# RLGCNt: Cardinality Estimation based on Rank Gauss Transform Coding and Attention

# RLGCNt：基于秩高斯变换编码和注意力机制的基数估计

No Author Given

未提供作者信息

No Institute Given

未提供机构信息

Abstract. Cost and cardinality estimation play a pivotal role in determining the selection of query execution plans. However, the cardinality estimation model based on deep learning will suffer from the problem of reduced accuracy when dealing with dynamic workloads. To address the degradation of cardinality estimation under dynamic workloads, we present a novel cardinality estimation model, namely RLGCNt. For query, unevenly distributed data is rank-transformed into a normal distribution using Gaussian transform-based query encoding. This solves the problem of reduced encoding efficiency under dynamic data loads. Furthermore, we incorporate a locality-sensitive hashing module and a Gaussian kernel function module to boost cardinality estimation and capture data similarities effectively. The RLGCNt model also introduces a causal attention mechanism to address the problem of how the query data arrangement affects the cardinality estimation. We conduct a comparison between RLGCNt and deep-learning-based cardinality estimation methods on the public STATS dataset. Experimental results demonstrate that RLGCNt outperforms mainstream cardinality estimation algorithms. Specifically, for dynamic workloads in the STATS dataset, RL-GCNt achieves an accuracy 10.7% higher than the baseline method, and for static loads,it attains an accuracy ${5.8}\%$ higher.

摘要：成本和基数估计在确定查询执行计划的选择中起着关键作用。然而，基于深度学习的基数估计模型在处理动态工作负载时会出现准确性下降的问题。为了解决动态工作负载下基数估计性能下降的问题，我们提出了一种新颖的基数估计模型，即RLGCNt。对于查询，使用基于高斯变换的查询编码将分布不均匀的数据进行秩变换，使其服从正态分布。这解决了动态数据负载下编码效率降低的问题。此外，我们引入了局部敏感哈希模块和高斯核函数模块，以提高基数估计的准确性并有效捕捉数据的相似性。RLGCNt模型还引入了因果注意力机制，以解决查询数据排列方式对基数估计的影响问题。我们在公共STATS数据集上对RLGCNt和基于深度学习的基数估计方法进行了比较。实验结果表明，RLGCNt优于主流的基数估计算法。具体而言，对于STATS数据集中的动态工作负载，RL - GCNt的准确率比基线方法高10.7%，对于静态负载，其准确率提高了${5.8}\%$。

Keywords: cardinality estimation - execution plans - dynamic - attention.

关键词：基数估计 - 执行计划 - 动态 - 注意力机制

## 1 Introduction

## 1 引言

In the context of query optimization, cost estimation and cardinality estimation are of paramount importance particularly in relational database management systems (RDBMS). Cardinality estimation, in particular, employs algorithms to estimate the size of a query result set. This estimation is instrumental in enabling the query optimizer to choose appropriate query plans, thereby enhancing the overall efficiency of the database system.

在查询优化的背景下，成本估计和基数估计尤为重要，特别是在关系数据库管理系统（RDBMS）中。基数估计尤其使用算法来估计查询结果集的大小。这种估计有助于查询优化器选择合适的查询计划，从而提高数据库系统的整体效率。

Traditional cardinality estimators $\left\lbrack  {6,9}\right\rbrack$ often ignore the correlation between features and predicates of queries, which reduces estimation accuracy. sampling methods $\left\lbrack  {2,6}\right\rbrack$ are less effective under dynamic workloads. For queries involving complex joins and multiple tables [13], applying deep learning neural networks [11] for cardinality estimation has become a trend. MSCN [4] based on deep learning addresses issues faced by traditional estimators, but small sample sizes can reduce effectiveness in large-scale databases and complex workloads. Additionally, the model does not fully capture local data patterns. In the ALECE [7] model, the attention mechanism is significantly influenced by the sequence position of features. Specifically, the one-hot encoding employed within ALECE demonstrates reduced effectiveness under conditions of uneven or highly skewed data distribution. This inefficiency of one-hot encoding can lead to suboptimal performance in the ALECE model, as it fails to adequately represent the data characteristics in such non-uniform data scenarios.

传统的基数估计器$\left\lbrack  {6,9}\right\rbrack$通常忽略查询特征和谓词之间的相关性，这会降低估计的准确性。采样方法$\left\lbrack  {2,6}\right\rbrack$在动态工作负载下效果较差。对于涉及复杂连接和多个表的查询[13]，应用深度学习神经网络[11]进行基数估计已成为一种趋势。基于深度学习的MSCN[4]解决了传统估计器面临的问题，但在大规模数据库和复杂工作负载中，小样本量会降低其有效性。此外，该模型没有充分捕捉局部数据模式。在ALECE[7]模型中，注意力机制受特征序列位置的影响很大。具体来说，ALECE中使用的独热编码在数据分布不均匀或高度偏斜的情况下效果较差。这种独热编码的低效性可能导致ALECE模型性能不佳，因为它无法在这种非均匀数据场景中充分表示数据特征。

The FACE [16] estimates cardinality through an autoregressive model and assumes that columns are independent of each other, but is affected by the arrangement of data in the query when facing dynamic workloads. Similarly, the PRICE [21] estimates cardinality from a high-dimensional joint probability density function, but is also affected by the arrangement of data in the query, which affects accuracy.

FACE[16]通过自回归模型估计基数，并假设各列相互独立，但在面对动态工作负载时会受到查询中数据排列的影响。同样，PRICE[21]从高维联合概率密度函数估计基数，但也会受到查询中数据排列的影响，从而影响准确性。

To address the limitations of existing technologies, we propose a novel cardinality estimation method based on attention mechanism and Gaussian kernel function. The core contributions of this paper include the following four parts:

为了解决现有技术的局限性，我们提出了一种基于注意力机制和高斯核函数的新颖基数估计方法。本文的核心贡献包括以下四个部分：

- Cardinality Estimation Model: We propose the RLGCNt cardinality estimation model, which improves accuracy for both dynamic and static workloads. By integrating locality-sensitive hashing, Gaussian kernel functions, and an attention mechanism, we cluster data, thus enhancing similarity search and local feature extraction in dynamic workload.

- 基数估计模型：我们提出了RLGCNt基数估计模型，该模型提高了动态和静态工作负载下的准确性。通过集成局部敏感哈希、高斯核函数和注意力机制，我们对数据进行聚类，从而增强了动态工作负载中的相似性搜索和局部特征提取。

- Query Encoding Method: We design a Rank Gaussian Transform-based query encoding method to enhance outlier resilience by transforming with a Gaussian rank function.

- 查询编码方法：我们设计了一种基于秩高斯变换的查询编码方法，通过使用高斯秩函数进行变换来增强对离群值的鲁棒性。

- Causal Attention Mechanism: We introduce a causal attention mechanism by adding causal masks to improve the handling of the effect of data arrangement on cardinality, addressing the limitations of traditional attention mechanisms.

- 因果注意力机制：我们通过添加因果掩码引入了因果注意力机制，以改善对数据排列对基数影响的处理，解决了传统注意力机制的局限性。

- Performance Comparison: RLGCNt is compared to mainstream models on the STATS dataset, which contains over 180,000 dynamic and static workloads. Experimental results show that RLGCNt is ${10.7}\%$ more accurate than baseline in dynamic workloads.

- 性能比较：在包含超过180,000个动态和静态工作负载的STATS数据集上，将RLGCNt与主流模型进行了比较。实验结果表明，在动态工作负载下，RLGCNt比基线模型的准确率提高了${10.7}\%$。

## 2 Related work

## 2 相关工作

### 2.1 Query-driven Model

### 2.1 查询驱动模型

The Query-driven model is a deep learning cardinality estimation model that encodes queries. The classic query-driven MSCN model [4] maps feature vectors to a low-dimensional space and then estimates cardinality through a fully connected layer. However, when this model handles dynamic workloads, the accuracy of the cardinality estimation decreases significantly. Deep Sketches [5] combine convolutional networks and materialized samples for cardinality estimation, but lack data classification, which prevents it from capturing similarities between data. Although this model requires extensive training to adapt to different query modes, it shows significant deviations when faced with unknown queries. Another query-driven model is DBEst [10], which uses a convolutional neural network combined with multiple encoding methods for cardinality estimation, but its performance is average under dynamic workloads because of its poor performance in handling outliers.NNGP [23] is also a query-driven cardinality estimation model, which improves the accuracy of cardinality estimation by using neural networks and capturing spatial similarities between different data in space. However, NNGP has problems in extracting local feature information of queries. In general, query-driven query encoding needs to be improved to address the problems of outliers and local feature extraction.

查询驱动模型是一种对查询进行编码的深度学习基数估计模型。经典的查询驱动MSCN模型[4]将特征向量映射到低维空间，然后通过全连接层进行基数估计。然而，当该模型处理动态工作负载时，基数估计的准确性会显著下降。深度草图（Deep Sketches）[5]将卷积网络和物化样本相结合进行基数估计，但缺乏数据分类，这使其无法捕捉数据之间的相似性。尽管该模型需要大量训练以适应不同的查询模式，但在面对未知查询时会出现显著偏差。另一个查询驱动模型是DBEst[10]，它使用卷积神经网络结合多种编码方法进行基数估计，但由于其在处理离群值方面表现不佳，在动态工作负载下的性能一般。NNGP[23]也是一种查询驱动的基数估计模型，它通过使用神经网络并捕捉不同数据在空间中的空间相似性来提高基数估计的准确性。然而，NNGP在提取查询的局部特征信息方面存在问题。总体而言，需要改进查询驱动的查询编码，以解决离群值和局部特征提取的问题。

### 2.2 Data-driven Model

### 2.2 数据驱动模型

Another cardinality estimation model is the data-driven model, which performs cardinality estimation based on the modeling of the underlying data. The first is the DeepDB model [3] that uses an RSPN-based data structure. Compared with the previous cardinality estimation methods, the DeepDB model achieves great improvement, but when the query changes, the model often suffers from insufficient generalization capabilities and limited processing of outliers. In addition, Neurocard [19] directly samples the connection results to build an autoregressive model and then trains the autoregressive model in the samples of the complete outer join. However, even if the columns are decomposed and the sub-columns become simpler to a certain extent, there are still several complex columns, and Neurocard fails to capture the similarities between data sufficiently. It is also limited in handling outliers. FACE [16] is a cardinality estimation model based on the rule flow model. It has different processing methods for different data types. If the data are discrete numerical, it is first quantized and processed to turn them into continuous data. Because the FACE model relies too much on continuous data, it affects the accuracy of cardinality estimation.

另一种基数估计模型是数据驱动模型，它基于对底层数据的建模进行基数估计。首先是使用基于RSPN的数据结构的DeepDB模型[3]。与之前的基数估计方法相比，DeepDB模型有了很大的改进，但当查询发生变化时，该模型往往存在泛化能力不足和对离群值处理有限的问题。此外，Neurocard[19]直接对连接结果进行采样以构建自回归模型，然后在完全外连接的样本中训练自回归模型。然而，即使对列进行了分解，子列在一定程度上变得更简单，但仍然存在几个复杂的列，并且Neurocard未能充分捕捉数据之间的相似性。它在处理离群值方面也受到限制。FACE[16]是一种基于规则流模型的基数估计模型。它对不同的数据类型有不同的处理方法。如果数据是离散数值型的，首先对其进行量化和处理，使其变为连续数据。由于FACE模型过于依赖连续数据，这会影响基数估计的准确性。

### 2.3 Hybrid Model

### 2.3 混合模型

In addition to the above two models, ALECE [7] combines query-driven and data-driven approaches. Based on the attention mechanism, it can efficiently handle dynamic workloads. It uses the MLP layer of the attention mechanism to merge data and query workload for cardinality estimation. However, the calculation of attention weights is affected both by the arrangement of query data and by its inability to perform fast approximate search, which can lead to errors. PRICE [21] is another hybrid-driven model, but it also ignores the impact of the data arrangement in the query on the cardinality, resulting in reduced efficiency.

除了上述两种模型外，ALECE[7]结合了查询驱动和数据驱动的方法。基于注意力机制，它可以有效地处理动态工作负载。它使用注意力机制的MLP层将数据和查询工作负载进行合并以进行基数估计。然而，注意力权重的计算既受查询数据排列的影响，又受其无法进行快速近似搜索的影响，这可能导致误差。PRICE[21]是另一种混合驱动模型，但它也忽略了查询中数据排列对基数的影响，导致效率降低。

The problems with the above three methods can be summarized as follows: the existing cardinality estimation models do not handle outliers in queries well, and ignore the arrangement of query data. Especially in dynamic workloads, these two problems are particularly prominent, and the local features of data and fast approximate search also need to be improved. Therefore, we address the above four problems.

上述三种方法的问题可以总结如下：现有的基数估计模型不能很好地处理查询中的离群值，并且忽略了查询数据的排列。特别是在动态工作负载下，这两个问题尤为突出，数据的局部特征和快速近似搜索也需要改进。因此，我们解决了上述四个问题。

## 3 Cardinality estimation

## 3 基数估计

### 3.1 Problem Definition

### 3.1 问题定义

Consider a set $M$ with attributes $\left\{  {{B}_{1},{B}_{2},\ldots ,{B}_{m}}\right\}$ and a relation $R$ containing $P$ tuples. Each tuple $u \in  R$ is represented as $u = \left( {{b}_{1},{b}_{2},\ldots ,{b}_{m}}\right)$ ,where ${b}_{j}$ is the value of attribute ${B}_{j}$ ,and $j$ ranges from 1 to $m$ . The function $o\left( u\right)$ indicates the number of tuple $u$ in the relation $R$ . A query $Q$ can be viewed as a function applied to each tuple $u$ . If $u$ meets the query conditions, $Q\left( u\right)  = 1$ ; otherwise, $Q\left( u\right)  = 0$ . The cardinality of the query $Q$ is defined as:

考虑一个具有属性$\left\{  {{B}_{1},{B}_{2},\ldots ,{B}_{m}}\right\}$的集合$M$和一个包含$P$个元组的关系$R$。每个元组$u \in  R$表示为$u = \left( {{b}_{1},{b}_{2},\ldots ,{b}_{m}}\right)$，其中${b}_{j}$是属性${B}_{j}$的值，并且$j$的范围是从1到$m$。函数$o\left( u\right)$表示关系$R$中元组$u$的数量。一个查询$Q$可以看作是应用于每个元组$u$的函数。如果$u$满足查询条件，则$Q\left( u\right)  = 1$；否则，$Q\left( u\right)  = 0$。查询$Q$的基数定义为：

$$
\operatorname{card}\left( Q\right)  = \left| {\{ u \in  R : Q\left( u\right)  = 1\} }\right| 
$$

which represents the number of tuples that satisfy the query $Q$ . The selectivity of the query $Q$ is defined as:

它表示满足查询$Q$的元组数量。查询$Q$的选择性定义为：

$$
\operatorname{sel}\left( Q\right)  = \frac{\operatorname{card}\left( Q\right) }{P}
$$

Cardinality Estimation (CE) [16,20] aims to predict the number of tuples card(Q)that satisfy the query conditions of $\mathrm{Q}$ without actually running the query.

基数估计（CE）[16,20]旨在在不实际运行查询的情况下预测满足$\mathrm{Q}$查询条件的元组数量card(Q)。

### 3.2 Cardinality Estimation of RLGCNt

### 3.2 RLGCNt的基数估计

Deep learning-based cardinality estimation models, which struggle with outliers and uneven data (leading to large estimation errors), are limited in dynamic query scenarios. Additionally, traditional attention mechanisms are easily affected by data arrangement in queries, further degrading estimation accuracy. To address these issues, we propose RLGCNt, which encodes the query using Rank Gaussian Transform coding, combines locality-sensitive hashing and Gaussian kernel functions for joint data-query representation, and finally uses the causal mask in the causal attention mechanism to capture the arrangement of query data. The internal structure of RLGCNt is shown in Fig. 1. In RLGCNt, table join conditions(SJCB, SARC and BRTL)are extracted from the query workload, which include the table join conditions, attribute ranges, and related tables. Then, Rank Gaussian Transform encoding is applied to normalize query distributions, enhancing the model's outlier robustness in large workloads. At the same time, histogram characterizations is used to process the table data. Both representations undergo locality-sensitive hashing (LSH), where high-dimensional data is classified into hash buckets (e.g., Bucket-A, Bucket-B), significantly improving the model's efficiency and robustness. Then, the Gaussian kernel function (GKF) is used to capture data similarities and local features. After LSH and GKF processing, the data is submitted to self-attention and cross-attention with multiple heads. The causal mask enables multi-head attention to dynamically prioritize query weights based on data arrangement. The causal mask captures the arrangement of the query data. The output is fed into an MLP layer with stacked fully connected (FC) layers. Similarly, the instance object reg, constructed by multiple MLPs, calculates the final cardinality. The regression module (reg ), constructed via multiple MLPs, is integrated into the loss function for adaptive weight adjustment in self- and cross-CMA layers.

基于深度学习的基数估计模型在处理离群值和不均匀数据时存在困难（会导致较大的估计误差），在动态查询场景中存在局限性。此外，传统的注意力机制容易受到查询中数据排列的影响，进一步降低了估计的准确性。为了解决这些问题，我们提出了RLGCNt，它使用秩高斯变换编码对查询进行编码，结合局部敏感哈希和高斯核函数进行联合数据 - 查询表示，最后使用因果注意力机制中的因果掩码来捕捉查询数据的排列。RLGCNt的内部结构如图1所示。在RLGCNt中，从查询工作负载中提取表连接条件（SJCB、SARC和BRTL），其中包括表连接条件、属性范围和相关表。然后，应用秩高斯变换编码对查询分布进行归一化，增强模型在大工作负载下对离群值的鲁棒性。同时，使用直方图表征来处理表数据。这两种表示都经过局部敏感哈希（LSH）处理，将高维数据分类到哈希桶中（例如，桶A、桶B），显著提高了模型的效率和鲁棒性。然后，使用高斯核函数（GKF）来捕捉数据的相似性和局部特征。经过LSH和GKF处理后，数据被提交到多头自注意力和交叉注意力机制中。因果掩码使多头注意力能够根据数据排列动态地确定查询权重的优先级。因果掩码捕捉查询数据的排列。输出被输入到一个由堆叠全连接（FC）层组成的多层感知机（MLP）层中。同样，由多个MLP构建的实例对象reg计算最终的基数。通过多个MLP构建的回归模块（reg）被集成到损失函数中，用于在自注意力和交叉注意力CMA层中进行自适应权重调整。

<!-- Media -->

<!-- figureText: Generating causal masks Generating causal masks Cardinality RLGCNt MLP reg FC MLP loss FC - Hash Bucket-B ...... Rank Gauss Transform SJCB CDF SARC Dynamic/static workload BRTL Sort the query Scaling Cross-attention K self-CMA layers cross-CMA layers K, Values, C GKF LSH Hash Bucket-A Histogram Data -->

<img src="https://cdn.noedgeai.com/0195b83e-3266-7865-aa76-bf1a6ae5de97_4.jpg?x=384&y=327&w=1028&h=753&r=0"/>

Fig. 1. Cardinality estimation based on RLGCNt.

图1. 基于RLGCNt的基数估计。

<!-- Media -->

## 4 RLGCNt Based on Attention Model

## 4 基于注意力模型的RLGCNt

### 4.1 The Rank Gauss Transform Coding

### 4.1 秩高斯变换编码

Since one-hot encoding is often ineffective when dealing with queries on massive, non-uniform data and categorical features, we use Rank Gauss Transform to encode each filtering condition, transforming queries from a nonuniform distribution to a normal distribution.

由于独热编码在处理大规模、非均匀数据和分类特征的查询时往往效果不佳，我们使用秩高斯变换对每个过滤条件进行编码，将查询从非均匀分布转换为正态分布。

Our approach makes categorical features easier for the attention mechanism to process and solves the problem of outliers and repeated values. The input consists of ${SJCB},{SARC}$ ,and ${BRTL}$ . The first Step involves sorting each element in the query data using the rank data method, specifically by the minimum value. ranked represents the result after sorting, as shown in Eq. (1). The minimum value sorting method we use assigns repeated values the minimum rank based on their first appearance. ${xk}$ represents the information vector related to the query,and $i$ represents the index. The next step encodes the input SQL query’s join conditions, attribute range conditions, and the list of relevant tables using the Rank Gauss Transform method,as shown in Eq. (2). ${ng}$ represents the total number of attributes. After that, the length of the SQL conditions is calculated, and the necessary variables are initialized. The process then iterates through all SQL conditions and accumulates the predicted values. The final step involves converting the scaled data into a normal distribution using the $\Phi$ function. $\Phi$ is the inverse of the Cumulative Distribution Function (CDF) of a probability distribution, and transformed is the new query feature obtained from the output of CDF. Eq. (3) shows that the standard normal distribution value of $x{k}_{i}$ is obtained through the inverse cumulative distribution function of the standard normal distribution. $\Phi$ represents the CDF of the standard normal distribution.

我们的方法使分类特征更易于注意力机制处理，并解决了离群值和重复值的问题。输入包括${SJCB},{SARC}$和${BRTL}$。第一步涉及使用秩数据方法对查询数据中的每个元素进行排序，具体是按最小值排序。ranked表示排序后的结果，如公式（1）所示。我们使用的最小值排序方法根据重复值的首次出现为其分配最小的秩。${xk}$表示与查询相关的信息向量，$i$表示索引。下一步使用秩高斯变换方法对输入SQL查询的连接条件、属性范围条件和相关表列表进行编码，如公式（2）所示。${ng}$表示属性的总数。之后，计算SQL条件的长度，并初始化必要的变量。然后，该过程遍历所有SQL条件并累加预测值。最后一步涉及使用$\Phi$函数将缩放后的数据转换为正态分布。$\Phi$是概率分布的累积分布函数（CDF）的反函数，transformed是从CDF输出中获得的新查询特征。公式（3）表明，$x{k}_{i}$的标准正态分布值是通过标准正态分布的逆累积分布函数获得的。$\Phi$表示标准正态分布的CDF。

$$
\operatorname{ranked}\left( {x{k}_{i}}\right)  = \operatorname{rankdata}\left( {x{k}_{i},\text{ method } = \min }\right)  \tag{1}
$$

$$
\text{ scaled_rank }\left( {x{k}_{i}}\right)  = \frac{\text{ ranked }\left( {x{k}_{i}}\right) }{{ng} + 1} \tag{2}
$$

$$
\operatorname{transformed}\left( {x{k}_{i}}\right)  = {\Phi }^{-1}\left( {\text{ scaled_rank }\left( {x{k}_{i}}\right) }\right)  \tag{3}
$$

Once the query is encoded using the Rank Gauss Transform, it is merged with the data features and fed into the multi-head cross-attention layer. Finally, the weight of the query is calculated, and the cardinality is predicted via stacked fully connected (FC) layers.

一旦使用秩高斯变换对查询进行编码，它就会与数据特征合并，并输入到多头交叉注意力层中。最后，计算查询的权重，并通过堆叠全连接（FC）层预测基数。

### 4.2 Locality Sensitive Hash

### 4.2 局部敏感哈希

The purpose of combining the attention mechanism with locality-sensitive hashing (LSH) is to address the issue that the attention mechanism cannot efficiently approximate query results. This combination reduces the number of query-key-value pair combinations that need to be calculated, thereby greatly speeding up the attention mechanism and improving the accuracy of cardinality estimation. The principle of LSH is to map two similar data points from the original dataset into the same hash bucket. The goal is to ensure that these points remain adjacent in the new bucket, thereby minimizing the probability that non-adjacent data points are mapped to the same bucket.

将注意力机制与局部敏感哈希（LSH）相结合的目的是解决注意力机制无法有效近似查询结果的问题。这种结合减少了需要计算的查询 - 键 - 值对组合的数量，从而大大加快了注意力机制的速度并提高了基数估计的准确性。LSH的原理是将原始数据集中的两个相似数据点映射到同一个哈希桶中。目标是确保这些点在新桶中仍然相邻，从而最小化非相邻数据点被映射到同一个桶中的概率。

As shown in Figure 2, ${tz}$ represents the merged features of the key-value (K,Values) pairs of the input data and the query(Q),which are then divided into different hash buckets such as Hash Bucket-A, Hash Bucket-B, and Hash Bucket-C,etc. The four-line intersecting circle $g$ points to is the similarity matrix generated by the Gaussian kernel function algorithm. The weights are calculated through the causal attention mechanism we proposed (in Section. 5) and then transformed through the residual connection and forward feedback layer to obtain the output results $q{x}_{1},q{x}_{2},q{x}_{3},\ldots q{x}_{n}$ . The blue circles represent neurons in the mechanism, which generates feature vectors (rectangles with pink dots), while the black rectangles represent the generated residuals, and finally the gray rectangles represent the feature vectors after processing by the fully connected layer. The remaining content pertains to the Gaussian kernel function and is thoroughly discussed in Section 4.3. Locality-sensitive hashing completes the mapping of data points by creating a hash function and then calculating the hash value to create a hash table. Eq. (4) represents the creation of a hash function,where ${W}_{nl}$ and ${b}_{nl}$ represent the weight matrix and bias vector used to create the hash function,and ${nl}$ represents the number of hash functions. Eq. (5) represents the calculation of the hash value. ${W}_{ib}$ and ${b}_{ib}$ represent the weight matrix and bias vector used by the ${ib}$ -th hash function to compute the hash value,where ReLU is the activation function used,and hash_value ${}_{ib}$ represents the calculated hash value corresponding to ${tz}$ . Eq. (6) represents the connection of the calculated hash values along the last dimension of the feature to obtain the final hash value. Eq. (7) represents the construction of a hash table to combine all hash values ${h}_{nl}$ ,where $h$ represents the hash function. $x$ is the output of locality sensitive hashing.

如图2所示，${tz}$表示输入数据的键值（K，Values）对与查询（Q）的合并特征，然后将这些特征划分为不同的哈希桶，如哈希桶A、哈希桶B和哈希桶C等。四行相交的圆$g$所指向的是由高斯核函数算法生成的相似度矩阵。通过我们提出的因果注意力机制（在第5节）计算权重，然后通过残差连接和前馈反馈层进行转换，以获得输出结果$q{x}_{1},q{x}_{2},q{x}_{3},\ldots q{x}_{n}$。蓝色圆圈表示该机制中的神经元，它会生成特征向量（带有粉色点的矩形），而黑色矩形表示生成的残差，最后灰色矩形表示经过全连接层处理后的特征向量。其余内容与高斯核函数有关，将在4.3节中进行详细讨论。局部敏感哈希通过创建哈希函数，然后计算哈希值来创建哈希表，从而完成数据点的映射。公式（4）表示创建哈希函数，其中${W}_{nl}$和${b}_{nl}$分别表示用于创建哈希函数的权重矩阵和偏置向量，${nl}$表示哈希函数的数量。公式（5）表示计算哈希值。${W}_{ib}$和${b}_{ib}$分别表示第${ib}$个哈希函数用于计算哈希值的权重矩阵和偏置向量，其中ReLU是使用的激活函数，哈希值${}_{ib}$表示与${tz}$对应的计算得到的哈希值。公式（6）表示沿着特征的最后一个维度连接计算得到的哈希值，以获得最终的哈希值。公式（7）表示构建哈希表以组合所有哈希值${h}_{nl}$，其中$h$表示哈希函数。$x$是局部敏感哈希的输出。

<!-- Media -->

<!-- figureText: LSH Add and Norm Feed-forward $q{x}_{3}$ ......... Hash Bucket-A Hash Bucket-B feature vector tz Hash Bucket-C ......... -->

<img src="https://cdn.noedgeai.com/0195b83e-3266-7865-aa76-bf1a6ae5de97_6.jpg?x=389&y=331&w=1026&h=369&r=0"/>

Fig. 2. The LSH of RLGCNt.

图2. RLGCNt的局部敏感哈希。

<!-- Media -->

Storage hash function:

存储哈希函数：

$$
\left( {{W}_{1},{b}_{1}}\right) ,\left( {{W}_{2},{b}_{2}}\right) ,\ldots ,\left( {{W}_{nl},{b}_{nl}}\right)  \tag{4}
$$

Calculate hash value:

计算哈希值：

$$
\text{hash_valu}{e}_{ib} = \operatorname{ReLU}\left( {{tz}{W}_{ib} + {b}_{ib}}\right) ,i \in  \left\lbrack  {1,{nl}}\right\rbrack   \tag{5}
$$

Concatenate hash values:

连接哈希值：

$$
\text{hashed_inputs} = \text{concat(hash_value1, hash_value2,} \tag{6}
$$

$$
\left. {\ldots ,{\text{ hash_value }}_{{n}_{l}},\text{ axis } =  - 1}\right) 
$$

Construct a hash table:

构建哈希表：

$$
x = \left( {{h}_{1},{h}_{2},\ldots ,{h}_{nl}}\right)  \tag{7}
$$

### 4.3 Gaussian Kernel Function-Cardinality Estimation

### 4.3 高斯核函数 - 基数估计

We use the combination of the Gaussian kernel function to enhance the ability to capture the local structure of data. The Gaussian kernel function calculates similarity weights to better handle non-linearly related data points and reduce the noise problem caused by attention. First, the feature dimension is expanded, as shown in Eq. (8) and Eq. (9). In these equations, $x$ represents the output of locality-sensitive hashing (LSH),where ${x}_{1}$ and ${x}_{2}$ are parts of $\mathrm{x}$ and the expand_dims(   ) function is first used to reshape the input tensor by adding new dimensions,so that ${x}_{1}$ and ${x}_{2}$ can be obtained from the input features for pairwise distance computation. The difference between them is that ${x}_{1}$ inserts a new dimension from the second-to-last dimension to make its shape match the correct form,while ${x}_{2}$ inserts a new dimension from the third-to-last dimension for the purpose of calculating the Euclidean distance. Then, the distance between them according to the Euclidean metric is computed. Eq. (10) shows that after the dimension is expanded, $S$ is obtained by calculating the sum of the squares of the differences between the two feature vectors,and $d$ represents a specific dimension. Therefore, ${x}_{1,d}$ and ${x}_{2,d}$ represent the components of ${x}_{1}$ and ${x}_{2}$ in dimension $d$ ,respectively. Eq. (11) uses $S$ to calculate the Euclidean distance between them,and $D$ represents the number of feature tuples in the input $x$ . Their similarity is then judged by the Gaussian distribution. Enabling the attention mechanism to better capture the similarity between data. The final Eq. (12) represents dividing the obtained Euclidean distance by ${\sigma }^{2}$ . The role of $\sigma$ here is to control the Gaussian kernel, which denotes the bandwidth parameter of the Gaussian kernel,and then the range is controlled in $(0,1\rbrack$ through the exponential function. If it is closer to 1 , it means that the similarity between the two is higher.

我们使用高斯核函数的组合来增强捕捉数据局部结构的能力。高斯核函数计算相似度权重，以更好地处理非线性相关的数据点，并减少注意力引起的噪声问题。首先，扩展特征维度，如公式（8）和公式（9）所示。在这些公式中，$x$表示局部敏感哈希（LSH）的输出，其中${x}_{1}$和${x}_{2}$是$\mathrm{x}$的部分，并且首先使用expand_dims( )函数通过添加新维度来重塑输入张量，以便可以从输入特征中获得${x}_{1}$和${x}_{2}$用于成对距离计算。它们之间的区别在于，${x}_{1}$从倒数第二个维度插入一个新维度，以使其形状匹配正确的形式，而${x}_{2}$从倒数第三个维度插入一个新维度，用于计算欧几里得距离。然后，根据欧几里得度量计算它们之间的距离。公式（10）表明，在维度扩展后，通过计算两个特征向量之间差异的平方和得到$S$，$d$表示特定的维度。因此，${x}_{1,d}$和${x}_{2,d}$分别表示${x}_{1}$和${x}_{2}$在维度$d$上的分量。公式（11）使用$S$计算它们之间的欧几里得距离，$D$表示输入$x$中特征元组的数量。然后通过高斯分布判断它们的相似度。使注意力机制能够更好地捕捉数据之间的相似度。最后的公式（12）表示将获得的欧几里得距离除以${\sigma }^{2}$。这里$\sigma$的作用是控制高斯核，它表示高斯核的带宽参数，然后通过指数函数将范围控制在$(0,1\rbrack$内。如果它更接近1，则表示两者之间的相似度更高。

$$
{x}_{1} = \text{expand_dims}\left( {x, - 2}\right)  \tag{8}
$$

$$
{x}_{2} = \text{expand_dims}\left( {x, - 3}\right)  \tag{9}
$$

$$
S = {\left( {x}_{1,d} - {x}_{2,d}\right) }^{2} \tag{10}
$$

$$
\text{squared_distance} = \mathop{\sum }\limits_{0}^{D}S \tag{11}
$$

$$
\text{ Gaussian Kernel } = \exp \left( {-\frac{1}{2} \cdot  \frac{\text{ squared_distance }}{{\sigma }^{2}}}\right)  \tag{12}
$$

### 4.4 Causal Attention Mechanism

### 4.4 因果注意力机制

We use the causal attention mechanism to address the issue where the multihead attention mechanism is affected by the arrangement of query data, thereby reducing the model's over-dependence on future data, limiting the scope of attention, improving the performance of cardinality estimation, and calculating the mask does not increase computational overhead.

我们使用因果注意力机制来解决多头注意力机制受查询数据排列影响的问题，从而减少模型对未来数据的过度依赖，限制注意力范围，提高基数估计的性能，并且计算掩码不会增加计算开销。

<!-- Media -->

<!-- figureText: ...... Feedback to original features Causal Mask Matrix(M) ...... ...... ...... Generating causal masks ...... Feature matrix ‵ $g\left\lbrack  {m,n}\right\rbrack$ Decompose feature vector into Jkey-value pairs and query ${Q}_{p},{V}_{p},{K}_{p}$ -->

<img src="https://cdn.noedgeai.com/0195b83e-3266-7865-aa76-bf1a6ae5de97_8.jpg?x=385&y=332&w=1026&h=514&r=0"/>

Fig. 3. Mechanism process diagram.

图3. 机制流程示意图。

<!-- Media -->

As shown in Fig. 3, the feature matrix on the left is the similarity matrix output by the Gaussian kernel function, represented as the pink matrix, while the blue color represents the query feature vector. First, the required query feature vector is extracted; specifically, the original feature vector is converted into a key-value pair by applying a linear transformation to the weights (as the two green arrows in Fig. 3). Then, the query feature vector is retrieved from the key-value pair (as shown by the yellow arrows). We use the query feature vector to generate a causal mask, as shown by the red arrow. First, a matrix of all ones is generated. Then, elements on and below the main diagonal are kept unchanged, while the rest are set to 0 , thus transforming the matrix into an upper triangular form, which serves as the causal mask. Eq. (13) and Eq. (14) represent the equations for generating masks and the equations for the attention mechanism,respectively,where $c$ and $j$ represent the row index and column index of the matrix respectively,and ${Q}_{p},{K}_{p}$ and ${V}_{p}$ in Eq. (14) represent the values of the key-value pairs after linear transformation,and $M$ represents the causal mask generated in Eq. (13). The final light blue-colored matrix represents the generated causal mask, which is then integrated back into the original feature matrix.

如图3所示，左侧的特征矩阵是高斯核函数输出的相似度矩阵，用粉色矩阵表示，而蓝色表示查询特征向量。首先，提取所需的查询特征向量；具体来说，通过对权重应用线性变换将原始特征向量转换为键值对（如图3中的两个绿色箭头所示）。然后，从键值对中检索查询特征向量（如黄色箭头所示）。我们使用查询特征向量生成因果掩码，如红色箭头所示。首先，生成一个全为1的矩阵。然后，保持主对角线及其下方的元素不变，其余元素设置为0，从而将矩阵转换为上三角形式，作为因果掩码。式(13)和式(14)分别表示生成掩码的方程和注意力机制的方程，其中$c$和$j$分别表示矩阵的行索引和列索引，式(14)中的${Q}_{p},{K}_{p}$和${V}_{p}$表示线性变换后键值对的值，$M$表示式(13)中生成的因果掩码。最终的浅蓝色矩阵表示生成的因果掩码，然后将其集成回原始特征矩阵。

$$
{\mathbf{M}}_{cj} = \left\{  \begin{array}{ll} 1, & \text{ if }c \geq  j \\  0, & \text{ if }c < j \end{array}\right.  \tag{13}
$$

$$
\operatorname{Attention}\left( {{Q}_{p},{K}_{p},{V}_{p}}\right)  = \operatorname{softmax}\left( {\frac{{Q}_{p}{K}_{p}^{\top }}{\sqrt{{d}_{k}}} + M}\right) {V}_{p} \tag{14}
$$

## 5 Loss function

## 5 损失函数

We use MSE [14] as part of the loss function to evaluate the optimization performance. Eq. (15) defines the loss function, which consists of two components. The first component calculates the mean squared error (MSE) between two values, representing the true cardinality and the estimated cardinality, respectively. The second component is the sum of the regularization losses incurred during the forward pass through each fully connected layer.

我们使用均方误差（MSE）[14]作为损失函数的一部分来评估优化性能。式(15)定义了损失函数，它由两部分组成。第一部分计算两个值之间的均方误差（MSE），这两个值分别表示真实基数和估计基数。第二部分是在通过每个全连接层的前向传播过程中产生的正则化损失之和。

MSE is given by Eq. (16). The mean squared error between the true cardinality and the estimated cardinality is computed for each sample, where labels represents the true cardinality, preds represents the estimated cardinality, and ${nv}$ denotes the total number of samples. Finally,we use the Adam optimizer [22] to optimize the model during backpropagation.

均方误差（MSE）由式(16)给出。针对每个样本计算真实基数和估计基数之间的均方误差，其中labels表示真实基数，preds表示估计基数，${nv}$表示样本总数。最后，我们使用Adam优化器[22]在反向传播过程中对模型进行优化。

$$
\text{Loss} = \operatorname{MSE}\left( \text{labels,preds}\right)  + \text{regularization} \tag{15}
$$

$$
{MSE} = \frac{1}{nv}\mathop{\sum }\limits_{{i = 1}}^{{nv}}{\left( {\text{ labels }}_{i} - {\text{ preds }}_{i}\right) }^{2} \tag{16}
$$

## 6 Experimental Evaluation

## 6 实验评估

In this section, we introduce the experimental configuration, dataset and dynamic workload used by RLGCNt.

在本节中，我们介绍RLGCNt使用的实验配置、数据集和动态工作负载。

### 6.1 Experimental Settings and Data Sets

### 6.1 实验设置和数据集

Experimental settings : Our experimental setup included a computer powered by an Intel i7-10700 processor, operating under Ubuntu 20.04.4 and evaluates the optimization method of this paper on more than 180,000 dynamic and static workloads.

实验设置：我们的实验环境包括一台由英特尔i7 - 10700处理器驱动的计算机，运行在Ubuntu 20.04.4系统下，并在超过180,000个动态和静态工作负载上评估本文的优化方法。

STATS ${}^{1}$ : The STATS consists of eight interrelated data tables,forming a complex data ecosystem. These tables include: user information table (storing key data such as user ID, user name, reputation, etc.), post details table (covering post ID, title, content, etc.), post link relationship table, post history change record table, comment interaction table (recording user comments on posts), voting behavior table (tracking voting status of posts and comments), badge recognition table (recording honor badges obtained by users), and label classification table (labeling posts with relevant labels). The dataset is extensive, containing 1,029,842 detailed records.

STATS ${}^{1}$：STATS由八个相互关联的数据表组成，形成一个复杂的数据生态系统。这些表包括：用户信息表（存储用户ID、用户名、信誉等关键数据）、帖子详情表（涵盖帖子ID、标题、内容等）、帖子链接关系表、帖子历史更改记录表、评论交互表（记录用户对帖子的评论）、投票行为表（跟踪帖子和评论的投票状态）、徽章识别表（记录用户获得的荣誉徽章）和标签分类表（为帖子添加相关标签）。该数据集规模庞大，包含1,029,842条详细记录。

Dynamic workloadss [7]: The dynamic workloads consist of three parts: insert, update, and delete statements. The initial data set randomly selects two-thirds of these statements in batches and puts them into the relational data table. The remaining one-third is used for update and insert operations.

动态工作负载[7]：动态工作负载由三部分组成：插入、更新和删除语句。初始数据集分批随机选择其中三分之二的语句放入关系数据表中。其余三分之一用于更新和插入操作。

---

<!-- Footnote -->

${}^{1}$ https://relational.fit.cvut.cz/dataset/Stats

${}^{1}$ https://relational.fit.cvut.cz/dataset/Stats

<!-- Footnote -->

---

### 6.2 Evaluation Indicators

### 6.2 评估指标

Q-error [12]: To show the effect of the model in this paper on cardinality estimation, Q-error is used as an indicator to measure the model in this article and other mainstream methods. The Eq. (17) of Q-error is as follows, where labels represents the true cardinality and preds represents the predicted cardinality.

Q误差[12]：为了展示本文模型对基数估计的效果，使用Q误差作为指标来衡量本文模型和其他主流方法。Q误差的式(17)如下，其中labels表示真实基数，preds表示预测基数。

$$
Q - \text{ error } = \max \left( {\frac{\text{ labels }}{\text{ preds }},\frac{\text{ preds }}{\text{ labels }}}\right)  \tag{17}
$$

E2E time [1]: The definition of E2E time is the sum of the time it takes to complete all query executions in the face of different workloads and data sets. It directly reflects the quality of DBMS query performance, and is also an important indicator for measuring the cardinality estimation for query execution speed.

端到端时间（E2E时间）[1]：端到端时间的定义是在面对不同工作负载和数据集时完成所有查询执行所需的时间总和。它直接反映了数据库管理系统（DBMS）查询性能的优劣，也是衡量查询执行速度的基数估计的重要指标。

### 6.3 Comparison Objects

### 6.3 比较对象

We use the following models to conduct comparative experiments with RLGCNt, including data-driven, query-driven, and hybrid-driven methods.

我们使用以下模型与RLGCNt进行对比实验，包括数据驱动、查询驱动和混合驱动方法。

PostgreSQL [17], mainly obtains the final cardinality through statistical information, which is initially collected manually by the analyze command or automatically collected through the model.

PostgreSQL [17]主要通过统计信息获取最终的基数，这些统计信息最初由analyze命令手动收集或通过模型自动收集。

Uni-Samp [8] performs a fast and accurate cardinality estimation without traversing all data sets.

Uni - Samp [8]无需遍历所有数据集即可进行快速准确的基数估计。

NeuroCard [19] uses Monte Carlo integration to perform progressive sampling in the data set during the probability inference process, which can effectively handle various query situations for the subset problems involved in the query.

NeuroCard [19]在概率推理过程中使用蒙特卡罗积分在数据集中进行渐进式采样，能够有效处理查询中涉及的子集问题的各种查询情况。

FLAT [24] is a data-driven model based on the SPN storage method.

FLAT [24]是一种基于SPN存储方法的数据驱动模型。

FactorJoin [18] proposes a new binning and upper bounding algorithm to approximate complex factor graph reasoning and combines it with histograms for cardinality estimation.

FactorJoin [18]提出了一种新的分箱和上界算法来近似复杂的因子图推理，并将其与直方图结合进行基数估计。

MLP [15] is not only used for cardinality estimation, but also predicts the final result by fitting the training data for the regression model analysis.

MLP [15]不仅用于基数估计，还通过拟合训练数据进行回归模型分析来预测最终结果。

MSCN [4] decomposes the query into tables, join conditions and predicates, and then encodes them one by one.

MSCN [4]将查询分解为表、连接条件和谓词，然后逐一进行编码。

NNGP [23] estimates the cardinality by the uncertainty of the result.

NNGP [23]通过结果的不确定性来估计基数。

ALECE [7] processes the underlying data through histogram normalization encoding, and then combines the unique hot encoding and the attention mechanism to perform cardinality estimation.

ALECE [7]通过直方图归一化编码处理底层数据，然后结合唯一热编码和注意力机制进行基数估计。

### 6.4 Experiment Analysis-Static Workloads

### 6.4 实验分析——静态工作负载

We use ALECE as the baseline for RLGCNt in both static and dynamic workloads. RLGCNt effectively solves the problems of outlier processing and fast approximate querying.

我们在静态和动态工作负载中都使用ALECE作为RLGCNt的基线。RLGCNt有效解决了离群值处理和快速近似查询的问题。

<!-- Media -->

Table 1. Ablation experiment results on STATS dataset-static workloads.

表1. STATS数据集静态工作负载的消融实验结果。

<table><tr><td rowspan="2">Model</td><td colspan="4">Q-error</td></tr><tr><td>50%</td><td>90%</td><td>95%</td><td>99%</td></tr><tr><td>ALECE</td><td>1.687</td><td>7.737</td><td>16.536</td><td>117.331</td></tr><tr><td>baseline+LSH+GKF</td><td>1.669</td><td>7.371</td><td>15.650</td><td>108.541</td></tr><tr><td>baseline+RGC+GKF</td><td>1.670</td><td>7.909</td><td>15.524</td><td>86.650</td></tr><tr><td>baseline+CAM</td><td>1.668</td><td>7.555</td><td>16.042</td><td>78.851</td></tr><tr><td>RLGCNt</td><td>1.601</td><td>6.582</td><td>13.497</td><td>83.285</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="4">Q误差</td></tr><tr><td>50%</td><td>90%</td><td>95%</td><td>99%</td></tr><tr><td>自适应局部误差补偿估计器（ALECE）</td><td>1.687</td><td>7.737</td><td>16.536</td><td>117.331</td></tr><tr><td>基线+局部敏感哈希（LSH）+高斯卡尔曼滤波（GKF）</td><td>1.669</td><td>7.371</td><td>15.650</td><td>108.541</td></tr><tr><td>基线+随机图卷积（RGC）+高斯卡尔曼滤波（GKF）</td><td>1.670</td><td>7.909</td><td>15.524</td><td>86.650</td></tr><tr><td>基线+通道注意力机制（CAM）</td><td>1.668</td><td>7.555</td><td>16.042</td><td>78.851</td></tr><tr><td>强化学习图卷积网络（RLGCNt）</td><td>1.601</td><td>6.582</td><td>13.497</td><td>83.285</td></tr></tbody></table>

Table 2. Comparative experimental results on the STATS dataset-static workloads.

表2. STATS数据集静态工作负载的对比实验结果。

<table><tr><td rowspan="2">Model</td><td colspan="4">Q-error</td><td rowspan="2">E2E Time (s)</td></tr><tr><td>50%</td><td>90%</td><td>95%</td><td>99%</td></tr><tr><td>PG</td><td>1.80</td><td>21.84</td><td>${1.1} \times  {10}^{5}$</td><td>${1.8} \times  {10}^{7}$</td><td>12,777</td></tr><tr><td>Uni-Samp</td><td>1.33</td><td>6.64</td><td>$> {10}^{10}$</td><td>$> {10}^{10}$</td><td>15,397</td></tr><tr><td>NeuroCard</td><td>2.91</td><td>192</td><td>1,511</td><td>${1.5} \times  {10}^{5}$</td><td>19,847</td></tr><tr><td>MSCN</td><td>3.85</td><td>39.56</td><td>99.81</td><td>1,273</td><td>15,162</td></tr><tr><td>NNGP</td><td>8.10</td><td>694</td><td>3,294</td><td>${2.3} \times  {10}^{5}$</td><td>20,181</td></tr><tr><td>RLGCNt</td><td>1.601</td><td>6.582</td><td>13.497</td><td>83.285</td><td>9,565</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="4">Q误差</td><td rowspan="2">端到端时间（秒）</td></tr><tr><td>50%</td><td>90%</td><td>95%</td><td>99%</td></tr><tr><td>策略梯度（Policy Gradient，PG）</td><td>1.80</td><td>21.84</td><td>${1.1} \times  {10}^{5}$</td><td>${1.8} \times  {10}^{7}$</td><td>12,777</td></tr><tr><td>统一采样（Uniform Sampling，Uni - Samp）</td><td>1.33</td><td>6.64</td><td>$> {10}^{10}$</td><td>$> {10}^{10}$</td><td>15,397</td></tr><tr><td>神经卡片（NeuroCard）</td><td>2.91</td><td>192</td><td>1,511</td><td>${1.5} \times  {10}^{5}$</td><td>19,847</td></tr><tr><td>多尺度卷积网络（Multi - Scale Convolutional Network，MSCN）</td><td>3.85</td><td>39.56</td><td>99.81</td><td>1,273</td><td>15,162</td></tr><tr><td>神经网络高斯过程（Neural Network Gaussian Process，NNGP）</td><td>8.10</td><td>694</td><td>3,294</td><td>${2.3} \times  {10}^{5}$</td><td>20,181</td></tr><tr><td>强化学习图卷积网络（Reinforcement Learning Graph Convolutional Network，RLGCNt）</td><td>1.601</td><td>6.582</td><td>13.497</td><td>83.285</td><td>9,565</td></tr></tbody></table>

<!-- Media -->

As shown in Table 1, LSH denotes the use of locality sensitive hashing before attention, GKF represents the use of the Rank Gaussian Transform instead of the original encoding method, RGC refers to the combination of the Gaussian kernel function with attention, CAM denotes the use of a different attention mechanism instead of the original attention, and RLGCNt represents the model proposed in this paper.

如表1所示，LSH表示在注意力机制之前使用局部敏感哈希（Locality Sensitive Hashing），GKF表示使用秩高斯变换（Rank Gaussian Transform）代替原始编码方法，RGC指的是将高斯核函数（Gaussian Kernel Function）与注意力机制相结合，CAM表示使用不同的注意力机制代替原始的注意力机制，RLGCNt表示本文提出的模型。

Compared with ALECE, which is based on the attention mechanism, RL-GCNt shows a decrease in all four indicators under static workloads.

与基于注意力机制的ALECE相比，RL - GCNt在静态工作负载下的四项指标均有所下降。

Table 1 shows that all four indicators have decreased compared with the existing model, with values of 1.669, 7.371, 15.650, and 108.541, respectively. The combination of Rank Gaussian Transform coding and the Gaussian kernel function makes the categorical features smoother and reduces the occurrence of outliers, adding new momentum to the attention-based cardinality estimation.

表1显示，与现有模型相比，四项指标均有所下降，数值分别为1.669、7.371、15.650和108.541。秩高斯变换编码与高斯核函数的结合使分类特征更加平滑，减少了异常值的出现，为基于注意力机制的基数估计增添了新的动力。

RLGCNt achieves an overall accuracy improvement of ${5.8}\%$ .

RLGCNt实现了总体准确率提升${5.8}\%$。

In Table 2, the four indicators are 1.601, 6.582, 13.497, and 83.285. Compared with traditional cardinality estimation methods such as Uni-Samp, the last three indicators show a significant decrease. Compared with mainstream cardinality estimation methods such as NNGP and MSCN, the four indicators show a significant decrease, with the last two exhibiting the most substantial decline.

在表2中，四项指标分别为1.601、6.582、13.497和83.285。与传统的基数估计方法（如Uni - Samp）相比，后三项指标显著下降。与主流的基数估计方法（如NNGP和MSCN）相比，四项指标均显著下降，其中后两项下降最为明显。

The results show that partial locality sensitive hashing, the Gaussian kernel function, and Rank Gaussian Transform coding proposed in this paper offer notable advantages, mitigating the impact of outliers and feature inconsistencies, thereby enhancing the effectiveness of the attention mechanism in cardinality estimation. RLGCNt achieves a substantial reduction in end-to-end processing time by transforming the query workload into a normal distribution and enabling fast approximate querying.

结果表明，本文提出的部分局部敏感哈希、高斯核函数和秩高斯变换编码具有显著优势，减轻了异常值和特征不一致的影响，从而提高了注意力机制在基数估计中的有效性。RLGCNt通过将查询工作负载转换为正态分布并实现快速近似查询，大幅减少了端到端处理时间。

### 6.5 Experiment Analysis-Dynamic Workloads

### 6.5 实验分析——动态工作负载

Table 3 shows that RLGCNt has a stronger ability to handle outliers and capture similarities in data features, thereby improving the accuracy of cardinality estimation by ${10.7}\%$ when facing dynamic workloads.

表3显示，RLGCNt在处理异常值和捕捉数据特征相似性方面具有更强的能力，因此在面对动态工作负载时，基数估计的准确率提高了${10.7}\%$。

<!-- Media -->

Table 3. Ablation experiment results on STATS dataset-dynamic Workloads.

表3. STATS数据集动态工作负载的消融实验结果。

<table><tr><td rowspan="2">Model</td><td colspan="4">Q-error</td></tr><tr><td>50%</td><td>90%</td><td>95%</td><td>99%</td></tr><tr><td>ALECE</td><td>2.447</td><td>11.596</td><td>24.899</td><td>198.961</td></tr><tr><td>baseline+RGC</td><td>1.745</td><td>7.879</td><td>15.954</td><td>68.847</td></tr><tr><td>baseline+LSH</td><td>1.557</td><td>8.454</td><td>18.999</td><td>143.885</td></tr><tr><td>baseline+GKF</td><td>1.570</td><td>7.307</td><td>14.419</td><td>148.007</td></tr><tr><td>baseline+CAM</td><td>2.111</td><td>10.001</td><td>22.654</td><td>166.228</td></tr><tr><td>RLGCNt</td><td>1.625</td><td>7.200</td><td>14.090</td><td>85.199</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="4">Q误差</td></tr><tr><td>50%</td><td>90%</td><td>95%</td><td>99%</td></tr><tr><td>自适应局部误差校正估计器（ALECE）</td><td>2.447</td><td>11.596</td><td>24.899</td><td>198.961</td></tr><tr><td>基线+随机图卷积（RGC）</td><td>1.745</td><td>7.879</td><td>15.954</td><td>68.847</td></tr><tr><td>基线+局部敏感哈希（LSH）</td><td>1.557</td><td>8.454</td><td>18.999</td><td>143.885</td></tr><tr><td>基线+高斯卡尔曼滤波（GKF）</td><td>1.570</td><td>7.307</td><td>14.419</td><td>148.007</td></tr><tr><td>基线+类激活映射（CAM）</td><td>2.111</td><td>10.001</td><td>22.654</td><td>166.228</td></tr><tr><td>递归图卷积网络（RLGCNt）</td><td>1.625</td><td>7.200</td><td>14.090</td><td>85.199</td></tr></tbody></table>

Table 4. Comparative experimental results on the STATS dataset-dynamic workloads.

表4. STATS数据集动态工作负载的对比实验结果。

<table><tr><td rowspan="2">Model</td><td colspan="4">Q-error</td><td rowspan="2">E2E Time (s)</td></tr><tr><td>50%</td><td>90%</td><td>95%</td><td>99%</td></tr><tr><td>PG</td><td>190</td><td>${1.4} \times  {10}^{5}$</td><td>${1.1} \times  {10}^{5}$</td><td>${1.8} \times  {10}^{7}$</td><td>7,790</td></tr><tr><td>Uni-Samp</td><td>1.35</td><td>12.47</td><td>$> {10}^{10}$</td><td>$> {10}^{10}$</td><td>6,002</td></tr><tr><td>NeuroCard</td><td>17.35</td><td>1,388</td><td>7,402</td><td>${3.0} \times  {10}^{5}$</td><td>$> {30},{000}$</td></tr><tr><td>FLAT</td><td>12.77</td><td>1,979</td><td>12,897</td><td>${8.6} \times  {10}^{5}$</td><td>> 30,000</td></tr><tr><td>FactorJoin</td><td>22.62</td><td>2,593</td><td>31,936</td><td>${1.6} \times  {10}^{6}$</td><td>> 30, 000</td></tr><tr><td>MLP</td><td>${2.4} \times  {10}^{6}$</td><td>$> {10}^{10}$</td><td>$> {10}^{10}$</td><td>$> {10}^{10}$</td><td>$> {30},{000}$</td></tr><tr><td>MSCN</td><td>20.09</td><td>2,870</td><td>17,037</td><td>${2.6} \times  {10}^{5}$</td><td>27,758</td></tr><tr><td>NNGP</td><td>9.88</td><td>827</td><td>4,652</td><td>${2.6} \times  {10}^{5}$</td><td>12,883</td></tr><tr><td>ALECE</td><td>2.447</td><td>11.596</td><td>24.899</td><td>198.961</td><td>2,901</td></tr><tr><td>RLGCNt</td><td>1.625</td><td>7.200</td><td>14.090</td><td>85.199</td><td>2,623</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="4">Q误差</td><td rowspan="2">端到端时间（秒）</td></tr><tr><td>50%</td><td>90%</td><td>95%</td><td>99%</td></tr><tr><td>策略梯度（Policy Gradient，PG）</td><td>190</td><td>${1.4} \times  {10}^{5}$</td><td>${1.1} \times  {10}^{5}$</td><td>${1.8} \times  {10}^{7}$</td><td>7,790</td></tr><tr><td>统一采样（Uniform Sampling，Uni - Samp）</td><td>1.35</td><td>12.47</td><td>$> {10}^{10}$</td><td>$> {10}^{10}$</td><td>6,002</td></tr><tr><td>神经卡片（NeuroCard）</td><td>17.35</td><td>1,388</td><td>7,402</td><td>${3.0} \times  {10}^{5}$</td><td>$> {30},{000}$</td></tr><tr><td>扁平网络（FLAT）</td><td>12.77</td><td>1,979</td><td>12,897</td><td>${8.6} \times  {10}^{5}$</td><td>> 30,000</td></tr><tr><td>因子连接（FactorJoin）</td><td>22.62</td><td>2,593</td><td>31,936</td><td>${1.6} \times  {10}^{6}$</td><td>> 30, 000</td></tr><tr><td>多层感知机（Multilayer Perceptron，MLP）</td><td>${2.4} \times  {10}^{6}$</td><td>$> {10}^{10}$</td><td>$> {10}^{10}$</td><td>$> {10}^{10}$</td><td>$> {30},{000}$</td></tr><tr><td>多阶段卷积网络（Multi - Stage Convolutional Network，MSCN）</td><td>20.09</td><td>2,870</td><td>17,037</td><td>${2.6} \times  {10}^{5}$</td><td>27,758</td></tr><tr><td>神经网络高斯过程（Neural Network Gaussian Process，NNGP）</td><td>9.88</td><td>827</td><td>4,652</td><td>${2.6} \times  {10}^{5}$</td><td>12,883</td></tr><tr><td>自适应学习与误差校正（Adaptive Learning and Error Correction，ALECE）</td><td>2.447</td><td>11.596</td><td>24.899</td><td>198.961</td><td>2,901</td></tr><tr><td>强化学习图卷积网络（Reinforcement Learning Graph Convolutional Network，RLGCNt）</td><td>1.625</td><td>7.200</td><td>14.090</td><td>85.199</td><td>2,623</td></tr></tbody></table>

<!-- Media -->

As can be seen from Table 4, although the traditional method Uni-Samp performed best in the first indicator, its overall accuracy is low, and the 99th percentile dropped significantly. The structure of RLGCNt proposed in this paper is similar to mainstream methods used for comparison. Compared with traditional cardinality estimation methods, such as PG, all four indicators have decreased significantly. From the last three indicators, RLGCNt reduces the cardinality estimation error to a lower range.

从表4可以看出，虽然传统方法Uni - Samp（统一采样）在第一个指标上表现最佳，但其整体准确率较低，且第99百分位数显著下降。本文提出的RLGCNt（基于因果注意力机制的基数估计模型）的结构与用于比较的主流方法相似。与传统的基数估计方法（如PG）相比，四个指标均显著下降。从最后三个指标来看，RLGCNt将基数估计误差降低到了更低的范围。

Compared with query-driven methods, such as MSCN and NNGP, RLGCNt has a significant impact in dealing with dynamic workloads, and it produces fewer extreme estimation values. Compared with data-driven methods, such as NeuroCard, it achieves similar results and demonstrates clear advantages in dynamic workloads.

与查询驱动的方法（如MSCN和NNGP）相比，RLGCNt在处理动态工作负载方面有显著影响，并且产生的极端估计值更少。与数据驱动的方法（如NeuroCard）相比，它取得了相似的结果，并且在动态工作负载方面显示出明显优势。

These results indicate that the accuracy of cardinality estimation has been significantly improved, demonstrating the effectiveness of the four optimization methods proposed in this paper for cardinality estimation.

这些结果表明，基数估计的准确性得到了显著提高，证明了本文提出的四种优化方法对基数估计的有效性。

Following the same principle applied to static loads, RLGCNt shortens the $\mathrm{E}2\mathrm{E}$ time for dynamic workloads.

遵循应用于静态负载的相同原则，RLGCNt缩短了动态工作负载的$\mathrm{E}2\mathrm{E}$时间。

## 7 Conclusion

## 7 结论

We propose a cardinality estimation model, RLGCNt, based on the causal attention mechanism. To demonstrate the effectiveness of our model, we conduct extensive experiments on static and dynamic workloads using the STATS dataset. The results show that RLGCNt performs faster and more accurately in cardinality estimation under both static and dynamic workloads. For future research, Fourier neural networks present a promising direction for cardinality estimation.

我们提出了一种基于因果注意力机制的基数估计模型RLGCNt。为了证明我们模型的有效性，我们使用STATS数据集对静态和动态工作负载进行了广泛的实验。结果表明，RLGCNt在静态和动态工作负载下的基数估计中都表现得更快、更准确。对于未来的研究，傅里叶神经网络为基数估计提供了一个有前景的方向。

## References

## 参考文献

1. Han, Y., Wu, Z., Wu, P., Zhu, R., Yang, J., Tan, L. W., Zeng, K., Cong, G., Qin, Y., Pfadler, A., et al.: Cardinality estimation in dbms: A comprehensive benchmark evaluation. arXiv preprint arXiv:2109.05877 (2021)

1. 韩，Y.，吴，Z.，吴，P.，朱，R.，杨，J.，谭，L. W.，曾，K.，丛，G.，秦，Y.，普法德勒，A.等：数据库管理系统中的基数估计：全面的基准评估。预印本arXiv:2109.05877 (2021)

2. Harmouch, H., Naumann, F.: Cardinality estimation: An experimental survey. Proceedings of the VLDB Endowment ${11}\left( 4\right) ,{499} - {512}\left( {2017}\right)$

2. 哈穆奇，H.，瑙曼，F.：基数估计：一项实验性调查。VLDB捐赠会议论文集 ${11}\left( 4\right) ,{499} - {512}\left( {2017}\right)$

3. Hilprecht, B., Schmidt, A., Kulessa, M., Molina, A., Kersting, K., Binnig, C.: Deepdb: Learn from data, not from queries! arXiv preprint arXiv:1909.00607 (2019)

3. 希尔普雷希特，B.，施密特，A.，库莱萨，M.，莫利纳，A.，克尔斯廷，K.，宾尼格，C.：Deepdb：从数据中学习，而不是从查询中学习！预印本arXiv:1909.00607 (2019)

4. Kipf, A., Kipf, T., Radke, B., Leis, V., Boncz, P., Kemper, A.: Learned cardinalities: Estimating correlated joins with deep learning. arXiv preprint arXiv:1809.00677 (2018)

4. 基普夫，A.，基普夫，T.，拉德克，B.，莱斯，V.，邦茨，P.，肯珀，A.：学习基数：使用深度学习估计相关连接。预印本arXiv:1809.00677 (2018)

5. Kipf, A., Vorona, D., Müller, J., Kipf, T., Radke, B., Leis, V., Boncz, P., Neumann, T., Kemper, A.: Estimating cardinalities with deep sketches. In: Proceedings of the 2019 International Conference on Management of Data. pp. 1937-1940 (2019)

5. 基普夫，A.，沃罗娜，D.，米勒，J.，基普夫，T.，拉德克，B.，莱斯，V.，邦茨，P.，诺伊曼，T.，肯珀，A.：使用深度草图估计基数。见：2019年国际数据管理会议论文集。第1937 - 1940页 (2019)

6. Leis, V., Radke, B., Gubichev, A., Kemper, A., Neumann, T.: Cardinality estimation done right: Index-based join sampling. In: Cidr (2017)

6. 莱斯，V.，拉德克，B.，古比切夫，A.，肯珀，A.，诺伊曼，T.：正确进行基数估计：基于索引的连接采样。见：Cidr (2017)

7. Li, P., Wei, W., Zhu, R., Ding, B., Zhou, J., Lu, H.: Alece: An attention-based learned cardinality estimator for spj queries on dynamic workloads. Proceedings of the VLDB Endowment $\mathbf{{17}}\left( 2\right) ,{197} - {210}\left( {2023}\right)$

7. 李，P.，魏，W.，朱，R.，丁，B.，周，J.，陆，H.：Alece：一种基于注意力的动态工作负载下SPJ查询的学习型基数估计器。VLDB捐赠会议论文集 $\mathbf{{17}}\left( 2\right) ,{197} - {210}\left( {2023}\right)$

8. Liang, X., Sintos, S., Shang, Z., Krishnan, S.: Combining aggregation and sampling (nearly) optimally for approximate query processing. In: Proceedings of the 2021 International Conference on Management of Data. pp. 1129-1141 (2021)

8. 梁，X.，辛托斯，S.，尚，Z.，克里什南，S.：（近乎）最优地结合聚合和采样进行近似查询处理。见：2021年国际数据管理会议论文集。第1129 - 1141页 (2021)

9. Lin, X., Zeng, X., Pu, X., Sun, Y., et al.: A cardinality estimation approach based on two level histograms. J. Inf. Sci. Eng. 31(5), 1733-1756 (2015)

9. 林，X.，曾，X.，蒲，X.，孙，Y.等：一种基于两级直方图的基数估计方法。信息科学与工程学报31(5)，1733 - 1756 (2015)

10. Ma, Q., Triantafillou, P.: Dbest: Revisiting approximate query processing engines with machine learning models. In: Proceedings of the 2019 International Conference on Management of Data. pp. 1553-1570 (2019)

10. 马（Ma），Q.，特里安塔菲卢（Triantafillou），P.：Dbest：用机器学习模型重新审视近似查询处理引擎。见：《2019 年国际数据管理会议论文集》。第 1553 - 1570 页（2019 年）

11. Marcus, R., Papaemmanouil, O.: Plan-structured deep neural network models for

11. 马库斯（Marcus），R.，帕帕埃马努伊尔（Papaemmanouil），O.：用于

query performance prediction. arXiv preprint arXiv:1902.00132 (2019)

查询性能预测的计划结构深度神经网络模型。预印本 arXiv:1902.00132（2019 年）

12. Moerkotte, G., Neumann, T., Steidl, G.: Preventing bad plans by bounding the impact of cardinality estimation errors. Proceedings of the VLDB Endowment $\mathbf{2}\left( 1\right) ,{982} - {993}\left( {2009}\right)$

12. 莫尔科特（Moerkotte），G.，诺伊曼（Neumann），T.，施泰德尔（Steidl），G.：通过限制基数估计误差的影响来防止产生糟糕的查询计划。《VLDB 捐赠会议论文集》$\mathbf{2}\left( 1\right) ,{982} - {993}\left( {2009}\right)$

13. Ning, Z., Zhang, Z., Sun, T., Tian, Y., Zhang, T., Li, T.J.J.: An empirical study of model errors and user error discovery and repair strategies in natural language database queries. In: Proceedings of the 28th International Conference on Intelligent User Interfaces. pp. 633-649 (2023)

13. 宁（Ning），Z.，张（Zhang），Z.，孙（Sun），T.，田（Tian），Y.，张（Zhang），T.，李（Li），T.J.J.：自然语言数据库查询中模型误差和用户错误发现与修复策略的实证研究。见：《第 28 届国际智能用户界面会议论文集》。第 633 - 649 页（2023 年）

14. Prasad, N.N., Rao, J.N.: The estimation of the mean squared error of small-area estimators. Journal of the American statistical association $\mathbf{{85}}\left( {409}\right) ,{163} - {171}\left( {1990}\right)$

14. 普拉萨德（Prasad），N.N.，拉奥（Rao），J.N.：小区域估计量均方误差的估计。《美国统计协会杂志》$\mathbf{{85}}\left( {409}\right) ,{163} - {171}\left( {1990}\right)$

15. Rumelhart, D.E., Hinton, G.E., Williams, R.J.: Learning internal representations by error propagation. pp. 318-362 (1986)

15. 鲁梅尔哈特（Rumelhart），D.E.，辛顿（Hinton），G.E.，威廉姆斯（Williams），R.J.：通过误差传播学习内部表示。第 318 - 362 页（1986 年）

16. Wang, J., Chai, C., Liu, J., Li, G.: Cardinality estimation using normalizing flow. The VLDB Journal $\mathbf{{33}}\left( 2\right) ,{323} - {348}\left( {2024}\right)$

16. 王（Wang），J.，柴（Chai），C.，刘（Liu），J.，李（Li），G.：使用归一化流进行基数估计。《VLDB 杂志》$\mathbf{{33}}\left( 2\right) ,{323} - {348}\left( {2024}\right)$

17. Woltmann, L., Olwig, D., Hartmann, C., Habich, D., Lehner, W.: Postcenn: post-gresql with machine learning models for cardinality estimation. Proceedings of the VLDB Endowment 14(12), 2715-2718 (2021)

17. 沃尔特曼（Woltmann），L.，奥尔维格（Olwig），D.，哈特曼（Hartmann），C.，哈比希（Habich），D.，勒纳（Lehner），W.：Postcenn：结合机器学习模型进行基数估计的 PostgreSQL。《VLDB 捐赠会议论文集》14(12)，2715 - 2718（2021 年）

18. Wu, Z., Negi, P., Alizadeh, M., Kraska, T., Madden, S.: Factorjoin: a new cardinality estimation framework for join queries. Proceedings of the ACM on Management of Data $\mathbf{1}\left( 1\right) ,1 - {27}\left( {2023}\right)$

18. 吴（Wu），Z.，内吉（Negi），P.，阿里扎德（Alizadeh），M.，克拉斯卡（Kraska），T.，马登（Madden），S.：Factorjoin：一种用于连接查询的新型基数估计框架。《ACM 数据管理会议论文集》$\mathbf{1}\left( 1\right) ,1 - {27}\left( {2023}\right)$

19. Yang, Z., Kamsetty, A., Luan, S., Liang, E., Duan, Y., Chen, X., Stoica, I.: Neu-rocard: one cardinality estimator for all tables. arXiv preprint arXiv:2006.08109 (2020)

19. 杨（Yang），Z.，卡姆塞蒂（Kamsetty），A.，栾（Luan），S.，梁（Liang），E.，段（Duan），Y.，陈（Chen），X.，斯托伊卡（Stoica），I.：Neurocard：适用于所有表的单一基数估计器。预印本 arXiv:2006.08109（2020 年）

20. Yang, Z., Liang, E., Kamsetty, A., Wu, C., Duan, Y., Chen, X., Abbeel, P., Heller-stein, J.M., Krishnan, S., Stoica, I.: Deep unsupervised cardinality estimation. arXiv preprint arXiv:1905.04278 (2019)

20. 杨（Yang），Z.，梁（Liang），E.，卡姆塞蒂（Kamsetty），A.，吴（Wu），C.，段（Duan），Y.，陈（Chen），X.，阿贝埃尔（Abbeel），P.，赫勒斯坦（Hellerstein），J.M.，克里什南（Krishnan），S.，斯托伊卡（Stoica），I.：深度无监督基数估计。预印本 arXiv:1905.04278（2019 年）

21. Zeng, T., Lan, J., Ma, J., Wei, W., Zhu, R., Li, P., Ding, B., Lian, D., Wei, Z., Zhou, J.: Price: a pretrained model for cross-database cardinality estimation. arXiv preprint arXiv:2406.01027 (2024)

21. 曾（Zeng），T.，兰（Lan），J.，马（Ma），J.，魏（Wei），W.，朱（Zhu），R.，李（Li），P.，丁（Ding），B.，连（Lian），D.，魏（Wei），Z.，周（Zhou），J.：Price：一种用于跨数据库基数估计的预训练模型。预印本 arXiv:2406.01027（2024 年）

22. Zhang, Y., Chen, C., Li, Z., Ding, T., Wu, C., Ye, Y., Luo, Z.Q., Sun, R.: Adam-mini: Use fewer learning rates to gain more. arXiv preprint arXiv:2406.16793 (2024)

22. 张（Zhang），Y.，陈（Chen），C.，李（Li），Z.，丁（Ding），T.，吴（Wu），C.，叶（Ye），Y.，罗（Luo），Z.Q.，孙（Sun），R.：Adam - mini：用更少的学习率获得更多收益。预印本 arXiv:2406.16793（2024 年）

23. Zhao, K., Yu, J.X., He, Z., Li, R., Zhang, H.: Lightweight and accurate cardinality estimation by neural network gaussian process. In: Proceedings of the 2022 International Conference on Management of Data. pp. 973-987 (2022)

23. 赵（Zhao），K.，余（Yu），J.X.，何（He），Z.，李（Li），R.，张（Zhang），H.：通过神经网络高斯过程实现轻量级且准确的基数估计。见：《2022 年国际数据管理会议论文集》。第 973 - 987 页（2022 年）

24. Zhu, R., Wu, Z., Han, Y., Zeng, K., Pfadler, A., Qian, Z., Zhou, J., Cui, B.: Flat: fast, lightweight and accurate method for cardinality estimation. arXiv preprint arXiv:2011.09022 (2020)

24. 朱（Zhu），R.，吴（Wu），Z.，韩（Han），Y.，曾（Zeng），K.，普法德勒（Pfadler），A.，钱（Qian），Z.，周（Zhou），J.，崔（Cui），B.：Flat：一种快速、轻量级且准确的基数估计方法。预印本 arXiv:2011.09022（2020 年）