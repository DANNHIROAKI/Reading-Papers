# PYRAFORMER: LOW-COMPLEXITY PYRAMIDAL AT- TENTION FOR LONG-RANGE TIME SERIES MODELING AND FORECASTING

# PYRAFORMER：用于长序列时间序列建模与预测的低复杂度金字塔注意力机制

Shizhan Liu ${}^{1,2 * }$ ,Hang Yu ${}^{1}$ ; Cong Liao ${}^{1}$ ,Jianguo Li ${}^{1}$ ; Weiyao Lin ${}^{2}$ ,Alex X. Liu ${}^{1}$ ,

刘世展 ${}^{1,2 * }$，余航 ${}^{1}$；廖聪 ${}^{1}$，李建国 ${}^{1}$；林伟耀 ${}^{2}$，刘亚历克斯·X ${}^{1}$，

and Schahram Dustdar ${}^{3}$

以及沙赫拉姆·达斯塔尔 ${}^{3}$

${}^{1}$ Ant Group, ${}^{2}$ Shanghai Jiaotong University, ${}^{3}$ TU Wien,Austria

${}^{1}$ 蚂蚁集团，${}^{2}$ 上海交通大学，${}^{3}$ 维也纳工业大学，奥地利

## Abstract

## 摘要

Accurate prediction of the future given the past based on time series data is of paramount importance, since it opens the door for decision making and risk management ahead of time. In practice, the challenge is to build a flexible but parsimonious model that can capture a wide range of temporal dependencies. In this paper, we propose Pyraformer by exploring the multi-resolution representation of the time series. Specifically, we introduce the pyramidal attention module (PAM) in which the inter-scale tree structure summarizes features at different resolutions and the intra-scale neighboring connections model the temporal dependencies of different ranges. Under mild conditions, the maximum length of the signal traversing path in Pyraformer is a constant (i.e., $\mathcal{O}\left( 1\right)$ ) with regard to the sequence length $L$ ,while its time and space complexity scale linearly with $L$ . Extensive experimental results show that Pyraformer typically achieves the highest prediction accuracy in both single-step and long-range multi-step forecasting tasks with the least amount of time and memory consumption, especially when the sequence is long ${}^{1}$

基于时间序列数据，根据过去准确预测未来至关重要，因为这为提前进行决策和风险管理创造了条件。在实践中，挑战在于构建一个灵活且简洁的模型，该模型能够捕捉广泛的时间依赖关系。在本文中，我们通过探索时间序列的多分辨率表示提出了金字塔变换器（Pyraformer）。具体而言，我们引入了金字塔注意力模块（PAM），其中跨尺度树结构总结了不同分辨率下的特征，而尺度内相邻连接对不同范围的时间依赖关系进行建模。在温和条件下，金字塔变换器中信号遍历路径的最大长度相对于序列长度 $L$ 是一个常数（即 $\mathcal{O}\left( 1\right)$ ），而其时间和空间复杂度与 $L$ 呈线性关系。大量实验结果表明，金字塔变换器通常在单步和长程多步预测任务中以最少的时间和内存消耗实现最高的预测精度，尤其是在序列较长时 ${}^{1}$

## 1 INTRODUCTION

## 1 引言

Time series forecasting is the cornerstone for downstream tasks such as decision making and risk management. As an example, reliable prediction of the online traffic for micro-services can yield early warnings of the potential risk in cloud systems. Furthermore, it also provides guidance for dynamic resource allocation, in order to minimize the cost without degrading the performance. In addition to online traffic, time series forecasting has also found vast applications in other fields, including disease propagation, energy management, and economics and finance.

时间序列预测是决策和风险管理等下游任务的基石。例如，对微服务在线流量的可靠预测可以对云系统中的潜在风险发出早期预警。此外，它还为动态资源分配提供指导，以便在不降低性能的情况下将成本降至最低。除了在线流量，时间序列预测在其他领域也有广泛的应用，包括疾病传播、能源管理以及经济和金融。

The major challenge of time series forecasting lies in constructing a powerful but parsimonious model that can compactly capture temporal dependencies of different ranges. Time series often exhibit both short-term and long-term repeating patterns (Lai et al. 2018), and taking them into account is the key to accurate prediction. Of particular note is the more difficult task of handling long-range dependencies, which is characterized by the length of the longest signal traversing path (see Proposition 2 for the definition) between any two positions in the time series (Vaswani et al. 2017). The shorter the path, the better the dependencies are captured. Additionally, to allow the models to learn these long-term patterns, the historical input to the models should also be long. To this end, low time and space complexity is a priority.

时间序列预测的主要挑战在于构建一个强大而简洁的模型，该模型能够紧凑地捕捉不同范围的时间依赖关系。时间序列通常呈现出短期和长期的重复模式（赖等人，2018年），考虑这些模式是准确预测的关键。特别值得注意的是，处理长距离依赖关系是一项更具挑战性的任务，其特征是时间序列中任意两个位置之间最长信号遍历路径的长度（长距离依赖关系的定义见命题2）（瓦斯瓦尼等人，2017年）。路径越短，对依赖关系的捕捉就越好。此外，为了让模型学习这些长期模式，模型的历史输入也应该足够长。为此，低时间和空间复杂度是首要考虑因素。

Unfortunately, the present state-of-the-art methods fail to accomplish these two objectives simultaneously. On one end, RNN (Salinas et al. 2020) and CNN (Munir et al. 2018) achieve a low time complexity that is linear in terms of the time series length $L$ ,yet their maximum length of the signal traversing path is $\mathcal{O}\left( L\right)$ ,thus rendering them difficult to learn dependencies between distant positions. On the other extreme,Transformer dramatically shortens the maximum path to be $\mathcal{O}\left( 1\right)$

遗憾的是，目前最先进的方法无法同时实现这两个目标。一方面，循环神经网络（RNN，萨利纳斯等人，2020年）和卷积神经网络（CNN，穆尼尔等人，2018年）实现了较低的时间复杂度，该复杂度相对于时间序列长度 $L$ 呈线性关系，但它们信号遍历路径的最大长度为 $\mathcal{O}\left( L\right)$ ，因此难以学习远距离位置之间的依赖关系。另一方面，Transformer 显著将最大路径缩短至 $\mathcal{O}\left( 1\right)$

---

<!-- Footnote -->

*Equal contribution. This work was done when Shizhan Liu was a research intern at Ant Group.

*同等贡献。这项工作是刘世展在蚂蚁集团担任研究实习生时完成的。

${}^{ \dagger  }$ Corresponding author

${}^{ \dagger  }$ 通讯作者

${}^{1}$ Code is available at: https://github.com/alipay/Pyraformer

${}^{1}$ 代码可在以下链接获取：https://github.com/alipay/Pyraformer

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Connection example Maximum signal traversing path (c) RNN Layer 3 Embeddings Layer 2 Layer 1 Embeddings Embeddings (f) LogTrans Layer 3 Embeddings Layer 2 Embeddings Layer 1 Embeddings Input (a) Full Attention (b) CNN (d) Pyraformer (e) ETC -->

<img src="https://cdn.noedgeai.com/01957b4b-1654-7ad6-8e71-2da613767aa1_1.jpg?x=305&y=229&w=1194&h=515&r=0"/>

Figure 1: Graphs of commonly used neural network models for sequence data.

图 1：用于序列数据的常用神经网络模型图。

Table 1: Comparison of the complexity and the maximum signal traveling path for different models, where $G$ is the number of global tokens in ETC. In practice,the $G$ increases with $L$ ,and so the complexity of ETC is super-linear.

表1：不同模型的复杂度和最大信号传播路径对比，其中 $G$ 是高效变压器压缩（ETC）中的全局标记数量。实际上，$G$ 随 $L$ 增加，因此高效变压器压缩（ETC）的复杂度呈超线性。

<table><tr><td>$\mathbf{{Method}}$</td><td>Complexity per layer</td><td>Maximum path length</td></tr><tr><td>CNN (Munir et al., 2018)</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td></tr><tr><td>RNN (Salinas et al., 2020)</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td></tr><tr><td>Full-Attention (Vaswani et al., 2017)</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$\mathcal{O}\left( 1\right)$</td></tr><tr><td>ETC (Ainslie et al., 2020)</td><td>$\mathcal{O}\left( {GL}\right)$</td><td>$\mathcal{O}\left( 1\right)$</td></tr><tr><td>Longformer (Beltagy et al., 2020)</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td></tr><tr><td>LogTrans (Li et al., 2019)</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {\log L}\right)$</td></tr><tr><td>Pyraformer</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( 1\right)$</td></tr></table>

<table><tbody><tr><td>$\mathbf{{Method}}$</td><td>每层复杂度</td><td>最大路径长度</td></tr><tr><td>卷积神经网络（CNN，穆尼尔等人，2018年）</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td></tr><tr><td>循环神经网络（RNN，萨利纳斯等人，2020年）</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td></tr><tr><td>全注意力机制（Full - Attention，瓦斯瓦尼等人，2017年）</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$\mathcal{O}\left( 1\right)$</td></tr><tr><td>增强型变换器压缩模型（ETC，安斯利等人，2020年）</td><td>$\mathcal{O}\left( {GL}\right)$</td><td>$\mathcal{O}\left( 1\right)$</td></tr><tr><td>长序列变换器（Longformer，贝尔塔吉等人，2020年）</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td></tr><tr><td>对数变换器（LogTrans，李等人，2019年）</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {\log L}\right)$</td></tr><tr><td>金字塔变换器（Pyraformer）</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( 1\right)$</td></tr></tbody></table>

<!-- Media -->

at the sacrifice of increasing the time complexity to $\mathcal{O}\left( {L}^{2}\right)$ . As a consequence,it cannot tackle very long sequences. To find a compromise between the model capacity and complexity, variants of Transformer are proposed, such as Longformer (Beltagy et al., 2020), Reformer (Kitaev et al., 2019), and Informer (Zhou et al. 2021). However, few of them can achieve a maximum path length less than $\mathcal{O}\left( L\right)$ while greatly reducing the time and space complexity.

以将时间复杂度增加到$\mathcal{O}\left( {L}^{2}\right)$为代价。因此，它无法处理非常长的序列。为了在模型容量和复杂度之间找到平衡，人们提出了Transformer的变体，如Longformer（贝尔塔吉等人，2020年）、Reformer（基塔耶夫等人，2019年）和Informer（周等人，2021年）。然而，它们中很少有能在大幅降低时间和空间复杂度的同时，实现小于$\mathcal{O}\left( L\right)$的最大路径长度。

In this paper, we propose a novel pyramidal attention based Transformer (Pyraformer) to bridge the gap between capturing the long-range dependencies and achieving a low time and space complexity. Specifically, we develop the pyramidal attention mechanism by passing messages based on attention in the pyramidal graph as shown in Figure 1(d). The edges in this graph can be divided into two groups: the inter-scale and the intra-scale connections. The inter-scale connections build a multiresolution representation of the original sequence: nodes at the finest scale correspond to the time points in the original time series (e.g., hourly observations), while nodes in the coarser scales represent features with lower resolutions (e.g., daily, weekly, and monthly patterns). Such latent coarser-scale nodes are initially introduced via a coarser-scale construction module. On the other hand, the intra-scale edges capture the temporal dependencies at each resolution by connecting neighboring nodes together. As a result, this model provides a compact representation for long-range temporal dependencies among far-apart positions by capturing such behavior at coarser resolutions, leading to a smaller length of the signal traversing path. Moreover, modeling temporal dependencies of different ranges at different scales with sparse neighboring intra-scale connections significantly reduces the computational cost. In short, our key contributions comprise:

在本文中，我们提出了一种基于金字塔注意力机制的新型Transformer模型（Pyraformer），以弥合捕捉长距离依赖关系与实现低时间和空间复杂度之间的差距。具体而言，我们通过在如图1(d)所示的金字塔图中基于注意力传递消息来开发金字塔注意力机制。该图中的边可以分为两组：跨尺度连接和同尺度连接。跨尺度连接构建了原始序列的多分辨率表示：最精细尺度的节点对应于原始时间序列中的时间点（例如，每小时的观测值），而较粗尺度的节点表示分辨率较低的特征（例如，每日、每周和每月的模式）。这种潜在的较粗尺度节点最初是通过较粗尺度构建模块引入的。另一方面，同尺度边通过将相邻节点连接在一起，捕捉每个分辨率下的时间依赖关系。因此，该模型通过在较粗分辨率下捕捉远距离位置之间的长距离时间依赖关系，提供了一种紧凑的表示，从而缩短了信号遍历路径的长度。此外，利用稀疏的相邻同尺度连接在不同尺度上对不同范围的时间依赖关系进行建模，显著降低了计算成本。简而言之，我们的主要贡献包括：

- We propose Pyraformer to simultaneously capture temporal dependencies of different ranges in a compact multi-resolution fashion. To distinguish Pyraformer from the state-of-the-art methods, we summarize all models from the perspective of graphs in Figure 1.

- 我们提出了金字塔变换器（Pyraformer），以紧凑的多分辨率方式同时捕捉不同范围的时间依赖关系。为了将金字塔变换器（Pyraformer）与最先进的方法区分开来，我们在图1中从图论的角度总结了所有模型。

- Theoretically, we prove that by choosing parameters appropriately, the maximum path length of $\mathcal{O}\left( 1\right)$ and the time and space complexity of $\mathcal{O}\left( L\right)$ can be reached concurrently. To highlight the appeal of the proposed model, we further compare different models in terms of the maximum path and the complexity in Table 1

- 从理论上讲，我们证明了通过适当选择参数，可以同时达到$\mathcal{O}\left( 1\right)$的最大路径长度以及$\mathcal{O}\left( L\right)$的时间和空间复杂度。为了突出所提出模型的优势，我们在表1中进一步比较了不同模型在最大路径和复杂度方面的差异。

- Experimentally, we show that the proposed Pyraformer yields more accurate predictions than the original Transformer and its variants on various real-world datasets under the scenario of both single-step and long-range multi-step forecasting, but with lower time and memory cost.

- 通过实验，我们表明，在单步和长程多步预测场景下，所提出的金字塔变换器（Pyraformer）在各种真实世界数据集上的预测比原始的变换器（Transformer）及其变体更准确，而且时间和内存成本更低。

## 2 RELATED WORKS

## 2 相关工作

### 2.1 TIME SERIES FORECASTING

### 2.1 时间序列预测

Time series forecasting methods can be roughly divided into statistical methods and neural network based methods. The first group involves ARIMA (Box & Jenkins, 1968) and Prophet (Taylor & Letham 2018). However, both of them need to fit each time series separately, and their performance pales when it comes to long-range forecasting.

时间序列预测方法大致可分为统计方法和基于神经网络的方法。第一类方法包括自回归积分滑动平均模型（ARIMA，Box & Jenkins，1968年）和先知模型（Prophet，Taylor & Letham 2018年）。然而，这两种方法都需要分别拟合每个时间序列，并且在进行长期预测时，它们的表现不佳。

More recently, the development of deep learning has spawned a tremendous increase in neural network based time series forecasting methods, including CNN (Munir et al., 2018), RNN (Salinas et al. 2020) and Transformer (Li et al. 2019). As mentioned in the previous section, CNN and RNN enjoy a low time and space complexity (i.e., $\mathcal{O}\left( L\right)$ ),but entail a path of $\mathcal{O}\left( L\right)$ to describe long-range dependence. We refer the readers to Appendix A for a more detailed review on related RNN-based models. By contrast, Transformer (Vaswani et al. 2017) can effectively capture the long-range dependence with a path of $\mathcal{O}\left( 1\right)$ steps,whereas the complexity increases vastly from $\mathcal{O}\left( L\right)$ to $\mathcal{O}\left( {L}^{2}\right)$ . To alleviate this computational burden, LogTrans (Li et al., 2019) and Informer (Zhou et al., 2021) are proposed: the former constrains that each point in the sequence can only attend to the point that is ${2}^{n}$ steps before it,where $n = 1,2,\cdots$ ,and the latter utilizes the sparsity of the attention score, resulting in substantial decrease in the complexity (i.e., $\mathcal{O}\left( {L\log L}\right)$ at the expense of introducing a longer maximum path length.

近年来，深度学习的发展促使基于神经网络的时间序列预测方法大幅增加，包括卷积神经网络（CNN，穆尼尔等人，2018年）、循环神经网络（RNN，萨利纳斯等人，2020年）和Transformer（李等人，2019年）。如前一节所述，卷积神经网络和循环神经网络具有较低的时间和空间复杂度（即$\mathcal{O}\left( L\right)$），但需要$\mathcal{O}\left( L\right)$的路径来描述长程依赖关系。有关基于循环神经网络的相关模型的更详细综述，请读者参考附录A。相比之下，Transformer（瓦斯瓦尼等人，2017年）可以通过$\mathcal{O}\left( 1\right)$步的路径有效捕捉长程依赖关系，但其复杂度从$\mathcal{O}\left( L\right)$大幅增加到$\mathcal{O}\left( {L}^{2}\right)$。为了减轻这种计算负担，人们提出了LogTrans（李等人，2019年）和Informer（周等人，2021年）：前者限制序列中的每个点只能关注其前${2}^{n}$步的点，其中$n = 1,2,\cdots$；后者利用注意力得分的稀疏性，大幅降低了复杂度（即$\mathcal{O}\left( {L\log L}\right)$），但代价是引入了更长的最大路径长度。

### 2.2 SPARSE TRANSFORMERS

### 2.2 稀疏变压器（Sparse Transformers）

In addition to the literature on time series forecasting, a plethora of methods have been proposed for enhancing the efficiency of Transformer in the field of natural language processing (NLP). Similar to CNN, Longformer (Beltagy et al. 2020) computes attention within a local sliding window or a dilated sliding window. Although the complexity is reduced to $\mathcal{O}\left( {AL}\right)$ ,where $A$ is the local window size, the limited window size makes it difficult to exchange information globally. The consequent maximum path length is $\mathcal{O}\left( {L/A}\right)$ . As an alternative,Reformer (Kitaev et al. 2019) exploits locality sensitive hashing (LSH) to divide the sequence into several buckets, and then performs attention within each bucket. It also employs reversible Transformer to further reduce memory consumption, and so an extremely long sequence can be processed. Its maximum path length is proportional to the number of buckets though, and worse still, a large bucket number is required to reduce the complexity. On the other hand, ETC (Ainslie et al., 2020) introduces an extra set of global tokens for the sake of global information exchange,leading to an $\mathcal{O}\left( {GL}\right)$ time and space complexity and an $\mathcal{O}\left( 1\right)$ maximum path length,where $G$ is the number of global tokens. However, $G$ typically increases with $L$ ,and the consequent complexity is still super-linear. Akin to ETC,the proposed Pyraformer also introduces global tokens, but in a multiscale manner, successfully reducing the complexity to $\mathcal{O}\left( L\right)$ without increasing the order of the maximum path length as in the original Transformer.

除了时间序列预测方面的文献外，在自然语言处理（NLP）领域，人们还提出了大量提高Transformer效率的方法。与卷积神经网络（CNN）类似，长序列Transformer（Longformer，贝尔塔吉等人，2020年）在局部滑动窗口或扩张滑动窗口内计算注意力。虽然复杂度降低到了$\mathcal{O}\left( {AL}\right)$，其中$A$是局部窗口大小，但有限的窗口大小使得全局信息交换变得困难。由此产生的最大路径长度为$\mathcal{O}\left( {L/A}\right)$。作为另一种选择，改革者Transformer（Reformer，基塔耶夫等人，2019年）利用局部敏感哈希（LSH）将序列划分为多个桶，然后在每个桶内执行注意力机制。它还采用可逆Transformer进一步减少内存消耗，因此可以处理极长的序列。不过，其最大路径长度与桶的数量成正比，更糟糕的是，需要大量的桶来降低复杂度。另一方面，增强型Transformer（ETC，安斯利等人，2020年）引入了一组额外的全局标记，以实现全局信息交换，这导致了$\mathcal{O}\left( {GL}\right)$的时间和空间复杂度以及$\mathcal{O}\left( 1\right)$的最大路径长度，其中$G$是全局标记的数量。然而，$G$通常会随着$L$的增加而增加，因此产生的复杂度仍然是超线性的。与ETC类似，本文提出的金字塔Transformer（Pyraformer）也引入了全局标记，但采用了多尺度的方式，成功地将复杂度降低到了$\mathcal{O}\left( L\right)$，且不会像原始Transformer那样增加最大路径长度的阶数。

### 2.3 HIERARCHICAL TRANSFORMERS

### 2.3 分层Transformer模型（Hierarchical Transformers）

Finally, we provide a brief review on methods that improve Transformer's ability to capture the hierarchical structure of natural language, although they have never been used for time series forecasting. HIBERT (Miculicich et al. 2018) first uses a Sent Encoder to extract the features of a sentence, and then forms the EOS tokens of sentences in the document as a new sequence and input it into the Doc Encoder. However, it is specialized for natural language and cannot be generalized to other sequence data. Multi-scale Transformer (Subramanian et al., 2020) learns the multi-scale representations of sequence data using both the top-down and bottom-up network structures. Such multi-scale representations help reduce the time and memory cost of the original Transformer, but it still suffers from the pitfall of the quadratic complexity. Alternatively, BP-Transformer (Ye et al. 2019) recursively partitions the entire input sequence into two until a partition only contains a single token. The partitioned sequences then form a binary tree. In the attention layer, each upper-scale node can attend to its own children,while the nodes at the bottom scale can attend to the adjacent $A$ nodes at the same scale and all coarser-scale nodes. Note that BP-Transformer initializes the nodes at coarser scale with zeros, whereas Pyraformer introduces the coarser-scale nodes using a construction module in a more flexible manner. Moreover, BP-Transformer is associated with a denser graph than Pyraformer,thus giving rise to a higher complexity of $\mathcal{O}\left( {L\log L}\right)$ .

最后，我们简要回顾了一些提高Transformer捕捉自然语言层次结构能力的方法，尽管这些方法从未用于时间序列预测。HIBERT（米库利奇等人，2018年）首先使用句子编码器（Sent Encoder）提取句子特征，然后将文档中句子的结束符（EOS）标记组成一个新序列，并将其输入文档编码器（Doc Encoder）。然而，它是专门为自然语言设计的，无法推广到其他序列数据。多尺度Transformer（苏布拉马尼亚姆等人，2020年）使用自上而下和自下而上的网络结构学习序列数据的多尺度表示。这种多尺度表示有助于降低原始Transformer的时间和内存成本，但它仍然存在二次复杂度的问题。另外，BP - Transformer（叶等人，2019年）将整个输入序列递归地划分为两部分，直到每个分区只包含一个标记。划分后的序列形成一棵二叉树。在注意力层中，每个上一级尺度的节点可以关注其自身的子节点，而最底层尺度的节点可以关注同一尺度的相邻$A$节点以及所有更粗尺度的节点。请注意，BP - Transformer用零初始化更粗尺度的节点，而金字塔Transformer（Pyraformer）使用构建模块以更灵活的方式引入更粗尺度的节点。此外，BP - Transformer关联的图比金字塔Transformer更密集，因此导致$\mathcal{O}\left( {L\log L}\right)$复杂度更高。

<!-- Media -->

<!-- figureText: Positional Prediction Strategy 1 Add & Norm Gather Output Features Linear Predictions 4 Feed Forward Prediction Strategy 2 Add & Norm A Decoder Linear Predictions PAM Encoding N× Observations Observation Embedding CSCM Covariates Covariates Embedding -->

<img src="https://cdn.noedgeai.com/01957b4b-1654-7ad6-8e71-2da613767aa1_3.jpg?x=305&y=227&w=1188&h=337&r=0"/>

Figure 2: The architecture of Pyraformer: The CSCM summarizes the embedded sequence at different scales and builds a multi-resolution tree structure. Then the PAM is used to exchange information between nodes efficiently.

图2：Pyraformer（金字塔变换器）的架构：跨尺度上下文模块（CSCM）在不同尺度上总结嵌入序列，并构建多分辨率树结构。然后使用位置感知模块（PAM）在节点之间高效地交换信息。

<!-- Media -->

## 3 METHOD

## 3 方法

The time series forecasting problem can be formulated as predicting the future $M$ steps ${\mathbf{z}}_{t + 1 : t + M}$ given the previous $L$ steps of observations ${\mathbf{z}}_{t - L + 1 : t}$ and the associated covariates ${\mathbf{x}}_{t - L + 1 : t + M}$ (e.g., hour-of-the-day). To move forward to this goal, we propose Pyraformer in this paper, whose overall architecture is summarized in Figure 2 As shown in the figure, we first embed the observed data, the covariates, and the positions separately and then add them together, in the same vein with Informer (Zhou et al. 2021). Next,we construct a multi-resolution $C$ -ary tree using the coarser-scale construction module (CSCM), where nodes at a coarser scale summarize the information of $C$ nodes at the corresponding finer scale. To further capture the temporal dependencies of different ranges, we introduce the pyramidal attention module (PAM) by passing messages using the attention mechanism in the pyramidal graph. Finally, depending on the downstream task, we employ different network structures to output the final predictions. In the sequel, we elaborate on each part of the proposed model. For ease of exposition, all notations in this paper are summarized in Table 4.

时间序列预测问题可以表述为：在给定先前$L$步观测值${\mathbf{z}}_{t - L + 1 : t}$以及相关协变量${\mathbf{x}}_{t - L + 1 : t + M}$（例如一天中的小时数）的情况下，预测未来$M$步${\mathbf{z}}_{t + 1 : t + M}$的值。为了实现这一目标，本文提出了Pyraformer（金字塔变换器），其整体架构总结如图2所示。如图所示，与Informer（周等人，2021年）类似，我们首先分别对观测数据、协变量和位置进行嵌入，然后将它们相加。接下来，我们使用粗尺度构建模块（CSCM）构建一个多分辨率$C$叉树，其中较粗尺度的节点总结了相应较细尺度下$C$个节点的信息。为了进一步捕捉不同范围的时间依赖关系，我们通过在金字塔图中使用注意力机制传递消息，引入了金字塔注意力模块（PAM）。最后，根据下游任务，我们采用不同的网络结构输出最终预测结果。接下来，我们将详细阐述所提出模型的各个部分。为便于阐述，本文中所有符号总结于表4。

### 3.1 Pyramidal Attention Module (PAM)

### 3.1 金字塔注意力模块（Pyramidal Attention Module，PAM）

We begin with the introduction of the PAM, since it lies at the heart of Pyraformer. As demonstrated in Figure 1(d), we leverage a pyramidal graph to describe the temporal dependencies of the observed time series in a multiresolution fashion. Such a multiresolution structure has proved itself an effective and efficient tool for long-range interaction modeling in the field of computer vision (Sun et al. 2019, Wang et al., 2021) and statistical signal processing (Choi et al., 2008; Yu et al., 2019). We can decompose the pyramidal graph into two parts: the inter-scale and the intra-scale connections. The inter-scale connections form a $C$ -ary tree,in which each parent has $C$ children. For example, if we associate the finest scale of the pyramidal graph with hourly observations of the original time series, the nodes at coarser scales can be regarded as the daily, weekly, and even monthly features of the time series. As a consequence, the pyramidal graph offers a multi-resolution representation of the original time series. Furthermore, it is easier to capture long-range dependencies (e.g., monthly dependence) in the coarser scales by simply connecting the neighboring nodes via the intra-scale connections. In other words, the coarser scales are instrumental in describing long-range correlations in a manner that is graphically far more parsimonious than could be solely captured with a single, finest scale model. Indeed, the original single-scale Transformer (see Figure 1(a)) adopts a full graph that connects every two nodes at the finest scale so as to model the long-range dependencies, leading to a computationally burdensome model with $\mathcal{O}\left( {L}^{2}\right)$ time and space complexity (Vaswani et al. 2017). In stark contrast, as illustrated below, the pyramidal graph in the proposed Pyraformer reduces the computational cost to $\mathcal{O}\left( L\right)$ without increasing the order of the maximum length of the signal traversing path.

我们首先介绍金字塔注意力模块（PAM），因为它是金字塔变换器（Pyraformer）的核心。如图1(d)所示，我们利用金字塔图以多分辨率的方式描述观测时间序列的时间依赖关系。这种多分辨率结构已被证明是计算机视觉领域（孙等人，2019；王等人，2021）和统计信号处理领域（崔等人，2008；余等人，2019）中进行长距离交互建模的一种有效工具。我们可以将金字塔图分解为两部分：跨尺度连接和尺度内连接。跨尺度连接形成一个$C$叉树，其中每个父节点有$C$个子节点。例如，如果我们将金字塔图的最精细尺度与原始时间序列的每小时观测值相关联，那么较粗尺度上的节点可以被视为时间序列的每日、每周甚至每月特征。因此，金字塔图为原始时间序列提供了一种多分辨率表示。此外，通过尺度内连接简单地连接相邻节点，更容易在较粗尺度上捕捉长距离依赖关系（例如，每月依赖关系）。换句话说，较粗尺度有助于以一种比仅使用单一最精细尺度模型更简洁的图形方式描述长距离相关性。实际上，原始的单尺度变换器（见图1(a)）采用了一个全连接图，在最精细尺度上连接每两个节点，以便对长距离依赖关系进行建模，这导致了一个计算负担沉重的模型，其时间和空间复杂度为$\mathcal{O}\left( {L}^{2}\right)$（瓦斯瓦尼等人，2017）。与之形成鲜明对比的是，如下所示，所提出的金字塔变换器中的金字塔图将计算成本降低到$\mathcal{O}\left( L\right)$，而不增加信号遍历路径的最大长度的阶数。

Before delving into the PAM,we first introduce the original attention mechanism. Let $\mathbf{X}$ and $\mathbf{Y}$ denote the input and output of a single attention head respectively. Note that multiple heads can be introduced to describe the temporal pattern from different perspectives. $\mathbf{X}$ is first linearly transformed into three distinct matrices,namely,the query $\mathbf{Q} = \mathbf{X}{\mathbf{W}}_{Q}$ ,the key $\mathbf{K} = \mathbf{X}{\mathbf{W}}_{K}$ ,and the value $\mathbf{V} = \mathbf{X}{\mathbf{W}}_{V}$ ,where ${\mathbf{W}}_{Q},{\mathbf{W}}_{K},{\mathbf{W}}_{V} \in  {\mathbb{R}}^{L \times  {D}_{K}}$ . For the $i$ -th row ${\mathbf{q}}_{i}$ in $\mathbf{Q}$ ,it can attend to any rows (i.e.,keys) in $\mathbf{K}$ . In other words,the corresponding output ${\mathbf{y}}_{i}$ can be expressed as:

在深入探讨PAM（位置注意力模块，Position Attention Module）之前，我们首先介绍原始的注意力机制。分别用$\mathbf{X}$和$\mathbf{Y}$表示单个注意力头的输入和输出。请注意，可以引入多个注意力头，以便从不同的视角描述时间模式。首先将$\mathbf{X}$线性变换为三个不同的矩阵，即查询矩阵$\mathbf{Q} = \mathbf{X}{\mathbf{W}}_{Q}$、键矩阵$\mathbf{K} = \mathbf{X}{\mathbf{W}}_{K}$和值矩阵$\mathbf{V} = \mathbf{X}{\mathbf{W}}_{V}$，其中${\mathbf{W}}_{Q},{\mathbf{W}}_{K},{\mathbf{W}}_{V} \in  {\mathbb{R}}^{L \times  {D}_{K}}$。对于$\mathbf{Q}$中的第$i$行${\mathbf{q}}_{i}$，它可以关注$\mathbf{K}$中的任意行（即键）。换句话说，相应的输出${\mathbf{y}}_{i}$可以表示为：

$$
{\mathbf{y}}_{i} = \mathop{\sum }\limits_{{\ell  = 1}}^{L}\frac{\exp \left( {{\mathbf{q}}_{i}{\mathbf{k}}_{\ell }^{T}/\sqrt{{D}_{K}}}\right) {\mathbf{v}}_{\ell }}{\mathop{\sum }\limits_{{\ell  = 1}}^{L}\exp \left( {{\mathbf{q}}_{i}{\mathbf{k}}_{\ell }^{T}/\sqrt{{D}_{K}}}\right) }, \tag{1}
$$

where ${\mathbf{k}}_{\ell }^{T}$ denotes the transpose of row $\ell$ in $\mathbf{K}$ . We emphasize that the number of query-key dot products (Q-K pairs) that need to be calculated and stored dictates the time and space complexity of the attention mechanism. Viewed another way, this number is proportional to the number of edges in the graph (see Figure 1(a)). Since all Q-K pairs are computed and stored in the full attention mechanism [1],the resulting time and space complexity is $\mathcal{O}\left( {L}^{2}\right)$ .

其中 ${\mathbf{k}}_{\ell }^{T}$ 表示 $\mathbf{K}$ 中行 $\ell$ 的转置（transpose）。我们强调，需要计算和存储的查询 - 键点积（Q - K 对）的数量决定了注意力机制的时间和空间复杂度。从另一个角度看，这个数量与图中的边的数量成正比（见图 1(a)）。由于在全注意力机制 [1] 中会计算并存储所有的 Q - K 对，因此产生的时间和空间复杂度为 $\mathcal{O}\left( {L}^{2}\right)$。

As opposed to the above full attention mechanism, every node only pays attention to a limited set of keys in the PAM,corresponding to the pyramidal graph in Figure 1d. Concretely,suppose that ${n}_{\rho }^{\left( s\right) }$ denotes the $\ell$ -th node at scale $s$ ,where $s = 1,\cdots ,S$ represents the bottom scale to the top scale sequentially. In general,each node in the graph can attend to a set of neighboring nodes ${\mathbb{N}}_{\ell }^{\left( s\right) }$ at three scales: the adjacent $A$ nodes at the same scale including the node itself (denoted as ${\mathbb{A}}_{\ell }^{\left( s\right) }$ ),the $C$ children it has in the $C$ -ary tree (denoted as ${\mathbb{C}}_{\ell }^{\left( s\right) }$ ),and the parent of it in the $C$ -ary tree (denoted ${\mathbb{P}}_{\ell }^{\left( s\right) }$ ),that is,

与上述全注意力机制不同，在金字塔注意力机制（PAM）中，每个节点仅关注有限的一组键，这对应于图1d中的金字塔图。具体而言，假设 ${n}_{\rho }^{\left( s\right) }$ 表示尺度 $s$ 下的第 $\ell$ 个节点，其中 $s = 1,\cdots ,S$ 依次表示从最底层尺度到最顶层尺度。一般来说，图中的每个节点可以关注三个尺度下的一组相邻节点 ${\mathbb{N}}_{\ell }^{\left( s\right) }$：同一尺度下的相邻 $A$ 个节点（包括该节点本身，记为 ${\mathbb{A}}_{\ell }^{\left( s\right) }$）、该节点在 $C$ 叉树中的 $C$ 个子节点（记为 ${\mathbb{C}}_{\ell }^{\left( s\right) }$）以及该节点在 $C$ 叉树中的父节点（记为 ${\mathbb{P}}_{\ell }^{\left( s\right) }$），即

$$
\left\{  {\begin{array}{l} {\mathbb{N}}_{\ell }^{\left( s\right) } = {\mathbb{A}}_{\ell }^{\left( s\right) } \cup  {\mathbb{C}}_{\ell }^{\left( s\right) } \cup  {\mathbb{P}}_{\ell }^{\left( s\right) } \\  {\mathbb{A}}_{\ell }^{\left( s\right) } = \left\{  {{n}_{j}^{\left( s\right) } : \left| {j - \ell }\right|  \leq  \frac{A - 1}{2},1 \leq  j \leq  \frac{L}{{C}^{s - 1}}}\right\}  \\  {\mathbb{C}}_{\ell }^{\left( s\right) } = \left\{  {{n}_{j}^{\left( s - 1\right) } : \left( {\ell  - 1}\right) C < j \leq  \ell C}\right\}  \text{ if }s \geq  2\text{ else }\varnothing \\  {\mathbb{P}}_{\ell }^{\left( s\right) } = \left\{  {{n}_{j}^{\left( s + 1\right) } : j = \lceil \frac{\ell }{C}\rceil }\right\}  \text{ if }s \leq  S - 1\text{ else }\varnothing  \end{array}.}\right.  \tag{2}
$$

It follows that the attention at node ${n}_{\ell }^{\left( s\right) }$ can be simplified as:

由此可知，节点 ${n}_{\ell }^{\left( s\right) }$ 处的注意力可以简化为：

$$
{\mathbf{y}}_{i} = \mathop{\sum }\limits_{{\ell  \in  {\mathbb{N}}_{\ell }^{\left( s\right) }}}\frac{\exp \left( {{\mathbf{q}}_{i}{\mathbf{k}}_{\ell }^{T}/\sqrt{{d}_{K}}}\right) {\mathbf{v}}_{\ell }}{\mathop{\sum }\limits_{{\ell  \in  {\mathbb{N}}_{l}^{\left( s\right) }}}\exp \left( {{\mathbf{q}}_{i}{\mathbf{k}}_{\ell }^{T}/\sqrt{{d}_{K}}}\right) }, \tag{3}
$$

We further denote the number of attention layers as $N$ . Without loss of generality,we assume that $L$ is divisible by ${C}^{S - 1}$ . We can then have the following lemma (cf. Appendix B for the proof and Table 4 for the meanings of the notations).

我们进一步将注意力层的数量表示为 $N$。不失一般性，我们假设 $L$ 能被 ${C}^{S - 1}$ 整除。然后我们可以得到以下引理（证明见附录 B，符号含义见表 4）。

Lemma 1. Given $A,C,L,N$ ,and $S$ that satisfy Equation (4),after $N$ stacked attention layers, nodes at the coarsest scale can obtain a global receptive field.

引理 1。给定满足方程 (4) 的 $A,C,L,N$ 和 $S$，经过 $N$ 个堆叠的注意力层后，最粗尺度的节点可以获得全局感受野。

$$
\frac{L}{{C}^{S - 1}} - 1 \leq  \frac{\left( {A - 1}\right) N}{2}. \tag{4}
$$

In addition,when the number of scales $S$ is fixed,the following two propositions summarize the time and space complexity and the order of the maximum path length for the proposed pyramidal attention mechanism. We refer the readers to Appendix C and D for proof.

此外，当尺度数量 $S$ 固定时，以下两个命题总结了所提出的金字塔注意力机制的时间和空间复杂度以及最大路径长度的阶。证明见附录 C 和 D。

Proposition 1. The time and space complexity for the pyramidal attention mechanism is $\mathcal{O}\left( {AL}\right)$ for given $A$ and $L$ and amounts to $\mathcal{O}\left( L\right)$ when $A$ is a constant w.r.t. $L$ .

命题1。对于给定的$A$和$L$，金字塔注意力机制的时间和空间复杂度为$\mathcal{O}\left( {AL}\right)$；当$A$相对于$L$为常数时，复杂度为$\mathcal{O}\left( L\right)$。

Proposition 2. Let the signal traversing path between two nodes in a graph denote the shortest path connecting them. Then the maximum length of signal traversing path between two arbitrary nodes in the pyramidal graph is $\mathcal{O}\left( {S + L/{C}^{S - 1}/A}\right)$ for given $A,C,L$ ,and $S$ . Suppose that $A$ and $S$ are fixed and $C$ satisfies Equation (5),the maximum path length is $\mathcal{O}\left( 1\right)$ for time series with length $L$ .

命题2。设图中两个节点之间的信号遍历路径为连接它们的最短路径。那么，对于给定的$A,C,L$和$S$，金字塔图中任意两个节点之间的信号遍历路径的最大长度为$\mathcal{O}\left( {S + L/{C}^{S - 1}/A}\right)$。假设$A$和$S$固定，且$C$满足方程(5)，则对于长度为$L$的时间序列，最大路径长度为$\mathcal{O}\left( 1\right)$。

$$
\sqrt[{S - 1}]{L} \geq  C \geq  \sqrt[{S - 1}]{\frac{L}{\left( {A - 1}\right) N/2 + 1}}. \tag{5}
$$

<!-- Media -->

<!-- figureText: $B \times  L \times  D$ $B \times  L \times  {D}_{K}$ $B \times  \left( {L/C}\right)  \times  {D}_{K}$ Linear $B \times  \left( {L + L/C + L/{C}^{2} + L/{C}^{3}}\right)  \times  D$ Inputs Linear Conv with stride C Conv with stride C $B \times  \left( {L/{C}^{3}}\right)  \times  {D}_{K}$ stride C -->

<img src="https://cdn.noedgeai.com/01957b4b-1654-7ad6-8e71-2da613767aa1_5.jpg?x=568&y=228&w=662&h=311&r=0"/>

Figure 3: Coarser-scale construction module: $B$ is the batch size and $D$ is the dimension of a node.

图3：较粗尺度构建模块：$B$为批量大小，$D$为节点的维度。

<!-- Media -->

In our experiments,we fix $S$ and $N$ ,and $A$ can only take 3 or 5,regardless of the sequence length $L$ . Therefore,the proposed PAM achieves the complexity of $\mathcal{O}\left( L\right)$ with the maximum path length of $\mathcal{O}\left( 1\right)$ . Note that in the PAM,a node can attend to at most $A + C + 1$ nodes. Unfortunately, such a sparse attention mechanism is not supported in the existing deep learning libraries, such as Pytorch and TensorFlow. A naive implementation of the PAM that can fully exploit the tensor operation framework is to first compute the product between all $\mathrm{Q} - \mathrm{K}$ pairs,i.e., ${\mathbf{q}}_{i}{\mathbf{k}}_{\ell }^{T}$ for $\ell  =$ $1,\cdots ,L$ ,and then mask out $\ell  \notin  {\mathbb{N}}_{\ell }^{\left( s\right) }$ . However,the resulting time and space complexity of this implementation is still $\mathcal{O}\left( {L}^{2}\right)$ . Instead,we build a customized CUDA kernel specialized for the PAM using TVM (Chen et al. 2018), practically reducing the computational time and memory cost and making the proposed model amenable to long time series. Longer historical input is typically helpful for improving the prediction accuracy, as more information is provided, especially when long-range dependencies are considered.

在我们的实验中，我们固定$S$和$N$，并且无论序列长度$L$如何，$A$只能取3或5。因此，所提出的PAM（路径注意力机制，Path Attention Mechanism）在最大路径长度为$\mathcal{O}\left( 1\right)$的情况下实现了$\mathcal{O}\left( L\right)$的复杂度。请注意，在PAM中，一个节点最多可以关注$A + C + 1$个节点。不幸的是，现有的深度学习库（如PyTorch和TensorFlow）不支持这种稀疏注意力机制。一种能充分利用张量运算框架的PAM的简单实现方法是，首先计算所有$\mathrm{Q} - \mathrm{K}$对之间的乘积，即对于$\ell  =$ $1,\cdots ,L$计算${\mathbf{q}}_{i}{\mathbf{k}}_{\ell }^{T}$，然后屏蔽掉$\ell  \notin  {\mathbb{N}}_{\ell }^{\left( s\right) }$。然而，这种实现方式的时间和空间复杂度仍然是$\mathcal{O}\left( {L}^{2}\right)$。相反，我们使用TVM（陈等人，2018年）为PAM构建了一个定制的CUDA内核，实际上减少了计算时间和内存成本，并使所提出的模型适用于长时间序列。更长的历史输入通常有助于提高预测准确性，因为它提供了更多信息，特别是在考虑长距离依赖关系时。

### 3.2 COARSER-SCALE CONSTRUCTION MODULE (CSCM)

### 3.2 较粗尺度构建模块（CSCM）

CSCM targets at initializing the nodes at the coarser scales of the pyramidal graph, so as to facilitate the subsequent PAM to exchange information between these nodes. Specifically, the coarse-scale nodes are introduced scale by scale from bottom to top by performing convolutions on the corresponding children nodes ${\mathbb{C}}_{\ell }^{\left( s\right) }$ . As demonstrated in Figure 3,several convolution layers with kernel size $C$ and stride $C$ are sequentially applied to the embedded sequence in the dimension of time, yielding a sequence with length $L/{C}^{s}$ at scale $s$ . The resulting sequences at different scales form a $C$ -ary tree. We concatenate these fine-to-coarse sequences before inputting them to the PAM. In order to reduce the amount of parameters and calculations, we reduce the dimension of each node by a fully connected layer before inputting the sequence into the stacked convolution layers and restore it after all convolutions. Such a bottleneck structure significantly reduces the number of parameters in the module and can guard against over-fitting.

CSCM（粗尺度初始化模块，Coarse Scale Initialization Module）旨在对金字塔图中较粗尺度的节点进行初始化，以便后续的PAM（位置注意力模块，Position Attention Module）在这些节点之间交换信息。具体而言，通过对相应的子节点执行卷积操作[公式0]，从下到上逐尺度引入粗尺度节点。如图3所示，在时间维度上对嵌入序列依次应用几个核大小为[公式1]、步长为[公式1]的卷积层，在尺度[公式3]上得到长度为[公式2]的序列。不同尺度上得到的序列构成一棵[公式1]叉树。在将这些序列输入到PAM之前，我们将这些从细到粗的序列进行拼接。为了减少参数数量和计算量，在将序列输入到堆叠卷积层之前，我们通过一个全连接层降低每个节点的维度，并在所有卷积操作完成后恢复其维度。这种瓶颈结构显著减少了模块中的参数数量，并且可以防止过拟合。

### 3.3 Prediction Module

### 3.3 预测模块

For single-step forecasting,we add an end token (by setting ${z}_{t + 1} = 0$ ) to the end of the historical sequence ${z}_{t - L + 1 : t}$ before inputting it into the embedding layer. After the sequence is encoded by the PAM, we gather the features given by the last nodes at all scales in the pyramidal graph, concatenate and then input them into a fully connected layer for prediction.

对于单步预测，我们在将历史序列${z}_{t - L + 1 : t}$输入嵌入层之前，在其末尾添加一个结束标记（通过设置${z}_{t + 1} = 0$）。在该序列由金字塔注意力模块（PAM）编码后，我们收集金字塔图中所有尺度上最后节点给出的特征，将它们拼接起来，然后输入到一个全连接层进行预测。

For multi-step forecasting, we propose two prediction modules. The first one is the same with the single-step forecasting module,but maps the last nodes at all scales to all $M$ future time steps in a batch. The second one, on the other hand, resorts to a decoder with two full attention layers. Specifically, similar to the original Transformer (Vaswani et al. 2017), we replace the observations at the future $M$ time steps with 0,embed them in the same manner with the historical observations, and refer to the summation of the observation, covariate, and positional embedding as the "prediction token" ${\mathbf{F}}_{p}$ . The first attention layer then takes the prediction tokens ${\mathbf{F}}_{p}$ as the query and the output of the encoder ${\mathbf{F}}_{e}$ (i.e.,all nodes in the PAM) as the key and the value,and yields ${\mathbf{F}}_{d1}$ . The second layer takes ${\mathbf{F}}_{d1}$ as the query,but takes the concatenated ${\mathbf{F}}_{d1}$ and ${\mathbf{F}}_{e}$ as the key and the value. The historical information ${\mathbf{F}}_{e}$ is fed directly into both attention layers,since such information is vital for accurate long-range forecasting. The final prediction is then obtained through a fully connected layer across the dimension of channels. Again, we output all future predictions together to avoid the problem of error accumulation in the autoregressive decoder of Transformer.

对于多步预测，我们提出了两个预测模块。第一个模块与单步预测模块相同，但会将所有尺度上的最后节点批量映射到所有 $M$ 未来时间步。另一方面，第二个模块采用了一个具有两个全注意力层的解码器。具体而言，与原始的Transformer（Vaswani等人，2017年）类似，我们将未来 $M$ 时间步的观测值替换为0，以与历史观测值相同的方式对其进行嵌入，并将观测值、协变量和位置嵌入的总和称为“预测令牌” ${\mathbf{F}}_{p}$。然后，第一个注意力层将预测令牌 ${\mathbf{F}}_{p}$ 作为查询，将编码器 ${\mathbf{F}}_{e}$ 的输出（即PAM中的所有节点）作为键和值，并生成 ${\mathbf{F}}_{d1}$。第二层将 ${\mathbf{F}}_{d1}$ 作为查询，但将拼接后的 ${\mathbf{F}}_{d1}$ 和 ${\mathbf{F}}_{e}$ 作为键和值。历史信息 ${\mathbf{F}}_{e}$ 会直接输入到两个注意力层中，因为此类信息对于准确的长程预测至关重要。最终的预测结果通过一个跨通道维度的全连接层获得。同样，我们将所有未来预测结果一起输出，以避免Transformer自回归解码器中的误差累积问题。

<!-- Media -->

Table 2: Single-step forecasting results on three datasets. "Q-K pairs" refer to the number of query-key dot products performed by all attention layers in the network, which encodes the time and space complexity. We write the number of attention layers by $N$ ,the number of attention heads by $H$ ,the number of scales by $S$ ,the dimension of a node by $D$ ,the dimension of a key by ${D}_{K}$ ,the maximum dimension of feed-forward layer by ${D}_{F}$ ,and the convolution stride by $C$ .

表2：三个数据集上的单步预测结果。“查询-键对（Q-K pairs）”指网络中所有注意力层执行的查询-键点积的数量，它编码了时间和空间复杂度。我们用$N$表示注意力层的数量，用$H$表示注意力头的数量，用$S$表示尺度的数量，用$D$表示节点的维度，用${D}_{K}$表示键的维度，用${D}_{F}$表示前馈层的最大维度，用$C$表示卷积步长。

<table><tr><td>Methods</td><td>Parameters</td><td>Datasets</td><td>NRMSE</td><td>$\mathbf{{ND}}$</td><td>Q-K pairs</td></tr><tr><td rowspan="3">Full-attention</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>Electricity</td><td>0.328</td><td>0.041</td><td>456976</td></tr><tr><td>Wind</td><td>0.175</td><td>0.082</td><td>589824</td></tr><tr><td>App Flow</td><td>0.407</td><td>0.080</td><td>589824</td></tr><tr><td rowspan="3">LogTrans</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>Electricity</td><td>0.333</td><td>0.041</td><td>50138</td></tr><tr><td>Wind</td><td>0.173</td><td>0.081</td><td>58272</td></tr><tr><td>App Flow</td><td>0.387</td><td>0.073</td><td>58272</td></tr><tr><td rowspan="3">Reformer</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>Electricity</td><td>0.359</td><td>0.047</td><td>677376</td></tr><tr><td>Wind</td><td>0.183</td><td>0.086</td><td>884736</td></tr><tr><td>App Flow</td><td>0.463</td><td>0.095</td><td>884736</td></tr><tr><td rowspan="3">ETC</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>Electricity</td><td>0.324</td><td>0.041</td><td>79536</td></tr><tr><td>Wind</td><td>0.167</td><td>0.074</td><td>102144</td></tr><tr><td>App Flow</td><td>0.397</td><td>0.069</td><td>102144</td></tr><tr><td rowspan="3">Longformer</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>Electricity</td><td>0.330</td><td>0.041</td><td>41360</td></tr><tr><td>Wind</td><td>0.166</td><td>0.075</td><td>52608</td></tr><tr><td>App Flow</td><td>0.377</td><td>0.07</td><td>52608</td></tr><tr><td rowspan="3">Pyraformer</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right.$ $\left. {+\left( {S - 1}\right) C{D}_{K}^{2}}\right)$</td><td>Electricity</td><td>0.324</td><td>0.041</td><td>17648</td></tr><tr><td>Wind</td><td>0.161</td><td>0.072</td><td>20176</td></tr><tr><td>App Flow</td><td>0.366</td><td>0.067</td><td>20176</td></tr></table>

<table><tbody><tr><td>方法</td><td>参数</td><td>数据集</td><td>归一化均方根误差（NRMSE）</td><td>$\mathbf{{ND}}$</td><td>查询-键对（Q-K pairs）</td></tr><tr><td rowspan="3">全注意力机制（Full-attention）</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>电力</td><td>0.328</td><td>0.041</td><td>456976</td></tr><tr><td>风力</td><td>0.175</td><td>0.082</td><td>589824</td></tr><tr><td>应用程序流程</td><td>0.407</td><td>0.080</td><td>589824</td></tr><tr><td rowspan="3">日志传输（LogTrans）</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>电力</td><td>0.333</td><td>0.041</td><td>50138</td></tr><tr><td>风力</td><td>0.173</td><td>0.081</td><td>58272</td></tr><tr><td>应用程序流程</td><td>0.387</td><td>0.073</td><td>58272</td></tr><tr><td rowspan="3">重整器</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>电力</td><td>0.359</td><td>0.047</td><td>677376</td></tr><tr><td>风力</td><td>0.183</td><td>0.086</td><td>884736</td></tr><tr><td>应用程序流程</td><td>0.463</td><td>0.095</td><td>884736</td></tr><tr><td rowspan="3">电子不停车收费系统（ETC）</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>电力</td><td>0.324</td><td>0.041</td><td>79536</td></tr><tr><td>风力</td><td>0.167</td><td>0.074</td><td>102144</td></tr><tr><td>应用程序流程</td><td>0.397</td><td>0.069</td><td>102144</td></tr><tr><td rowspan="3">长序列变换器（Longformer）</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right)$</td><td>电力</td><td>0.330</td><td>0.041</td><td>41360</td></tr><tr><td>风力</td><td>0.166</td><td>0.075</td><td>52608</td></tr><tr><td>应用程序流程</td><td>0.377</td><td>0.07</td><td>52608</td></tr><tr><td rowspan="3">金字塔变换器（Pyraformer）</td><td rowspan="3">$\mathcal{O}\left( {N\left( {{HD}{D}_{K} + D{D}_{F}}\right) }\right.$ $\left. {+\left( {S - 1}\right) C{D}_{K}^{2}}\right)$</td><td>电力</td><td>0.324</td><td>0.041</td><td>17648</td></tr><tr><td>风力</td><td>0.161</td><td>0.072</td><td>20176</td></tr><tr><td>应用程序流程</td><td>0.366</td><td>0.067</td><td>20176</td></tr></tbody></table>

<!-- Media -->

## 4 EXPERIMENTS

## 4 实验

### 4.1 DATASETS AND EXPERIMENT SETUP

### 4.1 数据集与实验设置

We demonstrated the advantages of the proposed Pyraformer on the four real-world datasets, including Wind, App Flow, Electricity, and ETT. The first three datasets were used for single-step forecasting, while the last two for long-range multi-step forecasting. We refer the readers to Appendix $E$ and $F$ for more details regarding the data description and the experiment setup.

我们在四个真实世界的数据集上展示了所提出的金字塔变换器（Pyraformer）的优势，这些数据集包括风力（Wind）、应用流量（App Flow）、电力（Electricity）和电力变压器温度（ETT）。前三个数据集用于单步预测，而后两个用于长程多步预测。有关数据描述和实验设置的更多详细信息，请读者参考附录 $E$ 和 $F$。

### 4.2 RESULTS AND ANALYSIS

### 4.2 结果与分析

#### 4.2.1 SINGLE-STEP FORECASTING

#### 4.2.1 单步预测

We conducted single-step prediction experiments on three datasets: Electricity, Wind and App Flow. The historical length is 169, 192 and 192, respectively, including the end token. We benchmarked Pyraformer against 5 other attention mechanisms, including the original full-attention (Vaswani et al. 2017), the log-sparse attention (i.e., LogTrans) (Li et al. 2019), the LSH attention (i.e., Reformer) (Kitaev et al. 2019), the sliding window attention with global nodes (i.e., ETC) (Ainslie et al. 2020), and the dilated sliding window attention (i.e., Longformer) (Beltagy et al. 2020). In particular for ETC, some nodes with equal intervals at the finest scale were selected as the global nodes. A global node can attend to all nodes across the sequence and all nodes can attend to it in turn(see Figure 1(e)). The training and testing schemes were the same for all models. We further investigated the usefulness of the pretraining strategy (see Appendix G), the weighted sampler, and the hard sample mining on all methods, and the best results were presented. We adopted the NRMSE (Normalized RMSE) and the ND (Normalized Deviation) as the evaluation indicators (see Appendix A for the definitions). The results are summarized in Table 2. For a fair comparison, except for full-attention, the overall dot product number of all attention mechanisms was controlled to the same order of magnitude.

我们在三个数据集上进行了单步预测实验：电力（Electricity）、风力（Wind）和应用流量（App Flow）。历史长度分别为169、192和192，包括结束标记。我们将Pyraformer与其他5种注意力机制进行了基准测试，包括原始的全注意力机制（Vaswani等人，2017年）、对数稀疏注意力机制（即LogTrans）（Li等人，2019年）、局部敏感哈希注意力机制（即Reformer）（Kitaev等人，2019年）、带全局节点的滑动窗口注意力机制（即ETC）（Ainslie等人，2020年），以及扩张滑动窗口注意力机制（即Longformer）（Beltagy等人，2020年）。特别对于ETC，在最精细的尺度上选择了一些等间隔的节点作为全局节点。一个全局节点可以关注序列中的所有节点，反过来所有节点也可以关注它（见图1(e)）。所有模型的训练和测试方案相同。我们进一步研究了预训练策略（见附录G）、加权采样器和难样本挖掘在所有方法中的有效性，并展示了最佳结果。我们采用归一化均方根误差（NRMSE，Normalized RMSE）和归一化偏差（ND，Normalized Deviation）作为评估指标（定义见附录A）。结果总结在表2中。为了进行公平比较，除了全注意力机制外，所有注意力机制的总体点积数量都控制在相同的数量级。

Our experimental results show that Pyraformer outperforms Transformer and its variants in terms of NRMSE and ND, with the least number of query-key dot products (a.k.a. Q-K pairs). Concretely, there are three major trends that can be gleaned from Table 2 (1) The proposed Pyraformer yields the most accurate prediction results, suggesting that the pyramidal graph can better explain the temporal interactions in the time series by considering dependencies of different ranges. Interestingly, for the Wind dataset, sparse attention mechanisms, namely, LogTrans, ETC, Longformer and Pyraformer, outperform the original full attention Transformer, probably because the data contains a large number of zeros and the promotion of adequate sparsity can help avoid over-fitting. (2) The number of Q-K pairs in Pyraformer is the smallest. Recall that this number characterizes the time and space complexity. Remarkably enough,it is ${65.4}\%$ fewer than that of LogTrans and 96.6% than that of the full attention. It is worth emphasizing that this computational gain will continue to increase for longer time series. (3) The number of parameters for Pyraformer is slightly larger than that of the other models, resulting from the CSCM. However, this module is very lightweight,which incurs merely 5% overhead in terms of model size compared to other models. Moreover,in practice,we can fix the hyper-parameters $A,S$ and $N$ ,and ensure that $C$ satisfies $C > \sqrt[{S - 1}]{L/\left( {\left( {A - 1}\right) N/2 + 1}\right) }$ . Consequently,the extra number of parameters introduced by the CSCM is only $\mathcal{O}\left( {\left( {S - 1}\right) C{D}_{K}^{2}}\right)  \approx  \mathcal{O}\left( \sqrt[{S - 1}]{L}\right)$ .

我们的实验结果表明，在归一化均方根误差（NRMSE）和归一化偏差（ND）方面，金字塔变换器（Pyraformer）优于变换器（Transformer）及其变体，且查询 - 键点积（即 Q - K 对）的数量最少。具体而言，从表 2 中可以总结出三个主要趋势：（1）所提出的金字塔变换器（Pyraformer）产生了最准确的预测结果，这表明金字塔图通过考虑不同范围的依赖关系，能够更好地解释时间序列中的时间交互。有趣的是，对于风电数据集（Wind dataset），稀疏注意力机制，即对数变换器（LogTrans）、增强型变换器（ETC）、长序列变换器（Longformer）和金字塔变换器（Pyraformer），优于原始的全注意力变换器（Transformer），这可能是因为数据中包含大量零值，而适当的稀疏性提升有助于避免过拟合。（2）金字塔变换器（Pyraformer）中的 Q - K 对数量最少。请记住，这个数量表征了时间和空间复杂度。值得注意的是，它比对数变换器（LogTrans）少${65.4}\%$，比全注意力机制少 96.6%。值得强调的是，对于更长的时间序列，这种计算优势将继续增加。（3）由于上下文感知卷积模块（CSCM），金字塔变换器（Pyraformer）的参数数量略多于其他模型。然而，该模块非常轻量级，与其他模型相比，在模型大小方面仅产生 5% 的开销。此外，在实践中，我们可以固定超参数$A,S$和$N$，并确保$C$满足$C > \sqrt[{S - 1}]{L/\left( {\left( {A - 1}\right) N/2 + 1}\right) }$。因此，上下文感知卷积模块（CSCM）引入的额外参数数量仅为$\mathcal{O}\left( {\left( {S - 1}\right) C{D}_{K}^{2}}\right)  \approx  \mathcal{O}\left( \sqrt[{S - 1}]{L}\right)$。

<!-- Media -->

Table 3: Long-range multi-step forecasting results.

表3：远程多步预测结果。

<table><tr><td rowspan="2">Methods</td><td rowspan="2">$\mathbf{{Metrics}}$</td><td colspan="3">ETTh1</td><td colspan="3">ETTm1</td><td colspan="3">Electricity</td></tr><tr><td>168</td><td>336</td><td>720</td><td>96</td><td>288</td><td>672</td><td>168</td><td>336</td><td>720</td></tr><tr><td rowspan="3">Informer</td><td>MSE</td><td>1.075</td><td>1.329</td><td>1.384</td><td>0.556</td><td>0.841</td><td>0.921</td><td>0.745</td><td>1.579</td><td>4.365</td></tr><tr><td>MAE</td><td>0.801</td><td>0.911</td><td>0.950</td><td>0.537</td><td>0.705</td><td>0.753</td><td>0.266</td><td>0.323</td><td>0.371</td></tr><tr><td>Q-K pairs</td><td>188040</td><td>188040</td><td>423360</td><td>276480</td><td>560640</td><td>560640</td><td>188040</td><td>188040</td><td>423360</td></tr><tr><td rowspan="3">LogTrans</td><td>MSE</td><td>0.983</td><td>1.100</td><td>1.411</td><td>0.554</td><td>0.786</td><td>1.169</td><td>0.791</td><td>1.584</td><td>4.362</td></tr><tr><td>MAE</td><td>0.766</td><td>0.839</td><td>0.991</td><td>0.499</td><td>0.676</td><td>0.868</td><td>0.340</td><td>0.336</td><td>0.366</td></tr><tr><td>Q-K pairs</td><td>74664</td><td>74664</td><td>216744</td><td>254760</td><td>648768</td><td>648768</td><td>74664</td><td>74664</td><td>216744</td></tr><tr><td rowspan="3">Longformer</td><td>MSE</td><td>0.860</td><td>0.975</td><td>1.091</td><td>0.526</td><td>0.767</td><td>1.021</td><td>0.766</td><td>1.591</td><td>4.361</td></tr><tr><td>MAE</td><td>0.710</td><td>0.769</td><td>0.832</td><td>0.507</td><td>0.663</td><td>0.788</td><td>0.311</td><td>0.343</td><td>0.368</td></tr><tr><td>Q-K pairs</td><td>63648</td><td>63648</td><td>249120</td><td>329760</td><td>1007136</td><td>1007136</td><td>63648</td><td>63648</td><td>249120</td></tr><tr><td rowspan="3">Reformer</td><td>MSE</td><td>0.958</td><td>1.044</td><td>1.458</td><td>0.543</td><td>0.924</td><td>0.981</td><td>0.783</td><td>1.584</td><td>4.374</td></tr><tr><td>MAE</td><td>0.741</td><td>0.787</td><td>0.987</td><td>0.528</td><td>0.722</td><td>0.778</td><td>0.332</td><td>0.334</td><td>0.374</td></tr><tr><td>Q-K pairs</td><td>1016064</td><td>1016064</td><td>2709504</td><td>5308416</td><td>14450688</td><td>14450688</td><td>1016064</td><td>1016064</td><td>2709504</td></tr><tr><td rowspan="3">ETC</td><td>MSE</td><td>1.025</td><td>1.084</td><td>1.137</td><td>0.762</td><td>1.227</td><td>1.272</td><td>0.777</td><td>1.586</td><td>4.361</td></tr><tr><td>MAE</td><td>0.771</td><td>0.811</td><td>0.866</td><td>0.653</td><td>0.880</td><td>0.908</td><td>0.326</td><td>0.340</td><td>0.368</td></tr><tr><td>Q-K pairs</td><td>125280</td><td>125280</td><td>288720</td><td>331344</td><td>836952</td><td>836952</td><td>125280</td><td>125280</td><td>288720</td></tr><tr><td rowspan="3">Pyraformer</td><td>MSE</td><td>0.808</td><td>0.945</td><td>1.022</td><td>0.480</td><td>0.754</td><td>0.857</td><td>0.719</td><td>1.533</td><td>4.312</td></tr><tr><td>MAE</td><td>0.683</td><td>0.766</td><td>0.806</td><td>0.486</td><td>0.659</td><td>0.707</td><td>0.256</td><td>0.291</td><td>0.346</td></tr><tr><td>Q-K pairs</td><td>26472</td><td>26472</td><td>74280</td><td>57264</td><td>96384</td><td>96384</td><td>26472</td><td>26472</td><td>74280</td></tr></table>

<table><tbody><tr><td rowspan="2">方法</td><td rowspan="2">$\mathbf{{Metrics}}$</td><td colspan="3">ETTh1</td><td colspan="3">ETTm1</td><td colspan="3">电力</td></tr><tr><td>168</td><td>336</td><td>720</td><td>96</td><td>288</td><td>672</td><td>168</td><td>336</td><td>720</td></tr><tr><td rowspan="3">Informer</td><td>均方误差（MSE）</td><td>1.075</td><td>1.329</td><td>1.384</td><td>0.556</td><td>0.841</td><td>0.921</td><td>0.745</td><td>1.579</td><td>4.365</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.801</td><td>0.911</td><td>0.950</td><td>0.537</td><td>0.705</td><td>0.753</td><td>0.266</td><td>0.323</td><td>0.371</td></tr><tr><td>查询-键对（Q-K pairs）</td><td>188040</td><td>188040</td><td>423360</td><td>276480</td><td>560640</td><td>560640</td><td>188040</td><td>188040</td><td>423360</td></tr><tr><td rowspan="3">对数变换（LogTrans）</td><td>均方误差（MSE）</td><td>0.983</td><td>1.100</td><td>1.411</td><td>0.554</td><td>0.786</td><td>1.169</td><td>0.791</td><td>1.584</td><td>4.362</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.766</td><td>0.839</td><td>0.991</td><td>0.499</td><td>0.676</td><td>0.868</td><td>0.340</td><td>0.336</td><td>0.366</td></tr><tr><td>查询-键对（Q-K pairs）</td><td>74664</td><td>74664</td><td>216744</td><td>254760</td><td>648768</td><td>648768</td><td>74664</td><td>74664</td><td>216744</td></tr><tr><td rowspan="3">长序列变换器（Longformer）</td><td>均方误差（MSE）</td><td>0.860</td><td>0.975</td><td>1.091</td><td>0.526</td><td>0.767</td><td>1.021</td><td>0.766</td><td>1.591</td><td>4.361</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.710</td><td>0.769</td><td>0.832</td><td>0.507</td><td>0.663</td><td>0.788</td><td>0.311</td><td>0.343</td><td>0.368</td></tr><tr><td>查询-键对（Q-K pairs）</td><td>63648</td><td>63648</td><td>249120</td><td>329760</td><td>1007136</td><td>1007136</td><td>63648</td><td>63648</td><td>249120</td></tr><tr><td rowspan="3">改革者模型（Reformer）</td><td>均方误差（MSE）</td><td>0.958</td><td>1.044</td><td>1.458</td><td>0.543</td><td>0.924</td><td>0.981</td><td>0.783</td><td>1.584</td><td>4.374</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.741</td><td>0.787</td><td>0.987</td><td>0.528</td><td>0.722</td><td>0.778</td><td>0.332</td><td>0.334</td><td>0.374</td></tr><tr><td>查询-键对（Q-K pairs）</td><td>1016064</td><td>1016064</td><td>2709504</td><td>5308416</td><td>14450688</td><td>14450688</td><td>1016064</td><td>1016064</td><td>2709504</td></tr><tr><td rowspan="3">增强型变换器压缩器（ETC）</td><td>均方误差（MSE）</td><td>1.025</td><td>1.084</td><td>1.137</td><td>0.762</td><td>1.227</td><td>1.272</td><td>0.777</td><td>1.586</td><td>4.361</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.771</td><td>0.811</td><td>0.866</td><td>0.653</td><td>0.880</td><td>0.908</td><td>0.326</td><td>0.340</td><td>0.368</td></tr><tr><td>查询-键对（Q-K pairs）</td><td>125280</td><td>125280</td><td>288720</td><td>331344</td><td>836952</td><td>836952</td><td>125280</td><td>125280</td><td>288720</td></tr><tr><td rowspan="3">派拉变换器（Pyraformer）</td><td>均方误差（MSE）</td><td>0.808</td><td>0.945</td><td>1.022</td><td>0.480</td><td>0.754</td><td>0.857</td><td>0.719</td><td>1.533</td><td>4.312</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.683</td><td>0.766</td><td>0.806</td><td>0.486</td><td>0.659</td><td>0.707</td><td>0.256</td><td>0.291</td><td>0.346</td></tr><tr><td>查询-键对（Q-K pairs）</td><td>26472</td><td>26472</td><td>74280</td><td>57264</td><td>96384</td><td>96384</td><td>26472</td><td>26472</td><td>74280</td></tr></tbody></table>

<!-- Media -->

#### 4.2.2 LONG-RANGE MULTI-STEP FORECASTING

#### 4.2.2 长距离多步预测

We evaluated the performance of Pyraformer for long-range forecasting on three datasets, that is, Electricity, ETTh1, and ETTm1. In particular for ETTh1 and ETTm1, we predicted the future oil temperature and the 6 power load features at the same time, which is a multivariate time series forecasting problem. Both prediction modules introduced in Section 3.3 were tested for all models and the better results are listed in Table 3.

我们在三个数据集（即电力数据集（Electricity）、ETTh1 数据集和 ETTm1 数据集）上评估了 Pyraformer 在长距离预测方面的性能。特别是对于 ETTh1 和 ETTm1 数据集，我们同时预测未来油温以及 6 个电力负荷特征，这是一个多元时间序列预测问题。对所有模型都测试了第 3.3 节中介绍的两个预测模块，并将较好的结果列于表 3 中。

It is evident that Pyraformer still achieves the best performance with the least number of Q-K pairs for all datasets regardless of the prediction length. More precisely, in comparison with Informer (Zhou et al. 2021), the MSE given by Pyraformer for ETTh1 is decreased by 24.8%, 28.9%, ${26.2}\%$ respectively when the prediction length is 168,336,and 720 . Once again,this bolsters our belief that it is more beneficial to employ the pyramidal graph when describing the temporal dependencies. Interestingly, we notice that for Pyraformer, the results given by the first prediction module are better than those by the second one. One possible explanation is that the second prediction module based on the full attention layers cannot differentiate features with different resolutions, while the first module based on a single fully connected layer can take full advantages of such features in an automated fashion. To better elucidate the modeling capacity of Pyraformer for long-range forecasting, we refer the readers to Appendix 1 for a detailed example on synthetic data.

显然，无论预测长度如何，Pyraformer（金字塔变换器）在所有数据集上仍能以最少数量的查询 - 键（Q - K）对实现最佳性能。更确切地说，与Informer（周等人，2021年）相比，当预测长度为168、336和720时，Pyraformer在ETTh1数据集上给出的均方误差（MSE）分别降低了24.8%、28.9%、${26.2}\%$。这再次证实了我们的观点，即在描述时间依赖关系时采用金字塔图更为有利。有趣的是，我们注意到对于Pyraformer，第一个预测模块给出的结果优于第二个预测模块。一种可能的解释是，基于全注意力层的第二个预测模块无法区分不同分辨率的特征，而基于单个全连接层的第一个模块可以自动充分利用这些特征。为了更好地阐明Pyraformer在长期预测方面的建模能力，我们请读者参考附录1中关于合成数据的详细示例。

<!-- Media -->

<!-- figureText: Time Memory $\rightarrow$ prob-sparse attention - Pyraformer-TVM $\begin{array}{lllllll} 0 & {2000} & {4000} & {6000} & {8000} & {1000012000} & {1400016000} \end{array}$ sequence length (b) 175 - full attention - prob-sparse attention — Pyraformer-TVM 2000 4000 6000 8000 10000120001400016000 sequence length (a) -->

<img src="https://cdn.noedgeai.com/01957b4b-1654-7ad6-8e71-2da613767aa1_8.jpg?x=340&y=238&w=1116&h=474&r=0"/>

Figure 4: Comparison of the time and memory consumption between the full, the prob-sparse, and the TVM implementation of the pyramidal attention: (a) computation time; (b) memory occupation.

图4：金字塔注意力的完整实现、概率稀疏实现和TVM实现之间的时间和内存消耗对比：(a) 计算时间；(b) 内存占用。

<!-- Media -->

#### 4.2.3 SPEED AND MEMORY CONSUMPTION

#### 4.2.3 速度和内存消耗

To check the efficiency of the customized CUDA kernel implemented based on TVM, we depicted the empirical computation time and memory cost as a function of the sequence length $L$ in Figure 4. Here we only compared Pyraformer with the full attention and the prob-sparse attention in Informer (Zhou et al. 2021). All the computations were performed on a 12 GB Titan Xp GPU with Ubuntu 16.04, CUDA 11.0, and TVM 0.8.0. Figure 4 shows that the time and memory cost of the proposed Pyraformer based on TVM is approximately a linear function of $L$ ,as expected. Furthermore, the time and memory consumption of the TVM implementation can be several orders of magnitude smaller than that of the full attention and the prob-sparse attention, especially for relatively long time series. Indeed, for a 12GB Titan Xp GPU, when the sequence length reaches 5800, full attention encounters the out-of-memory (OOM) problem, yet the TVM implementation of Pyraformer only occupies 1GB of memory. When it comes to a sequence with 20000 time points, even Informer incurs the OOM problem, whereas the memory cost of Pyraformer is only 1.91GB and the computation time per batch is only 0.082 s .

为了检验基于TVM实现的定制CUDA内核的效率，我们在图4中描绘了经验计算时间和内存成本随序列长度$L$的变化情况。在这里，我们仅将Pyraformer与Informer（Zhou等人，2021年）中的全注意力机制和概率稀疏注意力机制进行了比较。所有计算均在配备Ubuntu 16.04操作系统、CUDA 11.0和TVM 0.8.0的12GB Titan Xp GPU上进行。图4显示，基于TVM的Pyraformer的时间和内存成本如预期的那样，大致是$L$的线性函数。此外，TVM实现的时间和内存消耗比全注意力机制和概率稀疏注意力机制小几个数量级，尤其是对于相对较长的时间序列而言。实际上，对于12GB的Titan Xp GPU，当序列长度达到5800时，全注意力机制会出现内存不足（OOM）问题，而Pyraformer的TVM实现仅占用1GB内存。当涉及到具有20000个时间点的序列时，即使是Informer也会出现OOM问题，而Pyraformer的内存成本仅为1.91GB，每批次的计算时间仅为0.082秒。

### 4.3 ABLATION STUDY

### 4.3 消融研究

We also performed ablation studies to measure the impact of $A$ and $C$ ,the CSCM architecture,the history length, and the PAM on the prediction accuracy of Pyraformer. The results are displayed in Tables 7 10 Detailed Discussions on the results can be found in Appendix J. Here, we only provide an overview of the major findings: (1) it is better to increase $C$ with $L$ but fix $A$ to a small constant for the sake of reducing the prediction error; (2) convolution with bottleneck strikes a balance between the prediction accuracy and the number of parameters, and hence, we use it as the CSCM; (3) more history helps increase the accuracy of forecasting; (4) the PAM is essential for accurate prediction.

我们还进行了消融研究，以衡量$A$和$C$、CSCM架构、历史长度以及PAM对Pyraformer预测准确性的影响。结果显示在表7 - 10中。关于结果的详细讨论可在附录J中找到。在此，我们仅对主要发现进行概述：（1）为了减少预测误差，最好随着$L$的增加而增加$C$，但将$A$固定为一个小常数；（2）带瓶颈的卷积在预测准确性和参数数量之间取得了平衡，因此，我们将其用作CSCM；（3）更多的历史数据有助于提高预测的准确性；（4）PAM对于准确预测至关重要。

## 5 CONCLUSION AND OUTLOOK

## 5 结论与展望

In this paper, we propose Pyraformer, a novel model based on pyramidal attention that can effectively describe both short and long temporal dependencies with low time and space complexity. Concretely,we first exploit the CSCM to construct a $C$ -ary tree,and then design the PAM to pass messages in both the inter-scale and the intra-scale fashion. By adjusting $C$ and fixing other parameters when the sequence length $L$ increases,Pyraformer can achieve the theoretical $\mathcal{O}\left( L\right)$ complexity and $\mathcal{O}\left( 1\right)$ maximum signal traversing path length. Experimental results show that the proposed model outperforms the state-of-the-art models for both single-step and long-range multi-step prediction tasks, but with less computational time and memory cost. So far we only concentrate on the scenario where $A$ and $S$ are fixed and $C$ increases with $L$ when constructing the pyramidal graph. On the other hand, we have shown in Appendix 1 that other configurations of the hyper-parameters may further improve the performance of Pyraformer. In the future work, we would like to explore how to adaptively learn the hyper-parameters from the data. Also, it is interesting to extend Pyraformer to other fields, including natural language processing and computer vision.

在本文中，我们提出了Pyraformer（金字塔变换器），这是一种基于金字塔注意力机制的新型模型，能够以较低的时间和空间复杂度有效描述短期和长期时间依赖关系。具体而言，我们首先利用CSCM（跨尺度卷积模块）构建一个$C$叉树，然后设计PAM（金字塔注意力模块）以跨尺度和尺度内的方式传递信息。当序列长度$L$增加时，通过调整$C$并固定其他参数，Pyraformer可以达到理论上的$\mathcal{O}\left( L\right)$复杂度和$\mathcal{O}\left( 1\right)$最大信号遍历路径长度。实验结果表明，所提出的模型在单步和长程多步预测任务中均优于现有最先进的模型，且计算时间和内存成本更低。到目前为止，我们仅关注在构建金字塔图时$A$和$S$固定且$C$随$L$增加的场景。另一方面，我们在附录1中表明，超参数的其他配置可能会进一步提高Pyraformer的性能。在未来的工作中，我们希望探索如何从数据中自适应地学习超参数。此外，将Pyraformer扩展到其他领域，包括自然语言处理和计算机视觉，也是很有趣的。

## ACKNOWLEDGEMENT

## 致谢

In this work, Prof. Weiyao Lin was supported by Ant Group through Ant Research Program and in part by National Natural Science Foundation of China under grant U21B2013.

在这项工作中，林巍峣（Weiyao Lin）教授得到了蚂蚁集团“蚂蚁科研计划”的支持，部分得到了国家自然科学基金（项目编号U21B2013）的资助。

## REFERENCES

## 参考文献

Joshua Ainslie, Santiago Ontanon, Chris Alberti, Vaclav Cvicek, Zachary Fisher, Philip Pham, Anirudh Ravula, Sumit Sanghai, Qifan Wang, and Li Yang. Etc: Encoding long and structured inputs in transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 268-284, 2020.

约书亚·安斯利（Joshua Ainslie）、圣地亚哥·翁塔农（Santiago Ontanon）、克里斯·阿尔伯蒂（Chris Alberti）、瓦茨拉夫·奇维塞克（Vaclav Cvicek）、扎卡里·费舍尔（Zachary Fisher）、菲利普·范（Philip Pham）、阿尼鲁德·拉武拉（Anirudh Ravula）、苏米特·桑海（Sumit Sanghai）、王启凡（Qifan Wang）和杨立（Li Yang）。Etc：在Transformer中编码长且结构化的输入。《2020年自然语言处理经验方法会议（EMNLP）论文集》，第268 - 284页，2020年。

Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

伊兹·贝尔塔吉（Iz Beltagy）、马修·E·彼得斯（Matthew E Peters）和阿尔曼·科汉（Arman Cohan）。Longformer：长文档Transformer。预印本arXiv:2004.05150，2020年。

George EP Box and Gwilym M Jenkins. Some recent advances in forecasting and control. Journal of the Royal Statistical Society. Series C (Applied Statistics), 17(2):91-109, 1968.

乔治·E·P·博克斯（George EP Box）和格威利姆·M·詹金斯（Gwilym M Jenkins）。预测与控制的一些最新进展。《皇家统计学会杂志》C辑（应用统计学），17(2)：91 - 109，1968年。

Shiyu Chang, Yang Zhang, Wei Han, Mo Yu, Xiaoxiao Guo, Wei Tan, Xiaodong Cui, Michael Witbrock, Mark Hasegawa-Johnson, and Thomas S Huang. Dilated recurrent neural networks. Advances in Neural Information Processing Systems, 2017:77-87, 2017.

常世玉（Shiyu Chang）、张洋（Yang Zhang）、韩伟（Wei Han）、于默（Mo Yu）、郭笑笑（Xiaoxiao Guo）、谭伟（Wei Tan）、崔晓东（Xiaodong Cui）、迈克尔·维特布罗克（Michael Witbrock）、马克·长谷川 - 约翰逊（Mark Hasegawa - Johnson）和黄煦涛（Thomas S Huang）。扩张循环神经网络。《神经信息处理系统进展》，2017 年：77 - 87，2017 年。

Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, et al. \{TVM\}: An automated end-to-end optimizing compiler for deep learning. In 13th \{USENIX\} Symposium on Operating Systems Design and Implementation ( $\{$ OSDI $\}$ 18),pp. 578-594,2018.

陈天奇（Tianqi Chen）、蒂埃里·莫罗（Thierry Moreau）、蒋子恒（Ziheng Jiang）、郑连民（Lianmin Zheng）、严 Eddie（Eddie Yan）、沈海晨（Haichen Shen）、梅根·考恩（Meghan Cowan）、王乐源（Leyuan Wang）、胡雨薇（Yuwei Hu）、路易斯·塞泽（Luis Ceze）等。{TVM}：一种用于深度学习的自动化端到端优化编译器。在第 13 届{USENIX}操作系统设计与实现研讨会（$\{$OSDI$\}$18）上，第 578 - 594 页，2018 年。

M. J. Choi, V. Chandrasekaran, D. M. Malioutov, J. K. Johnson, and A. S. Willsky. Multiscale stochastic modeling for tractable inference and data assimilation. Computer Methods in Applied Mechanics and Engineering, 197(43-44):3492-3515, 2008.

M. J. 崔（M. J. Choi）、V. 钱德拉塞卡兰（V. Chandrasekaran）、D. M. 马柳托夫（D. M. Malioutov）、J. K. 约翰逊（J. K. Johnson）和 A. S. 威尔斯基（A. S. Willsky）。用于可处理推理和数据同化的多尺度随机建模。《应用力学与工程中的计算机方法》，197（43 - 44）：3492 - 3515，2008 年。

Junyoung Chung, Sungjin Ahn, and Yoshua Bengio. Hierarchical multiscale recurrent neural networks. In 5th International Conference on Learning Representations, ICLR 2017, 2019.

郑俊英（Junyoung Chung）、安成镇（Sungjin Ahn）和约书亚·本吉奥（Yoshua Bengio）。分层多尺度循环神经网络。发表于第五届学习表征国际会议（5th International Conference on Learning Representations，ICLR 2017），2019年。

Marta R Costa-jussà and José AR Fonollosa. Character-based neural machine translation. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 357-361, 2016.

玛尔塔·R·科斯塔 - 胡萨（Marta R Costa - jussà）和何塞·AR·福诺洛萨（José AR Fonollosa）。基于字符的神经机器翻译。发表于计算语言学协会第54届年会论文集（第2卷：短篇论文），第357 - 361页，2016年。

Jaeyoung Kim, Mostafa El-Khamy, and Jungwon Lee. Residual lstm: Design of a deep recurrent architecture for distant speech recognition. arXiv preprint arXiv:1701.03360, 2017.

金在英（Jaeyoung Kim）、穆斯塔法·埃尔 - 哈米（Mostafa El - Khamy）和李正元（Jungwon Lee）。残差长短期记忆网络（Residual LSTM）：用于远距离语音识别的深度循环架构设计。预印本arXiv:1701.03360，2017年。

Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations, 2019.

尼基塔·基塔耶夫（Nikita Kitaev）、卢卡斯·凯泽（Lukasz Kaiser）和安塞尔姆·列夫斯卡亚（Anselm Levskaya）。改革者：高效的Transformer。发表于学习表征国际会议，2019年。

Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pp. 95-104, 2018.

赖国坤（Guokun Lai）、张维正（Wei - Cheng Chang）、杨一鸣（Yiming Yang）和刘瀚霄（Hanxiao Liu）。使用深度神经网络对长期和短期时间模式进行建模。发表于第41届国际计算机协会信息检索研究与发展会议，第95 - 104页，2018年。

Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. Advances in Neural Information Processing Systems, 32:5243-5253, 2019.

李仕洋（Shiyang Li）、金晓勇（Xiaoyong Jin）、宣姚（Yao Xuan）、周西有（Xiyou Zhou）、陈文虎（Wenhu Chen）、王宇翔（Yu-Xiang Wang）和闫夕峰（Xifeng Yan）。增强Transformer在时间序列预测中的局部性并突破内存瓶颈。《神经信息处理系统进展》，32:5243 - 5253，2019年。

Lesly Miculicich, Dhananjay Ram, Nikolaos Pappas, and James Henderson. Document level neural machine translation with hierarchical attention networks. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), number CONF, 2018.

莱斯利·米库利奇（Lesly Miculicich）、达南杰伊·拉姆（Dhananjay Ram）、尼古拉斯·帕帕斯（Nikolaos Pappas）和詹姆斯·亨德森（James Henderson）。基于分层注意力网络的文档级神经机器翻译。《自然语言处理经验方法会议论文集》（EMNLP），编号CONF，2018年。

Mohsin Munir, Shoaib Ahmed Siddiqui, Andreas Dengel, and Sheraz Ahmed. Deepant: A deep learning approach for unsupervised anomaly detection in time series. Ieee Access, 7:1991-2005, 2018.

莫辛·穆尼尔（Mohsin Munir）、绍艾布·艾哈迈德·西迪基（Shoaib Ahmed Siddiqui）、安德里亚斯·登格尔（Andreas Dengel）和谢拉兹·艾哈迈德（Sheraz Ahmed）。Deepant：一种用于时间序列无监督异常检测的深度学习方法。《IEEE接入》，7:1991 - 2005，2018年。

David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3):1181- 1191, 2020.

大卫·萨利纳斯（David Salinas）、瓦伦丁·弗伦克特（Valentin Flunkert）、扬·加施豪斯（Jan Gasthaus）和蒂姆·亚努绍夫斯基（Tim Januschowski）。Deepar：基于自回归循环网络的概率预测。《国际预测期刊》，36(3):1181 - 1191，2020年。

M. Schuster. Bi-directional recurrent neural networks for speech recognition. In Proceeding of IEEE Canadian Conference on Electrical and ComputerEngineering, pp. 7-12, 1996.

M. 舒斯特（M. Schuster）。用于语音识别的双向循环神经网络。见《电气与电子工程师协会（IEEE）加拿大电气与计算机工程会议论文集》，第7 - 12页，1996年。

Sandeep Subramanian, Ronan Collobert, Marc'Aurelio Ranzato, and Y-Lan Boureau. Multi-scale transformer language models. arXiv preprint arXiv:2005.00581, 2020.

桑迪普·苏布拉马尼亚姆（Sandeep Subramanian）、罗南·科洛贝尔（Ronan Collobert）、马克·奥雷利奥·兰扎托（Marc'Aurelio Ranzato）和伊兰·布雷奥（Y-Lan Boureau）。多尺度变压器语言模型。预印本arXiv:2005.00581，2020年。

Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5693-5703, 2019.

孙珂（Ke Sun）、肖斌（Bin Xiao）、刘东（Dong Liu）和王景东（Jingdong Wang）。用于人体姿态估计的深度高分辨率表征学习。见《电气与电子工程师协会/计算机视觉基金会计算机视觉与模式识别会议论文集》，第5693 - 5703页，2019年。

Sean J Taylor and Benjamin Letham. Forecasting at scale. The American Statistician, 72(1):37-45, 2018.

肖恩·J·泰勒（Sean J Taylor）和本杰明·莱瑟姆（Benjamin Letham）。大规模预测。《美国统计学家》，72(1):37 - 45，2018年。

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pp. 5998-6008, 2017.

阿什ish·瓦斯瓦尼（Ashish Vaswani）、诺姆·沙泽尔（Noam Shazeer）、尼基·帕尔马尔（Niki Parmar）、雅各布·乌斯库雷特（Jakob Uszkoreit）、利昂·琼斯（Llion Jones）、艾丹·N·戈麦斯（Aidan N Gomez）、卢卡斯·凯泽（Łukasz Kaiser）和伊利亚·波洛苏金（Illia Polosukhin）。注意力就是你所需要的一切。见《神经信息处理系统进展》，第5998 - 6008页，2017年。

W. Wang, E. Xie, X. Li, D. P. Fan, and L. Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. 2021.

王（W. Wang）、谢（E. Xie）、李（X. Li）、范（D. P. Fan）和邵（L. Shao）。金字塔视觉变换器：一种无需卷积的用于密集预测的通用主干网络。2021年。

Zihao Ye, Qipeng Guo, Quan Gan, Xipeng Qiu, and Zheng Zhang. Bp-transformer: Modelling long-range context via binary partitioning. arXiv preprint arXiv:1911.04070, 2019.

叶梓豪（Zihao Ye）、郭启鹏（Qipeng Guo）、甘权（Quan Gan）、邱锡鹏（Xipeng Qiu）和张政（Zheng Zhang）。Bp-transformer：通过二元划分对长距离上下文进行建模。预印本arXiv:1911.04070，2019年。

Hang Yu, Luyin Xin, and Justin Dauwels. Variational wishart approximation for graphical model selection: Monoscale and multiscale models. IEEE Transactions on Signal Processing, 67(24): 6468-6482, 2019. doi: 10.1109/TSP.2019.2953651.

于航（Hang Yu）、辛璐荫（Luyin Xin）和贾斯汀·道韦尔斯（Justin Dauwels）。用于图模型选择的变分威沙特近似：单尺度和多尺度模型。《IEEE信号处理汇刊》，67(24)：6468 - 6482，2019年。doi: 10.1109/TSP.2019.2953651。

Hsiang-Fu Yu, Nikhil Rao, and Inderjit S Dhillon. Temporal regularized matrix factorization for high-dimensional time series prediction. Advances in neural information processing systems, 29: 847-855, 2016.

余翔富（Hsiang - Fu Yu）、尼基尔·拉奥（Nikhil Rao）和英德吉特·S·狄龙（Inderjit S Dhillon）。用于高维时间序列预测的时间正则化矩阵分解。《神经信息处理系统进展》，29：847 - 855，2016年。

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of AAAI, 2021.

周浩毅（Haoyi Zhou）、张上航（Shanghang Zhang）、彭杰琦（Jieqi Peng）、张帅（Shuai Zhang）、李建新（Jianxin Li）、熊辉（Hui Xiong）和张万财（Wancai Zhang）。Informer：超越高效Transformer的长序列时间序列预测方法。收录于《AAAI会议论文集》，2021年。

Simiao Zuo, Haoming Jiang, Zichong Li, Tuo Zhao, and Hongyuan Zha. Transformer hawkes process. In International Conference on Machine Learning, pp. 11692-11702. PMLR, 2020.

左思邈（Simiao Zuo）、蒋浩明（Haoming Jiang）、李子冲（Zichong Li）、赵拓（Tuo Zhao）和查宏远（Hongyuan Zha）。Transformer霍克斯过程。见《国际机器学习会议》，第11692 - 11702页。机器学习研究会议录（PMLR），2020年。

<!-- Media -->

Table 4: Meanings of notations.

表4：符号的含义。

<table><tr><td>Notation</td><td>Size</td><td>Meaning</td></tr><tr><td>$L$</td><td>Constant</td><td>The length of historical sequence.</td></tr><tr><td>$G$</td><td>Constant</td><td>The number of global tokens in ETC.</td></tr><tr><td>$M$</td><td>Constant</td><td>The length of future sequence to be predicted.</td></tr><tr><td>$B$</td><td>Constant</td><td>Batch size.</td></tr><tr><td>$D$</td><td>Constant</td><td>The dimension of each node.</td></tr><tr><td>${D}_{K}$</td><td>Constant</td><td>The dimension of a key.</td></tr><tr><td>$X$</td><td>$B \times  L \times  D$</td><td>Input of a single attention head.</td></tr><tr><td>Y</td><td>$B \times  L \times  D$</td><td>Output of a single attention head.</td></tr><tr><td>$Q$</td><td>$B \times  L \times  {D}_{K}$</td><td>The query.</td></tr><tr><td>$K$</td><td>$B \times  L \times  {D}_{K}$</td><td>The key.</td></tr><tr><td>V</td><td>$B \times  L \times  {D}_{K}$</td><td>The value.</td></tr><tr><td>${W}_{Q}$</td><td>$D \times  {D}_{K}$</td><td>The weight matrix of the query.</td></tr><tr><td>${\mathbf{W}}_{K}$</td><td>$D \times  {D}_{K}$</td><td>The weight matrix of the key.</td></tr><tr><td>${\mathbf{W}}_{V}$</td><td>$D \times  {D}_{K}$</td><td>The weight matrix of the value.</td></tr><tr><td>$S$</td><td>Constant</td><td>Number of scales.</td></tr><tr><td>$A$</td><td>Constant</td><td>Number of adjacent nodes at the same scale that a node can attend to.</td></tr><tr><td>$C$</td><td>Constant</td><td>Number of finer scale nodes that a coarser scale node can summarize.</td></tr><tr><td>$N$</td><td>Constant</td><td>Number of attention layers.</td></tr><tr><td>${n}_{l}^{\left( s\right) }$</td><td>$D$</td><td>The $\ell$ -th node at scale s.</td></tr><tr><td>${\mathbb{N}}_{\ell }^{\left( s\right) }$</td><td>$\operatorname{len}\left( {\mathbb{N}}_{\ell }^{\left( s\right) }\right)  \times  D$</td><td>The set of neighboring nodes of node ${n}_{l}^{\left( s\right) }$ .</td></tr><tr><td>${\mathbb{A}}_{\ell }^{\left( s\right) }$</td><td>$\operatorname{len}\left( {\mathbb{A}}_{\ell }^{\left( s\right) }\right)  \times  D$</td><td>The adjacent A nodes at the same scale with ${n}_{l}^{\left( s\right) }$ .</td></tr><tr><td>${\mathbb{C}}_{\ell }^{\left( s\right) }$</td><td>$\operatorname{len}\left( {\mathbb{C}}_{\ell }^{\left( s\right) }\right)  \times  D$</td><td>The children nodes of ${n}_{l}^{\left( s\right) }$ .</td></tr><tr><td>${\mathbb{P}}_{\ell }^{\left( s\right) }$</td><td>$\operatorname{len}\left( {\mathbb{P}}_{\ell }^{\left( s\right) }\right)  \times  D$</td><td>The parent node of ${n}_{l}^{\left( s\right) }$ .</td></tr><tr><td>${\mathbf{F}}_{p}$</td><td>$B \times  M \times  D$</td><td>The prediction tokens.</td></tr><tr><td>${\mathbf{F}}_{e}$</td><td>$B \times  {L}_{tot} \times  D$</td><td>The output of the encoder. ${L}_{tot}$ represents the output length of the encoder.</td></tr><tr><td>${\mathbf{F}}_{d1}$</td><td>$B \times  M \times  D$</td><td>The output of the first attention-based decoder layer.</td></tr><tr><td>$H$</td><td>Constant</td><td>The number of attention heads.</td></tr><tr><td>${D}_{F}$</td><td>Constant</td><td>The maximum dimension of the feed-forward layer.</td></tr></table>

<table><tbody><tr><td>符号表示</td><td>大小</td><td>含义</td></tr><tr><td>$L$</td><td>常量</td><td>历史序列的长度。</td></tr><tr><td>$G$</td><td>常量</td><td>以太坊经典（ETC）中全局令牌的数量。</td></tr><tr><td>$M$</td><td>常量</td><td>待预测的未来序列的长度。</td></tr><tr><td>$B$</td><td>常量</td><td>批量大小。</td></tr><tr><td>$D$</td><td>常量</td><td>每个节点的维度。</td></tr><tr><td>${D}_{K}$</td><td>常量</td><td>键（key）的维度。</td></tr><tr><td>$X$</td><td>$B \times  L \times  D$</td><td>单个注意力头的输入。</td></tr><tr><td>Y</td><td>$B \times  L \times  D$</td><td>单个注意力头的输出。</td></tr><tr><td>$Q$</td><td>$B \times  L \times  {D}_{K}$</td><td>查询项。</td></tr><tr><td>$K$</td><td>$B \times  L \times  {D}_{K}$</td><td>键。</td></tr><tr><td>V</td><td>$B \times  L \times  {D}_{K}$</td><td>值。</td></tr><tr><td>${W}_{Q}$</td><td>$D \times  {D}_{K}$</td><td>查询项的权重矩阵。</td></tr><tr><td>${\mathbf{W}}_{K}$</td><td>$D \times  {D}_{K}$</td><td>键的权重矩阵。</td></tr><tr><td>${\mathbf{W}}_{V}$</td><td>$D \times  {D}_{K}$</td><td>值的权重矩阵。</td></tr><tr><td>$S$</td><td>常量</td><td>尺度数量。</td></tr><tr><td>$A$</td><td>常量</td><td>一个节点在同一尺度上可以关注的相邻节点数量。</td></tr><tr><td>$C$</td><td>常量</td><td>一个较粗尺度节点可以汇总的较细尺度节点数量。</td></tr><tr><td>$N$</td><td>常量</td><td>注意力层的数量。</td></tr><tr><td>${n}_{l}^{\left( s\right) }$</td><td>$D$</td><td>尺度s上的第$\ell$个节点。</td></tr><tr><td>${\mathbb{N}}_{\ell }^{\left( s\right) }$</td><td>$\operatorname{len}\left( {\mathbb{N}}_{\ell }^{\left( s\right) }\right)  \times  D$</td><td>节点${n}_{l}^{\left( s\right) }$的相邻节点集合。</td></tr><tr><td>${\mathbb{A}}_{\ell }^{\left( s\right) }$</td><td>$\operatorname{len}\left( {\mathbb{A}}_{\ell }^{\left( s\right) }\right)  \times  D$</td><td>与 ${n}_{l}^{\left( s\right) }$ 处于同一尺度的相邻 A 节点。</td></tr><tr><td>${\mathbb{C}}_{\ell }^{\left( s\right) }$</td><td>$\operatorname{len}\left( {\mathbb{C}}_{\ell }^{\left( s\right) }\right)  \times  D$</td><td>${n}_{l}^{\left( s\right) }$ 的子节点。</td></tr><tr><td>${\mathbb{P}}_{\ell }^{\left( s\right) }$</td><td>$\operatorname{len}\left( {\mathbb{P}}_{\ell }^{\left( s\right) }\right)  \times  D$</td><td>${n}_{l}^{\left( s\right) }$ 的父节点。</td></tr><tr><td>${\mathbf{F}}_{p}$</td><td>$B \times  M \times  D$</td><td>预测标记。</td></tr><tr><td>${\mathbf{F}}_{e}$</td><td>$B \times  {L}_{tot} \times  D$</td><td>编码器的输出。${L}_{tot}$ 表示编码器的输出长度。</td></tr><tr><td>${\mathbf{F}}_{d1}$</td><td>$B \times  M \times  D$</td><td>第一个基于注意力机制的解码器层的输出。</td></tr><tr><td>$H$</td><td>常量</td><td>注意力头的数量。</td></tr><tr><td>${D}_{F}$</td><td>常量</td><td>前馈层的最大维度。</td></tr></tbody></table>

<!-- Media -->

## A A BRIEF REVIEW ON RELATED RNN-BASED MODELS

## 基于循环神经网络（RNN）的相关模型简要综述

In this section, we provide a brief review on the related RNN-based models. Multiscale temporal dependencies are successfully captured in HRNN (Costa-jussà & Fonollosa, 2016) and HM-RNN (Chung et al. 2019). The former requires expert knowledge to partition the sequence into different resolutions, while the latter learns the partition automatically from the data. Note that the theoretical maximum length of the signal traversing path in both models is still $\mathcal{O}\left( L\right)$ . Another line of works aim to shorten the signal traversing path by adding residual connections (Kim et al. 2017) or dilated connections to LSTMs (Chang et al. 2017). However, they do not consider the multires-olution temporal dependencies explicitly. Furthermore, all aforementioned RNNs only propagate information in one direction from the past to the future. An appealing approach that allows bidirectional information exchange is Bi-LSTM (Schuster, 1996). The forward and backward propagation is realized through two different LSTMs though, and so still incurs a long signal traversing path. As opposed to the abovementioned RNN-based models, the proposed Pyraformer enables bidirectional information exchange that can better describe the temporal dependencies, while providing a multiresolution representation of the observed sequence at the same time. We also notice that due to the unidirectional property of RNNs, it is difficult the realize the pyramidal graph in Figure 1d based on RNNs.

在本节中，我们对相关的基于循环神经网络（RNN）的模型进行简要回顾。分层循环神经网络（HRNN）（科斯塔 - 胡萨（Costa - jussà）和福诺洛萨（Fonollosa），2016年）和分层记忆循环神经网络（HM - RNN）（钟（Chung）等人，2019年）成功捕捉到了多尺度时间依赖关系。前者需要专业知识将序列划分为不同分辨率，而后者则从数据中自动学习划分。请注意，这两种模型中信号遍历路径的理论最大长度仍然是 $\mathcal{O}\left( L\right)$ 。另一类工作旨在通过向长短期记忆网络（LSTM）添加残差连接（金（Kim）等人，2017年）或扩张连接（张（Chang）等人，2017年）来缩短信号遍历路径。然而，它们并未明确考虑多分辨率时间依赖关系。此外，上述所有循环神经网络仅沿从过去到未来的单一方向传播信息。一种允许双向信息交换的有吸引力的方法是双向长短期记忆网络（Bi - LSTM）（舒斯特（Schuster），1996年）。不过，前向和后向传播是通过两个不同的长短期记忆网络实现的，因此仍然会产生较长的信号遍历路径。与上述基于循环神经网络的模型不同，本文提出的金字塔变换器（Pyraformer）能够实现双向信息交换，从而更好地描述时间依赖关系，同时为观测序列提供多分辨率表示。我们还注意到，由于循环神经网络的单向特性，基于循环神经网络实现图1d中的金字塔图较为困难。

## B Proof of Lemma 1

## B 引理1的证明

Proof. Let $S$ denote the number of scales in the pyramidal graph, $C$ the number of children nodes in the finer scale $s - 1$ that a node in the the coarser scale $s$ can summarize for $s = 2,\cdots ,S,A$ the number of adjacent nodes that a node can attend to within each scale, $N$ the number of attention layers,and $L$ the length of the input time series. We define the term "receptive field" of an arbitrary node ${n}_{a}$ in a graph as the set of nodes that ${n}_{a}$ can receive messages from. We further define the distance between two arbitrary nodes in a graph as the length of the shortest path between them (i.e., the number of steps to travel from one node to another). Note that in each attention layer, the messages can only travel by one step in the graph.

证明。设$S$表示金字塔图中尺度的数量，$C$表示较粗尺度$s$中的一个节点能够为较细尺度$s - 1$汇总的子节点数量，$s = 2,\cdots ,S,A$表示每个尺度内一个节点可以关注的相邻节点数量，$N$表示注意力层的数量，$L$表示输入时间序列的长度。我们将图中任意节点${n}_{a}$的“感受野”定义为${n}_{a}$可以接收消息的节点集合。我们进一步将图中任意两个节点之间的距离定义为它们之间最短路径的长度（即从一个节点到另一个节点所需的步数）。请注意，在每个注意力层中，消息在图中只能传播一步。

Without sacrificing generality,we assume that $L$ is divisible by ${C}^{S - 1}$ ,and then the number of nodes at the coarsest scale $S$ is $L/{C}^{S - 1}$ . Since every node is connected to $A$ closest nodes at the same scale, the distance between the leftmost and the rightmost node at the coarsest scale is $2\left( {L/{C}^{S - 1} - 1}\right) /\left( {A - 1}\right)$ . Hence,the leftmost and the rightmost node in the coarsest scale are in the receptive field of each other after the stack of $N \geq  2\left( {L/{C}^{S - 1} - 1}\right) /\left( {A - 1}\right)$ layers of the pyramidal attention. In addition, owing to the CSCM, nodes at the coarsest scale can be regarded as the summary of the nodes in the finer scales. As a result, when Equation (4) is satisfied, all nodes at the coarsest scale have a global receptive field, which closes the proof.

不失一般性，我们假设$L$能被${C}^{S - 1}$整除，那么最粗尺度$S$上的节点数量为$L/{C}^{S - 1}$。由于每个节点在同一尺度上都与$A$个最近的节点相连，所以最粗尺度上最左侧节点和最右侧节点之间的距离为$2\left( {L/{C}^{S - 1} - 1}\right) /\left( {A - 1}\right)$。因此，在经过$N \geq  2\left( {L/{C}^{S - 1} - 1}\right) /\left( {A - 1}\right)$层金字塔注意力堆叠后，最粗尺度上的最左侧节点和最右侧节点处于彼此的感受野内。此外，由于CSCM（粗尺度上下文建模，Coarse Scale Context Modeling）的存在，最粗尺度上的节点可以被视为更细尺度上节点的总结。结果，当满足方程(4)时，最粗尺度上的所有节点都具有全局感受野，证明完毕。

## C Proof of Proposition 1

## 命题1的证明

Proof. Suppose that ${L}^{\left( s\right) }$ denotes the number of nodes at scale $s$ ,that is,

证明。假设 ${L}^{\left( s\right) }$ 表示尺度 $s$ 下的节点数量，即

$$
{L}^{\left( s\right) } = \frac{L}{{C}^{s - 1}},1 \leq  s \leq  S. \tag{6}
$$

For a node ${n}_{\ell }^{\left( s\right) }$ in the pyramidal graph,the number of dot products ${P}_{\ell }^{\left( s\right) }$ it acts as the query can be decomposed into two parts:

对于金字塔图中的节点 ${n}_{\ell }^{\left( s\right) }$，它作为查询所参与的点积 ${P}_{\ell }^{\left( s\right) }$ 的数量可以分解为两部分：

$$
{P}_{\ell }^{\left( s\right) } = {P}_{\ell }^{\left( s\right) }{}_{\text{inter }} + {P}_{\ell }^{\left( s\right) }{}_{\text{intra }}, \tag{7}
$$

where ${P}_{\ell }^{\left( s\right) }$ intra and ${P}_{\ell }^{\left( s\right) }$ inter denotes the intra-scale and the inter-scale part respectively. According to the structure of the pyramidal graph, we can have the following inequalities:

其中 ${P}_{\ell }^{\left( s\right) }$ 内部和 ${P}_{\ell }^{\left( s\right) }$ 外部 分别表示内部尺度和跨尺度部分。根据金字塔图的结构，我们可以得到以下不等式：

$$
{P}_{\ell }^{\left( s\right) }{}_{\text{intra }} \leq  A, \tag{8}
$$

$$
{P}_{\ell }^{\left( s\right) }{}_{\text{inter }} \leq  C + 1 \tag{9}
$$

The first inequality (8) holds since a node typically attends to $A$ most adjacent nodes at the same scale but for the leftmost and the rightmost node, the number of in-scale nodes it can attend to is smaller than $A$ . On the other hand,the second inequality (9) holds because a node typically has $C$ children and 1 parent in the pyramidal graph but nodes at the top and the bottom scale can only attend to fewer than $C + 1$ nodes at adjacent scales.

第一个不等式 (8) 成立，因为一个节点通常在同一尺度上最多关注 $A$ 个相邻节点，但对于最左侧和最右侧的节点，它能关注的同尺度节点数量小于 $A$。另一方面，第二个不等式 (9) 成立，因为在金字塔图中，一个节点通常有 $C$ 个子节点和 1 个父节点，但顶层和底层尺度的节点在相邻尺度上只能关注少于 $C + 1$ 个节点。

In summary,the number of dot products that need to be calculated for scale $s$ is:

综上所述，尺度 $s$ 需要计算的点积数量为：

$$
{P}^{\left( s\right) } = \mathop{\sum }\limits_{{\ell  = 1}}^{{L}^{\left( s\right) }}\left( {{P}_{\ell }^{\left( s\right) }{}_{\text{intra }} + {P}_{\ell }^{\left( s\right) }{}_{\text{inter }}}\right)  \leq  {L}^{\left( s\right) }\left( {A + C + 1}\right) . \tag{10}
$$

Note that ${P}^{\left( 1\right) } \leq  L\left( {A + 1}\right)$ for the finest scale (i.e., $s = 1$ ) since nodes at this scale do not have any children. It follows that the number of dot products that need to be calculated for the entire pyramidal attention layer is:

注意，对于最精细的尺度（即 $s = 1$），${P}^{\left( 1\right) } \leq  L\left( {A + 1}\right)$ 成立，因为该尺度的节点没有任何子节点。由此可知，整个金字塔注意力层需要计算的点积数量为：

$$
P = \mathop{\sum }\limits_{{s = 1}}^{S}{P}^{\left( s\right) }
$$

$$
 \leq  L\left( {A + 1}\right)  + {L}^{\left( 2\right) }\left( {A + C + 1}\right)  + \ldots  + {L}^{\left( S\right) }\left( {A + C + 1}\right) 
$$

$$
 = L\left( {\mathop{\sum }\limits_{{s = 1}}^{S}{C}^{-\left( {s - 1}\right) }A + \mathop{\sum }\limits_{{s = 2}}^{S}{C}^{-\left( {s - 1}\right) } + \mathop{\sum }\limits_{{s = 1}}^{{S - 1}}{C}^{-\left( {s - 1}\right) } + 1}\right) 
$$

$$
 < L\left( {\left( {A + 2}\right) \mathop{\sum }\limits_{{s = 1}}^{S}{C}^{-\left( {s - 1}\right) } + 1}\right) . \tag{11}
$$

In order to guarantee that the nodes at the coarsest scale have a global receptive field,we choose $C$ such that $C \propto  \sqrt[{S - 1}]{L}$ . Consequently,the complexity of the proposed pyramidal attention is:

为了确保最粗尺度的节点具有全局感受野，我们选择$C$使得$C \propto  \sqrt[{S - 1}]{L}$成立。因此，所提出的金字塔注意力机制的复杂度为：

$$
\mathcal{O}\left( P\right)  \leq  \mathcal{O}\left( {L\left( {\left( {A + 2}\right) \mathop{\sum }\limits_{{s = 1}}^{S}{C}^{-\left( {s - 1}\right) } + 1}\right) }\right) 
$$

$$
 = \mathcal{O}\left( {L\left( {A + 2}\right) \mathop{\sum }\limits_{{s = 1}}^{S}{C}^{-\left( {s - 1}\right) }}\right) 
$$

$$
 = \mathcal{O}\left( \frac{\left( {A + 2}\right) {L}^{\frac{S}{S - 1}} - 1}{{L}^{\frac{1}{S - 1}} - 1}\right) 
$$

$$
 = \mathcal{O}\left( \frac{A{L}^{\frac{S}{S - 1}} - 1}{{L}^{\frac{1}{S - 1}} - 1}\right) . \tag{12}
$$

When $L$ approaches infinity,the above expression amounts to $\mathcal{O}\left( {AL}\right)$ . Since $A$ can be fixed when $L$ changes,the complexity can be further reduced to $\mathcal{O}\left( L\right)$ .

当$L$趋近于无穷大时，上述表达式等价于$\mathcal{O}\left( {AL}\right)$。由于当$L$变化时$A$可以固定，复杂度可以进一步降低到$\mathcal{O}\left( L\right)$。

## D Proof of Proposition 2

## D 命题2的证明

Proof. Let ${n}_{\ell }^{\left( s\right) }$ represent the $\ell$ -th node of the $s$ -th scale. It is evident that the distance between ${n}_{1}^{\left( 1\right) }$ and ${n}_{L}^{\left( 1\right) }$ is the largest among all pairs of nodes in the pyramidal graph. The shortest path to travel from ${n}_{1}^{\left( 1\right) }$ to ${n}_{L}^{\left( s\right) }$ is:

证明。设 ${n}_{\ell }^{\left( s\right) }$ 表示第 $s$ 层的第 $\ell$ 个节点。显然，在金字塔图的所有节点对中，${n}_{1}^{\left( 1\right) }$ 和 ${n}_{L}^{\left( 1\right) }$ 之间的距离是最大的。从 ${n}_{1}^{\left( 1\right) }$ 到 ${n}_{L}^{\left( s\right) }$ 的最短路径为：

$$
{n}_{1}^{\left( 1\right) } \rightarrow  {n}_{1}^{\left( 2\right) } \rightarrow  \cdots  \rightarrow  {n}_{1}^{\left( S\right) } \rightarrow  \cdots  \rightarrow  {n}_{{L}^{\left( S\right) }}^{\left( S\right) } \rightarrow  {n}_{{L}^{\left( S - 1\right) }}^{\left( S - 1\right) } \rightarrow  \cdots  \rightarrow  {n}_{L}^{\left( 1\right) }. \tag{13}
$$

Correspondingly, the length of the maximum path between two arbitrary nodes in the graph is:

相应地，图中任意两个节点之间的最长路径长度为：

$$
{L}_{\max } = 2\left( {S - 1}\right)  + \frac{2\left( {{L}^{\left( S\right) } - 1}\right) }{A - 1}. \tag{14}
$$

When $C$ satisfies Equation (5),that is, ${L}^{\left( S\right) } - 1 \leq  \left( {A - 1}\right) N/2$ ,we can obtain:

当 $C$ 满足方程 (5)，即 ${L}^{\left( S\right) } - 1 \leq  \left( {A - 1}\right) N/2$ 时，我们可以得到：

$$
\mathcal{O}\left( {L}_{\max }\right)  = \mathcal{O}\left( {2\left( {S - 1}\right)  + \frac{2\left( {{L}^{\left( S\right) } - 1}\right) }{A - 1}}\right) 
$$

$$
 = \mathcal{O}\left( {2\left( {S - 1}\right)  + \frac{2\left( {\frac{L}{{C}^{S - 1}} - 1}\right) }{A - 1}}\right) 
$$

$$
 = \mathcal{O}\left( {2\left( {S - 1}\right)  + N}\right) 
$$

$$
 = \mathcal{O}\left( {S + N}\right) \text{.} \tag{15}
$$

Since $A,S$ and $N$ are invariant with $L$ ,the order of the maximum path length ${L}_{\max }$ can be further simplified as $\mathcal{O}\left( 1\right)$ .

由于$A,S$和$N$相对于$L$是不变的，最大路径长度${L}_{\max }$的阶可以进一步简化为$\mathcal{O}\left( 1\right)$。

## E DATASETS

## E 数据集

We demonstrated the advantages of the proposed Pyraformer on the following four datasets. The first three datasets were used for single-step forecasting, while the last two for long-range multi-step forecasting.

我们在以下四个数据集上展示了所提出的Pyraformer（金字塔变换器）的优势。前三个数据集用于单步预测，而后两个用于长程多步预测。

Wind ${}^{2}$ . This dataset contains hourly estimation of the energy potential in 28 countries between 1986 and 2015 as a percentage of a power plant's maximum output. Compared with the remaining datasets, it is more sparse and periodically exhibits a large number of zeros. Due to the large size of this dataset, the ratio between training and testing set was roughly 32:1.

风能${}^{2}$。该数据集包含1986年至2015年间28个国家每小时的能源潜力估计值，以发电厂最大输出的百分比表示。与其余数据集相比，它更为稀疏，并且周期性地出现大量零值。由于该数据集规模较大，训练集和测试集的比例约为32:1。

App Flow: This dataset was collected at Ant Group ${}^{3}$ . It consists of hourly maximum traffic flow for 128 systems deployed on 16 logic data centers, resulting in 1083 different time series in total. The length of each series is more than 4 months. Each time series was divided into two segments for training and testing respectively, with a ratio of 32:1.

应用程序流量：该数据集由蚂蚁集团（Ant Group）${}^{3}$收集。它包含部署在16个逻辑数据中心的128个系统的每小时最大流量，总共产生了1083个不同的时间序列。每个序列的长度超过4个月。每个时间序列分别按32:1的比例划分为训练段和测试段。

Electricity ${}^{4}$ (Yu et al. 2016): This dataset contains time series of electricity consumption recorded every 15 minutes from 370 users. Following DeepAR (Salinas et al. 2020), we aggregated every 4 records to get the hourly observations. This dataset was employed for both single-step and long-range forecasting. We trained with data from 2011-01-01 to 2014-09-01 for single-step forecasting, and from 2011-04-01 to 2014-04-01 for long-range forecasting.

电力${}^{4}$（Yu等人，2016年）：该数据集包含370个用户每15分钟记录一次的电力消耗时间序列。按照DeepAR（Salinas等人，2020年）的方法，我们将每4条记录进行聚合，以获得每小时的观测值。该数据集用于单步和长期预测。对于单步预测，我们使用2011年1月1日至2014年9月1日的数据进行训练；对于长期预测，我们使用2011年4月1日至2014年4月1日的数据进行训练。

ETT ${}^{5}$ (Zhou et al. 2021): This dataset comprises 2 years of 2 electricity transformers collected from 2 stations, including the oil temperature and 6 power load features. Observations every hour (i.e., ETTh1) and every 15 minutes (i.e., ETTm1) are provided. This dataset is typically exploited for model assessment on long-range forecasting. Here, we followed Informer (Zhou et al. 2021) and partitioned the data into 12 and 4 months for training and testing respectively.

ETT ${}^{5}$（周等人，2021年）：该数据集包含从2个站点收集的2台电力变压器2年的数据，包括油温以及6个电力负荷特征。提供了每小时（即ETTh1）和每15分钟（即ETTm1）的观测数据。该数据集通常用于长期预测的模型评估。在此，我们遵循《Informer》（周等人，2021年）的方法，将数据分别划分为12个月和4个月用于训练和测试。

## F EXPERIMENT SETUP

## F 实验设置

We set $S = 4$ and $N = 4$ for Pyraformer in all experiments. When the historical length $L$ is not divisible by $C$ ,we only introduced $\lfloor L/C\rfloor$ nodes in the upper scale,where $\lfloor  \cdot  \rfloor$ denotes the round down operation. The last $L - \left( {\lfloor L/\bar{C}\rfloor  - 1}\right) C$ nodes at the bottom scale were all connected to the last node at the upper scale. For single-step forecasting,we set $C = 4,A = 3$ ,and $H = 4$ in all experiments. Both training and testing used a fixed-size historical sequence to predict the mean and variance of the Gaussian distribution of a single future value. We chose the MSE loss and the log-likelihood (Zuo et al. 2020) as our loss functions. The ratio between them was set to 100 . For optimization,we used Adam with the learning rate starting from ${10}^{-5}$ and halving in every epoch. We trained Pyraformer with 10 epochs. Weighted sampler based on each window's average value and hard sample mining were used to improve the generalization ability of the network. On the other hand,for long-range forecasting,we tested four combinations of $A$ and $C$ in each experiment, and the best results were presented. Specifically, when the prediction length is smaller than 600, we tested $A = 3,5$ and $C = 4,5$ . When the prediction length is larger than 600,we tested $A = 3,5$ and $C = 5,6$ . The resulting choice of hyper-parameters for each experiment is listed in Table 5 In addition, the loss function was the MSE loss only. We still used Adam as our optimizer, but the learning rate started from ${10}^{-4}$ and was reduced to one-tenth every epoch. We set the number of epochs to be 5 .

在所有实验中，我们为金字塔变换器（Pyraformer）设置了$S = 4$和$N = 4$。当历史长度$L$不能被$C$整除时，我们仅在上层尺度引入$\lfloor L/C\rfloor$个节点，其中$\lfloor  \cdot  \rfloor$表示向下取整运算。底层尺度的最后$L - \left( {\lfloor L/\bar{C}\rfloor  - 1}\right) C$个节点都与上层尺度的最后一个节点相连。对于单步预测，我们在所有实验中设置$C = 4,A = 3$和$H = 4$。训练和测试均使用固定大小的历史序列来预测单个未来值的高斯分布的均值和方差。我们选择均方误差损失（MSE loss）和对数似然（左等人，2020）作为损失函数。它们之间的比例设置为100。为了进行优化，我们使用Adam优化器，学习率从${10}^{-5}$开始，每一轮训练减半。我们对金字塔变换器（Pyraformer）进行了10轮训练。基于每个窗口的平均值的加权采样器和难样本挖掘被用于提高网络的泛化能力。另一方面，对于长程预测，我们在每个实验中测试了$A$和$C$的四种组合，并展示了最佳结果。具体而言，当预测长度小于600时，我们测试了$A = 3,5$和$C = 4,5$。当预测长度大于600时，我们测试了$A = 3,5$和$C = 5,6$。每个实验所选择的超参数结果列于表5中。此外，损失函数仅为均方误差损失（MSE loss）。我们仍然使用Adam作为优化器，但学习率从${10}^{-4}$开始，每一轮训练降低至原来的十分之一。我们将训练轮数设置为5。

## G PRETRAINING

## G 预训练

For single-step forecasting, the value to be predicted is usually close to the last value of history. Since we only use the last nodes of all scales to predict, the network tends to focus only on short-term dependencies. To force the network to capture long-range dependencies, we add additional supervision in the first few epochs of training. Specifically, in the first epoch, we form our network as an auto-encoder, as shown in Figure 5 . Apart from predicting future values, the PAM is also

对于单步预测，待预测的值通常接近历史数据的最后一个值。由于我们仅使用所有尺度的最后节点进行预测，网络往往只关注短期依赖关系。为了迫使网络捕捉长距离依赖关系，我们在训练的前几个轮次（epoch）中添加额外的监督。具体来说，在第一个轮次中，我们将网络构建为一个自编码器，如图 5 所示。除了预测未来值之外，PAM（位置注意力模块，Position Attention Module）也是

---

<!-- Footnote -->

${}^{2}$ Wind dataset can be downloaded at https://www.kaggle.com/sohier/30-years-of-european-wind-generation

${}^{2}$ 风能数据集可从以下链接下载：https://www.kaggle.com/sohier/30 - years - of - european - wind - generation

${}^{3}$ The App Flow dataset does not contain any Personal Identifiable Information and is desensitized and encrypted. Adequate data protection was carried out during the experiment to prevent the risk of data copy leakage, and the dataset was destroyed after the experiment. It is only used for academic research, it does not represent any real business situation. The download link is https://github.com/alipay/Pyraformer/tree/master/data/app_zone_rpc_hour_encrypted.csv

${}^{3}$ 应用流数据集不包含任何个人身份信息，且经过脱敏和加密处理。实验期间进行了充分的数据保护，以防止数据复制泄露的风险，实验结束后销毁了该数据集。它仅用于学术研究，不代表任何实际业务情况。下载链接为https://github.com/alipay/Pyraformer/tree/master/data/app_zone_rpc_hour_encrypted.csv

${}^{4}$ Electricity dataset can be downloaded at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20 112014

${}^{4}$ 电力数据集可从https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20 112014下载。

${}^{5}$ ETT dataset can be downloaded at https:// github.com/zhouhaoyi/ETDataset

${}^{5}$ ETT数据集可从https:// github.com/zhouhaoyi/ETDataset下载。

<!-- Footnote -->

---

<!-- Media -->

Table 5: Hyper-parameter settings of long-range experiments.

表5：远程实验的超参数设置。

<table><tr><td>Dataset</td><td>prediction length</td><td>$\mathbf{N}$</td><td>S</td><td>$\mathbf{H}$</td><td>A</td><td>C</td><td>historical length</td></tr><tr><td rowspan="3">ETTh1</td><td>168</td><td>4</td><td>4</td><td>6</td><td>3</td><td>4</td><td>168</td></tr><tr><td>336</td><td>4</td><td>4</td><td>6</td><td>3</td><td>4</td><td>168</td></tr><tr><td>720</td><td>4</td><td>4</td><td>6</td><td>5</td><td>4</td><td>336</td></tr><tr><td rowspan="3">ETTm1</td><td>96</td><td>4</td><td>4</td><td>6</td><td>3</td><td>5</td><td>384</td></tr><tr><td>288</td><td>4</td><td>4</td><td>6</td><td>5</td><td>5</td><td>672</td></tr><tr><td>672</td><td>4</td><td>4</td><td>6</td><td>3</td><td>6</td><td>672</td></tr><tr><td rowspan="3">Elect</td><td>168</td><td>4</td><td>4</td><td>6</td><td>3</td><td>4</td><td>168</td></tr><tr><td>336</td><td>4</td><td>4</td><td>6</td><td>3</td><td>4</td><td>168</td></tr><tr><td>720</td><td>4</td><td>4</td><td>6</td><td>3</td><td>5</td><td>336</td></tr></table>

<table><tbody><tr><td>数据集</td><td>预测长度</td><td>$\mathbf{N}$</td><td>S</td><td>$\mathbf{H}$</td><td>A</td><td>C</td><td>历史长度</td></tr><tr><td rowspan="3">ETTh1</td><td>168</td><td>4</td><td>4</td><td>6</td><td>3</td><td>4</td><td>168</td></tr><tr><td>336</td><td>4</td><td>4</td><td>6</td><td>3</td><td>4</td><td>168</td></tr><tr><td>720</td><td>4</td><td>4</td><td>6</td><td>5</td><td>4</td><td>336</td></tr><tr><td rowspan="3">ETTm1</td><td>96</td><td>4</td><td>4</td><td>6</td><td>3</td><td>5</td><td>384</td></tr><tr><td>288</td><td>4</td><td>4</td><td>6</td><td>5</td><td>5</td><td>672</td></tr><tr><td>672</td><td>4</td><td>4</td><td>6</td><td>3</td><td>6</td><td>672</td></tr><tr><td rowspan="3">电力</td><td>168</td><td>4</td><td>4</td><td>6</td><td>3</td><td>4</td><td>168</td></tr><tr><td>336</td><td>4</td><td>4</td><td>6</td><td>3</td><td>4</td><td>168</td></tr><tr><td>720</td><td>4</td><td>4</td><td>6</td><td>3</td><td>5</td><td>336</td></tr></tbody></table>

<img src="https://cdn.noedgeai.com/01957b4b-1654-7ad6-8e71-2da613767aa1_15.jpg?x=625&y=782&w=543&h=558&r=0"/>

Figure 5: The pretraining strategy for one-step prediction. Features of nodes surrounded by the dashed ellipses are concatenated to recover the corresponding input value.

图5：一步预测的预训练策略。用虚线椭圆圈出的节点特征被拼接起来以恢复相应的输入值。

<!-- Media -->

trained to recover the input values. Note that we test all methods with and without this pretraining strategy and the better results are displayed in Table 2

经过训练以恢复输入值。请注意，我们对所有方法分别测试了使用和不使用这种预训练策略的情况，表2中展示了更好的结果。

## H METRICS

## H 指标

Denote the target value as ${z}_{j,t}$ and the predicted value as ${\widehat{z}}_{j,t}$ ,where $j$ is the sample index and $t$ is the time index. Then NRMSE and ND are calculated as follows:

将目标值表示为 ${z}_{j,t}$，预测值表示为 ${\widehat{z}}_{j,t}$，其中 $j$ 是样本索引，$t$ 是时间索引。然后，归一化均方根误差（NRMSE）和归一化偏差（ND）的计算如下：

$$
\text{ NRMSE } = \frac{\sqrt{\frac{1}{NT}\mathop{\sum }\limits_{{j = 1}}^{N}\mathop{\sum }\limits_{{t = 1}}^{T}{\left( {z}_{j,t} - {\widehat{z}}_{j,t}\right) }^{2}}}{\frac{1}{NT}\mathop{\sum }\limits_{{j = 1}}^{N}\mathop{\sum }\limits_{{t = 1}}^{T}\left| {z}_{j,t}\right| }, \tag{16}
$$

$$
\mathrm{{ND}} = \frac{\mathop{\sum }\limits_{{j = 1}}^{N}\mathop{\sum }\limits_{{t = 1}}^{T}\left| {{z}_{j,t} - {\widehat{z}}_{j,t}}\right| }{\mathop{\sum }\limits_{{j = 1}}^{N}\mathop{\sum }\limits_{{t = 1}}^{T}\left| {z}_{j,t}\right| }. \tag{17}
$$

## I EXPERIMENTS ON SYNTHETIC DATA

## I 合成数据实验

To further evaluate Pyraformer's ability to capture different ranges of temporal dependencies, we synthesized an hourly dataset with multi-range dependencies and carried out experiments on it. Specifically, each time series in the synthetic dataset is a linear combination of three sine functions of different periods:24,168and 720 ,that is,

为了进一步评估Pyraformer捕捉不同范围时间依赖关系的能力，我们合成了一个具有多范围依赖关系的每小时数据集，并在该数据集上进行了实验。具体来说，合成数据集中的每个时间序列都是三个不同周期（24、168和720）的正弦函数的线性组合，即：

$$
f\left( t\right)  = {\beta }_{0} + {\beta }_{1}\sin \left( {\frac{2\pi }{24}t}\right)  + {\beta }_{2}\sin \left( {\frac{2\pi }{168}t}\right)  + {\beta }_{3}\sin \left( {\frac{2\pi }{720}t}\right) . \tag{18}
$$

In the above equation,the coefficients of the three sine functions ${\beta }_{1},{\beta }_{2}$ ,and ${\beta }_{3}$ for each time series are uniformly sampled from $\left\lbrack  {5,{10}}\right\rbrack  .{\beta }_{0}$ is a Gaussian process with a covariance function ${\sum }_{{t}_{1},{t}_{2}} = {\left| {t}_{1} - {t}_{2}\right| }^{-1}$ and ${\sum }_{{t}_{1}} = {\sum }_{{t}_{2}} = 1$ ,where ${t}_{1}$ and ${t}_{2}$ denote two arbitrary time stamps. Such polynomially decaying covariance functions are known to have long-range dependence, as oppose to the exponentially decaying covariance functions (Yu et al. 2019). The start time of each time series ${t}_{0}$ is uniformly sampled from [0,719]. We first generate 60 time series of length 14400,and then split each time series into sliding windows of width 1440 with a stride of 24 . In our experiments, we use the historical 720 time points to predict the future 720 points. Since both the deterministic and stochastic parts of the synthetic time series have long-range correlations, such dependencies should be well captured in the model in order to yield accurate predictions of the next 720 points. The results are summarized in Table 6 Here, we consider two different configurations of Pyraformer: 1) $C = 6$ for all scales in the pyramidal graph (denoted as Pyraformer ${}_{6,6,6}$ ); 2) $C = {12},7$ ,and 4 for the three layers sequentially from bottom to top (denoted as Pyraformer ${}_{{12},7,4}$ ).

在上述方程中，每个时间序列的三个正弦函数 ${\beta }_{1},{\beta }_{2}$ 和 ${\beta }_{3}$ 的系数是从 $\left\lbrack  {5,{10}}\right\rbrack  .{\beta }_{0}$ 中均匀采样得到的。$\left\lbrack  {5,{10}}\right\rbrack  .{\beta }_{0}$ 是一个具有协方差函数 ${\sum }_{{t}_{1},{t}_{2}} = {\left| {t}_{1} - {t}_{2}\right| }^{-1}$ 和 ${\sum }_{{t}_{1}} = {\sum }_{{t}_{2}} = 1$ 的高斯过程，其中 ${t}_{1}$ 和 ${t}_{2}$ 表示两个任意时间戳。与指数衰减协方差函数（Yu等人，2019年）不同，这种多项式衰减协方差函数具有长程依赖性。每个时间序列的起始时间 ${t}_{0}$ 是从 [0, 719] 中均匀采样得到的。我们首先生成60个长度为14400的时间序列，然后将每个时间序列分割成宽度为1440、步长为24的滑动窗口。在我们的实验中，我们使用历史720个时间点来预测未来720个时间点。由于合成时间序列的确定性部分和随机部分都具有长程相关性，因此模型应能很好地捕捉这些依赖关系，以便对接下来的720个时间点进行准确预测。结果总结在表6中。在这里，我们考虑Pyraformer的两种不同配置：1）金字塔图中所有尺度的 $C = 6$（表示为Pyraformer ${}_{6,6,6}$）；2）从下到上三层依次为 $C = {12},7$ 和4（表示为Pyraformer ${}_{{12},7,4}$）。

<!-- Media -->

Table 6: Long-range forecasting results on the synthetic dataset.

表6：合成数据集上的长期预测结果。

<table><tr><td>$\mathbf{{Method}}$</td><td>$\mathbf{{MSE}}$</td><td>$\mathbf{{MAE}}$</td></tr><tr><td>Full attention</td><td>3.550</td><td>1.477</td></tr><tr><td>LogTrans</td><td>3.007</td><td>1.366</td></tr><tr><td>ETC</td><td>4.742</td><td>5.509</td></tr><tr><td>Informer</td><td>7.546</td><td>2.092</td></tr><tr><td>Longformer</td><td>2.032</td><td>1.116</td></tr><tr><td>Reformer</td><td>1.538</td><td>3.069</td></tr><tr><td>Pyraformer ${}_{6,6,6}$</td><td>1.258</td><td>0.877</td></tr><tr><td>Pyraformer ${}_{{12},7,4}$</td><td>1.176</td><td>0.849</td></tr></table>

<table><tbody><tr><td>$\mathbf{{Method}}$</td><td>$\mathbf{{MSE}}$</td><td>$\mathbf{{MAE}}$</td></tr><tr><td>全注意力</td><td>3.550</td><td>1.477</td></tr><tr><td>日志转换器（LogTrans）</td><td>3.007</td><td>1.366</td></tr><tr><td>电子不停车收费系统（ETC）</td><td>4.742</td><td>5.509</td></tr><tr><td>信息者（Informer）</td><td>7.546</td><td>2.092</td></tr><tr><td>长序列转换器（Longformer）</td><td>2.032</td><td>1.116</td></tr><tr><td>改革者（Reformer）</td><td>1.538</td><td>3.069</td></tr><tr><td>金字塔变换器（Pyraformer） ${}_{6,6,6}$</td><td>1.258</td><td>0.877</td></tr><tr><td>金字塔变换器（Pyraformer） ${}_{{12},7,4}$</td><td>1.176</td><td>0.849</td></tr></tbody></table>

<!-- Media -->

It can be observed that Pyraformer ${}_{6,6,6}$ with the same $C$ for all scales already outperforms the benchmark methods by a large margin. In particular, the MSE given by Pyraformer is decreased by ${18.2}\%$ compared with Reformer,which produces the smallest MSE among the existing variants of Transformer. On the other hand,by exploiting the information of the known period,Pyraformer ${}_{{12},7,4}$ performs even better than Pyraformer ${}_{6,6,6}$ . Note that in Pyraformer ${}_{{12},7,4}$ ,nodes at scale 2,3,and 4 characterizes coarser temporal resolutions respectively corresponding to half a day, half a week, and half a month. We also tested Pyraformer ${}_{{24},7,4}$ ,but setting $C = {24}$ in the second scale degrades the performance, probably because the convolution layer with a kernel size of 24 is difficult to train.

可以观察到，在所有尺度上使用相同$C$的金字塔变换器（Pyraformer ${}_{6,6,6}$）已经大幅超越了基准方法。特别是，与在现有Transformer变体中产生最小均方误差（MSE）的Reformer相比，金字塔变换器（Pyraformer）给出的均方误差降低了${18.2}\%$。另一方面，通过利用已知周期的信息，金字塔变换器（Pyraformer ${}_{{12},7,4}$）的表现甚至比金字塔变换器（Pyraformer ${}_{6,6,6}$）更好。请注意，在金字塔变换器（Pyraformer ${}_{{12},7,4}$）中，尺度2、3和4的节点分别表征了更粗的时间分辨率，对应于半天、半周和半月。我们还测试了金字塔变换器（Pyraformer ${}_{{24},7,4}$），但在第二个尺度上设置$C = {24}$会降低性能，可能是因为核大小为24的卷积层难以训练。

We further visualized the forecasting results produced by Pyraformer ${}_{{12},7,4}$ in Figure 6 The blue solid curve and red dashed curve denote the true and predicted time series respectively. By capturing the temporal dependencies with different ranges, the prediction resulting from Pyraformer closely follows the ground truth.

我们进一步在图6中可视化了Pyraformer ${}_{{12},7,4}$产生的预测结果。蓝色实线曲线和红色虚线曲线分别表示真实时间序列和预测时间序列。通过捕捉不同范围的时间依赖关系，Pyraformer的预测结果与真实值非常接近。

On the other hand, to check whether Pyraformer can extract features with different temporal resolutions, we depicted the extracted features in a randomly selected channel across time at each scale in the pyramidal graph in Figure 7. It is apparent that the features at the coarser scales can be regarded as a lower resolution version of the features at the finer scales.

另一方面，为了检验Pyraformer是否能够提取不同时间分辨率的特征，我们在图7的金字塔图中描绘了在每个尺度上随机选择的一个通道随时间提取的特征。显然，较粗尺度上的特征可以看作是较细尺度上特征的低分辨率版本。

## J ABLATION STUDY

## J 消融研究

### J.1 IMPACT OF $A$ AND $C$

### J.1 $A$和$C$的影响

We studied the impact of $A$ and $C$ on the performance of Pyraformer for long-range time series forecasting, and showed the results in Table 7. Here, we focus on the dataset ETTh1. The history length is 336 and the prediction length is 720 . From Table 7, we can conclude that the receptive fields of the nodes at the coarsest scale in the PAM play an indispensable role in reducing the prediction error of Pyraformer. For instance,there are 42 nodes at the coarsest scale when $C = 2$ . Without the intra-scale connections, each node can only receive messages from 16 nodes at the finest scale. As the number of adjacent connections $A$ in each scale increases,the receptive fields of the coarsest-scale nodes also extend, and therefore, the prediction error decreases accordingly. However, as long as the nodes at the top scale have a global receptive field,further increasing $A$ will not bring large gains. For $C = 5$ ,the performance does not improve even though $A$ increases. Such observations indicate that it is better to set $A$ to be small once the uppermost nodes in the PAM have a global receptive field. In practice,we only increase $C$ with the increase of $L$ ,but keep $A$ small.

我们研究了$A$和$C$对Pyraformer在长序列时间预测性能上的影响，并将结果展示在表7中。这里，我们聚焦于数据集ETTh1。历史长度为336，预测长度为720。从表7中，我们可以得出结论：PAM（金字塔注意力模块，Pyramid Attention Module）中最粗粒度尺度下节点的感受野在降低Pyraformer的预测误差方面起着不可或缺的作用。例如，当$C = 2$时，最粗粒度尺度下有42个节点。如果没有尺度内连接，每个节点只能从最细粒度尺度下的16个节点接收信息。随着每个尺度内相邻连接数量$A$的增加，最粗粒度尺度节点的感受野也会扩大，因此预测误差也会相应降低。然而，只要最顶层尺度的节点具有全局感受野，进一步增加$A$不会带来显著收益。对于$C = 5$，即使$A$增加，性能也不会提升。这些观察结果表明，一旦PAM中最顶层的节点具有全局感受野，最好将$A$设置得小一些。在实践中，我们仅随着$L$的增加而增加$C$，但保持$A$较小。

<!-- Media -->

<!-- figureText: Ground truth Predicted values -->

<img src="https://cdn.noedgeai.com/01957b4b-1654-7ad6-8e71-2da613767aa1_17.jpg?x=325&y=227&w=1146&h=611&r=0"/>

Figure 6: Visualization of prediction results on the synthetic dataset.

图6：合成数据集上预测结果的可视化。

<!-- figureText: (b) -->

<img src="https://cdn.noedgeai.com/01957b4b-1654-7ad6-8e71-2da613767aa1_17.jpg?x=323&y=959&w=1152&h=328&r=0"/>

Figure 7: Visualization of the extracted features across time in second channel at different scales: (a) scale 1; (b) scale 2; (c) scale 3.

图7：不同尺度下第二通道随时间提取的特征的可视化：(a) 尺度1；(b) 尺度2；(c) 尺度3。

<!-- Media -->

### J.2 IMPACT OF THE CSCM ARCHITECTURE

### J.2 CSCM架构的影响

In addition to convolution,there exist other mechanisms for constructing the $C$ -ary tree,such as max pooling and average pooling. We studied the impact of different CSCM architectures on the performance for long-range forecasting on dataset ETTh1. The history and the prediction length are both 168 and $C = 4$ for all mechanisms. The results are listed in Table 8 . From Table 8,we can tell that: (1) Using pooling layers instead of convolution typically degrades the performance. However, the performance of Pyraformer based on max pooling is still superior to that of Informer, demonstrating the advantages of the PAM over the prob-sparse attention in Informer. (2) The MSE of convolution with the bottleneck is only ${1.51}\%$ larger than that without bottleneck,but the number of parameters is reduced by almost ${90}\%$ . Thus,we adopt the more compact module of convolution with bottleneck as our CSCM.

除卷积外，还存在其他构建$C$元树的机制，如最大池化和平均池化。我们研究了不同的CSCM架构对ETTh1数据集上长期预测性能的影响。所有机制的历史长度和预测长度均为168和$C = 4$。结果列于表8。从表8中我们可以看出：（1）使用池化层而非卷积通常会降低性能。然而，基于最大池化的Pyraformer的性能仍优于Informer，这表明PAM相对于Informer中的概率稀疏注意力具有优势。（2）带瓶颈的卷积的均方误差（MSE）仅比不带瓶颈的卷积大${1.51}\%$，但参数数量减少了近${90}\%$。因此，我们采用更紧凑的带瓶颈卷积模块作为我们的CSCM。

<!-- Media -->

Table 7: Impact of $A$ and $C$ on long-range forecasting. The history length is 336.

表7：$A$和$C$对长期预测的影响。历史长度为336。

<table><tr><td rowspan="2"/><td colspan="3">$A = 3$</td><td colspan="3">$A = 9$</td><td colspan="3">$A = {13}$</td></tr><tr><td>MSE</td><td>MAE</td><td>Q-K pairs</td><td>MSE</td><td>MAE</td><td>Q-K pairs</td><td>MSE</td><td>MAE</td><td>Q-K pairs</td></tr><tr><td>$C = 2$</td><td>1.035</td><td>0.811</td><td>73512</td><td>1.029</td><td>0.815</td><td>162648</td><td>1.003</td><td>0.807</td><td>221112</td></tr><tr><td>$C = 3$</td><td>1.029</td><td>0.817</td><td>58992</td><td>1.009</td><td>0.798</td><td>128976</td><td>1.056</td><td>0.805</td><td>174672</td></tr><tr><td>$C = 4$</td><td>1.001</td><td>0.802</td><td>53208</td><td>1.028</td><td>0.806</td><td>115848</td><td>1.027</td><td>0.804</td><td>156696</td></tr><tr><td>$C = 5$</td><td>0.999</td><td>0.796</td><td>49992</td><td>1.005</td><td>0.796</td><td>108744</td><td>1.017</td><td>0.797</td><td>147192</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="3">$A = 3$</td><td colspan="3">$A = 9$</td><td colspan="3">$A = {13}$</td></tr><tr><td>均方误差（Mean Squared Error，MSE）</td><td>平均绝对误差（Mean Absolute Error，MAE）</td><td>查询-键对（Query-Key pairs）</td><td>均方误差（Mean Squared Error，MSE）</td><td>平均绝对误差（Mean Absolute Error，MAE）</td><td>查询-键对（Query-Key pairs）</td><td>均方误差（Mean Squared Error，MSE）</td><td>平均绝对误差（Mean Absolute Error，MAE）</td><td>查询-键对（Query-Key pairs）</td></tr><tr><td>$C = 2$</td><td>1.035</td><td>0.811</td><td>73512</td><td>1.029</td><td>0.815</td><td>162648</td><td>1.003</td><td>0.807</td><td>221112</td></tr><tr><td>$C = 3$</td><td>1.029</td><td>0.817</td><td>58992</td><td>1.009</td><td>0.798</td><td>128976</td><td>1.056</td><td>0.805</td><td>174672</td></tr><tr><td>$C = 4$</td><td>1.001</td><td>0.802</td><td>53208</td><td>1.028</td><td>0.806</td><td>115848</td><td>1.027</td><td>0.804</td><td>156696</td></tr><tr><td>$C = 5$</td><td>0.999</td><td>0.796</td><td>49992</td><td>1.005</td><td>0.796</td><td>108744</td><td>1.017</td><td>0.797</td><td>147192</td></tr></tbody></table>

Table 8: Impact of the CSCM architecture on long-range forecasting. Parameters introduced by the normalization layers are relatively few, and thus, are ignored.

表8：CSCM架构对长期预测的影响。归一化层引入的参数相对较少，因此可忽略不计。

<table><tr><td>CSCM</td><td>$\mathbf{{MSE}}$</td><td>$\mathbf{{MAE}}$</td><td>Parameters</td></tr><tr><td>Max-pooling</td><td>0.842</td><td>0.700</td><td>0</td></tr><tr><td>Average-pooling</td><td>0.833</td><td>0.693</td><td>0</td></tr><tr><td>Conv.</td><td>0.796</td><td>0.679</td><td>3147264</td></tr><tr><td>Conv. w/bottleneck</td><td>0.808</td><td>0.683</td><td>328704</td></tr></table>

<table><tbody><tr><td>跨尺度上下文模块（Cross-Scale Context Module，CSCM）</td><td>$\mathbf{{MSE}}$</td><td>$\mathbf{{MAE}}$</td><td>参数</td></tr><tr><td>最大池化</td><td>0.842</td><td>0.700</td><td>0</td></tr><tr><td>平均池化</td><td>0.833</td><td>0.693</td><td>0</td></tr><tr><td>卷积（Convolution）</td><td>0.796</td><td>0.679</td><td>3147264</td></tr><tr><td>带瓶颈结构的卷积</td><td>0.808</td><td>0.683</td><td>328704</td></tr></tbody></table>

Table 9: Impact of history length. The prediction length is 1344.

表9：历史长度的影响。预测长度为1344。

<table><tr><td>History Length</td><td>$\mathbf{{MSE}}$</td><td>$\mathbf{{MAE}}$</td></tr><tr><td>84</td><td>1.234</td><td>0.856</td></tr><tr><td>168</td><td>1.226</td><td>0.868</td></tr><tr><td>336</td><td>1.108</td><td>0.835</td></tr><tr><td>672</td><td>1.057</td><td>0.806</td></tr><tr><td>1344</td><td>1.062</td><td>0.806</td></tr></table>

<table><tbody><tr><td>历史长度</td><td>$\mathbf{{MSE}}$</td><td>$\mathbf{{MAE}}$</td></tr><tr><td>84</td><td>1.234</td><td>0.856</td></tr><tr><td>168</td><td>1.226</td><td>0.868</td></tr><tr><td>336</td><td>1.108</td><td>0.835</td></tr><tr><td>672</td><td>1.057</td><td>0.806</td></tr><tr><td>1344</td><td>1.062</td><td>0.806</td></tr></tbody></table>

Table 10: Impact of the PAM.

表10：位置特异性氨基酸替换矩阵（PAM）的影响。

<table><tr><td>$\mathbf{{Method}}$</td><td>$\mathbf{{Metrics}}$</td><td>96</td><td>288</td><td>672</td></tr><tr><td rowspan="2">CSCM Only</td><td>MSE</td><td>0.576</td><td>0.782</td><td>0.883</td></tr><tr><td>MAE</td><td>0.544</td><td>0.683</td><td>0.752</td></tr><tr><td rowspan="2">Pyraformer</td><td>MSE</td><td>0.480</td><td>0.754</td><td>0.857</td></tr><tr><td>MAE</td><td>0.486</td><td>0.659</td><td>0.707</td></tr></table>

<table><tbody><tr><td>$\mathbf{{Method}}$</td><td>$\mathbf{{Metrics}}$</td><td>96</td><td>288</td><td>672</td></tr><tr><td rowspan="2">仅适用于供应链协同管理（CSCM）</td><td>均方误差（MSE）</td><td>0.576</td><td>0.782</td><td>0.883</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.544</td><td>0.683</td><td>0.752</td></tr><tr><td rowspan="2">金字塔变换器（Pyraformer）</td><td>均方误差（MSE）</td><td>0.480</td><td>0.754</td><td>0.857</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.486</td><td>0.659</td><td>0.707</td></tr></tbody></table>

<!-- Media -->

### J.3 IMPACT OF THE HISTORY LENGTH

### J.3 历史长度的影响

We also checked the influence of the history length on the prediction accuracy. The dataset is ETTm1, since its granularity is minute and contains more long-range dependencies. We fixed the prediction length to 1344 and changed the history length from 84 to 1344 in Table 9. As expected, a longer history typically improves prediction accuracy. On the other hand, this performance gain starts to level off when introducing more history stops providing new information. As shown in Figure 8, the time series with length 672 contains almost all periodicity information that is essential for prediction, while length 1344 introduces more noise.

我们还检验了历史长度对预测准确性的影响。数据集采用ETTm1，因为其粒度为分钟级，且包含更多长距离依赖关系。在表9中，我们将预测长度固定为1344，并将历史长度从84变化到1344。正如预期的那样，更长的历史通常会提高预测准确性。另一方面，当引入更多历史数据不再能提供新信息时，这种性能提升开始趋于平稳。如图8所示，长度为672的时间序列几乎包含了预测所需的所有周期性信息，而长度为1344的时间序列则引入了更多噪声。

### J.4 IMPACT OF THE PAM

### J.4 金字塔注意力模块（PAM）的影响

Finally, we investigated the importance of the PAM. We compared the performance of Pyraformer with and without the PAM on the dataset ETTm1. For a fair comparison, the number of parameters of the two methods were controlled to be within the same order of magnitude. More precisely, we increased the bottleneck dimension of "Conv. w/bottleneck" for the model only with the CSCM. The results are shown in Table 10. Obviously, the PAM is vital to yield accurate predictions.

最后，我们研究了金字塔注意力模块（PAM）的重要性。我们在数据集ETTm1上比较了有无金字塔注意力模块（PAM）的Pyraformer模型的性能。为了进行公平比较，我们将两种方法的参数数量控制在同一数量级内。更确切地说，我们仅针对仅使用卷积自相关模块（CSCM）的模型增加了“带瓶颈的卷积（Conv. w/bottleneck）”的瓶颈维度。结果如表10所示。显然，金字塔注意力模块（PAM）对于产生准确的预测至关重要。

## K DISCUSSION ON THE SELECTION OF HYPER-PARAMETERS

## K 超参数选择的讨论

We recommend to first determine the number of attention layers $N$ based on the available computing resources,as this number is directly related to the model size. Next,the number of scales $S$ can be determined by the granularity of the time series. For example, for hourly observations, we typically assume that it may also have daily,weekly and monthly periods. Therefore,we can set $S$ to be 4 . We then focus on the selection of $A$ and $C$ . According to the ablation study,we typically prefer a small $A$ ,such as 3 and 5 . Lastly,in order to ensure the network has a receptive field of $L$ ,we can select a $C$ that satisfies Equation (5). In practice,we can use a validation set to choose $C$ from its candidates that satisfies (5). It is also worthwhile to check whether choosing different $C$ for different scales based on the granularity of the time series can further improve the performance as we did in Appendix I

我们建议首先根据可用的计算资源确定注意力层的数量$N$，因为该数量与模型大小直接相关。接下来，可以根据时间序列的粒度确定尺度的数量$S$。例如，对于每小时的观测数据，我们通常假设它可能还具有每日、每周和每月的周期。因此，我们可以将$S$设置为 4。然后，我们重点关注$A$和$C$的选择。根据消融实验，我们通常倾向于选择较小的$A$，例如 3 和 5。最后，为了确保网络具有$L$的感受野，我们可以选择一个满足方程 (5) 的$C$。在实践中，我们可以使用验证集从满足 (5) 的候选值中选择$C$。同样值得检查的是，根据时间序列的粒度为不同尺度选择不同的$C$是否能像我们在附录 I 中所做的那样进一步提高性能

<!-- Media -->

<!-- figureText: 27.5 25.0 22.5 20.0 17.5 10.0 1000 1200 (d) 25.0 22.5 27.5 25.0 22.5 15.0 12.5 -->

<img src="https://cdn.noedgeai.com/01957b4b-1654-7ad6-8e71-2da613767aa1_19.jpg?x=313&y=238&w=1171&h=937&r=0"/>

Figure 8: Time series with different lengths in the ETTm1 dataset. The sequence length in (a) and (b) is 672, and that in (c) and (d) is 1344. The time series in (a) and (b) corresponds to the latter half of those in (c) and (d) respectively.

图8：ETTm1数据集中不同长度的时间序列。(a)和(b)中的序列长度为672，(c)和(d)中的序列长度为1344。(a)和(b)中的时间序列分别对应于(c)和(d)中时间序列的后半部分。

<!-- Media -->