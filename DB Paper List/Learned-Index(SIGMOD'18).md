# The Case for Learned Index Structures

# 关于学习型索引结构的论证

Tim Kraska*

蒂姆·克拉斯卡（Tim Kraska）*

MIT

kraska@mit.edu

Alex Beutel

亚历克斯·比特尔（Alex Beutel）

Google, Inc.

谷歌公司

abeutel@google.com

Ed H. Chi

埃德·H·池（Ed H. Chi）

Google, Inc.

谷歌公司

edchi@google.com

Jeffrey Dean

杰弗里·迪恩（Jeffrey Dean）

Google, Inc.

谷歌公司

jeff@google.com

Neoklis Polyzotis

尼奥克利斯·波利佐蒂斯（Neoklis Polyzotis）

Google, Inc.

谷歌公司

npoly@google.com

## Abstract

## 摘要

Indexes are models: a B-Tree-Index can be seen as a model to map a key to the position of a record within a sorted array, a Hash-Index as a model to map a key to a position of a record within an unsorted array, and a BitMap-Index as a model to indicate if a data record exists or not. In this exploratory research paper, we start from this premise and posit that all existing index structures can be replaced with other types of models, including deep-learning models, which we term learned indexes. We theoretically analyze under which conditions learned indexes outperform traditional index structures and describe the main challenges in designing learned index structures. Our initial results show that our learned indexes can have significant advantages over traditional indexes. More importantly, we believe that the idea of replacing core components of a data management system through learned models has far reaching implications for future systems designs and that this work provides just a glimpse of what might be possible.

索引即模型：B树索引（B-Tree-Index）可视为一种将键映射到排序数组中记录位置的模型，哈希索引（Hash-Index）可视为一种将键映射到未排序数组中记录位置的模型，位图索引（BitMap-Index）可视为一种指示数据记录是否存在的模型。在这篇探索性研究论文中，我们从这一前提出发，假定所有现有的索引结构都可以被其他类型的模型所取代，包括深度学习模型，我们将其称为学习型索引。我们从理论上分析了学习型索引在哪些条件下优于传统索引结构，并描述了设计学习型索引结构的主要挑战。我们的初步结果表明，我们的学习型索引相较于传统索引具有显著优势。更重要的是，我们认为通过学习型模型取代数据管理系统核心组件的想法对未来系统设计具有深远影响，而这项工作仅仅展示了可能实现的一小部分。

## ACM Reference Format:

## ACM引用格式：

Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean, and Neoklis Polyzotis. 2018. The Case for Learned Index Structures. In SIGMOD'18: 2018 International Conference on Management of Data, June 10-15, 2018, Houston, TX, USA. , 16 pages. https://doi.org/10.1145/3183713.3196909

蒂姆·克拉斯卡（Tim Kraska）、亚历克斯·比特尔（Alex Beutel）、埃德·H·池（Ed H. Chi）、杰弗里·迪恩（Jeffrey Dean）和尼奥克利斯·波利佐蒂斯（Neoklis Polyzotis）。2018年。关于学习型索引结构的论证。收录于SIGMOD'18：2018年国际数据管理会议，2018年6月10 - 15日，美国得克萨斯州休斯顿。共16页。https://doi.org/10.1145/3183713.3196909

## 1 INTRODUCTION

## 1 引言

Whenever efficient data access is needed, index structures are the answer, and a wide variety of choices exist to address the different needs of various access patterns. For example, B-Trees are the best choice for range requests (e.g., retrieve all records in a certain time frame); Hash-maps are hard to beat in performance for single key look-ups; and Bloom filters are typically used to check for record existence. Because of their importance for database systems and many other applications, indexes have been extensively tuned over the past decades to be more memory,cache and/or CPU efficient [11, 29, 36, 59].

每当需要高效的数据访问时，索引结构就是解决方案，并且存在多种选择来满足各种访问模式的不同需求。例如，B树（B-Trees）是范围查询（例如，检索特定时间范围内的所有记录）的最佳选择；哈希映射（Hash-maps）在单键查找性能方面很难被超越；布隆过滤器（Bloom filters）通常用于检查记录是否存在。由于索引对数据库系统和许多其他应用程序至关重要，在过去几十年中，人们对其进行了广泛的优化，以提高内存、缓存和/或CPU的使用效率 [11, 29, 36, 59]。

Yet, all of those indexes remain general purpose data structures; they assume nothing about the data distribution and do not take advantage of more common patterns prevalent in real world data. For example, if the goal is to build a highly-tuned system to store and query ranges of fixed-length records over

然而，所有这些索引仍然是通用的数据结构；它们对数据分布不做任何假设，也没有利用现实世界数据中普遍存在的更常见模式。例如，如果目标是构建一个高度优化的系统，用于存储和查询固定长度记录在

COND

This work is licensed under a Creative Commons

本作品采用知识共享

Attribution-NonCommercial-ShareAlike International 4.0 License.

署名 - 非商业性使用 - 相同方式共享 4.0 国际许可协议进行许可。

© 2018 Copyright held by the owner/author(s).

© 2018 版权归所有者/作者所有。

ACM ISBN 978-1-4503-4703-7/18/06. a set of continuous integer keys (e.g., the keys 1 to 100M), one would not use a conventional B-Tree index over the keys since the key itself can be used as an offset,making it an $O\left( 1\right)$ rather than $O\left( {\log n}\right)$ operation to look-up any key or the beginning of a range of keys. Similarly, the index memory size would be reduced from $O\left( n\right)$ to $O\left( 1\right)$ . Maybe surprisingly,similar optimizations are possible for other data patterns. In other words, knowing the exact data distribution enables highly optimizing almost any index structure.

ACM 国际标准书号 978 - 1 - 4503 - 4703 - 7/18/06。一组连续的整数键（例如，键 1 到 1 亿），人们不会对这些键使用传统的 B 树索引，因为键本身可以用作偏移量，这使得查找任何键或键范围的起始位置成为一个$O\left( 1\right)$操作，而不是$O\left( {\log n}\right)$操作。同样，索引内存大小将从$O\left( n\right)$减少到$O\left( 1\right)$。也许令人惊讶的是，其他数据模式也可以进行类似的优化。换句话说，了解确切的数据分布可以对几乎任何索引结构进行高度优化。

Of course, in most real-world use cases the data do not perfectly follow a known pattern and the engineering effort to build specialized solutions for every use case is usually too high. However, we argue that machine learning (ML) opens up the opportunity to learn a model that reflects the patterns in the data and thus to enable the automatic synthesis of specialized index structures, termed learned indexes, with low engineering cost.

当然，在大多数现实世界的用例中，数据并不完全遵循已知的模式，并且为每个用例构建专门解决方案的工程工作量通常过高。然而，我们认为机器学习（ML）提供了学习一个反映数据模式的模型的机会，从而能够以较低的工程成本自动合成专门的索引结构，即学习型索引。

In this paper, we explore the extent to which learned models, including neural networks, can be used to enhance, or even replace, traditional index structures from B-Trees to Bloom filters. This may seem counterintuitive because ML cannot provide the semantic guarantees we traditionally associate with these indexes, and because the most powerful ML models, neural networks, are traditionally thought of as being very compute expensive. Yet, we argue that none of these apparent obstacles are as problematic as they might seem. Instead, our proposal to use learned models has the potential for significant benefits, especially on the next generation of hardware.

在本文中，我们探讨了包括神经网络在内的学习模型在多大程度上可以用于增强甚至取代从 B 树到布隆过滤器等传统索引结构。这可能看起来有悖直觉，因为机器学习无法提供我们传统上与这些索引相关联的语义保证，而且最强大的机器学习模型，即神经网络，传统上被认为计算成本非常高。然而，我们认为这些明显的障碍都没有看起来那么成问题。相反，我们使用学习模型的提议有可能带来显著的好处，特别是在下一代硬件上。

In terms of semantic guarantees, indexes are already to a large extent learned models making it surprisingly straightforward to replace them with other types of ML models. For example, a B-Tree can be considered as a model which takes a key as an input and predicts the position of a data record in a sorted set (the data has to be sorted to enable efficient range requests). A Bloom filter is a binary classifier, which based on a key predicts if a key exists in a set or not. Obviously, there exists subtle but important differences. For example, a Bloom filter can have false positives but not false negatives. However, as we will show in this paper, it is possible to address these differences through novel learning techniques and/or simple auxiliary data structures.

就语义保证而言，索引在很大程度上已经是学习模型，因此用其他类型的机器学习模型取代它们出奇地容易。例如，B 树可以被视为一个模型，它将键作为输入，并预测数据记录在有序集合中的位置（数据必须排序才能实现高效的范围查询）。布隆过滤器是一个二元分类器，它根据键预测一个键是否存在于一个集合中。显然，存在一些细微但重要的差异。例如，布隆过滤器可能会产生误报，但不会产生漏报。然而，正如我们将在本文中展示的，通过新颖的学习技术和/或简单的辅助数据结构可以解决这些差异。

In terms of performance, we observe that every CPU already has powerful SIMD capabilities and we speculate that many laptops and mobile phones will soon have a Graphics Processing Unit (GPU) or Tensor Processing Unit (TPU). It is also reasonable to speculate that CPU-SIMD/GPU/TPUs will be increasingly powerful as it is much easier to scale the restricted set of (parallel) math operations used by neural nets than a general purpose instruction set. As a result the high cost to execute a neural net or other ML models might actually be negligible in the future. For instance, both Nvidia and Google's TPUs are already able to perform thousands if not tens of thousands of neural net operations in a single cycle [3]. Furthermore,it was stated that GPUs will improve ${1000} \times$ in performance by 2025, whereas Moore's law for CPUs is essentially dead [5]. By replacing branch-heavy index structures with neural networks, databases and other systems can benefit from these hardware trends. While we see the future of learned index structures on specialized hardware, like TPUs, this paper focuses entirely on CPUs and surprisingly shows that we can achieve significant advantages even in this case.

在性能方面，我们注意到每个 CPU 已经具备强大的单指令多数据（SIMD）能力，并且我们推测许多笔记本电脑和手机很快将配备图形处理单元（GPU）或张量处理单元（TPU）。同样合理的推测是，CPU - SIMD/GPU/TPU 将变得越来越强大，因为扩展神经网络使用的受限（并行）数学运算集比扩展通用指令集要容易得多。因此，执行神经网络或其他机器学习模型的高成本在未来实际上可能可以忽略不计。例如，英伟达和谷歌的 TPU 已经能够在单个周期内执行数千甚至数万个神经网络操作[3]。此外，有报告称到 2025 年 GPU 的性能将提高${1000} \times$，而 CPU 的摩尔定律实际上已经失效[5]。通过用神经网络取代分支密集的索引结构，数据库和其他系统可以从这些硬件发展趋势中受益。虽然我们看到学习型索引结构的未来在于专门的硬件，如 TPU，但本文完全聚焦于 CPU，并且令人惊讶地表明，即使在这种情况下我们也能获得显著的优势。

---

<!-- Footnote -->

*Work done while the author was affiliated with Google.

*本文作者在谷歌任职期间完成的工作。

<!-- Footnote -->

---

It is important to note that we do not argue to completely replace traditional index structures with learned indexes. Rather, the main contribution of this paper is to outline and evaluate the potential of a novel approach to build indexes, which complements existing work and, arguably, opens up an entirely new research direction for a decades-old field. This is based on the key observation that many data structures can be decomposed into a learned model and an auxiliary structure to provide the same semantic guarantees. The potential power of this approach comes from the fact that continuous functions, describing the data distribution, can be used to build more efficient data structures or algorithms. We empirically get very promising results when evaluating our approach on synthetic and real-world datasets for read-only analytical workloads. However, many open challenges still remain, such as how to handle write-heavy workloads, and we outline many possible directions for future work. Furthermore, we believe that we can use the same principle to replace other components and operations commonly used in (database) systems. If successful, the core idea of deeply embedding learned models into algorithms and data structures could lead to a radical departure from the way systems are currently developed.

需要注意的是，我们并非主张用学习型索引完全取代传统索引结构。相反，本文的主要贡献在于概述并评估一种构建索引的新方法的潜力，该方法对现有工作起到补充作用，并且可以说为这个已有数十年历史的领域开辟了全新的研究方向。这一观点基于一个关键观察结果，即许多数据结构可以分解为一个学习模型和一个辅助结构，以提供相同的语义保证。这种方法的潜在力量源于这样一个事实：描述数据分布的连续函数可用于构建更高效的数据结构或算法。当我们在合成数据集和真实世界数据集上针对只读分析工作负载评估我们的方法时，取得了非常有前景的结果。然而，仍然存在许多开放性挑战，例如如何处理写操作繁重的工作负载，我们还概述了未来工作的许多可能方向。此外，我们相信可以使用相同的原理来替换（数据库）系统中常用的其他组件和操作。如果成功，将学习模型深度嵌入算法和数据结构的核心思想可能会使系统的开发方式发生根本性的改变。

The remainder of this paper is outlined as follows: In the next two sections we introduce the general idea of learned indexes using B-Trees as an example. In Section 4 we extend this idea to Hash-maps and in Section 5 to Bloom filters. All sections contain a separate evaluation. Finally in Section 6 we discuss related work and conclude in Section 7.

本文的其余部分概述如下：在接下来的两节中，我们以B树为例介绍学习型索引的一般概念。在第4节中，我们将这一概念扩展到哈希映射，在第5节中扩展到布隆过滤器。所有章节都包含单独的评估。最后，在第6节中我们讨论相关工作，并在第7节中进行总结。

## 2 RANGE INDEX

## 2 范围索引

Range index structure, like B-Trees, are already models: given a key, they "predict" the location of a value within a key-sorted set. To see this, consider a B-Tree index in an analytics in-memory database (i.e., read-only) over the sorted primary key column as shown in Figure 1(a). In this case, the B-Tree provides a mapping from a look-up key to a position inside the sorted array of records with the guarantee that the key of the record at that position is the first key equal or higher than the look-up key. The data has to be sorted to allow for efficient range requests. This same general concept also applies to secondary indexes where the data would be the list of <key, record_pointer> pairs with the key being the indexed value and the pointer a reference to the record. ${}^{1}$

像B树这样的范围索引结构本身就是模型：给定一个键，它们“预测”该键在键排序集合中对应值的位置。为了说明这一点，考虑一个分析型内存数据库（即只读）中基于排序主键列的B树索引，如图1(a)所示。在这种情况下，B树提供了从查找键到排序记录数组中某个位置的映射，并保证该位置记录的键是第一个等于或大于查找键的键。数据必须进行排序，以便能够高效处理范围查询请求。同样的一般概念也适用于二级索引，其中数据是<键, 记录指针>对的列表，键是被索引的值，指针是对记录的引用。${}^{1}$

<!-- Media -->

<!-- figureText: (a) B-Tree Index (b) Learned Index Key Model (e.g., NN) pos pos - min_err pos + max_en Key BTree pos pos - 0 pos + pagezise -->

<img src="https://cdn.noedgeai.com/0195c8fc-f853-7b31-b5ff-dad98e8dad50_1.jpg?x=936&y=234&w=692&h=320&r=0"/>

Figure 1: Why B-Trees are models

图1：为什么B树是模型

<!-- Media -->

For efficiency reasons it is common not to index every single key of the sorted records, rather only the key of every n-th record, i.e., the first key of a page. Here we only assume fixed-length records and logical paging over a continuous memory region, i.e., a single array, not physical pages which are located in different memory regions (physical pages and variable length records are discussed in Appendix D.2). Indexing only the first key of every page helps to significantly reduce the number of keys the index has to store without any significant performance penalty. Thus, the B-Tree is a model, or in ML terminology, a regression tree: it maps a key to a position with a min- and max-error (a min-error of 0 and a max-error of the page-size), with a guarantee that the key can be found in that region if it exists. Consequently, we can replace the index with other types of ML models, including neural nets, as long as they are also able to provide similar strong guarantees about the min- and max-error.

出于效率考虑，通常不会对排序记录中的每个键都进行索引，而是仅对每第n条记录的键进行索引，即页面的第一个键。这里我们仅假设记录长度固定，并且在连续内存区域（即单个数组）上进行逻辑分页，而不是位于不同内存区域的物理页面（物理页面和可变长度记录在附录D.2中讨论）。仅对每个页面的第一个键进行索引有助于显著减少索引必须存储的键的数量，而不会对性能造成显著影响。因此，B树是一个模型，或者用机器学习术语来说，是一个回归树：它将一个键映射到一个具有最小和最大误差（最小误差为0，最大误差为页面大小）的位置，并保证如果该键存在，则可以在该区域中找到它。因此，我们可以用其他类型的机器学习模型（包括神经网络）来替换该索引，只要它们也能够提供关于最小和最大误差的类似强保证。

At first sight it may seem hard to provide the same guarantees with other types of ML models, but it is actually surprisingly simple. First, the B-Tree only provides the strong min- and max-error guarantee over the stored keys, not for all possible keys. For new data, B-Trees need to be re-balanced, or in machine learning terminology re-trained, to still be able to provide the same error guarantees. That is, for monotonic models the only thing we need to do is to execute the model for every key and remember the worst over- and under-prediction of a position to calculate the min- and max-error. ${}^{2}$ Second, second,and more importantly, the strong error bounds are not even needed. The data has to be sorted anyway to support range requests, so any error is easily corrected by a local search around the prediction (e.g., using exponential search) and thus, even allows for non-monotonic models. Consequently, we are able to replace B-Trees with any other type of regression model, including linear regression or neural nets (see Figure 1(b)).

乍一看，用其他类型的机器学习模型提供相同的保证似乎很困难，但实际上却出奇地简单。首先，B树仅对存储的键提供强最小和最大误差保证，而不是对所有可能的键都提供。对于新数据，B树需要进行重新平衡，或者用机器学习术语来说，需要重新训练，才能继续提供相同的误差保证。也就是说，对于单调模型，我们唯一需要做的就是对每个键执行模型，并记住位置的最大高估和低估，以计算最小和最大误差。${}^{2}$ 其次，更重要的是，甚至不需要强误差边界。无论如何，数据都必须进行排序以支持范围查询请求，因此任何误差都可以通过在预测位置附近进行局部搜索（例如，使用指数搜索）轻松纠正，因此，甚至允许使用非单调模型。因此，我们能够用任何其他类型的回归模型（包括线性回归或神经网络）来替换B树（见图1(b)）。

Now, there are other technical challenges that we need to address before we can replace B-Trees with learned indexes. For instance, B-Trees have a bounded cost for inserts and look-ups and are particularly good at taking advantage of the cache. Also, B-Trees can map keys to pages which are not continuously mapped to memory or disk. All of these are interesting challenges/research questions and are explained in more detail, together with potential solutions, throughout this section and in the appendix.

现在，在我们能够用学习型索引取代B树（B-Trees）之前，还有其他技术挑战需要解决。例如，B树的插入和查找操作成本是有界的，并且特别擅长利用缓存。此外，B树可以将键映射到并非连续映射到内存或磁盘的页面。所有这些都是有趣的挑战/研究问题，在本节以及附录中会更详细地解释这些问题，并给出潜在的解决方案。

---

<!-- Footnote -->

${}^{1}$ Note,that against some definitions for secondary indexes we do not consider the <key , record_pointer> pairs as part of the index; rather for secondary index the data are the <key , record_pointer> pairs. This is similar to how indexes are implemented in key value stores $\left\lbrack  {{12},{21}}\right\rbrack$ or how B-Trees on modern hardware are designed [44].

${}^{1}$ 请注意，与某些二级索引的定义不同，我们不将<键, 记录指针>对视为索引的一部分；相反，对于二级索引而言，数据就是<键, 记录指针>对。这与键值存储中索引的实现方式$\left\lbrack  {{12},{21}}\right\rbrack$或现代硬件上B树的设计方式类似[44]。

${}^{2}$ The model has to be monotonic to also guarantee the min- and max-error for look-up keys, which do not exist in the stored set.

${}^{2}$ 该模型必须是单调的，以确保对于存储集合中不存在的查找键也能保证最小和最大误差。

<!-- Footnote -->

---

At the same time, using other types of models as indexes can provide tremendous benefits. Most importantly, it has the potential to transform the $\log n$ cost of a B-Tree lookup into a constant operation. For example, assume a dataset with $1\mathrm{M}$ unique keys with a value from $1\mathrm{M}$ and $2\mathrm{M}$ (so the value 1,000,009 is stored at position 10). In this case, a simple linear model, which consists of a single multiplication and addition, can perfectly predict the position of any key for a point look-up or range scan, whereas a B-Tree would require $\log n$ operations. The beauty of machine learning,especially neural nets, is that they are able to learn a wide variety of data distributions, mixtures and other data peculiarities and patterns. The challenge is to balance the complexity of the model with its accuracy.

同时，使用其他类型的模型作为索引可以带来巨大的好处。最重要的是，它有可能将B树查找的$\log n$成本转化为常量操作。例如，假设有一个数据集，包含$1\mathrm{M}$个唯一键，键值范围在$1\mathrm{M}$和$2\mathrm{M}$之间（因此值1,000,009存储在位置10）。在这种情况下，一个简单的线性模型（由一次乘法和一次加法组成）可以完美地预测任何键在点查找或范围扫描中的位置，而B树则需要$\log n$次操作。机器学习，尤其是神经网络的美妙之处在于，它们能够学习各种各样的数据分布、混合情况以及其他数据特性和模式。挑战在于平衡模型的复杂度和准确性。

For most of the discussion in this paper, we keep the simplified assumptions of this section: we only index an in-memory dense array that is sorted by key. This may seem restrictive, but many modern hardware optimized B-Trees, e.g., FAST [44], make exactly the same assumptions, and these indexes are quite common for in-memory database systems for their superior performance $\left\lbrack  {{44},{48}}\right\rbrack$ over scanning or binary search. However, while some of our techniques translate well to some scenarios (e.g., disk-resident data with very large blocks, for example, as used in Bigtable [23]), for other scenarios (fine grained paging, insert-heavy workloads, etc.) more research is needed. In Appendix D. 2 we discuss some of those challenges and potential solutions in more detail.

在本文的大部分讨论中，我们保留本节的简化假设：我们只对按键排序的内存中密集数组进行索引。这可能看起来有局限性，但许多现代硬件优化的B树，例如FAST [44]，也做出了完全相同的假设，并且由于这些索引在扫描或二分查找方面具有优越的性能$\left\lbrack  {{44},{48}}\right\rbrack$，它们在内存数据库系统中相当常见。然而，虽然我们的一些技术在某些场景（例如，使用非常大的数据块的磁盘驻留数据，如Bigtable [23]中使用的那样）中效果很好，但对于其他场景（细粒度分页、插入密集型工作负载等），还需要更多的研究。在附录D. 2中，我们将更详细地讨论其中一些挑战和潜在的解决方案。

### 2.1 What Model Complexity Can We Afford?

### 2.1 我们能承受多大的模型复杂度？

To better understand the model complexity, it is important to know how many operations can be performed in the same amount of time it takes to traverse a B-Tree, and what precision the model needs to achieve to be more efficient than a B-Tree.

为了更好地理解模型复杂度，了解在遍历一棵B树所需的相同时间内可以执行多少次操作，以及模型需要达到何种精度才能比B树更高效是很重要的。

Consider a B-Tree that indexes ${100}\mathrm{M}$ records with a pagesize of 100 . We can think of every B-Tree node as a way to partition the space, decreasing the "error" and narrowing the region to find the data. We therefore say that the B-Tree with a page-size of 100 has a precision gain of $1/{100}$ per node and we need to traverse in total ${\log }_{100}N$ nodes. So the first node partitions the space from ${100M}$ to ${100M}/{100} = {1M}$ ,the second from ${1M}$ to ${1M}/{100} = {10k}$ and so on,until we find the record. Now, traversing a single B-Tree page with binary search takes roughly 50 cycles and is notoriously hard to parallelize ${}^{3}$ . In contrast, a modern CPU can do 8-16 SIMD operations per cycle. Thus, a model will be faster as long as it has a better precision gain than $1/{100}$ per ${50} * 8 = {400}$ arithmetic operations. Note that this calculation still assumes that all B-Tree pages are in the cache. A single cache-miss costs 50-100 additional cycles and would thus allow for even more complex models.

考虑一棵对${100}\mathrm{M}$条记录进行索引的B树，页面大小为100。我们可以将每个B树节点看作是一种划分空间的方式，它可以减少“误差”并缩小查找数据的区域。因此，我们说页面大小为100的B树每个节点的精度增益为$1/{100}$，并且我们总共需要遍历${\log }_{100}N$个节点。所以，第一个节点将空间从${100M}$划分到${100M}/{100} = {1M}$，第二个节点从${1M}$划分到${1M}/{100} = {10k}$，依此类推，直到我们找到记录。现在，使用二分查找遍历单个B树页面大约需要50个周期，并且众所周知很难并行化${}^{3}$。相比之下，现代CPU每个周期可以执行8 - 16次单指令多数据（SIMD）操作。因此，只要一个模型在每${50} * 8 = {400}$次算术运算中的精度增益优于$1/{100}$，它就会更快。请注意，此计算仍然假设所有B树页面都在缓存中。一次缓存未命中会额外消耗50 - 100个周期，因此可以允许使用更复杂的模型。

Additionally, machine learning accelerators are entirely changing the game. They allow to run much more complex models in the same amount of time and offload computation from the CPU. For example, NVIDIA's latest Tesla V100 GPU is able to achieve 120 TeraFlops of low-precision deep learning arithmetic operations ( $\approx  {60},{000}$ operations per cycle). Assuming that the entire learned index fits into the GPU's memory (we show in Section 3.7 that this is a very reasonable assumption), in just 30 cycles we could execute 1 million neural net operations. Of course, the latency for transferring the input and retrieving the result from a GPU is still significantly higher, but this problem is not insuperable given batching and/or the recent trend to more closely integrate CPU/GPU/T-PUs [4]. Finally, it can be expected that the capabilities and the number of floating/int operations per second of GPUs/TPUs will continue to increase, whereas the progress on increasing the performance of executing if-statements of CPUs essentially has stagnated [5]. Regardless of the fact that we consider GPUs/TPUs as one of the main reasons to adopt learned indexes in practice, in this paper we focus on the more limited CPUs to better study the implications of replacing and enhancing indexes through machine learning without the impact of hardware changes.

此外，机器学习加速器正在彻底改变现状。它们允许在相同的时间内运行更复杂的模型，并将计算任务从CPU卸载。例如，英伟达（NVIDIA）最新的特斯拉V100 GPU能够实现120万亿次低精度深度学习算术运算（每个周期$\approx  {60},{000}$次运算）。假设整个学习索引能够装入GPU的内存（我们将在3.7节中表明这是一个非常合理的假设），那么仅需30个周期，我们就可以执行100万次神经网络运算。当然，从GPU传输输入数据和获取结果的延迟仍然显著较高，但考虑到批处理和/或近期CPU/GPU/T - PU更紧密集成的趋势，这个问题并非不可克服[4]。最后，可以预计，GPU/TPU的每秒浮点/整数运算能力和数量将继续增加，而CPU执行条件语句性能的提升基本上已经停滞[5]。尽管我们认为GPU/TPU是在实践中采用学习索引的主要原因之一，但在本文中，我们专注于性能更有限的CPU，以便在不受硬件变化影响的情况下，更好地研究通过机器学习替换和增强索引的影响。

### 2.2 Range Index Models are CDF Models

### 2.2 范围索引模型即累积分布函数（CDF）模型

As stated in the beginning of the section, an index is a model that takes a key as an input and predicts the position of the record. Whereas for point queries the order of the records does not matter, for range queries the data has to be sorted according to the look-up key so that all data items in a range (e.g., in a time frame) can be efficiently retrieved. This leads to an interesting observation: a model that predicts the position given a key inside a sorted array effectively approximates the cumulative distribution function (CDF). We can model the CDF of the data to predict the position as:

如本节开头所述，索引是一种以键为输入并预测记录位置的模型。对于点查询而言，记录的顺序无关紧要，但对于范围查询，数据必须根据查找键进行排序，以便能够高效地检索某个范围内（例如，某个时间框架内）的所有数据项。这引出了一个有趣的观察结果：在排序数组中，根据键预测位置的模型实际上是在近似累积分布函数（CDF）。我们可以对数据的累积分布函数进行建模，以预测位置，具体如下：

$$
p = F\left( \mathrm{{Key}}\right)  * N \tag{1}
$$

where $p$ is the position estimate, $F\left( \mathrm{{Key}}\right)$ is the estimated cumulative distribution function for the data to estimate the likelihood to observe a key smaller or equal to the look-up key $P\left( {X \leq  \text{Key}}\right)$ ,and $N$ is the total number of keys (see also Figure 2). This observation opens up a whole new set of interesting directions: First, it implies that indexing literally requires learning a data distribution. A B-Tree "learns" the data distribution by building a regression tree. A linear regression model would learn the data distribution by minimizing the (squared) error of a linear function. Second, estimating the distribution for a dataset is a well known problem and learned indexes can benefit from decades of research. Third, learning the CDF plays also a key role in optimizing other types of index structures and potential algorithms as we will outline later in this paper. Fourth, there is a long history of research on how closely theoretical CDFs approximate empirical CDFs that gives a foothold to theoretically understand the benefits of this approach [28]. We give a high-level theoretical analysis of how well our approach scales in Appendix A.

其中，$p$是位置估计值，$F\left( \mathrm{{Key}}\right)$是用于估计观察到小于或等于查找键$P\left( {X \leq  \text{Key}}\right)$的键的可能性的数据的估计累积分布函数，$N$是键的总数（另见图2）。这一观察结果开辟了一系列全新的有趣研究方向：首先，这意味着索引实际上需要学习数据分布。B - 树通过构建回归树来“学习”数据分布。线性回归模型则通过最小化线性函数的（平方）误差来学习数据分布。其次，估计数据集的分布是一个众所周知的问题，学习索引可以从数十年的研究中受益。第三，学习累积分布函数在优化其他类型的索引结构和潜在算法方面也起着关键作用，我们将在本文后面进行阐述。第四，关于理论累积分布函数与经验累积分布函数的近似程度已有很长的研究历史，这为从理论上理解这种方法的益处提供了依据[28]。我们在附录A中对我们的方法的可扩展性进行了高层次的理论分析。

---

<!-- Footnote -->

${}^{3}$ There exist SIMD optimized index structures such as FAST [44],but they can only transform control dependencies to memory dependencies. These are often significantly slower than multiplications with simple in-cache data dependencies and as our experiments show SIMD optimized index structures, like FAST, are not significantly faster.

${}^{3}$ 存在诸如FAST [44]之类的单指令多数据（SIMD）优化索引结构，但它们只能将控制依赖转换为内存依赖。这些操作通常比具有简单缓存内数据依赖的乘法运算慢得多，而且正如我们的实验所示，像FAST这样的SIMD优化索引结构并没有显著更快。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Pos A Key Figure 2: Indexes as CDFs -->

<img src="https://cdn.noedgeai.com/0195c8fc-f853-7b31-b5ff-dad98e8dad50_3.jpg?x=233&y=255&w=569&h=293&r=0"/>

<!-- Media -->

### 2.3 A First, Naive Learned Index

### 2.3 第一个简单的学习索引

To better understand the requirements to replace B-Trees through learned models,we used ${200}\mathrm{M}$ web-server log records with the goal of building a secondary index over the timestamps using Tensorflow [9]. We trained a two-layer fully-connected neural network with 32 neurons per layer using ReLU activation functions; the timestamps are the input features and the positions in the sorted array are the labels. Afterwards we measured the look-up time for a randomly selected key (averaged over several runs disregarding the first numbers) with Tensorflow and Python as the front-end.

为了更好地理解用学习模型替换B - 树的要求，我们使用了${200}\mathrm{M}$条Web服务器日志记录，目标是使用TensorFlow [9]在时间戳上构建一个二级索引。我们训练了一个两层的全连接神经网络，每层有32个神经元，使用修正线性单元（ReLU）激活函数；时间戳作为输入特征，排序数组中的位置作为标签。之后，我们使用TensorFlow和Python作为前端，测量了随机选择的键的查找时间（多次运行的平均值，不考虑前几次的结果）。

In this setting we achieved $\approx  {1250}$ predictions per second, i.e.,it takes $\approx  {80},{000}$ nano-seconds (ns) to execute the model with Tensorflow, without the search time (the time to find the actual record from the predicted position). As a comparison point,a B-Tree traversal over the same data takes $\approx  {300}\mathrm{{ns}}$ and binary search over the entire data roughly $\approx  {900}\mathrm{{ns}}$ . With a closer look, we find our naïve approach is limited in a few key ways: (1) Tensorflow was designed to efficiently run larger models, not small models, and thus, has a significant invocation overhead, especially with Python as the front-end. (2) B-Trees, or decision trees in general, are really good in overfitting the data with a few operations as they recursively divide the space using simple if-statements. In contrast, other models can be significantly more efficient to approximate the general shape of a CDF, but have problems being accurate at the individual data instance level. To see this, consider again Figure 2. The figure demonstrates, that from a top-level view, the CDF function appears very smooth and regular. However, if one zooms in to the individual records, more and more irregularities show; a well known statistical effect. Thus models like neural nets, polynomial regression, etc. might be more CPU and space efficient to narrow down the position for an item from the entire dataset to a region of thousands, but a single neural net usually requires significantly more space and CPU time for the "last mile" to reduce the error further down from thousands to hundreds. (3) B-Trees are extremely cache- and operation-efficient as they keep the top nodes always in cache and access other pages if needed. In contrast, standard neural nets require all weights to compute a prediction, which has a high cost in the number of multiplications.

在这种设置下，我们实现了每秒$\approx  {1250}$次预测，即使用TensorFlow执行模型（不包括搜索时间，即从预测位置查找实际记录的时间）需要$\approx  {80},{000}$纳秒（ns）。作为对比，对相同数据进行B树遍历需要$\approx  {300}\mathrm{{ns}}$，对整个数据进行二分查找大约需要$\approx  {900}\mathrm{{ns}}$。仔细观察后，我们发现我们的简单方法在几个关键方面存在局限性：（1）TensorFlow旨在高效运行大型模型，而非小型模型，因此存在显著的调用开销，尤其是以Python作为前端时。（2）一般来说，B树或决策树非常擅长通过少量操作对数据进行过拟合，因为它们使用简单的if语句递归地划分空间。相比之下，其他模型在近似累积分布函数（CDF）的总体形状方面可能效率更高，但在单个数据实例层面的准确性存在问题。为了说明这一点，请再次参考图2。该图表明，从顶层视角看，CDF函数看起来非常平滑和规则。然而，如果放大到单个记录，会发现越来越多的不规则性，这是一种众所周知的统计效应。因此，像神经网络、多项式回归等模型在将数据集中某个项目的位置从整个数据集缩小到数千个数据的区域时，可能在CPU和空间使用上更高效，但单个神经网络通常需要更多的空间和CPU时间来完成“最后一英里”，将误差从数千进一步降低到数百。（3）B树在缓存和操作方面极其高效，因为它们总是将顶层节点保存在缓存中，并在需要时访问其他页面。相比之下，标准的神经网络需要所有权重来进行预测，这在乘法运算次数上成本很高。

## 3 THE RM-INDEX

## 3 RM索引

In order to overcome the challenges and explore the potential of models as index replacements or optimizations, we developed the learning index framework (LIF), recursive-model indexes (RMI), and standard-error-based search strategies. We primarily focus on simple, fully-connected neural nets because of their simplicity and flexibility, but we believe other types of models may provide additional benefits.

为了克服这些挑战并探索将模型用作索引替代或优化的潜力，我们开发了学习索引框架（LIF）、递归模型索引（RMI）和基于标准误差的搜索策略。我们主要关注简单的全连接神经网络，因为它们简单且灵活，但我们相信其他类型的模型可能会带来额外的好处。

### 3.1 The Learning Index Framework (LIF)

### 3.1 学习索引框架（LIF）

The LIF can be regarded as an index synthesis system; given an index specification, LIF generates different index configurations, optimizes them, and tests them automatically. While LIF can learn simple models on-the-fly (e.g., linear regression models), it relies on Tensorflow for more complex models (e.g., NN). However, it never uses Tensorflow at inference Rather, given a trained Tensorflow model, LIF automatically extracts all weights from the model and generates efficient index structures in $\mathrm{C} +  +$ based on the model specification. Our code-generation is particularly designed for small models and removes all unnecessary overhead and instrumentation that Tensorflow has to manage the larger models. Here we leverage ideas from [25], which already showed how to avoid unnecessary overhead from the Spark-runtime. As a result, we are able to execute simple models on the order of 30 nano-seconds. However, it should be pointed out that LIF is still an experimental framework and is instrumentalized to quickly evaluate different index configurations (e.g., ML models, page-sizes, search strategies, etc.), which introduces additional overhead in form of additional counters, virtual function calls, etc. Also besides the vectorization done by the compiler, we do not make use of special SIMD intrinisics. While these inefficiencies do not matter in our evaluation as we ensure a fair comparison by always using our framework, for a production setting or when comparing the reported performance numbers with other implementations, these inefficiencies should be taking into account/be avoided.

LIF可以被视为一个索引合成系统；给定一个索引规范，LIF会生成不同的索引配置，对其进行优化，并自动进行测试。虽然LIF可以即时学习简单的模型（例如线性回归模型），但对于更复杂的模型（例如神经网络），它依赖于TensorFlow。然而，在推理时它从不使用TensorFlow。相反，给定一个训练好的TensorFlow模型，LIF会自动从模型中提取所有权重，并根据模型规范在$\mathrm{C} +  +$中生成高效的索引结构。我们的代码生成特别针对小型模型设计，去除了TensorFlow管理大型模型时所需的所有不必要的开销和工具。在这里，我们借鉴了文献[25]的思路，该文献已经展示了如何避免Spark运行时的不必要开销。因此，我们能够在大约30纳秒内执行简单模型。然而，应该指出的是，LIF仍然是一个实验性框架，用于快速评估不同的索引配置（例如机器学习模型、页面大小、搜索策略等），这会以额外的计数器、虚函数调用等形式引入额外的开销。此外，除了编译器进行的向量化操作外，我们没有使用特殊的单指令多数据（SIMD）内在函数。虽然在我们的评估中这些低效性并不重要，因为我们始终使用我们的框架以确保公平比较，但在生产环境中或在将报告的性能指标与其他实现进行比较时，应该考虑/避免这些低效性。

### 3.2 The Recursive Model Index

### 3.2 递归模型索引

As outlined in Section 2.3 one of the key challenges of building alternative learned models to replace B-Trees is the accuracy for last-mile search. For example, reducing the prediction error to the order of hundreds from ${100}\mathrm{M}$ records using a single model is often difficult. At the same time, reducing the error to ${10}\mathrm{\;k}$ from ${100}\mathrm{M}$ ,e.g.,a precision gain of ${100} * {100} = {10000}$ to replace the first 2 layers of a B-Tree through a model, is much easier to achieve even with simple models. Similarly, reducing the error from ${10}\mathrm{k}$ to 100 is a simpler problem as the model can focus only on a subset of the data.

如第2.3节所述，构建替代学习模型以取代B树的关键挑战之一是最后一英里搜索的准确性。例如，使用单个模型将预测误差从${100}\mathrm{M}$条记录降低到数百条记录通常很困难。同时，将误差从${100}\mathrm{M}$降低到${10}\mathrm{\;k}$，例如，通过一个模型将精度提高${100} * {100} = {10000}$以取代B树的前两层，即使使用简单的模型也更容易实现。同样，将误差从${10}\mathrm{k}$降低到100是一个更简单的问题，因为模型可以只关注数据的一个子集。

Based on that observation and inspired by the mixture of experts work [62], we propose the recursive regression model (see Figure 3). That is, we build a hierarchy of models, where at each stage the model takes the key as an input and based on it picks another model, until the final stage predicts the position. More formally,for our model $f\left( x\right)$ where $x$ is the key and $y \in  \lbrack 0,N)$ the position,we assume at stage $\ell$ there are ${M}_{\ell }$ models. We train the model at stage $0,{f}_{0}\left( x\right)  \approx  y$ . As such, model $k$ in stage $\ell$ ,denoted by ${f}_{\ell }^{\left( k\right) }$ ,is trained with loss:

基于该观察结果，并受到专家混合模型研究 [62] 的启发，我们提出了递归回归模型（见图 3）。即，我们构建一个模型层次结构，在每个阶段，模型将键作为输入，并基于此选择另一个模型，直到最后阶段预测位置。更正式地说，对于我们的模型 $f\left( x\right)$，其中 $x$ 是键，$y \in  \lbrack 0,N)$ 是位置，我们假设在阶段 $\ell$ 有 ${M}_{\ell }$ 个模型。我们在阶段 $0,{f}_{0}\left( x\right)  \approx  y$ 训练模型。因此，阶段 $\ell$ 中的模型 $k$，用 ${f}_{\ell }^{\left( k\right) }$ 表示，使用以下损失函数进行训练：

$$
{L}_{\ell } = \mathop{\sum }\limits_{\left( x,y\right) }{\left( {f}_{\ell }^{\left( \left\lfloor  {M}_{\ell }{f}_{\ell  - 1}\left( x\right) /N\right\rfloor  \right) }\left( x\right)  - y\right) }^{2}\;{L}_{0} = \mathop{\sum }\limits_{\left( x,y\right) }{\left( {f}_{0}\left( x\right)  - y\right) }^{2}
$$

<!-- Media -->

<!-- figureText: Model 2.1 Model 1.1 Model 2.2 Model 2.3 Model 3.3 Model 3.4 Figure 3: Staged models Stage 3 Model 3.1 Model 3.2 Position -->

<img src="https://cdn.noedgeai.com/0195c8fc-f853-7b31-b5ff-dad98e8dad50_4.jpg?x=195&y=225&w=633&h=372&r=0"/>

<!-- Media -->

Note,we use here the notation of ${f}_{\ell  - 1}\left( x\right)$ recursively executing ${f}_{\ell  - 1}\left( x\right)  = {f}_{\ell  - 1}^{\left( \left\lfloor  {M}_{\ell  - 1}{f}_{\ell  - 2}\left( x\right) /N\right\rfloor  \right) }\left( x\right)$ . In total,we iteratively train each stage with loss ${L}_{\ell }$ to build the complete model.

注意，我们在这里使用 ${f}_{\ell  - 1}\left( x\right)$ 递归执行 ${f}_{\ell  - 1}\left( x\right)  = {f}_{\ell  - 1}^{\left( \left\lfloor  {M}_{\ell  - 1}{f}_{\ell  - 2}\left( x\right) /N\right\rfloor  \right) }\left( x\right)$ 的表示法。总体而言，我们使用损失函数 ${L}_{\ell }$ 迭代训练每个阶段，以构建完整的模型。

One way to think about the different models is that each model makes a prediction with a certain error about the position for the key and that the prediction is used to select the next model, which is responsible for a certain area of the key-space to make a better prediction with a lower error. However, recursive model indexes do not have to be trees. As shown in Figure 3 it is possible that different models of one stage pick the same models at the stage below. Furthermore, each model does not necessarily cover the same amount of records like B-Trees do (i.e., a B-Tree with a page-size of 100 covers 100 or less records). ${}^{4}$ Finally,depending on the used models the predictions between the different stages can not necessarily be interpreted as positions estimates, rather should be considered as picking an expert which has a better knowledge about certain keys (see also [62]).

理解不同模型的一种方式是，每个模型针对键的位置进行带有一定误差的预测，并且该预测用于选择下一个模型，该模型负责键空间的特定区域，以进行误差更小的更好预测。然而，递归模型索引不必是树状结构。如图 3 所示，一个阶段的不同模型有可能选择下一个阶段的相同模型。此外，每个模型不一定像 B 树那样覆盖相同数量的记录（即，页大小为 100 的 B 树覆盖 100 条或更少的记录）。${}^{4}$ 最后，根据所使用的模型，不同阶段之间的预测不一定能被解释为位置估计，而应被视为选择一个对某些键有更深入了解的专家（另见 [62]）。

This model architecture has several benefits: (1) It separates model size and complexity from execution cost. (2) It leverages the fact that it is easy to learn the overall shape of the data distribution. (3) It effectively divides the space into smaller subranges, like a B-Tree, to make it easier to achieve the required "last mile" accuracy with fewer operations. (4) There is no search process required in-between the stages. For example, the output of Model 1.1 is directly used to pick the model in the next stage. This not only reduces the number of instructions to manage the structure, but also allows representing the entire index as a sparse matrix-multiplication for a TPU/GPU.

这种模型架构有几个优点：（1）它将模型大小和复杂度与执行成本分离。（2）它利用了易于学习数据分布整体形状这一事实。（3）它像 B 树一样有效地将空间划分为更小的子范围，以便用更少的操作更容易达到所需的“最后一公里”精度。（4）阶段之间不需要搜索过程。例如，模型 1.1 的输出直接用于选择下一阶段的模型。这不仅减少了管理结构所需的指令数量，还允许将整个索引表示为适用于 TPU/GPU 的稀疏矩阵乘法。

### 3.3 Hybrid Indexes

### 3.3 混合索引

Another advantage of the recursive model index is, that we are able to build mixtures of models. For example, whereas on the top-layer a small ReLU neural net might be the best choice as they are usually able to learn a wide-range of complex data distributions, the models at the bottom of the model hierarchy might be thousands of simple linear regression models as they are inexpensive in space and execution time. Furthermore, we can even use traditional B-Trees at the bottom stage if the data is particularly hard to learn.

递归模型索引的另一个优点是，我们能够构建模型的混合体。例如，虽然在顶层，小型 ReLU 神经网络可能是最佳选择，因为它们通常能够学习广泛的复杂数据分布，但模型层次结构底层的模型可能是数千个简单的线性回归模型，因为它们在空间和执行时间上成本较低。此外，如果数据特别难以学习，我们甚至可以在底层阶段使用传统的 B 树。

For this paper, we focus on 2 types of models, simple neural nets with zero to two fully-connected hidden layers and ReLU activation functions and a layer width of up to 32 neurons and B-Trees (a.k.a. decision trees). Note, that a zero hidden-layer NN is equivalent to linear regression. Given an index configuration, which specifies the number of stages and the number of models per stage as an array of sizes, the end-to-end training for hybrid indexes is done as shown in Algorithm 1

在本文中，我们专注于两种类型的模型：具有零到两个全连接隐藏层和 ReLU 激活函数且层宽度最多为 32 个神经元的简单神经网络，以及 B 树（又称决策树）。注意，零隐藏层的神经网络等同于线性回归。给定一个索引配置，它以大小数组的形式指定阶段数量和每个阶段的模型数量，混合索引的端到端训练如算法 1 所示

<!-- Media -->

Algorithm 1: Hybrid End-To-End Training

算法 1：混合端到端训练

---

Input: int threshold, int stages[], NN_complexity

输入：整数阈值，整数数组 stages[]，神经网络复杂度

Data: record data[], Model index[][]

数据：记录数组 data[]，二维模型数组 index[][]

Result: trained index

结果：训练好的索引

$M =$ stages.size;

$M =$ 阶段数组的大小;

tmp_records[][];

临时记录二维数组 tmp_records[][];

tmp_records[1][1] = all_data;

tmp_records[1][1] = 所有数据;

for $i \leftarrow  1$ to $M$ do

从 $i \leftarrow  1$ 到 $M$ 执行循环

	for $j \leftarrow  1$ to stages $\left\lbrack  i\right\rbrack$ do

	  从 $j \leftarrow  1$ 到 stages $\left\lbrack  i\right\rbrack$ 执行循环

		index[i][j] = new NN trained on tmp_records[i][j];

		index[i][j] = 基于临时记录tmp_records[i][j]训练得到的新神经网络（NN）模型;

		if $i < M$ then

		if $i < M$ 那么

			for $r \in$ tmp_records[i][j] do

			对于 $r \in$ 临时记录tmp_records[i][j] 执行

				$p = \operatorname{index}\left\lbrack  \mathrm{i}\right\rbrack  \left\lbrack  \mathrm{j}\right\rbrack  \left( {r.{key}}\right) /\operatorname{stages}\left\lbrack  {i + 1}\right\rbrack$ ;

				tmp_records $\left\lbrack  {i + 1}\right\rbrack  \left\lbrack  p\right\rbrack$ .add(r);

				临时记录 $\left\lbrack  {i + 1}\right\rbrack  \left\lbrack  p\right\rbrack$ .添加(r);

for $j \leftarrow  1$ to index $\left\lbrack  M\right\rbrack$ .size do

从 $j \leftarrow  1$ 到索引 $\left\lbrack  M\right\rbrack$ 的大小 执行

	index $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack$ .calc_err(tmp_records $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack$ );

	索引 $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack$ .计算临时记录 $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack$ 的误差;

	if index $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack$ .max_abs_err > threshold then

	如果索引 $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack$ 的最大绝对误差 > 阈值 那么

		index $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack   =$ new B-Tree trained on tmp_records $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack$ ;

		索引 $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack   =$ 为基于临时记录 $\left\lbrack  M\right\rbrack  \left\lbrack  j\right\rbrack$ 训练得到的新B树;

return index;

返回索引;

---

<!-- Media -->

Starting from the entire dataset (line 3), it trains first the top-node model. Based on the prediction of this top-node model, it then picks the model from the next stage (lines 9 and 10) and adds all keys which fall into that model (line 10). Finally, in the case of hybrid indexes, the index is optimized by replacing NN models with B-Trees if absolute min-/max-error is above a predefined threshold (lines 11-14).

从整个数据集开始（第3行），首先训练顶层节点模型。基于该顶层节点模型的预测结果，接着从下一阶段选择模型（第9行和第10行），并添加所有属于该模型的键（第10行）。最后，对于混合索引的情况，如果绝对最小/最大误差高于预定义的阈值，则通过用B树替换神经网络（NN）模型来优化索引（第11 - 14行）。

Note, that we store the standard and min- and max-error for every model on the last stage. That has the advantage, that we can individually restrict the search space based on the used model for every key. Currently, we tune the various parameters of the model (i.e., number of stages, hidden layers per model, etc.) with a simple simple grid-search. However, many potential optimizations exists to speed up the training process from ML auto tuning to sampling.

注意，我们在最后一个阶段为每个模型存储标准误差、最小误差和最大误差。这样做的优点是，我们可以根据每个键所使用的模型单独限制搜索空间。目前，我们使用简单的网格搜索来调整模型的各种参数（即阶段数、每个模型的隐藏层数等）。然而，从机器学习自动调优到采样，存在许多潜在的优化方法可以加速训练过程。

Note, that hybrid indexes allow us to bound the worst case performance of learned indexes to the performance of B-Trees. That is, in the case of an extremely difficult to learn data distribution, all models would be automatically replaced by B-Trees, making it virtually an entire B-Tree.

注意，混合索引使我们能够将学习型索引的最坏情况性能限制在B树的性能范围内。也就是说，在数据分布极难学习的情况下，所有模型将自动被B树替换，实际上使其成为一个完整的B树。

### 3.4 Search Strategies and Monotonicity

### 3.4 搜索策略与单调性

Range indexes usually implement an upper_bound(key) [lower_ bound(key)] interface to find the position of the first key within the sorted array that is equal or higher [lower] than the lookup key to efficiently support range requests. For learned range indexes we therefore have to find the first key higher [lower] from the look-up key based on the prediction. Despite many efforts, it was repeatedly reported [8] that binary search or scanning for records with small payloads are usually the fastest strategies to find a key within a sorted array as the additional complexity of alternative techniques rarely pays off. However, learned indexes might have an advantage here: the models actually predict the position of the key, not just the region (i.e., page) of the key. Here we discuss two simple search strategies which take advantage of this information:

范围索引通常实现一个upper_bound(key) [lower_bound(key)] 接口，以在排序数组中找到第一个等于或高于 [低于] 查找键的键的位置，从而有效地支持范围查询。因此，对于学习型范围索引，我们必须根据预测结果从查找键开始找到第一个更高 [更低] 的键。尽管付出了很多努力，但多次有报告 [8] 指出，对于小负载的记录，二分查找或扫描通常是在排序数组中查找键的最快策略，因为替代技术的额外复杂性很少能带来回报。然而，学习型索引在此可能具有优势：模型实际上预测的是键的位置，而不仅仅是键所在的区域（即页）。这里我们讨论两种利用此信息的简单搜索策略：

---

<!-- Footnote -->

${}^{4}$ Note,that we currently train stage-wise and not fully end-to-end. End-to-end training would be even better and remains future work.

${}^{4}$ 注意，我们目前是按阶段进行训练，而不是完全端到端的训练。端到端训练会更好，这留待未来研究。

<!-- Footnote -->

---

Model Biased Search: Our default search strategy, which only varies from traditional binary search in that the first middle point is set to the value predicted by the model.

模型偏置搜索：我们的默认搜索策略，它与传统二分查找的唯一区别在于，第一个中间点被设置为模型预测的值。

Biased Quaternary Search: Quaternary search takes instead of one split point three points with the hope that the hardware pre-fetches all three data points at once to achieve better performance if the data is not in cache. In our implementation, we defined the initial three middle points of quaternary search as ${pos} - \sigma ,{pos},{pos} + \sigma$ . That is we make a guess that most of our predictions are accurate and focus our attention first around the position estimate and then we continue with traditional quaternary search.

有偏四元搜索：四元搜索采用三个分割点而非一个，期望在数据未缓存时，硬件能一次性预取所有三个数据点以实现更好的性能。在我们的实现中，我们将四元搜索的初始三个中间点定义为 ${pos} - \sigma ,{pos},{pos} + \sigma$。也就是说，我们假设大部分预测是准确的，首先将注意力集中在位置估计附近，然后继续进行传统的四元搜索。

For all our experiments we used the min- and max-error as the search area for all techniques. That is, we executed the RMI model for every key and stored the worst over- and under-prediction per last-stage model. While this technique guarantees to find all existing keys, for non-existing keys it might return the wrong upper or lower bound if the RMI model is not monotonic. To overcome this problem, one option is to force our RMI model to be monotonic, as has been studied in machine learning $\left\lbrack  {{41},{71}}\right\rbrack$ .

在我们所有的实验中，我们将最小和最大误差作为所有技术的搜索区域。也就是说，我们为每个键执行 RMI 模型，并存储每个最后阶段模型的最差过预测和欠预测。虽然这种技术保证能找到所有现有的键，但对于不存在的键，如果 RMI 模型不是单调的，它可能会返回错误的上界或下界。为了克服这个问题，一种选择是强制我们的 RMI 模型具有单调性，正如机器学习中所研究的那样 $\left\lbrack  {{41},{71}}\right\rbrack$。

Alternatively, for non-monotonic models we can automatically adjust the search area. That is, if the found upper (lower) bound key is on the boundary of the search area defined by the min- and max-error, we incrementally adjust the search area. Yet, another possibility is, to use exponential search techniques. Assuming a normal distributed error, those techniques on average should work as good as alternative search strategies while not requiring to store any min- and max-errors.

或者，对于非单调模型，我们可以自动调整搜索区域。也就是说，如果找到的上（下）界键位于由最小和最大误差定义的搜索区域的边界上，我们会逐步调整搜索区域。此外，另一种可能性是使用指数搜索技术。假设误差呈正态分布，这些技术平均而言应该与其他搜索策略一样有效，同时不需要存储任何最小和最大误差。

### 3.5 Indexing Strings

### 3.5 字符串索引

We have primarily focused on indexing real valued keys, but many databases rely on indexing strings, and luckily, significant machine learning research has focused on modeling strings. As before, we need to design a model of strings that is efficient yet expressive. Doing this well for strings opens a number of unique challenges.

我们主要专注于对实值键进行索引，但许多数据库依赖于对字符串进行索引，幸运的是，大量机器学习研究都集中在对字符串进行建模上。和之前一样，我们需要设计一个高效且表达能力强的字符串模型。要做好字符串的索引工作会带来一些独特的挑战。

The first design consideration is how to turn strings into features for the model, typically called tokenization. For simplicity and efficiency,we consider an $n$ -length string to be a feature vector $\mathbf{x} \in  {\mathbb{R}}^{n}$ where ${\mathbf{x}}_{i}$ is the ASCII decimal value (or Unicode decimal value depending on the strings). Further, most ML models operate more efficiently if all inputs are of equal size. As such,we will set a maximum input length $N$ . Because the data is sorted lexicographically, we will truncate the keys to length $N$ before tokenization. For strings with length $n < N$ ,we set ${\mathbf{x}}_{i} = 0$ for $i > n$ .

第一个设计考虑是如何将字符串转换为模型的特征，通常称为分词。为了简单和高效起见，我们将一个 $n$ 长度的字符串视为一个特征向量 $\mathbf{x} \in  {\mathbb{R}}^{n}$，其中 ${\mathbf{x}}_{i}$ 是 ASCII 十进制值（或者根据字符串情况为 Unicode 十进制值）。此外，如果所有输入的大小相等，大多数机器学习模型的运行效率会更高。因此，我们将设置一个最大输入长度 $N$。由于数据是按字典序排序的，我们将在分词之前将键截断为长度 $N$。对于长度为 $n < N$ 的字符串，我们为 $i > n$ 设置 ${\mathbf{x}}_{i} = 0$。

For efficiency, we generally follow a similar modeling approach as we did for real valued inputs. We learn a hierarchy of relatively small feed-forward neural networks. The one difference is that the input is not a single real value $x$ but a vector $\mathbf{x}$ . Linear models $\mathbf{w} \cdot  \mathbf{x} + \mathbf{b}$ scale the number of multiplications and additions linearly with the input length $N$ . Feed-forward neural networks with even a single hidden layer of width $h$ will scale $O\left( {hN}\right)$ multiplications and additions.

为了提高效率，我们通常采用与处理实值输入类似的建模方法。我们学习一个相对较小的前馈神经网络层次结构。唯一的区别是输入不是单个实值 $x$，而是一个向量 $\mathbf{x}$。线性模型 $\mathbf{w} \cdot  \mathbf{x} + \mathbf{b}$ 的乘法和加法数量与输入长度 $N$ 呈线性比例关系。即使是具有宽度为 $h$ 的单个隐藏层的前馈神经网络，其乘法和加法数量也会按 $O\left( {hN}\right)$ 比例增长。

Ultimately, we believe there is significant future research that can optimize learned indexes for string keys. For example, we could easily imagine other tokenization algorithms. There is a large body of research in natural language processing on string tokenization to break strings into more useful segments for ML models, e.g., wordpieces in translation [70]. Further, it might be interesting to combine the idea of suffix-trees with learned indexes as well as explore more complex model architectures (e.g., recurrent and convolutional neural networks).

最终，我们认为未来有大量研究可以优化针对字符串键的学习索引。例如，我们可以轻松想象其他分词算法。自然语言处理领域有大量关于字符串分词的研究，旨在将字符串拆分为对机器学习模型更有用的片段，例如翻译中的词块 [70]。此外，将后缀树的思想与学习索引相结合，以及探索更复杂的模型架构（例如循环和卷积神经网络）可能会很有趣。

### 3.6 Training

### 3.6 训练

While the training (i.e., loading) time is not the focus of this paper, it should be pointed out that all of our models, shallow NNs or even simple linear/multi-variate regression models, train relatively fast. Whereas simple NNs can be efficiently trained using stochastic gradient descent and can converge in less than one to a few passes over the randomized data, a closed form solution exists for linear multi-variate models (e.g., also 0-layer NN) and they can be trained in a single pass over the sorted data. Therefore,for ${200}\mathrm{M}$ records training a simple RMI index does not take much longer than a few seconds, (of course, depending on how much auto-tuning is performed); neural nets can train on the order of minutes per model, depending on the complexity. Also note that training the top model over the entire data is usually not necessary as those models converge often even before a single scan over the entire randomized data. This is in part because we use simple models and do not care much about the last few digit points in precision, as it has little effect on indexing performance. Finally, research on improving learning time from the ML community $\left\lbrack  {{27},{72}}\right\rbrack$ applies in our context and we expect a lot of future research in this direction.

虽然训练（即加载）时间并非本文的重点，但应当指出，我们所有的模型，无论是浅层神经网络（NN），还是简单的线性/多元回归模型，训练速度都相对较快。简单的神经网络可以使用随机梯度下降法进行高效训练，并且在对随机数据进行不到一轮至几轮遍历后即可收敛；而线性多元模型（例如，零层神经网络）存在闭式解，并且可以在对排序数据进行一轮遍历后完成训练。因此，对于${200}\mathrm{M}$条记录，训练一个简单的递归模型索引（RMI）所需的时间不会超过几秒（当然，这取决于进行了多少自动调优）；神经网络每个模型的训练时间可能在几分钟左右，具体取决于模型的复杂度。此外，值得注意的是，通常无需对整个数据集训练顶层模型，因为这些模型往往在对整个随机数据集进行一轮扫描之前就已经收敛。这在一定程度上是因为我们使用的是简单模型，并且不太在意精度的最后几位小数，因为这对索引性能的影响很小。最后，机器学习社区中关于缩短学习时间的研究$\left\lbrack  {{27},{72}}\right\rbrack$同样适用于我们的场景，我们期待未来在这方面有更多的研究。

### 3.7 Results

### 3.7 结果

We evaluated learned range indexes in regard to their space and speed on several real and synthetic data sets against other read-optimized index structures.

我们在多个真实和合成数据集上，针对其他读优化索引结构，评估了学习型范围索引在空间和速度方面的性能。

3.7.1 Integer Datasets. As a first experiment we compared learned indexes using a 2-stage RMI model and different second-stage sizes(10k,50k,100k,and200k)with a read-optimized B-Tree with different page sizes on three different integer data sets. For the data we used 2 real-world datasets, (1) Weblogs and (2) Maps [56], and (3) a synthetic dataset, Lognormal. The Weblogs dataset contains ${200}\mathrm{M}$ log entries for every request to a major university web-site over several years. We use the unique request timestamps as the index keys. This dataset is almost a worst-case scenario for the learned index as it contains very complex time patterns caused by class schedules, weekends, holidays, lunch-breaks, department events, semester breaks, etc., which are notoriously hard to learn. For the maps dataset we indexed the longitude of $\approx  {200}\mathrm{M}$ user-maintained features (e.g., roads, museums, coffee shops) across the world. Unsurprisingly, the longitude of locations is relatively linear and has fewer irregularities than the Weblogs dataset. Finally, to test how the index works on heavy-tail distributions,we generated a synthetic dataset of ${190}\mathrm{M}$ unique values sampled from a log-normal distribution with $\mu  = 0$ and $\sigma  = 2$ . The values are scaled up to be integers up to $1\mathrm{\;B}$ . This data is of course highly non-linear, making the CDF more difficult to learn using neural nets. For all B-Tree experiments we used 64-bit keys and 64-bit payload/value.

3.7.1 整数数据集。作为第一个实验，我们在三个不同的整数数据集上，将使用两阶段递归模型索引（RMI）模型且不同第二阶段大小（10k、50k、100k 和 200k）的学习型索引，与具有不同页面大小的读优化 B 树进行了比较。我们使用了两个真实世界的数据集：（1）网络日志（Weblogs）和（2）地图（Maps）[56]，以及（3）一个合成数据集，对数正态分布数据集（Lognormal）。网络日志数据集包含一所主要大学网站多年来每次请求的${200}\mathrm{M}$条日志条目。我们使用唯一的请求时间戳作为索引键。这个数据集对于学习型索引来说几乎是最坏的情况，因为它包含由课程表、周末、假期、午休、部门活动、学期假期等因素导致的非常复杂的时间模式，这些模式极难学习。对于地图数据集，我们对全球$\approx  {200}\mathrm{M}$个用户维护的地理特征（例如，道路、博物馆、咖啡店）的经度进行了索引。不出所料，地理位置的经度相对呈线性，与网络日志数据集相比，不规则性较少。最后，为了测试索引在重尾分布上的性能，我们生成了一个合成数据集，其中包含从参数为$\mu  = 0$和$\sigma  = 2$的对数正态分布中采样得到的${190}\mathrm{M}$个唯一值。这些值被放大为最大到$1\mathrm{\;B}$的整数。当然，这个数据集具有高度的非线性，这使得使用神经网络学习累积分布函数（CDF）变得更加困难。在所有 B 树实验中，我们使用 64 位键和 64 位有效负载/值。

<!-- Media -->

<table><tr><td colspan="2"/><td colspan="3">Map Data</td><td colspan="3">Web Data</td><td colspan="3">Log-Normal Data</td></tr><tr><td>Type</td><td>Config</td><td>Size (MB)</td><td>Lookup (ns)</td><td>Model (ns)</td><td>Size (MB)</td><td>Lookup (ns)</td><td>Model (ns)</td><td>Size (MB)</td><td>Lookup (ns)</td><td>Model (ns)</td></tr><tr><td rowspan="5">Btree</td><td>page size: 32</td><td>52.45 (4.00x)</td><td>274 (0.97x)</td><td>198 (72.3%)</td><td>${51.93}\left( {{4.00}\mathrm{x}}\right)$</td><td>276 (0.94x)</td><td>201 (72.7%)</td><td>49.83(4.00x)</td><td>274 (0.96x)</td><td>198 (72.1%)</td></tr><tr><td>page size: 64</td><td>26.23(2.00x)</td><td>277 (0.96x)</td><td>172 (62.0%)</td><td>25.97 (2.00x)</td><td>274 (0.95x)</td><td>171 (62.4%)</td><td>24.92 (2.00x)</td><td>274 (0.96x)</td><td>169 (61.7%)</td></tr><tr><td>page size: 128</td><td>13.11 (1.00x)</td><td>265 (1.00x)</td><td>134 (50.8%)</td><td>12.98 (1.00x)</td><td>260 (1.00x)</td><td>132 (50.8%)</td><td>12.46 (1.00x)</td><td>263 (1.00x)</td><td>131 (50.0%)</td></tr><tr><td>page size: 256</td><td>6.56 (0.50x)</td><td>267 (0.99x)</td><td>114 (42.7%)</td><td>6.49 (0.50x)</td><td>266 (0.98x)</td><td>114 (42.9%)</td><td>6.23 (0.50x)</td><td>271 (0.97x)</td><td>117 (43.2%)</td></tr><tr><td>page size: 512</td><td>3.28(0.25x)</td><td>286 (0.93x)</td><td>101 (35.3%)</td><td>3.25 (0.25x)</td><td>291 (0.89x)</td><td>100 (34.3%)</td><td>3.11 (0.25x)</td><td>293 (0.90x)</td><td>101 (34.5%)</td></tr><tr><td rowspan="4">Learned Index</td><td>2nd stage models: 10k</td><td>0.15 (0.01x)</td><td>98 (2.70x)</td><td>31 (31.6%)</td><td>0.15 (0.01x)</td><td>222 (1.17x)</td><td>29 (13.1%)</td><td>${0.15}\left( {0.01x}\right)$</td><td>178 (1.47x)</td><td>26 (14.6%)</td></tr><tr><td>2nd stage models: 50k</td><td>0.76(0.06x)</td><td>85 (3.11x)</td><td>39 (45.9%)</td><td>0.76(0.06x)</td><td>162 (1.60x)</td><td>36 (22.2%)</td><td>${0.76}\left( {0.06x}\right)$</td><td>162 (1.62x)</td><td>35 (21.6%)</td></tr><tr><td>2nd stage models: 100k</td><td>1.53(0.12x)</td><td>82 (3.21x)</td><td>41 (50.2%)</td><td>1.53 (0.12x)</td><td>144 (1.81x)</td><td>39 (26.9%)</td><td>1.53 (0.12x)</td><td>152 (1.73x)</td><td>36 (23.7%)</td></tr><tr><td>2nd stage models: 200k</td><td>3.05 (0.23x)</td><td>86 (3.08x)</td><td>50 (58.1%)</td><td>3.05 (0.24x)</td><td>126(2.07x)</td><td>41 (32.5%)</td><td>3.05(0.24x)</td><td>146 (1.79x)</td><td>40 (27.6%)</td></tr></table>

<table><tbody><tr><td colspan="2"></td><td colspan="3">地图数据</td><td colspan="3">网络数据</td><td colspan="3">对数正态分布数据</td></tr><tr><td>类型</td><td>配置</td><td>大小（兆字节）</td><td>查找时间（纳秒）</td><td>模型时间（纳秒）</td><td>大小（兆字节）</td><td>查找时间（纳秒）</td><td>模型时间（纳秒）</td><td>大小（兆字节）</td><td>查找时间（纳秒）</td><td>模型时间（纳秒）</td></tr><tr><td rowspan="5">二叉搜索树（B树）</td><td>页面大小：32</td><td>52.45 (4.00x)</td><td>274 (0.97x)</td><td>198 (72.3%)</td><td>${51.93}\left( {{4.00}\mathrm{x}}\right)$</td><td>276 (0.94x)</td><td>201 (72.7%)</td><td>49.83(4.00x)</td><td>274 (0.96x)</td><td>198 (72.1%)</td></tr><tr><td>页面大小：64</td><td>26.23(2.00x)</td><td>277 (0.96x)</td><td>172 (62.0%)</td><td>25.97 (2.00x)</td><td>274 (0.95x)</td><td>171 (62.4%)</td><td>24.92 (2.00x)</td><td>274 (0.96x)</td><td>169 (61.7%)</td></tr><tr><td>页面大小：128</td><td>13.11 (1.00x)</td><td>265 (1.00x)</td><td>134 (50.8%)</td><td>12.98 (1.00x)</td><td>260 (1.00x)</td><td>132 (50.8%)</td><td>12.46 (1.00x)</td><td>263 (1.00x)</td><td>131 (50.0%)</td></tr><tr><td>页面大小：256</td><td>6.56 (0.50x)</td><td>267 (0.99x)</td><td>114 (42.7%)</td><td>6.49 (0.50x)</td><td>266 (0.98x)</td><td>114 (42.9%)</td><td>6.23 (0.50x)</td><td>271 (0.97x)</td><td>117 (43.2%)</td></tr><tr><td>页面大小：512</td><td>3.28(0.25x)</td><td>286 (0.93x)</td><td>101 (35.3%)</td><td>3.25 (0.25x)</td><td>291 (0.89x)</td><td>100 (34.3%)</td><td>3.11 (0.25x)</td><td>293 (0.90x)</td><td>101 (34.5%)</td></tr><tr><td rowspan="4">学习索引</td><td>第二阶段模型：10000个</td><td>0.15 (0.01x)</td><td>98 (2.70x)</td><td>31 (31.6%)</td><td>0.15 (0.01x)</td><td>222 (1.17x)</td><td>29 (13.1%)</td><td>${0.15}\left( {0.01x}\right)$</td><td>178 (1.47x)</td><td>26 (14.6%)</td></tr><tr><td>第二阶段模型：50000个</td><td>0.76(0.06x)</td><td>85 (3.11x)</td><td>39 (45.9%)</td><td>0.76(0.06x)</td><td>162 (1.60x)</td><td>36 (22.2%)</td><td>${0.76}\left( {0.06x}\right)$</td><td>162 (1.62x)</td><td>35 (21.6%)</td></tr><tr><td>第二阶段模型：100000个</td><td>1.53(0.12x)</td><td>82 (3.21x)</td><td>41 (50.2%)</td><td>1.53 (0.12x)</td><td>144 (1.81x)</td><td>39 (26.9%)</td><td>1.53 (0.12x)</td><td>152 (1.73x)</td><td>36 (23.7%)</td></tr><tr><td>第二阶段模型：200000个</td><td>3.05 (0.23x)</td><td>86 (3.08x)</td><td>50 (58.1%)</td><td>3.05 (0.24x)</td><td>126(2.07x)</td><td>41 (32.5%)</td><td>3.05(0.24x)</td><td>146 (1.79x)</td><td>40 (27.6%)</td></tr></tbody></table>

Figure 4: Learned Index vs B-Tree

图4：学习索引与B树对比

<!-- Media -->

As our baseline, we used a production quality B-Tree implementation which is similar to the stx::btree but with further cache-line optimization, dense pages (i.e., fill factor of 100%), and very competitive performance. To tune the 2-stage learned indexes we used simple grid-search over neural nets with zero to two hidden layers and layer-width ranging from 4 to 32 nodes. In general we found that a simple ( 0 hidden layers) to semi-complex ( 2 hidden layers and 8- or 16-wide) models for the first stage work the best. For the second stage, simple, linear models, had the best performance. This is not surprising as for the last mile it is often not worthwhile to execute complex models, and linear models can be learned optimally.

作为基线，我们使用了一个生产级质量的B树实现，它类似于stx::btree，但进行了进一步的缓存行优化、采用了密集页面（即填充因子为100%），并且性能极具竞争力。为了调整两阶段学习索引，我们对具有零到两个隐藏层且层宽度从4到32个节点的神经网络进行了简单的网格搜索。总体而言，我们发现第一阶段使用简单（0个隐藏层）到半复杂（2个隐藏层且宽度为8或16）的模型效果最佳。对于第二阶段，简单的线性模型性能最佳。这并不奇怪，因为在最后一步执行复杂模型通常不值得，而线性模型可以得到最优学习。

Learned Index vs B-Tree performance: The main results are shown in Figure 4. Note, that the page size for B-Trees indicates the number of keys per page not the size in Bytes, which is actually larger. As the main metrics we show the size in MB, the total look-up time in nano-seconds, and the time to execution the model (either B-Tree traversal or ML model) also in nano-seconds and as a percentage compared to the total time in paranthesis. Furthermore, we show the speedup and space savings compared to a B-Tree with page size of 128 in parenthesis as part of the size and lookup column. We choose a page size of 128 as the fixed reference point as it provides the best lookup performance for B-Trees (note, that it is always easy to save space at the expense of lookup performance by simply having no index at all). The color-encoding in the speedup and size columns indicates how much faster or slower (larger or smaller) the index is against the reference point.

学习索引与B树性能对比：主要结果如图4所示。注意，B树的页面大小指的是每页的键数量，而非字节大小，实际字节大小更大。作为主要指标，我们展示了以MB为单位的大小、以纳秒为单位的总查找时间，以及执行模型（B树遍历或机器学习模型）的时间（同样以纳秒为单位，并在括号中给出其占总时间的百分比）。此外，在大小和查找列中，我们在括号内展示了与页面大小为128的B树相比的加速比和空间节省情况。我们选择页面大小为128作为固定参考点，因为它为B树提供了最佳的查找性能（注意，通过完全不使用索引，总是可以很容易地以牺牲查找性能为代价来节省空间）。加速比和大小列中的颜色编码表示该索引相对于参考点的速度快慢（大小）程度。

As can be seen, the learned index dominates the B-Tree index in almost all configurations by being up to ${1.5} - 3 \times$ faster while being up to two orders-of-magnitude smaller. Of course, B-Trees can be further compressed at the cost of CPUtime for decompressing. However, most of these optimizations are orthogonal and apply equally (if not more) to neural nets. For example, neural nets can be compressed by using 4 - or 8-bit integers instead of 32- or 64-bit floating point values to represent the model parameters (a process referred to as quantization). This level of compression can unlock additional gains for learned indexes.

可以看出，在几乎所有配置中，学习索引都优于B树索引，速度最高可达${1.5} - 3 \times$倍，而大小最多小两个数量级。当然，B树可以进一步压缩，但需要付出CPU解压缩时间的代价。然而，大多数这些优化是正交的，并且同样（如果不是更多）适用于神经网络。例如，通过使用4位或8位整数而不是32位或64位浮点值来表示模型参数（这一过程称为量化），可以对神经网络进行压缩。这种压缩水平可以为学习索引带来额外的收益。

Unsurprisingly the second stage size has a significant impact on the index size and look-up performance. Using 10,000 or more models in the second stage is particularly impressive with respect to the analysis in $\$ {2.1}$ ,as it demonstrates that our first-stage model can make a much larger jump in precision than a single node in the B-Tree. Finally, we do not report on hybrid models or other search techniques than binary search for these datasets as they did not provide significant benefit.

不出所料，第二阶段的大小对索引大小和查找性能有显著影响。根据$\$ {2.1}$中的分析，在第二阶段使用10000个或更多模型尤其令人印象深刻，因为这表明我们的第一阶段模型在精度上的提升比B树中的单个节点大得多。最后，对于这些数据集，我们没有报告混合模型或除二分查找之外的其他搜索技术，因为它们没有带来显著的好处。

Learned Index vs Alternative Baselines: In addition to the detailed evaluation of learned indexes against our read-optimized B-Trees, we also compared learned indexes against other alternative baselines, including third party implementations. In the following, we discuss some alternative baselines and compare them against learned indexes if appropriate:

学习索引与其他基线对比：除了对学习索引与我们针对读操作优化的B树进行详细评估之外，我们还将学习索引与其他替代基线进行了比较，包括第三方实现。下面，我们将讨论一些替代基线，并在适当的时候将它们与学习索引进行比较：

Histogram: B-Trees approximate the CDF of the underlying data distribution. An obvious question is whether histograms can be used as a CDF model. In principle the answer is yes, but to enable fast data access, the histogram must be a low-error approximation of the CDF. Typically this requires a large number of buckets, which makes it expensive to search the histogram itself. This is especially true, if the buckets have varying bucket boundaries to efficiently handle data skew, so that only few buckets are empty or too full. The obvious solutions to this issues would yield a B-Tree, and histograms are therefore not further discussed.

直方图：B树近似于底层数据分布的累积分布函数（CDF）。一个明显的问题是，直方图是否可以用作CDF模型。原则上答案是肯定的，但为了实现快速数据访问，直方图必须是CDF的低误差近似。通常这需要大量的桶，这使得搜索直方图本身的成本很高。如果桶具有不同的边界以有效处理数据倾斜，从而使只有少数桶为空或过满，情况尤其如此。解决这些问题的明显方法会产生一个B树，因此不再进一步讨论直方图。

Lookup-Table: A simple alternative to B-Trees are (hierarchical) lookup-tables. Often lookup-tables have a fixed size and structure (e.g., 64 slots for which each slot points to another 64 slots, etc.). The advantage of lookup-tables is that because of their fixed size they can be highly optimized using AVX instructions. We included a comparison against a 3-stage lookup table, which is constructed by taking every 64th key and putting it into an array including padding to make it a multiple of 64 . Then we repeat that process one more time over the array without padding, creating two arrays in total. To lookup a key, we use binary search on the top table followed by an AVX optimized branch-free scan [14] for the second table and the data itself. This configuration leads to the fastest lookup times compared to alternatives (e.g., using scanning on the top layer, or binary search on the 2nd array or the data).

查找表：B树的一种简单替代方案是（分层）查找表。通常，查找表具有固定的大小和结构（例如，64个槽位，每个槽位指向另外64个槽位，依此类推）。查找表的优点在于，由于其大小固定，可以使用AVX指令对其进行高度优化。我们纳入了与一个三级查找表的对比，该查找表的构建方式是：每隔64个键选取一个键，并将其放入一个数组中，同时进行填充以使数组长度为64的倍数。然后，我们对未填充的数组再重复一次该过程，总共创建两个数组。为了查找一个键，我们先在顶层表上使用二分查找，然后在第二层表和数据本身使用AVX优化的无分支扫描 [14]。与其他方案（例如，在顶层使用扫描，或在第二个数组或数据上使用二分查找）相比，这种配置的查找时间最快。

<!-- Media -->

<table><tr><td/><td>Lookup Table w/ AVX search</td><td>FAST</td><td>Fixe-Size Btree w/ interpol. search</td><td>Multivariate Learned Index</td></tr><tr><td>Time</td><td>199 ns</td><td>189 ns</td><td>280 ns</td><td>105 ns</td></tr><tr><td>Size</td><td>16.3 MB</td><td>1024 MB</td><td>1.5 MB</td><td>1.5 MB</td></tr></table>

<table><tbody><tr><td></td><td>带AVX搜索的查找表</td><td>快速</td><td>带插值搜索的固定大小B树</td><td>多元学习索引</td></tr><tr><td>时间</td><td>199纳秒</td><td>189纳秒</td><td>280纳秒</td><td>105纳秒</td></tr><tr><td>大小</td><td>16.3兆字节</td><td>1024兆字节</td><td>1.5兆字节</td><td>1.5兆字节</td></tr></tbody></table>

Figure 5: Alternative Baselines

图5：替代基线

<!-- Media -->

FAST: FAST [44] is a highly SIMD optimized data structure. We used the code from [47] for the comparison. However, it should be noted that FAST always requires to allocate memory in the power of 2 to use the branch free SIMD instructions, which can lead to significantly larger indexes.

FAST：FAST [44] 是一种高度SIMD优化的数据结构。我们使用了文献 [47] 中的代码进行比较。然而，需要注意的是，FAST始终需要以2的幂次方来分配内存，以使用无分支的SIMD指令，这可能会导致索引显著增大。

Fixed-size B-Tree & interpolation search: Finally, as proposed in a recent blog post [1] we created a fixed-height B-Tree with interpolation search. The B-Tree height is set, so that the total size of the tree is ${1.5}\mathrm{{MB}}$ ,similar to our learned model.

固定大小的B树与插值搜索：最后，正如最近一篇博客文章 [1] 所提出的，我们创建了一个带有插值搜索的固定高度B树。B树的高度被设置为使得树的总大小为 ${1.5}\mathrm{{MB}}$，这与我们的学习型模型类似。

Learned indexes without overhead: For our learned index we used a 2-staged RMI index with a multivariate linear regression model at the top and simple linear models at the bottom. We used simple automatic feature engineering for the top model by automatically creating and selecting features in the form of key, $\log \left( \text{key}\right) ,{\operatorname{key}}^{2}$ ,etc. Multivariate linear regression is an interesting alternative to NN as it is particularly well suited to fit nonlinear patterns with only a few operations. Furthermore, we implemented the learned index outside of our benchmarking framework to ensure a fair comparison.

无开销的学习型索引：对于我们的学习型索引，我们使用了一个两级的RMI（递归模型索引）索引，顶层采用多元线性回归模型，底层采用简单线性模型。我们为顶层模型使用了简单的自动特征工程，通过自动创建和选择以键、$\log \left( \text{key}\right) ,{\operatorname{key}}^{2}$ 等形式的特征。多元线性回归是神经网络（NN）的一个有趣替代方案，因为它特别适合用少量操作来拟合非线性模式。此外，我们在基准测试框架之外实现了学习型索引，以确保公平比较。

For the comparison we used the Lognormal data with a payload of an eight-byte pointer. The results can be seen in Figure 5. As can be seen for the dataset under fair conditions, learned indexes provide the best overall performance while saving significant amount of memory. It should be noted, that the FAST index is big because of the alignment requirement.

为了进行比较，我们使用了带有八字节指针负载的对数正态分布数据。结果如图5所示。从公平条件下的数据集可以看出，学习型索引在节省大量内存的同时提供了最佳的整体性能。需要注意的是，由于对齐要求，FAST索引较大。

While the results are very promising, we by no means claim that learned indexes will always be the best choice in terms of size or speed. Rather, learned indexes provide a new way to think about indexing and much more research is needed to fully understand the implications.

虽然结果非常有前景，但我们绝不声称学习型索引在大小或速度方面总是最佳选择。相反，学习型索引为索引设计提供了一种新的思考方式，需要更多的研究来充分理解其影响。

3.7.2 String Datasets. We also created a secondary index over ${10}\mathrm{M}$ non-continuous document-ids of a large web index used as part of a real product at Google to test how learned indexes perform on strings. The results for the string-based document-id dataset are shown in Figure 6, which also now includes hybrid models. In addition, we include our best model in the table, which is a non-hybrid RMI model index with quaternary search, named "Learned QS" (bottom of the table). All RMI indexes used 10,000 models on the 2nd stage and for hybrid indexes we used two thresholds, 128 and 64, as the maximum tolerated absolute error for a model before it is replaced with a B-Tree.

3.7.2 字符串数据集。我们还在谷歌一款实际产品中使用的大型网络索引的 ${10}\mathrm{M}$ 个非连续文档ID上创建了一个二级索引，以测试学习型索引在字符串上的性能。基于字符串的文档ID数据集的结果如图6所示，现在该图还包括了混合模型。此外，我们在表中列出了我们的最佳模型，它是一个带有四元搜索的非混合RMI模型索引，名为“学习型QS”（表的底部）。所有RMI索引在第二阶段使用了10,000个模型，对于混合索引，我们使用了两个阈值128和64，作为在模型被B树替代之前允许的最大绝对误差。

As can be seen, the speedups for learned indexes over B-Trees for strings are not as prominent. Part of the reason is the comparably high cost of model execution, a problem that GPU/TPUs would remove. Furthermore, searching over strings is much more expensive thus higher precision often pays off; the reason why hybrid indexes, which replace bad performing models through B-Trees, help to improve performance.

可以看出，学习型索引在字符串上相对于B树的加速效果并不那么显著。部分原因是模型执行的成本相对较高，而GPU/TPU可以解决这个问题。此外，对字符串进行搜索的成本要高得多，因此更高的精度通常是值得的；这就是混合索引（通过B树替换性能不佳的模型）有助于提高性能的原因。

<!-- Media -->

<table><tr><td/><td>Config</td><td>Size(MB)</td><td>Lookup (ns)</td><td>Model (ns)</td></tr><tr><td rowspan="4">Btree</td><td>page size: 32</td><td>13.11(4.00x)</td><td>1247 (1.03x)</td><td>643 (52%)</td></tr><tr><td>page size: 64</td><td>6.56 (2.00x)</td><td>1280 (1.01x)</td><td>500 (39%)</td></tr><tr><td>page size: 128</td><td>3.28 (1.00x)</td><td>1288 (1.00x)</td><td>377 (29%)</td></tr><tr><td>page size: 256</td><td>1.64 (0.50x)</td><td>1398 (0.92x)</td><td>330 (24%)</td></tr><tr><td rowspan="2">Learned Index</td><td>1 hidden layer</td><td>1.22 (0.37x)</td><td>1605 (0.80x)</td><td>503 (31%)</td></tr><tr><td>2 hidden layers</td><td>2.26(0.69x)</td><td>${1660}\left( {0.78x}\right)$</td><td>598 (36%)</td></tr><tr><td rowspan="4">Hybrid Index</td><td>t=128, 1 hidden layer</td><td>1.67 (0.51x)</td><td>1397 (0.92x)</td><td>472 (34%)</td></tr><tr><td>t=128, 2 hidden layers</td><td>2.33 (0.71x)</td><td>${1620}\left( {{0.80}\mathrm{x}}\right)$</td><td>591 (36%)</td></tr><tr><td>t= 64, 1 hidden layer</td><td>2.50 (0.76x)</td><td>1220 (1.06x)</td><td>440 (36%)</td></tr><tr><td>t= 64. 2 hidden lavers</td><td>2.79 (0.85x)</td><td>1447 (0.89x)</td><td>556 (38%)</td></tr><tr><td>Learned QS</td><td>1 hidden layer</td><td>1.22 (0.37x)</td><td>1155 (1.12x)</td><td>496 (43%)</td></tr></table>

<table><tbody><tr><td></td><td>配置</td><td>大小(兆字节)</td><td>查找时间(纳秒)</td><td>模型时间(纳秒)</td></tr><tr><td rowspan="4">B树</td><td>页面大小: 32</td><td>13.11(4.00x)</td><td>1247 (1.03x)</td><td>643 (52%)</td></tr><tr><td>页面大小: 64</td><td>6.56 (2.00x)</td><td>1280 (1.01x)</td><td>500 (39%)</td></tr><tr><td>页面大小: 128</td><td>3.28 (1.00x)</td><td>1288 (1.00x)</td><td>377 (29%)</td></tr><tr><td>页面大小: 256</td><td>1.64 (0.50x)</td><td>1398 (0.92x)</td><td>330 (24%)</td></tr><tr><td rowspan="2">学习型索引</td><td>1个隐藏层</td><td>1.22 (0.37x)</td><td>1605 (0.80x)</td><td>503 (31%)</td></tr><tr><td>2个隐藏层</td><td>2.26(0.69x)</td><td>${1660}\left( {0.78x}\right)$</td><td>598 (36%)</td></tr><tr><td rowspan="4">混合索引</td><td>t=128, 1个隐藏层</td><td>1.67 (0.51x)</td><td>1397 (0.92x)</td><td>472 (34%)</td></tr><tr><td>t=128, 2个隐藏层</td><td>2.33 (0.71x)</td><td>${1620}\left( {{0.80}\mathrm{x}}\right)$</td><td>591 (36%)</td></tr><tr><td>t= 64, 1个隐藏层</td><td>2.50 (0.76x)</td><td>1220 (1.06x)</td><td>440 (36%)</td></tr><tr><td>t= 64, 2个隐藏层</td><td>2.79 (0.85x)</td><td>1447 (0.89x)</td><td>556 (38%)</td></tr><tr><td>学习型QS</td><td>1个隐藏层</td><td>1.22 (0.37x)</td><td>1155 (1.12x)</td><td>496 (43%)</td></tr></tbody></table>

Figure 6: String data: Learned Index vs B-Tree

图6：字符串数据：学习索引与B树对比

<!-- Media -->

Because of the cost of searching, the different search strategies make a bigger difference. For example, the search time for a NN with 1-hidden layer and biased binary search is ${1102}\mathrm{{ns}}$ as shown in Figure 6. In contrast, our biased quaternary search with the same model only takes ${658}\mathrm{\;{ns}}$ ,a significant improvement. The reason why biased search and quaternary search perform better is that they take the model error into account.

由于搜索成本的原因，不同的搜索策略会产生较大差异。例如，如图6所示，具有1个隐藏层的神经网络（NN）和有偏二分搜索的搜索时间为${1102}\mathrm{{ns}}$。相比之下，使用相同模型的有偏四叉搜索仅需${658}\mathrm{\;{ns}}$，有显著改进。有偏搜索和四叉搜索表现更好的原因是它们考虑了模型误差。

## 4 POINT INDEX

## 4 点索引

Next to range indexes, Hash-maps for point look-ups play a similarly important role in DBMS. Conceptually Hash-maps use a hash-function to deterministically map keys to positions inside an array (see Figure 7(a)). The key challenge for any efficient Hash-map implementation is to prevent too many distinct keys from being mapped to the same position inside the Hash-map, henceforth referred to as a conflict. For example, let's assume 100M records and a Hash-map size of 100M. For a hash-function which uniformly randomizes the keys, the number of expected conflicts can be derived similarly to the birthday paradox and in expectation would be around 33% or 33M slots. For each of these conflicts, the Hash-map architecture needs to deal with this conflict. For example, separate chaining Hash-maps would create a linked-list to handle the conflict (see Figure 7(a)). However, many alternatives exist including secondary probing, using buckets with several slots, up to simultaneously using more than one hash function (e.g., as done by Cuckoo Hashing [57]).

除了范围索引之外，用于点查找的哈希映射在数据库管理系统（DBMS）中也起着类似重要的作用。从概念上讲，哈希映射使用哈希函数将键确定性地映射到数组内的位置（见图7(a)）。任何高效哈希映射实现的关键挑战是防止过多不同的键被映射到哈希映射内的同一位置，此后称为冲突。例如，假设存在1亿条记录，哈希映射大小为1亿。对于一个能均匀随机化键的哈希函数，预期冲突的数量可以类似生日悖论那样推导得出，预计约为33%，即3300万个槽位。对于这些冲突中的每一个，哈希映射架构都需要处理。例如，分离链接哈希映射会创建一个链表来处理冲突（见图7(a)）。然而，存在许多替代方法，包括二次探测、使用带有多个槽位的桶，甚至同时使用多个哈希函数（例如，布谷鸟哈希[57]的做法）。

However, regardless of the Hash-map architecture, conflicts can have a significant impact of the performance and/or storage requirement, and machine learned models might provide an alternative to reduce the number of conflicts. While the idea of learning models as a hash-function is not new, existing techniques do not take advantage of the underlying data distribution. For example, the various perfect hashing techniques [26] also try to avoid conflicts but the data structure used as part of the hash functions grow with the data size; a property learned models might not have (recall, the example of indexing all keys between 1 and ${100}\mathrm{M}$ ). To our knowledge it has not been explored if it is possible to learn models which yield more efficient point indexes.

然而，无论哈希映射架构如何，冲突都会对性能和/或存储需求产生重大影响，而机器学习模型可能提供一种减少冲突数量的替代方法。虽然将学习模型用作哈希函数的想法并不新鲜，但现有技术并未利用底层数据分布。例如，各种完美哈希技术[26]也试图避免冲突，但作为哈希函数一部分使用的数据结构会随着数据大小增长；而学习模型可能不具备这一特性（回想一下，对1到${100}\mathrm{M}$之间的所有键进行索引的例子）。据我们所知，是否有可能学习出能产生更高效点索引的模型尚未得到探索。

<!-- Media -->

<!-- figureText: (a) Traditional Hash-Map (b) Learned Hash-Map Model Hash- Function -->

<img src="https://cdn.noedgeai.com/0195c8fc-f853-7b31-b5ff-dad98e8dad50_8.jpg?x=184&y=231&w=648&h=316&r=0"/>

Figure 7: Traditional Hash-map vs Learned Hash-map

图7：传统哈希映射与学习哈希映射对比

<!-- Media -->

### 4.1 The Hash-Model Index

### 4.1 哈希模型索引

Surprisingly, learning the CDF of the key distribution is one potential way to learn a better hash function. However, in contrast to range indexes, we do not aim to store the records compactly or in strictly sorted order. Rather we can scale the CDF by the targeted size $M$ of the Hash-map and use $h\left( K\right)  = F\left( K\right)  * M$ ,with key $K$ as our hash-function. If the model $F$ perfectly learned the empirical CDF of the keys,no conflicts would exist. Furthermore, the hash-function is orthogonal to the actual Hash-map architecture and can be combined with separate chaining or any other Hash-map type.

令人惊讶的是，学习键分布的累积分布函数（CDF）是学习更好哈希函数的一种潜在方法。然而，与范围索引不同，我们的目标不是紧凑地存储记录或以严格排序的顺序存储。相反，我们可以将CDF按哈希映射的目标大小$M$进行缩放，并使用$h\left( K\right)  = F\left( K\right)  * M$，以键$K$作为我们的哈希函数。如果模型$F$完美地学习了键的经验累积分布函数，就不会存在冲突。此外，哈希函数与实际的哈希映射架构无关，可以与分离链接或任何其他类型的哈希映射结合使用。

For the model, we can again leverage the recursive model architecture from the previous section. Obviously, like before, there exists a trade-off between the size of the index and performance, which is influenced by the model and dataset.

对于模型，我们可以再次利用上一节中的递归模型架构。显然，和之前一样，索引大小和性能之间存在权衡，这受模型和数据集的影响。

Note, that how inserts, look-ups, and conflicts are handled is dependent on the Hash-map architecture. As a result, the benefits learned hash functions provide over traditional hash functions, which map keys to a uniformly distributed space depend on two key factors: (1) How accurately the model represents the observed CDF. For example, if the data is generated by a uniform distribution, a simple linear model will be able to learn the general data distribution, but the resulting hash function will not be better than any sufficiently randomized hash function. (2) Hash map architecture: depending on the architecture, implementation details, the payload (i.e., value), the conflict resolution policy, as well as how much more memory (i.e., slots) will or can be allocated, significantly influences the performance. For example, for small keys and small or no values, traditional hash functions with Cuckoo hashing will probably work well, whereas larger payloads or distributed hash maps might benefit more from avoiding conflicts, and thus from learned hash functions.

注意，插入、查找和冲突的处理方式取决于哈希映射架构。因此，学习哈希函数相对于将键映射到均匀分布空间的传统哈希函数的优势取决于两个关键因素：(1) 模型对观察到的累积分布函数的表示精度。例如，如果数据是由均匀分布生成的，一个简单的线性模型将能够学习到一般的数据分布，但得到的哈希函数不会比任何充分随机化的哈希函数更好。(2) 哈希映射架构：根据架构、实现细节、有效负载（即值）、冲突解决策略，以及将或可以分配多少额外内存（即槽位），会显著影响性能。例如，对于小键和小值或无值的情况，使用布谷鸟哈希的传统哈希函数可能效果很好，而较大的有效负载或分布式哈希映射可能从避免冲突中受益更多，从而从学习哈希函数中受益更多。

### 4.2 Results

### 4.2 结果

We evaluated the conflict rate of learned hash functions over the three integer data sets from the previous section. As our model hash-functions we used the 2-stage RMI models from the previous section with ${100}\mathrm{k}$ models on the 2nd stage and without any hidden layers. As the baseline we used a simple MurmurHash3-like hash-function and compared the number of conflicts for a table with the same number of slots as records.

我们评估了学习哈希函数在前面章节的三个整数数据集上的冲突率。作为我们的模型哈希函数，我们使用了上一节中的两阶段递归模型索引（RMI）模型，第二阶段有${100}\mathrm{k}$个模型且没有任何隐藏层。作为基线，我们使用了一个类似MurmurHash3的简单哈希函数，并比较了与记录数量相同槽位数的表的冲突数量。

As can be seen in Figure 8, the learned models can reduce the number of conflicts by up to ${77}\%$ over our datasets by learning the empirical CDF at a reasonable cost; the execution time is the same as the model execution time in Figure 4, around 25-40ns.

如图8所示，通过以合理的成本学习经验累积分布函数（empirical CDF），所学习的模型在我们的数据集上最多可将冲突数量减少${77}\%$；执行时间与图4中的模型执行时间相同，约为25 - 40纳秒。

<!-- Media -->

<table><tr><td/><td>% Conflicts Hash Map</td><td>% Conflicts Model</td><td>Reduction</td></tr><tr><td>Map Data</td><td>35.3%</td><td>07.9%</td><td>77.5%</td></tr><tr><td>Web Data</td><td>35.3%</td><td>24.7%</td><td>30.0%</td></tr><tr><td>Log Normal</td><td>35.4%</td><td>25.9%</td><td>26.7%</td></tr></table>

<table><tbody><tr><td></td><td>% 冲突哈希映射（Conflicts Hash Map）</td><td>% 冲突模型（Conflicts Model）</td><td>约简</td></tr><tr><td>地图数据（Map Data）</td><td>35.3%</td><td>07.9%</td><td>77.5%</td></tr><tr><td>网络数据（Web Data）</td><td>35.3%</td><td>24.7%</td><td>30.0%</td></tr><tr><td>对数正态（Log Normal）</td><td>35.4%</td><td>25.9%</td><td>26.7%</td></tr></tbody></table>

Figure 8: Reduction of Conflicts

图8：冲突减少情况

<!-- Media -->

How beneficial the reduction of conflicts is given the model execution time depends on the Hash-map architecture, payload, and many other factors. For example, our experiments (see Appendix B) show that for a separate chaining Hash-map architecture with 20 Byte records learned hash functions can reduce the wasted amount of storage by up to ${80}\%$ at an increase of only 13ns in latency compared to random hashing. The reason why it only increases the latency by 13ns and not 40ns is, that often fewer conflicts also yield to fewer cache misses, and thus better performance. On the other hand, for very small payloads Cuckoo-hashing with standard hash-maps probably remains the best choice. However, as we show in Appendix $\mathrm{C}$ ,for larger payloads a chained-hashmap with learned hash function can be faster than cuckoo-hashing and/or traditional randomized hashing. Finally, we see the biggest potential for distributed settings. For example, NAM-DB [74] employs a hash function to look-up data on remote machines using RDMA. Because of the extremely high cost for every conflict (i.e., every conflict requires an additional RDMA request which is in the order of micro-seconds), the model execution time is negligible and even small reductions in the conflict rate can significantly improve the overall performance. To conclude, learned hash functions are independent of the used Hash-map architecture and depending on the Hash-map architecture their complexity may or may not pay off.

在给定模型执行时间的情况下，冲突减少的益处取决于哈希映射架构、负载和许多其他因素。例如，我们的实验（见附录B）表明，对于具有20字节记录的分离链接哈希映射架构，与随机哈希相比，学习到的哈希函数可以将浪费的存储量最多减少${80}\%$，而延迟仅增加13纳秒。它只将延迟增加13纳秒而不是40纳秒的原因是，通常较少的冲突也会导致较少的缓存未命中，从而提高性能。另一方面，对于非常小的负载，使用标准哈希映射的布谷鸟哈希可能仍然是最佳选择。然而，正如我们在附录$\mathrm{C}$中所示，对于较大的负载，使用学习到的哈希函数的链式哈希映射可能比布谷鸟哈希和/或传统的随机哈希更快。最后，我们看到分布式环境具有最大的潜力。例如，NAM - DB [74]使用哈希函数通过远程直接内存访问（RDMA）在远程机器上查找数据。由于每次冲突的成本极高（即，每次冲突都需要额外的RDMA请求，其耗时约为微秒级），模型执行时间可以忽略不计，即使冲突率的小幅降低也能显著提高整体性能。总之，学习到的哈希函数与所使用的哈希映射架构无关，并且根据哈希映射架构的不同，其复杂度可能会带来收益，也可能不会。

## 5 EXISTENCE INDEX

## 5 存在索引

The last common index type of DBMS are existence indexes, most importantly Bloom filters, a space efficient probabilistic data structure to test whether an element is a member of a set. They are commonly used to determine if a key exists on cold storage. For example, Bigtable uses them to determine if a key is contained in an SSTable [23].

数据库管理系统（DBMS）的最后一种常见索引类型是存在索引，其中最重要的是布隆过滤器（Bloom filters），它是一种空间高效的概率数据结构，用于测试一个元素是否是集合的成员。它们通常用于确定某个键是否存在于冷存储中。例如，Bigtable使用它们来确定某个键是否包含在SSTable中 [23]。

Internally,Bloom filters use a bit array of size $m$ and $k$ hash functions,which each map a key to one of the $m$ array positions (see Figure9(a)). To add an element to the set, a key is fed to the $k$ hash-functions and the bits of the returned positions are set to 1 . To test if a key is a member of the set, the key is again fed into the $k$ hash functions to receive $k$ array positions. If any of the bits at those $k$ positions is 0,the key is not a member of a set. In other words, a Bloom filter does guarantee that there exists no false negatives, but has potential false positives.

在内部，布隆过滤器使用一个大小为$m$的位数组和$k$个哈希函数，每个哈希函数将一个键映射到$m$个数组位置之一（见图9(a)）。要将一个元素添加到集合中，将一个键输入到$k$个哈希函数中，并将返回位置的位设置为1。要测试一个键是否是集合的成员，再次将该键输入到$k$个哈希函数中以获得$k$个数组位置。如果这些$k$个位置中的任何一位为0，则该键不是集合的成员。换句话说，布隆过滤器保证不存在假阴性，但可能存在假阳性。

While Bloom filters are highly space-efficient, they can still occupy a significant amount of memory. For example for one billion records roughly $\approx  {1.76}$ Gigabytes are needed. For a FPR of 0.01% we would require $\approx  {2.23}$ Gigabytes. There have been several attempts to improve the efficiency of Bloom filters [52], but the general observation remains.

虽然布隆过滤器具有很高的空间效率，但它们仍然可能占用大量内存。例如，对于十亿条记录，大约需要$\approx  {1.76}$GB的内存。对于误报率（FPR）为0.01%的情况，我们需要$\approx  {2.23}$GB的内存。已经有一些尝试来提高布隆过滤器的效率 [52]，但总体情况仍然如此。

<!-- Media -->

<!-- figureText: (a) Traditional Bloom-Filter Insertion (b) Learned Bloom-Filter Insertion (c) Bloom filters as a classification problem key2 key3 No Bloom Key Model filter Model Model Yes key1 key2 key3 key1 Model -->

<img src="https://cdn.noedgeai.com/0195c8fc-f853-7b31-b5ff-dad98e8dad50_9.jpg?x=156&y=228&w=1462&h=229&r=0"/>

Figure 9: Bloom filters Architectures

图9：布隆过滤器架构

<!-- Media -->

Yet, if there is some structure to determine what is inside versus outside the set, which can be learned, it might be possible to construct more efficient representations. Interestingly, for existence indexes for database systems, the latency and space requirements are usually quite different than what we saw before. Given the high latency to access cold storage (e.g., disk or even band), we can afford more complex models while the main objective is to minimize the space for the index and the number of false positives. We outline two potential ways to build existence indexes using learned models.

然而，如果存在某种结构可以确定集合内和集合外的元素，并且这种结构是可以学习的，那么就有可能构建更高效的表示。有趣的是，对于数据库系统的存在索引，其延迟和空间要求通常与我们之前看到的情况大不相同。鉴于访问冷存储（例如磁盘甚至磁带）的延迟很高，我们可以使用更复杂的模型，而主要目标是最小化索引的空间和假阳性的数量。我们概述了两种使用学习模型构建存在索引的潜在方法。

### 5.1 Learned Bloom filters

### 5.1 学习型布隆过滤器

While both range and point indexes learn the distribution of keys, existence indexes need to learn a function that separates keys from everything else. Stated differently, a good hash function for a point index is one with few collisions among keys, whereas a good hash function for a Bloom filter would be one that has lots of collisions among keys and lots of collisions among non-keys, but few collisions of keys and non-keys. We consider below how to learn such a function $f$ and how to incorporate it into an existence index.

范围索引和点索引都学习键的分布，而存在索引需要学习一个将键与其他所有内容分开的函数。换句话说，对于点索引来说，一个好的哈希函数是在键之间冲突较少的函数，而对于布隆过滤器来说，一个好的哈希函数是在键之间有大量冲突、在非键之间也有大量冲突，但键和非键之间冲突较少的函数。我们下面将考虑如何学习这样一个函数$f$，以及如何将其纳入存在索引中。

While traditional Bloom filters guarantee a false negative rate (FNR) of zero and a specific false positive rate (FPR) for any set of queries chosen a-priori [22], we follow the notion that we want to provide a specific FPR for realistic queries in particular while maintaining a FNR of zero. That is, we measure the FPR over a heldout dataset of queries, as is common in evaluating ML systems [30]. While these definitions differ, we believe the assumption that we can observe the distribution of queries, e.g., from historical logs, holds in many applications, especially within databases ${}^{5}$ .

虽然传统的布隆过滤器保证对于任何预先选择的查询集，假阴性率（FNR）为零，并且有特定的假阳性率（FPR） [22]，但我们遵循这样的理念，即我们希望为实际的查询提供特定的FPR，同时保持FNR为零。也就是说，我们像评估机器学习系统时常见的那样 [30]，在一个保留的查询数据集上测量FPR。虽然这些定义有所不同，但我们相信，在许多应用中，特别是在数据库中 ${}^{5}$，我们可以观察到查询分布的假设是成立的，例如从历史日志中获取。

Traditionally, existence indexes make no use of the distribution of keys nor how they differ from non-keys, but learned Bloom filters can. For example, if our database included all integers $x$ for $0 \leq  x < n$ ,the existence index could be computed in constant time and with almost no memory footprint by just computing $f\left( x\right)  \equiv  \mathbb{1}\left\lbrack  {0 \leq  x < n}\right\rbrack$ .

传统上，存在索引不利用键的分布，也不考虑它们与非键的差异，但基于学习的布隆过滤器（Bloom filters）可以做到。例如，如果我们的数据库包含所有整数 $x$（其中 $0 \leq  x < n$ ），那么只需计算 $f\left( x\right)  \equiv  \mathbb{1}\left\lbrack  {0 \leq  x < n}\right\rbrack$ ，就可以在常数时间内且几乎不占用内存的情况下计算出存在索引。

In considering the data distribution for ML purposes, we must consider a dataset of non-keys. In this work, we consider the case where non-keys come from observable historical queries and we assume that future queries come from the same distribution as historical queries. When this assumption does not hold, one could use randomly generated keys, non-keys generated by a machine learning model [34], importance weighting to directly address covariate shift [18], or adversarial training for robustness [65]; we leave this as future work. We denote the set of keys by $\mathcal{K}$ and the set of non-keys by $\mathcal{U}$ .

出于机器学习的目的考虑数据分布时，我们必须考虑一个非键的数据集。在这项工作中，我们考虑非键来自可观察的历史查询的情况，并假设未来的查询与历史查询来自相同的分布。当这个假设不成立时，我们可以使用随机生成的键、由机器学习模型生成的非键 [34]、直接解决协变量偏移的重要性加权方法 [18]，或者进行对抗训练以增强鲁棒性 [65]；我们将这些留作未来的工作。我们用 $\mathcal{K}$ 表示键的集合，用 $\mathcal{U}$ 表示非键的集合。

5.1.1 Bloom filters as a Classification Problem. One way to frame the existence index is as a binary probabilistic classification task. That is,we want to learn a model $f$ that can predict if a query $x$ is a key or non-key. For example,for strings we can train a recurrent neural network (RNN) or convolutional neural network $\left( \mathrm{{CNN}}\right) \left\lbrack  {{37},{64}}\right\rbrack$ with $\mathcal{D} = \left\{  {\left( {{x}_{i},{y}_{i} = 1}\right)  \mid  {x}_{i} \in  }\right.$ $\mathcal{K}\}  \cup  \left\{  {\left( {{x}_{i},{y}_{i} = 0}\right)  \mid  {x}_{i} \in  \mathcal{U}}\right\}$ . Because this is a binary classification task, our neural network has a sigmoid activation to produce a probability and is trained to minimize the log loss: $L = \mathop{\sum }\limits_{{\left( {x,y}\right)  \in  \mathcal{D}}}y\log f\left( x\right)  + \left( {1 - y}\right) \log \left( {1 - f\left( x\right) }\right) .$

5.1.1 将布隆过滤器视为分类问题。将存在索引构建为一个二元概率分类任务是一种可行的方法。也就是说，我们希望学习一个模型 $f$ ，它能够预测一个查询 $x$ 是键还是非键。例如，对于字符串，我们可以使用 $\mathcal{D} = \left\{  {\left( {{x}_{i},{y}_{i} = 1}\right)  \mid  {x}_{i} \in  }\right.$ $\mathcal{K}\}  \cup  \left\{  {\left( {{x}_{i},{y}_{i} = 0}\right)  \mid  {x}_{i} \in  \mathcal{U}}\right\}$ 来训练一个循环神经网络（RNN）或卷积神经网络 $\left( \mathrm{{CNN}}\right) \left\lbrack  {{37},{64}}\right\rbrack$ 。由于这是一个二元分类任务，我们的神经网络使用 sigmoid 激活函数来产生一个概率，并通过训练来最小化对数损失：$L = \mathop{\sum }\limits_{{\left( {x,y}\right)  \in  \mathcal{D}}}y\log f\left( x\right)  + \left( {1 - y}\right) \log \left( {1 - f\left( x\right) }\right) .$

The output of $f\left( x\right)$ can be interpreted as the probability that $x$ is a key in our database. Thus,we can turn the model into an existence index by choosing a threshold $\tau$ above which we will assume that the key exists in our database. Unlike Bloom filters, our model will likely have a non-zero FPR and FNR; in fact, as the FPR goes down, the FNR will go up. In order to preserve the no false negatives constraint of existence indexes, we create an overflow Bloom filter. That is,we consider ${\mathcal{K}}_{\tau }^{ - } =$ $\{ x \in  \mathcal{K} \mid  f\left( x\right)  < \tau \}$ to be the set of false negatives from $f$ and create a Bloom filter for this subset of keys. We can then run our existence index as in Figure 9(c): if $f\left( x\right)  \geq  \tau$ ,the key is believed to exist; otherwise, check the overflow Bloom filter.

$f\left( x\right)$ 的输出可以解释为 $x$ 是我们数据库中一个键的概率。因此，我们可以通过选择一个阈值 $\tau$ ，将该模型转换为一个存在索引，当概率高于这个阈值时，我们就假设该键存在于我们的数据库中。与布隆过滤器不同，我们的模型可能会有非零的误报率（FPR）和漏报率（FNR）；实际上，随着误报率的降低，漏报率会升高。为了保留存在索引无漏报的约束条件，我们创建了一个溢出布隆过滤器。也就是说，我们将 ${\mathcal{K}}_{\tau }^{ - } =$ $\{ x \in  \mathcal{K} \mid  f\left( x\right)  < \tau \}$ 视为 $f$ 的漏报集合，并为这个键的子集创建一个布隆过滤器。然后，我们可以按照图 9(c) 所示的方式运行我们的存在索引：如果 $f\left( x\right)  \geq  \tau$ ，则认为该键存在；否则，检查溢出布隆过滤器。

One question is how to set $\tau$ so that our learned Bloom filter has the desired FPR ${p}^{ * }$ . We denote the FPR of our model by ${\mathrm{{FPR}}}_{\tau } \equiv  \frac{\mathop{\sum }\limits_{{x \in  \widetilde{\mathcal{U}}}}\mathbb{1}\left( {f\left( x\right)  > \tau }\right) }{\left| \widetilde{\mathcal{U}}\right| }$ where $\widetilde{\mathcal{U}}$ is a held-out set of non-keys. We denote the FPR of our overflow Bloom filter by ${\mathrm{{FPR}}}_{B}$ . The overall FPR of our system therefore is ${\mathrm{{FPR}}}_{O} = {\mathrm{{FPR}}}_{\tau } + (1 -$ $\left. {\mathrm{{FPR}}}_{\tau }\right) {\mathrm{{FPR}}}_{B}$ [53]. For simplicity,we set ${\mathrm{{FPR}}}_{\tau } = {\mathrm{{FPR}}}_{B} = \frac{{p}^{ * }}{2}$ so that ${\mathrm{{FPR}}}_{O} \leq  {p}^{ * }$ . We tune $\tau$ to achieve this FPR on $\widetilde{\mathcal{U}}$ .

一个问题是如何设置$\tau$，以使我们学习得到的布隆过滤器（Bloom filter）具有期望的误判率（False Positive Rate，FPR）${p}^{ * }$。我们用${\mathrm{{FPR}}}_{\tau } \equiv  \frac{\mathop{\sum }\limits_{{x \in  \widetilde{\mathcal{U}}}}\mathbb{1}\left( {f\left( x\right)  > \tau }\right) }{\left| \widetilde{\mathcal{U}}\right| }$表示我们模型的误判率，其中$\widetilde{\mathcal{U}}$是一个保留的非键（non - key）集合。我们用${\mathrm{{FPR}}}_{B}$表示我们的溢出布隆过滤器的误判率。因此，我们系统的总体误判率为${\mathrm{{FPR}}}_{O} = {\mathrm{{FPR}}}_{\tau } + (1 -$ $\left. {\mathrm{{FPR}}}_{\tau }\right) {\mathrm{{FPR}}}_{B}$ [53]。为了简单起见，我们设置${\mathrm{{FPR}}}_{\tau } = {\mathrm{{FPR}}}_{B} = \frac{{p}^{ * }}{2}$，使得${\mathrm{{FPR}}}_{O} \leq  {p}^{ * }$。我们调整$\tau$以在$\widetilde{\mathcal{U}}$上实现这个误判率。

This setup is effective in that the learned model can be fairly small relative to the size of the data. Further, because Bloom filters scale with the size of key set, the overflow Bloom filter will scale with the FNR. We will see experimentally that this combination is effective in decreasing the memory footprint of the existence index. Finally, the learned model computation can benefit from machine learning accelerators, whereas traditional Bloom filters tend to be heavily dependent on the random access latency of the memory system.

这种设置是有效的，因为相对于数据的大小，学习得到的模型可以相当小。此外，由于布隆过滤器的规模与键集的大小成比例，溢出布隆过滤器的规模将与漏判率（False Negative Rate，FNR）成比例。我们将通过实验看到，这种组合在减少存在索引的内存占用方面是有效的。最后，学习得到的模型计算可以受益于机器学习加速器，而传统的布隆过滤器往往严重依赖于内存系统的随机访问延迟。

5.1.2 Bloom filters with Model-Hashes. An alternative approach to building existence indexes is to learn a hash function with the goal to maximize collisions among keys and among non-keys while minimizing collisions of keys and non-keys. Interestingly, we can use the same probabilistic classification model as before to achieve that. That is, we can create a hash function $d$ ,which maps $f$ to a bit array of size $m$ by scaling its output as $d = \lfloor f\left( x\right)  * m\rfloor$ As such,we can use $d$ as a hash function just like any other in a Bloom filter. This has the advantage of $f$ being trained to map most keys to the higher range of bit positions and non-keys to the lower range of bit positions (see Figure9(b)). A more detailed explanation of the approach is given in Appendix E.

5.1.2 带有模型哈希的布隆过滤器。构建存在索引的另一种方法是学习一个哈希函数，其目标是最大化键之间和非键之间的冲突，同时最小化键和非键之间的冲突。有趣的是，我们可以使用与之前相同的概率分类模型来实现这一点。也就是说，我们可以创建一个哈希函数$d$，它通过将其输出按$d = \lfloor f\left( x\right)  * m\rfloor$进行缩放，将$f$映射到一个大小为$m$的位阵列。因此，我们可以像在布隆过滤器中使用任何其他哈希函数一样使用$d$。这样做的优点是，$f$经过训练，可以将大多数键映射到位位置的较高范围，将非键映射到位位置的较低范围（见图9(b)）。附录E中给出了该方法的更详细解释。

---

<!-- Footnote -->

${}^{5}$ We would like to thank Michael Mitzenmacher for valuable conversations in articulating the relationship between these definitions as well as improving the overall chapter through his insightful comments.

${}^{5}$我们要感谢迈克尔·米岑马赫（Michael Mitzenmacher）进行了有价值的讨论，阐明了这些定义之间的关系，并通过他富有洞察力的评论改进了整个章节。

<!-- Footnote -->

---

### 5.2 Results

### 5.2 结果

In order to test this idea experimentally, we explore the application of an existence index for keeping track of blacklisted phishing URLs. We consider data from Google's transparency report as our set of keys to keep track of. This dataset consists of ${1.7}\mathrm{M}$ unique URLs. We use a negative set that is a mixture of random (valid) URLs and whitelisted URLs that could be mistaken for phishing pages. We split our negative set randomly into train, validation and test sets. We train a character-level RNN (GRU [24], in particular) to predict which set a URL belongs to; we set $\tau$ based on the validation set and also report the FPR on the test set.

为了通过实验验证这一想法，我们探索了使用存在索引来跟踪列入黑名单的网络钓鱼URL的应用。我们将谷歌透明度报告中的数据作为我们要跟踪的键集。这个数据集包含${1.7}\mathrm{M}$个唯一的URL。我们使用一个负样本集，它是随机（有效）URL和可能被误认为是网络钓鱼页面的白名单URL的混合。我们将负样本集随机划分为训练集、验证集和测试集。我们训练一个字符级的循环神经网络（特别是门控循环单元（GRU）[24]）来预测一个URL属于哪个集合；我们根据验证集设置$\tau$，并报告测试集上的误判率。

A normal Bloom filter with a desired 1% FPR requires 2.04MB. We consider a 16-dimensional GRU with a 32-dimensional embedding for each character; this model is ${0.0259}\mathrm{{MB}}$ . When building a comparable learned index,we set $\tau$ for ${0.5}\%$ FPR on the validation set; this gives a FNR of 55%. (The FPR on the test set is 0.4976%, validating the chosen threshold.) As described above, the size of our Bloom filter scales with the FNR. Thus, we find that our model plus the spillover Bloom filter uses 1.31MB,a 36% reduction in size. If we want to enforce an overall FPR of 0.1%, we have a FNR of 76%, which brings the total Bloom filter size down from 3.06MB to 2.59MB, a 15% reduction in memory. We observe this general relationship in Figure 10. Interestingly, we see how different size models balance the accuracy vs. memory trade-off differently.

一个期望误报率（FPR）为1%的普通布隆过滤器需要2.04MB的空间。我们考虑一个16维的门控循环单元（GRU），每个字符采用32维的嵌入；该模型为${0.0259}\mathrm{{MB}}$。在构建一个可比较的学习索引时，我们在验证集上针对${0.5}\%$的误报率设置$\tau$；这会产生55%的漏报率（FNR）。（测试集上的误报率为0.4976%，验证了所选的阈值。）如上所述，我们的布隆过滤器的大小与漏报率相关。因此，我们发现我们的模型加上溢出布隆过滤器使用1.31MB的空间，大小减少了36%。如果我们想将总体误报率控制在0.1%，则漏报率为76%，这会使布隆过滤器的总大小从3.06MB降至2.59MB，内存使用减少了15%。我们在图10中观察到了这种普遍关系。有趣的是，我们看到不同大小的模型如何以不同的方式平衡准确性与内存占用之间的权衡。

We consider briefly the case where there is covariate shift in our query distribution that we have not addressed in the model. When using validation and test sets with only random URLs we find that we can save ${60}\%$ over a Bloom filter with a FPR of 1%. When using validation and test sets with only the whitelisted URLs we find that we can save ${21}\%$ over a Bloom filter with a FPR of $1\%$ . Ultimately,the choice of negative set is application specific and covariate shift could be more directly addressed, but these experiments are intended to give intuition for how the approach adapts to different situations.

我们简要考虑一下在查询分布中存在协变量偏移（covariate shift）的情况，而我们在模型中并未处理这种偏移。当使用仅包含随机URL的验证集和测试集时，我们发现与误报率为1%的布隆过滤器相比，我们可以节省${60}\%$的空间。当使用仅包含白名单URL的验证集和测试集时，我们发现与误报率为$1\%$的布隆过滤器相比，我们可以节省${21}\%$的空间。最终，负样本集的选择取决于具体应用，并且协变量偏移可以更直接地处理，但这些实验旨在让我们直观了解该方法如何适应不同的情况。

Clearly, the more accurate our model is, the better the savings in Bloom filter size. One interesting property of this is that there is no reason that our model needs to use the same features as the Bloom filter. For example, significant research has worked on using ML to predict if a webpage is a phish-ing page $\left\lbrack  {{10},{15}}\right\rbrack$ . Additional features like WHOIS data or IP information could be incorporated in the model, improving accuracy, decreasing Bloom filter size, and keeping the property of no false negatives.

显然，我们的模型越准确，布隆过滤器的大小节省就越多。这其中一个有趣的特性是，我们的模型没有必要使用与布隆过滤器相同的特征。例如，有大量研究致力于使用机器学习来预测网页是否为钓鱼页面$\left\lbrack  {{10},{15}}\right\rbrack$。诸如WHOIS数据或IP信息等额外特征可以融入模型中，从而提高准确性、减小布隆过滤器的大小，并保持无假阴性的特性。

Further, we give additional results following the approach in Section 5.1.2 in Appendix E.

此外，我们按照附录E中第5.1.2节的方法给出了额外的结果。

<!-- Media -->

<!-- figureText: emory Footprint (Megabytes) 5 BloomFilter W=128,E=32 W=32,E=32 W=16,E=32 1.0 1.5 False Positive Rate (%) 4 3 2 0 0.5 -->

<img src="https://cdn.noedgeai.com/0195c8fc-f853-7b31-b5ff-dad98e8dad50_10.jpg?x=974&y=239&w=616&h=462&r=0"/>

Figure 10: Learned Bloom filter improves memory footprint at a wide range of FPRs. (Here $W$ is the RNN width and $E$ is the embedding size for each character.)

图10：学习型布隆过滤器在广泛的误报率范围内改善了内存占用。（这里$W$是循环神经网络（RNN）的宽度，$E$是每个字符的嵌入大小。）

<!-- Media -->

## 6 RELATED WORK

## 6 相关工作

The idea of learned indexes builds upon a wide range of research in machine learning and indexing techniques. In the following, we highlight the most important related areas.

学习索引的思想建立在机器学习和索引技术的广泛研究基础之上。在下面，我们将重点介绍最重要的相关领域。

B-Trees and variants: Over the last decades a variety of different index structures have been proposed [36], such as B+-trees [17] for disk based systems and T-trees [46] or balanced/red-black trees $\left\lbrack  {{16},{20}}\right\rbrack$ for in-memory systems. As the original main-memory trees had poor cache behavior, several cache conscious B+-tree variants were proposed, such as the CSB+-tree [58]. Similarly, there has been work on making use of SIMD instructions such as FAST [44] or even taking advantage of GPUs $\left\lbrack  {{43},{44},{61}}\right\rbrack$ . Moreover,many of these (in-memory) indexes are able to reduce their storage-needs by using offsets rather than pointers between nodes. There exists also a vast array of research on index structures for text, such as tries/radix-trees $\left\lbrack  {{19},{31},{45}}\right\rbrack$ ,or other exotic index structures, which combine ideas from B-Trees and tries [48].

B树及其变体：在过去的几十年中，人们提出了各种不同的索引结构[36]，例如用于基于磁盘的系统的B + 树[17]，以及用于内存系统的T树[46]或平衡/红黑树$\left\lbrack  {{16},{20}}\right\rbrack$。由于最初的主存树的缓存性能较差，人们提出了几种考虑缓存的B + 树变体，例如CSB + 树[58]。同样，也有利用单指令多数据（SIMD）指令的工作，如FAST [44]，甚至还有利用图形处理器（GPU）的工作$\left\lbrack  {{43},{44},{61}}\right\rbrack$。此外，许多（内存中的）索引能够通过使用节点之间的偏移量而非指针来减少存储需求。此外，还有大量关于文本索引结构的研究，如字典树/基数树$\left\lbrack  {{19},{31},{45}}\right\rbrack$，或者其他结合了B树和字典树思想的奇特索引结构[48]。

However, all of these approaches are orthogonal to the idea of learned indexes as none of them learn from the data distribution to achieve a more compact index representation or performance gains. At the same time, like with our hybrid indexes, it might be possible to more tightly integrate the existing hardware-conscious index strategies with learned models for further performance gains.

然而，所有这些方法都与学习索引的思想正交，因为它们都没有从数据分布中学习以实现更紧凑的索引表示或性能提升。同时，就像我们的混合索引一样，有可能将现有的考虑硬件的索引策略与学习模型更紧密地集成，以进一步提高性能。

Since B+-trees consume significant memory, there has also been a lot of work in compressing indexes, such as prefix/suffix truncation, dictionary compression, key normalization [33, 36, 55], or hybrid hot/cold indexes [75]. However, we presented a radical different way to compress indexes, which-dependent on the data distribution-is able to achieve orders-of-magnitude smaller indexes and faster look-up times and potentially even changes the storage complexity class (e.g., $O\left( n\right)$ to $O\left( 1\right)$ ). Interestingly though,some of the existing compression techniques are complimentary to our approach and could help to further improve the efficiency. For example, dictionary compression can be seen as a form of embedding (i.e., representing a string as a unique integer).

由于B + 树会消耗大量内存，因此在索引压缩方面也有很多工作，例如前缀/后缀截断、字典压缩、键归一化[33, 36, 55]，或者冷热混合索引[75]。然而，我们提出了一种截然不同的索引压缩方法，该方法依赖于数据分布，能够使索引缩小几个数量级，加快查找速度，甚至有可能改变存储复杂度类别（例如，从$O\left( n\right)$变为$O\left( 1\right)$）。不过有趣的是，一些现有的压缩技术与我们的方法是互补的，可以帮助进一步提高效率。例如，字典压缩可以看作是一种嵌入形式（即，将字符串表示为唯一整数）。

Probably most related to this paper are A-Trees [32], BF-Trees [13], and B-Tree interpolation search [35]. BF-Trees uses a B+-tree to store information about a region of the dataset, but leaf nodes are Bloom filters and do not approximate the CDF. In contrast, A-Trees use piece-wise linear functions to reduce the number of leaf-nodes in a B-Tree, and [35] proposes to use interpolation search within a B-Tree page. However, learned indexes go much further and propose to replace the entire index structure using learned models.

可能与本文最相关的是A - 树[32]、BF - 树[13]和B - 树插值搜索[35]。BF - 树使用B + 树来存储数据集某个区域的信息，但叶节点是布隆过滤器，并不近似累积分布函数（CDF）。相比之下，A - 树使用分段线性函数来减少B - 树中叶节点的数量，而文献[35]提出在B - 树页面内使用插值搜索。然而，学习型索引更进一步，提议使用学习模型来替代整个索引结构。

Finally, sparse indexes like Hippo [73], Block Range Indexes [63], and Small Materialized Aggregates (SMAs) [54] all store information about value ranges but again do not take advantage of the underlying properties of the data distribution.

最后，像Hippo[73]、块范围索引[63]和小型物化聚合（SMA）[54]这样的稀疏索引都存储了关于值范围的信息，但同样没有利用数据分布的底层属性。

Learning Hash Functions for ANN Indexes: There has been a lot of research on learning hash functions $\lbrack {49},{59},{67}$ , 68]. Most notably, there has been work on learning locality-sensitive hash (LSH) functions to build Approximate Nearest Neighborhood (ANN) indexes. For example, $\left\lbrack  {{40},{66},{68}}\right\rbrack$ explore the use of neural networks as a hash function, whereas [69] even tries to preserve the order of the multi-dimensional input space. However, the general goal of LSH is to group similar items into buckets to support nearest neighborhood queries, usually involving learning approximate similarity measures in high-dimensional input space using some variant of hamming distances. There is no direct way to adapt previous approaches to learn the fundamental data structures we consider, and it is not clear whether they can be adapted.

为近似最近邻（ANN）索引学习哈希函数：关于学习哈希函数$\lbrack {49},{59},{67}$，68]已有大量研究。最值得注意的是，有研究致力于学习局部敏感哈希（LSH）函数以构建近似最近邻（ANN）索引。例如，$\left\lbrack  {{40},{66},{68}}\right\rbrack$探索了使用神经网络作为哈希函数，而文献[69]甚至试图保留多维输入空间的顺序。然而，LSH的总体目标是将相似的项分组到桶中以支持最近邻查询，通常涉及使用某种汉明距离变体在高维输入空间中学习近似相似度度量。没有直接的方法可以将以前的方法应用于学习我们所考虑的基本数据结构，而且不清楚它们是否可以被应用。

Perfect Hashing: Perfect hashing [26] is very related to our use of models for Hash-maps. Like our CDF models, perfect hashing tries to avoid conflicts. However, in all approaches of which we are aware, learning techniques have not been considered, and the size of the function grows with the size of the data. In contrast, learned hash functions can be independent of the size. For example, a linear model for mapping every other integer between 0 and ${200}\mathrm{M}$ would not create any conflicts and is independent of the size of the data. In addition, perfect hashing is also not useful for B-Trees or Bloom filters.

完美哈希：完美哈希[26]与我们在哈希映射中使用模型密切相关。与我们的累积分布函数（CDF）模型一样，完美哈希试图避免冲突。然而，据我们所知，在所有方法中都没有考虑学习技术，并且函数的大小会随着数据大小的增加而增长。相比之下，学习型哈希函数可以与数据大小无关。例如，将0到${200}\mathrm{M}$之间的每隔一个整数进行映射的线性模型不会产生任何冲突，并且与数据大小无关。此外，完美哈希对于B - 树或布隆过滤器也没有用处。

Bloom filters: Finally, our existence indexes directly builds upon the existing work in Bloom filters $\left\lbrack  {{11},{29}}\right\rbrack$ . Yet again our work takes a different perspective on the problem by proposing a Bloom filter enhanced classification model or using models as special hash functions with a very different optimization goal than the hash-models we created for Hash-maps.

布隆过滤器：最后，我们的存在性索引直接基于现有的布隆过滤器工作$\left\lbrack  {{11},{29}}\right\rbrack$。不过，我们的工作从不同的角度看待这个问题，提出了一种布隆过滤器增强的分类模型，或者将模型用作特殊的哈希函数，其优化目标与我们为哈希映射创建的哈希模型截然不同。

Succinct Data Structures: There exists an interesting connection between learned indexes and succinct data structures, especially rank-select dictionaries such as wavelet trees [38, 39]. However, many succinct data structures focus on H0 entropy (i.e., the number of bits that are necessary to encode each element in the index), whereas learned indexes try to learn the underlying data distribution to predict the position of each element. Thus, learned indexes might achieve a higher compression rate than $\mathrm{H}0$ entropy potentially at the cost of slower operations. Furthermore, succinct data structures normally have to be carefully constructed for each use case, whereas learned indexes "automate" this process through machine learning. Yet, succinct data structures might provide a framework to further study learned indexes.

简洁数据结构：学习型索引和简洁数据结构之间存在着有趣的联系，特别是像小波树[38, 39]这样的秩选择字典。然而，许多简洁数据结构关注H0熵（即，对索引中的每个元素进行编码所需的比特数），而学习型索引试图学习底层数据分布以预测每个元素的位置。因此，学习型索引可能比$\mathrm{H}0$熵实现更高的压缩率，但可能会以操作速度变慢为代价。此外，简洁数据结构通常必须针对每个用例进行精心构建，而学习型索引通过机器学习“自动化”了这个过程。不过，简洁数据结构可能为进一步研究学习型索引提供一个框架。

Modeling CDFs: Our models for both range and point indexes are closely tied to models of the CDF. Estimating the CDF is non-trivial and has been studied in the machine learning community [50] with a few applications such as ranking [42]. How to most effectively model the CDF is still an open question worth further investigation.

建模累积分布函数（CDF）：我们用于范围索引和点索引的模型都与累积分布函数（CDF）模型密切相关。估计累积分布函数并非易事，机器学习社区已经对此进行了研究[50]，并在一些应用中有所体现，如排序[42]。如何最有效地对累积分布函数进行建模仍然是一个值得进一步研究的开放问题。

Mixture of Experts: Our RMI architecture follows a long line of research on building experts for subsets of the data [51]. With the growth of neural networks, this has become more common and demonstrated increased usefulness [62]. As we see in our setting, it nicely lets us to decouple model size and model computation, enabling more complex models that are not more expensive to execute.

专家混合模型：我们的RMI（递归模型索引）架构遵循了为数据子集构建专家模型的长期研究思路[51]。随着神经网络的发展，这种方法变得更为常见，并且显示出了更大的实用性[62]。在我们的场景中可以看到，它很好地让我们能够将模型大小和模型计算解耦，从而能够使用更复杂但执行成本不会更高的模型。

## 7 CONCLUSION AND FUTURE WORK

## 7 结论与未来工作

We have shown that learned indexes can provide significant benefits by utilizing the distribution of data being indexed. This opens the door to many interesting research questions.

我们已经表明，学习型索引可以通过利用被索引数据的分布来提供显著的优势。这为许多有趣的研究问题打开了大门。

Other ML Models: While our focus was on linear models and neural nets with mixture of experts, there exist many other ML model types and ways to combine them with traditional data structures, which are worth exploring.

其他机器学习模型：虽然我们的重点是线性模型和带有专家混合的神经网络，但还存在许多其他类型的机器学习模型，以及将它们与传统数据结构相结合的方法，这些都值得探索。

Multi-Dimensional Indexes: Arguably the most exciting research direction for the idea of learned indexes is to extend them to multi-dimensional indexes. Models, especially NNs, are extremely good at capturing complex high-dimensional relationships. Ideally, this model would be able to estimate the position of all records filtered by any combination of attributes.

多维索引：可以说，学习型索引理念最令人兴奋的研究方向是将其扩展到多维索引。模型，尤其是神经网络，非常擅长捕捉复杂的高维关系。理想情况下，这个模型应该能够估计由任何属性组合过滤后的所有记录的位置。

Beyond Indexing: Learned Algorithms Maybe surprisingly, a CDF model has also the potential to speed-up sorting and joins, not just indexes. For instance, the basic idea to speedup sorting is to use an existing CDF model $F$ to put the records roughly in sorted order and then correct the nearly perfectly sorted data, for example, with insertion sort.

超越索引：学习型算法 也许令人惊讶的是，累积分布函数（CDF）模型不仅有可能加速索引，还有可能加速排序和连接操作。例如，加速排序的基本思路是使用现有的CDF模型$F$将记录大致按排序顺序排列，然后对几乎已经排好序的数据进行修正，比如使用插入排序。

GPU/TPUs Finally, as mentioned several times throughout this paper, GPU/TPUs will make the idea of learned indexes even more valuable. At the same time, GPU/TPUs also have their own challenges, most importantly the high invocation latency. While it is reasonable to assume that probably all learned indexes will fit on the GPU/TPU because of the exceptional compression ratio as shown before, it still requires 2-3 micro-seconds to invoke any operation on them. At the same time, the integration of machine learning accelerators with the CPU is getting better $\left\lbrack  {4,6}\right\rbrack$ and with techniques like batching requests the cost of invocation can be amortized, so that we do not believe the invocation latency is a real obstacle.

GPU/TPU 最后，正如本文多次提到的，GPU/TPU将使学习型索引的理念更具价值。同时，GPU/TPU也有自身的挑战，最重要的是高调用延迟。虽然由于之前展示的出色压缩比，可以合理假设可能所有学习型索引都能放在GPU/TPU上，但在它们上面调用任何操作仍然需要2 - 3微秒。同时，机器学习加速器与CPU的集成正在不断改进$\left\lbrack  {4,6}\right\rbrack$，并且通过批量请求等技术可以分摊调用成本，因此我们认为调用延迟并不是真正的障碍。

In summary, we have demonstrated that machine learned models have the potential to provide significant benefits over state-of-the-art indexes, and we believe this is a fruitful direction for future research.

综上所述，我们已经证明机器学习模型有可能比最先进的索引提供显著的优势，并且我们相信这是未来研究的一个富有成果的方向。

Acknowledgements: We would like to thank Michael Mitzenmacher, Chris Olston, Jonathan Bischof and many others at Google for their helpful feedback during the preparation of this paper.

致谢：我们要感谢迈克尔·米岑马赫（Michael Mitzenmacher）、克里斯·奥尔森（Chris Olston）、乔纳森·比肖夫（Jonathan Bischof）以及谷歌的许多其他人，感谢他们在本文准备过程中提供的有益反馈。

## REFERENCES

## 参考文献

[1] Database architects blog: The case for b-tree index structures. http://databasearchitects.blogspot.de/2017/12/ the-case-for-b-tree-index-structures.html.

[2] Google's sparsehash documentation. https://github.com/sparsehash/ sparsehash/blob/master/src/sparsehash/sparse_hash_map.

[3] An in-depth look at google's first tensor processing unit (tpu). https://cloud.google.com/blog/big-data/2017/05/ an-in-depth-look-at-googles-first-tensor-processing-unit-tpu.

[4] Intel Xeon Phi. https://www.intel.com/content/www/us/en/products/ processors/xeon-phi/xeon-phi-processors.html.

[5] Moore Law is Dead but GPU will get ${1000}\mathrm{X}$ faster by 2025. https://www.nextbigfuture.com/2017/06/ moore-law-is-dead-but-gpu-will-get-1000x-faster-by-2025.html.

[6] NVIDIA NVLink High-Speed Interconnect. http://www.nvidia.com/object/ nvlink.html.

[7] Stanford DAWN cuckoo hashing. https://github.com/stanford-futuredata/ index-baselines.

[8] Trying to speed up binary search. http://databasearchitects.blogspot.com/ 2015/09/trying-to-speed-up-binary-search.html.

[9] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghe-mawat, G. Irving, M. Isard, et al. Tensorflow: A system for large-scale machine learning. In OSDI, volume 16, pages 265-283, 2016.

[10] S. Abu-Nimeh, D. Nappa, X. Wang, and S. Nair. A comparison of machine learning techniques for phishing detection. In eCrime, pages 60-69, 2007.

[11] K. Alexiou, D. Kossmann, and P.-A. Larson. Adaptive range filters for cold data: Avoiding trips to siberia. Proc. VLDB Endow., 6(14):1714-1725, Sept. 2013.

[12] M. Armbrust, A. Fox, D. A. Patterson, N. Lanham, B. Trushkowsky, J. Trutna, and H. Oh. SCADS: scale-independent storage for social computing applications. In ${CIDR},{2009}$ .

[13] M. Athanassoulis and A. Ailamaki. BF-tree: Approximate Tree Indexing. In ${VLDB}$ ,pages 1881-1892,2014.

[14] Performance comparison: linear search vs binary search. https://dirtyhandscoding.wordpress.com/2017/08/25/ performance-comparison-linear-search-vs-binary-search/.

[15] R. B. Basnet, S. Mukkamala, and A. H. Sung. Detection of phishing attacks: A machine learning approach. Soft Computing Applications in Industry, 226:373-383, 2008.

[16] R. Bayer. Symmetric binary b-trees: Data structure and maintenance algorithms. Acta Inf., 1(4):290-306, Dec. 1972.

[17] R. Bayer and E. McCreight. Organization and maintenance of large ordered indices. In SIGFIDET (Now SIGMOD), pages 107-141, 1970.

[18] S. Bickel, M. Brückner, and T. Scheffer. Discriminative learning under covariate shift. Journal of Machine Learning Research, 10(Sep):2137-2155, 2009.

[19] M. Böhm, B. Schlegel, P. B. Volk, U. Fischer, D. Habich, and W. Lehner. Efficient in-memory indexing with generalized prefix trees. In ${BTW}$ ,pages 227-246, 2011.

[20] J. Boyar and K. S. Larsen. Efficient rebalancing of chromatic search trees. Journal of Computer and System Sciences, 49(3):667 - 682, 1994. 30th IEEE Conference on Foundations of Computer Science.

[21] M. Brantner, D. Florescu, D. A. Graf, D. Kossmann, and T. Kraska. Building a database on S3. In SIGMOD, pages 251-264, 2008.

[22] A. Broder and M. Mitzenmacher. Network applications of bloom filters: A survey. Internet mathematics, 1(4):485-509, 2004.

[23] F. Chang, J. Dean, S. Ghemawat, W. C. Hsieh, D. A. Wallach, M. Burrows, T. Chandra, A. Fikes, and R. Gruber. Bigtable: A distributed storage system for structured data (awarded best paper!). In OSDI, pages 205-218, 2006.

[24] K. Cho, B. van Merrienboer, Ç. Gülcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In EMNLP, pages 1724-1734, 2014.

[25] A. Crotty, A. Galakatos, K. Dursun, T. Kraska, C. Binnig, U. Cetintemel, and S. Zdonik. An architecture for compiling udf-centric workflows. PVLDB, 8(12):1466-1477, 2015.

[26] M. Dietzfelbinger, A. Karlin, K. Mehlhorn, F. Meyer auF der Heide, H. Rohn-ert, and R. E. Tarjan. Dynamic perfect hashing: Upper and lower bounds. SIAM Journal on Computing, 23(4):738-761, 1994.

[27] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121-2159, 2011.

[28] A. Dvoretzky, J. Kiefer, and J. Wolfowitz. Asymptotic minimax character of the sample distribution function and of the classical multinomial estimator. The Annals of Mathematical Statistics, pages 642-669, 1956.

[29] B. Fan, D. G. Andersen, M. Kaminsky, and M. D. Mitzenmacher. Cuckoo filter: Practically better than bloom. In CoNEXT, pages 75-88, 2014.

[30] T. Fawcett. An introduction to roc analysis. Pattern recognition letters, 27(8):861-874, 2006.

[31] E. Fredkin. Trie memory. Commun. ACM, 3(9):490-499, Sept. 1960.

[32] A. Galakatos, M. Markovitch, C. Binnig, R. Fonseca, and T. Kraska. A-tree: A bounded approximate index structure. CoRR, abs/1801.10207, 2018.

[33] J. Goldstein, R. Ramakrishnan, and U. Shaft. Compressing Relations and Indexes. In ICDE, pages 370-379, 1998.

[34] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In NIPS, pages 2672-2680, 2014.

[35] G. Graefe. B-tree indexes, interpolation search, and skew. In DaMoN, 2006.

[36] G. Graefe and P. A. Larson. B-tree indexes and CPU caches. In ICDE, pages 349-358, 2001.

[37] A. Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

[38] R. Grossi, A. Gupta, and J. S. Vitter. High-order entropy-compressed text indexes. In SODA, pages 841-850. Society for Industrial and Applied Mathematics, 2003.

[39] R. Grossi and G. Ottaviano. The wavelet trie: Maintaining an indexed sequence of strings in compressed space. In PODS, pages 203-214, 2012.

[40] J. Guo and J. Li. CNN based hashing for image retrieval. CoRR, abs/1509.01354, 2015.

[41] M. Gupta, A. Cotter, J. Pfeifer, K. Voevodski, K. Canini, A. Mangylov, W. Moczydlowski, and A. Van Esbroeck. Monotonic calibrated interpolated look-up tables. The Journal of Machine Learning Research, 17(1):3790-3836, 2016.

[42] J. C. Huang and B. J. Frey. Cumulative distribution networks and the derivative-sum-product algorithm: Models and inference for cumulative distribution functions on graphs. J. Mach. Learn. Res., 12:301-348, Feb. 2011.

[43] K. Kaczmarski. B + -Tree Optimized for GPGPU. 2012.

[44] C. Kim, J. Chhugani, N. Satish, E. Sedlar, A. D. Nguyen, T. Kaldewey, V. W. Lee, S. A. Brandt, and P. Dubey. Fast: Fast architecture sensitive tree search on modern cpus and gpus. In SIGMOD, pages 339-350, 2010.

[45] T. Kissinger, B. Schlegel, D. Habich, and W. Lehner. Kiss-tree: Smart latch-free in-memory indexing on modern architectures. In ${DaMoN}$ ,pages ${16} - {23}$ , 2012.

[46] T. J. Lehman and M. J. Carey. A study of index structures for main memory database management systems. In VLDB, pages 294-303, 1986.

[47] V. Leis. FAST source. http://www-db.in.tum.de/âLijleis/index/fast.cpp.

[48] V. Leis, A. Kemper, and T. Neumann. The adaptive radix tree: Artful indexing for main-memory databases. In ICDE, pages 38-49, 2013.

[49] W. Litwin. Readings in database systems. chapter Linear Hashing: A New Tool for File and Table Addressing., pages 570-581. Morgan Kaufmann Publishers Inc., 1988.

[50] M. Magdon-Ismail and A. F. Atiya. Neural networks for density estimation. In M. J. Kearns, S. A. Solla, and D. A. Cohn, editors, NIPS, pages 522-528. MIT Press, 1999.

[51] D. J. Miller and H. S. Uyar. A mixture of experts classifier with learning based on both labelled and unlabelled data. In NIPS, pages 571-577, 1996.

[52] M. Mitzenmacher. Compressed bloom filters. In PODC, pages 144-150, 2001.

[53] M. Mitzenmacher. A model for learned bloom filters and related structures. arXiv preprint arXiv:1802.00884, 2018.

[54] G. Moerkotte. Small Materialized Aggregates: A Light Weight Index Structure for Data Warehousing. In VLDB, pages 476-487, 1998.

[55] T. Neumann and G. Weikum. RDF-3X: A RISC-style Engine for RDF. Proc. VLDB Endow., pages 647-659, 2008.

[56] OpenStreetMap database ©OpenStreetMap contributors. https://aws.amazon.com/public-datasets/osm.

[57] R. Pagh and F. F. Rodler. Cuckoo hashing. Journal of Algorithms, 51(2):122- 144, 2004.

[58] J. Rao and K. A. Ross. Making b+- trees cache conscious in main memory. In SIGMOD, pages 475-486, 2000.

[59] S. Richter, V. Alvarez, and J. Dittrich. A seven-dimensional analysis of hashing methods and its implications on query processing. Proc. VLDB Endow., 9(3):96-107, Nov. 2015.

[60] D. G. Severance and G. M. Lohman. Differential files: Their application to the maintenance of large data bases. In SIGMOD, pages 43-43, 1976.

[61] A. Shahvarani and H.-A. Jacobsen. A hybrid b+-tree as solution for in-memory indexing on cpu-gpu heterogeneous computing platforms. In SIGMOD, pages 1523-1538, 2016.

[62] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

[63] M. Stonebraker and L. A. Rowe. The Design of POSTGRES. In SIGMOD, pages 340-355, 1986.

[64] I. Sutskever, O. Vinyals, and Q. V. Le. Sequence to sequence learning with neural networks. In NIPS, pages 3104-3112, 2014.

[65] F. Tramèr, A. Kurakin, N. Papernot, D. Boneh, and P. McDaniel. Ensemble adversarial training: Attacks and defenses. arXiv preprint arXiv:1705.07204, 2017.

[66] M. Turcanik and M. Javurek. Hash function generation by neural network. In NTSP, pages 1-5, Oct 2016.

[67] J. Wang, W. Liu, S. Kumar, and S. F. Chang. Learning to hash for indexing big data;a survey. Proceedings of the IEEE, 104(1):34-57, Jan 2016.

[68] J. Wang, H. T. Shen, J. Song, and J. Ji. Hashing for similarity search: A survey. CoRR, abs/1408.2927, 2014.

[69] J. Wang, J. Wang, N. Yu, and S. Li. Order preserving hashing for approximate nearest neighbor search. In ${MM}$ ,pages 133-142,2013.

[70] Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

[71] S. You, D. Ding, K. Canini, J. Pfeifer, and M. Gupta. Deep lattice networks and partial monotonic functions. In NIPS, pages 2985-2993, 2017.

[72] Y. You, Z. Zhang, C. Hsieh, J. Demmel, and K. Keutzer. Imagenet training in minutes. CoRR, abs/1709.05011, 2017.

[73] J. Yu and M. Sarwat. Two Birds, One Stone: A Fast, Yet Lightweight, Indexing Scheme for Modern Database Systems. In VLDB, pages 385-396, 2016.

[74] E. Zamanian, C. Binnig, T. Kraska, and T. Harris. The end of a myth: Distributed transaction can scale. PVLDB, 10(6):685-696, 2017.

[75] H. Zhang, D. G. Andersen, A. Pavlo, M. Kaminsky, L. Ma, and R. Shen. Reducing the storage overhead of main-memory OLTP databases with hybrid indexes. In SIGMOD, pages 1567-1581, 2016.

## A THEORETICAL ANALYSIS OF SCALING LEARNED RANGE INDEXES

## A 学习型范围索引可扩展性的理论分析

One advantage of framing learned range indexes as modeling the cumulative distribution function (CDF) of the data is that we can build on the long research literature on modeling the CDF. Significant research has studied the relationship between a theoretical CDF $F\left( x\right)$ and the empirical CDF of data sampled from $F\left( x\right)$ . We consider the case where we have sampled i.i.d. $N$ datapoints, $\mathcal{Y}$ ,from some distribution,and we will use ${\widehat{F}}_{N}\left( x\right)$ to denote the empirical cumulative distribution function:

将学习型范围索引构建为对数据的累积分布函数（CDF）进行建模的一个优势在于，我们可以借鉴关于CDF建模的长期研究文献。大量研究已经探讨了理论CDF $F\left( x\right)$ 与从 $F\left( x\right)$ 中采样的数据的经验CDF之间的关系。我们考虑从某个分布中独立同分布地采样 $N$ 个数据点 $\mathcal{Y}$ 的情况，并且我们将使用 ${\widehat{F}}_{N}\left( x\right)$ 来表示经验累积分布函数：

$$
{\widehat{F}}_{N}\left( x\right)  = \frac{\mathop{\sum }\limits_{{y \in  y}}{\mathbf{1}}_{y \leq  x}}{N}. \tag{2}
$$

One theoretical question about learned indexes is: how well do they scale with the size of the data $N$ ? In our setting,we learn a model $F\left( x\right)$ to approximate the distribution of our data ${\widehat{F}}_{N}\left( x\right)$ . Here,we assume we know the distribution $F\left( x\right)$ that generated the data and analyze the error inherent in the data being sampled from that distribution ${}^{6}$ . That is,we consider the error between the distribution of data ${\widehat{F}}_{N}\left( x\right)$ and our model of the distribution $F\left( x\right)$ . Because ${\widehat{F}}_{N}\left( x\right)$ is a binomial random variable with mean $F\left( x\right)$ ,we find that the expected squared error between our data and our model is given by

关于学习型索引的一个理论问题是：它们如何随数据大小 $N$ 进行扩展？在我们的场景中，我们学习一个模型 $F\left( x\right)$ 来近似我们的数据 ${\widehat{F}}_{N}\left( x\right)$ 的分布。这里，我们假设我们知道生成数据的分布 $F\left( x\right)$ ，并分析从该分布 ${}^{6}$ 中采样数据所固有的误差。也就是说，我们考虑数据 ${\widehat{F}}_{N}\left( x\right)$ 的分布与我们的分布模型 $F\left( x\right)$ 之间的误差。因为 ${\widehat{F}}_{N}\left( x\right)$ 是一个均值为 $F\left( x\right)$ 的二项随机变量，我们发现我们的数据与我们的模型之间的期望平方误差由下式给出

$$
\mathbf{E}\left\lbrack  {\left( F\left( x\right)  - {\widehat{F}}_{N}\left( x\right) \right) }^{2}\right\rbrack   = \frac{F\left( x\right) \left( {1 - F\left( x\right) }\right) }{N}. \tag{3}
$$

In our application the look-up time scales with the average error in the number of positions in the sorted data; that is, we

在我们的应用中，查找时间与排序数据中位置数量的平均误差成比例；也就是说，我们

<!-- Media -->

<table><tr><td>Dataset</td><td>Slots</td><td>Hash Type</td><td>Time (ns)</td><td>Empty Slots</td><td>Space</td></tr><tr><td rowspan="6">Map</td><td rowspan="2">75%</td><td>Model Hash</td><td>67</td><td>0.18GB</td><td>0.21x</td></tr><tr><td>Random Has</td><td>52</td><td>0.84GB</td><td/></tr><tr><td rowspan="2">100%</td><td>Model Hash</td><td>53</td><td>0.35GB</td><td>0.22x</td></tr><tr><td>Random Has</td><td>48</td><td>1.58GB</td><td/></tr><tr><td rowspan="2">125%</td><td>Model Hash</td><td>64</td><td>1.47GB</td><td>0.60x</td></tr><tr><td>Random Has</td><td>49</td><td>2.43GB</td><td/></tr><tr><td rowspan="6">Web</td><td rowspan="2">75%</td><td>Model Hash</td><td>78</td><td>0.64GB</td><td>0.77x</td></tr><tr><td>Random Has</td><td>53</td><td>0.83GB</td><td/></tr><tr><td rowspan="2">100%</td><td>Model Hash</td><td>63</td><td>1.09GB</td><td>0.70x</td></tr><tr><td>Random Has</td><td>50</td><td>1.56GB</td><td/></tr><tr><td rowspan="2">125%</td><td>Model Hash</td><td>77</td><td>2.20GB</td><td>0.91x</td></tr><tr><td>Random Has</td><td>50</td><td>2.41GB</td><td/></tr><tr><td rowspan="6">Log Normal</td><td rowspan="2">75%</td><td>Model Hash</td><td>79</td><td>0.63GB</td><td>0.79x</td></tr><tr><td>Random Has</td><td>52</td><td>0.80GB</td><td/></tr><tr><td rowspan="2">100%</td><td>Model Hash</td><td>66</td><td>1.10GB</td><td>0.73x</td></tr><tr><td>Random Has</td><td>46</td><td>1.50GB</td><td/></tr><tr><td rowspan="2">125%</td><td>Model Hash</td><td>77</td><td>2.16GB</td><td>0.94x</td></tr><tr><td>Random Has</td><td>46</td><td>2.31GB</td><td/></tr></table>

<table><tbody><tr><td>数据集</td><td>槽位</td><td>哈希类型</td><td>时间（纳秒）</td><td>空槽位</td><td>空间</td></tr><tr><td rowspan="6">映射</td><td rowspan="2">75%</td><td>模型哈希</td><td>67</td><td>0.18GB</td><td>0.21x</td></tr><tr><td>随机哈希</td><td>52</td><td>0.84GB</td><td></td></tr><tr><td rowspan="2">100%</td><td>模型哈希</td><td>53</td><td>0.35GB</td><td>0.22x</td></tr><tr><td>随机哈希</td><td>48</td><td>1.58GB</td><td></td></tr><tr><td rowspan="2">125%</td><td>模型哈希</td><td>64</td><td>1.47GB</td><td>0.60x</td></tr><tr><td>随机哈希</td><td>49</td><td>2.43GB</td><td></td></tr><tr><td rowspan="6">网络</td><td rowspan="2">75%</td><td>模型哈希</td><td>78</td><td>0.64GB</td><td>0.77x</td></tr><tr><td>随机哈希</td><td>53</td><td>0.83GB</td><td></td></tr><tr><td rowspan="2">100%</td><td>模型哈希</td><td>63</td><td>1.09GB</td><td>0.70x</td></tr><tr><td>随机哈希</td><td>50</td><td>1.56GB</td><td></td></tr><tr><td rowspan="2">125%</td><td>模型哈希</td><td>77</td><td>2.20GB</td><td>0.91x</td></tr><tr><td>随机哈希</td><td>50</td><td>2.41GB</td><td></td></tr><tr><td rowspan="6">对数正态</td><td rowspan="2">75%</td><td>模型哈希</td><td>79</td><td>0.63GB</td><td>0.79x</td></tr><tr><td>随机哈希</td><td>52</td><td>0.80GB</td><td></td></tr><tr><td rowspan="2">100%</td><td>模型哈希</td><td>66</td><td>1.10GB</td><td>0.73x</td></tr><tr><td>随机哈希</td><td>46</td><td>1.50GB</td><td></td></tr><tr><td rowspan="2">125%</td><td>模型哈希</td><td>77</td><td>2.16GB</td><td>0.94x</td></tr><tr><td>随机哈希</td><td>46</td><td>2.31GB</td><td></td></tr></tbody></table>

Figure 11: Model vs Random Hash-map

图11：模型哈希映射与随机哈希映射

<!-- Media -->

are concerned with the error between our model ${NF}\left( x\right)$ and the key position $N{\widehat{F}}_{N}\left( x\right)$ . With some minor manipulation of Eq. (3), we find that the average error in the predicted positions grows at a rate of $O\left( \sqrt{N}\right)$ . Note that this sub-linear scaling in error for a constant-sized model is an improvement over the linear scaling achieved by a constant-sized B-Tree. This provides preliminary understanding of the scalability of our approach and demonstrates how framing indexing as learning the CDF lends itself well to theoretical analysis.

我们关注的是我们的模型${NF}\left( x\right)$与键位置$N{\widehat{F}}_{N}\left( x\right)$之间的误差。通过对公式(3)进行一些小的处理，我们发现预测位置的平均误差以$O\left( \sqrt{N}\right)$的速率增长。请注意，对于固定大小的模型，这种误差的次线性缩放是对固定大小的B树所实现的线性缩放的一种改进。这为我们的方法的可扩展性提供了初步的理解，并展示了将索引构建为学习累积分布函数（CDF）如何有助于进行理论分析。

## B SEPARATED CHAINING HASH-MAP

## B 分离链接哈希映射

We evaluated the potential of learned hash functions using a separate chaining Hash-map; records are stored directly within an array and only in the case of a conflict is the record attached to the linked-list. That is without a conflict there is at most one cache miss. Only in the case that several keys map to the same position, additional cache-misses might occur. We choose that design as it leads to the best look-up performance even for larger payloads. For example, we also tested a commercial-grade dense Hash-map with a bucket-based in-place overflow (i.e., the Hash-map is divided into buckets to minimize overhead and uses in-place overflow if a bucket is full [2]). While it is possible to achieve a lower footprint using this technique, we found that it is also twice as slow as the separate chaining approach. Furthermore,at ${80}\%$ or more memory utilization the dense Hash-maps degrade further in performance. Of course many further (orthogonal) optimizations are possible and by no means do we claim that this is the most memory or CPU efficient implementation of a Hash-map. Rather we aim to demonstrate the general potential of learned hash functions.

我们使用分离链接哈希映射评估了学习哈希函数的潜力；记录直接存储在数组中，只有在发生冲突的情况下，记录才会附加到链表上。也就是说，在没有冲突的情况下，最多只会有一次缓存未命中。只有当多个键映射到同一位置时，才可能会发生额外的缓存未命中。我们选择这种设计是因为即使对于较大的有效负载，它也能带来最佳的查找性能。例如，我们还测试了一种商业级的密集哈希映射，它采用基于桶的就地溢出（即，哈希映射被划分为多个桶以最小化开销，并且如果桶已满则使用就地溢出[2]）。虽然使用这种技术可以实现更小的内存占用，但我们发现它的速度是分离链接方法的两倍。此外，在${80}\%$或更高的内存利用率下，密集哈希映射的性能会进一步下降。当然，还可以进行许多其他（正交）优化，我们绝不是声称这是哈希映射最节省内存或CPU效率最高的实现方式。相反，我们旨在展示学习哈希函数的一般潜力。

As the baseline for this experiment we used our Hash-map implementation with a MurmurHash3-like hash-function. As the data we used the three integer datasets from Section 3.7 and as the model-based Hash-map the 2-stage RMI model with ${100}\mathrm{k}$ models on the 2nd stage and no hidden layers from the same section. For all experiments we varied the number of available slots from 75% to 125% of the data. That is, with 75% there are ${25}\%$ less slots in the Hash-map than data records. Forcing less slots than the data size, minimizes the empty slots within the Hash-map at the expense of longer linked lists. However, for Hash-maps we store the full records, which consist of a 64bit key, 64bit payload, and a 32bit meta-data field for delete flags, version nb, etc. (so a record has a fixed length of 20 Bytes); note that our chained hash-map adds another 32bit pointer, making it a 24Byte slot.

作为该实验的基线，我们使用了带有类似MurmurHash3哈希函数的哈希映射实现。作为数据，我们使用了第3.7节中的三个整数数据集，作为基于模型的哈希映射，我们使用了来自同一节的二阶RMI模型，该模型在第二阶段有${100}\mathrm{k}$个模型且没有隐藏层。在所有实验中，我们将可用槽位的数量从数据量的75%变化到125%。也就是说，当为75%时，哈希映射中的槽位比数据记录少${25}\%$个。强制使槽位数量少于数据大小，以更长的链表为代价，最小化了哈希映射中的空槽位。然而，对于哈希映射，我们存储完整的记录，这些记录由一个64位的键、64位的有效负载和一个32位的元数据字段（用于删除标志、版本号等）组成（因此一条记录的固定长度为20字节）；请注意，我们的链式哈希映射还会添加另一个32位的指针，使其成为一个24字节的槽位。

---

<!-- Footnote -->

${}^{6}$ Learning $F\left( x\right)$ can improve or worsen the error,but we take this as a reasonable assumption for some applications, such as data keyed by a random hash.

${}^{6}$学习$F\left( x\right)$可能会改善或恶化误差，但对于某些应用，如由随机哈希键控的数据，我们将此视为一个合理的假设。

<!-- Footnote -->

---

The results are shown in Figure 11, listing the average lookup time, the number of empty slots in GB and the space improvement as a factor of using a randomized hash function. Note, that in contrast to the B-Tree experiments, we do include the data size. The main reason is that in order to enable 1 cache-miss look-ups, the data itself has to be included in the Hash-map, whereas in the previous section we only counted the extra index overhead excluding the sorted array itself.

结果如图11所示，列出了平均查找时间、以GB为单位的空槽位数量以及使用随机哈希函数时的空间改进因子。请注意，与B树实验不同，我们确实将数据大小考虑在内。主要原因是为了实现一次缓存未命中的查找，数据本身必须包含在哈希映射中，而在前面的章节中，我们只计算了不包括排序数组本身的额外索引开销。

As can be seen in Figure 11, the index with the model hash function overall has similar performance while utilizing the memory better. For example, for the map dataset the model hash function only "wastes" 0.18GB in slots, an almost 80% reduction compared to using a random hash function. Obviously, the moment we increase the Hash-map in size to have 25% more slots, the savings are not as large, as the Hash-map is also able to better spread out the keys. Surprisingly if we decrease the space to 75% of the number of keys, the learned Hash-map still has an advantage because of the still prevalent birthday paradox.

从图11中可以看出，使用模型哈希函数的索引在整体上具有相似的性能，同时能更好地利用内存。例如，对于地图数据集，模型哈希函数仅在槽位中“浪费”了0.18GB，与使用随机哈希函数相比，减少了近80%。显然，当我们将哈希映射的大小增加25%以拥有更多槽位时，节省的空间就没有那么大了，因为哈希映射也能够更好地分散键。令人惊讶的是，如果我们将空间减少到键数量的75%，由于生日悖论仍然普遍存在，学习哈希映射仍然具有优势。

## C HASH-MAP COMPARISON AGAINST ALTERNATIVE BASELINES

## C 哈希映射与其他基线的比较

In addition to the separate chaining Hash-map architecture, we also compared learned point indexes against four alternative Hash-map architectures and configurations:

除了分离链接哈希映射架构之外，我们还将学习点索引与四种其他哈希映射架构和配置进行了比较：

AVX Cuckoo Hash-map: We used an AVX optimized Cuckoo Hash-map from [7].

AVX布谷鸟哈希映射：我们使用了来自[7]的AVX优化布谷鸟哈希映射。

Commercial Cuckoo Hash-map: The implementation of [7] is highly tuned, but does not handle all corner cases. We therefore also compared against a commercially used Cuckoo Hash-map.

商业布谷鸟哈希映射：[7]中的实现经过了高度调优，但不能处理所有的边界情况。因此，我们还与商业使用的布谷鸟哈希映射进行了比较。

In-place chained Hash-map with learned hash functions: One significant downside of separate chaining is that it requires additional memory for the linked list. As an alternative, we implemented a chained Hash-map, which uses a two pass algorithm: in the first pass, the learned hash function is used to put items into slots. If a slot is already taken, the item is skipped. Afterwards we use a separate chaining approach for every skipped item except that we use the remaining free slots with offsets as pointers for them. As a result, the utilization can be ${100}\%$ (recall,we do not consider inserts) and the quality of the learned hash function can only make an impact on the performance not the size: the fewer conflicts, the fewer cache misses. We used a simple single stage multi-variate model as the learned hash function and implemented the Hash-map including the model outside of our benchmarking framework to ensure a fair comparison.

采用学习型哈希函数的原地链式哈希映射：分离链接法的一个显著缺点是，它需要为链表额外分配内存。作为一种替代方案，我们实现了一个链式哈希映射，它采用了两阶段算法：在第一阶段，使用学习型哈希函数将元素放入槽位中。如果某个槽位已被占用，则跳过该元素。之后，我们对每个被跳过的元素采用分离链接法，但不同的是，我们使用剩余的空闲槽位及其偏移量作为这些元素的指针。因此，利用率可以达到${100}\%$（请记住，我们不考虑插入操作），并且学习型哈希函数的质量只会影响性能，而不会影响大小：冲突越少，缓存未命中的情况就越少。我们使用一个简单的单阶段多元模型作为学习型哈希函数，并在基准测试框架之外实现了包含该模型的哈希映射，以确保进行公平的比较。

<!-- Media -->

<table><tr><td>Type</td><td>Time (ns)</td><td>Utilization</td></tr><tr><td>AVX Cuckoo, 32-bit value</td><td>31ns</td><td>99%</td></tr><tr><td>AVX Cuckoo, 20 Byte record</td><td>43ns</td><td>99%</td></tr><tr><td>Comm. Cuckoo, 20Byte record</td><td>90ns</td><td>95%</td></tr><tr><td>In-place chained Hash-map with learned hash functions, record</td><td>35ns</td><td>100%</td></tr></table>

<table><tbody><tr><td>类型</td><td>时间（纳秒）</td><td>利用率</td></tr><tr><td>AVX布谷鸟哈希（AVX Cuckoo），32位值</td><td>31纳秒</td><td>99%</td></tr><tr><td>AVX布谷鸟哈希（AVX Cuckoo），20字节记录</td><td>43纳秒</td><td>99%</td></tr><tr><td>普通布谷鸟哈希（Comm. Cuckoo），20字节记录</td><td>90纳秒</td><td>95%</td></tr><tr><td>使用学习型哈希函数的原地链式哈希映射（In-place chained Hash-map with learned hash functions），记录</td><td>35纳秒</td><td>100%</td></tr></tbody></table>

Table 1: Hash-map alternative baselines

表1：哈希映射替代基线

<!-- Media -->

Like in Section B our records are 20 Bytes large and consist of a 64bit key, 64bit payload, and a 32bit meta-data field as commonly found in real applications (e.g., for delete flags, version numbers, etc.). For all Hash-map architectures we tried to maximize utilization and used records, except for the AVX Cuckoo Hash-map where we also measured the performance for 32bit values. As the dataset we used the log-normal data and the same hardware as before. The results are shown in Table 1.

与B节一样，我们的记录大小为20字节，由一个64位键、64位有效负载和一个32位元数据字段组成，这在实际应用中很常见（例如，用于删除标志、版本号等）。对于所有哈希映射架构，我们都试图最大限度地提高利用率并使用记录，但对于AVX布谷鸟哈希映射，我们还测量了32位值的性能。我们使用对数正态数据作为数据集，并使用与之前相同的硬件。结果如表1所示。

The results for the AVX cuckoo Hash-map show that the payload has a significant impact on the performance. Going from 8 Byte to 20 Byte decreases the performance by almost 40%. Furthermore, the commercial implementation which handles all corner cases but is not very AVX optimized slows down the lookup by another factor of 2 . In contrast, our learned hash functions with in-place chaining can provide better lookup performance than even the cuckoo Hash-map for our records. The main take-aways from this experiment is that learned hash functions can be used with different Hash-map architectures and that the benefits and disadvantages highly depend on the implementation, data and workload.

AVX布谷鸟哈希映射的结果表明，有效负载对性能有显著影响。从8字节增加到20字节会使性能下降近40%。此外，处理所有边界情况但未针对AVX进行优化的商业实现会使查找速度再慢2倍。相比之下，我们采用就地链接的学习哈希函数甚至可以为我们的记录提供比布谷鸟哈希映射更好的查找性能。该实验的主要结论是，学习哈希函数可以与不同的哈希映射架构一起使用，并且其优缺点在很大程度上取决于实现方式、数据和工作负载。

## D FUTURE DIRECTIONS FOR LEARNED B-TREES

## D 学习型B树的未来方向

In the main part of the paper, we have focused on index-structures for read-only, in-memory database systems. Here we outline how the idea of learned index structures could be extended in the future.

在论文的主要部分，我们专注于只读内存数据库系统的索引结构。在这里，我们概述了学习型索引结构的思想在未来如何扩展。

### D.1 Inserts and Updates

### D.1 插入和更新

On first sight, inserts seem to be the Achilles heel of learned indexes because of the potentially high cost for learning models, but yet again learned indexes might have a significant advantage for certain workloads. In general we can distinguish between two types of inserts: (1) appends and (2) inserts in the middle like updating a secondary index on the customer-id over an order table.

乍一看，由于学习模型的潜在成本较高，插入似乎是学习型索引的致命弱点，但对于某些工作负载，学习型索引可能仍具有显著优势。一般来说，我们可以将插入分为两种类型：（1）追加插入和（2）中间插入，例如更新订单表上基于客户ID的二级索引。

Let's for the moment focus on the first case: appends. For example, it is reasonable to assume that for an index over the timestamps of web-logs, like in our previous experiments, most if not all inserts will be appends with increasing timestamps. Now, let us further assume that our model generalizes and is able to learn the patterns, which also hold for the future data. As a result,updating the index structure becomes an $O\left( 1\right)$ operation; it is a simple append and no change of the model itself is needed,whereas a B-Tree requires $O\left( {\log n}\right)$ operations to keep the B-Tree balance. A similar argument can also be made for inserts in the middle, however, those might require to move data or reserve space within the data, so that the new items can be put into the right place.

目前，让我们先关注第一种情况：追加插入。例如，合理的假设是，对于像我们之前实验中那样基于网络日志时间戳的索引，大多数（如果不是全部）插入将是时间戳递增的追加插入。现在，让我们进一步假设我们的模型具有泛化能力，能够学习到这些模式，并且这些模式对未来的数据也适用。结果，更新索引结构变成了一个$O\left( 1\right)$操作；这只是一个简单的追加操作，不需要对模型本身进行更改，而B树需要$O\left( {\log n}\right)$操作来保持B树的平衡。对于中间插入也可以提出类似的论点，然而，这些插入可能需要移动数据或在数据中预留空间，以便将新项放入正确的位置。

Obviously, this observation also raises several questions. First, there seems to be an interesting trade-off in the generalizability of the model and the "last mile" performance; the better the "last mile" prediction, arguably, the more the model is overfitting and less able to generalize to new data items.

显然，这一观察结果也引发了几个问题。首先，模型的泛化能力和“最后一英里”性能之间似乎存在一个有趣的权衡；可以说，“最后一英里”预测越好，模型就越容易过拟合，越难以对新数据项进行泛化。

Second, what happens if the distribution changes? Can it be detected, and is it possible to provide similar strong guarantees as B-Trees which always guarantee $O$ (logn) look-up and insertion costs? While answering this question goes beyond the scope of this paper, we believe that it is possible for certain models to achieve it. More importantly though, machine learning offers new ways to adapt the models to changes in the data distribution, such as online learning, which might be more effective than traditional B-Tree balancing techniques. Exploring them also remains future work.

其次，如果数据分布发生变化会怎样？能否检测到这种变化，是否有可能提供与B树一样强大的保证，即始终保证$O$ (logn)的查找和插入成本？虽然回答这个问题超出了本文的范围，但我们相信某些模型有可能实现这一点。不过，更重要的是，机器学习提供了新的方法来使模型适应数据分布的变化，例如在线学习，这可能比传统的B树平衡技术更有效。探索这些方法也是未来的工作。

Finally, it should be pointed out that there always exists a much simpler alternative to handling inserts by building a delta-index [60]. All inserts are kept in buffer and from time to time merged with a potential retraining of the model. This approach is already widely used, for example in Bigtable [23] and many other systems, and was recently explored in [32] for learned indexes.

最后，应该指出的是，通过构建增量索引[60]来处理插入总是存在一种更简单的替代方法。所有插入都保存在缓冲区中，并时不时地与模型的潜在重新训练一起进行合并。这种方法已经被广泛使用，例如在Bigtable [23]和许多其他系统中，最近在[32]中也对学习型索引进行了探索。

### D.2 Paging

### D.2 分页

Throughout this section we assumed that the data, either the actual records or the <key , pointer> pairs, are stored in one continuous block. However, especially for indexes over data stored on disk, it is quite common to partition the data into larger pages that are stored in separate regions on disk. To that end, our observation that a model learns the CDF no longer holds true as $\operatorname{pos} = \Pr \left( {X < \text{Key}}\right)  * N$ is violated. In the following we outline several options to overcome this issue:

在本节中，我们假设数据（无论是实际记录还是<键，指针>对）都存储在一个连续的块中。然而，特别是对于存储在磁盘上的数据的索引，将数据划分为较大的页面并存储在磁盘的不同区域是很常见的。为此，我们关于模型学习累积分布函数（CDF）的观察不再成立，因为$\operatorname{pos} = \Pr \left( {X < \text{Key}}\right)  * N$被违反了。下面我们概述几种解决这个问题的方法：

Leveraging the RMI structure: The RMI structure already partitions the space into regions. With small modifications to the learning process, we can minimize how much models overlap in the regions they cover. Furthermore, it might be possible to duplicate any records which might be accessed by more than one model.

利用RMI结构（Recursive Model Index，递归模型索引结构）：RMI结构已经将空间划分为多个区域。通过对学习过程进行小的修改，我们可以最小化模型在其覆盖区域内的重叠程度。此外，复制可能被多个模型访问的任何记录也是可行的。

Another option is to have an additional translation table in the form of <first_key, disk-position>. With the translation table the rest of the index structure remains the same. However, this idea will work best if the disk pages are very large. At the same time it is possible to use the predicted position with the min- and max-error to reduce the number of bytes which have to be read from a large page, so that the impact of the page size might be negligible.

另一种选择是使用一个额外的以<第一键, 磁盘位置>形式存在的转换表。有了这个转换表，索引结构的其余部分保持不变。然而，如果磁盘页面非常大，这个想法的效果会最佳。同时，可以使用带有最小和最大误差的预测位置来减少必须从大页面中读取的字节数，这样页面大小的影响可能可以忽略不计。

With more complex models, it might actually be possible to learn the actual pointers of the pages. Especially if a file-system is used to determine the page on disk with a systematic numbering of the blocks on disk (e.g., block1, ..., block100) the learning process can remain the same.

对于更复杂的模型，实际上有可能学习到页面的实际指针。特别是如果使用文件系统通过对磁盘上的块进行系统编号（例如，块1，...，块100）来确定磁盘上的页面，学习过程可以保持不变。

Obviously, more investigation is required to better understand the impact of learned indexes for disk-based systems. At the same time the significant space savings as well as speed benefits make it a very interesting avenue for future work.

显然，需要进行更多的研究来更好地理解基于磁盘的系统中学习型索引的影响。同时，显著的空间节省以及速度优势使其成为未来工作中一个非常有吸引力的方向。

## E FURTHER BLOOM FILTER RESULTS

## E 布隆过滤器的更多结果

In Section 5.1.2, we propose an alternative approach to a learned Bloom filter where the classifier output is discretized and used as an additional hash function in the traditional Bloom filter setup. Preliminary results demonstrate that this approach in some cases outperforms the results listed in Section 5.2, but as the results depend on the discretization scheme, further analysis is worthwhile. We describe below these additional experiments.

在5.1.2节中，我们提出了一种针对学习型布隆过滤器的替代方法，即将分类器的输出进行离散化，并将其用作传统布隆过滤器设置中的一个额外哈希函数。初步结果表明，这种方法在某些情况下优于5.2节中列出的结果，但由于结果取决于离散化方案，因此进一步分析是值得的。我们在下面描述这些额外的实验。

As before,we assume we have a model model $f\left( x\right)  \rightarrow  \left\lbrack  {0,1}\right\rbrack$ that maps keys to the range $\left\lbrack  {0,1}\right\rbrack$ . In this case,we allocate $m$ bits for a bitmap $M$ where we set $M\left\lbrack  {\lfloor {mf}\left( x\right) \rfloor }\right\rbrack   = 1$ for all inserted keys $x \in  \mathcal{K}$ . We can then observe the FPR by observing what percentage of non-keys in the validation set map to a location in the bitmap with a value of 1,i.e. ${\mathrm{{FPR}}}_{m} \equiv$ $\frac{\mathop{\sum }\limits_{{x \in  \widetilde{\mathcal{U}}}}M\left\lbrack  \left\lfloor  {f\left( x\right) m}\right\rfloor  \right\rbrack  }{\left| \widetilde{\mathcal{U}}\right| }$ . In addition,we have a traditional Bloom filter with false positive rate ${\mathrm{{FPR}}}_{B}$ . We say that a query $q$ is predicted to be a key if $M\left\lbrack  \left\lfloor  {f\left( q\right) m}\right\rfloor  \right\rbrack   = 1$ and the Bloom filter also returns that it is a key. As such, the overall FPR of the system is ${\mathrm{{FPR}}}_{m} \times  {\mathrm{{FPR}}}_{B}$ ; we can determine the size of the traditional Bloom filter based on it's false positive rate ${\mathrm{{FPR}}}_{B} = \frac{{p}^{ * }}{{\mathrm{{FPR}}}_{m}}$ where ${p}^{ * }$ is the desired FPR for the whole system.

和之前一样，我们假设我们有一个模型model $f\left( x\right)  \rightarrow  \left\lbrack  {0,1}\right\rbrack$，它将键映射到范围$\left\lbrack  {0,1}\right\rbrack$。在这种情况下，我们为位图$M$分配$m$位，对于所有插入的键$x \in  \mathcal{K}$，我们设置$M\left\lbrack  {\lfloor {mf}\left( x\right) \rfloor }\right\rbrack   = 1$。然后，我们可以通过观察验证集中非键映射到位图中值为1的位置的百分比来观察误报率（False Positive Rate，FPR），即${\mathrm{{FPR}}}_{m} \equiv$ $\frac{\mathop{\sum }\limits_{{x \in  \widetilde{\mathcal{U}}}}M\left\lbrack  \left\lfloor  {f\left( x\right) m}\right\rfloor  \right\rbrack  }{\left| \widetilde{\mathcal{U}}\right| }$。此外，我们有一个误报率为${\mathrm{{FPR}}}_{B}$的传统布隆过滤器。我们说，如果$M\left\lbrack  \left\lfloor  {f\left( q\right) m}\right\rfloor  \right\rbrack   = 1$且布隆过滤器也返回查询$q$是一个键，那么该查询被预测为一个键。因此，系统的整体误报率为${\mathrm{{FPR}}}_{m} \times  {\mathrm{{FPR}}}_{B}$；我们可以根据传统布隆过滤器的误报率${\mathrm{{FPR}}}_{B} = \frac{{p}^{ * }}{{\mathrm{{FPR}}}_{m}}$来确定其大小，其中${p}^{ * }$是整个系统所需的误报率。

As in Section 5.2, we test our learned Bloom filter on data from Google's transparency report. We use the same character RNN trained with a 16-dimensional width and 32-dimensional character embeddings. Scanning over different values for $m$ , we can observe the total size of the model, bitmap for the learned Bloom filter, and the traditional Bloom filter. For a desired total FPR ${p}^{ * } = {0.1}\%$ ,we find that setting $m = {1000000}$ gives a total size of ${2.21}\mathrm{{MB}}$ ,a ${27.4}\%$ reduction in memory, compared to the ${15}\%$ reduction following the approach in Section 5.1.1 and reported in Section 5.2. For a desired total FPR ${p}^{ * } = 1\%$ we get a total size of ${1.19}\mathrm{{MB}}$ ,a ${41}\%$ reduction in memory,compared to the ${36}\%$ reduction reported in Section 5.2.

与5.2节一样，我们在谷歌透明度报告的数据上测试我们学习得到的布隆过滤器（Bloom filter）。我们使用相同的字符循环神经网络（RNN），其训练时采用16维宽度和32维字符嵌入。通过扫描$m$的不同值，我们可以观察到模型的总大小、学习得到的布隆过滤器的位图以及传统布隆过滤器的情况。对于期望的总误报率（FPR）${p}^{ * } = {0.1}\%$，我们发现设置$m = {1000000}$可使总大小为${2.21}\mathrm{{MB}}$，与5.1.1节的方法以及5.2节报告的结果相比，内存减少了${27.4}\%$。对于期望的总误报率${p}^{ * } = 1\%$，我们得到的总大小为${1.19}\mathrm{{MB}}$，与5.2节报告的结果相比，内存减少了${41}\%$。

These results are a significant improvement over those shown in Section 5.2. However, typical measures of accuracy or calibration do not match this discretization procedure, and as such further analysis would be valuable to understand how well model accuracy aligns with it's suitability as a hash function.

这些结果相较于5.2节所示的结果有了显著改进。然而，典型的准确性或校准度量与这种离散化过程并不匹配，因此，进一步分析以了解模型准确性与其作为哈希函数的适用性之间的契合程度是很有价值的。