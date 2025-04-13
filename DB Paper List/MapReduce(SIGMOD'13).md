# Minimal MapReduce Algorithms

# 最小化MapReduce算法

Yufei Tao ${}^{1,2}$ Wenqing Lin ${}^{3}\;$ Xiaokui Xiao ${}^{3}$

陶宇飞 ${}^{1,2}$ 林文清 ${}^{3}\;$ 肖晓奎 ${}^{3}$

${}^{1}$ Chinese University of Hong Kong,Hong Kong

${}^{1}$ 香港中文大学，中国香港

${}^{2}$ Korea Advanced Institute of Science and Technology,Korea

${}^{2}$ 韩国科学技术院，韩国

${}^{3}$ Nanyang Technological University,Singapore

${}^{3}$ 南洋理工大学，新加坡

## Abstract

## 摘要

MapReduce has become a dominant parallel computing paradigm for big data, i.e., colossal datasets at the scale of tera-bytes or higher. Ideally, a MapReduce system should achieve a high degree of load balancing among the participating machines, and minimize the space usage, CPU and I/O time, and network transfer at each machine. Although these principles have guided the development of MapReduce algorithms, limited emphasis has been placed on enforcing serious constraints on the aforementioned metrics simultaneously. This paper presents the notion of minimal algorithm, that is, an algorithm that guarantees the best parallelization in multiple aspects at the same time, up to a small constant factor. We show the existence of elegant minimal algorithms for a set of fundamental database problems, and demonstrate their excellent performance with extensive experiments.

MapReduce已成为大数据（即规模达到TB级或更高的庞大数据集）的主流并行计算范式。理想情况下，MapReduce系统应在参与计算的机器之间实现高度的负载均衡，并最小化每台机器的空间使用、CPU和I/O时间以及网络传输。尽管这些原则指导了MapReduce算法的开发，但在同时对上述指标施加严格约束方面的重视程度有限。本文提出了最小化算法的概念，即一种能在多个方面同时保证最佳并行性（误差在一个小常数因子范围内）的算法。我们证明了针对一组基本数据库问题存在优雅的最小化算法，并通过大量实验展示了它们的卓越性能。

## Categories and Subject Descriptors

## 类别与主题描述

F2.2 [Analysis of algorithms and problem complexity]: Nonnumerical algorithms and problems

F2.2 [算法分析与问题复杂度]：非数值算法与问题

## Keywords

## 关键词

Minimal algorithm, MapReduce, big data

最小化算法，MapReduce，大数据

## 1. INTRODUCTION

## 1. 引言

We are in an era of information explosion, where industry, academia, and governments are accumulating data at an unprecedentedly high speed. This brings forward the urgent need of big data processing, namely, fast computation over colossal datasets whose sizes can reach the order of tera-bytes or higher. In recent years, the database community has responded to this grand challenge by building massive parallel computing platforms which use hundreds or even thousands of commodity machines. The most notable platform, which has attracted a significant amount of research attention, is MapReduce.

我们正处于信息爆炸的时代，工业界、学术界和政府都在以前所未有的高速积累数据。这就迫切需要进行大数据处理，即对规模可达TB级或更高的庞大数据集进行快速计算。近年来，数据库领域通过构建使用数百甚至数千台商用机器的大规模并行计算平台来应对这一重大挑战。其中最引人注目的平台是MapReduce，它吸引了大量的研究关注。

Since its invention [16], MapReduce has gone through years of improvement into a mature paradigm (see Section 2 for a review). At a high level, a MapReduce system involves a number of share-nothing machines which communicate only by sending messages over the network. A MapReduce algorithm instructs these machines to perform a computational task collaboratively Initially, the input dataset is distributed across the machines, typically in a non-replicate manner, i.e., each object on one machine. The algorithm executes in rounds (sometimes also called jobs in the literature), each having three phases: map, shuffle, and reduce. The first two enable the machines to exchange data: in the map phase, each machine prepares the information to be delivered to other machines, while the shuffle phase takes care of the actual data transfer. No network communication occurs in the reduce phase, where each machine performs calculation from its local storage. The current round finishes after the reduce phase. If the computational task has not completed, another round starts.

自发明以来 [16]，MapReduce经过多年的改进，已成为一种成熟的范式（相关回顾见第2节）。从高层次来看，MapReduce系统涉及多台无共享机器，它们仅通过网络发送消息进行通信。MapReduce算法指导这些机器协同执行计算任务。最初，输入数据集通常以非复制的方式分布在各台机器上，即每台机器上有一个对象。该算法按轮次执行（文献中有时也称为作业），每一轮有三个阶段：映射（map）、洗牌（shuffle）和归约（reduce）。前两个阶段使机器能够交换数据：在映射阶段，每台机器准备要发送给其他机器的信息，而洗牌阶段负责实际的数据传输。在归约阶段不发生网络通信，每台机器从其本地存储进行计算。归约阶段结束后，当前轮次完成。如果计算任务尚未完成，则开始下一轮。

Motivation. As with traditional parallel computing, a MapReduce system aims at a high degree of load balancing, and the minimization of space, CPU, I/O, and network costs at each individual machine. Although these principles have guided the design of MapReduce algorithms, the previous practices have mostly been on a best-effort basis, paying relatively less attention to enforcing serious constraints on different performance metrics. This work aims to remedy the situation by studying algorithms that promise outstanding efficiency in multiple aspects simultaneously.

动机。与传统并行计算一样，MapReduce系统旨在实现高度的负载均衡，并最小化每台机器的空间、CPU、I/O和网络成本。尽管这些原则指导了MapReduce算法的设计，但以往的实践大多是尽力而为，相对较少关注对不同性能指标施加严格约束。本工作旨在通过研究能在多个方面同时保证卓越效率的算法来改善这种情况。

Minimal MapReduce Algorithms. Denote by $S$ the set of input objects for the underlying problem. Let $n$ ,the problem cardinality, be the number of objects in $S$ ,and $t$ be the number of machines used in the system. Define $m = n/t$ ,namely, $m$ is the number of objects per machine when $S$ is evenly distributed across the machines. Consider an algorithm for solving a problem on $S$ . We say that the algorithm is minimal if it has all of the following properties.

最小化MapReduce算法。用 $S$ 表示底层问题的输入对象集合。设 $n$（问题基数）为 $S$ 中的对象数量，$t$ 为系统中使用的机器数量。定义 $m = n/t$，即 $m$ 是 $S$ 均匀分布在各台机器上时每台机器的对象数量。考虑一个用于解决 $S$ 上问题的算法。如果该算法具有以下所有属性，我们称其为最小化算法。

- Minimum footprint: at all times, each machine uses only $O\left( m\right)$ space of storage.

- 最小占用空间：在任何时候，每台机器仅使用 $O\left( m\right)$ 的存储空间。

- Bounded net-traffic: in each round, every machine sends and receives at most $O\left( m\right)$ words of information over the network.

- 有限网络流量：在每一轮中，每台机器通过网络发送和接收的信息最多为 $O\left( m\right)$ 个单词。

- Constant round: the algorithm must terminate after a constant number of rounds.

- 固定轮数：算法必须在固定的轮数后终止。

- Optimal computation: every machine performs only $O\left( {{T}_{\text{seq }}/t}\right)$ amount of computation in total (i.e.,summing over all rounds),where ${T}_{seq}$ is the time needed to solve the same problem on a single sequential machine. Namely, the algorithm should achieve a speedup of $t$ by using $t$ machines in parallel.

- 最优计算：每台机器总共仅执行 $O\left( {{T}_{\text{seq }}/t}\right)$ 的计算量（即，对所有轮次求和），其中 ${T}_{seq}$ 是在单台顺序机器上解决相同问题所需的时间。即，该算法应通过并行使用 $t$ 台机器实现 $t$ 倍的加速。

It is fairly intuitive why minimal algorithms are appealing. First, minimum footprint ensures that,each machine keeps $O\left( {1/t}\right)$ of the dataset $S$ at any moment. This effectively prevents partition skew, where some machines are forced to handle considerably more than $m$ objects,as is a major cause of inefficiency in MapReduce [36].

为什么最小化算法具有吸引力是相当直观的。首先，最小占用空间确保每台机器在任何时刻都保留数据集 $S$ 的 $O\left( {1/t}\right)$ 。这有效地防止了分区倾斜，即某些机器被迫处理远超过 $m$ 个对象的情况，而这是MapReduce [36] 中效率低下的主要原因。

<!-- Media -->

<!-- figureText: $\ell  = 5$ $\operatorname{window}\left( o\right)$ window sum $= {55}$ window max $= {20}$ -->

<img src="https://cdn.noedgeai.com/0195c8fe-f266-7d5f-9ab0-051892ebd600_1.jpg?x=263&y=147&w=485&h=162&r=0"/>

Figure 1: Sliding aggregates

图1：滑动聚合

<!-- Media -->

Second, bounded net-traffic guarantees that, the shuffle phase of each round transfers at most $O\left( {m \cdot  t}\right)  = O\left( n\right)$ words of network traffic overall. The duration of the phase equals roughly the time for a machine to send and receive $O\left( m\right)$ words,because the data transfers to/from different machines are in parallel. Furthermore, this property is also useful when one wants to make an algorithm stateless for the purpose of fault tolerance, as discussed in Section 2.1.

其次，有限网络流量保证每一轮的洗牌阶段总体上最多传输 $O\left( {m \cdot  t}\right)  = O\left( n\right)$ 个单词的网络流量。该阶段的持续时间大致等于一台机器发送和接收 $O\left( m\right)$ 个单词所需的时间，因为与不同机器之间的数据传输是并行的。此外，当出于容错目的希望使算法无状态时，此属性也很有用，如第2.1节所述。

The third property constant round is not new, as it has been the goal of many previous MapReduce algorithms. Importantly, this and the previous properties imply that there can be only $O\left( n\right)$ words of network traffic during the entire algorithm. Finally, optimal computation echoes the very original motivation of MapReduce to accomplish a computational task $t$ times faster than leveraging only one machine.

第三个属性固定轮数并不新鲜，因为它一直是许多先前MapReduce算法的目标。重要的是，此属性和前面的属性意味着在整个算法期间最多只有 $O\left( n\right)$ 个单词的网络流量。最后，最优计算呼应了MapReduce最初的动机，即比仅使用一台机器快 $t$ 倍地完成计算任务。

Contributions. The core of this work comprises of neat minimal algorithms for two problems:

贡献。这项工作的核心包括针对两个问题的简洁最小化算法：

Sorting. The input is a set $S$ of $n$ objects drawn from an ordered domain. When the algorithm terminates, all the objects must have been distributed across the $t$ machines in a sorted fashion. That is,we can order the machines from 1 to $t$ such that all objects in machine $i$ precede those in machine $j$ for all $1 \leq  i <$ $j \leq  t$ .

排序。输入是一个从有序域中抽取的 $n$ 个对象的集合 $S$ 。当算法终止时，所有对象必须以排序的方式分布在 $t$ 台机器上。即，我们可以将机器从1到 $t$ 进行排序，使得对于所有 $1 \leq  i <$ $j \leq  t$ ，机器 $i$ 中的所有对象都排在机器 $j$ 中的对象之前。

Sliding Aggregation. The input includes

滑动聚合。输入包括

- a set $S$ of $n$ objects from an ordered domain,where every object $o \in  S$ is associated with a numeric weight

- 一个从有序域中抽取的 $n$ 个对象的集合 $S$ ，其中每个对象 $o \in  S$ 都与一个数值权重相关联

- an integer $\ell  \leq  n$

- 一个整数 $\ell  \leq  n$

- and a distributive aggregate function AGG (e.g., sum, max, min).

- 以及一个可分配的聚合函数AGG（例如，求和、最大值、最小值）。

Denote by window(o)the set of $\ell$ largest objects in $S$ not exceeding $o$ . The window aggregate of $o$ is the result of applying AGG to the weights of the objects in window(o). The sliding aggregation problem is to report the window aggregate of every object in $S$ .

用window(o)表示 $S$ 中不超过 $o$ 的 $\ell$ 个最大对象的集合。 $o$ 的窗口聚合是对window(o)中对象的权重应用AGG的结果。滑动聚合问题是报告 $S$ 中每个对象的窗口聚合。

Figure 1 illustrates an example where $\ell  = 5$ . Each black dot represents an object in $S$ . Some relevant weights are given on top of the corresponding objects. For the object $o$ as shown, its window aggregate is 55 and20for $\mathrm{{AGG}} =$ sum and max, respectively.

图1展示了一个 $\ell  = 5$ 的示例。每个黑点代表 $S$ 中的一个对象。一些相关的权重显示在相应对象的上方。如图所示的对象 $o$ ，其窗口聚合对于 $\mathrm{{AGG}} =$ 求和和最大值分别为55和20。

The significance of sorting is obvious: a minimal algorithm for this problem leads to minimal algorithms for several fundamental database problems, including ranking, group-by, semi-join and skyline, as we will discuss in this paper.

排序的重要性显而易见：正如我们将在本文中讨论的那样，针对该问题的最优算法会引出针对几个基本数据库问题的最优算法，这些问题包括排名、分组、半连接和天际线查询。

The importance of the second problem probably deserves a bit more explanation. Sliding aggregates are crucial in studying time series. For example, consider a time series that records the Nasdaq index in history, with one value per minute. It makes good senses to examine moving statistics, that is, statistics aggregated from a sliding window. For example, a 6-month average/maximum with respect to a day equals the average/maximum Nasdaq index in a 6-month period ending on that very day. The 6-month averages/maximums of all days can be obtained by solving a sliding aggregation problem (note that an average can be calculated by dividing a window sum by the period length $\ell$ ).

第二个问题的重要性或许值得多做一些解释。滑动聚合在研究时间序列时至关重要。例如，考虑一个记录历史纳斯达克指数的时间序列，每分钟记录一个值。研究移动统计量是很有意义的，即从一个滑动窗口中聚合得到的统计量。例如，某一天的6个月平均值/最大值等于以该天为结束日的6个月期间内纳斯达克指数的平均值/最大值。通过解决一个滑动聚合问题可以得到所有日期的6个月平均值/最大值（注意，平均值可以通过将窗口总和除以时间段长度$\ell$来计算）。

Sorting and sliding aggregation can both be settled in $O\left( {n\log n}\right)$ time on a sequential computer. There has been progress in developing MapReduce algorithms for sorting. The state of the art is TeraSort [50], which won the Jim Gray's benchmark contest in 2009. TeraSort comes close to being minimal when a crucial parameter is set appropriately. As will be clear later, the algorithm requires manual tuning of the parameter, an improper choice of which can incur severe performance penalty. Sliding aggregation has also been studied in MapReduce by Beyer et al. [6]. However, as explained shortly, the algorithm is far from being minimal, and is efficient only when the window length $\ell$ is short - the authors of [6] commented that this problem is "notoriously difficult".

在顺序计算机上，排序和滑动聚合都可以在$O\left( {n\log n}\right)$时间内解决。在开发用于排序的MapReduce算法方面已经取得了进展。目前最先进的是TeraSort [50]，它在2009年赢得了吉姆·格雷基准测试竞赛。当一个关键参数设置适当时，TeraSort接近最优。正如后面会清楚看到的，该算法需要手动调整参数，参数选择不当会导致严重的性能损失。Beyer等人[6]也在MapReduce中研究了滑动聚合。然而，正如稍后将解释的，该算法远非最优，并且仅当窗口长度$\ell$较短时才有效——[6]的作者评论说这个问题“出了名地难”。

Technical Overview. This work was initialized by an attempt to justify theoretically why TeraSort often achieves excellent sorting time with only 2 rounds. In the first round, the algorithm extracts a random sample set ${S}_{\text{samp }}$ of the input $S$ ,and then picks $t - 1$ sampled objects as the boundary objects. Conceptually, these boundary objects divide $S$ into $t$ segments. In the second round, each of the $t$ machines acquires all the objects in a distinct segment, and sorts them. The size of ${S}_{\text{samp }}$ is the key to efficiency. If ${S}_{\text{samp }}$ is too small, the boundary objects may be insufficiently scattered, which can cause partition skew in the second round. Conversely, an over-sized ${S}_{\text{samp }}$ entails expensive sampling overhead. In the standard implementation of TeraSort, the sample size is left as a parameter, although it always seems to admit a good choice that gives outstanding performance [50].

技术概述。这项工作始于试图从理论上解释为什么TeraSort通常仅用两轮就能实现出色的排序时间。在第一轮中，该算法从输入$S$中提取一个随机样本集${S}_{\text{samp }}$，然后选择$t - 1$个采样对象作为边界对象。从概念上讲，这些边界对象将$S$划分为$t$个段。在第二轮中，$t$台机器中的每一台获取一个不同段中的所有对象，并对它们进行排序。${S}_{\text{samp }}$的大小是效率的关键。如果${S}_{\text{samp }}$太小，边界对象可能分布不够分散，这可能会导致第二轮中的分区倾斜。相反，过大的${S}_{\text{samp }}$会带来高昂的采样开销。在TeraSort的标准实现中，样本大小作为一个参数保留，尽管似乎总有一个能带来出色性能的好选择[50]。

In this paper, we provide rigorous explanation for the above phenomenon. Our theoretical analysis clarifies how to set the size of ${S}_{\text{samp }}$ to guarantee the minimality of TeraSort. In the meantime, we also remedy a conceptual drawback of TeraSort. As elaborated later, strictly speaking, this algorithm does not fit in the MapReduce framework, because it requires that (besides network messages) the machines should be able to communicate by reading/writing a common (distributed) file. Once this is disabled, the algorithm requires one more round. We present an elegant fix so that the algorithm still terminates in 2 rounds even by strictly adhering to MapReduce. Our findings of TeraSort have immediate practical significance, given the essential role of sorting in a large number of MapReduce programs.

在本文中，我们为上述现象提供了严格的解释。我们的理论分析阐明了如何设置${S}_{\text{samp }}$的大小以保证TeraSort的最优性。同时，我们还弥补了TeraSort的一个概念性缺陷。正如后面详细阐述的，严格来说，该算法并不符合MapReduce框架，因为它要求（除了网络消息之外）机器应该能够通过读写一个公共（分布式）文件进行通信。一旦禁止这样做，该算法就需要多一轮。我们提出了一个巧妙的修正方法，使得即使严格遵循MapReduce，该算法仍然可以在两轮内结束。鉴于排序在大量MapReduce程序中的重要作用，我们对TeraSort的研究结果具有直接的实际意义。

Regarding sliding aggregation,the difficulty lies in that $\ell$ is not a constant,but can be any value up to $n$ . Intuitively,when $\ell  \gg  m$ ,window(o)is so large that the objects in window(o) cannot be found on one machine under the minimum footprint constraint. Instead,window(o)would potentially span many machines, making it essential to coordinate the searching of machines judiciously to avoid a disastrous cost blowup. In fact, this pitfall has captured the existing algorithm of [6], whose main idea is to ensure that every sliding window be sent to a machine for aggregation (various windows may go to different machines). This suffers from prohibitive communication and processing cost when the window length $\ell$ is long. Our algorithm,on the other hand, achieves minimality with a novel idea of perfectly balancing the input objects across the machines while still maintaining their sorted order.

关于滑动聚合，难点在于$\ell$不是一个常量，而是可以取到$n$的任意值。直观地说，当$\ell  \gg  m$时，窗口(o)非常大，在最小占用空间约束下，窗口(o)中的对象无法在一台机器上找到。相反，窗口(o)可能会跨越多台机器，因此必须明智地协调各机器的搜索，以避免成本灾难性地增加。事实上，现有的[6]算法就陷入了这个陷阱，其主要思想是确保每个滑动窗口都被发送到一台机器进行聚合（不同的窗口可能会发送到不同的机器）。当窗口长度$\ell$较长时，这种方法会产生过高的通信和处理成本。另一方面，我们的算法通过一个新颖的想法实现了最优性，即在机器之间完美平衡输入对象，同时仍保持它们的有序性。

Outline. Section 2 reviews the previous work related to ours. Section 3 analyzes TeraSort and modifies it into a minimal algorithm, which Section 4 deploys to solve a set of fundamental problems minimally. Section 5 gives our minimal algorithm for the sliding aggregation problem. Section 6 evaluates the practical efficiency of the proposed techniques with extensive experiments. Finally, Section 7 concludes the paper with a summary of findings.

大纲。第2节回顾与我们的工作相关的过往研究。第3节分析TeraSort算法并将其改进为一种最小化算法，第4节运用该算法以最小化方式解决一系列基础问题。第5节给出用于滑动聚合问题的最小化算法。第6节通过大量实验评估所提出技术的实际效率。最后，第7节总结研究结果并结束本文。

## 2. PRELIMINARY AND RELATED WORK

## 2. 预备知识与相关工作

In Section 2.1, we expand the MapReduce introduction in Section 1 with more details to pave the way for our discussion. Section 2.2 reviews the existing studies on MapReduce, while Section 2.3 points out the relevance of minimal algorithms to the previous work.

在2.1节中，我们将详细扩展第1节中对MapReduce的介绍，为后续讨论奠定基础。2.2节回顾现有的MapReduce相关研究，而2.3节指出最小化算法与过往工作的相关性。

### 2.1 MapReduce

### 2.1 MapReduce

As explained earlier, a MapReduce algorithm proceeds in rounds, where each round has three phases: map, shuffle, and reduce. As all machines execute a program in the same way, next we focus on one specific machine $\mathcal{M}$ .

如前文所述，MapReduce算法按轮次执行，每一轮包含三个阶段：映射（map）、洗牌（shuffle）和归约（reduce）。由于所有机器以相同方式执行程序，接下来我们聚焦于一台特定的机器$\mathcal{M}$。

Map. In this phase, $\mathcal{M}$ generates a list of key-value pairs(k,v) from its local storage. While the key $k$ is usually numeric,the value $v$ can contain arbitrary information. As clarified shortly,the pair(k,v)will be transmitted to another machine in the shuffle phase,such that the recipient machine is determined solely by $k$ .

映射（Map）。在此阶段，$\mathcal{M}$从其本地存储生成一个键值对列表(k, v)。键$k$通常为数值型，而值$v$可以包含任意信息。稍后会说明，键值对(k, v)将在洗牌阶段传输到另一台机器，接收机器仅由$k$决定。

Shuffle. Let $L$ be the list of key-value pairs that all the machines produced in the map phase. The shuffle phase distributes $L$ across the machines adhering to the constraint that, pairs with the same key must be delivered to the same machine. That is, if $\left( {k,{v}_{1}}\right) ,\left( {k,{v}_{2}}\right) ,\ldots ,\left( {k,{v}_{x}}\right)$ are the pairs in $L$ having a common key $k$ ,all of them will arrive at an identical machine.

洗牌（Shuffle）。设$L$为所有机器在映射阶段生成的键值对列表。洗牌阶段将$L$分配到各机器，遵循相同键的键值对必须发送到同一台机器的约束。也就是说，如果$\left( {k,{v}_{1}}\right) ,\left( {k,{v}_{2}}\right) ,\ldots ,\left( {k,{v}_{x}}\right)$是$L$中具有相同键$k$的键值对，它们都将到达同一台机器。

Reduce. $\mathcal{M}$ incorporates the key-value pairs received from the previous phase into its local storage. Then, it carries out whatever processing as needed on its local data. After all machines have completed the reduce phase, the current round terminates.

归约（Reduce）。$\mathcal{M}$将从上一阶段接收到的键值对合并到其本地存储中。然后，它对本地数据进行所需的处理。所有机器完成归约阶段后，当前轮次结束。

Discussion. It is clear from the above that, the machines communicate only in the shuffle phase, whereas in the other phases each machine executes the algorithm sequentially, focusing on its own storage. Overall, parallel computing happens mainly in reduce. The major role of map and shuffle is to swap data among the machines, so that computation can take place on different combinations of objects.

讨论。从上述内容可以明显看出，机器仅在洗牌阶段进行通信，而在其他阶段，每台机器按顺序执行算法，专注于自身存储。总体而言，并行计算主要发生在归约阶段。映射和洗牌的主要作用是在机器之间交换数据，以便对不同的对象组合进行计算。

Simplified View for Our Algorithms. Let us number the $t$ machines of the MapReduce system arbitrarily from 1 to $t$ . In the map phase,all our algorithms will adopt the convention that $\mathcal{M}$ generates a key-value pair(k,v)if and only if it wants to send $v$ to machine $k$ . In other words,the key field is explicitly the id of the recipient machine.

我们算法的简化视图。我们任意地将MapReduce系统的$t$台机器从1到$t$编号。在映射阶段，我们所有的算法都遵循这样的约定：当且仅当$\mathcal{M}$想将$v$发送到机器$k$时，它才生成一个键值对(k, v)。换句话说，键字段明确为接收机器的ID。

This convention admits a conceptually simpler modeling. In describing our algorithms, we will combine the map and shuffle phases into one called map-shuffle. By saying succinctly that "in the map-shuffle phase, $\mathcal{M}$ delivers $v$ to machine $k$ ",we mean that $\mathcal{M}$ creates(k,v)in the map phase,which is then transmitted to machine $k$ in the shuffle phase. The equivalence also explains why the simplification is only at the logical level, while physically all our algorithms are still implemented in the standard MapReduce paradigm.

这一约定允许进行概念上更简单的建模。在描述我们的算法时，我们将映射和洗牌阶段合并为一个称为映射 - 洗牌（map - shuffle）的阶段。简洁地说，“在映射 - 洗牌阶段，$\mathcal{M}$将$v$发送到机器$k$”，意味着$\mathcal{M}$在映射阶段创建(k, v)，然后在洗牌阶段将其传输到机器$k$。这种等价性也解释了为什么这种简化仅在逻辑层面，而实际上我们所有的算法仍在标准的MapReduce范式中实现。

Statelessness for Fault Tolerance. Some MapReduce implementations (e.g., Hadoop) place the requirement that, at the end of a round, each machine should send all the data in its storage to a distributed file system (DFS), which in our context can be understood as a "disk in the cloud" that guarantees consistent storage (i.e., it never fails). The objective is to improve the system's robustness in the scenario where a machine collapses during the algorithm's execution. In such a case, the system can replace this machine with another one, ask the new machine to load the storage of the old machine at the end of the previous round, and re-do the current round (where the machine failure occurred). Such a system is called stateless because intuitively no machine is responsible for remembering any state of the algorithm [58].

无状态以实现容错。一些MapReduce实现（例如Hadoop）要求在一轮结束时，每台机器应将其存储中的所有数据发送到分布式文件系统（DFS），在我们的语境中，这可以理解为一个保证一致存储（即永不失败）的“云端磁盘”。其目标是在算法执行过程中机器崩溃的情况下提高系统的鲁棒性。在这种情况下，系统可以用另一台机器替换这台故障机器，要求新机器加载上一轮结束时旧机器的存储数据，并重新执行当前轮次（即发生机器故障的轮次）。这样的系统被称为无状态系统，因为直观上没有机器负责记住算法的任何状态[58]。

The four minimality conditions defined in Section 1 ensure efficient enforcement of statelessness. In particular, minimum footprint guarantees that,at each round,every machine sends $O\left( m\right)$ words to the DFS, as is still consistent with bounded traffic.

第1节中定义的四个最小化条件确保了无状态性的有效实施。特别是，最小占用空间保证了在每一轮中，每台机器向DFS发送$O\left( m\right)$个单词，这仍然符合流量限制。

### 2.2 Previous Research on MapReduce

### 2.2 MapReduce的过往研究

The existing investigation on MapReduce can be coarsely classified into two categories, which focus on improving the internal working of the framework, and employing MapReduce to solve concrete problems, respectively. In the sequel, we survey each category separately.

目前对MapReduce（映射归约）的研究大致可分为两类，分别侧重于改进该框架的内部工作机制，以及利用MapReduce解决具体问题。接下来，我们将分别对这两类研究进行综述。

Framework Implementation. Hadoop is perhaps the most popular open-source implementation of MapReduce nowadays. It was first described by Abouzeid et al. [1], and has been improved significantly by the collective findings of many studies. Specifically, Dittrich et al. [18] provided various user-defined functions that can substantially reduce the running time of MapReduce programs. Nykiel et al. [47], Elghandour and Agoulnaga [19] achieved further performance gains by allowing a subsequent round of an algorithm to re-use the outputs of the previous rounds. Eltabakh et al. [20] and He et al. [27] discussed the importance of keeping relevant data at the same machine in order to reduce network traffic. Floratou et al. [22] presented a column-based implementation and demonstrated superior performance in certain environments. Shinnar et al. [53] proposed to eliminate disk I/Os by fitting data in memory as much as possible. Gufler et al. [26], Kolb et al. [33], and Kwon et al. [36] designed methods to rectify skewness, i.e., imbalance in the workload of different machines.

框架实现。Hadoop（哈adoop）可能是目前最流行的MapReduce开源实现。它最初由Abouzeid等人[1]提出，并在众多研究成果的共同推动下得到了显著改进。具体而言，Dittrich等人[18]提供了各种用户自定义函数，这些函数可以大幅缩短MapReduce程序的运行时间。Nykiel等人[47]、Elghandour和Agoulnaga[19]通过允许算法的后续轮次复用前一轮次的输出来进一步提高性能。Eltabakh等人[20]和He等人[27]讨论了将相关数据存储在同一台机器上以减少网络流量的重要性。Floratou等人[22]提出了一种基于列的实现方式，并证明了其在某些环境下具有卓越的性能。Shinnar等人[53]建议尽可能将数据存储在内存中以消除磁盘I/O。Gufler等人[26]、Kolb等人[33]和Kwon等人[36]设计了纠正数据倾斜（即不同机器工作负载不平衡）的方法。

Progress has been made towards building an execution optimizer that can automatically coordinate different components of the system for the best overall efficiency. The approach of Herodotou and Babu [28] is based on profiling the cost of a MapReduce program. Jahani et al. [29] proposed a strategy that works by analyzing the programming logic of MapReduce codes. Lim et al. [40] focused on optimizing as a whole multiple MapReduce programs that are interconnected by a variety of factors.

在构建执行优化器方面已经取得了进展，该优化器可以自动协调系统的不同组件以实现最佳的整体效率。Herodotou和Babu[28]的方法基于对MapReduce程序成本的分析。Jahani等人[29]提出了一种通过分析MapReduce代码编程逻辑来工作的策略。Lim等人[40]专注于对由多种因素相互关联的多个MapReduce程序进行整体优化。

There has also been development of administration tools for MapReduce systems. Lang and Patel [37] suggested strategies for minimizing energy consumption. Morton et al. [46] devised techniques for estimating the progress (in completion percentage) of a MapReduce program. Khoussainova et al. [32] presented a mechanism to facilitate the debugging of MapReduce programs.

MapReduce系统的管理工具也得到了发展。Lang和Patel[37]提出了最小化能耗的策略。Morton等人[46]设计了估算MapReduce程序完成进度（以完成百分比表示）的技术。Khoussainova等人[32]提出了一种便于调试MapReduce程序的机制。

MapReduce, which after all is a computing framework, lacks many features of a database. One, in particular, is an expressive language that allows users to describe queries supportable by MapReduce. To fill this void, a number of languages have been designed, together with the corresponding translators that convert a query to a MapReduce program. Examples include SCOPE [9], Pig [49], Dremel [43], HIVE [55], Jaql [6], Tenzing [10], and SystemML [24].

毕竟MapReduce是一个计算框架，它缺乏数据库的许多特性。特别是，它缺少一种表达性强的语言，允许用户描述MapReduce可支持的查询。为了填补这一空白，人们设计了多种语言，以及将查询转换为MapReduce程序的相应翻译器。例如SCOPE[9]、Pig[49]、Dremel[43]、HIVE[55]、Jaql[6]、Tenzing[10]和SystemML[24]。

Algorithms on MapReduce. Considerable work has been devoted to processing joins on relational data. Blanas et al. [7] compared the implementations of traditional join algorithms in MapReduce. Afrati and Ullman [3] provided specialized algorithms for multiway equi-joins. Lin et al. [41] tackled the same problem utilizing column-based storage. Okcan and Riedewald [48] devised algorithms for reporting the cartesian product of two tables. Zhang et al. [62] discussed efficient processing of multiway theta-joins.

MapReduce上的算法。在处理关系数据的连接操作方面已经开展了大量工作。Blanas等人[7]比较了传统连接算法在MapReduce中的实现。Afrati和Ullman[3]为多路等值连接提供了专门的算法。Lin等人[41]利用基于列的存储方式解决了同样的问题。Okcan和Riedewald[48]设计了报告两个表笛卡尔积的算法。Zhang等人[62]讨论了多路θ连接的高效处理方法。

Regarding joins on non-relational data, Vernica et al. [59], Metwally and Faloutsos [44] studied set-similarity join. Afrati et al. [2] re-visited this problem and its variants under the constraint that an algorithm must terminate in a single round. Lu et al. [42], on the other hand,investigated $k$ nearest neighbor join in Euclidean space.

关于非关系数据的连接操作，Vernica等人[59]、Metwally和Faloutsos[44]研究了集合相似度连接。Afrati等人[2]在算法必须在单轮内终止的约束下重新探讨了这个问题及其变体。另一方面，Lu等人[42]研究了欧几里得空间中的$k$最近邻连接。

MapReduce has been proven useful for processing massive graphs. Suri, Vassilvitskii [54], and Tsourakakis et al. [56] considered triangle counting, Morales et al. [45] dealt with b-matching, Bahmani et al. [5] focused on the discovery of densest subgraphs, Karloff et al. [31] analyzed computing connected components and spanning trees, while Lattanzi et al. [39] studied maximal matching, vertex/edge cover, and minimum cut.

事实证明，MapReduce在处理大规模图数据方面非常有用。Suri、Vassilvitskii[54]和Tsourakakis等人[56]考虑了三角形计数问题，Morales等人[45]处理了b匹配问题，Bahmani等人[5]专注于发现最密集子图，Karloff等人[31]分析了计算连通分量和生成树的问题，而Lattanzi等人[39]研究了最大匹配、顶点/边覆盖和最小割问题。

Data mining and statistical analysis are also popular topics on MapReduce. Clustering was investigated by Das et al. [15], Cordeiro et al. [13], and Ene et al. [21]. Classification and regression were studied by Panda et al. [51]. Ghoting et al. [23] developed an integrated toolkit to facilitate machine learning tasks. Pansare et al. [52] and Laptev et al. [38] explained how to compute aggregates over a gigantic file. Grover and Carey [25] focused on extracting a set of samples satisfying a given predicate. Chen [11] described techniques for supporting operations of data warehouses.

数据挖掘和统计分析也是MapReduce上的热门话题。Das等人[15]、Cordeiro等人[13]和Ene等人[21]对聚类问题进行了研究。Panda等人[51]研究了分类和回归问题。Ghoting等人[23]开发了一个集成工具包以方便机器学习任务。Pansare等人[52]和Laptev等人[38]解释了如何对大型文件进行聚合计算。Grover和Carey[25]专注于提取满足给定谓词的一组样本。Chen[11]描述了支持数据仓库操作的技术。

Among the other algorithmic studies on MapReduce, Chierichetti et al. [12] attacked approximation versions of the set cover problem. Wang et al. [60] described algorithms for the simulation of real-world events. Bahmani et al. [4] proposed methods for calculating personalized page ranks. Jestes et al. [30] investigated the construction of wavelet histograms.

在其他关于MapReduce（映射规约）的算法研究中，基里切蒂（Chierichetti）等人 [12] 研究了集合覆盖问题的近似版本。王（Wang）等人 [60] 描述了用于模拟现实世界事件的算法。巴赫马尼（Bahmani）等人 [4] 提出了计算个性化网页排名的方法。杰斯特斯（Jestes）等人 [30] 研究了小波直方图的构建。

### 2.3 Relevance to Minimal Algorithms

### 2.3 与最小算法的相关性

Our study of minimal algorithms is orthogonal to the framework implementation category as mentioned in Section 2.2. Even a minimal algorithm can benefit from clever optimization at the system level. On the other hand, a minimal algorithm may considerably simplify optimization. For instance, as the minimal requirements already guarantee excellent load balancing in storage, computation, and communication, there would be little skewness to deserve specialized optimization. As another example, the cost of a minimal algorithm is by definition highly predictable, which is a precious advantage appreciated by cost-based optimizers (e.g., [28, 40]).

我们对最小算法的研究与2.2节中提到的框架实现类别是相互独立的。即使是最小算法也能从系统级的巧妙优化中受益。另一方面，最小算法可能会大大简化优化过程。例如，由于最小要求已经保证了存储、计算和通信方面的出色负载均衡，因此几乎不存在需要专门优化的偏差。再举一个例子，根据定义，最小算法的成本具有很高的可预测性，这是基于成本的优化器（例如 [28, 40]）所看重的宝贵优势。

This work belongs to the algorithms on MapReduce category. However, besides dealing with different problems, we also differ from the existing studies in that we emphasize on an algorithm's minimality. Remember that the difficulty of designing a minimal algorithm lies in excelling in all the four aspects (see Section 1) at the same time. Often times, it is easy to do well in only certain aspects (e.g., constant rounds), while losing out in the rest. Parallel algorithms on classic platforms are typically compared under multiple metrics. We believe that MapReduce should not be an exception.

这项工作属于MapReduce（映射规约）算法类别。然而，除了解决不同的问题之外，我们与现有研究的不同之处在于我们强调算法的最小性。请记住，设计最小算法的难点在于同时在四个方面（见第1节）都表现出色。很多时候，只在某些方面（例如常数轮数）表现良好，而在其他方面表现不佳是很容易的。经典平台上的并行算法通常在多个指标下进行比较。我们认为MapReduce（映射规约）也不应例外。

From a theoretical perspective, minimal algorithms are reminiscent of algorithms under the bulk synchronous parallel (BSP) model [57] and coarse-grained multicomputer (CGM) model [17]. Both models are well-studied branches of theoretical parallel computing. Our algorithmic treatment, however, is system oriented, i.e., easy to implement, while offering excellent performance in practice. In contrast, theoretical solutions in BSP/CGM are often rather involved, and usually carry large hidden constants in their complexities, not to mention that they are yet to be migrated to MapReduce. It is worth mentioning that there has been work on extending the MapReduce framework to enhance its power so as to solve difficult problems efficiently. We refer the interested readers to the recent work of [34].

从理论角度来看，最小算法让人联想到批量同步并行（BSP）模型 [57] 和粗粒度多计算机（CGM）模型 [17] 下的算法。这两种模型都是理论并行计算中经过充分研究的分支。然而，我们的算法处理方式是以系统为导向的，即易于实现，同时在实践中提供出色的性能。相比之下，BSP/CGM中的理论解决方案通常相当复杂，并且在其复杂度中通常包含较大的隐藏常数，更不用说它们还需要迁移到MapReduce（映射规约）中。值得一提的是，已经有工作致力于扩展MapReduce（映射规约）框架以增强其能力，从而有效地解决难题。我们建议感兴趣的读者参考 [34] 的最新工作。

## 3. SORTING

## 3. 排序

In the sorting problem,the input is a set $S$ of $n$ objects from an ordered domain. For simplicity, we assume that objects are real values because our discussion easily generalizes to other ordered domains. Denote by ${\mathcal{M}}_{1},\ldots ,{\mathcal{M}}_{t}$ the machines in the MapReduce system. Initially, $S$ is distributed across these machines,each storing $O\left( m\right)$ objects where $m = n/t$ . At the end of sorting,all objects in ${\mathcal{M}}_{i}$ must precede those in ${\mathcal{M}}_{j}$ for any $1 \leq  i < j \leq  t$ .

在排序问题中，输入是一个来自有序域的 $n$ 个对象的集合 $S$。为了简单起见，我们假设对象是实数值，因为我们的讨论很容易推广到其他有序域。用 ${\mathcal{M}}_{1},\ldots ,{\mathcal{M}}_{t}$ 表示MapReduce（映射规约）系统中的机器。最初，$S$ 分布在这些机器上，每台机器存储 $O\left( m\right)$ 个对象，其中 $m = n/t$ 。排序结束时，对于任何 $1 \leq  i < j \leq  t$ ，${\mathcal{M}}_{i}$ 中的所有对象必须排在 ${\mathcal{M}}_{j}$ 中的对象之前。

### 3.1 TeraSort

### 3.1 万亿级排序（TeraSort）

Parameterized by $\rho  \in  (0,1\rbrack$ ,TeraSort [50] runs as follows:

以 $\rho  \in  (0,1\rbrack$ 为参数，万亿级排序（TeraSort） [50] 运行如下：

<!-- Media -->

Round 1. Map-shuffle $\left( \rho \right)$

第1轮。映射 - 洗牌 $\left( \rho \right)$

---

		Every ${\mathcal{M}}_{i}\left( {1 \leq  i \leq  t}\right)$ samples each object from its local

		每台 ${\mathcal{M}}_{i}\left( {1 \leq  i \leq  t}\right)$ 以概率 ${\mathcal{M}}_{i}\left( {1 \leq  i \leq  t}\right)$ 独立地从其本地存储中对每个对象进行采样。

		storage with probability $\rho$ independently. It sends all the

		它将所有采样的对象发送到 $\rho$。

		sampled objects to ${\mathcal{M}}_{1}$ .

		

Reduce (only on ${\mathcal{M}}_{1}$ )

归约（仅在 ${\mathcal{M}}_{1}$ 上）

	1. Let ${S}_{\text{samp }}$ be the set of samples received by ${\mathcal{M}}_{1}$ ,and $s =$

	1. 设 ${S}_{\text{samp }}$ 是 ${\mathcal{M}}_{1}$ 接收到的样本集合，以及 $s =$

		$\left| {S}_{\text{samp }}\right|$ .

	2. Sort ${S}_{\text{samp }}$ ,and pick ${b}_{1},\ldots ,{b}_{t - 1}$ where ${b}_{i}$ is the $i\lceil s/t\rceil$ -th

	2. 对 ${S}_{\text{samp }}$ 进行排序，并选取 ${b}_{1},\ldots ,{b}_{t - 1}$ ，其中 ${b}_{i}$ 是第 $i\lceil s/t\rceil$ 个

		smallest object in ${S}_{\text{samp }}$ ,for $1 \leq  i \leq  t - 1$ . Each ${b}_{i}$ is a

		${S}_{\text{samp }}$ 中的最小对象，对于 $1 \leq  i \leq  t - 1$ 而言。每个 ${b}_{i}$ 都是一个

		boundary object.

		边界对象。

Round 2. Map-shuffle (assumption: ${b}_{1},\ldots ,{b}_{t - 1}$ have been sent to

第二轮。映射 - 洗牌（假设：${b}_{1},\ldots ,{b}_{t - 1}$ 已被发送到

all machines)

所有机器）

		Every ${\mathcal{M}}_{i}$ sends the objects in $\left( {{b}_{j - 1},{b}_{j}}\right\rbrack$ from its local

		每个 ${\mathcal{M}}_{i}$ 将其本地存储中 $\left( {{b}_{j - 1},{b}_{j}}\right\rbrack$ 里的对象发送到 ${\mathcal{M}}_{i}$，对于每个 $\left( {{b}_{j - 1},{b}_{j}}\right\rbrack$，其中 [latex2] 和

		storage to ${\mathcal{M}}_{j}$ ,for each $1 \leq  j \leq  t$ ,where ${b}_{0} =  - \infty$ and

		${\mathcal{M}}_{j}$ 是虚拟边界对象。

		${b}_{t} = \infty$ are dummy boundary objects.

		${b}_{t} = \infty$ 是虚拟边界对象。

## Reduce:

## 归约：

		Every ${\mathcal{M}}_{i}$ sorts the objects received in the previous phase.

		每个 ${\mathcal{M}}_{i}$ 对前一阶段收到的对象进行排序。

---

<!-- Media -->

For convenience, the above description sometimes asks a machine $\mathcal{M}$ to send data to itself. Needless to say,such data "transfer" occurs internally in $\mathcal{M}$ ,with no network transmission. Also note the assumption at the map-shuffle phase of Round 2, which we call the broadcast assumption, and will deal with later in Section 3.3.

为方便起见，上述描述有时会要求机器 $\mathcal{M}$ 将数据发送给自己。不用说，这种数据“传输”是在 $\mathcal{M}$ 内部进行的，无需网络传输。另请注意第二轮映射 - 洗牌阶段的假设，我们称之为广播假设，将在 3.3 节中进一步讨论。

In [50], $\rho$ was left as an open parameter. Next,we analyze the setting of this value to make TeraSort a minimal algorithm.

在文献 [50] 中，$\rho$ 是一个未确定的参数。接下来，我们分析该值的设置，以使万亿级排序（TeraSort）成为一种最优算法。

### 3.2 Choice of $\rho$

### 3.2 $\rho$ 的选择

Define ${S}_{i} = S \cap  \left( {{b}_{i - 1},{b}_{i}}\right\rbrack$ ,for $1 \leq  i \leq  t$ . In Round 2,all the objects in ${S}_{i}$ are gathered by ${\mathcal{M}}_{i}$ ,which sorts them in the reduce phase. For TeraSort to be minimal, it must hold:

定义 ${S}_{i} = S \cap  \left( {{b}_{i - 1},{b}_{i}}\right\rbrack$，对于 $1 \leq  i \leq  t$ 而言。在第二轮中，${S}_{i}$ 中的所有对象都由 ${\mathcal{M}}_{i}$ 收集，${\mathcal{M}}_{i}$ 在归约阶段对它们进行排序。为使万亿级排序（TeraSort）达到最优，必须满足：

$$
{\mathcal{P}}_{1} \cdot  s = O\left( m\right) .
$$

$$
{\mathcal{P}}_{2}.\left| {S}_{i}\right|  = O\left( m\right) \text{for all}1 \leq  i \leq  t\text{.}
$$

Specifically, ${\mathcal{P}}_{1}$ is because ${\mathcal{M}}_{1}$ receives $O\left( s\right)$ objects over the network in the map-shuffle phase of Round 1, which has to be $O\left( m\right)$ to satisfy bounded net-traffic (see Section 1). ${\mathcal{P}}_{2}$ is because ${\mathcal{M}}_{i}$ must receive and store $O\left( \left| {S}_{i}\right| \right)$ words in Round 2,which needs to be $O\left( m\right)$ to qualify bounded net-traffic and minimum footprint.

具体而言，${\mathcal{P}}_{1}$ 是因为 ${\mathcal{M}}_{1}$ 在第一轮的映射 - 洗牌阶段通过网络接收了 $O\left( s\right)$ 个对象，为满足有限网络流量的要求（见第 1 节），这些对象数量必须为 $O\left( m\right)$。${\mathcal{P}}_{2}$ 是因为 ${\mathcal{M}}_{i}$ 在第二轮必须接收并存储 $O\left( \left| {S}_{i}\right| \right)$ 个单词，为符合有限网络流量和最小占用空间的要求，这些单词数量需要为 $O\left( m\right)$。

We now establish an important fact about TeraSort:

我们现在建立关于万亿级排序（TeraSort）的一个重要事实：

THEOREM 1. When $m \geq  t\ln \left( {nt}\right) ,{\mathcal{P}}_{1}$ and ${\mathcal{P}}_{2}$ hold simultaneously with probability at least $1 - O\left( \frac{1}{n}\right)$ by setting $\rho  =$ $\frac{1}{m}\ln \left( {nt}\right)$ .

定理 1。当通过设置 $\rho  =$ $\frac{1}{m}\ln \left( {nt}\right)$ 使得 $m \geq  t\ln \left( {nt}\right) ,{\mathcal{P}}_{1}$ 和 ${\mathcal{P}}_{2}$ 同时成立的概率至少为 $1 - O\left( \frac{1}{n}\right)$ 时。

Proof. We will consider $t \geq  9$ because otherwise $m = \Omega \left( n\right)$ , in which case ${\mathcal{P}}_{1}$ and ${\mathcal{P}}_{2}$ hold trivially. Our proof is based on the Chernoff bound ${}^{1}$ and an interesting bucketing argument.

证明。我们将考虑$t \geq  9$，因为否则$m = \Omega \left( n\right)$，在这种情况下，${\mathcal{P}}_{1}$和${\mathcal{P}}_{2}$显然成立。我们的证明基于切尔诺夫界（Chernoff bound）${}^{1}$和一个有趣的分桶论证。

First,it is easy to see that $\mathbf{E}\left\lbrack  s\right\rbrack   = {m\rho t} = t\ln \left( {nt}\right)$ . A simple application of Chernoff bound results in:

首先，很容易看出$\mathbf{E}\left\lbrack  s\right\rbrack   = {m\rho t} = t\ln \left( {nt}\right)$。简单应用切尔诺夫界可得：

$$
\Pr \left\lbrack  {s \geq  {1.6} \cdot  t\ln \left( {nt}\right) }\right\rbrack   \leq  \exp \left( {-{0.12} \cdot  t\ln \left( {nt}\right) }\right)  \leq  1/n
$$

where the last inequality used the fact that $t \geq  9$ . The above implies that ${\mathcal{P}}_{1}$ can fail with probability at most $1/n$ . Next,we analyze ${\mathcal{P}}_{2}$ under the event $s < {1.6t}\ln \left( {nt}\right)  = O\left( m\right)$ .

其中最后一个不等式利用了$t \geq  9$这一事实。上述结果表明，${\mathcal{P}}_{1}$不成立的概率至多为$1/n$。接下来，我们在事件$s < {1.6t}\ln \left( {nt}\right)  = O\left( m\right)$发生的条件下分析${\mathcal{P}}_{2}$。

Imagine that $S$ has been sorted in ascending order. We divide the sorted list into $\lfloor t/8\rfloor$ sub-lists as evenly as possible,and call each sub-list a bucket. Each bucket has between ${8n}/t = {8m}$ and ${16m}$ objects. We observe that ${\mathcal{P}}_{2}$ holds if every bucket covers at least one boundary object. To understand why, notice that under this condition, no bucket can fall between two consecutive boundary objects (counting also the dummy ones) ${}^{2}$ . Hence,every ${S}_{i},1 \leq$ $i \leq  t$ ,can contain objects in at most 2 buckets,i.e., $\left| {S}_{i}\right|  \leq  {32m} =$ $O\left( m\right)$ .

假设 $S$ 已按升序排序。我们将排序后的列表尽可能均匀地划分为 $\lfloor t/8\rfloor$ 个子列表，并将每个子列表称为一个桶。每个桶包含的对象数量在 ${8n}/t = {8m}$ 到 ${16m}$ 之间。我们观察到，如果每个桶至少覆盖一个边界对象，则 ${\mathcal{P}}_{2}$ 成立。要理解原因，请注意，在这种条件下，没有桶可以落在两个连续的边界对象（也包括虚拟对象）${}^{2}$ 之间。因此，每个 ${S}_{i},1 \leq$ $i \leq  t$ 最多可以包含 2 个桶中的对象，即 $\left| {S}_{i}\right|  \leq  {32m} =$ $O\left( m\right)$。

A bucket $\beta$ definitely includes a boundary object if $\beta$ covers more than ${1.6}\ln \left( {nt}\right)  > s/t$ samples (i.e.,objects from ${S}_{\text{samp }}$ ), as a boundary object is taken every $\lceil s/t\rceil$ consecutive samples. Let $\left| \beta \right|  \geq  {8m}$ be the number of objects in $\beta$ . Define random variable ${x}_{j},1 \leq  j \leq  \left| \beta \right|$ ,to be 1 if the $j$ -th object in $\beta$ is sampled,and 0 otherwise. Define:

如果桶 $\beta$ 覆盖的样本数量超过 ${1.6}\ln \left( {nt}\right)  > s/t$（即来自 ${S}_{\text{samp }}$ 的对象），则该桶肯定包含一个边界对象，因为每 $\lceil s/t\rceil$ 个连续样本中会选取一个边界对象。设 $\left| \beta \right|  \geq  {8m}$ 为桶 $\beta$ 中的对象数量。定义随机变量 ${x}_{j},1 \leq  j \leq  \left| \beta \right|$，如果桶 $\beta$ 中的第 $j$ 个对象被采样，则 ${x}_{j},1 \leq  j \leq  \left| \beta \right|$ 为 1，否则为 0。定义：

$$
X = \mathop{\sum }\limits_{{j = 1}}^{\left| \beta \right| }{x}_{j} = \left| {\beta  \cap  {S}_{\text{samp }}}\right| .
$$

Clearly, $\mathbf{E}\left\lbrack  X\right\rbrack   \geq  {8m\rho } = 8\ln \left( {nt}\right)$ . We have:

显然，$\mathbf{E}\left\lbrack  X\right\rbrack   \geq  {8m\rho } = 8\ln \left( {nt}\right)$。我们有：

$$
\Pr \left\lbrack  {X \leq  {1.6}\ln \left( {nt}\right) }\right\rbrack   = \Pr \left\lbrack  {X \leq  \left( {1 - 4/5}\right) 8\ln \left( {nt}\right) }\right\rbrack  
$$

$$
 \leq  \Pr \left\lbrack  {X \leq  \left( {1 - 4/5}\right) \mathbf{E}\left\lbrack  X\right\rbrack  }\right\rbrack  
$$

$$
\text{(by Chernoff)} \leq  \exp \left( {-\frac{16}{25}\frac{\mathbf{E}\left\lbrack  X\right\rbrack  }{3}}\right) 
$$

$$
 \leq  \exp \left( {-\frac{16}{25} \cdot  \frac{8\ln \left( {nt}\right) }{3}}\right) 
$$

$$
 \leq  \exp \left( {-\ln \left( {nt}\right) }\right) 
$$

$$
 \leq  1/\left( {nt}\right) \text{.}
$$

We say that $\beta$ fails if it covers no boundary object. The above derivation shows that $\beta$ fails with probability at most $1/\left( {nt}\right)$ . As there are at most $t/8$ buckets,the probability that at least one bucket fails is at most $1/\left( {8n}\right)$ . Hence, ${\mathcal{P}}_{2}$ can be violated with probability at most $1/\left( {8n}\right)$ under the event $s < {1.6t}\ln \left( {nt}\right)$ ,i.e.,at most $9/{8n}$ overall.

我们称如果桶 $\beta$ 不覆盖任何边界对象，则该桶失败。上述推导表明，桶 $\beta$ 失败的概率至多为 $1/\left( {nt}\right)$。由于最多有 $t/8$ 个桶，至少有一个桶失败的概率至多为 $1/\left( {8n}\right)$。因此，在事件 $s < {1.6t}\ln \left( {nt}\right)$ 下，${\mathcal{P}}_{2}$ 被违反的概率至多为 $1/\left( {8n}\right)$，即总体上至多为 $9/{8n}$。

Therefore, ${\mathcal{P}}_{1}$ and ${\mathcal{P}}_{2}$ hold at the same time with probability at least $1 - {17}/\left( {8n}\right)$ .

因此，${\mathcal{P}}_{1}$ 和 ${\mathcal{P}}_{2}$ 同时成立的概率至少为 $1 - {17}/\left( {8n}\right)$。

Discussion. For large $n$ ,the success probability $1 - O\left( {1/n}\right)$ in Theorem 1 is so high that the failure probability $O\left( {1/n}\right)$ is negligible,i.e., ${\mathcal{P}}_{1}$ and ${\mathcal{P}}_{2}$ are almost never violated.

讨论。对于较大的 $n$，定理 1 中的成功概率 $1 - O\left( {1/n}\right)$ 非常高，以至于失败概率 $O\left( {1/n}\right)$ 可以忽略不计，即 ${\mathcal{P}}_{1}$ 和 ${\mathcal{P}}_{2}$ 几乎不会被违反。

The condition about $m$ in Theorem 1 is tight within a logarithmic factor because $m \geq  t$ is an implicit condition for TeraSort to work, noticing that both the reduce phase of Round 1 and the map-shuffle phase of Round 2 require a machine to store $t - 1$ boundary objects.

定理 1 中关于 $m$ 的条件在对数因子范围内是严格的，因为 $m \geq  t$ 是 TeraSort 算法能够正常工作的一个隐含条件，注意到第一轮的归约阶段和第二轮的映射 - 洗牌阶段都要求一台机器存储 $t - 1$ 个边界对象。

In reality,typically $m \gg  t$ ,namely,the memory size of a machine is significantly greater than the number of machines. More specifically, $m$ is at the order of at least ${10}^{6}$ (this is using only a few mega bytes per machine),while $t$ is at the order of ${10}^{4}$ or lower. Therefore, $m \geq  t\ln \left( {nt}\right)$ is a (very) reasonable assumption,which explains why TeraSort has excellent efficiency in practice.

实际上，通常 $m \gg  t$，即一台机器的内存大小明显大于机器的数量。更具体地说，$m$ 至少是 ${10}^{6}$ 这个数量级（这仅需每台机器使用几兆字节的内存），而 $t$ 是 ${10}^{4}$ 或更低的数量级。因此，$m \geq  t\ln \left( {nt}\right)$ 是一个（非常）合理的假设，这解释了为什么 TeraSort 在实践中具有出色的效率。

Minimality. We now establish the minimality of TeraSort, temporarily ignoring how to fulfill the broadcast assumption. Properties ${\mathcal{P}}_{1}$ and ${\mathcal{P}}_{2}$ indicate that each machine needs to store only $O\left( m\right)$ objects at any time,consistent with minimum footprint. Regarding the network cost,a machine $\mathcal{M}$ in each round sends only objects that were already on $\mathcal{M}$ when the algorithm started. Hence, $\mathcal{M}$ sends $O\left( m\right)$ network data per round. Furthermore, ${\mathcal{M}}_{1}$ receives only $O\left( m\right)$ objects by ${\mathcal{P}}_{1}$ . Therefore,bounded-bandwidth is fulfilled. Constant round is obviously satisfied. Finally, the computation time of each machine ${\mathcal{M}}_{i}\left( {1 \leq  i \leq  t}\right)$ is dominated by the cost of sorting ${S}_{i}$ in Round 2,i.e., $O\left( {m\log m}\right)  = O\left( {\frac{n}{t}\log n}\right)$ by ${\mathcal{P}}_{2}$ . As this is $1/t$ of the $O\left( {n\log n}\right)$ time of a sequential algorithm, optimal computation is also achieved.

极小性。我们现在来证明TeraSort算法的极小性，暂时忽略如何满足广播假设。性质${\mathcal{P}}_{1}$和${\mathcal{P}}_{2}$表明，任何时候每台机器只需存储$O\left( m\right)$个对象，这与最小占用空间的要求相符。关于网络成本，每一轮中机器$\mathcal{M}$仅发送算法开始时就已存在于$\mathcal{M}$上的对象。因此，$\mathcal{M}$每轮发送$O\left( m\right)$的网络数据。此外，根据${\mathcal{P}}_{1}$，${\mathcal{M}}_{1}$仅接收$O\left( m\right)$个对象。所以，满足有界带宽的要求。显然，常数轮数的条件也能满足。最后，每台机器${\mathcal{M}}_{i}\left( {1 \leq  i \leq  t}\right)$的计算时间主要取决于第二轮中对${S}_{i}$进行排序的成本，即根据${\mathcal{P}}_{2}$为$O\left( {m\log m}\right)  = O\left( {\frac{n}{t}\log n}\right)$。由于这是顺序算法$O\left( {n\log n}\right)$时间的$1/t$，因此也实现了最优计算。

### 3.3 Removing the Broadcast Assumption

### 3.3 消除广播假设

Before Round 2 of TeraSort, ${\mathcal{M}}_{1}$ needs to broadcast the boundary objects ${b}_{1},\ldots ,{b}_{t - 1}$ to the other machines. We have to be careful because a naive solution would ask ${\mathcal{M}}_{1}$ to send $O\left( t\right)$ words to every other machine,and hence,incur $O\left( {t}^{2}\right)$ network traffic overall. This not only requires one more round, but also violates bounded net-traffic if $t$ exceeds $\sqrt{m}$ by a non-constant factor.

在TeraSort算法的第二轮之前，${\mathcal{M}}_{1}$需要将边界对象${b}_{1},\ldots ,{b}_{t - 1}$广播给其他机器。我们必须谨慎行事，因为简单的解决方案会要求${\mathcal{M}}_{1}$向其他每台机器发送$O\left( t\right)$个单词，因此，总体上会产生$O\left( {t}^{2}\right)$的网络流量。这不仅需要额外一轮，而且如果$t$超过$\sqrt{m}$的倍数不是常数，还会违反有界网络流量的要求。

In [50], this issue was circumvented by assuming that all the machines can access a distributed file system. In this scenario, ${\mathcal{M}}_{1}$ can simply write the boundary objects to a file on that system, after which each ${M}_{i},2 \leq  i \leq  t$ ,gets them from the file. In other words, a brute-force file accessing step is inserted between the two rounds. This is allowed by the current Hadoop implementation (on which TeraSort was based [50]).

在文献[50]中，通过假设所有机器都可以访问分布式文件系统来规避这个问题。在这种情况下，${\mathcal{M}}_{1}$可以简单地将边界对象写入该系统上的一个文件，然后每台${M}_{i},2 \leq  i \leq  t$从该文件中获取这些对象。换句话说，在两轮之间插入了一个简单的文件访问步骤。当前的Hadoop实现（TeraSort算法基于此实现[50]）允许这样做。

Technically, however, the above approach destroys the elegance of TeraSort because it requires that, besides sending key-value pairs to each other, the machines should also communicate via a distributed file. This implies that the machines are not share-nothing because they are essentially sharing the file Furthermore, as far as this paper is concerned, the artifact is inconsistent with the definition of minimal algorithms. As sorting lingers in all the problems to be discussed later, we are motivated to remove the artifact to keep our analytical framework clean.

然而，从技术上讲，上述方法破坏了TeraSort算法的简洁性，因为它要求机器除了相互发送键值对之外，还应该通过分布式文件进行通信。这意味着这些机器并非无共享架构，因为它们实际上在共享该文件。此外，就本文而言，这种做法与极小算法的定义不一致。由于排序问题贯穿于后续要讨论的所有问题中，我们有动力消除这种做法，以保持我们分析框架的简洁性。

We now provide an elegant remedy, which allows TeraSort to still terminate in 2 rounds, and retain its minimality. The idea is to give all machines a copy of ${S}_{\text{samp }}$ . Specifically,we modify Round 1 of TeraSort as:

我们现在提供一种简洁的解决方案，使TeraSort算法仍然可以在两轮内终止，并保持其极小性。其思路是为所有机器提供一份${S}_{\text{samp }}$的副本。具体来说，我们将TeraSort算法的第一轮修改为：

<!-- Media -->

---

Round 1. Map-shuffle $\left( \rho \right)$

第一轮。映射 - 洗牌$\left( \rho \right)$

	After sampling as in TeraSort,each ${\mathcal{M}}_{i}$ sends its sampled

	在像TeraSort算法那样进行采样之后，每台${\mathcal{M}}_{i}$将其采样的

	objects to all machines (not just to ${\mathcal{M}}_{1}$ ).

	对象发送给所有机器（而不仅仅是发送给${\mathcal{M}}_{1}$）。

Reduce

归约

	Same as TeraSort but performed on all machines (not just on

	与TeraSort算法相同，但在所有机器上执行（而不仅仅是在

	$\left. {\mathcal{M}}_{1}\right)$ .

---

<!-- Media -->

---

<!-- Footnote -->

${}^{1}$ Let ${X}_{1},\ldots ,{X}_{n}$ be independent Bernoulli variables with $\Pr \left\lbrack  {{X}_{i} = }\right.$ $1\rbrack  = {p}_{i}$ ,for $1 \leq  i \leq  n$ . Set $X = \mathop{\sum }\limits_{{i = 1}}^{n}{X}_{i}$ and $\mu  = \mathbf{E}\left\lbrack  X\right\rbrack   =$ $\mathop{\sum }\limits_{{i = 1}}^{n}{p}_{i}$ . The Chernoff bound states (i) for any $0 < \alpha  < 1$ , $\Pr \left\lbrack  {X \geq  \left( {1 + \alpha }\right) \mu }\right\rbrack   \leq  \exp \left( {-{\alpha }^{2}\mu /3}\right)$ while $\Pr \left\lbrack  {X \leq  \left( {1 - \alpha }\right) \mu }\right\rbrack   \leq$ $\exp \left( {-{\alpha }^{2}\mu /3}\right)$ ,and (ii) $\Pr \left\lbrack  {X \geq  {6\mu }}\right\rbrack   \leq  {2}^{-{6\mu }}$ .

${}^{1}$ 设 ${X}_{1},\ldots ,{X}_{n}$ 为独立的伯努利变量（Bernoulli variables），其中 $\Pr \left\lbrack  {{X}_{i} = }\right.$ $1\rbrack  = {p}_{i}$ ，对于 $1 \leq  i \leq  n$ 。令 $X = \mathop{\sum }\limits_{{i = 1}}^{n}{X}_{i}$ 且 $\mu  = \mathbf{E}\left\lbrack  X\right\rbrack   =$ $\mathop{\sum }\limits_{{i = 1}}^{n}{p}_{i}$ 。切尔诺夫界（Chernoff bound）表明：（i）对于任意 $0 < \alpha  < 1$ ， $\Pr \left\lbrack  {X \geq  \left( {1 + \alpha }\right) \mu }\right\rbrack   \leq  \exp \left( {-{\alpha }^{2}\mu /3}\right)$ ，而 $\Pr \left\lbrack  {X \leq  \left( {1 - \alpha }\right) \mu }\right\rbrack   \leq$ $\exp \left( {-{\alpha }^{2}\mu /3}\right)$ ；（ii） $\Pr \left\lbrack  {X \geq  {6\mu }}\right\rbrack   \leq  {2}^{-{6\mu }}$ 。

${}^{2}$ If there was one,the bucket would not be able to cover any boundary object.

${}^{2}$ 如果存在一个（边界对象），那么桶将无法覆盖任何边界对象。

<!-- Footnote -->

---

Round 2 still proceeds as before. The correctness follows from the fact that, in the reduce phase, every machine picks boundary objects in exactly the same way from an identical ${S}_{\text{samp }}$ . Therefore, all machines will obtain the same boundary objects, thus eliminating the need of broadcasting. Henceforth, we will call the modified algorithm pure TeraSort.

第二轮仍按之前的方式进行。其正确性源于以下事实：在归约阶段，每台机器都以完全相同的方式从相同的 ${S}_{\text{samp }}$ 中选取边界对象。因此，所有机器都会得到相同的边界对象，从而无需进行广播。此后，我们将这种改进后的算法称为纯TeraSort算法。

At first glance, the new map-shuffle phase of Round 1 may seem to require a machine $\mathcal{M}$ to send out considerable data,because every sample necessitates $O\left( t\right)$ words of network traffic (i.e., $O\left( 1\right)$ to every other machine). However, as every object is sampled with probability $\rho  = \frac{1}{m}\ln \left( {nt}\right)$ ,the number of words sent by $\mathcal{M}$ is only $O\left( {m \cdot  t \cdot  \rho }\right)  = O\left( {t\ln \left( {nt}\right) }\right)$ in expectation. The lemma below gives a much stronger fact:

乍一看，第一轮新的映射 - 洗牌阶段似乎要求机器 $\mathcal{M}$ 发送大量数据，因为每个样本都需要 $O\left( t\right)$ 字的网络流量（即向其他每台机器发送 $O\left( 1\right)$ ）。然而，由于每个对象被采样的概率为 $\rho  = \frac{1}{m}\ln \left( {nt}\right)$ ，机器 $\mathcal{M}$ 发送的字数期望仅为 $O\left( {m \cdot  t \cdot  \rho }\right)  = O\left( {t\ln \left( {nt}\right) }\right)$ 。下面的引理给出了一个更强的结论：

LEMMA 1. With probability at least $1 - \frac{1}{n}$ ,every machine sends $O\left( {t\ln \left( {nt}\right) }\right)$ words over the network in Round 1 of pure TeraSort.

引理1. 在纯TeraSort算法的第一轮中，每台机器通过网络发送 $O\left( {t\ln \left( {nt}\right) }\right)$ 字的概率至少为 $1 - \frac{1}{n}$ 。

Proof. Consider an arbitrary machine $\mathcal{M}$ . Let random variable $X$ be the number of objects sampled from $\mathcal{M}$ . Hence, $\mathbf{E}\left\lbrack  X\right\rbrack   =$ ${m\rho } = \ln \left( {nt}\right)$ . A straightforward application of Chernoff bound gives:

证明. 考虑任意一台机器 $\mathcal{M}$ 。设随机变量 $X$ 为从 $\mathcal{M}$ 中采样的对象数量。因此， $\mathbf{E}\left\lbrack  X\right\rbrack   =$ ${m\rho } = \ln \left( {nt}\right)$ 。直接应用切尔诺夫界（Chernoff bound）可得：

$$
\Pr \left\lbrack  {X \geq  6\ln \left( {nt}\right) }\right\rbrack   \leq  {2}^{-6\ln \left( {nt}\right) } \leq  1/\left( {nt}\right) .
$$

Hence, $\mathcal{M}$ sends more than $O\left( {t\ln \left( {nt}\right) }\right)$ words in Round 1 with probability at most $1/\left( {nt}\right)$ . By union bound,the probability that this is true for all $t$ machines is at least $1 - 1/n$ .

因此，机器 $\mathcal{M}$ 在第一轮中发送超过 $O\left( {t\ln \left( {nt}\right) }\right)$ 字的概率至多为 $1/\left( {nt}\right)$ 。根据联合界（union bound），所有 $t$ 台机器都满足此条件的概率至少为 $1 - 1/n$ 。

Combining the above lemma with Theorem 1 and the minimality analysis in Section 3.2, we can see that pure TeraSort is a minimal algorithm with probability at least $1 - O\left( {1/n}\right)$ when $m \geq  t\ln \left( {nt}\right)$ .

将上述引理与定理1以及3.2节中的最小性分析相结合，我们可以看出，当 $m \geq  t\ln \left( {nt}\right)$ 时，纯TeraSort算法是一种最小算法的概率至少为 $1 - O\left( {1/n}\right)$ 。

We close this section by pointing out that, the fix of TeraSort is of mainly theoretical concerns. Its purpose is to convince the reader that the broadcast assumption is not a technical "loose end" in achieving minimality. In practice, TeraSort has nearly the same performance as our pure version, at least on Hadoop where (as mentioned before) the brute-force approach of TeraSort is well supported.

在结束本节时，我们指出，TeraSort的修正主要是出于理论上的考虑。其目的是让读者相信，广播假设并非实现最小化过程中的一个技术“漏洞”。实际上，TeraSort的性能与我们的纯版本几乎相同，至少在Hadoop上是如此（如前所述），在Hadoop中，TeraSort的暴力方法得到了很好的支持。

## 4. BASIC MINIMAL ALGORITHMS IN DATABASES

## 4. 数据库中的基本最小算法

A minimal sorting algorithm also gives rise to minimal algorithms for other database problems. We demonstrate so for ranking, group-by, semi-join, and 2D skyline in this section. For all these problems, our objective is to terminate in one more round after sorting,in which a machine entails only $O\left( t\right)$ words of network traffic where $t$ is the number of machines.

一种最小排序算法也会引出用于解决其他数据库问题的最小算法。在本节中，我们将针对排名、分组、半连接和二维天际线问题进行说明。对于所有这些问题，我们的目标是在排序后的一轮内终止，在这一轮中，一台机器仅产生$O\left( t\right)$个网络流量字，其中$t$是机器的数量。

As before,each of the machines ${\mathcal{M}}_{1},\ldots ,{\mathcal{M}}_{t}$ is permitted $O\left( m\right)$ space of storage where $m = n/t$ ,and $n$ is the problem cardinality. In the rest of the paper,we will concentrate on $m \geq  t\ln \left( {nt}\right)$ ,i.e., the condition under which TeraSort is minimal (see Theorem 1).

和之前一样，每台机器${\mathcal{M}}_{1},\ldots ,{\mathcal{M}}_{t}$被允许使用$O\left( m\right)$的存储空间，其中$m = n/t$，并且$n$是问题的基数。在本文的其余部分，我们将集中讨论$m \geq  t\ln \left( {nt}\right)$，即TeraSort达到最小化的条件（见定理1）。

### 4.1 Ranking and Skyline

### 4.1 排名与天际线

Prefix Sum. Let $S$ be a set of $n$ objects from an ordered domain, such that each object $o \in  S$ carries a real-valued weight $w\left( o\right)$ . Define prefix(o,S),the prefix sum of $o$ ,to be the total weight of the objects ${o}^{\prime } \in  S$ such that ${o}^{\prime } < o$ . The prefix sum problem is to report the prefix sums of all objects in $S$ . The problem can be settled in $O\left( {n\log n}\right)$ time on a sequential machine. Next,we present an efficient MapReduce algorithm.

前缀和。设$S$是来自一个有序域的$n$个对象的集合，使得每个对象$o \in  S$都带有一个实值权重$w\left( o\right)$。定义前缀(o,S)，即$o$的前缀和，为满足${o}^{\prime } < o$的对象${o}^{\prime } \in  S$的总权重。前缀和问题是报告$S$中所有对象的前缀和。该问题可以在一台顺序机器上以$O\left( {n\log n}\right)$的时间复杂度解决。接下来，我们将介绍一种高效的MapReduce算法。

First,sort $S$ with TeraSort. Let ${S}_{i}$ be the set of objects on machine ${\mathcal{M}}_{i}$ after sorting,for $1 \leq  i \leq  t$ . We solve the prefix sum problem in another round:

首先，使用TeraSort对$S$进行排序。设${S}_{i}$是排序后机器${\mathcal{M}}_{i}$上的对象集合，其中$1 \leq  i \leq  t$。我们在另一轮中解决前缀和问题：

<!-- Media -->

---

		${\mathcal{M}}_{i}$ sends ${W}_{i} = \mathop{\sum }\limits_{{o \in  {S}_{i}}}w\left( o\right)$ to ${\mathcal{M}}_{i + 1},\ldots ,{\mathcal{M}}_{t}$ .

		${\mathcal{M}}_{i}$将${W}_{i} = \mathop{\sum }\limits_{{o \in  {S}_{i}}}w\left( o\right)$发送给${\mathcal{M}}_{i + 1},\ldots ,{\mathcal{M}}_{t}$。

Reduce (on each ${\mathcal{M}}_{i}$ ):

归约（在每台${\mathcal{M}}_{i}$上）：

	1. ${V}_{i} = \mathop{\sum }\limits_{{j \leq  i - 1}}{W}_{j}$ .

	2. Obtain prefix $\left( {o,{S}_{i}}\right)$ for $o \in  {S}_{i}$ by solving the prefix sum

	2. 通过在$\left( {o,{S}_{i}}\right)$上局部解决前缀和问题，为$o \in  {S}_{i}$获取前缀$\left( {o,{S}_{i}}\right)$。

		problem on ${S}_{i}$ locally.

		在${S}_{i}$上局部解决前缀和问题。

	3. prefix $\left( {o,S}\right)  = {V}_{i} + \operatorname{prefix}\left( {o,{S}_{i}}\right)$ for each $o \in  {S}_{i}$ .

	3. 为每个$o \in  {S}_{i}$计算前缀$\left( {o,S}\right)  = {V}_{i} + \operatorname{prefix}\left( {o,{S}_{i}}\right)$。

---

<!-- Media -->

In the above map-shuffle phase, every machine sends and receives exactly $t - 1$ values in total: precisely, ${\mathcal{M}}_{i}\left( {1 \leq  i \leq  t}\right)$ sends $t - i$ and receives $i - 1$ values. This satisfies bounded net-traffic because $t \leq  m$ . Furthermore,the reduce phase takes $O\left( m\right)  = O\left( {n/t}\right)$ time,by leveraging the sorted order of ${S}_{i}$ . Omitting the other trivial details, we conclude that our prefix sum algorithm is minimal.

在上述的映射 - 洗牌阶段，每台机器总共发送和接收恰好$t - 1$个值：具体来说，${\mathcal{M}}_{i}\left( {1 \leq  i \leq  t}\right)$发送$t - i$个值并接收$i - 1$个值。由于$t \leq  m$，这满足有界网络流量的条件。此外，归约阶段利用${S}_{i}$的有序性，需要$O\left( m\right)  = O\left( {n/t}\right)$的时间。省略其他琐碎的细节，我们得出结论：我们的前缀和算法是最小的。

Prefix Min. The prefix min problem is almost the same as prefix sum,except that $\operatorname{prefix}\left( {o,S}\right)$ is defined as the prefix min of $o$ ,which is the minimum weight of the objects ${o}^{\prime } \in  S$ such that ${o}^{\prime } < o$ . This problem can also be settled by the above algorithm minimally with three simple changes: redefine (i) ${W}_{i} = \mathop{\min }\limits_{{o \in  {S}_{i}}}w\left( o\right)$ in the map-shuffle phase,(ii-iii) ${V}_{i} =$ $\mathop{\min }\limits_{{j \leq  i - 1}}{W}_{j}$ at Line 1 of the reduce phase,and prefix $\left( {o,S}\right)  =$ $\min \left\{  {{V}_{i},\operatorname{prefix}\left( {o,{S}_{i}}\right) }\right\}$ at Line 3 .

前缀最小值。前缀最小值问题几乎与前缀和问题相同，不同之处在于将$\operatorname{prefix}\left( {o,S}\right)$定义为$o$的前缀最小值，即满足${o}^{\prime } < o$的对象${o}^{\prime } \in  S$的最小权重。这个问题也可以通过上述算法以最少的改动解决，只需进行三处简单修改：（i）在映射 - 洗牌阶段重新定义${W}_{i} = \mathop{\min }\limits_{{o \in  {S}_{i}}}w\left( o\right)$；（ii - iii）在归约阶段的第1行修改为${V}_{i} =$ $\mathop{\min }\limits_{{j \leq  i - 1}}{W}_{j}$，并在第3行将前缀修改为$\left( {o,S}\right)  =$ $\min \left\{  {{V}_{i},\operatorname{prefix}\left( {o,{S}_{i}}\right) }\right\}$。

Ranking. Let $S$ be a set of objects from an ordered domain. The ranking problem reports the rank of each object $o \in  S$ ,which equals $\left| \left\{  {{o}^{\prime } \in  S \mid  {o}^{\prime } \leq  o}\right\}  \right|$ ; in other words,the smallest object has rank 1, the second smallest rank 2, etc. This can be solved as a special prefix sum problem where all objects have weight 1 (i.e. prefix count).

排名。设$S$是来自一个有序域的一组对象。排名问题是报告每个对象$o \in  S$的排名，其等于$\left| \left\{  {{o}^{\prime } \in  S \mid  {o}^{\prime } \leq  o}\right\}  \right|$；换句话说，最小的对象排名为1，第二小的对象排名为2，依此类推。这可以作为一个特殊的前缀和问题来解决，其中所有对象的权重都为1（即前缀计数）。

Skyline. Let ${x}_{p}\left( {y}_{p}\right)$ be the $\mathrm{x}$ - (y-) coordinate of a 2D point $p$ . A point $p$ dominates another ${p}^{\prime }$ if ${x}_{p} \leq  {x}_{{p}^{\prime }}$ and ${y}_{p} \leq  {y}_{{p}^{\prime }}$ . For a set $P$ of $n$ 2D points,the skyline is the set of points $p \in  P$ such that $p$ is not dominated by any other point in $P$ . The skyline problem [8] is to report the skyline of $P$ ,and admits a sequential algorithm of $O\left( {n\log n}\right)$ time [35].

天际线。设${x}_{p}\left( {y}_{p}\right)$为二维点$p$的$\mathrm{x}$（y）坐标。如果${x}_{p} \leq  {x}_{{p}^{\prime }}$且${y}_{p} \leq  {y}_{{p}^{\prime }}$，则点$p$支配另一个点${p}^{\prime }$。对于由$n$个二维点组成的集合$P$，天际线是满足$p$不被$P$中任何其他点支配的点集$p \in  P$。天际线问题[8]是报告$P$的天际线，并且存在一个时间复杂度为$O\left( {n\log n}\right)$的顺序算法[35]。

The problem is essentially prefix min in disguise. Specifically, let $S = \left\{  {{x}_{p} \mid  p \in  P}\right\}$ where ${x}_{p}$ carries a "weight" ${y}_{p}$ . Define the prefix min of ${x}_{p}$ as the minimum "weight" of the values in $S$ preceding ${}^{3}{x}_{p}$ . It is rudimentary to show that $p$ is in the skyline of $P$ ,if and only if the prefix min of ${x}_{p}$ is strictly greater than ${y}_{p}$ . Therefore, our prefix min algorithm also settles the skyline problem minimally.

这个问题本质上是伪装的前缀最小值问题。具体来说，设$S = \left\{  {{x}_{p} \mid  p \in  P}\right\}$，其中${x}_{p}$带有一个“权重”${y}_{p}$。将${x}_{p}$的前缀最小值定义为$S$中先于${}^{3}{x}_{p}$的值的最小“权重”。很容易证明，当且仅当${x}_{p}$的前缀最小值严格大于${y}_{p}$时，$p$才在$P$的天际线中。因此，我们的前缀最小值算法也能以最少的改动解决天际线问题。

### 4.2 Group By

### 4.2 分组

Let $S$ be a set of $n$ objects,where each object $o \in  S$ carries a key $k\left( o\right)$ and a weight $w\left( o\right)$ ,both of which are real values. A group $G$ is a maximal set of objects with the same key. The aggregate of $G$ is the result of applying a distributive ${}^{4}$ aggregate function AGG to the weights of the objects in $G$ . The group-by problem is to report the aggregates of all groups. It is easy to do so in $O\left( {n\log n}\right)$ time on a sequential machine. Next, we discuss MapReduce, assuming for simplicity $\mathrm{{AGG}} =$ sum because it is straightforward to generalize the discussion to other AGG.

设$S$是由$n$个对象组成的集合，其中每个对象$o \in  S$都带有一个键$k\left( o\right)$和一个权重$w\left( o\right)$，两者均为实数值。组$G$是具有相同键的最大对象集合。$G$的聚合是对$G$中对象的权重应用一个可分配的${}^{4}$聚合函数AGG的结果。分组问题是报告所有组的聚合结果。在顺序机器上，很容易在$O\left( {n\log n}\right)$时间内完成此操作。接下来，我们讨论MapReduce，为简单起见，假设$\mathrm{{AGG}} =$为求和，因为将讨论推广到其他AGG是很直接的。

---

<!-- Footnote -->

${}^{3}$ Precisely,given points $p$ and ${p}^{\prime },{x}_{p}$ precedes ${x}_{{p}^{\prime }}$ if (i) ${x}_{p} < {x}_{{p}^{\prime }}$ or (ii) ${x}_{p} = {x}_{{p}^{\prime }}$ but ${y}_{p} < {y}_{{p}^{\prime }}$ .

${}^{3}$ 确切地说，给定的点 $p$ 和 ${p}^{\prime },{x}_{p}$ ，若满足 (i) ${x}_{p} < {x}_{{p}^{\prime }}$ 或 (ii) ${x}_{p} = {x}_{{p}^{\prime }}$ 但 ${y}_{p} < {y}_{{p}^{\prime }}$ ，则 $p$ 和 ${p}^{\prime },{x}_{p}$ 先于 ${x}_{{p}^{\prime }}$ 。

${}^{4}$ An aggregate function AGG is distributive on a set $S$ if $\operatorname{AGG}\left( S\right)$ can be obtained in constant time from $\operatorname{AGG}\left( {S}_{1}\right)$ and $\operatorname{AGG}\left( {S}_{2}\right)$ , where ${S}_{1}$ and ${S}_{2}$ form a partition of $S$ ,i.e., ${S}_{1} \cup  {S}_{2} = S$ and ${S}_{1} \cap  {S}_{2} = \varnothing$ .

${}^{4}$ 如果可以在常数时间内从 $\operatorname{AGG}\left( {S}_{1}\right)$ 和 $\operatorname{AGG}\left( {S}_{2}\right)$ 得到 $\operatorname{AGG}\left( S\right)$ ，其中 ${S}_{1}$ 和 ${S}_{2}$ 构成集合 $S$ 的一个划分，即 ${S}_{1} \cup  {S}_{2} = S$ 且 ${S}_{1} \cap  {S}_{2} = \varnothing$ ，那么聚合函数 AGG（Aggregate Function）在集合 $S$ 上是可分配的。

Map-shuffle (on each ${\mathcal{M}}_{i},1 \leq  i \leq  t$ )

映射 - 洗牌（在每个 ${\mathcal{M}}_{i},1 \leq  i \leq  t$ 上）

<!-- Footnote -->

---

The main issue is to handle large groups that do not fit in one machine. Our algorithm starts by sorting the objects by keys, breaking ties by ids. Consider an arbitrary machine $\mathcal{M}$ after sorting. If a group $G$ is now completely in $\mathcal{M}$ ,its aggregate can be obtained locally in $\mathcal{M}$ . Motivated by this,let ${k}_{min}\left( \mathcal{M}\right)$ and ${k}_{max}\left( \mathcal{M}\right)$ be the smallest and largest keys on $\mathcal{M}$ currently. Clearly,groups of keys $k$ where ${k}_{\min }\left( \mathcal{M}\right)  < k < {k}_{\max }\left( \mathcal{M}\right)$ are entirely stored in $\mathcal{M}$ ,which can obtain their aggregates during sorting,and remove them from further consideration.

主要问题是处理无法在一台机器中容纳的大型分组。我们的算法首先按键对对象进行排序，若键相同则按 ID 排序。考虑排序后任意一台机器 $\mathcal{M}$ 。如果一个分组 $G$ 现在完全在 $\mathcal{M}$ 中，那么它的聚合结果可以在 $\mathcal{M}$ 本地获得。受此启发，设 ${k}_{min}\left( \mathcal{M}\right)$ 和 ${k}_{max}\left( \mathcal{M}\right)$ 分别是当前 $\mathcal{M}$ 上最小和最大的键。显然，键为 $k$ （其中 ${k}_{\min }\left( \mathcal{M}\right)  < k < {k}_{\max }\left( \mathcal{M}\right)$ ）的分组完全存储在 $\mathcal{M}$ 中，这些分组的聚合结果可以在排序过程中获得，并将它们从后续考虑中移除。

Each machine $\mathcal{M}$ has at most 2 groups remaining,i.e.,with keys ${k}_{min}\left( \mathcal{M}\right)$ and ${k}_{max}\left( \mathcal{M}\right)$ ,respectively. Hence,there are at most ${2t}$ such groups on all machines. To handle them, we ask each machine to send at most 4 values to ${\mathcal{M}}_{1}$ (i.e.,to just a single machine). The following elaborates how:

每台机器 $\mathcal{M}$ 最多剩下 2 个分组，即键分别为 ${k}_{min}\left( \mathcal{M}\right)$ 和 ${k}_{max}\left( \mathcal{M}\right)$ 的分组。因此，所有机器上最多有 ${2t}$ 个这样的分组。为了处理这些分组，我们要求每台机器最多向 ${\mathcal{M}}_{1}$ （即仅向一台机器）发送 4 个值。以下详细说明如何操作：

<!-- Media -->

Map-shuffle (on each ${\mathcal{M}}_{i},1 \leq  i \leq  t$ ):

映射 - 洗牌（在每个 ${\mathcal{M}}_{i},1 \leq  i \leq  t$ 上）:

---

	1. Obtain the total weight ${W}_{\min }\left( {\mathcal{M}}_{i}\right)$ of group ${k}_{\min }\left( {\mathcal{M}}_{i}\right)$ ,i.e.,

	1. 获得分组 ${k}_{\min }\left( {\mathcal{M}}_{i}\right)$ 的总权重 ${W}_{\min }\left( {\mathcal{M}}_{i}\right)$ ，即

		by considering only objects in ${\mathcal{M}}_{i}$ .

		仅考虑 ${\mathcal{M}}_{i}$ 中的对象。

	2. Send pair $\left( {{k}_{\min }\left( {\mathcal{M}}_{i}\right) ,{W}_{\min }\left( {\mathcal{M}}_{i}\right) }\right)$ to ${\mathcal{M}}_{1}$ .

	2. 将对 $\left( {{k}_{\min }\left( {\mathcal{M}}_{i}\right) ,{W}_{\min }\left( {\mathcal{M}}_{i}\right) }\right)$ 发送到 ${\mathcal{M}}_{1}$ 。

	3. If ${k}_{\min }\left( {\mathcal{M}}_{i}\right)  \neq  {k}_{\max }\left( {\mathcal{M}}_{i}\right)$ ,send pair $\left( {{k}_{\max }\left( {\mathcal{M}}_{i}\right) }\right.$ ,

	3. 如果 ${k}_{\min }\left( {\mathcal{M}}_{i}\right)  \neq  {k}_{\max }\left( {\mathcal{M}}_{i}\right)$ ，则发送对 $\left( {{k}_{\max }\left( {\mathcal{M}}_{i}\right) }\right.$ ，

		$\left. {{W}_{\max }\left( {\mathcal{M}}_{i}\right) }\right)$ to ${\mathcal{M}}_{1}$ ,where the definition of ${k}_{\max }\left( {\mathcal{M}}_{i}\right)$ is

		$\left. {{W}_{\max }\left( {\mathcal{M}}_{i}\right) }\right)$ 到 ${\mathcal{M}}_{1}$ ，其中 ${k}_{\max }\left( {\mathcal{M}}_{i}\right)$ 的定义

		similar to ${k}_{\min }\left( {\mathcal{M}}_{i}\right)$ .

		与 ${k}_{\min }\left( {\mathcal{M}}_{i}\right)$ 类似。

Reduce (only on ${\mathcal{M}}_{1}$ ):

归约（仅在 ${\mathcal{M}}_{1}$ 上）:

		Let $\left( {{k}_{1},{w}_{1}}\right) ,\ldots ,\left( {{k}_{x},{w}_{x}}\right)$ be the pairs received in the

		设 $\left( {{k}_{1},{w}_{1}}\right) ,\ldots ,\left( {{k}_{x},{w}_{x}}\right)$ 为在上一阶段接收到的成对数据

		previous phase where $x$ is some value between $t$ and ${2t}$ . For

		其中 $x$ 是介于 $t$ 和 ${2t}$ 之间的某个值。对于

		each group whose key $k$ is in one of the $x$ pairs,output its

		每个键 $k$ 位于 $x$ 对中的某一对里的组，输出其

		final aggregate $\mathop{\sum }\limits_{{j \mid  {k}_{j} = k}}{w}_{j}$ .

		最终聚合值 $\mathop{\sum }\limits_{{j \mid  {k}_{j} = k}}{w}_{j}$。

---

<!-- Media -->

The minimality of our group-by algorithm is easy to verify. It suffices to point out that the reduce phase of the last round takes $O\left( {t\log t}\right)  = O\left( {\frac{n}{t}\log n}\right)$ time (since $t \leq  m = n/t$ ).

我们的分组算法的最优性很容易验证。只需指出最后一轮的归约阶段需要 $O\left( {t\log t}\right)  = O\left( {\frac{n}{t}\log n}\right)$ 时间（因为 $t \leq  m = n/t$）即可。

Categorical Keys. We have assumed that the key $k\left( o\right)$ of an object is numeric. This is in fact unnecessary because the key ordering does not affect the correctness of group by. Hence,even if $k\left( o\right)$ is categorical, we can simply sort the keys alphabetically by their binary representations.

分类键。我们假设对象的键 $k\left( o\right)$ 是数值型的。实际上这并非必要，因为键的排序并不影响分组的正确性。因此，即使 $k\left( o\right)$ 是分类键，我们也可以简单地按照它们的二进制表示按字母顺序对键进行排序。

Term Frequency. MapReduce is often introduced with the term frequency problem. The input is a document $D$ ,which can be regarded as a multi-set of strings. The goal is to report, for every distinct string $s \in  D$ ,the number of occurrences of $s$ in $D$ . In their pioneering work,Dean and Ghemawat [16] gave an algorithm which works by sending all occurrences of a string to an identical machine. The algorithm is not minimal in the scenario where a string has an exceedingly high frequency. Note, on the other hand, that the term frequency problem is merely a group-by problem with every distinct string representing a group. Hence, our group-by algorithm provides a minimal alternative to counting term frequencies.

词频。MapReduce 通常会结合词频问题进行介绍。输入是一个文档 $D$，它可以被视为一个字符串的多重集。目标是针对每个不同的字符串 $s \in  D$，报告 $s$ 在 $D$ 中出现的次数。在他们的开创性工作中，迪恩（Dean）和格玛沃特（Ghemawat）[16] 提出了一种算法，该算法通过将一个字符串的所有出现情况发送到同一台机器来工作。在某个字符串出现频率极高的场景中，该算法并非最优。另一方面，需要注意的是，词频问题仅仅是一个分组问题，每个不同的字符串代表一个组。因此，我们的分组算法为统计词频提供了一种最优的替代方案。

### 4.3 Semi-Join

### 4.3 半连接

Let $R$ and $T$ be two sets of objects from the same domain. Each object $o$ in $R$ or $T$ carries a key $k\left( o\right)$ . The semi-join problem is to report all the objects $o \in  R$ that have a match ${o}^{\prime } \in  T$ ,i.e., $k\left( o\right)  = k\left( {o}^{\prime }\right)$ . The problem can be solved in $O\left( {n\log n}\right)$ time sequentially,where $n$ is the total number of objects in $R \cup  T$ .

设 $R$ 和 $T$ 是来自同一域的两个对象集合。$R$ 或 $T$ 中的每个对象 $o$ 都带有一个键 $k\left( o\right)$。半连接问题是报告所有有匹配项 ${o}^{\prime } \in  T$ 的对象 $o \in  R$，即 $k\left( o\right)  = k\left( {o}^{\prime }\right)$。该问题可以按顺序在 $O\left( {n\log n}\right)$ 时间内解决，其中 $n$ 是 $R \cup  T$ 中对象的总数。

In MapReduce, we approach the problem in a way analogous to how group-by was tackled. The difference is that, now objects with the same key do not "collapse" into an aggregate; instead, we must output all of them if their (common) key has a match in $T$ . For this reason, we will need to transfer more network data than group by, as will be clear shortly,but still $O\left( t\right)$ words per machine.

在 MapReduce 中，我们处理这个问题的方式与处理分组问题类似。不同之处在于，现在具有相同键的对象不会“合并”为一个聚合值；相反，如果它们（共同的）键在 $T$ 中有匹配项，我们必须输出所有这些对象。出于这个原因，我们需要传输的网络数据会比分组问题更多，这一点很快就会清楚，但每台机器仍然是 $O\left( t\right)$ 个单词。

Define $S = R \cup  T$ . We sort the objects of the mixed set $S$ by their keys across the $t$ machines. Consider any machine $\mathcal{M}$ after sorting. Let ${k}_{\min }\left( {T,\mathcal{M}}\right)$ and ${k}_{\max }\left( {T,\mathcal{M}}\right)$ be the smallest and largest keys respectively,among the $T$ -objects stored on $\mathcal{M}$ (a $T$ -object is an object from $T$ ). The semi-join problem can be settled with an extra round:

定义 $S = R \cup  T$。我们在 $t$ 台机器上按对象的键对混合集合 $S$ 中的对象进行排序。考虑排序后的任意一台机器 $\mathcal{M}$。设 ${k}_{\min }\left( {T,\mathcal{M}}\right)$ 和 ${k}_{\max }\left( {T,\mathcal{M}}\right)$ 分别是存储在 $\mathcal{M}$ 上的 $T$ -对象（$T$ -对象是来自 $T$ 的对象）中的最小键和最大键。半连接问题可以通过额外一轮来解决：

<!-- Media -->

Map-shuffle (on each ${\mathcal{M}}_{i},1 \leq  i \leq  t$ ):

映射 - 洗牌（在每台 ${\mathcal{M}}_{i},1 \leq  i \leq  t$ 上）：

---

		Send ${k}_{min}\left( {T,{\mathcal{M}}_{i}}\right)$ and ${k}_{max}\left( {T,{\mathcal{M}}_{i}}\right)$ to all machines.

		将 ${k}_{min}\left( {T,{\mathcal{M}}_{i}}\right)$ 和 ${k}_{max}\left( {T,{\mathcal{M}}_{i}}\right)$ 发送到所有机器。

Reduce (on each ${\mathcal{M}}_{i}$ ):

归约（在每台 ${\mathcal{M}}_{i}$ 上）：

	1. ${K}_{\text{border }} =$ the set of keys received from the last round.

	1. ${K}_{\text{border }} =$ 是上一轮接收到的键的集合。

	2. $K\left( {\mathcal{M}}_{i}\right)  =$ the set of keys of the $T$ -objects stored in ${\mathcal{M}}_{i}$ .

	2. $K\left( {\mathcal{M}}_{i}\right)  =$ 存储在 ${\mathcal{M}}_{i}$ 中的 $T$ 对象的键集合。

	3. For every $R$ -object $o$ stored in ${\mathcal{M}}_{i}$ ,output it if $k\left( o\right)  \in$

	3. 对于存储在 ${\mathcal{M}}_{i}$ 中的每个 $R$ 对象 $o$，如果 $k\left( o\right)  \in$ 则输出它。

		$K\left( {\mathcal{M}}_{i}\right)  \cup  {K}_{\text{border }}$ .

---

<!-- Media -->

Every machine sends and receives ${2t}$ keys in the map-shuffle phase. The reduce phase can be implemented in $O\left( {m + t\log t}\right)  =$ $O\left( {\frac{n}{t}\log n}\right)$ time,using the fact that the $R$ -objects on ${\mathcal{M}}_{i}$ are already sorted. The overall semi-join algorithm is minimal.

每台机器在映射 - 洗牌（map - shuffle）阶段发送和接收 ${2t}$ 个键。利用 ${\mathcal{M}}_{i}$ 上的 $R$ 对象已经排序这一事实，归约（reduce）阶段可以在 $O\left( {m + t\log t}\right)  =$ $O\left( {\frac{n}{t}\log n}\right)$ 时间内实现。整个半连接算法是最优的。

## 5. SLIDING AGGREGATION

## 5. 滑动聚合

This section is devoted to the sliding aggregation problem. Recall that the input is: (i) a set $S$ of $n$ objects from an ordered domain,(ii) an integer $\ell  \leq  n$ ,and (iii) a distributive aggregate function AGG. We will focus on AGG $=$ sum because extension to other AGG is straightforward. Each object $o \in  S$ is associated with a real-valued weight $w\left( o\right)$ . The window of $o$ ,denoted as window(o),is the set of $\ell$ largest objects not exceeding $o$ (see Figure 1). The window sum of $o$ equals

本节致力于解决滑动聚合问题。回顾一下，输入包括：（i）一个来自有序域的 $n$ 对象集合 $S$；（ii）一个整数 $\ell  \leq  n$；（iii）一个可分配的聚合函数 AGG。我们将重点关注 AGG $=$ 求和，因为扩展到其他 AGG 函数很直接。每个对象 $o \in  S$ 都与一个实值权重 $w\left( o\right)$ 相关联。$o$ 的窗口，记为 window(o)，是不超过 $o$ 的 $\ell$ 个最大对象的集合（见图 1）。$o$ 的窗口和等于

$$
\operatorname{win}\operatorname{sum}\left( o\right)  = \mathop{\sum }\limits_{{{o}^{\prime } \in  \operatorname{window}\left( o\right) }}w\left( {o}^{\prime }\right) 
$$

The objective is to report win-sum(o)for all $o \in  S$ .

目标是报告所有 $o \in  S$ 的窗口和 win - sum(o)。

### 5.1 Sorting with Perfect Balance

### 5.1 完美平衡排序

Let us first tackle a variant of sorting which we call the perfect sorting problem. The input is a set $S$ of $n$ objects from an ordered domain. We want to distribute them among the $t$ MapReduce machines ${\mathcal{M}}_{1},\ldots ,{\mathcal{M}}_{t}$ such that ${\mathcal{M}}_{i},1 \leq  i \leq  t - 1$ ,stores exactly $\lceil m\rceil$ objects,and ${\mathcal{M}}_{t}$ stores all the remaining objects, where $m = n/t$ . In the meantime,the sorted order must be maintained,i.e.,all objects on ${\mathcal{M}}_{i}$ precede those on ${\mathcal{M}}_{j}$ ,for any $1 \leq  i < j \leq  t$ . We will assume that $m$ is an integer; if not,simply pad at most $t - 1$ dummy objects to make $n$ a multiple of $t$ .

让我们首先处理一种排序的变体，我们称之为完美排序问题。输入是一个来自有序域的 $n$ 对象集合 $S$。我们希望将它们分配到 $t$ 个 MapReduce 机器 ${\mathcal{M}}_{1},\ldots ,{\mathcal{M}}_{t}$ 上，使得 ${\mathcal{M}}_{i},1 \leq  i \leq  t - 1$ 恰好存储 $\lceil m\rceil$ 个对象，${\mathcal{M}}_{t}$ 存储所有剩余的对象，其中 $m = n/t$。同时，必须保持排序顺序，即对于任何 $1 \leq  i < j \leq  t$，${\mathcal{M}}_{i}$ 上的所有对象都先于 ${\mathcal{M}}_{j}$ 上的对象。我们假设 $m$ 是一个整数；如果不是，只需填充最多 $t - 1$ 个虚拟对象，使 $n$ 是 $t$ 的倍数。

The problem is in fact nothing but a small extension to ranking. Our algorithm first invokes the ranking algorithm in Section 4.1 to obtain the rank of each $o \in  S$ ,denoted as $r\left( o\right)$ . Then,we finish in one more round:

这个问题实际上只是对排名问题的一个小扩展。我们的算法首先调用 4.1 节中的排名算法来获取每个 $o \in  S$ 的排名，记为 $r\left( o\right)$。然后，我们再进行一轮操作来完成：

Map-shuffle (on each ${\mathcal{M}}_{i},1 \leq  i \leq  t$ ):

映射 - 洗牌（在每个 ${\mathcal{M}}_{i},1 \leq  i \leq  t$ 上）：

For each object $o$ currently on ${\mathcal{M}}_{i}$ ,send it to ${\mathcal{M}}_{j}$ where $j = \lceil r\left( o\right) /m\rceil$ .

对于当前在 ${\mathcal{M}}_{i}$ 上的每个对象 $o$，将其发送到 ${\mathcal{M}}_{j}$，其中 $j = \lceil r\left( o\right) /m\rceil$。

Reduce: No action is needed.

归约：无需操作。

---

<!-- Footnote -->

The above algorithm is clearly minimal.

上述算法显然是最优的。

<!-- Footnote -->

---

### 5.2 Sliding Aggregate Computation

### 5.2 滑动聚合计算

We now return to the sliding aggregation problem, assuming that $S$ has been perfectly sorted across ${\mathcal{M}}_{1},\ldots ,{\mathcal{M}}_{t}$ as described earlier. The objective is to settle the problem in just one more round. Once again,we assume that $n$ is a multiple of $t$ ; if not,pad at most $t - 1$ dummy objects with zero weights.

现在我们回到滑动聚合问题，假设 $S$ 已经如前文所述在 ${\mathcal{M}}_{1},\ldots ,{\mathcal{M}}_{t}$ 上进行了完美排序。目标是再进行一轮操作来解决这个问题。我们再次假设 $n$ 是 $t$ 的倍数；如果不是，最多填充 $t - 1$ 个权重为零的虚拟对象。

By virtue of the perfect balancing,the objects on machine $i$ form a rank range $\left\lbrack  {\left( {i - 1}\right) m + 1,{im}}\right\rbrack$ ,for $1 \leq  i \leq  t$ . Consider an object $o$ with window $\left( o\right)  = \left\lbrack  {r\left( o\right)  - \ell  + 1,r\left( o\right) }\right\rbrack$ ,i.e.,the range of ranks of the objects in window(o). Clearly,window(o)intersects the rank ranges of machines from $\alpha$ to $\beta$ ,where $\alpha  = \lceil \left( {r\left( o\right)  - \ell  + 1}\right) /m\rceil$ to $\beta  = \lceil r\left( o\right) /m\rceil$ . If $\alpha  = \beta$ ,win-sum(o)can be calculated locally by ${\mathcal{M}}_{\beta }$ ,so next we focus on $\alpha  < \beta$ . Note that when $\alpha  < \beta  - 1$ , window(o)spans the rank ranges of machines $\alpha  + 1,\ldots ,\beta  - 1$ .

由于完美的平衡，机器$i$上的对象形成了一个排名范围$\left\lbrack  {\left( {i - 1}\right) m + 1,{im}}\right\rbrack$，其中$1 \leq  i \leq  t$。考虑一个窗口为$\left( o\right)  = \left\lbrack  {r\left( o\right)  - \ell  + 1,r\left( o\right) }\right\rbrack$的对象$o$，即窗口(o)中对象的排名范围。显然，窗口(o)与从$\alpha$到$\beta$的机器的排名范围相交，其中$\alpha  = \lceil \left( {r\left( o\right)  - \ell  + 1}\right) /m\rceil$到$\beta  = \lceil r\left( o\right) /m\rceil$。如果$\alpha  = \beta$，win - sum(o)可以通过${\mathcal{M}}_{\beta }$在本地计算，因此接下来我们关注$\alpha  < \beta$。注意，当$\alpha  < \beta  - 1$时，窗口(o)跨越了机器$\alpha  + 1,\ldots ,\beta  - 1$的排名范围。

Let ${W}_{i}$ be the total weight of all the objects on ${\mathcal{M}}_{i},1 \leq  i \leq  t$ . We will ensure that every machine knows ${W}_{1},\ldots ,{W}_{t}$ . Then,to calculate win-sum (o) at ${\mathcal{M}}_{\beta }$ ,the only information ${\mathcal{M}}_{\beta }$ does not have locally is the objects on ${\mathcal{M}}_{\alpha }$ enclosed in window(o). We say that those objects are remotely relevant to ${\mathcal{M}}_{\beta }$ . Objects from machines $\alpha  + 1,\ldots ,\beta  - 1$ are not needed because their contributions to win-sum(o)have been summarized by ${W}_{\alpha  + 1},\ldots ,{W}_{\beta  - 1}$ .

设${W}_{i}$为${\mathcal{M}}_{i},1 \leq  i \leq  t$上所有对象的总权重。我们将确保每台机器都知道${W}_{1},\ldots ,{W}_{t}$。然后，为了在${\mathcal{M}}_{\beta }$处计算win - sum(o)，${\mathcal{M}}_{\beta }$本地没有的唯一信息是窗口(o)中包含的${\mathcal{M}}_{\alpha }$上的对象。我们称这些对象与${\mathcal{M}}_{\beta }$远程相关。不需要来自机器$\alpha  + 1,\ldots ,\beta  - 1$的对象，因为它们对win - sum(o)的贡献已由${W}_{\alpha  + 1},\ldots ,{W}_{\beta  - 1}$汇总。

The lemma below points out a crucial fact.

下面的引理指出了一个关键事实。

LEMMA 2. Every object is remotely relevant to at most 2 machines.

引理2. 每个对象最多与2台机器远程相关。

Proof. Consider a machine ${\mathcal{M}}_{i}$ for some $i \in  \left\lbrack  {1,t}\right\rbrack$ . If a machine ${\mathcal{M}}_{j}$ stores at least an object remotely relevant to ${\mathcal{M}}_{i}$ ,we say that ${\mathcal{M}}_{j}$ is pertinent to ${\mathcal{M}}_{i}$ .

证明。考虑对于某个$i \in  \left\lbrack  {1,t}\right\rbrack$的一台机器${\mathcal{M}}_{i}$。如果一台机器${\mathcal{M}}_{j}$存储了至少一个与${\mathcal{M}}_{i}$远程相关的对象，我们称${\mathcal{M}}_{j}$与${\mathcal{M}}_{i}$相关。

Recall that the left endpoint of window(o)lies in machine $\alpha  =$ $\lceil \left( {r\left( o\right)  - \ell  + 1}\right) /m\rceil$ . When $r\left( o\right)  \in  \left\lbrack  {\left( {i - 1}\right) m + 1,{im}}\right\rbrack$ ,i.e.,the rank range of ${\mathcal{M}}_{i}$ ,it holds that

回顾一下，窗口(o)的左端点位于机器$\alpha  =$$\lceil \left( {r\left( o\right)  - \ell  + 1}\right) /m\rceil$中。当$r\left( o\right)  \in  \left\lbrack  {\left( {i - 1}\right) m + 1,{im}}\right\rbrack$，即${\mathcal{M}}_{i}$的排名范围时，有

$$
\left\lceil  \frac{\left( {i - 1}\right) m + 1 - \ell  + 1}{m}\right\rceil   \leq  \alpha  \leq  \left\lceil  \frac{{im} - \ell  + 1}{m}\right\rceil   \Rightarrow  
$$

$$
\left( {i - 1}\right)  - \left\lfloor  \frac{\ell  - 1}{m}\right\rfloor   \leq  \alpha  \leq  i - \left\lfloor  \frac{\ell  - 1}{m}\right\rfloor   \tag{1}
$$

where the last step used the fact that $\lceil x - y\rceil  = x - \lfloor y\rfloor$ for any integer $x$ and real value $y$ .

其中最后一步使用了对于任何整数$x$和实数值$y$有$\lceil x - y\rceil  = x - \lfloor y\rfloor$这一事实。

There are two useful observations. First,integer $\alpha$ has only two choices satisfying (1), namely, at most 2 machines are pertinent to ${\mathcal{M}}_{i}$ . Second,as is grows by 1,the two permissible values of $\alpha$ both increase by 1 . This means that each machine can be pertinent to at most 2 machines, thus completing the proof.

有两个有用的观察结果。首先，整数 $\alpha$ 只有两个值满足条件 (1)，即最多有 2 台机器与 ${\mathcal{M}}_{i}$ 相关。其次，当 $\alpha$ 增加 1 时，其两个允许的值都会增加 1。这意味着每台机器最多与 2 台机器相关，从而完成了证明。

COROLLARY 1. Objects in ${\mathcal{M}}_{i},1 \leq  i \leq  t$ ,can be remotely relevant only to

推论 1. ${\mathcal{M}}_{i},1 \leq  i \leq  t$ 中的对象只能与以下对象远程相关

- machine $i + 1$ ,if $\ell  \leq  m$

- 机器 $i + 1$，如果 $\ell  \leq  m$

- machines $i + \lfloor \left( {\ell  - 1}\right) /m\rfloor$ and $i + 1 + \lfloor \left( {\ell  - 1}\right) /m\rfloor$ ,otherwise.

- 机器 $i + \lfloor \left( {\ell  - 1}\right) /m\rfloor$ 和 $i + 1 + \lfloor \left( {\ell  - 1}\right) /m\rfloor$，否则。

In the above,if a machine id exceeds $m$ ,ignore it.

在上述情况中，如果机器 ID 超过 $m$，则忽略它。

Proof. Directly from (1).

证明：直接由 (1) 得出。

We are now ready to explain how to solve the sliding aggregation problem in one round:

现在我们准备解释如何在一轮中解决滑动聚合问题：

Map-shuffle (on each ${\mathcal{M}}_{i},1 \leq  i \leq  t$ ):

映射 - 洗牌（在每个 ${\mathcal{M}}_{i},1 \leq  i \leq  t$ 上）：

1. Send ${W}_{i}$ to all machines.

1. 将 ${W}_{i}$ 发送到所有机器。

2. Send all the objects in ${\mathcal{M}}_{i}$ to one or two machines as instructed by Corollary 1.

2. 根据推论 1 的指示，将 ${\mathcal{M}}_{i}$ 中的所有对象发送到一台或两台机器。

<!-- Media -->

---

Reduce (on each ${\mathcal{M}}_{i}$ ):

归约（在每个 ${\mathcal{M}}_{i}$ 上）：

	For each object $o$ already in ${\mathcal{M}}_{i}$ after perfect sorting:

	对于经过完美排序后已存在于 ${\mathcal{M}}_{i}$ 中的每个对象 $o$：

	1. $\alpha  = \lceil \left( {r\left( o\right)  - \ell  + 1}\right) /m\rceil$

	2. ${w}_{1} =$ the total weight of the objects in ${\mathcal{M}}_{\alpha }$ that fall in

	2. ${w}_{1} =$ 是 ${\mathcal{M}}_{\alpha }$ 中落在

		window(o)(if $\alpha  < i$ ,such objects were received in the last

		窗口 (o) 内的对象的总权重（如果 $\alpha  < i$，这些对象是在上一

		phase).

		阶段收到的）。

	3. ${w}_{2} = \mathop{\sum }\limits_{{j = \alpha  + 1}}^{{i - 1}}{W}_{j}$ .

	4. If $\alpha  = i$ ,set ${w}_{3} = 0$ ; otherwise, ${w}_{3}$ is the total weight of the

	4. 如果 $\alpha  = i$，则设置 ${w}_{3} = 0$；否则，${w}_{3}$ 是

		objects in ${\mathcal{M}}_{i}$ that fall in window(o).

		${\mathcal{M}}_{i}$中落在窗口(o)内的对象。

	5. $\operatorname{win} - \operatorname{sum}\left( o\right)  = {w}_{1} + {w}_{2} + {w}_{3}$ .

---

<!-- Media -->

We now analyze the algorithm's minimality. It is clear that every machine sends and receives $O\left( {t + m}\right)  = O\left( m\right)$ words of data over the network in the map-shuffle phase. Hence, each machine requires only $O\left( m\right)$ storage. It remains to prove that the reduce phase terminates in $O\left( {\frac{n}{t}\log n}\right)$ time. We create a range sum structure ${}^{5}$ respectively on: (i) the local objects in ${\mathcal{M}}_{i}$ ,(ii) the objects received from (at most) two machines in the map-reduce phase,and (iii) the set $\left\{  {{W}_{1},\ldots ,{W}_{t}}\right\}$ . These structures can be built in $O\left( {m\log m}\right)$ time,and allow us to compute ${w}_{1},{w}_{2},{w}_{3}$ in Lines 2-4 using $O\left( {\log m}\right)$ time. It follows that the reduce phase takes $O\left( {m\log m}\right)  = O\left( {\frac{n}{t}\log m}\right)$ time.

我们现在分析该算法的最小性。显然，在映射 - 混洗阶段，每台机器通过网络发送和接收$O\left( {t + m}\right)  = O\left( m\right)$个数据字。因此，每台机器仅需要$O\left( m\right)$的存储空间。还需证明归约阶段在$O\left( {\frac{n}{t}\log n}\right)$时间内终止。我们分别在以下内容上创建一个范围求和结构${}^{5}$：(i) ${\mathcal{M}}_{i}$中的本地对象；(ii) 在映射 - 归约阶段从（最多）两台机器接收的对象；(iii) 集合$\left\{  {{W}_{1},\ldots ,{W}_{t}}\right\}$。这些结构可以在$O\left( {m\log m}\right)$时间内构建，并允许我们在第2 - 4行使用$O\left( {\log m}\right)$时间计算${w}_{1},{w}_{2},{w}_{3}$。由此可知，归约阶段需要$O\left( {m\log m}\right)  = O\left( {\frac{n}{t}\log m}\right)$时间。

## 6. EXPERIMENTS

## 6. 实验

This section experimentally evaluates our algorithms on an in-house cluster with one master and 56 slave nodes, each of which has four Intel Xeon 2.4GHz CPUs and 24GB RAM. We implement all algorithms on Hadoop (version 1.0), and allocate 4GB of RAM to the Java Virtual Machine on each node (i.e., each node can use up to $4\mathrm{{GB}}$ of memory for a Hadoop task). Table 1 lists the Hadoop parameters in our experiments.

本节在一个内部集群上对我们的算法进行实验评估，该集群有一个主节点和56个从节点，每个从节点配备四个英特尔至强2.4GHz CPU和24GB内存。我们在Hadoop（版本1.0）上实现所有算法，并为每个节点上的Java虚拟机分配4GB内存（即每个节点在Hadoop任务中最多可使用$4\mathrm{{GB}}$的内存）。表1列出了我们实验中的Hadoop参数。

<!-- Media -->

<table><tr><td>$\mathbf{{ParameterName}}$</td><td>Value</td></tr><tr><td>fs.block.size</td><td>128MB</td></tr><tr><td>io.sort.mb</td><td>512MB</td></tr><tr><td>io.sort.record.percentage</td><td>0.1</td></tr><tr><td>io.sort.spill.percentage</td><td>0.9</td></tr><tr><td>io.sort.factor</td><td>300</td></tr><tr><td>dfs.replication</td><td>3</td></tr></table>

<table><tbody><tr><td>$\mathbf{{ParameterName}}$</td><td>值</td></tr><tr><td>文件系统块大小（fs.block.size）</td><td>128兆字节</td></tr><tr><td>排序缓冲区大小（io.sort.mb）</td><td>512兆字节</td></tr><tr><td>排序记录百分比（io.sort.record.percentage）</td><td>0.1</td></tr><tr><td>排序溢出百分比（io.sort.spill.percentage）</td><td>0.9</td></tr><tr><td>排序因子（io.sort.factor）</td><td>300</td></tr><tr><td>分布式文件系统副本数（dfs.replication）</td><td>3</td></tr></tbody></table>

Table 1: Parameters of Hadoop

表1：Hadoop的参数

<!-- Media -->

We deploy two real datasets named LIDAR ${}^{6}$ and PageView ${}^{7}$ , respectively. 514GB in size, LIDAR contains 7.35 billion records, each of which is a 3D point representing a location in North Carolina. We use LIDAR for experiments on sorting, skyline, group by, and semi-join. PageView is 332GB in size and contains 11.8 billion tuples. Each tuple corresponds to a page on Wikipedia, and records the number of times the page was viewed in a certain hour during Jan-Sep 2012. We impose a total order on all the tuples by their timestamps, and use the data for experiments on sliding aggregation. In addition, we also generate synthetic datasets to investigate the effect of data distribution on the performance of different algorithms. In each experiment, we run an algorithm 5 times and report the average reading.

我们分别部署了两个真实数据集，名为激光雷达（LIDAR）${}^{6}$和页面浏览量（PageView）${}^{7}$。激光雷达数据集大小为514GB，包含73.5亿条记录，每条记录是一个代表北卡罗来纳州某个位置的三维点。我们使用激光雷达数据集进行排序、天际线、分组和半连接实验。页面浏览量数据集大小为332GB，包含118亿个元组。每个元组对应维基百科上的一个页面，并记录了该页面在2012年1月至9月期间某一小时内的浏览次数。我们根据所有元组的时间戳对它们进行全排序，并使用该数据进行滑动聚合实验。此外，我们还生成了合成数据集，以研究数据分布对不同算法性能的影响。在每个实验中，我们将一个算法运行5次，并报告平均结果。

---

<!-- Footnote -->

${}^{5}$ Let $S$ be a set of $n$ real values,each associated with a numeric weight. Given an interval $I$ ,a range sum query returns the total weight of the values in $S \cap  I$ . A simple augmented binary tree [14] uses $O\left( n\right)$ space,answers a query in $O\left( {\log n}\right)$ time,and can be built in $O\left( {n\log n}\right)$ time.

${}^{5}$ 设 $S$ 为一组包含 $n$ 个实数值的集合，每个实数值都关联一个数值权重。给定一个区间 $I$ ，范围求和查询返回 $S \cap  I$ 中值的总权重。一个简单的扩充二叉树 [14] 使用 $O\left( n\right)$ 空间，在 $O\left( {\log n}\right)$ 时间内回答查询，并可以在 $O\left( {n\log n}\right)$ 时间内构建完成。

${}^{6}$ Http://www.ncfloodmaps.com.

${}^{6}$ 网址：Http://www.ncfloodmaps.com。

${}^{7}$ Http://dumps.wikimedia.org/other/pagecounts-raw.

${}^{7}$ 网址：Http://dumps.wikimedia.org/other/pagecounts-raw。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Pure TeraSort ${HS}$ 100 maximum local data (GB) 80 60 40 20 0 100 200 300 400 500 dataset size (GB) (b) Max. data volume on a slave Figure 2: Pure TeraSort vs. HS on LIDAR. ${HS}$ 160 maximum local data (GB) 120 80 40 0 100 200 300 400 500 dataset size (GB) (b) Max. data volume on a slave Figure 3: Pure TeraSort vs. HS on modified LIDAR maximum local data (GB) 20 15 10 5 ${6}^{-1}$ ${6}^{1}$ ${6}^{2}$ ${6}^{3}$ sample set size $\left( {\times t\ln \left( {nt}\right) }\right)$ (b) Max. data volume on a slave 10000 total processing time (sec, 8000 6000 4000 2000 100 200 300 400 500 dataset size (GB) (a) Total time Pure TeraSort 20000 total processing time (sec) 15000 10000 5000 100 200 300 400 500 dataset size (GB) (a) Total time 3000 total processing time (sec) 2000 1000 ${6}^{ - }$ ${6}^{1}$ ${6}^{2}$ ${6}^{3}$ sample set size $\left( {\times t\ln \left( {nt}\right) }\right)$ (a) Total time -->

<img src="https://cdn.noedgeai.com/0195c8fe-f266-7d5f-9ab0-051892ebd600_8.jpg?x=134&y=147&w=726&h=1234&r=0"/>

Figure 4: Effects of sample size on pure TeraSort

图4：样本大小对纯TeraSort算法的影响

<!-- Media -->

### 6.1 Sorting

### 6.1 排序

The first set of experiments compares pure TeraSort (proposed in Section 3.3) with Hadoop's default sorting algorithm, referred to as ${HS}$ henceforth.

第一组实验将纯TeraSort算法（在3.3节中提出）与Hadoop的默认排序算法进行了比较，此后将默认排序算法称为 ${HS}$。

Given a dataset of $k$ blocks long in the Hadoop Distributed File System (HDFS), HS first asks the master node to gather the first $\left\lceil  {{10}^{5}/k}\right\rceil$ records of each block into a set $S$ - call them the pilot records. Next,the master identifies ${t}_{\text{slave }} - 1$ boundary points ${b}_{1},{b}_{2},\ldots ,{b}_{{t}_{\text{slave }} - 1}$ ,where ${b}_{i}$ is the $i\left\lceil  {{10}^{5}/{t}_{\text{slave }}}\right\rceil$ -th smallest record in $S$ ,and ${t}_{\text{slave }}$ is the number of slave nodes. The mater then launches a one-round algorithm where all records in $\left( {{b}_{i - 1},{b}_{i}}\right\rbrack$ are sent to the $i$ -th $\left( {i \in  \left\lbrack  {1,{t}_{\text{slave }}}\right\rbrack  }\right)$ slave for sorting,where ${b}_{0} = 0$ and ${b}_{t} = \infty$ are dummies. Clearly,the efficiency of ${HS}$ relies on the distribution of the pilot records. If their distribution is the same as the whole dataset, each slave sorts approximately an equal number of tuples. Otherwise, certain slaves may receive an excessive amount of data and thus become the bottleneck of sorting.

给定Hadoop分布式文件系统（HDFS）中长度为 $k$ 个块的数据集，HS算法首先要求主节点将每个块的前 $\left\lceil  {{10}^{5}/k}\right\rceil$ 条记录收集到一个集合 $S$ 中 —— 称它们为引导记录。接下来，主节点确定 ${t}_{\text{slave }} - 1$ 个边界点 ${b}_{1},{b}_{2},\ldots ,{b}_{{t}_{\text{slave }} - 1}$ ，其中 ${b}_{i}$ 是 $S$ 中第 $i\left\lceil  {{10}^{5}/{t}_{\text{slave }}}\right\rceil$ 小的记录， ${t}_{\text{slave }}$ 是从节点的数量。然后，主节点启动一轮算法，将 $\left( {{b}_{i - 1},{b}_{i}}\right\rbrack$ 中的所有记录发送到第 $i$ 个 $\left( {i \in  \left\lbrack  {1,{t}_{\text{slave }}}\right\rbrack  }\right)$ 从节点进行排序，其中 ${b}_{0} = 0$ 和 ${b}_{t} = \infty$ 是虚拟变量。显然， ${HS}$ 的效率依赖于引导记录的分布。如果它们的分布与整个数据集相同，每个从节点将对大致相等数量的元组进行排序。否则，某些从节点可能会收到过多的数据，从而成为排序的瓶颈。

<!-- Media -->

<!-- figureText: minimal-Sky MR-SFS 50 maximum local data (GB) 40 30 20 10 0 100 200 300 400 500 dataset size (GB) (b) Max. data volume on a slave 20000 total processing time (sec) 15000 10000 5000 0 100 200 300 400 500 dataset size (GB) (a) Total time -->

<img src="https://cdn.noedgeai.com/0195c8fe-f266-7d5f-9ab0-051892ebd600_8.jpg?x=912&y=148&w=708&h=391&r=0"/>

Figure 5: Minimal-Sky vs. MR-SFS on LIDAR

图5：在激光雷达数据集上Minimal - Sky算法与MR - SFS算法的比较

<!-- Media -->

We implement pure TeraSort in a way similar to HS with the difference in how pilot records are picked. Specifically, the master now forms $S$ by randomly sampling $t\ln t$ records from the dataset. Figure 2a illustrates the running time of ${HS}$ and pure TeraSort in sorting LIDAR by its first dimension, when the dataset size varies from 51.4GB to ${514}\mathrm{{GB}}$ (a dataset with size smaller than ${514}\mathrm{{GB}}$ consists of random tuples from LIDAR, preserving their original ordering.) Pure TeraSort consistently outperforms ${HS}$ ,with the difference becoming more significant as the size grows. To reveal the reason behind, we plot in Figure 2b the maximum data amount on a slave node in the above experiments. Evidently, while pure TeraSort distributes the data evenly to the slaves, ${HS}$ sends a large portion to a single slave, thus incurring enormous overhead.

我们以类似于HS算法的方式实现了纯TeraSort算法，不同之处在于引导记录的选取方式。具体来说，主节点现在通过从数据集中随机采样 $t\ln t$ 条记录来形成 $S$。图2a展示了当数据集大小从51.4GB变化到 ${514}\mathrm{{GB}}$ 时， ${HS}$ 和纯TeraSort算法按激光雷达数据集的第一维进行排序的运行时间（大小小于 ${514}\mathrm{{GB}}$ 的数据集由激光雷达数据集中的随机元组组成，并保留它们的原始顺序）。纯TeraSort算法始终优于 ${HS}$ ，并且随着数据集大小的增加，这种差异变得更加显著。为了揭示背后的原因，我们在图2b中绘制了上述实验中从节点上的最大数据量。显然，纯TeraSort算法将数据均匀地分配给从节点，而 ${HS}$ 则将大部分数据发送到单个从节点，从而产生了巨大的开销。

To further demonstrate the deficiency of ${HS}$ ,Figure 3a shows the time taken by pure TeraSort and HS to sort a modified version of LIDAR, where tuples with small first coordinates are put to the beginning of each block. The efficiency of ${HS}$ deteriorates dramatically, as shown in Figure 3b, confirming the intuition that its cost is highly sensitive to the distribution of pilot records. In contrast, the performance of pure TeraSort is not affected, owning to the fact that its sampling procedure is not sensitive to original data ordering at all.

为了进一步证明${HS}$的不足，图3a展示了纯TeraSort和HS对修改后的激光雷达（LIDAR）数据进行排序所需的时间，在修改后的数据中，首坐标较小的元组被放置在每个数据块的开头。如图3b所示，${HS}$的效率急剧下降，这证实了一种直觉，即其成本对引导记录的分布高度敏感。相比之下，纯TeraSort的性能不受影响，因为其采样过程根本不依赖于原始数据的顺序。

To demonstrate the effect of sample size, Figure 4a shows the cost of pure Terasort on LIDAR as the number of pilot tuples changes. The result suggests that $t\ln \left( {nt}\right)$ is a nice choice. When the sample size decreases, pure Terasort is slower due to the increased unbalance in the distribution of data across the slaves. as can be observed from Figure 4b. On the opposite side, when the sample size grows, the running time also lengthens because sampling itself is more expensive.

为了展示样本大小的影响，图4a展示了随着引导元组数量的变化，纯TeraSort对激光雷达（LIDAR）数据进行排序的成本。结果表明$t\ln \left( {nt}\right)$是一个不错的选择。从图4b可以看出，当样本大小减小时，由于数据在各个从节点上分布的不平衡性增加，纯TeraSort的速度会变慢。相反，当样本大小增大时，运行时间也会变长，因为采样本身的成本更高。

### 6.2 Skyline

### 6.2 天际线（Skyline）

The second set of experiments evaluates our skyline algorithm, referred to as minimal-Sky, against MR-SFS [61], a recently developed method for skyline computation in MapReduce. We use exactly the implementation of ${MR}$ -SFS from its authors. Figure 5a compares the cost of minimal-Sky and MR-SFS in finding the skyline on the first two dimensions of LIDAR, as the dataset size increases. Minimal-Sky significantly outperforms MR-SFS in all cases. The reason is that MR-SFS, which is not a minimal algorithm, may force a slave node to process an excessive amount of data, as shown in Figure 5b.

第二组实验将我们的天际线算法（称为最小天际线算法，minimal - Sky）与MR - SFS [61]进行了比较，MR - SFS是最近开发的一种用于MapReduce中天际线计算的方法。我们使用了${MR}$ - SFS作者提供的精确实现。图5a比较了随着数据集大小的增加，最小天际线算法（minimal - Sky）和MR - SFS在查找激光雷达（LIDAR）数据前两个维度的天际线时的成本。在所有情况下，最小天际线算法（minimal - Sky）都明显优于MR - SFS。原因在于，MR - SFS不是一种最小化算法，如图5b所示，它可能会迫使一个从节点处理过多的数据。

Figure 6 illustrates the performance of minimal-Sky and MR-SFS on three synthetic datasets that follow a correlated, anti-correlated, and independent distribution, respectively. 120GB in size, each dataset contains 2.5 billion 2D points generated by a publicly available toolkit ${}^{8}$ . Clearly,MR-SFS is rather sensitive to the dataset distribution, whereas the efficiency of Minimal-Sky is not affected at all.

图6展示了最小天际线算法（minimal - Sky）和MR - SFS在三个合成数据集上的性能，这三个数据集分别遵循相关分布、反相关分布和独立分布。每个数据集大小为120GB，包含由公开可用的工具包${}^{8}$生成的25亿个二维点。显然，MR - SFS对数据集分布相当敏感，而最小天际线算法（minimal - Sky）的效率根本不受影响。

---

<!-- Footnote -->

${}^{8}$ Http://pgfoundry.org/projects/randdataset.

${}^{8}$ Http://pgfoundry.org/projects/randdataset.

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: minimal-Sky MR-SFS 0222 maximum local data (GB) 40 35 15 10 data distribution (b) Max. data volume on a slave base-GB maximum local data (GB) 50 40 30 20 10 100 200 300 400 500 dataset size (GB) (b) Max. data volume on a slave 12000 total processing time (sec) 10000 8000 6000 4000 2000 0 anti-correlated and correlated data distribution (a) Total time minimal-GB 10000 total processing time (sec) 8000 6000 4000 2000 100 200 300 400 500 dataset size (GB) (a) Total time -->

<img src="https://cdn.noedgeai.com/0195c8fe-f266-7d5f-9ab0-051892ebd600_9.jpg?x=134&y=146&w=716&h=823&r=0"/>

Figure 7: Minimal-GB vs. base-GB on LIDAR

图7：最小分组算法（Minimal - GB）与基准分组算法（base - GB）在激光雷达（LIDAR）数据上的比较

<!-- Media -->

### 6.3 Group By

### 6.3 分组（Group By）

Next, we compare our group by algorithm, referred to as minimal- ${GB}$ ,with a baseline approach called base- ${GB}$ . Suppose that we are to group a dataset $D$ by an attribute $A$ . Base- ${GB}$ first invokes a map phase where each tuple $t \in  D$ spawns a key-value pair $\left( {t\left\lbrack  A\right\rbrack  ,t}\right)$ ,where $t\left\lbrack  A\right\rbrack$ is the value of $t$ on $A$ . Then,all key-value pairs are distributed to the slave nodes using Hadoop's Partitioner program. Finally, every slave aggregates the key-value pairs it receives to compute the group by results.

接下来，我们将我们的分组算法（称为最小${GB}$算法，minimal - ${GB}$）与一种称为基准${GB}$算法（base - ${GB}$）的基线方法进行比较。假设我们要根据属性$A$对数据集$D$进行分组。基准${GB}$算法（base - ${GB}$）首先调用一个映射阶段，在该阶段中，每个元组$t \in  D$生成一个键值对$\left( {t\left\lbrack  A\right\rbrack  ,t}\right)$，其中$t\left\lbrack  A\right\rbrack$是$t$在$A$上的值。然后，使用Hadoop的分区程序将所有键值对分配到从节点。最后，每个从节点聚合其接收到的键值对以计算分组结果。

Figure 7a presents the cost of minimal- ${GB}$ and base- ${GB}$ in grouping LIDAR by its first attribute. Regardless of the dataset size, minimal- ${GB}$ is considerably faster than base- ${GB}$ which,as shown in Figure 7b, is because Hadoop's Partitioner does not distribute data across the slaves as evenly as minimal- ${GB}$ .

图7a展示了最小${GB}$算法（minimal - ${GB}$）和基准${GB}$算法（base - ${GB}$）根据激光雷达（LIDAR）数据的第一个属性进行分组的成本。无论数据集大小如何，最小${GB}$算法（minimal - ${GB}$）都比基准${GB}$算法（base - ${GB}$）快得多，如图7b所示，这是因为Hadoop的分区程序在从节点之间分配数据的均匀性不如最小${GB}$算法（minimal - ${GB}$）。

To evaluate the effect of dataset distribution, we generate 2D synthetic datasets where the first dimension (i) has an integer domain $\left\lbrack  {1,{2.5} \times  {10}^{8}}\right\rbrack$ ,and (ii) follows a Zipf distribution with a skew factor between 0 and $1.{}^{9}$ Each dataset contains 5 billion tuples and is ${90}\mathrm{{GB}}$ in size. Figure 8 illustrates the performance of minimal- ${GB}$ and base- ${GB}$ on grouping the synthetic datasets by their first attributes. The efficiency of base- ${GB}$ deteriorates as the skew factor increases. This is because base-GB always sends tuples with an identical group-by key to the same slave node. When the group-by keys are skewed, the data distribution is very uneven on the slaves, leading to severe performance penalty. In contrast, minimal- ${GB}$ is completely insensitive to data skewness.

为了评估数据集分布的影响，我们生成了二维合成数据集，其中第一个维度（i）的整数域为$\left\lbrack  {1,{2.5} \times  {10}^{8}}\right\rbrack$，并且（ii）遵循偏斜因子在0到$1.{}^{9}$之间的齐普夫分布（Zipf distribution）。每个数据集包含50亿个元组，大小为${90}\mathrm{{GB}}$。图8展示了最小${GB}$算法（minimal - ${GB}$）和基准${GB}$算法（base - ${GB}$）根据合成数据集的第一个属性进行分组的性能。随着偏斜因子的增加，基准${GB}$算法（base - ${GB}$）的效率下降。这是因为基准分组算法（base - GB）总是将具有相同分组键的元组发送到同一个从节点。当分组键存在偏斜时，数据在从节点上的分布非常不均匀，导致严重的性能损失。相比之下，最小${GB}$算法（minimal - ${GB}$）对数据偏斜完全不敏感。

<!-- Media -->

<!-- figureText: minimal-GB base-GB 15 maximum local data (GB) 10 5 0 0 0.2 0.4 0.6 0.8 1 skew factor (b) Max. data volume on a slave Figure 8: Minimal-GB vs. base-GB on synthetic data PSSU maximum local data (GB) 15 10 5 0 ${10}^{-3}$ ${10}^{-2}$ ${10}^{-1}$ referencing factor (b) Max. data volume on a slave 10000 stal processing time (sec) 8000 6000 4000 2000 0 0 0.2 0.4 0.6 0.8 skew factor (a) Total time 700 total processing time (sec, 600 500 400 300 200 100 0 ${10}^{-3}$ ${10}^{-2}$ ${10}^{-1}$ referencing factor (a) Total time -->

<img src="https://cdn.noedgeai.com/0195c8fe-f266-7d5f-9ab0-051892ebd600_9.jpg?x=917&y=144&w=711&h=820&r=0"/>

Figure 9: Minimal-SJ vs. PSSJ on various referencing factors

图9：最小半连接（Minimal - SJ）与按分割半连接（PSSJ）在不同引用因子下的比较

<!-- Media -->

### 6.4 Semi-Join

### 6.4 半连接

We now proceed to evaluate our semi-join algorithm, referred to as minimal-SJ, with Per-Split Semi-Join (PSSJ) [7], which is the best existing MapReduce semi-join algorithm. We adopt the implementation of PSSJ that has been made available online at sites.google.com/site/hadoopcs561. Following [7], we generate synthetic tables $T$ and $R$ as follows. The attributes of $T$ are ${A}_{1}$ and ${A}_{2}$ ,both of which have an integer domain of $\left\lbrack  {1,{2.5} \times  {10}^{8}}\right\rbrack$ . $T$ has 5 billion tuples whose ${A}_{1}$ values follow a Zipf distribution (some tuples may share an identical value). Their ${A}_{2}$ values are unimportant and arbitrarily decided. Similarly, $R$ has 10 million tuples with integer attributes ${A}_{1}$ and ${A}_{3}$ of domain $\left\lbrack  {1,{2.5} \times  {10}^{8}}\right\rbrack$ . A fraction $r$ of the tuples in $R$ carry ${A}_{1}$ values present in $T$ ,while the other tuples have ${A}_{1}$ values absent from $T$ . We refer to $r$ as the referencing factor. Tuples’ ${A}_{3}$ values are unimportant and arbitrarily determined.

现在，我们将评估我们的半连接算法（称为最小半连接，minimal - SJ）与按分割半连接（Per - Split Semi - Join，PSSJ）[7]的性能，PSSJ是现有的最佳MapReduce半连接算法。我们采用了可在sites.google.com/site/hadoopcs561上在线获取的PSSJ实现。按照文献[7]的方法，我们按如下方式生成合成表$T$和$R$。$T$的属性为${A}_{1}$和${A}_{2}$，二者的整数域均为$\left\lbrack  {1,{2.5} \times  {10}^{8}}\right\rbrack$。$T$有50亿个元组，其${A}_{1}$值遵循齐普夫分布（有些元组可能具有相同的值）。它们的${A}_{2}$值并不重要，可任意确定。类似地，$R$有1000万个元组，其整数属性${A}_{1}$和${A}_{3}$的域为$\left\lbrack  {1,{2.5} \times  {10}^{8}}\right\rbrack$。$R$中有比例为$r$的元组的${A}_{1}$值存在于$T$中，而其他元组的${A}_{1}$值不存在于$T$中。我们将$r$称为引用因子。元组的${A}_{3}$值并不重要，可任意确定。

Figure 9 compares minimal-SJ and PSSJ under different referencing factors,when the skew factor of $T.{A}_{1}$ equals 0.4 . In all scenarios, Minimal-SJ beats PSSJ by a wide margin. Figure 10a presents the running time of minimal-SJ and PSSJ as a function of the skew factor of $T.{A}_{1}$ ,setting the reference factor $r$ to 0.1 . The efficiency of PSSJ degrades rapidly as the skew factor grows which, as shown Figure 10b, is because PSSJ fails to distribute the workload evenly among the slaves. Minimal-SJ is not affected by skewness.

图9展示了当$T.{A}_{1}$的倾斜因子等于0.4时，最小半连接（Minimal - SJ）和按分割半连接（PSSJ）在不同引用因子下的比较情况。在所有场景中，最小半连接（Minimal - SJ）都大幅优于按分割半连接（PSSJ）。图10a展示了最小半连接（Minimal - SJ）和按分割半连接（PSSJ）的运行时间随$T.{A}_{1}$倾斜因子的变化情况，其中引用因子$r$设为0.1。随着倾斜因子的增大，按分割半连接（PSSJ）的效率迅速下降，如图10b所示，这是因为按分割半连接（PSSJ）未能在从节点之间均匀分配工作负载。而最小半连接（Minimal - SJ）不受倾斜度的影响。

### 6.5 Sliding Aggregation

### 6.5 滑动聚合

In the last set of experiments, we evaluate our sliding aggregation algorithm, referred to as minimal-SA, against a baseline solution referred to as Jaql, which corresponds to the algorithm proposed in [6]. Suppose that we want to perform sliding aggregation over a set $S$ of objects using a window size $l \leq  n$ . Jaql first sorts $S$ with our ranking algorithm (Section 4.1). It then maps each record $t \in  S$ to $l$ key-value pair $\left( {{t}_{1},t}\right) ,\ldots ,\left( {{t}_{l},t}\right)$ ,where ${t}_{1},\ldots ,{t}_{l}$ are the $l$ largest objects not exceeding $t$ . Then,Jaql distributes the key-value pairs to the slaves by applying Hadoop's Partitioner, and instructs each slave to aggregate the key-value pairs with the same key. Besides our own implementation of Jaql, we also examine the original implementation released by the authors of [6], henceforth called Jaql-original.

在最后一组实验中，我们将评估我们的滑动聚合算法（称为最小滑动聚合，minimal - SA）与一种基线解决方案（称为Jaql）的性能，Jaql对应于文献[6]中提出的算法。假设我们要使用窗口大小$l \leq  n$对对象集合$S$执行滑动聚合。Jaql首先使用我们的排序算法（第4.1节）对$S$进行排序。然后，它将每条记录$t \in  S$映射为$l$键值对$\left( {{t}_{1},t}\right) ,\ldots ,\left( {{t}_{l},t}\right)$，其中${t}_{1},\ldots ,{t}_{l}$是不超过$t$的$l$个最大对象。接着，Jaql通过应用Hadoop的分区器将键值对分配给从节点，并指示每个从节点对具有相同键的键值对进行聚合。除了我们自己实现的Jaql，我们还测试了文献[6]作者发布的原始实现，此后称为原始Jaql（Jaql - original）。

---

<!-- Footnote -->

${}^{9}$ Data are more skewed when the skew factor is higher. In particular, when the factor is 0 , the distribution degenerates into uniformity.

${}^{9}$ 倾斜因子越高，数据的倾斜程度越大。特别地，当倾斜因子为0时，分布退化为均匀分布。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: minimal-SJ PSSJ 25 maximum local data (GB) 20 15 10 5 0 0 0.2 0.4 0.6 0.8 1 skew factor (b) Max. data volume on a slave Figure 10: Minimal-SJ vs. PSSJ on various skew factors Jaql ✘ Jaql-original 40 maximum local data (GB) 30 20 10 0 2 6 10 window length (b) Max. data volume on a slave 800 total processing time (sec) 600 400 200 0 0 0.2 0.4 0.6 0.8 1 skew factor (a) Total time ${10}^{6}$ total processing time (sec) ${10}^{5}$ ${10}^{4}$ ${10}^{3}$ 2 6 8 10 window length (a) Total time -->

<img src="https://cdn.noedgeai.com/0195c8fe-f266-7d5f-9ab0-051892ebd600_10.jpg?x=141&y=147&w=712&h=818&r=0"/>

Figure 11: Sliding aggregation on small window sizes

图11：小窗口大小下的滑动聚合

<!-- Media -->

Figure 11 demonstrates the performance of minimal-SA, Jaql, and Jaql-original on the PageView dataset,varying $l$ from 2 to 10 . Minimal-SA is superior to Jaql in all settings, except for a single case $l = 2$ . In addition,minimal- ${SA}$ is not affected by $l$ ,while Jaql deteriorates linearly. Jaql-original is slower than the other two methods by a factor of over an order of magnitude. It is not included in Figure 11b because it needs to keep almost the entire database on a single machine, which becomes the system's bottleneck.

图11展示了最小滑动聚合（minimal - SA）、Jaql和原始Jaql（Jaql - original）在PageView数据集上的性能，其中$l$从2变化到10。除了一种情况$l = 2$外，在所有设置下最小滑动聚合（minimal - SA）都优于Jaql。此外，最小${SA}$不受$l$的影响，而Jaql的性能呈线性下降。原始Jaql（Jaql - original）比其他两种方法慢一个数量级以上。图11b中未包含它，因为它需要将几乎整个数据库保留在一台机器上，这成为了系统的瓶颈。

Focusing on large $l$ ,Figure 12 plots the running time of minimal-SA when $l$ increases from ${10}^{5}$ to ${10}^{9}$ . We omit the Jaql implementations because they are prohibitively expensive, and worse than minimal-SA by more than a thousand times.

聚焦于较大的$l$，图12展示了最小自连接算法（minimal - SA）在$l$从${10}^{5}$增加到${10}^{9}$时的运行时间。我们省略了Jaql实现，因为它们的成本过高，并且比最小自连接算法（minimal - SA）慢一千多倍。

## 7. CONCLUSIONS

## 7. 结论

MapReduce has grown into an extremely popular architecture for large-scaled parallel computation. Even though there have been a great variety of algorithms developed for MapReduce, few are able to achieve the ideal goal of parallelization: balanced workload across the participating machines, and a speedup over a sequential algorithm linear to the number of machines. In particular, currently there is a void at the conceptual level as to what it means to be a "good" MapReduce algorithm.

MapReduce已发展成为一种非常流行的大规模并行计算架构。尽管已经为MapReduce开发了各种各样的算法，但很少有算法能够实现并行化的理想目标：参与计算的机器之间工作负载均衡，并且相对于顺序算法的加速比与机器数量呈线性关系。特别是，目前在概念层面上对于什么是“好的”MapReduce算法还存在空白。

We believe that a major contribution of this paper is to fill the aforementioned void with the new notion of "minimal MapReduce algorithm". This notion puts together for the first time four strong criteria towards (at least asymptotically) the highest parallel degree. At first glance, the conditions of minimality appear to be fairly stringent. Nonetheless, we prove the existence of simple yet elegant algorithms that minimally settle an array of important database problems. Our extensive experimentation demonstrates the immediate benefit brought forward by minimality that, the proposed algorithms significantly improve the existing state of the art for all the problems tackled.

我们认为，本文的一个主要贡献是用“最小MapReduce算法”这一新概念填补了上述空白。这一概念首次将四个强有力的标准结合起来，以（至少渐近地）实现最高的并行度。乍一看，最小性的条件似乎相当严格。尽管如此，我们证明了存在简单而优雅的算法，能够以最小的方式解决一系列重要的数据库问题。我们广泛的实验表明，最小性带来了直接的好处，即所提出的算法显著改进了所有已解决问题的现有技术水平。

<!-- Media -->

<!-- figureText: 6000 total processing time (sec) maximum local data (GB) 5 4 3 2 ${10}^{2}$ ${10}^{6}$ ${10}^{7}$ ${10}^{8}$ ${10}^{9}$ window length (b) Max. data volume on a slave 5000 4000 3000 2000 1000 0 ${10}^{5}$ ${10}^{6}$ ${10}^{7}$ ${10}^{8}$ ${10}^{9}$ window length (a) Total time -->

<img src="https://cdn.noedgeai.com/0195c8fe-f266-7d5f-9ab0-051892ebd600_10.jpg?x=917&y=150&w=704&h=358&r=0"/>

Figure 12: Minimal-SA on large window sizes

图12：大窗口大小下的最小自连接算法（Minimal - SA）

<!-- Media -->

## ACKNOWLEDGEMENTS

## 致谢

Yufei Tao was supported in part by (i) projects GRF 4166/10. 4165/11, and 4164/12 from HKRGC, and (ii) the WCU (World Class University) program under the National Research Foundation of Korea, and funded by the Ministry of Education, Science and Technology of Korea (Project No: R31-30007). Wenqing Lin and Xiaokui Xiao were supported by the Nanyang Technological University under SUG Grant M58020016, and by the Agency for Science, Technology, and Research (Singapore) under SERC Grant 102-158-0074. The authors would like to thank the anonymous reviewers for their insightful comments.

陶宇飞部分得到了以下资助：（i）香港研究资助局（HKRGC）的GRF 4166/10、4165/11和4164/12项目；（ii）韩国国家研究基金会（National Research Foundation of Korea）的世界级大学（WCU）计划，由韩国教育、科学和技术部资助（项目编号：R31 - 30007）。林文清和肖晓奎得到了南洋理工大学SUG资助项目M58020016的支持，以及新加坡科技研究局（Agency for Science, Technology, and Research）SERC资助项目102 - 158 - 0074的支持。作者感谢匿名审稿人提出的深刻见解。

## 8. REFERENCES

## 8. 参考文献

[1] A. Abouzeid, K. Bajda-Pawlikowski, D. J. Abadi, A. Rasin, and

[1] A. 阿布泽德（A. Abouzeid）、K. 巴伊达 - 帕夫利科夫斯基（K. Bajda - Pawlikowski）、D. J. 阿巴迪（D. J. Abadi）、A. 拉辛（A. Rasin）和

A. Silberschatz. Hadoopdb: An architectural hybrid of mapreduce and dbms technologies for analytical workloads. PVLDB, 2(1):922-933, 2009.

[2] F. N. Afrati, A. D. Sarma, D. Menestrina, A. G. Parameswaran, and J. D. Ullman. Fuzzy joins using mapreduce. In ICDE, pages 498-509, 2012.

[3] F. N. Afrati and J. D. Ullman. Optimizing multiway joins in a map-reduce environment. TKDE, 23(9):1282-1298, 2011.

[4] B. Bahmani, K. Chakrabarti, and D. Xin. Fast personalized pagerank on mapreduce. In SIGMOD, pages 973-984, 2011.

[5] B. Bahmani, R. Kumar, and S. Vassilvitskii. Densest subgraph in streaming and mapreduce. PVLDB, 5(5):454-465, 2012.

[6] K. S. Beyer, V. Ercegovac, R. Gemulla, A. Balmin, M. Y. Eltabakh, C.-C. Kanne, F. Özcan, and E. J. Shekita. Jaql: A scripting language for large scale semistructured data analysis. PVLDB, 4(12):1272-1283, 2011.

[7] S. Blanas, J. M. Patel, V. Ercegovac, J. Rao, E. J. Shekita, and Y. Tian. A comparison of join algorithms for log processing in mapreduce. In SIGMOD, pages 975-986, 2010.

[8] S. Borzsonyi, D. Kossmann, and K. Stocker. The skyline operator. In ICDE, pages 421-430, 2001.

[9] R. Chaiken, B. Jenkins, P. ake Larson, B. Ramsey, D. Shakib, S. Weaver, and J. Zhou. Scope: easy and efficient parallel processing of massive data sets. PVLDB, 1(2):1265-1276, 2008.

[10] B. Chattopadhyay, L. Lin, W. Liu, S. Mittal, P. Aragonda, V. Lychagina, Y. Kwon, and M. Wong. Tenzing a sql implementation on the mapreduce framework. PVLDB, 4(12):1318-1327, 2011.

[11] S. Chen. Cheetah: A high performance, custom data warehouse on top of mapreduce. PVLDB, 3(2):1459-1468, 2010.

[12] F. Chierichetti, R. Kumar, and A. Tomkins. Max-cover in map-reduce. In ${WWW}$ ,pages ${231} - {240},{2010}$ .

[13] R. L. F. Cordeiro, C. T. Jr., A. J. M. Traina, J. Lopez, U. Kang, and C. Faloutsos. Clustering very large multi-dimensional datasets with mapreduce. In SIGKDD, pages 690-698, 2011.

[14] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein. Introduction to Algorithms, Second Edition. The MIT Press, 2001.

[14] T. H. 科尔曼（T. H. Cormen）、C. E. 莱瑟森（C. E. Leiserson）、R. L. 里弗斯特（R. L. Rivest）和C. 斯坦（C. Stein）。《算法导论》（Introduction to Algorithms），第二版。麻省理工学院出版社，2001年。

[15] A. Das, M. Datar, A. Garg, and S. Rajaram. Google news

[15] A. 达斯（A. Das）、M. 达塔尔（M. Datar）、A. 加尔格（A. Garg）和S. 拉贾拉姆（S. Rajaram）。谷歌新闻

personalization: scalable online collaborative filtering. In WWW, pages 271-280, 2007.

[16] J. Dean and S. Ghemawat. Mapreduce: Simplified data processing on large clusters. In OSDI, pages 137-150, 2004.

[17] F. K. H. A. Dehne, A. Fabri, and A. Rau-Chaplin. Scalable parallel geometric algorithms for coarse grained multicomputers. In ${SoCG}$ , pages 298-307, 1993.

[18] J. Dittrich, J.-A. Quiane-Ruiz, A. Jindal, Y. Kargin, V. Setty, and J. Schad. Hadoop++: Making a yellow elephant run like a cheetah (without it even noticing). PVLDB, 3(1):518-529, 2010.

[19] I. Elghandour and A. Aboulnaga. Restore: Reusing results of mapreduce jobs. PVLDB, 5(6):586-597, 2012.

[20] M. Y. Eltabakh, Y. Tian, F. Ozcan, R. Gemulla, A. Krettek, and J. McPherson. Cohadoop: Flexible data placement and its exploitation in hadoop. PVLDB, 4(9):575-585, 2011.

[21] A. Ene, S. Im, and B. Moseley. Fast clustering using mapreduce. In SIGKDD, pages 681-689, 2011.

[22] A. Floratou, J. M. Patel, E. J. Shekita, and S. Tata. Column-oriented storage techniques for mapreduce. PVLDB, 4(7):419-429, 2011.

[23] A. Ghoting, P. Kambadur, E. P. D. Pednault, and R. Kannan. Nimble: a toolkit for the implementation of parallel data mining and machine learning algorithms on mapreduce. In SIGKDD, pages 334-342, 2011.

[24] A. Ghoting, R. Krishnamurthy, E. P. D. Pednault, B. Reinwald, V. Sindhwani, S. Tatikonda, Y. Tian, and S. Vaithyanathan. Systemml: Declarative machine learning on mapreduce. In ${ICDE}$ ,pages 231-242, 2011.

[25] R. Grover and M. J. Carey. Extending map-reduce for efficient predicate-based sampling. In ICDE, pages 486-497, 2012.

[26] B. Gufler, N. Augsten, A. Reiser, and A. Kemper. Load balancing in mapreduce based on scalable cardinality estimates. In ${ICDE}$ ,pages 522-533, 2012.

[27] Y. He, R. Lee, Y. Huai, Z. Shao, N. Jain, X. Zhang, and Z. Xu. Rcfile: A fast and space-efficient data placement structure in mapreduce-based warehouse systems. In ICDE, pages 1199-1208, 2011.

[28] H. Herodotou and S. Babu. Profiling, what-if analysis, and cost-based optimization of mapreduce programs. PVLDB, 4(11):1111-1122, 2011.

[29] E. Jahani, M. J. Cafarella, and C. Re. Automatic optimization for mapreduce programs. PVLDB, 4(6):385-396, 2011.

[30] J. Jestes, F. Li, and K. Yi. Building wavelet histograms on large data in mapreduce. In PVLDB, pages 617-620, 2012.

[31] H. J. Karloff, S. Suri, and S. Vassilvitskii. A model of computation for mapreduce. In ${SODA}$ ,pages ${938} - {948},{2010}$ .

[32] N. Khoussainova, M. Balazinska, and D. Suciu. Perfxplain: Debugging mapreduce job performance. PVLDB, 5(7):598-609, 2012.

[33] L. Kolb, A. Thor, and E. Rahm. Load balancing for mapreduce-based entity resolution. In ${ICDE}$ ,pages ${618} - {629},{2012}$ .

[34] P. Koutris and D. Suciu. Parallel evaluation of conjunctive queries. In PODS, pages 223-234, 2011.

[35] H. T. Kung, F. Luccio, and F. P. Preparata. On finding the maxima of a set of vectors. JACM, 22(4):469-476, 1975.

[36] Y. Kwon, M. Balazinska, B. Howe, and J. A. Rolia. Skewtune: mitigating skew in mapreduce applications. In SIGMOD, pages 25-36, 2012.

[37] W. Lang and J. M. Patel. Energy management for mapreduce clusters. PVLDB, 3(1):129-139, 2010.

[38] N. Laptev, K. Zeng, and C. Zaniolo. Early accurate results for advanced analytics on mapreduce. PVLDB, 5(10):1028-1039, 2012.

[39] S. Lattanzi, B. Moseley, S. Suri, and S. Vassilvitskii. Filtering: a method for solving graph problems in mapreduce. In ${SPAA}$ ,pages 85-94, 2011.

[40] H. Lim, H. Herodotou, and S. Babu. Stubby: A transformation-based optimizer for mapreduce workflows. PVLDB, 5(11):1196-1207, 2012.

[41] Y. Lin, D. Agrawal, C. Chen, B. C. Ooi, and S. Wu. Llama: leveraging columnar storage for scalable join processing in the mapreduce framework. In SIGMOD, pages 961-972, 2011.

[42] W. Lu, Y. Shen, S. Chen, and B. C. Ooi. Efficient processing of k nearest neighbor joins using mapreduce. PVLDB, 5(10):1016-1027, 2012.

[43] S. Melnik, A. Gubarev, J. J. Long, G. Romer, S. Shivakumar, M. Tolton, and T. Vassilakis. Dremel: Interactive analysis of web-scale datasets. PVLDB, 3(1):330-339, 2010.

[44] A. Metwally and C. Faloutsos. V-smart-join: A scalable mapreduce framework for all-pair similarity joins of multisets and vectors. PVLDB, 5(8):704-715, 2012.

[45] G. D. F. Morales, A. Gionis, and M. Sozio. Social content matching in mapreduce. PVLDB, 4(7):460-469, 2011.

[46] K. Morton, M. Balazinska, and D. Grossman. Paratimer: a progress indicator for mapreduce dags. In SIGMOD, pages 507-518, 2010.

[47] T. Nykiel, M. Potamias, C. Mishra, G. Kollios, and N. Koudas. Mrshare: Sharing across multiple queries in mapreduce. PVLDB, 3(1):494-505, 2010.

[48] A. Okcan and M. Riedewald. Processing theta-joins using mapreduce. In ${SIGMOD}$ ,pages 949-960,2011.

[49] C. Olston, B. Reed, U. Srivastava, R. Kumar, and A. Tomkins. Pig latin: a not-so-foreign language for data processing. In SIGMOD, pages 1099-1110, 2008.

[50] O. O'Malley. Terabyte sort on apache hadoop. Technical report, Yahoo, 2008.

[51] B. Panda, J. Herbach, S. Basu, and R. J. Bayardo. Planet: Massively parallel learning of tree ensembles with mapreduce. PVLDB, 2(2):1426-1437, 2009.

[52] N. Pansare, V. R. Borkar, C. Jermaine, and T. Condie. Online aggregation for large mapreduce jobs. PVLDB, 4(11):1135-1145, 2011.

[53] A. Shinnar, D. Cunningham, B. Herta, and V. A. Saraswat. M3r: Increased performance for in-memory hadoop jobs. PVLDB, 5(12):1736-1747, 2012.

[54] S. Suri and S. Vassilvitskii. Counting triangles and the curse of the last reducer. In ${WWW}$ ,pages ${607} - {614},{2011}$ .

[55] A. Thusoo, J. S. Sarma, N. Jain, Z. Shao, P. Chakka, N. Zhang, S. Anthony, H. Liu, and R. Murthy. Hive - a petabyte scale data warehouse using hadoop. In ICDE, pages 996-1005, 2010.

[56] C. E. Tsourakakis, U. Kang, G. L. Miller, and C. Faloutsos. Doulion: counting triangles in massive graphs with a coin. In ${SIGKDD}$ ,pages 837-846, 2009.

[57] L. G. Valiant. A bridging model for parallel computation. Commun. ${ACM},{33}\left( 8\right)  : {103} - {111},{1990}$ .

[58] R. Vernica, A. Balmin, K. S. Beyer, and V. Ercegovac. Adaptive mapreduce using situation-aware mappers. In ${EDBT}$ ,pages 420-431, 2012.

[59] R. Vernica, M. J. Carey, and C. Li. Efficient parallel set-similarity joins using mapreduce. In SIGMOD, pages 495-506, 2010.

[60] G. Wang, M. A. V. Salles, B. Sowell, X. Wang, T. Cao, A. J. Demers, J. Gehrke, and W. M. White. Behavioral simulations in mapreduce. PVLDB, 3(1):952-963, 2010.

[61] B. Zhang, S. Zhou, and J. Guan. Adapting skyline computation to the mapreduce framework: Algorithms and experiments. In DASFAA Workshops, pages 403-414, 2011.

[62] X. Zhang, L. Chen, and M. Wang. Efficient multi-way theta-join processing using mapreduce. PVLDB, 5(11):1184-1195, 2012.