# Pregel: A System for Large-Scale Graph Processing

# Pregel：大规模图处理系统

Grzegorz Malewicz, Matthew H. Austern, Aart J. C. Bik, James C. Dehnert, Ilan Horn, Naty Leiser, and Grzegorz Czajkowski

格热戈日·马莱维奇（Grzegorz Malewicz）、马修·H·奥斯特恩（Matthew H. Austern）、阿特·J·C·比克（Aart J. C. Bik）、詹姆斯·C·德内特（James C. Dehnert）、伊兰·霍恩（Ilan Horn）、纳蒂·莱泽（Naty Leiser）和格热戈日·恰伊科夫斯基（Grzegorz Czajkowski）

Google, Inc.

谷歌公司

\{malewicz,austern,ajcbik,dehnert,ilan,naty,gczaj\}@google.com

\{malewicz,austern,ajcbik,dehnert,ilan,naty,gczaj\}@google.com

## Abstract

## 摘要

Many practical computing problems concern large graphs. Standard examples include the Web graph and various social networks. The scale of these graphs-in some cases billions of vertices, trillions of edges-poses challenges to their efficient processing. In this paper we present a computational model suitable for this task. Programs are expressed as a sequence of iterations, in each of which a vertex can receive messages sent in the previous iteration, send messages to other vertices, and modify its own state and that of its outgoing edges or mutate graph topology. This vertex-centric approach is flexible enough to express a broad set of algorithms. The model has been designed for efficient, scalable and fault-tolerant implementation on clusters of thousands of commodity computers, and its implied synchronic-ity makes reasoning about programs easier. Distribution-related details are hidden behind an abstract API. The result is a framework for processing large graphs that is expressive and easy to program.

许多实际的计算问题都与大型图有关。典型的例子包括网页图和各种社交网络。这些图的规模（在某些情况下，有数十亿个顶点、数万亿条边）给它们的高效处理带来了挑战。在本文中，我们提出了一种适合此任务的计算模型。程序被表示为一系列迭代，在每次迭代中，一个顶点可以接收上一次迭代中发送的消息，向其他顶点发送消息，并修改其自身状态以及其出边的状态，或者改变图的拓扑结构。这种以顶点为中心的方法足够灵活，可以表达广泛的算法。该模型旨在在数千台商用计算机组成的集群上实现高效、可扩展和容错的计算，并且其隐含的同步性使程序推理更加容易。与分布式相关的细节被隐藏在一个抽象的应用程序编程接口（API）后面。其结果是一个用于处理大型图的框架，该框架具有很强的表达能力且易于编程。

## Categories and Subject Descriptors

## 类别和主题描述符

D.1.3 [Programming Techniques]: Concurrent Programming-Distributed programming; D.2.13 [Software Engineering]: Reusable Software-Reusable libraries

D.1.3 [编程技术]：并发编程 - 分布式编程；D.2.13 [软件工程]：可复用软件 - 可复用库

## General Terms

## 通用术语

Design, Algorithms

设计、算法

## Keywords

## 关键词

Distributed computing, graph algorithms

分布式计算、图算法

## 1. INTRODUCTION

## 1. 引言

The Internet made the Web graph a popular object of analysis and research. Web 2.0 fueled interest in social networks. Other large graphs - for example induced by transportation routes, similarity of newspaper articles, paths of disease outbreaks, or citation relationships among published scientific work - have been processed for decades. Frequently applied algorithms include shortest paths computations, different flavors of clustering, and variations on the page rank theme. There are many other graph computing problems of practical value, e.g., minimum cut and connected components.

互联网使网页图成为分析和研究的热门对象。Web 2.0激发了人们对社交网络的兴趣。其他大型图（例如由交通路线、报纸文章的相似度、疾病爆发路径或已发表科学著作之间的引用关系所构成的图）已经被处理了数十年。常用的算法包括最短路径计算、不同类型的聚类以及网页排名主题的各种变体。还有许多其他具有实际价值的图计算问题，例如最小割和连通分量。

Efficient processing of large graphs is challenging. Graph algorithms often exhibit poor locality of memory access, very little work per vertex, and a changing degree of parallelism over the course of execution $\left\lbrack  {{31},{39}}\right\rbrack$ . Distribution over many machines exacerbates the locality issue, and increases the probability that a machine will fail during computation. Despite the ubiquity of large graphs and their commercial importance, we know of no scalable general-purpose system for implementing arbitrary graph algorithms over arbitrary graph representations in a large-scale distributed environment.

大型图的高效处理具有挑战性。图算法通常表现出较差的内存访问局部性，每个顶点的计算量非常小，并且在执行过程中并行度会发生变化$\left\lbrack  {{31},{39}}\right\rbrack$。在多台机器上进行分布式计算会加剧局部性问题，并增加计算过程中机器发生故障的概率。尽管大型图无处不在且具有商业重要性，但据我们所知，目前还没有一种可扩展的通用系统能够在大规模分布式环境中针对任意的图表示实现任意的图算法。

Implementing an algorithm to process a large graph typically means choosing among the following options:

实现一个处理大型图的算法通常意味着在以下选项中进行选择：

1. Crafting a custom distributed infrastructure, typically requiring a substantial implementation effort that must be repeated for each new algorithm or graph representation.

1. 构建自定义的分布式基础设施，通常需要大量的实现工作，并且针对每个新算法或图表示都必须重复进行。

2. Relying on an existing distributed computing platform, often ill-suited for graph processing. MapReduce [14], for example, is a very good fit for a wide array of large-scale computing problems. It is sometimes used to mine large graphs $\left\lbrack  {{11},{30}}\right\rbrack$ ,but this can lead to suboptimal performance and usability issues. The basic models for processing data have been extended to facilitate aggregation [41] and SQL-like queries [40, 47], but these extensions are usually not ideal for graph algorithms that often better fit a message passing model.

2. 依赖现有的分布式计算平台，而这些平台往往不适合图处理。例如，MapReduce [14] 非常适合处理各种大规模计算问题。它有时被用于挖掘大型图 $\left\lbrack  {{11},{30}}\right\rbrack$，但这可能会导致性能不佳和可用性问题。用于处理数据的基本模型已得到扩展，以方便进行聚合操作 [41] 和类似 SQL 的查询 [40, 47]，但这些扩展通常并不适合图算法，因为图算法更适合消息传递模型。

3. Using a single-computer graph algorithm library, such as BGL [43], LEDA [35], NetworkX [25], JDSL [20], Stanford GraphBase [29], or FGL [16], limiting the scale of problems that can be addressed.

3. 使用单计算机图算法库，如 BGL [43]、LEDA [35]、NetworkX [25]、JDSL [20]、斯坦福图基库（Stanford GraphBase） [29] 或 FGL [16]，这会限制可处理问题的规模。

4. Using an existing parallel graph system. The Parallel BGL [22] and CGMgraph [8] libraries address parallel graph algorithms, but do not address fault tolerance or other issues that are important for very large scale distributed systems.

4. 使用现有的并行图系统。并行 BGL（Parallel BGL） [22] 和 CGMgraph [8] 库可处理并行图算法，但未解决容错或其他对于超大规模分布式系统很重要的问题。

None of these alternatives fit our purposes. To address distributed processing of large scale graphs, we built a scalable and fault-tolerant platform with an API that is sufficiently flexible to express arbitrary graph algorithms. This paper describes the resulting system,called ${\mathrm{{Pregel}}}^{1}$ ,and reports our experience with it.

这些替代方案都不符合我们的需求。为了解决大规模图的分布式处理问题，我们构建了一个可扩展且容错的平台，其 API 足够灵活，可以表达任意的图算法。本文介绍了由此产生的系统，称为 ${\mathrm{{Pregel}}}^{1}$，并报告了我们使用该系统的经验。

---

<!-- Footnote -->

bear this notice and the full citation on the first page. To copy otherwise, to

在第一页保留此声明和完整引用。否则进行复制，

<!-- Footnote -->

---

The high-level organization of Pregel programs is inspired by Valiant's Bulk Synchronous Parallel model [45]. Pregel computations consist of a sequence of iterations,called ${su}$ - persteps. During a superstep the framework invokes a user-defined function for each vertex, conceptually in parallel. The function specifies behavior at a single vertex $V$ and a single superstep $S$ . It can read messages sent to $V$ in su-perstep $S - 1$ ,send messages to other vertices that will be received at superstep $S + 1$ ,and modify the state of $V$ and its outgoing edges. Messages are typically sent along outgoing edges, but a message may be sent to any vertex whose identifier is known.

Pregel 程序的高层组织受到了瓦利安特（Valiant）的整体同步并行（Bulk Synchronous Parallel）模型 [45] 的启发。Pregel 计算由一系列迭代组成，称为 ${su}$ - 超步（superstep）。在一个超步期间，框架会为每个顶点调用一个用户定义的函数，从概念上讲是并行执行的。该函数指定了单个顶点 $V$ 在单个超步 $S$ 中的行为。它可以读取在超步 $S - 1$ 发送给 $V$ 的消息，向其他顶点发送消息（这些消息将在超步 $S + 1$ 被接收），并修改 $V$ 及其出边的状态。消息通常沿着出边发送，但也可以发送给任何已知标识符的顶点。

The vertex-centric approach is reminiscent of MapReduce in that users focus on a local action, processing each item independently, and the system composes these actions to lift computation to a large dataset. By design the model is well suited for distributed implementations: it doesn't expose any mechanism for detecting order of execution within a superstep,and all communication is from superstep $S$ to superstep $S + 1$ .

以顶点为中心的方法让人联想到 MapReduce，因为用户专注于局部操作，独立处理每个项目，而系统将这些操作组合起来，将计算扩展到大型数据集。从设计上看，该模型非常适合分布式实现：它不暴露任何用于检测超步内执行顺序的机制，并且所有通信都是从超步 $S$ 到超步 $S + 1$。

The synchronicity of this model makes it easier to reason about program semantics when implementing algorithms, and ensures that Pregel programs are inherently free of deadlocks and data races common in asynchronous systems. In principle the performance of Pregel programs should be competitive with that of asynchronous systems given enough parallel slack $\left\lbrack  {{28},{34}}\right\rbrack$ . Because typical graph computations have many more vertices than machines, one should be able to balance the machine loads so that the synchronization between supersteps does not add excessive latency.

该模型的同步性使得在实现算法时更容易推理程序语义，并确保 Pregel 程序本质上不会出现异步系统中常见的死锁和数据竞争问题。原则上，在有足够的并行松弛度 $\left\lbrack  {{28},{34}}\right\rbrack$ 的情况下，Pregel 程序的性能应该与异步系统的性能相当。由于典型的图计算中的顶点数量远多于机器数量，应该能够平衡机器负载，使得超步之间的同步不会增加过多的延迟。

The rest of the paper is structured as follows. Section 2 describes the model. Section 3 describes its expression as a C++ API. Section 4 discusses implementation issues, including performance and fault tolerance. In Section 5 we present several applications of this model to graph algorithm problems, and in Section 6 we present performance results. Finally, we discuss related work and future directions.

本文的其余部分结构如下。第 2 节描述模型。第 3 节描述其作为 C++ API 的表达方式。第 4 节讨论实现问题，包括性能和容错。在第 5 节中，我们介绍该模型在图算法问题中的几个应用，在第 6 节中，我们展示性能结果。最后，我们讨论相关工作和未来的研究方向。

## 2. MODEL OF COMPUTATION

## 2. 计算模型

The input to a Pregel computation is a directed graph in which each vertex is uniquely identified by a string vertex identifier. Each vertex is associated with a modifiable, user defined value. The directed edges are associated with their source vertices, and each edge consists of a modifiable, user defined value and a target vertex identifier.

Pregel 计算的输入是一个有向图，其中每个顶点由一个字符串顶点标识符唯一标识。每个顶点都与一个可修改的用户定义值相关联。有向边与它们的源顶点相关联，每条边由一个可修改的用户定义值和一个目标顶点标识符组成。

A typical Pregel computation consists of input, when the graph is initialized, followed by a sequence of supersteps separated by global synchronization points until the algorithm terminates, and finishing with output.

典型的 Pregel 计算包括输入阶段（此时图被初始化），接着是一系列由全局同步点分隔的超步，直到算法终止，最后是输出阶段。

Within each superstep the vertices compute in parallel, each executing the same user-defined function that expresses the logic of a given algorithm. A vertex can modify its state or that of its outgoing edges, receive messages sent to it in the previous superstep, send messages to other vertices (to be received in the next superstep), or even mutate the topology of the graph. Edges are not first-class citizens in this model, having no associated computation.

在每个超步内，顶点并行计算，每个顶点都执行同一个用户定义的函数，该函数表达了给定算法的逻辑。一个顶点可以修改其自身状态或其出边的状态，接收上一个超步发送给它的消息，向其他顶点发送消息（将在下一个超步被接收），甚至可以改变图的拓扑结构。在这个模型中，边不是一等公民，没有关联的计算。

<!-- Media -->

<!-- figureText: Active Vote to halt Inactive Message received -->

<img src="https://cdn.noedgeai.com/0195c906-e8f1-7dd6-9e85-95dc221a0758_1.jpg?x=963&y=167&w=643&h=171&r=0"/>

Figure 1: Vertex State Machine

图 1：顶点状态机

<!-- Media -->

Algorithm termination is based on every vertex voting to halt. In superstep 0 , every vertex is in the active state; all active vertices participate in the computation of any given superstep. A vertex deactivates itself by voting to halt. This means that the vertex has no further work to do unless triggered externally, and the Pregel framework will not execute that vertex in subsequent supersteps unless it receives a message. If reactivated by a message, a vertex must explicitly deactivate itself again. The algorithm as a whole terminates when all vertices are simultaneously inactive and there are no messages in transit. This simple state machine is illustrated in Figure 1.

算法终止基于每个顶点投票停止。在第 0 超步中，每个顶点都处于活跃状态；所有活跃顶点都会参与任何给定超步的计算。顶点通过投票停止来使自身停用。这意味着该顶点除非受到外部触发，否则没有更多工作要做，并且 Pregel 框架在后续超步中不会执行该顶点，除非它收到消息。如果顶点因消息而重新激活，则必须再次显式地使自身停用。当所有顶点同时处于非活跃状态且没有消息在传输时，整个算法终止。这个简单的状态机如图 1 所示。

The output of a Pregel program is the set of values explicitly output by the vertices. It is often a directed graph isomorphic to the input, but this is not a necessary property of the system because vertices and edges can be added and removed during computation. A clustering algorithm, for example, might generate a small set of disconnected vertices selected from a large graph. A graph mining algorithm might simply output aggregated statistics mined from the graph.

Pregel 程序的输出是顶点显式输出的值的集合。它通常是一个与输入同构的有向图，但这不是该系统的必要属性，因为在计算过程中可以添加和删除顶点和边。例如，聚类算法可能会从一个大图中生成一小部分不相连的顶点。图挖掘算法可能只是输出从图中挖掘出的聚合统计信息。

Figure 2 illustrates these concepts using a simple example: given a strongly connected graph where each vertex contains a value, it propagates the largest value to every vertex. In each superstep, any vertex that has learned a larger value from its messages sends it to all its neighbors. When no further vertices change in a superstep, the algorithm terminates.

图 2 使用一个简单的示例说明了这些概念：给定一个强连通图，其中每个顶点都包含一个值，它将最大值传播到每个顶点。在每个超步中，任何从其消息中得知更大值的顶点都会将该值发送给其所有邻居。当在一个超步中没有更多顶点发生变化时，算法终止。

We chose a pure message passing model, omitting remote reads and other ways of emulating shared memory, for two reasons. First, message passing is sufficiently expressive that there is no need for remote reads. We have not found any graph algorithms for which message passing is insufficient. Second, this choice is better for performance. In a cluster environment, reading a value from a remote machine incurs high latency that can't easily be hidden. Our message passing model allows us to amortize latency by delivering messages asynchronously in batches.

我们选择纯消息传递模型，省略远程读取和其他模拟共享内存的方式，有两个原因。首先，消息传递具有足够的表达能力，因此不需要远程读取。我们还没有发现任何消息传递不足以实现的图算法。其次，这种选择对性能更有利。在集群环境中，从远程机器读取值会产生高延迟，而且这种延迟不容易被隐藏。我们的消息传递模型允许我们通过批量异步传递消息来分摊延迟。

Graph algorithms can be written as a series of chained MapReduce invocations $\left\lbrack  {{11},{30}}\right\rbrack$ . We chose a different model for reasons of usability and performance. Pregel keeps vertices and edges on the machine that performs computation, and uses network transfers only for messages. MapReduce, however, is essentially functional, so expressing a graph algorithm as a chained MapReduce requires passing the entire state of the graph from one stage to the next-in general requiring much more communication and associated serialization overhead. In addition, the need to coordinate the steps of a chained MapReduce adds programming complexity that is avoided by Pregel's iteration over supersteps.

图算法可以写成一系列链式的 MapReduce 调用 $\left\lbrack  {{11},{30}}\right\rbrack$。出于可用性和性能的原因，我们选择了不同的模型。Pregel 将顶点和边保留在执行计算的机器上，仅使用网络传输消息。然而，MapReduce 本质上是函数式的，因此将图算法表示为链式 MapReduce 需要将图的整个状态从一个阶段传递到下一个阶段 —— 通常需要更多的通信和相关的序列化开销。此外，协调链式 MapReduce 的步骤需要增加编程复杂性，而 Pregel 通过超步迭代避免了这一点。

---

<!-- Footnote -->

${}^{1}$ The name honors Leonhard Euler. The Bridges of Königs-berg, which inspired his famous theorem, spanned the Pregel river.

${}^{1}$ 这个名字是为了纪念莱昂哈德·欧拉（Leonhard Euler）。启发他著名定理的哥尼斯堡七桥（The Bridges of Königs - berg）横跨普雷格尔河（Pregel river）。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 3 6 2 Superstep 0 6 Superstep 1 6 Superstep 2 Superstep 3 6 6 2 6 6 6 6 6 -->

<img src="https://cdn.noedgeai.com/0195c906-e8f1-7dd6-9e85-95dc221a0758_2.jpg?x=227&y=167&w=564&h=465&r=0"/>

Figure 2: Maximum Value Example. Dotted lines are messages. Shaded vertices have voted to halt.

图 2：最大值示例。虚线表示消息。阴影顶点已投票停止。

<!-- Media -->

### 3.THE C++ API

### 3. C++ 应用程序编程接口

This section discusses the most important aspects of Pre-gel's C++ API, omitting relatively mechanical issues.

本节讨论 Pregel 的 C++ 应用程序编程接口（API）的最重要方面，省略相对机械的问题。

Writing a Pregel program involves subclassing the predefined Vertex class (see Figure 3). Its template arguments define three value types, associated with vertices, edges, and messages. Each vertex has an associated value of the specified type. This uniformity may seem restrictive, but users can manage it by using flexible types like protocol buffers [42]. The edge and message types behave similarly.

编写 Pregel 程序涉及对预定义的 Vertex 类进行子类化（见图 3）。其模板参数定义了三种值类型，分别与顶点、边和消息相关联。每个顶点都有一个指定类型的关联值。这种一致性可能看起来有局限性，但用户可以通过使用像协议缓冲区 [42] 这样的灵活类型来处理。边和消息类型的行为类似。

The user overrides the virtual Compute(   ) method, which will be executed at each active vertex in every superstep. Predefined Vertex methods allow Compute(   ) to query information about the current vertex and its edges, and to send messages to other vertices. Compute(   ) can inspect the value associated with its vertex via GetValue(   ) or modify it via MutableValue(   ). It can inspect and modify the values of out-edges using methods supplied by the out-edge iterator. These state updates are visible immediately. Since their visibility is confined to the modified vertex, there are no data races on concurrent value access from different vertices.

用户重写虚拟的 Compute( ) 方法，该方法将在每个超步中的每个活跃顶点处执行。预定义的 Vertex 方法允许 Compute( ) 查询有关当前顶点及其边的信息，并向其他顶点发送消息。Compute( ) 可以通过 GetValue( ) 检查与其顶点关联的值，或者通过 MutableValue( ) 修改该值。它可以使用出边迭代器提供的方法检查和修改出边的值。这些状态更新会立即生效。由于它们的可见性仅限于被修改的顶点，因此在不同顶点并发访问值时不会出现数据竞争。

The values associated with the vertex and its edges are the only per-vertex state that persists across supersteps. Limiting the graph state managed by the framework to a single value per vertex or edge simplifies the main computation cycle, graph distribution, and failure recovery.

与顶点及其边关联的值是唯一在超步之间持久存在的每个顶点的状态。将框架管理的图状态限制为每个顶点或边的单个值，简化了主要计算周期、图分布和故障恢复。

### 3.1 Message Passing

### 3.1 消息传递

Vertices communicate directly with one another by sending messages, each of which consists of a message value and the name of the destination vertex. The type of the message value is specified by the user as a template parameter of the Vertex class.

顶点通过发送消息直接相互通信，每条消息由一个消息值和目标顶点的名称组成。消息值的类型由用户作为 Vertex 类的模板参数指定。

A vertex can send any number of messages in a superstep. All messages sent to vertex $V$ in superstep $S$ are available, via an iterator,when $V$ ’s Compute(   ) method is called in superstep $S + 1$ . There is no guaranteed order of messages in the iterator, but it is guaranteed that messages will be delivered and that they will not be duplicated.

在一个超步（superstep）中，一个顶点可以发送任意数量的消息。当在超步 $S + 1$ 中调用顶点 $V$ 的 Compute() 方法时，通过一个迭代器可以获取在超步 $S$ 中发送给顶点 $V$ 的所有消息。迭代器中的消息顺序没有保证，但可以保证消息会被传递且不会重复。

A common usage pattern is for a vertex $V$ to iterate over its outgoing edges, sending a message to the destination vertex of each edge, as shown in the PageRank algorithm in Figure 4 (Section 5.1 below). However, dest_vertex need not be a neighbor of $V$ . A vertex could learn the identifier of a non-neighbor from a message received earlier, or vertex identifiers could be known implicitly. For example, the graph could be a clique, with well-known vertex identifiers ${V}_{1}$ through ${V}_{n}$ ,in which case there may be no need to even keep explicit edges in the graph.

一种常见的使用模式是，顶点 $V$ 遍历其所有出边（outgoing edges），并向每条边的目标顶点发送一条消息，如图 4（见下面的 5.1 节）中的 PageRank 算法所示。然而，目标顶点（dest_vertex）不必是 $V$ 的邻居。一个顶点可以从之前收到的消息中得知非邻居顶点的标识符，或者顶点标识符可以隐式已知。例如，图可以是一个完全图（clique），其顶点标识符从 ${V}_{1}$ 到 ${V}_{n}$ 是已知的，在这种情况下，甚至可能不需要在图中显式保留边。

<!-- Media -->

---

template <typename VertexValue,

template <typename VertexValue,

	typename EdgeValue,

	typename EdgeValue,

	typename MessageValue>

	typename MessageValue>

class Vertex \{

class Vertex {

public:

public:

virtual void Compute(MessageIterator* msgs) = 0 ;

virtual void Compute(MessageIterator* msgs) = 0 ;

const string& vertex_id(   ) const;

const string& vertex_id( ) const;

int64 superstep(   ) const;

int64 superstep( ) const;

const VertexValue& GetValue(   );

const VertexValue& GetValue( );

VertexValue* MutableValue(   );

VertexValue* MutableValue( );

OutEdgeIterator GetOutEdgeIterator(   );

OutEdgeIterator GetOutEdgeIterator( );

void SendMessageTo(const string& dest_vertex,

void SendMessageTo(const string& dest_vertex,

			const MessageValue& message);

			const MessageValue& message);

void VoteToHalt(   );

void VoteToHalt( );

---

\};

Figure 3: The Vertex API foundations.

图3：顶点API基础。

<!-- Media -->

When the destination vertex of any message does not exist, we execute user-defined handlers. A handler could, for example, create the missing vertex or remove the dangling edge from its source vertex.

当任何消息的目标顶点不存在时，我们会执行用户定义的处理程序。例如，处理程序可以创建缺失的顶点，或者从其源顶点移除悬空边。

### 3.2 Combiners

### 3.2 合并器

Sending a message, especially to a vertex on another machine, incurs some overhead. This can be reduced in some cases with help from the user. For example, suppose that Compute(   ) receives integer messages and that only the sum matters, as opposed to the individual values. In that case the system can combine several messages intended for a vertex $V$ into a single message containing their sum,reducing the number of messages that must be transmitted and buffered.

发送消息，尤其是发送到另一台机器上的顶点，会产生一些开销。在某些情况下，借助用户的帮助可以减少这种开销。例如，假设Compute( )接收整数消息，并且只有总和重要，而不是单个值。在这种情况下，系统可以将发往顶点$V$的多条消息合并为一条包含它们总和的消息，从而减少必须传输和缓冲的消息数量。

Combiners are not enabled by default, because there is no mechanical way to find a useful combining function that is consistent with the semantics of the user's Compute(   ) method. To enable this optimization the user subclasses the Combiner class, overriding a virtual Combine(   ) method. There are no guarantees about which (if any) messages are combined, the groupings presented to the combiner, or the order of combining, so combiners should only be enabled for commutative and associative operations.

合并器默认是禁用的，因为没有机械的方法可以找到一个与用户的Compute( )方法的语义一致的有用合并函数。要启用此优化，用户需要继承Combiner类，并重写虚拟的Combine( )方法。对于哪些（如果有的话）消息会被合并、呈现给合并器的分组，或者合并的顺序，没有任何保证，因此合并器只应针对可交换和可结合的操作启用。

For some algorithms, such as single-source shortest paths (Section 5.2), we have observed more than a fourfold reduction in message traffic by using combiners.

对于一些算法，如单源最短路径算法（5.2节），我们观察到使用合并器可以使消息流量减少四倍以上。

### 3.3 Aggregators

### 3.3 聚合器

Pregel aggregators are a mechanism for global communication, monitoring, and data. Each vertex can provide a value to an aggregator in superstep $S$ ,the system combines those values using a reduction operator, and the resulting value is made available to all vertices in superstep $S + 1$ . Pregel includes a number of predefined aggregators, such as min, max, or sum operations on various integer or string types.

Pregel聚合器是一种用于全局通信、监控和数据处理的机制。每个顶点可以在超步$S$中向聚合器提供一个值，系统使用归约运算符合并这些值，并且得到的值会在超步$S + 1$中提供给所有顶点。Pregel包含许多预定义的聚合器，例如对各种整数或字符串类型的最小值、最大值或求和操作。

Aggregators can be used for statistics. For instance, a sum aggregator applied to the out-degree of each vertex yields the total number of edges in the graph. More complex reduction operators can generate histograms of a statistic.

聚合器可用于统计。例如，对每个顶点的出度应用求和聚合器可以得到图中边的总数。更复杂的归约运算符可以生成统计数据的直方图。

Aggregators can also be used for global coordination. For instance, one branch of Compute(   ) can be executed for the supersteps until an and aggregator determines that all vertices satisfy some condition, and then another branch can be executed until termination. A min or max aggregator, applied to the vertex ID, can be used to select a vertex to play a distinguished role in an algorithm.

聚合器还可用于全局协调。例如，Compute( )的一个分支可以在超步中执行，直到与聚合器确定所有顶点都满足某个条件，然后可以执行另一个分支直到终止。应用于顶点ID的最小值或最大值聚合器可用于选择一个顶点在算法中扮演特殊角色。

To define a new aggregator, a user subclasses the predefined Aggregator class, and specifies how the aggregated value is initialized from the first input value and how multiple partially aggregated values are reduced to one. Aggregation operators should be commutative and associative.

要定义一个新的聚合器，用户需要继承预定义的Aggregator类，并指定如何从第一个输入值初始化聚合值，以及如何将多个部分聚合的值归约为一个。聚合运算符应该是可交换和可结合的。

By default an aggregator only reduces input values from a single superstep, but it is also possible to define a sticky aggregator that uses input values from all supersteps. This is useful, for example, for maintaining a global edge count that is adjusted only when edges are added or removed.

默认情况下，聚合器只归约来自单个超步的输入值，但也可以定义一个粘性聚合器，它使用来自所有超步的输入值。例如，这对于维护一个仅在添加或移除边时才调整的全局边计数很有用。

More advanced uses are possible. For example, an aggregator can be used to implement a distributed priority queue for the $\Delta$ -stepping shortest paths algorithm [37]. Each vertex is assigned to a priority bucket based on its tentative distance. In one superstep, the vertices contribute their indices to a min aggregator. The minimum is broadcast to all workers in the next superstep, and the vertices in the lowest-index bucket relax edges.

还可以有更高级的用法。例如，聚合器可用于为$\Delta$ -步最短路径算法[37]实现一个分布式优先队列。每个顶点根据其临时距离被分配到一个优先级桶中。在一个超步中，顶点将其索引贡献给一个最小值聚合器。在下一个超步中，最小值会广播给所有工作节点，并且最低索引桶中的顶点会松弛边。

### 3.4 Topology Mutations

### 3.4 拓扑变更

Some graph algorithms need to change the graph's topology. A clustering algorithm, for example, might replace each cluster with a single vertex, and a minimum spanning tree algorithm might remove all but the tree edges. Just as a user's Compute(   ) function can send messages, it can also issue requests to add or remove vertices or edges.

一些图算法需要改变图的拓扑结构。例如，聚类算法可能会用一个单一顶点替换每个聚类，而最小生成树算法可能会移除除树边之外的所有边。就像用户的Compute( )函数可以发送消息一样，它也可以发出添加或移除顶点或边的请求。

Multiple vertices may issue conflicting requests in the same superstep (e.g.,two requests to add a vertex $V$ ,with different initial values). We use two mechanisms to achieve determinism: partial ordering and handlers.

多个顶点可能在同一超步中发出冲突的请求（例如，两个添加顶点$V$的请求，具有不同的初始值）。我们使用两种机制来实现确定性：部分排序和处理程序。

As with messages, mutations become effective in the su-perstep after the requests were issued. Within that super-step removals are performed first, with edge removal before vertex removal, since removing a vertex implicitly removes all of its out-edges. Additions follow removals, with vertex addition before edge addition, and all mutations precede calls to Compute(   ). This partial ordering yields deterministic results for most conflicts.

与消息一样，突变（mutation）在发出请求后的超步（superstep）中生效。在该超步中，首先执行删除操作，先删除边再删除顶点，因为删除一个顶点会隐式删除其所有出边。删除操作之后是添加操作，先添加顶点再添加边，并且所有突变操作都在调用 Compute( ) 之前执行。这种部分排序为大多数冲突产生确定性的结果。

The remaining conflicts are resolved by user-defined handlers. If there are multiple requests to create the same vertex in the same superstep, then by default the system just picks one arbitrarily, but users with special needs may specify a better conflict resolution policy by defining an appropriate handler method in their Vertex subclass. The same handler mechanism is used to resolve conflicts caused by multiple vertex removal requests, or by multiple edge addition or removal requests. We delegate the resolution to handlers to keep the code of Compute(   ) simple, which limits the interaction between a handler and Compute(   ), but has not been an issue in practice.

其余的冲突由用户定义的处理程序解决。如果在同一个超步中有多个创建相同顶点的请求，那么默认情况下系统会任意选择一个，但有特殊需求的用户可以通过在其 Vertex 子类中定义一个合适的处理程序方法来指定更好的冲突解决策略。相同的处理程序机制用于解决由多个顶点删除请求、多个边添加或删除请求引起的冲突。我们将冲突解决委托给处理程序，以保持 Compute( ) 代码的简单性，这限制了处理程序和 Compute( ) 之间的交互，但在实践中并未成为问题。

Our coordination mechanism is lazy: global mutations do not require coordination until the point when they are applied. This design choice facilitates stream processing. The intuition is that conflicts involving modification of a vertex $V$ are handled by $V$ itself.

我们的协调机制是惰性的：全局突变在应用之前不需要进行协调。这种设计选择便于流处理。其原理是，涉及修改顶点 $V$ 的冲突由 $V$ 自身处理。

Pregel also supports purely local mutations, i.e., a vertex adding or removing its own outgoing edges or removing itself. Local mutations cannot introduce conflicts and making them immediately effective simplifies distributed programming by using an easier sequential programming semantics.

Pregel 还支持纯局部突变，即一个顶点添加或删除其自身的出边，或者删除自身。局部突变不会引入冲突，并且使它们立即生效通过使用更简单的顺序编程语义简化了分布式编程。

### 3.5 Input and output

### 3.5 输入和输出

There are many possible file formats for graphs, such as a text file, a set of vertices in a relational database, or rows in Bigtable [9]. To avoid imposing a specific choice of file format, Pregel decouples the task of interpreting an input file as a graph from the task of graph computation. Similarly, output can be generated in an arbitrary format and stored in the form most suitable for a given application. The Pregel library provides readers and writers for many common file formats, but users with unusual needs can write their own by subclassing the abstract base classes Reader and Writer.

图有许多可能的文件格式，例如文本文件、关系数据库中的一组顶点或 Bigtable [9] 中的行。为了避免强制规定特定的文件格式选择，Pregel 将将输入文件解释为图的任务与图计算任务解耦。类似地，输出可以以任意格式生成，并以最适合给定应用程序的形式存储。Pregel 库为许多常见的文件格式提供了读取器和写入器，但有特殊需求的用户可以通过继承抽象基类 Reader 和 Writer 来编写自己的读取器和写入器。

## 4. IMPLEMENTATION

## 4. 实现

Pregel was designed for the Google cluster architecture, which is described in detail in [3]. Each cluster consists of thousands of commodity PCs organized into racks with high intra-rack bandwidth. Clusters are interconnected but distributed geographically.

Pregel 是为谷歌集群架构设计的，该架构在 [3] 中有详细描述。每个集群由数千台商用 PC 组成，这些 PC 被组织成机架，机架内带宽很高。集群之间相互连接，但在地理上是分布式的。

Our applications typically execute on a cluster management system that schedules jobs to optimize resource allocation, sometimes killing instances or moving them to different machines. The system includes a name service, so that instances can be referred to by logical names independent of their current binding to a physical machine. Persistent data is stored as files on a distributed storage system, GFS [19], or in Bigtable [9], and temporary data such as buffered messages on local disk.

我们的应用程序通常在集群管理系统上执行，该系统调度作业以优化资源分配，有时会终止实例或将它们迁移到不同的机器上。该系统包括一个名称服务，因此实例可以通过逻辑名称引用，而与它们当前绑定的物理机器无关。持久数据以文件形式存储在分布式存储系统 GFS [19] 或 Bigtable [9] 中，临时数据（如缓冲消息）存储在本地磁盘上。

### 4.1 Basic architecture

### 4.1 基本架构

The Pregel library divides a graph into partitions, each consisting of a set of vertices and all of those vertices' outgoing edges. Assignment of a vertex to a partition depends solely on the vertex ID, which implies it is possible to know which partition a given vertex belongs to even if the vertex is owned by a different machine, or even if the vertex does not yet exist. The default partitioning function is just hash(ID) ${\;\operatorname{mod}\;N}$ ,where $N$ is the number of partitions,but users can replace it.

Pregel 库将图划分为多个分区，每个分区由一组顶点及其所有顶点的出边组成。一个顶点分配到哪个分区仅取决于顶点 ID，这意味着即使该顶点由不同的机器拥有，甚至该顶点尚不存在，也有可能知道给定顶点属于哪个分区。默认的分区函数是 hash(ID) ${\;\operatorname{mod}\;N}$ ，其中 $N$ 是分区的数量，但用户可以替换它。

The assignment of vertices to worker machines is the main place where distribution is not transparent in Pregel. Some applications work well with the default assignment, but some benefit from defining custom assignment functions to better exploit locality inherent in the graph. For example, a typical heuristic employed for the Web graph is to colocate vertices representing pages of the same site.

在 Pregel 中，顶点分配到工作机器的过程是分布式不透明性最明显的地方。一些应用程序使用默认分配方式效果很好，但有些应用程序通过定义自定义分配函数来更好地利用图中固有的局部性会受益。例如，对于 Web 图，一种典型的启发式方法是将代表同一网站页面的顶点放置在同一位置。

In the absence of faults, the execution of a Pregel program consists of several stages:

在没有故障的情况下，Pregel 程序的执行包括几个阶段：

1. Many copies of the user program begin executing on a cluster of machines. One of these copies acts as the master. It is not assigned any portion of the graph, but is responsible for coordinating worker activity. The workers use the cluster management system's name service to discover the master's location, and send registration messages to the master.

1. 用户程序的多个副本开始在一组机器集群上执行。其中一个副本充当主节点。它不分配图的任何部分，但负责协调工作节点的活动。工作节点使用集群管理系统的名称服务来发现主节点的位置，并向主节点发送注册消息。

2. The master determines how many partitions the graph will have, and assigns one or more partitions to each worker machine. The number may be controlled by the user. Having more than one partition per worker allows parallelism among the partitions and better load balancing, and will usually improve performance. Each worker is responsible for maintaining the state of its section of the graph, executing the user's Compute(   ) method on its vertices, and managing messages to and from other workers. Each worker is given the complete set of assignments for all workers.

2. 主节点确定图将有多少个分区，并将一个或多个分区分配给每个工作机器。分区数量可以由用户控制。每个工作节点有多个分区可以实现分区之间的并行性和更好的负载均衡，通常会提高性能。每个工作节点负责维护其负责的图部分的状态，在其顶点上执行用户的 Compute( ) 方法，并管理与其他工作节点之间的消息。每个工作节点都会获得所有工作节点的完整分配信息。

3. The master assigns a portion of the user's input to each worker. The input is treated as a set of records, each of which contains an arbitrary number of vertices and edges. The division of inputs is orthogonal to the partitioning of the graph itself, and is typically based on file boundaries. If a worker loads a vertex that belongs to that worker's section of the graph, the appropriate data structures (Section 4.3) are immediately updated. Otherwise the worker enqueues a message to the remote peer that owns the vertex. After the input has finished loading, all vertices are marked as active.

3. 主节点将用户输入的一部分分配给每个工作节点。输入被视为一组记录，每条记录包含任意数量的顶点和边。输入的划分与图本身的分区是正交的，通常基于文件边界。如果工作节点加载了属于该工作节点负责的图部分的顶点，则会立即更新相应的数据结构（第4.3节）。否则，工作节点会向拥有该顶点的远程对等节点入队一条消息。输入加载完成后，所有顶点都被标记为活跃状态。

4. The master instructs each worker to perform a super-step. The worker loops through its active vertices, using one thread for each partition. The worker calls Compute(   ) for each active vertex, delivering messages that were sent in the previous superstep. Messages are sent asynchronously, to enable overlapping of computation and communication and batching, but are delivered before the end of the superstep. When the worker is finished it responds to the master, telling the master how many vertices will be active in the next superstep.

4. 主节点指示每个工作节点执行一个超步。工作节点遍历其活跃顶点，为每个分区使用一个线程。工作节点为每个活跃顶点调用Compute( )方法，并传递上一个超步中发送的消息。消息是异步发送的，以实现计算和通信的重叠以及批量处理，但会在超步结束前送达。工作节点完成后会向主节点响应，告知主节点下一个超步中将有多少个顶点处于活跃状态。

This step is repeated as long as any vertices are active, or any messages are in transit.

只要有任何顶点处于活跃状态，或者有任何消息正在传输中，就会重复此步骤。

5. After the computation halts, the master may instruct each worker to save its portion of the graph.

5. 计算停止后，主节点可以指示每个工作节点保存其负责的图部分。

### 4.2 Fault tolerance

### 4.2 容错性

Fault tolerance is achieved through checkpointing. At the beginning of a superstep, the master instructs the workers to save the state of their partitions to persistent storage, including vertex values, edge values, and incoming messages; the master separately saves the aggregator values.

容错性通过检查点机制（checkpointing）实现。在每个超步（superstep）开始时，主节点（master）指示工作节点（worker）将其分区的状态保存到持久存储中，包括顶点值、边值和传入消息；主节点单独保存聚合器值。

Worker failures are detected using regular "ping" messages that the master issues to workers. If a worker does not receive a ping message after a specified interval, the worker process terminates. If the master does not hear back from a worker, the master marks that worker process as failed.

工作节点故障通过主节点定期向工作节点发送的“心跳”消息来检测。如果工作节点在指定时间间隔后未收到心跳消息，工作节点进程将终止。如果主节点未收到工作节点的响应，则将该工作节点进程标记为故障。

When one or more workers fail, the current state of the partitions assigned to these workers is lost. The master reassigns graph partitions to the currently available set of workers, and they all reload their partition state from the most recent available checkpoint at the beginning of a superstep $S$ . That checkpoint may be several supersteps earlier than the latest superstep ${S}^{\prime }$ completed by any partition before the failure, requiring that recovery repeat the missing su-persteps. We select checkpoint frequency based on a mean time to failure model [13], balancing checkpoint cost against expected recovery cost.

当一个或多个工作节点发生故障时，分配给这些工作节点的分区的当前状态将丢失。主节点将图分区重新分配给当前可用的工作节点集合，并且它们都在超步开始时从最近可用的检查点重新加载其分区状态 $S$。该检查点可能比故障发生前任何分区完成的最新超步 ${S}^{\prime }$ 早几个超步，这就需要恢复过程重复缺失的超步。我们根据平均故障间隔时间模型 [13] 选择检查点频率，在检查点成本和预期恢复成本之间进行平衡。

Confined recovery is under development to improve the cost and latency of recovery. In addition to the basic checkpoints, the workers also log outgoing messages from their assigned partitions during graph loading and supersteps. Recovery is then confined to the lost partitions, which are recovered from checkpoints. The system recomputes the missing supersteps up to ${S}^{\prime }$ using logged messages from healthy partitions and recalculated ones from recovering partitions.

受限恢复（Confined recovery）正在开发中，以提高恢复的成本和延迟。除了基本的检查点外，工作节点还会在图加载和超步期间记录其分配分区的传出消息。然后，恢复过程将仅限于丢失的分区，这些分区从检查点中恢复。系统使用健康分区记录的消息和恢复分区重新计算的消息，重新计算直到 ${S}^{\prime }$ 的缺失超步。

This approach saves compute resources during recovery by only recomputing lost partitions, and can improve the latency of recovery since each worker may be recovering fewer partitions. Saving the outgoing messages adds overhead, but a typical machine has adequate disk bandwidth to ensure that I/O does not become the bottleneck.

这种方法在恢复过程中仅重新计算丢失的分区，从而节省了计算资源，并且由于每个工作节点可能恢复的分区较少，因此可以提高恢复的延迟。保存传出消息会增加开销，但典型的机器具有足够的磁盘带宽，以确保 I/O 不会成为瓶颈。

Confined recovery requires the user algorithm to be deterministic, to avoid inconsistencies due to mixing saved messages from the original execution with new messages from the recovery. Randomized algorithms can be made deterministic by seeding a pseudorandom number generator deterministically based on the superstep and the partition Nondeterministic algorithms can disable confined recovery and fall back to the basic recovery mechanism.

受限恢复要求用户算法具有确定性，以避免由于将原始执行中保存的消息与恢复过程中的新消息混合而导致的不一致性。可以通过基于超步和分区确定性地播种伪随机数生成器，使随机算法具有确定性。非确定性算法可以禁用受限恢复，退回到基本恢复机制。

### 4.3 Worker implementation

### 4.3 工作节点实现

A worker machine maintains the state of its portion of the graph in memory. Conceptually this can be thought of as a map from vertex ID to the state of each vertex, where the state of each vertex consists of its current value, a list of its outgoing edges (the vertex ID for the edge's target, and the edge's current value), a queue containing incoming messages, and a flag specifying whether the vertex is active. When the worker performs a superstep it loops through all vertices and calls Compute(   ), passing it the current value, an iterator to the incoming messages, and an iterator to the outgoing edges. There is no access to incoming edges because each incoming edge is part of a list owned by the source vertex, in general on a different machine.

工作节点机器在内存中维护其负责的图部分的状态。从概念上讲，这可以看作是一个从顶点 ID 到每个顶点状态的映射，其中每个顶点的状态包括其当前值、其传出边列表（边的目标顶点 ID 和边的当前值）、包含传入消息的队列以及一个指定顶点是否活跃的标志。当工作节点执行一个超步时，它会遍历所有顶点并调用 Compute( ) 函数，将当前值、传入消息的迭代器和传出边的迭代器传递给它。由于每个传入边是源顶点所拥有列表的一部分，通常位于不同的机器上，因此无法访问传入边。

For performance reasons, the active vertex flags are stored separately from the incoming message queues. Furthermore, while only a single copy of the vertex and edge values exists, two copies of the active vertex flags and the incoming message queue exist: one for the current superstep and one for the next superstep. While a worker processes its vertices in superstep $S$ it is simultaneously,in another thread, receiving messages from other workers executing the same superstep. Since vertices receive messages that were sent in the previous superstep (see Section 2), messages for super-steps $S$ and $S + 1$ must be kept separate. Similarly,arrival of a message for a vertex $V$ means that $V$ will be active in the next superstep, not necessarily the current one.

出于性能原因，活跃顶点标志与传入消息队列分开存储。此外，虽然顶点和边的值只有一份副本，但活跃顶点标志和传入消息队列有两份副本：一份用于当前超步，一份用于下一个超步。当工作节点在超步 $S$ 中处理其顶点时，它会在另一个线程中同时接收来自执行相同超步的其他工作节点的消息。由于顶点接收的是上一个超步发送的消息（见第 2 节），因此必须将超步 $S$ 和 $S + 1$ 的消息分开保存。同样，顶点 $V$ 收到消息意味着 $V$ 将在下一个超步中活跃，而不一定是当前超步。

When Compute(   ) requests sending a message to another vertex, the worker process first determines whether the destination vertex is owned by a remote worker machine, or by the same worker that owns the sender. In the remote case the message is buffered for delivery to the destination worker. When the buffer sizes reach a threshold, the largest buffers are asynchronously flushed, delivering each to its destination worker as a single network message. In the local case an optimization is possible: the message is placed directly in the destination vertex's incoming message queue.

当 Compute( ) 函数请求向另一个顶点发送消息时，工作节点进程首先确定目标顶点是由远程工作节点机器拥有，还是由拥有发送者的同一工作节点拥有。在远程情况下，消息将被缓冲以便发送到目标工作节点。当缓冲区大小达到阈值时，最大的缓冲区将被异步刷新，作为单个网络消息发送到其目标工作节点。在本地情况下，可以进行优化：消息直接放入目标顶点的传入消息队列中。

If the user has provided a Combiner (Section 3.2), it is applied when messages are added to the outgoing message queue and when they are received at the incoming message queue. The latter does not reduce network usage, but does reduce the space needed to store messages.

如果用户提供了合并器（Combiner，见第 3.2 节），则在消息添加到传出消息队列时以及在传入消息队列中接收消息时应用该合并器。后者不会减少网络使用，但会减少存储消息所需的空间。

### 4.4 Master implementation

### 4.4 主节点实现

The master is primarily responsible for coordinating the activities of workers. Each worker is assigned a unique identifier at the time of its registration. The master maintains a list of all workers currently known to be alive, including the worker's unique identifier, its addressing information, and which portion of the graph it has been assigned. The size of the master's data structures is proportional to the number of partitions, not the number of vertices or edges, so a single master can coordinate computation for even a very large graph.

主节点主要负责协调工作节点的活动。每个工作节点在注册时会被分配一个唯一标识符。主节点维护着一份当前已知存活的所有工作节点的列表，包括工作节点的唯一标识符、其寻址信息以及它被分配处理的图的部分。主节点的数据结构大小与分区数量成正比，而非与顶点或边的数量成正比，因此单个主节点甚至可以协调非常大的图的计算。

Most master operations, including input, output, computation, and saving and resuming from checkpoints, are terminated at barriers: the master sends the same request to every worker that was known to be alive at the time the operation begins, and waits for a response from every worker. If any worker fails, the master enters recovery mode as described in section 4.2. If the barrier synchronization succeeds, the master proceeds to the next stage. In the case of a computation barrier, for example, the master increments the global superstep index and proceeds to the next super-step.

大多数主节点操作，包括输入、输出、计算以及从检查点保存和恢复，都在屏障处终止：主节点向操作开始时已知存活的每个工作节点发送相同的请求，并等待每个工作节点的响应。如果任何工作节点出现故障，主节点将进入如4.2节所述的恢复模式。如果屏障同步成功，主节点将进入下一阶段。例如，在计算屏障的情况下，主节点会增加全局超步索引并进入下一个超步。

The master also maintains statistics about the progress of computation and the state of the graph, such as the total size of the graph, a histogram of its distribution of out-degrees, the number of active vertices, the timing and message traffic of recent supersteps, and the values of all user-defined aggregators. To enable user monitoring, the master runs an HTTP server that displays this information.

主节点还会维护有关计算进度和图状态的统计信息，例如图的总大小、出度分布的直方图、活跃顶点的数量、最近超步的时间和消息流量，以及所有用户定义聚合器的值。为了便于用户监控，主节点运行一个HTTP服务器来显示这些信息。

### 4.5 Aggregators

### 4.5 聚合器

An aggregator (Section 3.3) computes a single global value by applying an aggregation function to a set of values that the user supplies. Each worker maintains a collection of aggregator instances, identified by a type name and instance name. When a worker executes a superstep for any partition of the graph, the worker combines all of the values supplied to an aggregator instance into a single local value: an aggregator that is partially reduced over all of the worker's vertices in the partition. At the end of the superstep workers form a tree to reduce partially reduced aggregators into global values and deliver them to the master. We use a tree-based reduction-rather than pipelining with a chain of workers-to parallelize the use of CPU during reduction. The master sends the global values to all workers at the beginning of the next superstep.

聚合器（第3.3节）通过对用户提供的一组值应用聚合函数来计算单个全局值。每个工作节点维护着一组聚合器实例，这些实例由类型名称和实例名称标识。当工作节点为图的任何分区执行一个超步时，工作节点会将提供给一个聚合器实例的所有值合并为一个单一的局部值：这是一个在分区中该工作节点的所有顶点上进行了部分归约的聚合器。在超步结束时，工作节点形成一棵树，将部分归约的聚合器归约为全局值并将其传递给主节点。我们使用基于树的归约方式，而非通过工作节点链进行流水线操作，以便在归约过程中并行使用CPU。主节点在下一个超步开始时将全局值发送给所有工作节点。

## 5. APPLICATIONS

## 5. 应用

This section presents four examples that are simplified versions of algorithms developed by Pregel users to solve real problems: Page Rank, Shortest Paths, Bipartite Matching, and a Semi-Clustering algorithm.

本节介绍四个示例，它们是Pregel用户为解决实际问题而开发的算法的简化版本：网页排名（Page Rank）、最短路径（Shortest Paths）、二分匹配（Bipartite Matching）和半聚类算法（Semi - Clustering algorithm）。

### 5.1 PageRank

### 5.1 网页排名（PageRank）

A Pregel implementation of a PageRank algorithm [7] is shown in Figure 4. The PageRankVertex class inherits from Vertex. Its vertex value type is double to store a tentative PageRank, and its message type is double to carry PageR-ank fractions, while the edge value type is void because edges do not store information. We assume that the graph is initialized so that in superstep 0 , the value of each vertex is 1 / NumVertices(   ). In each of the first 30 supersteps, each vertex sends along each outgoing edge its tentative

图4展示了网页排名算法 [7] 的Pregel实现。PageRankVertex类继承自Vertex。其顶点值类型为双精度浮点数，用于存储临时的网页排名值；其消息类型为双精度浮点数，用于携带网页排名分数；而边值类型为空，因为边不存储信息。我们假设图已初始化，使得在超步0中，每个顶点的值为1 / NumVertices( )。在前30个超步中的每一个超步里，每个顶点都会沿着每条出边发送其临时的

<!-- Media -->

---

class PageRankVertex

class PageRankVertex

		: public Vertex<double, void, double> \{

		: public Vertex<double, void, double> {

	public:

	public:

	virtual void Compute(MessageIterator* msgs) \{

	virtual void Compute(MessageIterator* msgs) {

		if (superstep(   ) >= 1) \{

		if (superstep( ) >= 1) {

			double sum = 0 ;

			double sum = 0 ;

			for (; !msgs->Done(   ); msgs->Next(   ))

			for (; !msgs->Done( ); msgs->Next( ))

				sum += msgs->Value(   );

				总和 += 消息集合的值(   );

			*MutableValue(   ) =

			*可变值(   ) =

					0.15 / NumVertices(   ) + 0.85 * sum;

					0.15 / 顶点数量(   ) + 0.85 * 总和;

		\}

		if (superstep(   ) < 30) \{

		if (超步(   ) < 30) {

			const int64 n = GetOutEdgeIterator(   ).size(   );

			const int64 n = 出边迭代器(   ).大小(   );

			SendMessageToAllNeighbors(GetValue(   ) / n);

			向所有邻居发送消息(当前值(   ) / n);

		\} else \{

		} else {

			VoteToHalt(   );

			投票停止(   );

		\}

	\}

\};

---

## Figure 4: PageRank implemented in Pregel.

## 图4：在Pregel中实现的PageRank算法。

<!-- Media -->

PageRank divided by the number of outgoing edges. Starting from superstep 1 , each vertex sums up the values arriving on messages into sum and sets its own tentative PageRank to ${0.15}/$ NumVertices $\left( \right)  + {0.85} \times$ sum. After reaching super-step 30 , no further messages are sent and each vertex votes to halt. In practice, a PageRank algorithm would run until convergence was achieved, and aggregators would be useful for detecting the convergence condition.

PageRank（网页排名）值除以出边数量。从超步1开始，每个顶点将消息中的值累加到总和中，并将其暂定的PageRank值设为 ${0.15}/$ 顶点数量 $\left( \right)  + {0.85} \times$ 总和。达到第30个超步后，不再发送消息，每个顶点投票停止。实际上，PageRank算法会一直运行直到收敛，聚合器有助于检测收敛条件。

### 5.2 Shortest Paths

### 5.2 最短路径

Shortest paths problems are among the best known problems in graph theory and arise in a wide variety of applications $\left\lbrack  {{10},{24}}\right\rbrack$ ,with several important variants. The single-source shortest paths problem requires finding a shortest path between a single source vertex and every other vertex in the graph. The $s - t$ shortest path problem requires finding a single shortest path between given vertices $s$ and $t$ ; it has obvious practical applications like driving directions and has received a great deal of attention. It is also relatively easy-solutions in typical graphs like road networks visit a tiny fraction of vertices, with Lumsdaine et al [31] observing visits to 80,000 vertices out of 32 million in one example. A third variant, all-pairs shortest paths, is impractical for large graphs because of its $O\left( {\left| V\right| }^{2}\right)$ storage requirements.

最短路径问题是图论中最著名的问题之一，在各种应用中都会出现 $\left\lbrack  {{10},{24}}\right\rbrack$ ，并且有几个重要的变体。单源最短路径问题要求找到图中一个源顶点到其他每个顶点的最短路径。 $s - t$ 最短路径问题要求找到给定顶点 $s$ 和 $t$ 之间的一条最短路径；它在实际应用中很明显，比如导航，并且受到了广泛关注。在典型的图（如道路网络）中，解决该问题相对容易，因为只需要访问一小部分顶点，例如Lumsdaine等人 [31] 观察到在一个包含3200万个顶点的图中只访问了80000个顶点。第三个变体，所有顶点对之间的最短路径问题，由于其 $O\left( {\left| V\right| }^{2}\right)$ 的存储要求，对于大型图来说是不切实际的。

For simplicity and conciseness, we focus here on the single-source variant that fits Pregel's target of large-scale graphs very well, but offers more interesting scaling data than the $s$ - $t$ shortest path problem. An implementation is shown in Figure 5.

为了简单和简洁，我们这里主要关注单源最短路径变体，它非常适合Pregel处理大规模图的目标，并且比 $s$ - $t$ 最短路径问题提供了更有趣的可扩展性数据。图5展示了其实现。

In this algorithm, we assume the value associated with each vertex is initialized to INF (a constant larger than any feasible distance in the graph from the source vertex). In each superstep, each vertex first receives, as messages from its neighbors, updated potential minimum distances from the source vertex. If the minimum of these updates is less than the value currently associated with the vertex, then this vertex updates its value and sends out potential updates to its neighbors, consisting of the weight of each outgoing edge added to the newly found minimum distance. In the first superstep, only the source vertex will update its value (from INF to zero) and send updates to its immediate neighbors. These neighbors in turn will update their values and send

在这个算法中，我们假设每个顶点关联的值初始化为INF（一个比图中从源顶点出发的任何可行距离都大的常量）。在每个超步中，每个顶点首先从其邻居那里接收消息，这些消息包含从源顶点出发的更新后的潜在最小距离。如果这些更新值中的最小值小于当前顶点关联的值，那么该顶点更新其值，并向其邻居发送潜在的更新消息，这些消息由每条出边的权重加上新找到的最小距离组成。在第一个超步中，只有源顶点会更新其值（从INF更新为零），并向其直接邻居发送更新消息。这些邻居反过来会更新它们的值并发送

---

class ShortestPathVertex

类 最短路径顶点

			: public Vertex<int, int, int> \{

			: 公共继承 顶点<int, int, int> {

	void Compute(MessageIterator* msgs) \{

	void Compute(MessageIterator* msgs) {

			int mindist = IsSource(vertex_id(   )) ? 0 : INF;

			int mindist = IsSource(vertex_id( )) ? 0 : INF;（如果当前顶点是源顶点，则最小距离为0，否则为无穷大（INF））

			for (; !msgs->Done(   ); msgs->Next(   ))

			for (; !msgs->Done( ); msgs->Next( ))（只要消息迭代器未完成，就继续迭代下一条消息）

				mindist = min(mindist, msgs->Value(   ));

				mindist = min(mindist, msgs->Value( ));（取当前最小距离和消息值中的较小值作为新的最小距离）

			if (mindist < GetValue(   )) \{

			if (mindist < GetValue( )) {（如果最小距离小于当前顶点的值）

				*MutableValue(   ) = mindist;

				*MutableValue( ) = mindist;（将最小距离赋值给当前顶点的可变值）

				OutEdgeIterator iter = GetOutEdgeIterator(   );

				OutEdgeIterator iter = GetOutEdgeIterator( );（获取当前顶点的出边迭代器）

				for (; !iter.Done(   ); iter.Next(   ))

				for (; !iter.Done( ); iter.Next( ))（只要出边迭代器未完成，就继续迭代下一条出边）

						SendMessageTo(iter.Target(   ),

						SendMessageTo(iter.Target( ),（向出边的目标顶点发送消息）

															mindist + iter.GetValue(   ));

															mindist + iter.GetValue( ));（消息内容为最小距离加上出边的值）

			\}

			VoteToHalt(   );

			VoteToHalt( );（投票停止计算）

	\}

\};

---

Figure 5: Single-source shortest paths. messages, resulting in a wavefront of updates through the graph. The algorithm terminates when no more updates occur, after which the value associated with each vertex denotes the minimum distance from the source vertex to that vertex. (The value INF denotes that the vertex cannot be reached at all.) Termination is guaranteed if all edge weights are non-negative.

图5：单源最短路径。消息在图中传播，形成更新的波前。当不再有更新发生时，算法终止，此后每个顶点关联的值表示从源顶点到该顶点的最小距离。（值INF表示该顶点根本无法到达。）如果所有边的权重均为非负，则可保证算法终止。

<!-- Media -->

---

class MinIntCombiner : public Combiner<int> \{

class MinIntCombiner : public Combiner<int> {（定义一个名为MinIntCombiner的类，继承自Combiner<int>）

	virtual void Combine(MessageIterator* msgs) \{

	virtual void Combine(MessageIterator* msgs) {（虚函数，用于合并消息）

			int mindist = INF;

			int mindist = INF;（初始化最小距离为无穷大（INF））

			for (; !msgs->Done(   ); msgs->Next(   ))

			for (; !msgs->Done( ); msgs->Next( ))（只要消息迭代器未完成，就继续迭代下一条消息）

				mindist = min(mindist, msgs->Value(   ));

				最小距离 = 取最小距离和消息值中的较小值;

			Output ("combined_source", mindist);

			输出("组合源", 最小距离);

	\}

\};

---

Figure 6: Combiner that takes minimum of message values.

图6：取消息值最小值的组合器。

<!-- Media -->

Messages in this algorithm consist of potential shorter distances. Since the receiving vertex is ultimately only interested in the minimum, this algorithm is amenable to optimization using a combiner (Section 3.2). The combiner shown in Figure 6 greatly reduces the amount of data sent between workers, as well as the amount of data buffered prior to executing the next superstep. While the code in Figure 5 only computes distances, modifying it to compute the shortest paths tree as well is quite straightforward.

此算法中的消息包含可能的更短距离。由于接收顶点最终只对最小值感兴趣，因此该算法适合使用组合器进行优化（第3.2节）。图6所示的组合器大大减少了工作节点之间传输的数据量，以及执行下一个超步之前缓冲的数据量。虽然图5中的代码仅计算距离，但对其进行修改以同时计算最短路径树也相当简单。

This algorithm may perform many more comparisons than sequential counterparts such as Dijkstra or Bellman-Ford [5, ${15},{17},{24}\rbrack$ ,but it is able to solve the shortest paths problem at a scale that is infeasible with any single-machine implementation. More advanced parallel algorithms exist, e.g., Thorup [44] or the $\Delta$ -stepping method [37],and have been used as the basis for special-purpose parallel shortest paths implementations $\left\lbrack  {{12},{32}}\right\rbrack$ . Such advanced algorithms can also be expressed in the Pregel framework. The simplicity of the implementation in Figure 5, however, together with the already acceptable performance (see Section 6), may appeal to users who can't do extensive tuning or customization.

与Dijkstra或Bellman - Ford等顺序算法相比[5, ${15},{17},{24}\rbrack$，该算法可能会进行更多的比较，但它能够解决单台机器实现无法处理规模的最短路径问题。存在更高级的并行算法，例如Thorup [44]或$\Delta$ - 步进方法[37]，并且已被用作专用并行最短路径实现的基础$\left\lbrack  {{12},{32}}\right\rbrack$。这种高级算法也可以在Pregel框架中表达。然而，图5中实现的简单性，再加上已经可以接受的性能（见第6节），可能会吸引那些无法进行大量调优或定制的用户。

### 5.3 Bipartite Matching

### 5.3 二分图匹配

The input to a bipartite matching algorithm consists of two distinct sets of vertices with edges only between the sets, and the output is a subset of edges with no common endpoints. A maximal matching is one to which no additional edge can be added without sharing an endpoint. We implemented a randomized maximal matching algorithm [1] and a maximum-weight bipartite matching algorithm [4]; we describe the former here.

二分图匹配算法的输入由两个不同的顶点集组成，边仅存在于这两个集合之间，输出是没有公共端点的边的子集。最大匹配是指在不共享端点的情况下不能再添加额外边的匹配。我们实现了一种随机最大匹配算法[1]和一种最大权重二分图匹配算法[4]；我们在此描述前者。

In the Pregel implementation of this algorithm the vertex value is a tuple of two values: a flag indicating which set the vertex is in(LorR),and the name of its matched vertex once known. The edge value has type void (edges carry no information), and the messages are boolean. The algorithm proceeds in cycles of four phases, where the phase index is just the superstep index ${\;\operatorname{mod}\;4}$ ,using a three-way handshake.

在该算法的Pregel实现中，顶点值是一个包含两个值的元组：一个标志，指示顶点所在的集合（L或R），以及一旦确定的匹配顶点的名称。边值的类型为void（边不携带信息），消息为布尔类型。该算法以四个阶段为一个周期进行，阶段索引就是超步索引${\;\operatorname{mod}\;4}$，使用三方握手协议。

In phase 0 of a cycle, each left vertex not yet matched sends a message to each of its neighbors to request a match, and then unconditionally votes to halt. If it sent no messages (because it is already matched, or has no outgoing edges), or if all the message recipients are already matched, it will never be reactivated. Otherwise, it will receive a response in two supersteps and reactivate.

在一个周期的第0阶段，每个尚未匹配的左顶点向其每个邻居发送消息以请求匹配，然后无条件投票停止。如果它没有发送消息（因为它已经匹配，或者没有出边），或者如果所有消息接收者都已经匹配，它将不会被重新激活。否则，它将在两个超步后收到响应并重新激活。

In phase 1 of a cycle, each right vertex not yet matched randomly chooses one of the messages it receives, sends a message granting that request, and sends messages to other requestors denying it. Then it unconditionally votes to halt.

在一个周期的第1阶段，每个尚未匹配的右顶点随机选择它收到的一条消息，发送一条消息批准该请求，并向其他请求者发送消息拒绝请求。然后它无条件投票停止。

In phase 2 of a cycle, each left vertex not yet matched chooses one of the grants it receives and sends an acceptance message. Left vertices that are already matched will never execute this phase, since they will not have sent a message in phase 0 .

在一个周期的第2阶段，每个尚未匹配的左顶点选择它收到的一个批准消息并发送一个接受消息。已经匹配的左顶点永远不会执行此阶段，因为它们在第0阶段不会发送消息。

Finally, in phase 3, an unmatched right vertex receives at most one acceptance message. It notes the matched node and unconditionally votes to halt - it has nothing further to do.

最后，在第3阶段，一个未匹配的右顶点最多接收一条接受消息。它记录匹配的节点并无条件投票停止——它没有其他事情可做。

### 5.4 Semi-Clustering

### 5.4 半聚类

Pregel has been used for several different versions of clustering. One version, semi-clustering, arises in social graphs.

Pregel已用于几种不同版本的聚类。其中一种版本，半聚类，出现在社交图中。

Vertices in a social graph typically represent people, and edges represent connections between them. Edges may be based on explicit actions (e.g., adding a friend in a social networking site), or may be inferred from people's behavior (e.g., email conversations or co-publication). Edges may have weights, to represent the interactions' frequency or strength.

社交图中的顶点通常代表人，边代表他们之间的联系。边可能基于明确的行为（例如，在社交网络站点中添加朋友），或者可能从人们的行为中推断出来（例如，电子邮件对话或共同发表文章）。边可能有权重，以表示交互的频率或强度。

A semi-cluster in a social graph is a group of people who interact frequently with each other and less frequently with others. What distinguishes it from ordinary clustering is that a vertex may belong to more than one semi-cluster.

社交图中的半聚类是一组彼此频繁交互且与其他人交互较少的人。它与普通聚类的区别在于，一个顶点可能属于多个半聚类。

This section describes a parallel greedy semi-clustering algorithm. Its input is a weighted, undirected graph (represented in Pregel by constructing each edge twice, once in each direction) and its output is at most ${C}_{\max }$ semi-clusters, each containing at most ${V}_{\max }$ vertices,where ${C}_{\max }$ and ${V}_{\max }$ are user-specified parameters.

本节描述了一种并行贪心半聚类算法。其输入是一个加权无向图（在Pregel中通过将每条边构建两次来表示，每个方向各一次），其输出最多为${C}_{\max }$个半聚类，每个半聚类最多包含${V}_{\max }$个顶点，其中${C}_{\max }$和${V}_{\max }$是用户指定的参数。

A semi-cluster $c$ is assigned a score,

为半聚类$c$分配一个得分，

$$
{S}_{c} = \frac{{I}_{c} - {f}_{B}{B}_{c}}{{V}_{c}\left( {{V}_{c} - 1}\right) /2}, \tag{1}
$$

where ${I}_{c}$ is the sum of the weights of all internal edges, ${B}_{c}$ is the sum of the weights of all boundary edges (i.e., edges connecting a vertex in the semi-cluster to one outside it), ${V}_{c}$ is the number of vertices in the semi-cluster,and ${f}_{B}$ ,the boundary edge score factor, is a user-specified parameter, usually between 0 and 1 . The score is normalized, i.e., divided by the number of edges in a clique of size ${V}_{c}$ ,so that large clusters do not receive artificially high scores.

其中${I}_{c}$是所有内部边的权重之和，${B}_{c}$是所有边界边（即连接半聚类内顶点与半聚类外顶点的边）的权重之和，${V}_{c}$是半聚类中的顶点数量，${f}_{B}$（边界边得分因子）是用户指定的参数，通常介于0和1之间。该得分经过归一化处理，即除以大小为${V}_{c}$的完全图中的边数，这样大的聚类就不会获得人为的高分。

Each vertex $V$ maintains a list containing at most ${C}_{\max }$ semi-clusters,sorted by score. In superstep ${0V}$ enters itself in that list as a semi-cluster of size 1 and score 1 , and publishes itself to all of its neighbors. In subsequent supersteps:

每个顶点$V$维护一个列表，该列表最多包含${C}_{\max }$个半聚类，并按得分排序。在超步${0V}$中，顶点$V$将自身作为大小为1、得分也为1的半聚类加入该列表，并将自身发布给所有邻居。在后续的超步中：

- Vertex $V$ iterates over the semi-clusters ${c}_{1},\ldots ,{c}_{k}$ sent to it on the previous superstep. If a semi-cluster $c$ does not already contain $V$ ,and ${V}_{c} < {M}_{\max }$ ,then $V$ is added to $c$ to form ${c}^{\prime }$ .

- 顶点$V$遍历上一个超步发送给它的半聚类${c}_{1},\ldots ,{c}_{k}$。如果半聚类$c$尚未包含$V$，并且${V}_{c} < {M}_{\max }$，则将$V$添加到$c$中以形成${c}^{\prime }$。

- The semi-clusters ${c}_{1},\ldots ,{c}_{k},{c}_{1}^{\prime },\ldots ,{c}_{k}^{\prime }$ are sorted by their scores,and the best ones are sent to $V$ ’s neighbors.

- 半聚类${c}_{1},\ldots ,{c}_{k},{c}_{1}^{\prime },\ldots ,{c}_{k}^{\prime }$按其得分排序，并将得分最高的半聚类发送给$V$的邻居。

- Vertex $V$ updates its list of semi-clusters with the semi-clusters from ${c}_{1},\ldots ,{c}_{k},{c}_{1}^{\prime },\ldots ,{c}_{k}^{\prime }$ that contain $V$ .

- 顶点$V$用来自${c}_{1},\ldots ,{c}_{k},{c}_{1}^{\prime },\ldots ,{c}_{k}^{\prime }$且包含$V$的半聚类更新其半聚类列表。

The algorithm terminates either when the semi-clusters stop changing or (to improve performance) when the number of supersteps reaches a user-specified limit. At that point the list of best semi-cluster candidates for each vertex may be aggregated into a global list of best semi-clusters.

当半聚类不再变化时，或者（为了提高性能）当超步的数量达到用户指定的限制时，算法终止。此时，每个顶点的最佳半聚类候选列表可以聚合为一个全局的最佳半聚类列表。

## 6. EXPERIMENTS

## 6. 实验

We conducted various experiments with the single-source shortest paths (SSSP) implementation of Section 5.2 on a cluster of 300 multicore commodity PCs. We report run-times for binary trees (to study scaling properties) and lognormal random graphs (to study the performance in a more realistic setting) using various graph sizes with the weights of all edges implicitly set to 1 .

我们在由300台多核商用PC组成的集群上，对第5.2节中的单源最短路径（SSSP）实现进行了各种实验。我们报告了二叉树（用于研究可扩展性）和对数正态随机图（用于研究更现实场景下的性能）的运行时间，使用了各种图大小，并且所有边的权重都隐式设置为1。

The time for initializing the cluster, generating the test graphs in-memory, and verifying results is not included in the measurements. Since all experiments could run in a relatively short time, failure probability was low, and checkpointing was disabled.

测量中不包括初始化集群、在内存中生成测试图以及验证结果的时间。由于所有实验都可以在相对较短的时间内运行，失败概率较低，因此禁用了检查点机制。

As an indication of how Pregel scales with worker tasks, Figure 7 shows shortest paths runtimes for a binary tree with a billion vertices (and, thus, a billion minus one edges) when the number of Pregel workers varies from 50 to 800 . The drop from 174 to 17.3 seconds using 16 times as many workers represents a speedup of about 10 .

为了说明Pregel如何随工作任务进行扩展，图7展示了一个具有十亿个顶点（因此有十亿减一条边）的二叉树的最短路径运行时间，此时Pregel工作节点的数量从50个变化到800个。使用16倍数量的工作节点时，运行时间从174秒降至17.3秒，这表示加速比约为10。

To show how Pregel scales with graph size, Figure 8 presents shortest paths runtimes for binary trees varying in size from a billion to 50 billion vertices, now using a fixed number of 800 worker tasks scheduled on 300 multicore machines. Here the increase from 17.3 to 702 seconds demonstrates that for graphs with a low average outdegree the runtime increases linearly in the graph size.

为了展示Pregel如何随图的大小进行扩展，图8展示了大小从十亿个顶点到500亿个顶点变化的二叉树的最短路径运行时间，现在使用固定数量的800个工作任务，这些任务调度在300台多核机器上。这里从17.3秒增加到702秒表明，对于平均出度较低的图，运行时间随图的大小线性增加。

<!-- Media -->

<!-- figureText: 180 400 500 600 700 800 Number of worker tasks 160 Runtime (seconds) 140 100 80 40 20 100 200 300 -->

<img src="https://cdn.noedgeai.com/0195c906-e8f1-7dd6-9e85-95dc221a0758_7.jpg?x=249&y=1621&w=535&h=359&r=0"/>

Figure 7: SSSP-1 billion vertex binary tree: varying number of worker tasks scheduled on 300 multicore machines

图7：单源最短路径 - 十亿个顶点的二叉树：在300台多核机器上调度的不同数量的工作任务

<!-- figureText: 800 25G 30G 35G 40G 45G 50G Number of vertices 700 Runtime (seconds) 600 500 400 300 200 100 5G 10G 15G 20G -->

<img src="https://cdn.noedgeai.com/0195c906-e8f1-7dd6-9e85-95dc221a0758_7.jpg?x=996&y=168&w=584&h=364&r=0"/>

Figure 8: SSSP-binary trees: varying graph sizes on 800 worker tasks scheduled on 300 multicore machines

图8：单源最短路径 - 二叉树：在300台多核机器上调度的800个工作任务下不同的图大小

<!-- Media -->

Although the previous experiments give an indication of how Pregel scales in workers and graph size, binary trees are obviously not representative of graphs encountered in practice. Therefore, we also conducted experiments with random graphs that use a log-normal distribution of outdegrees,

尽管之前的实验表明了Pregel在工作节点和图大小方面的扩展性，但二叉树显然不能代表实际中遇到的图。因此，我们还对使用出度对数正态分布的随机图进行了实验，

$$
p\left( d\right)  = \frac{1}{\sqrt{2\pi }{\sigma d}}{e}^{-{\left( \ln d - \mu \right) }^{2}/2{\sigma }^{2}} \tag{2}
$$

with $\mu  = 4$ and $\sigma  = {1.3}$ ,for which the mean outdegree is 127.1. Such a distribution resembles many real-world large-scale graphs, such as the web graph or social networks, where most vertices have a relatively small degree but some outliers are much larger-a hundred thousand or more. Figure 9 shows shortest paths runtimes for such graphs varying in size from 10 million to a billion vertices (and thus over 127 billion edges), again with 800 worker tasks scheduled on 300 multicore machines. Running shortest paths for the largest graph took a little over 10 minutes.

对于$\mu  = 4$和$\sigma  = {1.3}$，其平均出度为127.1。这种分布与许多现实世界中的大规模图类似，例如网页图或社交网络，其中大多数顶点的度相对较小，但有些离群点的度要大得多——达到十万甚至更多。图9展示了此类图在顶点数量从1000万到10亿（因此边数超过1270亿）变化时的最短路径运行时间，同样是在300台多核机器上调度800个工作任务。对最大的图运行最短路径算法耗时略超过10分钟。

In all experiments the graph was partitioned among workers using the default partitioning function based on a random hash; a topology-aware partitioning function would give better performance. Also, a naïve parallel shortest paths algorithm was used; here too a more advanced algorithm would perform better. Therefore, the results of the experiments in this section should not be interpreted as the best possible runtime of shortest paths using Pregel. Instead, the results are meant to show that satisfactory performance can be obtained with relatively little coding effort. In fact, our results for one billion vertices and edges are comparable to the $\Delta$ -stepping results from Parallel BGL [31] mentioned in the next section for a cluster of 112 processors on a graph of 256 million vertices and one billion edges, and Pregel scales better beyond that size.

在所有实验中，图使用基于随机哈希的默认分区函数在工作节点之间进行分区；一个感知拓扑结构的分区函数会带来更好的性能。此外，使用了一种简单的并行最短路径算法；在这里，更先进的算法也会表现得更好。因此，本节实验的结果不应被解读为使用Pregel算法计算最短路径的最佳可能运行时间。相反，这些结果旨在表明，只需相对较少的编码工作就能获得令人满意的性能。事实上，我们在处理10亿个顶点和边时的结果，与下一节提到的并行BGL [31]在拥有2.56亿个顶点和10亿条边的图上使用112个处理器的集群所得到的$\Delta$步结果相当，并且Pregel在更大规模上的扩展性更好。

<!-- Media -->

<!-- figureText: 800 Number of vertices 700 Runtime (seconds) 600 500 400 300 200 100 -->

<img src="https://cdn.noedgeai.com/0195c906-e8f1-7dd6-9e85-95dc221a0758_7.jpg?x=988&y=1589&w=597&h=359&r=0"/>

Figure 9: SSSP-log-normal random graphs, mean out-degree 127.1 (thus over 127 billion edges in the largest case): varying graph sizes on 800 worker tasks scheduled on 300 multicore machines

图9：单源最短路径（SSSP）——对数正态随机图，平均出度为127.1（因此在最大情况下边数超过1270亿）：在300台多核机器上调度800个工作任务时不同图规模的情况

<!-- Media -->

## 7. RELATED WORK

## 7. 相关工作

Pregel is a distributed programming framework, focused on providing users with a natural API for programming graph algorithms while managing the details of distribution invisibly, including messaging and fault tolerance. It is similar in concept to MapReduce [14], but with a natural graph API and much more efficient support for iterative computations over the graph. This graph focus also distinguishes it from other frameworks that hide distribution details such as Sawzall [41], Pig Latin [40], and Dryad [27, 47]. Pregel is also different because it implements a stateful model where long-lived processes compute, communicate, and modify local state, rather than a dataflow model where any process computes solely on input data and produces output data input by other processes.

Pregel是一个分布式编程框架，专注于为用户提供用于编写图算法的自然API，同时以不可见的方式管理分布式细节，包括消息传递和容错。它在概念上与MapReduce [14] 类似，但具有自然的图API，并且对图的迭代计算提供了更高效的支持。这种对图的关注也使其与其他隐藏分布式细节的框架（如Sawzall [41]、Pig Latin [40] 和Dryad [27, 47]）区分开来。Pregel的不同之处还在于，它实现了一种有状态模型，其中长期存在的进程进行计算、通信和修改本地状态，而不是一种数据流模型，在数据流模型中，任何进程仅对输入数据进行计算并产生由其他进程输入的输出数据。

Pregel was inspired by the Bulk Synchronous Parallel model [45], which provides its synchronous superstep model of computation and communication. There have been a number of general BSP library implementations, for example the Oxford BSP Library [38], Green BSP library [21], BSPlib [26] and Paderborn University BSP library [6]. They vary in the set of communication primitives provided, and in how they deal with distribution issues such as reliability (machine failure), load balancing, and synchronization. To our knowledge, the scalability and fault-tolerance of BSP implementations has not been evaluated beyond several dozen machines, and none of them provides a graph-specific API.

Pregel的灵感来自批量同步并行（Bulk Synchronous Parallel，BSP）模型 [45]，该模型为其提供了计算和通信的同步超步模型。已经有许多通用的BSP库实现，例如牛津BSP库 [38]、绿色BSP库 [21]、BSPlib [26] 和帕德博恩大学BSP库 [6]。它们在提供的通信原语集以及处理诸如可靠性（机器故障）、负载平衡和同步等分布式问题的方式上有所不同。据我们所知，BSP实现的可扩展性和容错性尚未在几十台以上的机器上进行评估，并且它们都没有提供特定于图的API。

The closest matches to Pregel are the Parallel Boost Graph Library and CGMgraph. The Parallel BGL [22, 23] specifies several key generic concepts for defining distributed graphs, provides implementations based on MPI [18], and implements a number of algorithms based on them. It attempts to maintain compatibility with the (sequential) BGL [43] to facilitate porting algorithms. It implements property maps to hold information associated with vertices and edges in the graph, using ghost cells to hold values associated with remote components. This can lead to scaling problems if reference to many remote components is required. Pregel uses an explicit message approach to acquiring remote information and does not replicate remote values locally. The most critical difference is that Pregel provides fault-tolerance to cope with failures during computation, allowing it to function in a huge cluster environment where failures are common, e.g., due to hardware failures or preemption by higher-priority jobs.

与Pregel最接近的是并行Boost图库（Parallel Boost Graph Library）和CGMgraph。并行BGL [22, 23] 为定义分布式图指定了几个关键的通用概念，提供了基于MPI [18] 的实现，并基于这些概念实现了许多算法。它试图与（顺序）BGL [43] 保持兼容，以方便算法的移植。它实现了属性映射来保存与图中的顶点和边相关的信息，使用幽灵单元（ghost cells）来保存与远程组件相关的值。如果需要引用许多远程组件，这可能会导致可扩展性问题。Pregel使用显式消息方法来获取远程信息，并且不在本地复制远程值。最关键的区别在于，Pregel提供了容错功能，以应对计算过程中的故障，使其能够在故障常见的大型集群环境中运行，例如由于硬件故障或被高优先级作业抢占资源。

CGMgraph [8] is similar in concept, providing a number of parallel graph algorithms using the Coarse Grained Multicomputer (CGM) model based on MPI. Its underlying distribution mechanisms are much more exposed to the user, and the focus is on providing implementations of algorithms rather than an infrastructure to be used to implement them. CGMgraph uses an object-oriented programming style, in contrast to the generic programming style of Parallel BGL and Pregel, at some performance cost.

CGMgraph [8] 在概念上类似，它使用基于MPI的粗粒度多计算机（Coarse Grained Multicomputer，CGM）模型提供了许多并行图算法。其底层的分布式机制对用户的暴露程度更高，并且重点在于提供算法的实现，而不是用于实现这些算法的基础设施。与并行BGL和Pregel的通用编程风格不同，CGMgraph使用面向对象的编程风格，这在一定程度上牺牲了性能。

Other than Pregel and Parallel BGL, there have been few systems reporting experimental results for graphs at the scale of billions of vertices. The largest have reported results from custom implementations of $s - t$ shortest path, rather than from general frameworks. Yoo et al [46] report on a BlueGene/L implementation of breadth-first search ( $s$ - $t$ shortest path) on 32,768 PowerPC processors with a high-performance torus network, achieving 1.5 seconds for a Poisson distributed random graph with 3.2 billion vertices and 32 billion edges. Bader and Madduri [2] report on a Cray MTA- 2 implementation of a similar problem on a 10 node, highly multithreaded system, achieving .43 seconds for a scale-free R-MAT random graph with 134 million vertices and 805 million edges. Lumsdaine et al [31] compare a Parallel BGL result on a x86-64 Opteron cluster of 200 processors to the BlueGene/L implementation, achieving .43 seconds for an Erdós-Renyi random graph of 4 billion vertices and 20 billion edges. They attribute the better performance to ghost cells, and observe that their implementation begins to get worse performance above 32 processors.

除了Pregel和并行BGL之外，很少有系统报告过针对数十亿顶点规模的图的实验结果。规模最大的报告结果来自于对 $s - t$ 最短路径的定制实现，而不是来自通用框架。Yoo等人 [46] 报告了在具有高性能环形网络的32,768个PowerPC处理器上对广度优先搜索（ $s$ - $t$ 最短路径）的BlueGene/L实现，对于一个具有32亿个顶点和320亿条边的泊松分布随机图，实现了1.5秒的计算时间。Bader和Madduri [2] 报告了在一个10节点、高度多线程的系统上对类似问题的Cray MTA - 2实现，对于一个具有1.34亿个顶点和8.05亿条边的无标度R - MAT随机图，实现了0.43秒的计算时间。Lumsdaine等人 [31] 将在200个处理器的x86 - 64 Opteron集群上的并行BGL结果与BlueGene/L实现进行了比较，对于一个具有40亿个顶点和200亿条边的埃尔德什 - 雷尼（Erdós - Renyi）随机图，实现了0.43秒的计算时间。他们将更好的性能归因于幽灵单元，并观察到他们的实现在超过32个处理器时性能开始变差。

Results for the single-source shortest paths problem on an Erdős-Renyi random graph with 256 million vertices and uniform out-degree 4,using the $\Delta$ -stepping algorithm,are reported for the Cray MTA-2 (40 processors, 2.37 sec, [32]), and for Parallel BGL on Opterons (112 processors, 35 sec., [31]). The latter time is similar to our 400-worker result for a binary tree with 1 billion nodes and edges. We do not know of any reported SSSP results on the scale of our 1 billion vertex and 127.1 billion edge log-normal graph.

对于具有2.56亿个顶点和均匀出度为4的埃尔德什 - 雷尼随机图上的单源最短路径问题，使用 $\Delta$ 步进算法的结果，报告了Cray MTA - 2（40个处理器，2.37秒，[32]）和Opterons上的并行BGL（112个处理器，35秒，[31]）的情况。后者的时间与我们在具有10亿个节点和边的二叉树上使用400个工作节点的结果相似。我们不知道有任何针对我们的具有10亿个顶点和1271亿条边的对数正态图规模的单源最短路径问题（SSSP）的报告结果。

Another line of research has tackled use of external disk memory to handle huge problems with single machines, e.g., $\left\lbrack  {{33},{36}}\right\rbrack$ ,but these implementations require hours for graphs of a billion vertices.

另一类研究致力于利用外部磁盘内存，以单台机器处理大规模问题，例如$\left\lbrack  {{33},{36}}\right\rbrack$，但这些实现方式在处理包含十亿个顶点的图时需要数小时。

## 8. CONCLUSIONS AND FUTURE WORK

## 8. 结论与未来工作

The contribution of this paper is a model suitable for large-scale graph computing and a description of its production quality, scalable, fault-tolerant implementation.

本文的贡献在于提出了一个适用于大规模图计算的模型，并对其具备生产质量、可扩展且容错的实现方式进行了描述。

Based on the input from our users we think we have succeeded in making this model useful and usable. Dozens of Pregel applications have been deployed, and many more are being designed, implemented, and tuned. The users report that once they switch to the "think like a vertex" mode of programming, the API is intuitive, flexible, and easy to use. This is not surprising, since we have worked with early adopters who influenced the API from the outset. For example, aggregators were added to remove limitations users found in the early Pregel model. Other usability aspects of Pregel motivated by user experience include a set of status pages with detailed information about the progress of Pregel programs, a unittesting framework, and a single-machine mode which helps with rapid prototyping and debugging.

根据用户反馈，我们认为已成功使该模型兼具实用性和易用性。数十个Pregel应用已部署，还有更多正在设计、实现和调优中。用户反馈称，一旦切换到“以顶点视角思考”的编程模式，该应用程序编程接口（API）直观、灵活且易于使用。这并不意外，因为我们与早期采用者合作，他们从一开始就对API产生了影响。例如，添加聚合器（aggregators）是为了消除用户在早期Pregel模型中发现的限制。受用户体验启发，Pregel的其他易用性方面还包括一组状态页面，提供Pregel程序进度的详细信息、一个单元测试框架，以及有助于快速原型开发和调试的单机模式。

The performance, scalability, and fault-tolerance of Pregel are already satisfactory for graphs with billions of vertices. We are investigating techniques for scaling to even larger graphs, such as relaxing the synchronicity of the model to avoid the cost of faster workers having to wait frequently at inter-superstep barriers.

对于包含数十亿个顶点的图，Pregel的性能、可扩展性和容错性已令人满意。我们正在研究进一步扩展到更大规模图的技术，例如放宽模型的同步性，以避免更快的工作节点频繁在超步间屏障处等待的成本。

Currently the entire computation state resides in RAM. We already spill some data to local disk, and will continue in this direction to enable computations on large graphs when terabytes of main memory are not available.

目前，整个计算状态都驻留在随机存取存储器（RAM）中。我们已经将部分数据溢出到本地磁盘，并将继续朝这个方向发展，以便在没有数TB主存可用时也能对大型图进行计算。

Assigning vertices to machines to minimize inter-machine communication is a challenge. Partitioning of the input graph based on topology may suffice if the topology corresponds to the message traffic, but it may not. We would like to devise dynamic re-partitioning mechanisms.

将顶点分配到不同机器以最小化机器间通信是一项挑战。如果输入图的拓扑结构与消息流量相符，基于拓扑结构对输入图进行分区可能就足够了，但情况并非总是如此。我们希望设计动态重新分区机制。

Pregel is designed for sparse graphs where communication occurs mainly over edges, and we do not expect that focus to change. Although care has been taken to support high fan-out and fan-in traffic, performance will suffer when most vertices continuously send messages to most other vertices. However, realistic dense graphs are rare, as are algorithms with dense communication over a sparse graph. Some such algorithms can be transformed into more Pregel-friendly variants, for example by using combiners, aggrega-tors, or topology mutations, and of course such computations are difficult for any highly distributed system.

Pregel专为稀疏图设计，其中通信主要通过边进行，我们预计这一重点不会改变。尽管已采取措施支持高扇出和扇入流量，但当大多数顶点持续向大多数其他顶点发送消息时，性能仍会受到影响。然而，现实中的稠密图很少见，在稀疏图上进行密集通信的算法也不多。一些此类算法可以转换为更适合Pregel的变体，例如通过使用合并器（combiners）、聚合器（aggregators）或拓扑变异，当然，这类计算对任何高度分布式系统来说都具有挑战性。

A practical concern is that Pregel is becoming a piece of production infrastructure for our user base. We are no longer at liberty to change the API without considering compatibility. However, we believe that the programming interface we have designed is sufficiently abstract and flexible to be resilient to the further evolution of the underlying system.

一个实际问题是，Pregel正成为我们用户群体的生产基础设施的一部分。我们不能随意更改API而不考虑兼容性。然而，我们相信所设计的编程接口足够抽象和灵活，能够适应底层系统的进一步发展。

## 9. ACKNOWLEDGMENTS

## 9. 致谢

We thank Pregel's very early users-Lorenz Huelsbergen, Galina Shubina, Zoltan Gyongyi-for their contributions to the model. Discussions with Adnan Aziz, Yossi Matias, and Steffen Meschkat helped refine several aspects of Pregel. Our interns, Punyashloka Biswal and Petar Maymounkov, provided initial evidence of Pregel's applicability to matchings and clustering, and Charles Reiss automated checkpointing decisions. The paper benefited from comments on its earlier drafts from Jeff Dean, Tushar Chandra, Luiz Barroso, Urs Hölzle, Robert Henry, Marián Dvorský, and the anonymous reviewers. Sierra Michels-Slettvet advertised Pregel to various teams within Google computing over interesting but less known graphs. Finally, we thank all the users of Pregel for feedback and many great ideas.

我们感谢Pregel的早期用户——洛伦兹·胡尔斯贝根（Lorenz Huelsbergen）、加林娜·舒比娜（Galina Shubina）、佐尔坦·久尼奥伊（Zoltan Gyongyi）——对该模型所做的贡献。与阿德南·阿齐兹（Adnan Aziz）、约西·马蒂亚斯（Yossi Matias）和斯特芬·梅施卡特（Steffen Meschkat）的讨论有助于完善Pregel的多个方面。我们的实习生普尼亚什洛卡·比斯瓦尔（Punyashloka Biswal）和彼得·梅蒙科夫（Petar Maymounkov）初步证明了Pregel在匹配和聚类方面的适用性，查尔斯·赖斯（Charles Reiss）实现了检查点决策的自动化。本文受益于杰夫·迪恩（Jeff Dean）、图沙尔·钱德拉（Tushar Chandra）、路易斯·巴罗索（Luiz Barroso）、乌尔斯·霍尔茨勒（Urs Hölzle）、罗伯特·亨利（Robert Henry）、马里安·德沃尔斯基（Marián Dvorský）和匿名审稿人对早期草稿的评论。西拉·米歇尔斯 - 斯莱特维特（Sierra Michels - Slettvet）向谷歌计算部门的各个团队宣传了Pregel，用于处理有趣但不太为人所知的图。最后，我们感谢所有Pregel用户的反馈和众多宝贵建议。

## 10. REFERENCES

## 10. 参考文献

[1] Thomas Anderson, Susan Owicki, James Saxe, and Charles Thacker, High-Speed Switch Scheduling for Local-Area Networks. ACM Trans. Comp. Syst. 11(4), 1993, 319-352.

[2] David A. Bader and Kamesh Madduri, Designing multithreaded algorithms for breadth-first search and st-connectivity on the Cray MTA-2, in Proc. 35th Intl. Conf. on Parallel Processing (ICPP'06), Columbus, OH, August 2006, 523-530.

[3] Luiz Barroso, Jeffrey Dean, and Urs Hoelzle, Web search for a planet: The Google Cluster Architecture. IEEE Micro ${23}\left( 2\right) ,{2003},{22} - {28}$ .

[4] Mohsen Bayati, Devavrat Shah, and Mayank Sharma, Maximum Weight Matching via Max-Product Belief Propagation. in Proc. IEEE Intl. Symp. on Information Theory, 2005, 1763-1767.

[5] Richard Bellman, On a routing problem. Quarterly of Applied Mathematics 16(1), 1958, 87-90.

[6] Olaf Bonorden, Ben H.H. Juurlink, Ingo von Otte, and

Ingo Rieping, The Paderborn University BSP (PUB)

英戈·里平（Ingo Rieping），帕德博恩大学批量同步并行模型（Paderborn University BSP，PUB）

Library. Parallel Computing 29(2), 2003, 187-207.

[7] Sergey Brin and Lawrence Page, The Anatomy of a Large-Scale Hypertextual Web Search Engine. in Proc. 7th Intl. Conf. on the World Wide Web, 1998, 107-117.

[8] Albert Chan and Frank Dehne, ${CGMGRAPH}/{CGMLIB}$ : Implementing and Testing CGM Graph Algorithms on PC Clusters and Shared Memory Machines. Intl. J. of High Performance Computing Applications 19(1), 2005, 81-97.

[9] Fay Chang, Jeffrey Dean, Sanjay Ghemawat, Wilson C. Hsieh, Deborah A. Wallach, Mike Burrows, Tushar Chandra, Andrew Fikes, Robert E. Gruber, Bigtable: A Distributed Storage System for Structured Data. ACM Trans. Comp. Syst. 26(2), Art. 4, 2008.

[10] Boris V. Cherkassky, Andrew V. Goldberg, and Tomasz Radzik, Shortest paths algorithms: Theory and experimental evaluation. Mathematical Programming 73, 1996, 129-174.

[11] Jonathan Cohen, Graph Twiddling in a MapReduce World. Comp. in Science & Engineering, July/August 2009, 29-41.

[12] Joseph R. Crobak, Jonathan W. Berry, Kamesh Madduri, and David A. Bader, Advanced Shortest Paths Algorithms on a Massively-Multithreaded Architecture. in Proc. First Workshop on Multithreaded Architectures and Applications, 2007, $1 - 8$ .

[13] John T. Daly, A higher order estimate of the optimum checkpoint interval for restart dumps. Future Generation Computer Systems 22, 2006, 303-312.

[14] Jeffrey Dean and Sanjay Ghemawat, MapReduce: Simplified Data Processing on Large Clusters. in Proc. 6th USENIX Symp. on Operating Syst. Design and Impl., 2004, 137-150.

[15] Edsger W. Dijkstra, A Note on Two Problems in Connexion with Graphs. Numerische Mathematik 1, 1959, 269-271.

[16] Martin Erwig, Inductive Graphs and Functional Graph Algorithms. J. Functional Programming 1(5), 2001, 467-492.

[17] Lester R. Ford, L. R. and Delbert R. Fulkerson, Flows in Networks. Princeton University Press, 1962.

[18] Ian Foster and Carl Kesselman (Eds), The Grid 2: Blueprint for a New Computing Infrastructure (2nd edition). Morgan Kaufmann, 2003.

[19] Sanjay Ghemawat, Howard Gobioff, and Shun-Tak Leung, The Google File System. in Proc. 19th ACM Symp. on Operating Syst. Principles, 2003, 29-43.

[20] Michael T. Goodrich and Roberto Tamassia, Data Structures and Algorithms in JAVA. (second edition). John Wiley and Sons, Inc., 2001.

[21] Mark W. Goudreau, Kevin Lang, Satish B. Rao, Torsten Suel, and Thanasis Tsantilas, Portable and Efficient Parallel Computing Using the BSP Model. IEEE Trans. Comp. 48(7), 1999, 670-689.

[22] Douglas Gregor and Andrew Lumsdaine, The Parallel ${BGL} : A$ Generic Library for Distributed Graph Computations. Proc. of Parallel Object-Oriented Scientific Computing (POOSC), July 2005.

[23] Douglas Gregor and Andrew Lumsdaine, Lifting Sequential Graph Algorithms for Distributed-Memory Parallel Computation. in Proc. 2005 ACM SIGPLAN Conf. on Object-Oriented Prog., Syst., Lang., and Applications (OOPSLA'05), October 2005, 423-437.

[24] Jonathan L. Gross and Jay Yellen, Graph Theory and Its Applications. (2nd Edition). Chapman and Hall/CRC, 2005.

[25] Aric A. Hagberg, Daniel A. Schult, and Pieter J. Swart, Exploring network structure, dynamics, and function using NetworkX. in Proc. 7th Python in Science Conf., 2008, 11-15.

[26] Jonathan Hill, Bill McColl, Dan Stefanescu, Mark Goudreau, Kevin Lang, Satish Rao, Torsten Suel, Thanasis Tsantilas, and Rob Bisseling, BSPlib: The BSP Programming Library. Parallel Computing 24, 1998, 1947-1980.

[27] Michael Isard, Mihai Budiu, Yuan Yu, Andrew Birrell, and Dennis Fetterly, Dryad: Distributed Data-Parallel Programs from Sequential Building Blocks. in Proc. European Conf. on Computer Syst., 2007, 59-72.

[28] Paris C. Kanellakis and Alexander A. Shvartsman, Fault-Tolerant Parallel Computation. Kluwer Academic Publishers, 1997.

[29] Donald E. Knuth, Stanford GraphBase: A Platform for Combinatorial Computing. ACM Press, 1994.

[30] U Kung, Charalampos E. Tsourakakis, and Christos Faloutsos, Pegasus: A Peta-Scale Graph Mining System - Implementation and Observations. Proc. Intl. Conf. Data Mining, 2009, 229-238.

[31] Andrew Lumsdaine, Douglas Gregor, Bruce Hendrickson, and Jonathan W. Berry, Challenges in Parallel Graph Processing. Parallel Processing Letters 17, 2007, 5-20.

[32] Kamesh Madduri, David A. Bader, Jonathan W. Berry, and Joseph R. Crobak, Parallel Shortest Path Algorithms for Solving Large-Scale Graph Instances. DIMACS Implementation Challenge - The Shortest Path Problem, 2006.

[33] Kamesh Madduri, David Ediger, Karl Jiang, David A. Bader, and Daniel Chavarria-Miranda, A Faster Parallel Algorithm and Efficient Multithreaded Implementation for Evaluating Betweenness Centrality on Massive Datasets, in Proc. 3rd Workshop on Multithreaded Architectures and Applications (MTAAP'09), Rome, Italy, May 2009.

[34] Grzegorz Malewicz, A Work-Optimal Deterministic Algorithm for the Certified Write-All Problem with a Nontrivial Number of Asynchronous Processors. SIAM J. Comput. 34(4), 2005, 993-1024.

[35] Kurt Mehlhorn and Stefan Näher, The LEDA Platform of Combinatorial and Geometric Computing. Cambridge University Press, 1999.

[36] Ulrich Meyer and Vitaly Osipov, Design and Implementation of a Practical I/O-efficient Shortest Paths Algorithm. in Proc. 3rd Workshop on Multithreaded Architectures and Applications (MTAAP'09), Rome, Italy, May 2009.

[37] Ulrich Meyer and Peter Sanders, $\Delta$ -stepping: $A$ Parallelizable Shortest Path Algorithm. J. Algorithms 49(1), 2003, 114-152.

[38] Richard Miller, A Library for Bulk-Synchronous Parallel Programming. in Proc. British Computer Society Parallel Processing Specialist Group Workshop on General Purpose Parallel Computing, 1993.

[39] Kameshwar Munagala and Abhiram Ranade, $I/O$ -complexity of graph algorithms. in Proc. 10th Annual ACM-SIAM Symp. on Discrete Algorithms, 1999, 687-694.

[40] Christopher Olston, Benjamin Reed, Utkarsh Srivastava, Ravi Kumar, and Andrew Tomkins, Pig Latin: A Not-So-Foreign Language for Data Processing. in Proc. ACM SIGMOD Intl. Conf. on Management of Data, 2008, 1099-1110.

[41] Rob Pike, Sean Dorward, Robert Griesemer, and Sean Quinlan, Interpreting the Data: Parallel Analysis with Sawzall. Scientific Programming Journal 13(4), Special Issue on Grids and Worldwide Computing Programming Models and Infrastructure, 2005, 227-298.

[42] Protocol Buffers-Google's data interchange format. http://code.google.com/p/protobuf/ 2009.

[43] Jeremy G. Siek, Lie-Quan Lee, and Andrew Lumsdaine, The Boost Graph Library: User Guide and Reference Manual. Addison Wesley, 2002.

[44] Mikkel Thorup, Undirected Single-Source Shortest Paths with Positive Integer Weights in Linear Time. J. ACM 46(3), May 1999, 362-394.

[45] Leslie G. Valiant, A Bridging Model for Parallel Computation. Comm. ACM 33(8), 1990, 103-111.

[46] Andy Yoo, Edmond Chow, Keith Henderson, William McLendon, Bruce Hendrickson, and Umit Catalyurek, A Scalable Distributed Parallel Breadth-First Search Algorithm on BlueGene/L, in Proc. 2005 ACM/IEEE Conf. on Supercomputing (SC'05), 2005, 25-43.

[47] Yuan Yu, Michael Isard, Dennis Fetterly, Mihai Budiu, Ulfar Erlingsson, Pradeep Kumar Gunda, and Jon Currey, DryadLINQ: A System for General-Purpose Distributed Data-Parallel Computing Using a High-Level Language. in Proc. 8th USENIX Symp. on Operating Syst. Design and Implementation, 2008, 10-14.