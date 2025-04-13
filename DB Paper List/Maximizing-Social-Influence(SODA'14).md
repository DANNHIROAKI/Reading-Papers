# Maximizing Social Influence in Nearly Optimal Time

# 在近乎最优时间内最大化社会影响力

Christian Borgs* Michael Brautbar ${}^{ \dagger  }$ Jennifer Chayes ${}^{ \ddagger  }$ Brendan Lucier ${}^{§}$

克里斯蒂安·博格斯* 迈克尔·布劳特巴尔 ${}^{ \dagger  }$ 詹妮弗·查耶斯 ${}^{ \ddagger  }$ 布伦丹·卢西尔 ${}^{§}$

## Abstract

## 摘要

Diffusion is a fundamental graph process, underpinning such phenomena as epidemic disease contagion and the spread of innovation by word-of-mouth. We address the algorithmic problem of finding a set of $k$ initial seed nodes in a network so that the expected size of the resulting cascade is maximized, under the standard independent cascade model of network diffusion. Runtime is a primary consideration for this problem due to the massive size of the relevant input networks.

扩散是一种基本的图过程，是诸如传染病传播和创新通过口碑传播等现象的基础。在网络扩散的标准独立级联模型下，我们解决了在网络中寻找一组 $k$ 初始种子节点的算法问题，以使由此产生的级联的预期规模最大化。由于相关输入网络的规模巨大，运行时间是该问题的主要考虑因素。

We provide a fast algorithm for the influence maximization problem, obtaining the near-optimal approximation factor of $\left( {1 - \frac{1}{e} - \epsilon }\right)$ ,for any $\epsilon  > 0$ ,in time $O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log n}\right)$ . Our algorithm is runtime-optimal (up to a logarithmic factor) and substantially improves upon the previously best-known algorithms which run in time $\Omega \left( {{mnk} \cdot  \operatorname{POLY}\left( {\epsilon }^{-1}\right) }\right)$ . Furthermore,our algorithm can be modified to allow early termination: if it is terminated after $O\left( {\beta \left( {m + n}\right) \log n}\right)$ steps for some $\beta  < 1$ (which can depend on $n$ ),then it returns a solution with approximation factor $O\left( \beta \right)$ . Finally,we show that this runtime is optimal (up to logarithmic factors) for any $\beta$ and fixed seed size $k$ .

我们为影响力最大化问题提供了一种快速算法，对于任何 $\epsilon  > 0$ ，在时间 $O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log n}\right)$ 内获得 $\left( {1 - \frac{1}{e} - \epsilon }\right)$ 的近似最优近似因子。我们的算法在运行时间上是最优的（最多相差一个对数因子），并且显著改进了之前已知的运行时间为 $\Omega \left( {{mnk} \cdot  \operatorname{POLY}\left( {\epsilon }^{-1}\right) }\right)$ 的最佳算法。此外，我们的算法可以进行修改以允许提前终止：如果在 $O\left( {\beta \left( {m + n}\right) \log n}\right)$ 步后终止（其中 $\beta  < 1$ 可以依赖于 $n$ ），则它返回一个近似因子为 $O\left( \beta \right)$ 的解。最后，我们证明对于任何 $\beta$ 和固定的种子大小 $k$ ，这个运行时间是最优的（最多相差对数因子）。

## 1 Introduction

## 1 引言

Diffusion is a fundamental process in the study of complex networks, modeling the spread of disease, ideas, or product adoption through a population. The common feature in each case is that local interactions between individuals can lead to epidemic outcomes. This is the idea behind word-of-mouth advertising, in which information about a product travels via links between individuals; see,for example $\lbrack {35},8,{17},9,3$ , 10, 19]. A prominent application is a viral marketing campaign which aims to use a small number of targeted interventions to initiate cascades of influence that create a global increase in product adoption [16, 23, 26, 17].

扩散是复杂网络研究中的一个基本过程，用于模拟疾病、思想或产品在人群中的传播。每种情况的共同特征是个体之间的局部相互作用可能导致大规模的结果。这就是口碑广告背后的理念，即产品信息通过个体之间的联系传播；例如，参见 $\lbrack {35},8,{17},9,3$ 、10、19]。一个突出的应用是病毒式营销活动，其目的是利用少量有针对性的干预措施引发影响力级联，从而在全球范围内提高产品的采用率 [16、23、26、17]。

This application gives rise to an algorithmic problem: given a network, how can we determine which individuals should be targeted to maximize the magnitude of a resulting cascade $\left\lbrack  {{16},{34},{23}}\right\rbrack$ ? Supposing that there is a limit $k$ on the number of nodes to target (e.g. due to advertising budgets), the goal is to efficiently find an appropriate set of $k$ nodes with which to "seed" a diffusion process. This problem has been studied for various models of influence spread, leading to the development of polynomial-time algorithms that achieve constant approximations $\left\lbrack  {{23},{24},{32}}\right\rbrack$ .

这个应用引出了一个算法问题：给定一个网络，我们如何确定应该针对哪些个体来最大化由此产生的级联的规模 $\left\lbrack  {{16},{34},{23}}\right\rbrack$ ？假设要针对的节点数量有一个限制 $k$ （例如，由于广告预算的原因），目标是有效地找到一组合适的 $k$ 个节点来“播种”扩散过程。针对各种影响力传播模型对这个问题进行了研究，从而开发出了能实现常数近似的多项式时间算法 $\left\lbrack  {{23},{24},{32}}\right\rbrack$。

Relevant networks for this problem can have massive size, on the order of billions of edges. The running time of an influence maximization algorithm is therefore a primary consideration. This is compounded by the fact that social network data and influence parameters tend to be volatile, necessitating recomputation of solutions over time. For these reasons, near-linear runtime is a practical necessity for algorithms that work with massive network data. This stringent runtime requirement has spawned a large body of work aimed at developing fast, heuristic methods of finding influential individuals in social networks, despite the existence of the above-mentioned approximation algorithms. See, for example, $\left\lbrack  {{13},{12},{14},{27},{38},{30},{22},{25}}\right\rbrack$ . However, to date, this line of work has focused primarily on empirical methods. Currently, the fastest algorithms with constant-factor approximation guarantees have runtime $\Omega \left( {nmk}\right) \left\lbrack  {13}\right\rbrack$ .

该问题相关的网络规模可能非常大，边的数量可达数十亿。因此，影响力最大化算法的运行时间是主要考虑因素。此外，社交网络数据和影响力参数往往不稳定，这就需要随着时间重新计算解决方案。出于这些原因，对于处理大规模网络数据的算法来说，近乎线性的运行时间是实际必需的。尽管存在上述近似算法，但这种严格的运行时间要求催生了大量旨在开发快速启发式方法以在社交网络中寻找有影响力个体的工作。例如，参见 $\left\lbrack  {{13},{12},{14},{27},{38},{30},{22},{25}}\right\rbrack$。然而，到目前为止，这方面的工作主要集中在实证方法上。目前，具有常数因子近似保证的最快算法的运行时间为 $\Omega \left( {nmk}\right) \left\lbrack  {13}\right\rbrack$。

In this paper we bridge this gap by developing a constant-factor approximation algorithm for the influence maximization problem, under the standard independent cascade model of influence spread, that runs in quasilinear time. Our algorithm can also be modified to run in sublinear time, with a correspondingly reduced approximation factor. Before describing these results in detail, we first provide some background into the influence model.

在本文中，我们通过开发一种在标准影响力传播独立级联模型下解决影响力最大化问题的常数因子近似算法来弥补这一差距，该算法的运行时间为拟线性。我们的算法也可以修改为以亚线性时间运行，相应地降低近似因子。在详细描述这些结果之前，我们首先介绍一下影响力模型的一些背景知识。

The Model: Independent Cascades We adopt the independent cascade (IC) model of diffusion, formalized by Kempe et al. [23]. In this model we are given a directed edge-weighted graph $\mathcal{G}$ with $n$ nodes and $m$ edges, representing the underlying network. Influence spreads via a random process that begins at a set $S$ of seed nodes. Each node, once infected, has a chance of subsequently infecting its neighbors: the weight of edge $e = \left( {v,u}\right)$ represents the probability that the process spreads along edge $e$ from $v$ to $u$ . If we write $I\left( S\right)$ for the (random) number of nodes that are eventually infected by this process, then we think of the expectation of $I\left( S\right)$ as the influence of set $S$ . Our optimization problem,then,is to find set $S$ maximizing $\mathbb{E}\left\lbrack  {I\left( S\right) }\right\rbrack$ subject to $\left| S\right|  \leq  k$ .

模型：独立级联 我们采用由肯普（Kempe）等人[23]正式提出的独立级联（IC）扩散模型。在该模型中，我们有一个带权有向图$\mathcal{G}$，它有$n$个节点和$m$条边，代表底层网络。影响力通过一个随机过程传播，该过程从一组种子节点$S$开始。每个节点一旦被激活，就有机会随后激活其邻居节点：边$e = \left( {v,u}\right)$的权重表示该过程沿着边$e$从$v$传播到$u$的概率。如果我们用$I\left( S\right)$表示最终被该过程激活的（随机）节点数量，那么我们将$I\left( S\right)$的期望值视为集合$S$的影响力。那么，我们的优化问题就是在满足$\left| S\right|  \leq  k$的条件下，找到使$\mathbb{E}\left\lbrack  {I\left( S\right) }\right\rbrack$最大化的集合$S$。

---

<!-- Footnote -->

*Microsoft Research

*微软研究院

${}^{ \dagger  }$ Computer and Information Science,University of Pennsylvania. Now in The Laboratory for Information and Decision Systems, Massachusetts Institute of Technology.

${}^{ \dagger  }$ 宾夕法尼亚大学计算机与信息科学系。现就职于麻省理工学院信息与决策系统实验室。

${}^{ \ddagger  }$ Microsoft Research

${}^{ \ddagger  }$ 微软研究院

§Microsoft Research

§微软研究院

<!-- Footnote -->

---

The IC model captures the intuition that influence can spread stochastically through a network, much like a disease $\left\lbrack  {{23},{17},{15}}\right\rbrack$ . Since its introduction,it has become one of the prominent models of influence spread; see for example [13, 12, 14, 27, 38, 24]. Kempe et al. show that $\mathbb{E}\left\lbrack  {I\left( \cdot \right) }\right\rbrack$ is a submodular monotone function [23], and hence the problem of maximizing $\mathbb{E}\left\lbrack  {I\left( \cdot \right) }\right\rbrack$ can be approximated to within a factor of $\left( {1 - \frac{1}{e} - \epsilon }\right)$ for any $\epsilon  > 0$ ,in polynomial time,via a greedy hill-climbing method. In contrast, many other formulations of the influence maximization problem have been shown to have strong lower bounds on polynomial-time approximability $\left\lbrack  {{33},{31},{11},4}\right\rbrack$ .

独立级联（IC）模型体现了这样一种直觉，即影响力可以像疾病$\left\lbrack  {{23},{17},{15}}\right\rbrack$一样在网络中随机传播。自提出以来，它已成为影响力传播的主要模型之一；例如，参见文献[13, 12, 14, 27, 38, 24]。肯普（Kempe）等人证明$\mathbb{E}\left\lbrack  {I\left( \cdot \right) }\right\rbrack$是一个次模单调函数[23]，因此，对于任意$\epsilon  > 0$，通过贪心爬山法可以在多项式时间内将最大化$\mathbb{E}\left\lbrack  {I\left( \cdot \right) }\right\rbrack$的问题近似到$\left( {1 - \frac{1}{e} - \epsilon }\right)$的因子范围内。相比之下，许多其他影响力最大化问题的公式化表述已被证明在多项式时间近似性方面有很强的下界$\left\lbrack  {{33},{31},{11},4}\right\rbrack$。

The greedy approach to maximizing influence in the IC model described above takes time $O\left( {kn}\right)$ ,given oracle access to the function $\mathbb{E}\left\lbrack  {I\left( \cdot \right) }\right\rbrack$ . However,influence values must be computed from the underlying network topology, by (for example) repeated simulation of the diffusion process. This leads ultimately ${}^{1}$ to a runtime ${}^{2}$ of $\Omega \left( {{mnk} \cdot  \operatorname{POLY}\left( {\epsilon }^{-1}\right) }\right)$ .

上述在独立级联（IC）模型中最大化影响力的贪心方法，在可以调用函数$\mathbb{E}\left\lbrack  {I\left( \cdot \right) }\right\rbrack$的情况下，需要$O\left( {kn}\right)$的时间。然而，影响力值必须从底层网络拓扑结构中计算得出，例如通过对扩散过程进行重复模拟。这最终${}^{1}$导致运行时间${}^{2}$为$\Omega \left( {{mnk} \cdot  \operatorname{POLY}\left( {\epsilon }^{-1}\right) }\right)$。

Our Result: A Quasi-Linear Time Algorithm Our main result is an algorithm for finding $\left( {1 - 1/e - \epsilon }\right)$ - approximately optimal seed sets in arbitrary directed networks,which runs in time $O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log n}\right)$ . Notably, the runtime of our algorithm is independent of the number of seeds $k$ . Moreover,this runtime is close to optimal,as we give a lower bound of $\Omega \left( {m + n}\right)$ on the time required to obtain a constant approximation, assuming an adjacency list representation of the network as well as the the ability to uniformly sample nodes. We also note that this approximation factor is nearly optimal, as no polytime algorithm achieves approximation $\left( {1 - \frac{1}{e} + \epsilon }\right)$ for any $\epsilon  > 0$ unless $\mathrm{P} = \mathrm{{NP}}\left\lbrack  {{23},{24}}\right\rbrack$ . Our algorithm is randomized, and it succeeds with probability $3/5$ ; moreover,failure is detectable,so this success probability can be amplified through repetition.

我们的成果：准线性时间算法 我们的主要成果是一种在任意有向网络中寻找$\left( {1 - 1/e - \epsilon }\right)$ - 近似最优种子集的算法，其运行时间为$O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log n}\right)$。值得注意的是，我们算法的运行时间与种子数量$k$无关。此外，这个运行时间接近最优，因为我们给出了在假设网络采用邻接表表示且能够均匀采样节点的情况下，获得常数近似所需时间的下界$\Omega \left( {m + n}\right)$。我们还注意到，这个近似因子几乎是最优的，因为除非$\mathrm{P} = \mathrm{{NP}}\left\lbrack  {{23},{24}}\right\rbrack$，否则对于任意$\epsilon  > 0$，没有多项式时间算法能达到近似$\left( {1 - \frac{1}{e} + \epsilon }\right)$。我们的算法是随机的，它以概率$3/5$成功；此外，失败是可检测的，因此可以通过重复执行来提高这个成功概率。

We assume that the network topology is described in the sparse representation of an (arbitrarily ordered) adjacency list for each vertex, as is natural for sparse graphs such as social networks. Our algorithms access the network structure in a very limited way: the only queries used by our algorithms are uniform vertex sampling and traversing the edges incident to a previously-accessed vertex.

我们假设网络拓扑以每个顶点的（任意排序的）邻接表的稀疏表示形式描述，这对于社交网络等稀疏图来说是很自然的。我们的算法以非常有限的方式访问网络结构：我们的算法使用的唯一查询是均匀顶点采样和遍历与先前访问的顶点相关联的边。

To describe our approach, let us first consider the problem of finding the single node with highest influence. One strategy would be to estimate the influence of every node directly, e.g., via repeated simulation, but this is computationally expensive. Alternatively, consider the following "polling" process: select a node $v$ uniformly at random, and determine the set of nodes that would have influenced $v$ . Intuitively,if we repeat this process multiple times,and a certain node $u$ appears often as an "influencer," then $u$ is likely a good candidate for the most influential node. Indeed, we show that the probability a node $u$ appears in a set of influencers is proportional to $\mathbb{E}\left\lbrack  {I\left( u\right) }\right\rbrack$ ,and standard concentration bounds show that this probability can be estimated accurately with relatively few repetitions of the polling process. Moreover, it is possible to efficiently find the set of nodes that would have influenced a node $v$ : this can be done by simulating the influence process, starting from $v$ ,in the transpose graph (i.e.,the original network with edge directions reversed).

为了描述我们的方法，让我们首先考虑找到具有最高影响力的单个节点的问题。一种策略是直接估计每个节点的影响力，例如，通过重复模拟，但这在计算上是昂贵的。或者，考虑以下“投票”过程：随机均匀地选择一个节点$v$，并确定会影响$v$的节点集合。直观地说，如果我们多次重复这个过程，并且某个节点$u$经常作为“影响者”出现，那么$u$很可能是最有影响力节点的一个很好的候选者。实际上，我们表明节点$u$出现在影响者集合中的概率与$\mathbb{E}\left\lbrack  {I\left( u\right) }\right\rbrack$成正比，并且标准的集中界表明，通过相对较少次数的投票过程重复就可以准确估计这个概率。此外，可以有效地找到会影响节点$v$的节点集合：这可以通过在转置图（即，边方向反转的原始网络）中从$v$开始模拟影响过程来完成。

This motivates our algorithm, which proceeds in two steps. First, we repeatedly apply the random sampling technique described above to generate a sparse hypergraph representation of the network. Each hyper-graph edge corresponds to a set of individuals that was influenced by a randomly selected node in the transpose graph. This preprocessing is done once, resulting in a structure of size $O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right) }\right)$ . This hy-pergraph encodes our influence estimates: for a set of nodes $S$ ,the total degree of $S$ in the hypergraph is approximately proportional to the influence of $S$ in the original graph. In the second step, we run a standard greedy algorithm on this hypergraph to return a set of size $k$ of approximately maximal total degree.

这启发了我们的算法，该算法分两步进行。首先，我们反复应用上述随机采样技术来生成网络的稀疏超图表示。每个超图边对应于在转置图中由一个随机选择的节点影响的一组个体。这个预处理只进行一次，得到一个大小为$O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right) }\right)$的结构。这个超图编码了我们的影响力估计：对于一组节点$S$，$S$在超图中的总度数大约与$S$在原始图中的影响力成正比。在第二步中，我们在这个超图上运行一个标准的贪心算法，以返回一个大小为$k$的近似最大总度数的集合。

To make this approach work one needs to overcome several inherent difficulties. First, note that our sampling method allows us to estimate the influence of a node, but not the marginal benefit of adding a node to a partially constructed seed set. Thus, unlike prior algorithms, we do not repeat our estimation procedure to incrementally construct a solution. Instead, we perform all of our sampling up front and then select the entire seed set using the resulting hypergraph.

为了使这种方法可行，需要克服几个固有的困难。首先，请注意我们的采样方法允许我们估计一个节点的影响力，但不能估计将一个节点添加到部分构建的种子集的边际效益。因此，与先前的算法不同，我们不会重复我们的估计过程来逐步构建一个解决方案。相反，我们预先进行所有的采样，然后使用得到的超图选择整个种子集。

Second, our algorithm has a stringent runtime constraint - we must construct our hypergraph in time $O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right) }\right)$ . To meet this bound,we must be flexible in the number of hyperedges we construct. Instead of building a certain fixed number of hyper-edges, we repeatedly build edges until the total sum of all edge sizes exceeds $O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right) }\right)$ . Intuitively speaking, this works because the number of hyperedges needed to accurately estimate influence values, times the expected size of each hyperedge, is roughly constant. Indeed, we should expect to see large hyperedges only if the network contains many influential nodes, but high influence values require fewer samples to estimate accurately.

其次，我们的算法有一个严格的运行时间约束——我们必须在时间$O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right) }\right)$内构建我们的超图。为了满足这个界限，我们必须在构建的超边数量上保持灵活性。我们不是构建一定数量的固定超边，而是反复构建边，直到所有边大小的总和超过$O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right) }\right)$。直观地说，这是可行的，因为准确估计影响力值所需的超边数量乘以每个超边的预期大小大致是恒定的。实际上，只有当网络包含许多有影响力的节点时，我们才应该期望看到大的超边，但高影响力值需要较少的样本就能准确估计。

---

<!-- Footnote -->

${}^{1}$ After simple optimizations,such as reusing simulations for multiple nodes.

${}^{1}$ 在进行简单的优化后，例如为多个节点重用模拟。

${}^{2}$ The best implementations appear to have running time $O\left( {{mnk}\log \left( n\right)  \cdot  \operatorname{POLY}\left( {\epsilon }^{-1}\right) }\right) \left\lbrack  {13}\right\rbrack$ ,though to the best of our knowledge a formal analysis of this runtime has not appeared in the literature.

${}^{2}$ 最好的实现似乎具有运行时间$O\left( {{mnk}\log \left( n\right)  \cdot  \operatorname{POLY}\left( {\epsilon }^{-1}\right) }\right) \left\lbrack  {13}\right\rbrack$，尽管据我们所知，对这个运行时间的正式分析尚未在文献中出现。

<!-- Footnote -->

---

Finally, in order to prevent errors from accumulating when we apply the greedy algorithm to the hyper-graph, it is important that our estimator for the influence function (i.e. total hypergraph degree) is itself a monotone submodular function.

最后，为了防止在我们对超图应用贪心算法时误差累积，重要的是我们对影响函数（即超图总度数）的估计器本身是一个单调次模函数。

Early Termination and Sublinear Time We next show how to modify our approximation algorithm to allow early termination, providing a tradeoff between runtime and approximation quality. Specifically, if the algorithm is allowed to run for $O\left( {\beta \left( {n + m}\right) \log \left( n\right) }\right)$ steps, and is then terminated without warning, it can immediately return a solution with approximation factor $O\left( \beta \right)$ . We also provide a lower bound of $\Omega (\beta (m +$ $n)/\min \{ 1/\beta ,k\} )$ on the runtime needed to obtain an $O\left( \beta \right)$ -approximation. Our algorithm is therefore nearly runtime-optimal (up to logarithmic factors) for any fixed seed size $k$ . Our method is randomized,and it succeeds with probability $3/5$ . As before,these results assume that the input network is provided in adjacency list format and an algorithm is allowed to perform uniform sampling of the nodes. Notably, our algorithm accesses the network structure in a very limited way and is directly implementable (with no additional cost) in a wide array of graph access models, including the ones used for sublinear-time and property testing algorithms on sparse graphs $\left\lbrack  {{36},{18}}\right\rbrack$ and the jump and crawl paradigm of [7].

提前终止与亚线性时间 接下来，我们将展示如何修改我们的近似算法以实现提前终止，从而在运行时间和近似质量之间进行权衡。具体而言，如果允许该算法运行 $O\left( {\beta \left( {n + m}\right) \log \left( n\right) }\right)$ 步，然后在无预警的情况下终止，它可以立即返回一个近似因子为 $O\left( \beta \right)$ 的解。我们还给出了获得 $O\left( \beta \right)$ -近似解所需运行时间的下界 $\Omega (\beta (m +$ $n)/\min \{ 1/\beta ,k\} )$。因此，对于任何固定的种子集大小 $k$，我们的算法在运行时间上几乎是最优的（达到对数因子级别）。我们的方法是随机化的，并且以概率 $3/5$ 成功。和之前一样，这些结果假设输入网络以邻接表格式提供，并且算法可以对节点进行均匀采样。值得注意的是，我们的算法以非常有限的方式访问网络结构，并且可以在多种图访问模型中直接实现（无需额外成本），包括用于稀疏图的亚线性时间和属性测试算法的模型 $\left\lbrack  {{36},{18}}\right\rbrack$ 以及文献 [7] 中的跳跃和爬行范式。

The intuition behind our modified algorithm is that a tradeoff between execution time and approximation factor can be achieved by constructing fewer edges in our hypergraph representation. Given an upper bound on runtime, we can build edges until that time has expired, then run the influence maximization algorithm using the resulting (impoverished) hypergraph. We show that this approach generates a solution whose quality degrades gracefully with the preprocessing time, with an important caveat. If the network contains many nodes with high influence, it may be that a reduction in runtime prevents us from achieving enough concentration to estimate the influence of any node. However, in this case, the fact that many individuals have high influence enables an alternative approach: a node chosen at random, according to the degree distribution of nodes in the hypergraph representation, is likely to have high influence.

我们改进算法背后的直觉是，通过在超图表示中构建更少的边，可以在执行时间和近似因子之间实现权衡。给定运行时间的上限，我们可以在该时间耗尽之前构建边，然后使用得到的（不完整的）超图运行影响力最大化算法。我们证明了这种方法生成的解的质量会随着预处理时间的减少而适度下降，但有一个重要的注意事项。如果网络中包含许多具有高影响力的节点，那么运行时间的减少可能会使我们无法获得足够的集中度来估计任何节点的影响力。然而，在这种情况下，许多个体具有高影响力这一事实使得我们可以采用另一种方法：根据超图表示中节点的度分布随机选择一个节点，该节点很可能具有高影响力。

Given the above, our algorithm will proceed by constructing two possible seed sets: one using the greedy algorithm applied to the constructed hypergraph, and the other by randomly selecting a singleton according to the hypergraph degree distribution. If $k > 1$ we will return a union of these two solutions. When $k = 1$ we cannot use both solutions, so we must choose; in this case, it turns out that we can determine which of the two solutions will achieve the desired approximation by examining the maximum degree in the hypergraph.

基于上述情况，我们的算法将通过构建两个可能的种子集来进行：一个是对构建的超图应用贪心算法得到的，另一个是根据超图的度分布随机选择一个单元素集得到的。如果 $k > 1$，我们将返回这两个解的并集。当 $k = 1$ 时，我们不能同时使用这两个解，因此必须做出选择；在这种情况下，事实证明，我们可以通过检查超图中的最大度来确定这两个解中哪一个将达到所需的近似度。

Finally, to allow early termination without warning, the algorithm can pause its hypergraph construction and compute a tentative solution at predetermined intervals (e.g., repeatedly doubling the number of steps between computations). Then, upon a request to terminate, the algorithm returns the most recent solution.

最后，为了允许无预警的提前终止，算法可以暂停其超图构建，并在预定的时间间隔（例如，在计算之间的步数反复加倍）计算一个临时解。然后，当收到终止请求时，算法返回最近的解。

1.1 Related Work Models of influence spread in networks, covering both cascade and threshold phenomena, are well-studied in the sociology and marketing literature $\left\lbrack  {{21},{35},{17}}\right\rbrack$ . The problem of finding the most influential set of nodes to target for a diffusive process was first posed by Domingos and Richardson [16, 34]. A formal development of the IC model, along with a greedy algorithm based upon submodular maximization, was given by Kempe et al. [23]. Many subsequent works have studied the nature of diffusion in online social networks, using empirical data to estimate influence probabilities and infer network topology; see $\left\lbrack  {{29},{19},{28}}\right\rbrack$ .

1.1 相关工作 网络中影响力传播的模型，涵盖级联和阈值现象，在社会学和营销文献中已有深入研究 $\left\lbrack  {{21},{35},{17}}\right\rbrack$。寻找用于扩散过程的最具影响力的节点集的问题最初由多明戈斯（Domingos）和理查森（Richardson）提出 [16, 34]。凯姆普（Kempe）等人给出了独立级联（IC）模型的正式发展以及基于子模最大化的贪心算法 [23]。许多后续工作使用实证数据来估计影响力概率并推断网络拓扑结构，研究了在线社交网络中扩散的性质；参见 $\left\lbrack  {{29},{19},{28}}\right\rbrack$。

It has been shown that many alternative formulations of the influence maximization problem are computationally difficult. The problem of finding, in a linear threshold model, a set of minimal size that influences the entire network was shown to be inapproximable within $O\left( {n}^{1 - \epsilon }\right)$ by Chen [11]. The problem of determining influence spread given a seed set in the IC model is $\# \mathrm{P}$ - hard [12].

研究表明，影响力最大化问题的许多替代公式在计算上是困难的。陈（Chen）证明了在线性阈值模型中，找到一个影响整个网络的最小规模节点集的问题在 $O\left( {n}^{1 - \epsilon }\right)$ 内是不可近似的 [11]。在独立级联（IC）模型中，给定一个种子集确定影响力传播的问题是 $\# \mathrm{P}$ -难的 [12]。

There has been a line of work aimed at improving the runtime of the algorithm by Kempe et al. [23]. These have focused largely on heuristics, such as assuming that all nodes have relatively low influence or that the input graph is clustered $\left\lbrack  {{12},{14},{25},{38}}\right\rbrack$ ,as well as empirically-motivated implementation improvements $\left\lbrack  {{27},{13}}\right\rbrack$ . One particular approach of note involves first attempting to sparsify the input graph, then estimating influence on the reduced network $\left\lbrack  {{14},{30}}\right\rbrack$ . Unfortunately, these sparsification problems are shown to be computationally intractible in general.

有一系列工作旨在改进肯普（Kempe）等人 [23] 提出的算法的运行时间。这些工作主要集中在启发式方法上，例如假设所有节点的影响力相对较低，或者输入图是聚类的 $\left\lbrack  {{12},{14},{25},{38}}\right\rbrack$，以及基于经验的实现改进 $\left\lbrack  {{27},{13}}\right\rbrack$。一个特别值得注意的方法是先尝试对输入图进行稀疏化，然后在简化后的网络上估计影响力 $\left\lbrack  {{14},{30}}\right\rbrack$。不幸的是，一般来说，这些稀疏化问题在计算上是难以处理的。

Various alternative formulations of influence spread as a submodular process have been proposed and analyzed in the literature $\left\lbrack  {{32},{24}}\right\rbrack$ ,including those that include interations between multiple diffusive processes $\left\lbrack  {{20},5}\right\rbrack$ . We focus specifically on the IC model,and leave open the question of whether our methods can be extended to apply to these alternative models.

文献中已经提出并分析了将影响力传播作为子模过程的各种替代公式 $\left\lbrack  {{32},{24}}\right\rbrack$，包括那些考虑多个扩散过程之间相互作用的公式 $\left\lbrack  {{20},5}\right\rbrack$。我们特别关注独立级联（IC）模型，并保留我们的方法是否可以扩展应用于这些替代模型的问题。

The influence estimation problem shares some commonality with the problems of local graph partitioning, as well as estimating pagerank and personalized pager-ank vectors $\left\lbrack  {1,6,{37},2}\right\rbrack$ . These problems admit local algorithms based on sampling short random walks. To the best of our understanding, these methods do not seem directly applicable to influence maximization.

影响力估计问题与局部图划分问题以及估计网页排名（PageRank）和个性化网页排名向量问题有一些共同之处 $\left\lbrack  {1,6,{37},2}\right\rbrack$。这些问题可以采用基于采样短随机游走的局部算法。据我们所知，这些方法似乎不能直接应用于影响力最大化问题。

## 2 Model and Preliminaries

## 2 模型与预备知识

The Independent Cascade Model In the independent cascade (IC) model, influence spreads via an edge-weighted directed graph $\mathcal{G}$ . An infection begins at a set $S$ of seed nodes,and spreads through the network in rounds. Each infected node $v$ has a single chance, upon first becoming infected, of subsequently infecting his neighbors. Each directed edge $e = \left( {v,u}\right)$ has a weight ${p}_{e} \in  \left\lbrack  {0,1}\right\rbrack$ representing the probability that the process spreads along edge $e$ to node $u$ in the round following the round in which $v$ was first infected.

独立级联模型 在独立级联（IC）模型中，影响力通过一个带边权的有向图 $\mathcal{G}$ 传播。感染从一组种子节点 $S$ 开始，并在网络中逐轮传播。每个被感染的节点 $v$ 在首次被感染时，有一次机会去感染其邻居节点。每条有向边 $e = \left( {v,u}\right)$ 都有一个权重 ${p}_{e} \in  \left\lbrack  {0,1}\right\rbrack$，表示在 $v$ 首次被感染的下一轮中，感染过程沿着边 $e$ 传播到节点 $u$ 的概率。

As noted in [23], the above process has the following equivalent description. We can interpret $\mathcal{G}$ as a distribution over unweighted directed graphs,where each edge $e$ is independently realized with probability ${p}_{e}$ . If we realize a graph $g$ according to this probability distribution, then we can associate the set of infected nodes in the original process with the set of nodes reachable from seed set $S$ in $g$ . We will make use of this alternative formulation of the IC model throughout the paper.

正如 [23] 中所指出的，上述过程有以下等价描述。我们可以将 $\mathcal{G}$ 解释为无权有向图的一个分布，其中每条边 $e$ 以概率 ${p}_{e}$ 独立实现。如果我们根据这个概率分布生成一个图 $g$，那么我们可以将原始过程中被感染的节点集与图 $g$ 中从种子集 $S$ 可达的节点集联系起来。在本文中，我们将始终使用独立级联模型的这种替代公式。

Notation We let $m$ and $n$ denote the number of edges and nodes, respectively, in the weighted directed graph $\mathcal{G}$ . We write $g \sim  \mathcal{G}$ to mean that $g$ is drawn from the random graph distribution $\mathcal{G}$ . Given set $S$ of vertices and (unweighted) directed graph $g$ ,write ${C}_{g}\left( S\right)$ for the set of nodes reachable from $S$ in $g$ . When $g$ is drawn from $\mathcal{G}$ ,we will refer to this as the set of nodes influenced by $S$ . We write ${I}_{g}\left( S\right)  = \left| {{C}_{g}\left( S\right) }\right|$ for the number of nodes influenced by $S$ ,which we call the influence of $S$ in $g$ . We write ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   = {\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  {{I}_{g}\left( S\right) }\right\rbrack$ for the expected influence of $S$ in $\mathcal{G}$ .

符号表示 我们分别用 $m$ 和 $n$ 表示带权有向图 $\mathcal{G}$ 中的边数和节点数。我们用 $g \sim  \mathcal{G}$ 表示 $g$ 是从随机图分布 $\mathcal{G}$ 中抽取的。给定顶点集 $S$ 和（无权）有向图 $g$，用 ${C}_{g}\left( S\right)$ 表示在图 $g$ 中从 $S$ 可达的节点集。当 $g$ 是从 $\mathcal{G}$ 中抽取时，我们将其称为受 $S$ 影响的节点集。我们用 ${I}_{g}\left( S\right)  = \left| {{C}_{g}\left( S\right) }\right|$ 表示受 $S$ 影响的节点数，我们称之为 $S$ 在图 $g$ 中的影响力。我们用 ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   = {\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  {{I}_{g}\left( S\right) }\right\rbrack$ 表示 $S$ 在 $\mathcal{G}$ 中的期望影响力。

Given two sets of nodes $S$ and $W$ ,we write ${C}_{g}\left( {S \mid  W}\right)$ for the set of nodes reachable from $S$ but not from $W$ . That is, ${C}_{g}\left( {S \mid  W}\right)  = {C}_{g}\left( S\right)  \smallsetminus  {C}_{g}\left( W\right)$ . As before, we write ${I}_{g}\left( {S \mid  W}\right)  = \left| {{C}_{g}\left( {S \mid  W}\right) }\right|$ ; we refer to this as the marginal influence of $S$ given $W$ . The expected marginal influence of $S$ given $W$ is ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( {S \mid  W}\right) }\right\rbrack   =$ ${\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  {{I}_{g}\left( {S \mid  W}\right) }\right\rbrack$ .

给定两组节点$S$和$W$，我们用${C}_{g}\left( {S \mid  W}\right)$表示从$S$可达但从$W$不可达的节点集合。即${C}_{g}\left( {S \mid  W}\right)  = {C}_{g}\left( S\right)  \smallsetminus  {C}_{g}\left( W\right)$。和之前一样，我们记为${I}_{g}\left( {S \mid  W}\right)  = \left| {{C}_{g}\left( {S \mid  W}\right) }\right|$；我们将其称为给定$W$时$S$的边际影响。给定$W$时$S$的期望边际影响为${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( {S \mid  W}\right) }\right\rbrack   =$ ${\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  {{I}_{g}\left( {S \mid  W}\right) }\right\rbrack$。

In general, a vertex in the subscript of an expectation or probability denotes the vertex being selected uniformly at random from the set of vertices of $\mathcal{G}$ . For example, ${\mathbb{E}}_{v,\mathcal{G}}\left\lbrack  {I\left( v\right) }\right\rbrack$ is the average,over all graph nodes $v$ ,of the expected influence of $v$ .

一般来说，期望或概率下标中的顶点表示从$\mathcal{G}$的顶点集合中均匀随机选择的顶点。例如，${\mathbb{E}}_{v,\mathcal{G}}\left\lbrack  {I\left( v\right) }\right\rbrack$是所有图节点$v$的期望影响的平均值。

For a given graph $g$ ,define ${g}^{T}$ to be the transpose graph of $g : \left( {u,v}\right)  \in  g$ iff $\left( {v,u}\right)  \in  {g}^{T}$ . We apply this notation to both unweighted and weighted graphs.

对于给定的图$g$，当且仅当$\left( {v,u}\right)  \in  {g}^{T}$时，定义${g}^{T}$为$g : \left( {u,v}\right)  \in  g$的转置图。我们将此符号表示法应用于无权图和有权图。

The Influence Maximization Problem Given graph $\mathcal{G}$ and integer $k \geq  1$ ,the influence maximization problem is to find a set $S$ of at most $k$ nodes maximizing the value of ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack$ . For $\beta  \leq  1$ ,we say that a particular set of nodes $S$ with $\left| S\right|  \leq  k$ is a $\beta$ - approximation to the influence maximization problem if ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  \beta  \cdot  \mathop{\max }\limits_{{T : \left| T\right|  = k}}{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack$ . We assume that graph $\mathcal{G}$ is provided in adjacency list format,with the neighbors of a given vertex $v$ ordered arbitrarily.

影响力最大化问题 给定图$\mathcal{G}$和整数$k \geq  1$，影响力最大化问题是找到一个最多包含$k$个节点的集合$S$，使得${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack$的值最大。对于$\beta  \leq  1$，如果${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  \beta  \cdot  \mathop{\max }\limits_{{T : \left| T\right|  = k}}{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack$，我们称具有$\left| S\right|  \leq  k$的特定节点集合$S$是影响力最大化问题的$\beta$ - 近似解。我们假设图$\mathcal{G}$以邻接表格式提供，给定顶点$v$的邻居节点任意排序。

A Simulation Primitive Our algorithms we will make use of a primitive that realizes an instance of the nodes influenced by a given vertex $u$ in weighted graph $\mathcal{G}$ ,and returns this set of nodes. Conceptually,this is done by realizing some $g \sim  \mathcal{G}$ and traversing ${C}_{g}\left( u\right)$ .

一种模拟原语 我们的算法将使用一个原语，该原语用于实现有权图$\mathcal{G}$中受给定顶点$u$影响的节点实例，并返回这个节点集合。从概念上讲，这是通过实现某个$g \sim  \mathcal{G}$并遍历${C}_{g}\left( u\right)$来完成的。

Let us briefly discuss the implementation of such a primitive. Given node $u$ ,we can run a depth first search in $\mathcal{G}$ starting at node $u$ . Before traversing any given edge $e$ ,we perform a random test: with probability ${p}_{e}$ we traverse the edge as normal, and with probability $1 - {p}_{e}$ we do not traverse edge $e$ and ignore it from that point onward. The set of nodes traversed in this manner is equivalent to ${C}_{g}\left( u\right)$ for $g \sim  \mathcal{G}$ ,due to deferred randomness. We then return the set of nodes traversed. The runtime of this procedure is precisely the sum of the degrees (in $\mathcal{G}$ ) of the vertices in ${C}_{g}\left( u\right)$ .

让我们简要讨论一下这种原语的实现。给定节点 $u$ ，我们可以从节点 $u$ 开始在 $\mathcal{G}$ 中进行深度优先搜索。在遍历任何给定的边 $e$ 之前，我们进行一个随机测试：以概率 ${p}_{e}$ 正常遍历该边，以概率 $1 - {p}_{e}$ 不遍历边 $e$ 并从那时起忽略它。由于延迟随机性，以这种方式遍历的节点集等同于 $g \sim  \mathcal{G}$ 时的 ${C}_{g}\left( u\right)$ 。然后我们返回遍历的节点集。此过程的运行时间恰好是 ${C}_{g}\left( u\right)$ 中顶点的度数（在 $\mathcal{G}$ 中）之和。

We can implement this procedure for a traversal of ${g}^{T}$ ,rather than $g$ ,by following in-links rather than out-links in our tree traversal.

我们可以通过在树遍历中遵循入边而不是出边，来实现对 ${g}^{T}$ 而不是 $g$ 的遍历过程。

## 3 An Approximation Algorithm for Influence Maximization

## 3 影响力最大化的近似算法

In this section we present an algorithm for the influence maximization problem on arbitrary directed graphs. Our algorithm returns a $\left( {1 - \frac{1}{e} - \epsilon }\right)$ -approximation to the influence maximization problem, with success probability $3/5$ ,in time $O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log n}\right)$ . We discuss how to amplify this success probability in Section 3.1.

在本节中，我们提出一种用于任意有向图上影响力最大化问题的算法。我们的算法在时间 $O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log n}\right)$ 内以成功概率 $3/5$ 返回影响力最大化问题的 $\left( {1 - \frac{1}{e} - \epsilon }\right)$ -近似解。我们将在 3.1 节讨论如何提高这个成功概率。

The algorithm is described formally as Algorithm 1 , but let us begin by describing our construction informally. Our approach proceeds in two steps. The first step, BuildHypergraph, stochastically generates a sparse,undirected hypergraph representation $\mathcal{H}$ of our underlying graph $g$ . This is done by repeatedly simulating the influence spread process on the transpose of the input graph, ${g}^{T}$ . This simulation process is performed as described in Section 2: we begin at a random node $u$ and proceed via depth-first search, where each encountered edge $e$ is traversed independently with probability ${p}_{e}$ . The set of nodes encountered becomes an edge in $\mathcal{H}$ . We then repeat this process, generating multiple hyper-edges. The BuildHypergraph subroutine takes as input a bound $R$ on its runtime; we continue building edges until a total of $R$ steps has been taken by the simulation process. (Note that the number of steps taken by the process is equal to the number of edges considered by the depth-first search process). Once $R$ steps have been taken in total over all simulations, we return the resulting hypergraph.

该算法在算法 1 中进行了正式描述，但让我们先非正式地描述一下我们的构造。我们的方法分两步进行。第一步，BuildHypergraph（构建超图），随机生成我们底层图 $g$ 的稀疏无向超图表示 $\mathcal{H}$ 。这是通过在输入图的转置 ${g}^{T}$ 上反复模拟影响力传播过程来完成的。这个模拟过程如第 2 节所述进行：我们从一个随机节点 $u$ 开始，通过深度优先搜索进行，其中每个遇到的边 $e$ 以概率 ${p}_{e}$ 独立遍历。遇到的节点集成为 $\mathcal{H}$ 中的一条边。然后我们重复这个过程，生成多条超边。BuildHypergraph 子程序将其运行时间的一个界限 $R$ 作为输入；我们继续构建边，直到模拟过程总共进行了 $R$ 步。（注意，该过程进行的步数等于深度优先搜索过程考虑的边数）。一旦所有模拟总共进行了 $R$ 步，我们就返回得到的超图。

<!-- Media -->

Algorithm 1 Maximize Influence

算法 1 最大化影响力

---

Require: Precision parameter $\epsilon  \in  \left( {0,1}\right)$ ,directed edge-

要求：精度参数 $\epsilon  \in  \left( {0,1}\right)$ ，有向边

		weighted graph $\mathcal{G}$ .

		带权图 $\mathcal{G}$ 。

		$R \leftarrow  {144}\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right)$

		$\mathcal{H} \leftarrow$ BuildHypergraph(R)

		$\mathcal{H} \leftarrow$ BuildHypergraph(R)

		return BuildSeedSet(H,k)

		return BuildSeedSet(H,k)

BuildHypergraph(R):

BuildHypergraph(R):

		Initialize $\mathcal{H} = \left( {V,\varnothing }\right)$ .

		初始化 $\mathcal{H} = \left( {V,\varnothing }\right)$ 。

		repeat

		重复

			Choose node $u$ from $\mathcal{G}$ uniformly at random.

			从 $\mathcal{G}$ 中均匀随机选择节点 $u$ 。

			Simulate influence spread,starting from $u$ ,in ${\mathcal{G}}^{T}$ .

			在 ${\mathcal{G}}^{T}$ 中从 $u$ 开始模拟影响力传播。

			Let $Z$ be the set of nodes discovered.

			令 $Z$ 为发现的节点集。

			Add $Z$ to the edge set of $\mathcal{H}$ .

			将$Z$添加到$\mathcal{H}$的边集。

		until $R$ steps have been taken in total by the

		直到模拟过程总共执行了$R$步

		simulation process.

		模拟过程。

		return $\mathcal{H}$

		返回$\mathcal{H}$

BuildSeedSet(H,k):

构建种子集（H，k）：

		for $i = 1,\ldots ,k$ do

		对于$i = 1,\ldots ,k$执行

			${v}_{i} \leftarrow  {\operatorname{argmax}}_{v}\left\{  {{de}{g}_{\mathcal{H}}\left( v\right) }\right\}$

			Remove ${v}_{i}$ and all incident edges from $\mathcal{H}$

			从$\mathcal{H}$中移除${v}_{i}$及其所有关联边

		return $\left\{  {{v}_{1},\ldots ,{v}_{k}}\right\}$

		返回$\left\{  {{v}_{1},\ldots ,{v}_{k}}\right\}$

---

<!-- Media -->

In the second step, BuildSeedSet, we use our hyper-graph representation to construct our output set. This is done by repeatedly choosing the node with highest degree in $\mathcal{H}$ ,then removing that node and all incident edges from $\mathcal{H}$ . The resulting set of $k$ nodes is the generated seed set.

在第二步“构建种子集”中，我们使用超图表示法来构建输出集。具体做法是，反复选择$\mathcal{H}$中度数最高的节点，然后从$\mathcal{H}$中移除该节点及其所有关联边。最终得到的$k$个节点的集合即为生成的种子集。

We now turn to provide a detailed analysis of Algorithm 1. Fix $k$ and a weighted directed graph $\mathcal{G}$ . Let $\operatorname{OPT} = \mathop{\max }\limits_{{S : \left| S\right|  = k}}\left\{  {{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack  }\right\}$ ,the maximum expected influence of a set of $k$ nodes.

现在我们来详细分析算法1。固定$k$和一个加权有向图$\mathcal{G}$。设$\operatorname{OPT} = \mathop{\max }\limits_{{S : \left| S\right|  = k}}\left\{  {{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack  }\right\}$为$k$个节点集合的最大期望影响力。

THEOREM 3.1. Given any $\epsilon  > 0$ ,Algorithm 1 returns a set $S$ with ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  \left( {1 - \frac{1}{e} - \epsilon }\right) \mathrm{{OPT}}$ ,with probability at least $3/5$ ,and runs in time $O\left( \frac{\left( {m + n}\right) \log \left( n\right) }{{\epsilon }^{3}}\right)$ .

定理3.1。给定任意$\epsilon  > 0$，算法1返回一个集合$S$，满足${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  \left( {1 - \frac{1}{e} - \epsilon }\right) \mathrm{{OPT}}$，概率至少为$3/5$，且运行时间为$O\left( \frac{\left( {m + n}\right) \log \left( n\right) }{{\epsilon }^{3}}\right)$。

The idea behind the proof of Theorem 3.1 is as follows. First, we observe that the influence of a set of nodes $S$ is precisely $n$ times the probability that a randomly selected node $u$ influences any node from $S$ in the transpose graph ${g}^{T}$ . Observation 1. For each subset of nodes $S \subseteq  \mathcal{G}$ , ${\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  {{I}_{g}\left( S\right) }\right\rbrack   = n\mathop{\Pr }\limits_{{u,g \sim  \mathcal{G}}}\left\lbrack  {S \cap  {C}_{{g}^{T}}\left( u\right)  \neq  \varnothing }\right\rbrack  .$

定理3.1证明背后的思路如下。首先，我们观察到节点集合$S$的影响力恰好是随机选择的节点$u$在转置图${g}^{T}$中影响$S$中任意节点的概率的$n$倍。观察1。对于节点的每个子集$S \subseteq  \mathcal{G}$，${\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  {{I}_{g}\left( S\right) }\right\rbrack   = n\mathop{\Pr }\limits_{{u,g \sim  \mathcal{G}}}\left\lbrack  {S \cap  {C}_{{g}^{T}}\left( u\right)  \neq  \varnothing }\right\rbrack  .$

Proof.

证明。

$$
{\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  {{I}_{g}\left( S\right) }\right\rbrack   = \mathop{\sum }\limits_{{u \in  g}}\mathop{\Pr }\limits_{{g \sim  \mathcal{G}}}\left\lbrack  {\exists v \in  S\text{ such that }u \in  {C}_{g}\left( v\right) }\right\rbrack  
$$

$$
 = \mathop{\sum }\limits_{{u \in  g}}\mathop{\Pr }\limits_{{g \sim  \mathcal{G}}}\left\lbrack  {\exists v \in  S\text{ such that }v \in  {C}_{{g}^{T}}\left( u\right) }\right\rbrack  
$$

$$
 = n\mathop{\Pr }\limits_{{u,g \sim  \mathcal{G}}}\left\lbrack  {\exists v \in  S\text{ such that }v \in  {C}_{{g}^{T}}\left( u\right) }\right\rbrack  
$$

$$
 = n\mathop{\Pr }\limits_{{u,g \sim  \mathcal{G}}}\left\lbrack  {S \cap  {C}_{{g}^{T}}\left( u\right)  \neq  \varnothing }\right\rbrack  .
$$

Observation 1 implies that we can estimate ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack$ by estimating the probability of the event $S \cap  {C}_{{g}^{T}}\left( u\right)  \neq$ $\varnothing$ . The degree of a node $v$ in $\mathcal{H}$ is precisely the number of times we observed that $v$ was influenced by a randomly selected node $u$ . We can therefore think of $\mathcal{H}$ as encoding an approximation to ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( \cdot \right) }\right\rbrack$ ,the influence function in graph $\mathcal{G}$ .

观察1表明，我们可以通过估计事件$S \cap  {C}_{{g}^{T}}\left( u\right)  \neq$ $\varnothing$的概率来估计${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack$。节点$v$在$\mathcal{H}$中的度数恰好是我们观察到$v$受随机选择的节点$u$影响的次数。因此，我们可以将$\mathcal{H}$视为对图$\mathcal{G}$中的影响力函数${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( \cdot \right) }\right\rbrack$的一种近似编码。

We now show that the algorithm takes enough samples to accurately estimate the influences of the nodes in the network. This requires two steps. First, we show that runtime $R = {144}\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right)$ is enough to build a sufficiently rich hypergraph structure, with high probability over the random outcomes of the influence cascade model. The idea behind the proof is to establish that $\mathrm{{OPT}} \cdot  \frac{m}{n}$ is an upper bound on the expected number of steps needed to build an edge, so it is unlikely that significantly fewer than $R/\left( {\mathrm{{OPT}} \cdot  \frac{m}{n}}\right)$ hyperedges have been completed after $R$ steps.

我们现在证明，该算法会采集足够多的样本，以准确估计网络中节点的影响力。这需要两个步骤。首先，我们证明运行时间 $R = {144}\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right)$ 足以构建一个足够丰富的超图结构，在影响级联模型的随机结果上具有高概率。证明背后的思路是确定 $\mathrm{{OPT}} \cdot  \frac{m}{n}$ 是构建一条边所需的期望步数的上界，因此在 $R$ 步之后完成的超边数量显著少于 $R/\left( {\mathrm{{OPT}} \cdot  \frac{m}{n}}\right)$ 的可能性不大。

LEMMA 3.1. Hypergraph H will contain at least $\frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ edges,with probability at least $\frac{2}{3}$ .

引理 3.1。超图 H 至少包含 $\frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ 条边的概率至少为 $\frac{2}{3}$。

Proof. Given a vertex $u$ and an edge $e = \left( {v,w}\right)$ ,consider the random event indicating whether edge $e$ is checked as part of the process of growing a depth-first search rooted at $u$ in the IC process corresponding to graph ${g}^{T} \sim  {\mathcal{G}}^{T}$ . Note that edge $e$ is checked if and only if node $v$ is influenced by node $u$ in this invocation of the IC process. In other words,edge $e = \left( {v,w}\right)$ is checked as part of the influence spread process on line 4 of BuildHypergraph if and only if $v \in  Z$ . Write ${m}_{{g}^{T}}\left( u\right)$ for the random variable indicating the number of edges that are checked as part of building the influence set $Z$ starting at node $u$ in ${g}^{T}$ .

证明。给定一个顶点 $u$ 和一条边 $e = \left( {v,w}\right)$，考虑一个随机事件，该事件指示在对应于图 ${g}^{T} \sim  {\mathcal{G}}^{T}$ 的独立级联（IC）过程中，以 $u$ 为根进行深度优先搜索扩展时，边 $e$ 是否被检查。注意，当且仅当在这次 IC 过程调用中节点 $v$ 受到节点 $u$ 的影响时，边 $e$ 才会被检查。换句话说，当且仅当 $v \in  Z$ 时，边 $e = \left( {v,w}\right)$ 会在 BuildHypergraph 算法第 4 行的影响传播过程中被检查。用 ${m}_{{g}^{T}}\left( u\right)$ 表示一个随机变量，它表示在 ${g}^{T}$ 中从节点 $u$ 开始构建影响集 $Z$ 时被检查的边的数量。

Let $X = \frac{{48n}\log \left( n\right) }{{OPT}{\epsilon }^{3}}$ for notational convenience. Consider the first (up to) $X$ iterations of the loop on lines 2-6 of BuildHypergraph. Note that $\mathcal{H}$ will have at least $X$ edges if the total runtime of the first $X$ iterations is at most $R$ . The expected runtime of the algorithm over these iterations is

为了符号表示方便，设 $X = \frac{{48n}\log \left( n\right) }{{OPT}{\epsilon }^{3}}$。考虑 BuildHypergraph 算法第 2 - 6 行循环的前（至多）$X$ 次迭代。注意，如果前 $X$ 次迭代的总运行时间至多为 $R$，那么 $\mathcal{H}$ 将至少有 $X$ 条边。算法在这些迭代上的期望运行时间为

$$
X \cdot  {\mathbb{E}}_{u,g \sim  \mathcal{G}}\left\lbrack  {1 + {m}_{{g}^{T}}\left( u\right) }\right\rbrack   = X + \frac{X}{n}{\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  {\mathop{\sum }\limits_{u}{m}_{{g}^{T}}\left( u\right) }\right\rbrack  
$$

$$
 = X + \frac{{48}\log \left( n\right) }{{OPT}{\epsilon }^{3}}{\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  {\mathop{\sum }\limits_{u}{m}_{{g}^{T}}\left( u\right) }\right\rbrack  
$$

$$
 = X + \frac{{48}\log \left( n\right) }{{OPT}{\epsilon }^{3}}\mathop{\sum }\limits_{{e = \left( {v,w}\right)  \in  {\mathcal{G}}^{T}}}{\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  \left| \left\{  {u : v \in  {C}_{{g}^{T}}\left( u\right) }\right\}  \right| \right\rbrack  
$$

$$
 = X + \frac{{48}\log \left( n\right) }{{OPT}{\epsilon }^{3}}\mathop{\sum }\limits_{{e = \left( {v,w}\right)  \in  {\mathcal{G}}^{T}}}{\mathbb{E}}_{g \sim  \mathcal{G}}\left\lbrack  \left| \left\{  {u : u \in  {C}_{g}\left( v\right) }\right\}  \right| \right\rbrack  
$$

$$
 \leq  \frac{{48n}\log \left( n\right) }{{\epsilon }^{3}} + \frac{{48}\log \left( n\right) }{{OPT}{\epsilon }^{3}}\mathop{\sum }\limits_{{e = \left( {v,w}\right)  \in  {\mathcal{G}}^{T}}}{OPT}
$$

$$
 = \frac{{48}\left( {m + n}\right) \log \left( n\right) }{{\epsilon }^{3}}.
$$

Here, the second equality (line 4 from above) follows by noting that an edge $\left( {v,w}\right)  \in  {\mathcal{G}}^{T}$ is traversed as part of ${m}_{{g}^{T}}\left( u\right)$ if and only if $v$ appears in ${C}_{{g}^{T}}\left( u\right)$ .

这里，第二个等式（从上面数第 4 行）是通过注意到当且仅当 $v$ 出现在 ${C}_{{g}^{T}}\left( u\right)$ 中时，边 $\left( {v,w}\right)  \in  {\mathcal{G}}^{T}$ 才会作为 ${m}_{{g}^{T}}\left( u\right)$ 的一部分被遍历得到的。

Thus, by the Markov inequality, the probability that the runtime over the first $X$ iterations is greater than $R = {144}\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right)$ is at most $\frac{1}{3}$ . The probability that at least $X$ edges are present in hypergraph $\mathcal{H}$ is therefore at least $\frac{2}{3}$ ,as required.

因此，根据马尔可夫不等式，前 $X$ 次迭代的运行时间大于 $R = {144}\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right)$ 的概率至多为 $\frac{1}{3}$。因此，超图 $\mathcal{H}$ 中至少存在 $X$ 条边的概率至少为 $\frac{2}{3}$，符合要求。

Next, we show that the resulting hypergraph is of sufficient size to estimate the influence of each set, up to an additive error that shrinks with $\epsilon$ ,with high probability. Write $m\left( \mathcal{H}\right)$ and ${\deg }_{\mathcal{H}}\left( S\right)$ for the number of edges of $\mathcal{H}$ and the number of edges from $\mathcal{H}$ incident with a node from $S$ ,respectively. Our approach is to apply concentration bounds to the random variable ${\deg }_{\mathcal{H}}\left( S\right)$ , which can be viewed as the sum of $m\left( \mathcal{H}\right)$ Bernoulli random variables. However, an important subtlety is that the $m\left( \mathcal{H}\right)$ is itself a random variable,determined by the stopping condition of BuildHypergraph. We must bound the correlation between $m\left( \mathcal{H}\right)$ and our influence estimation. We show that the value of $m\left( \mathcal{H}\right)$ is sufficiently concentrated that the resulting bias is insignificant.

接下来，我们证明所得到的超图具有足够的规模，能够以高概率估计每个集合的影响力，且加性误差会随着 $\epsilon$ 缩小。分别用 $m\left( \mathcal{H}\right)$ 和 ${\deg }_{\mathcal{H}}\left( S\right)$ 表示 $\mathcal{H}$ 的边数以及 $\mathcal{H}$ 中与 $S$ 中的节点相关联的边数。我们的方法是对随机变量 ${\deg }_{\mathcal{H}}\left( S\right)$ 应用集中不等式，该随机变量可视为 $m\left( \mathcal{H}\right)$ 个伯努利随机变量之和。然而，一个重要的微妙之处在于，$m\left( \mathcal{H}\right)$ 本身也是一个随机变量，它由构建超图（BuildHypergraph）的停止条件决定。我们必须对 $m\left( \mathcal{H}\right)$ 与我们的影响力估计之间的相关性进行界定。我们证明 $m\left( \mathcal{H}\right)$ 的值足够集中，使得由此产生的偏差可以忽略不计。

LEMMA 3.2. Suppose that $m\left( \mathcal{H}\right)  \geq  \frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ . Then,for any set of nodes $S \subseteq  V,\Pr \left\lbrack  {\left| {{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   - \frac{n}{m\left( \mathcal{H}\right) }{\deg }_{\mathcal{H}}\left( S\right) }\right|  > }\right.$ $\epsilon \mathrm{{OPT}}\rbrack  < \frac{1}{{n}^{3}}$ ,with probability taken over randomness in $\mathcal{H}$ .

引理3.2。假设 $m\left( \mathcal{H}\right)  \geq  \frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ 。那么，对于任意节点集合 $S \subseteq  V,\Pr \left\lbrack  {\left| {{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   - \frac{n}{m\left( \mathcal{H}\right) }{\deg }_{\mathcal{H}}\left( S\right) }\right|  > }\right.$ $\epsilon \mathrm{{OPT}}\rbrack  < \frac{1}{{n}^{3}}$ ，概率取自 $\mathcal{H}$ 中的随机性。

Proof. We will think of $\mathcal{H}$ as being built incrementally, one edge at a time, as in the execution of BuildHyper-graph. Since we will discuss the state of $\mathcal{H}$ at various points of execution,we will write $M$ for the number of edges in $\mathcal{H}$ when BuildHypergraph terminates; we think of $M$ as a random variable,corresponding to $m\left( \mathcal{H}\right)$ in the statement of the lemma. For a given $J \geq  \frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ , Let ${D}_{S}^{J}$ denote the degree of $S$ in $\mathcal{H}$ after $J$ edges have been added to $\mathcal{H}$ . We allow $J > M$ ,by considering the path of execution of BuildHypergraph had there not been a runtime bound. Thinking of ${D}_{S}^{J}$ as a random variable,we have that ${D}_{S}^{J}$ is the sum of $J$ identically distributed Bernoulli random variables each with probability ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack  /n \geq  {\epsilon OPT}/n$ ,by Observation 1. Our goal is to show that

证明。我们将像执行构建超图（BuildHyper - graph）算法那样，把 $\mathcal{H}$ 看作是一次添加一条边逐步构建起来的。由于我们将讨论 $\mathcal{H}$ 在执行过程中不同时刻的状态，我们用 $M$ 表示构建超图（BuildHypergraph）算法终止时 $\mathcal{H}$ 中的边数；我们将 $M$ 视为一个随机变量，对应于引理陈述中的 $m\left( \mathcal{H}\right)$ 。对于给定的 $J \geq  \frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ ，设 ${D}_{S}^{J}$ 表示在向 $\mathcal{H}$ 中添加了 $J$ 条边之后，$S$ 在 $\mathcal{H}$ 中的度数。通过考虑如果没有运行时间限制时构建超图（BuildHypergraph）算法的执行路径，我们允许 $J > M$ 。将 ${D}_{S}^{J}$ 视为一个随机变量，根据观察1，我们可知 ${D}_{S}^{J}$ 是 $J$ 个同分布的伯努利随机变量之和，每个变量的概率为 ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack  /n \geq  {\epsilon OPT}/n$ 。我们的目标是证明

(3.1)

$$
\Pr \left\lbrack  {\left| {{D}_{S}^{M} - \mathbb{E}\left\lbrack  {D}_{S}^{M}\right\rbrack  }\right|  > \frac{\epsilon  \cdot  M \cdot  {OPT}}{n}}\right\rbrack   < \frac{1}{{n}^{3}}.
$$

We will first establish that $\Pr \left\lbrack  {\left| {{D}_{S}^{J} - \mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack  }\right|  > \frac{\epsilon  \cdot  J \cdot  {OPT}}{n}}\right\rbrack   < \frac{1}{{n}^{12}}$ for a fixed $J$ . We consider two cases. First,suppose ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  {\epsilon OPT}$ . In this case,

我们首先证明对于固定的 $J$ ，有 $\Pr \left\lbrack  {\left| {{D}_{S}^{J} - \mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack  }\right|  > \frac{\epsilon  \cdot  J \cdot  {OPT}}{n}}\right\rbrack   < \frac{1}{{n}^{12}}$ 。我们考虑两种情况。首先，假设 ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  {\epsilon OPT}$ 。在这种情况下，

$$
\mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack   = {\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \cdot  \frac{J}{n} \geq  \frac{\epsilon  \cdot  J \cdot  {OPT}}{n} \geq  {48}{\epsilon }^{-2}\log n.
$$

The Multiplicative Chernoff bound (A.1) then implies that

乘法切尔诺夫界（A.1）则意味着

$$
\Pr \left\lbrack  {\left| {{D}_{S}^{J} - \mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack  }\right|  > \frac{\epsilon  \cdot  J \cdot  {OPT}}{n}}\right\rbrack  
$$

$$
 \leq  \Pr \left\lbrack  {\left| {{D}_{S}^{J} - \mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack  }\right|  < \epsilon \mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack  }\right\rbrack  
$$

$$
 < {e}^{-\mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack  {\epsilon }^{2}/4} < {e}^{-{12}\log \left( n\right) } = \frac{1}{{n}^{12}}.
$$

Next suppose ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   < {\epsilon OPT}$ ,so $\mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack   < \epsilon  \cdot  {OPT}$ . $J/n$ . In this case,we have ${D}_{S}^{J} \geq  \mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack   - \frac{\epsilon  \cdot  J \cdot  {OPT}}{n}$ surely, and the Multiplicative Chernoff bound (A.1) implies that

接下来假设 ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   < {\epsilon OPT}$，因此 $\mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack   < \epsilon  \cdot  {OPT}$。$J/n$。在这种情况下，我们肯定有 ${D}_{S}^{J} \geq  \mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack   - \frac{\epsilon  \cdot  J \cdot  {OPT}}{n}$，并且乘法切尔诺夫界（A.1）表明

$$
\Pr \left\lbrack  {{D}_{S}^{J} > \mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack   + \frac{\epsilon  \cdot  J \cdot  {OPT}}{n}}\right\rbrack  
$$

$$
 = \Pr \left\lbrack  {{D}_{S}^{J} > \mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack  \left( {1 + \frac{\epsilon  \cdot  J \cdot  {OPT}}{n\mathbb{E}\left\lbrack  {D}_{S}^{J}\right\rbrack  }}\right) }\right\rbrack  
$$

$$
 < {e}^{-\frac{\epsilon  \cdot  J \cdot  {OPT}}{2n}} < {e}^{-{12}\log \left( n\right) } \leq  \frac{1}{{n}^{12}}.
$$

Thus, in all cases, the probability that the event of interest occurs is at most $\frac{1}{{n}^{12}}$ .

因此，在所有情况下，感兴趣的事件发生的概率至多为 $\frac{1}{{n}^{12}}$。

To complete the proof, we must show that the result holds for $J = M$ ,noting that $M$ is a random variable that is correlated with the event in (3.1). Suppose first that $\epsilon  > {n}^{-4/3}$ . Then since we know $M \leq  R =$ ${144}\left( {m + n}\right) \log \left( n\right) /{\epsilon }^{3} = o\left( {n}^{7}\right)$ ,we can take the union bound over all $o\left( {n}^{7}\right)$ values of $J$ lying between $\frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ and $R$ to obtain (3.1),as required.

为了完成证明，我们必须证明该结果对于 $J = M$ 成立，注意到 $M$ 是一个与（3.1）中的事件相关的随机变量。首先假设 $\epsilon  > {n}^{-4/3}$。那么由于我们知道 $M \leq  R =$ ${144}\left( {m + n}\right) \log \left( n\right) /{\epsilon }^{3} = o\left( {n}^{7}\right)$，我们可以对介于 $\frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ 和 $R$ 之间的 $J$ 的所有 $o\left( {n}^{7}\right)$ 个值取联合界，以得到所需的（3.1）。

Next suppose $\epsilon  < {n}^{-4/3}$ . We claim that $M$ is concentrated about its expectation.

接下来假设 $\epsilon  < {n}^{-4/3}$。我们断言 $M$ 集中在其期望值附近。

CLAIM 1. If $\epsilon  < {n}^{-4/3}$ ,then there exist ${J}_{1},{J}_{2} \geq$ $\frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ ,with ${J}_{1} \leq  {J}_{2} \leq  \left( {1 + \epsilon /2}\right) {J}_{1}$ ,such that $\Pr \left\lbrack  {M \in  \left\lbrack  {{J}_{1},{J}_{2}}\right\rbrack  }\right\rbrack   > 1 - \frac{1}{{n}^{4}}$ . Proof. Let $\mu$ be the expected size of an edge generated by BuildHypergraph,and let $\widetilde{M} = R/\mu$ . We claim that,with probability at least $1 - \frac{1}{{n}^{4}},M \in  (1 \pm$ $\left. \frac{\epsilon }{4}\right) \widetilde{M}$ ; bounding this range from below by $\frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ then completes the claim.

断言 1. 如果 $\epsilon  < {n}^{-4/3}$，那么存在 ${J}_{1},{J}_{2} \geq$ $\frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$，其中 ${J}_{1} \leq  {J}_{2} \leq  \left( {1 + \epsilon /2}\right) {J}_{1}$，使得 $\Pr \left\lbrack  {M \in  \left\lbrack  {{J}_{1},{J}_{2}}\right\rbrack  }\right\rbrack   > 1 - \frac{1}{{n}^{4}}$。证明。设 $\mu$ 为通过构建超图生成的一条边的期望大小，并设 $\widetilde{M} = R/\mu$。我们断言，概率至少为 $1 - \frac{1}{{n}^{4}},M \in  (1 \pm$ $\left. \frac{\epsilon }{4}\right) \widetilde{M}$；然后用 $\frac{{48n}\log \left( n\right) }{\mathrm{{OPT}}{\epsilon }^{3}}$ 从下方界定这个范围就完成了该断言。

Let $X$ denote the sum of the sizes of the first $\left( {1 - \frac{\epsilon }{4}}\right) \widetilde{M}$ edges of $\mathcal{H}$ . Since each edge has size at most $n$ ,and $\mathbb{E}\left\lbrack  X\right\rbrack   = \left( {1 - \frac{\epsilon }{4}}\right) \widetilde{M} \cdot  \mu  = \left( {1 - \frac{\epsilon }{4}}\right) R$ ,the Hoeffding bound implies

设 $X$ 表示 $\mathcal{H}$ 的前 $\left( {1 - \frac{\epsilon }{4}}\right) \widetilde{M}$ 条边的大小之和。由于每条边的大小至多为 $n$，并且 $\mathbb{E}\left\lbrack  X\right\rbrack   = \left( {1 - \frac{\epsilon }{4}}\right) \widetilde{M} \cdot  \mu  = \left( {1 - \frac{\epsilon }{4}}\right) R$，霍夫丁界表明

$$
\Pr \left\lbrack  {X > R}\right\rbrack   \leq  \Pr \left\lbrack  {\left| {X - \mathbb{E}\left\lbrack  X\right\rbrack  }\right|  > \frac{\epsilon R}{4}}\right\rbrack  
$$

$$
 < 2{e}^{-\frac{{R}^{2}{\epsilon }^{2}}{{16}{n}^{2}\left( {1 - \epsilon }\right) R/\mu }}
$$

$$
 < 2{e}^{-\frac{R{\epsilon }^{2}}{{16}{n}^{2}}} < 2{e}^{-\log n/{16\epsilon n}} < \frac{2}{{n}^{6}}.
$$

We therefore have $X < R$ ,which implies $M > \left( {1 - \frac{\epsilon }{4}}\right) \widetilde{M}$ , with probability at least $1 - \frac{2}{{n}^{6}}$ . A similar argument yields $M < \left( {1 + \frac{\epsilon }{4}}\right) \widetilde{M}$ with probability at least $1 - \frac{2}{{n}^{6}}$ . We conclude (via union bound) that $M \in  \left( {1 \pm  \frac{\epsilon }{4}}\right) \widetilde{M}$ with probability at least $1 - \frac{4}{{n}^{6}}$ ,which proves the claim.

因此，我们有$X < R$，这意味着$M > \left( {1 - \frac{\epsilon }{4}}\right) \widetilde{M}$，且概率至少为$1 - \frac{2}{{n}^{6}}$。类似的论证可得，$M < \left( {1 + \frac{\epsilon }{4}}\right) \widetilde{M}$成立的概率至少为$1 - \frac{2}{{n}^{6}}$。我们（通过联合界）得出结论：$M \in  \left( {1 \pm  \frac{\epsilon }{4}}\right) \widetilde{M}$成立的概率至少为$1 - \frac{4}{{n}^{6}}$，这就证明了该命题。

Condition on the event that $M \in  \left\lbrack  {{J}_{1},{J}_{2}}\right\rbrack$ ,which has probability at least $1 - \frac{1}{{n}^{4}}$ . We know that with probability at least $1 - \frac{1}{{n}^{4}},\left| {{D}_{S}^{{J}_{2}} - \mathbb{E}\left\lbrack  {D}_{S}^{{J}_{2}}\right\rbrack  }\right|  < \frac{\epsilon  \cdot  {J}_{2} \cdot  {OPT}}{2n}$ ; we will condition on this event as well. Since ${D}_{S}^{M}$ is dominated by ${D}_{S}^{{J}_{2}}$ ,and since ${J}_{2} < \left( {1 + \epsilon /2}\right) {J}_{1} \leq$ $\left( {1 + \epsilon /2}\right) M$ ,we conclude that

假设事件$M \in  \left\lbrack  {{J}_{1},{J}_{2}}\right\rbrack$发生，该事件发生的概率至少为$1 - \frac{1}{{n}^{4}}$。我们知道，$1 - \frac{1}{{n}^{4}},\left| {{D}_{S}^{{J}_{2}} - \mathbb{E}\left\lbrack  {D}_{S}^{{J}_{2}}\right\rbrack  }\right|  < \frac{\epsilon  \cdot  {J}_{2} \cdot  {OPT}}{2n}$成立的概率至少为$1 - \frac{1}{{n}^{4}},\left| {{D}_{S}^{{J}_{2}} - \mathbb{E}\left\lbrack  {D}_{S}^{{J}_{2}}\right\rbrack  }\right|  < \frac{\epsilon  \cdot  {J}_{2} \cdot  {OPT}}{2n}$；我们也将基于此事件进行条件分析。由于${D}_{S}^{M}$受${D}_{S}^{{J}_{2}}$支配，并且因为${J}_{2} < \left( {1 + \epsilon /2}\right) {J}_{1} \leq$ $\left( {1 + \epsilon /2}\right) M$，我们得出结论：

$$
{D}_{S}^{M} \leq  {D}_{S}^{{J}_{2}} < \mathbb{E}\left\lbrack  {D}_{S}^{{J}_{2}}\right\rbrack   + \frac{\epsilon OPT}{2n}{J}_{2}
$$

$$
 = \mathbb{E}\left\lbrack  {D}_{S}\right\rbrack   \cdot  \frac{{J}_{2}}{M} + \frac{\epsilon OPT}{2n}{J}_{2}
$$

$$
 < \mathbb{E}\left\lbrack  {D}_{S}\right\rbrack   + \frac{\epsilon }{2}\mathbb{E}\left\lbrack  {D}_{S}\right\rbrack   + \frac{\epsilon OPT}{2n}M
$$

$$
 < \mathbb{E}\left\lbrack  {D}_{S}\right\rbrack   + \frac{\epsilon OPT}{n}M
$$

as required. Similarly,the fact that ${D}_{S}$ dominates ${D}_{S}^{{J}_{1}}$ implies ${D}_{S} > \mathbb{E}\left\lbrack  {D}_{S}\right\rbrack   - \frac{\epsilon OPT}{n}M$ ,conditioning on an event with probability at least $1 - \frac{1}{{n}^{4}}$ ,as required. Taking the union bound over the complement of the conditioned events, we conclude that the unconditional probability of (3.1) is at most $\frac{3}{{n}^{4}} < \frac{1}{{n}^{3}}$ as required.

如所需。类似地，${D}_{S}$支配${D}_{S}^{{J}_{1}}$这一事实意味着，在一个概率至少为$1 - \frac{1}{{n}^{4}}$的事件条件下，${D}_{S} > \mathbb{E}\left\lbrack  {D}_{S}\right\rbrack   - \frac{\epsilon OPT}{n}M$成立，如所需。对条件事件的补集取联合界，我们得出结论：(3.1)的无条件概率至多为$\frac{3}{{n}^{4}} < \frac{1}{{n}^{3}}$，如所需。

Finally, we must show that the greedy algorithm applied to $\mathcal{H}$ in BuildSeedSet returns a good approximation to the original optimization problem. Recall that, in general, the greedy algorithm for submodular function maximization proceeds by repeatedly selecting the singleton with maximal contribution to the function value, up to the cardinality constraint. The following lemma shows that if one submodular function is approximated sufficiently well by a distribution of sub-modular functions, then applying the greedy algorithm to a function drawn from the distribution yields a good approximation with respect to the original.

最后，我们必须证明，在BuildSeedSet中对$\mathcal{H}$应用贪心算法能得到原优化问题的一个良好近似解。回顾一下，一般来说，子模函数最大化的贪心算法是通过反复选择对函数值贡献最大的单元素集，直到达到基数约束为止。以下引理表明，如果一个子模函数能被子模函数的一个分布充分近似，那么对从该分布中抽取的一个函数应用贪心算法，相对于原函数能得到一个良好的近似解。

LEMMA 3.3. Choose $\delta  > 0$ and suppose that $f : {2}^{V} \rightarrow$ ${\mathbb{R}}_{ \geq  0}$ is a non-decreasing submodular function. Let $D$ be a distribution over non-decreasing submodular functions with the property that,for all sets $S$ with $\left| S\right|  \leq  k$ , $\mathop{\Pr }\limits_{{\widehat{f} \sim  D}}\left\lbrack  {\left| {f\left( S\right)  - \widehat{f}\left( S\right) }\right|  > \delta }\right\rbrack   < 1/{n}^{3}$ . If we write ${S}_{\widehat{f}}$ for the set returned by the greedy algorithm on input $\widehat{f}$ ,then

引理3.3。选择$\delta  > 0$，并假设$f : {2}^{V} \rightarrow$ ${\mathbb{R}}_{ \geq  0}$是一个非递减的子模函数。设$D$是一个关于非递减子模函数的分布，其性质为：对于所有满足$\left| S\right|  \leq  k$的集合$S$，有$\mathop{\Pr }\limits_{{\widehat{f} \sim  D}}\left\lbrack  {\left| {f\left( S\right)  - \widehat{f}\left( S\right) }\right|  > \delta }\right\rbrack   < 1/{n}^{3}$。如果我们用${S}_{\widehat{f}}$表示贪心算法在输入$\widehat{f}$时返回的集合，那么

$$
\mathop{\Pr }\limits_{{\widehat{f} \sim  D}}\left\lbrack  {f\left( {S}_{\widehat{f}}\right)  < \left( {1 - 1/e}\right) \left( {\mathop{\max }\limits_{{S : \left| S\right|  = k}}f\left( S\right) }\right)  - {2\delta }}\right\rbrack   < 1/n.
$$

Proof. Lemma 3.3 Choose ${S}^{ * } \in  {\operatorname{argmax}}_{\left| S\right|  = k}\{ f\left( S\right) \}$ . With probability at least $1 - 1/{n}^{3},\widehat{f}\left( {S}^{ * }\right)  \geq  f\left( {S}^{ * }\right)  - \delta$ . So,in particular, $\mathop{\max }\limits_{{\left| S\right|  = k}}\widehat{f}\left( S\right)  \geq  f\left( {S}^{ * }\right)  - \delta$ .

证明。引理3.3 选择${S}^{ * } \in  {\operatorname{argmax}}_{\left| S\right|  = k}\{ f\left( S\right) \}$ 。概率至少为$1 - 1/{n}^{3},\widehat{f}\left( {S}^{ * }\right)  \geq  f\left( {S}^{ * }\right)  - \delta$ 。因此，特别地，$\mathop{\max }\limits_{{\left| S\right|  = k}}\widehat{f}\left( S\right)  \geq  f\left( {S}^{ * }\right)  - \delta$ 。

We run the greedy algorithm on function $\widehat{f}$ ; let ${S}_{i}$ be the set of nodes selected up to and including iteration $i$ (with ${S}_{0} = \varnothing$ ). On iteration $i$ ,we consider each set of the form ${S}_{i - 1} \cup  \{ x\}$ where $x$ is a singleton. There are at most $n$ of these sets,and hence the union bound implies that $f$ and $\widehat{f}$ differ by at most $\delta$ on each of these sets,with probability at least $1 - 1/{n}^{2}$ . In particular, $\left| {f\left( {S}_{i}\right)  - \widehat{f}\left( {S}_{i}\right) }\right|  < \delta$ . Taking the union bound over all iterations,we have that $\left| {f\left( {S}_{k}\right)  - \widehat{f}\left( {S}_{k}\right) }\right|  < \delta$ with probability at least $1 - 1/n$ . We therefore have

我们对函数$\widehat{f}$ 运行贪心算法；设${S}_{i}$ 为直至并包括第$i$ 次迭代（其中${S}_{0} = \varnothing$ ）所选择的节点集合。在第$i$ 次迭代中，我们考虑形如${S}_{i - 1} \cup  \{ x\}$ 的每个集合，其中$x$ 是单元素集。这些集合最多有$n$ 个，因此并集界意味着在这些集合中的每一个上，$f$ 和$\widehat{f}$ 的差值最多为$\delta$ ，概率至少为$1 - 1/{n}^{2}$ 。特别地，$\left| {f\left( {S}_{i}\right)  - \widehat{f}\left( {S}_{i}\right) }\right|  < \delta$ 。对所有迭代取并集界，我们有概率至少为$1 - 1/n$ 时$\left| {f\left( {S}_{k}\right)  - \widehat{f}\left( {S}_{k}\right) }\right|  < \delta$ 。因此我们有

$$
f\left( {S}_{k}\right)  \geq  \widehat{f}\left( {S}_{k}\right)  - \delta 
$$

$$
 \geq  \left( {1 - 1/e}\right) \mathop{\max }\limits_{S}\widehat{f}\left( S\right)  - \delta 
$$

$$
 \geq  \left( {1 - 1/e}\right) f\left( {S}^{ * }\right)  - {2\delta }
$$

conditioning on an event of probability $1 - 1/n$ .

在概率为$1 - 1/n$ 的事件条件下。

We are now ready to complete our proof of Theorem 3.1.

我们现在准备完成定理3.1的证明。

Proof. [Proof of Theorem 3.1] Lemma 3.1 and 3.2 together imply that, conditioning on an event of probability at least $3/5$ ,we will have

证明。[定理3.1的证明] 引理3.1和3.2共同表明，在概率至少为$3/5$ 的事件条件下，我们将有

$$
\Pr \left\lbrack  {\left| {{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   - \frac{n \cdot  {\deg }_{\mathcal{H}}\left( S\right) }{m\left( \mathcal{H}\right) }}\right|  > \epsilon \mathrm{{OPT}}}\right\rbrack   < \frac{1}{{n}^{3}}
$$

for each $S \subseteq  V$ . We then apply Lemma 3.3 with $f\left( S\right)  \mathrel{\text{:=}} {\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack  ,\widehat{f}\left( S\right)  \mathrel{\text{:=}} \frac{n \cdot  {\deg }_{\mathcal{H}}\left( S\right) }{m\left( \mathcal{H}\right) }$ (drawn from distribution corresponding to distribution of $\mathcal{H}$ returned by BuildHypergraph),and $\delta  = \epsilon \mathrm{{OPT}}$ . Lemma 3.3 implies that,with probability at least $1 - \frac{1}{n}$ ,the greedy algorithm applied to $\mathcal{H}$ returns a set $S$ with ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq$ $\left( {1 - 1/e}\right) \mathrm{{OPT}} - {2\epsilon }\mathrm{{OPT}} = \left( {1 - 1/e - {2\epsilon }}\right) \mathrm{{OPT}}$ . Noting that this is precisely the set returned by BuildSeedSet gives the desired bound on the approximation factor (rescaling $\epsilon$ by a factor of 2). Thus the claim holds with probability at least $2/3 - 1/n \geq  3/5$ (for $n \geq  {20}$ ).

对于每个$S \subseteq  V$ 。然后我们使用$f\left( S\right)  \mathrel{\text{:=}} {\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack  ,\widehat{f}\left( S\right)  \mathrel{\text{:=}} \frac{n \cdot  {\deg }_{\mathcal{H}}\left( S\right) }{m\left( \mathcal{H}\right) }$ （从与BuildHypergraph返回的$\mathcal{H}$ 的分布相对应的分布中抽取）和$\delta  = \epsilon \mathrm{{OPT}}$ 应用引理3.3。引理3.3表明，概率至少为$1 - \frac{1}{n}$ 时，对$\mathcal{H}$ 应用贪心算法返回一个集合$S$ ，满足${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq$ $\left( {1 - 1/e}\right) \mathrm{{OPT}} - {2\epsilon }\mathrm{{OPT}} = \left( {1 - 1/e - {2\epsilon }}\right) \mathrm{{OPT}}$ 。注意到这正是BuildSeedSet返回的集合，这就给出了近似因子的期望界（将$\epsilon$ 缩放2倍）。因此，对于$n \geq  {20}$ ，该命题成立的概率至少为$2/3 - 1/n \geq  3/5$ 。

Finally, we argue that our algorithm can be implemented in the appropriate runtime. The fact that Build-Hypergraph executes in the required time follows from the explicit bound on its runtime. For BuildSeedSet, we will maintain a list of vertices sorted by their degree in $\mathcal{H}$ ; this will allow us to repeatedly select the maximum-degree node in constant time. The initial sort takes time $O\left( {n\log n}\right)$ . We must bound the time needed to remove an edge from $\mathcal{H}$ and correspondingly update the sorted list. We will implement the sorted list as a doubly linked list of groups of vertices, where each group itself is implemented as a doubly linked list containing all vertices of a given degree (with only non-empty groups present). Each edge of $\mathcal{H}$ will maintain a list of pointers to its vertices. When an edge is removed, the degree of each vertex in the edge decreases by 1 ; we modify the list by shifting any decremented vertex to the preceding group (creating new groups and removing empty groups as necessary). Removing an edge from $\mathcal{H}$ and updating the sorted list therefore takes time proportional to the size of the edge. Since each edge in $\mathcal{H}$ can be removed at most once over all iterations of BuildSeedSet, the total runtime is at most the sum of node degrees in $\mathcal{H}$ ,which is at most $R = O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right) }\right)$ .

最后，我们认为我们的算法可以在合适的运行时间内实现。Build - Hypergraph（构建超图）能在所需时间内执行这一事实，源于其运行时间的显式边界。对于BuildSeedSet（构建种子集），我们将维护一个按顶点在$\mathcal{H}$中的度数排序的顶点列表；这将使我们能够在常数时间内反复选择最大度数的节点。初始排序需要$O\left( {n\log n}\right)$的时间。我们必须界定从$\mathcal{H}$中移除一条边并相应更新排序列表所需的时间。我们将把排序列表实现为顶点组的双向链表，其中每个组本身实现为一个双向链表，包含给定度数的所有顶点（仅存在非空组）。$\mathcal{H}$的每条边将维护一个指向其顶点的指针列表。当移除一条边时，该边中每个顶点的度数减1；我们通过将任何度数减少的顶点移到前一个组来修改列表（必要时创建新组并移除空组）。因此，从$\mathcal{H}$中移除一条边并更新排序列表所需的时间与边的大小成正比。由于在BuildSeedSet的所有迭代中，$\mathcal{H}$中的每条边最多只能被移除一次，所以总运行时间最多为$\mathcal{H}$中节点度数之和，即最多为$R = O\left( {\left( {m + n}\right) {\epsilon }^{-3}\log \left( n\right) }\right)$。

3.1 Amplifying the Success Probability Algorithm 1 returns a set of influence at least $\left( {1 - \frac{1}{e} - \epsilon }\right)$ with probability at least $3/5$ . The failure probability is due to Lemma 3.1: hypergraph $\mathcal{H}$ may not have sufficiently many edges after $R$ steps have been taken by the simulation process in line 4 of the BuildHypergraph subprocedure. However, note that this failure condition is detectable via repetition: we can repeat Algorithm 1 multiple times, and use only the iteration that generates the most edges. The success rate can then be improved by repeated invocation, up to a maximum of $1 - 1/n$ with $\log \left( n\right)$ repetitions (at which point the error probability due to Lemma 3.2 becomes dominant).

3.1 提高成功概率 算法1以至少$3/5$的概率返回一个影响力至少为$\left( {1 - \frac{1}{e} - \epsilon }\right)$的集合。失败概率源于引理3.1：在BuildHypergraph子过程的第4行的模拟过程执行$R$步之后，超图$\mathcal{H}$可能没有足够多的边。然而，请注意，这种失败条件可以通过重复检测：我们可以多次重复算法1，并且只使用生成最多边的那次迭代。然后可以通过重复调用提高成功率，最多达到$1 - 1/n$，需要$\log \left( n\right)$次重复（此时，由于引理3.2导致的错误概率将占主导地位）。

We next note that,for any $\ell  > 1$ ,the error bound in Lemma 3.2 can be improved to $\frac{1}{{n}^{\ell }}$ ,by increasing the value of $R$ by a factor of $\ell$ ,since this error derives from Chernoff bounds. This would allow the success rate of the algorithm to be improved up to a maximum of $1 - \frac{1}{{n}^{\ell }}$ by further repeated invocation. To summarize, the error rate of the algorithm can be improved to $1 - \frac{1}{{n}^{\ell }}$ for any $\ell$ ,at the cost of increasing the runtime of the algorithm by a factor of ${\ell }^{2}\log \left( n\right)$ .

接下来我们注意到，对于任何$\ell  > 1$，通过将$R$的值增加$\ell$倍，引理3.2中的误差界可以提高到$\frac{1}{{n}^{\ell }}$，因为这个误差源于切尔诺夫界（Chernoff bounds）。这将允许通过进一步重复调用将算法的成功率提高到最多$1 - \frac{1}{{n}^{\ell }}$。综上所述，对于任何$\ell$，算法的错误率可以提高到$1 - \frac{1}{{n}^{\ell }}$，代价是将算法的运行时间增加${\ell }^{2}\log \left( n\right)$倍。

## 4 Approximate Influence Maximization in Sublinear Time

## 4 亚线性时间内的近似影响力最大化

We now describe a modified algorithm that provides a tradeoff between runtime and approximation quality. For an an arbitrary $\beta  < 1$ ,our algorithm will obtain an $O\left( \beta \right)$ -approximation to the influence maximization problem,in time $O\left( {\beta \left( {n + m}\right) \log \left( n\right) }\right)$ ,with probability at least $3/5$ . In Section 4.1 we describe an implementation of this algorithm that supports termination after an arbitrary number of steps, rather than being given the value of $\beta$ in advance.

我们现在描述一种改进的算法，该算法在运行时间和近似质量之间进行权衡。对于任意的$\beta  < 1$，我们的算法将在$O\left( {\beta \left( {n + m}\right) \log \left( n\right) }\right)$的时间内，以至少$3/5$的概率获得影响力最大化问题的$O\left( \beta \right)$ - 近似解。在4.1节中，我们描述了该算法的一种实现，它支持在任意步数后终止，而不是预先给定$\beta$的值。

Our algorithm is listed as Algorithm 2. The intuition behind our construction is as follows. We wish to find a set of nodes with high expected influence. One approach would be to apply Algorithm 1 and simply impose a tighter constraint on the amount of time that can be used to construct hypergraph $\mathcal{H}$ . This might correspond to reducing the value of parameter $R$ by,say, a factor of $\beta$ . Unfortunately,the precision of our sampling method does not always degrade gracefully with fewer samples: if $\beta$ is sufficiently small,we may not have enough data to guess at a maximum-influence node (even if we allow ourselves a factor of $\beta$ in the approximation ratio). In these cases, the sampling approach fails to provide a good approximation.

我们的算法列为算法2。我们构造背后的直觉如下。我们希望找到一组具有高预期影响力的节点。一种方法是应用算法1，并简单地对构建超图 $\mathcal{H}$ 所能使用的时间量施加更严格的约束。这可能对应于将参数 $R$ 的值降低，例如，降低 $\beta$ 倍。不幸的是，我们的采样方法的精度并不总是随着样本数量的减少而平稳下降：如果 $\beta$ 足够小，我们可能没有足够的数据来猜测最大影响力节点（即使我们允许自己在近似比率上有 $\beta$ 倍的误差）。在这些情况下，采样方法无法提供良好的近似。

However, as we will show, our sampling fails precisely because many of the edges in our hypergraph construction were large, and (with constant probability) this can occur only if many of the nodes that make up those edges have high influence. In this case, we could proceed by selecting a node from the hypergraph at random, with probability proportional to its hyper-graph degree. We prove that this procedure is likely to return a node of very high influence precisely in settings where the original sampling approach would fail.

然而，正如我们将展示的，我们的采样失败恰恰是因为我们构建的超图中的许多边都很大，并且（以恒定概率）只有当构成这些边的许多节点具有高影响力时才会发生这种情况。在这种情况下，我们可以通过从超图中随机选择一个节点来继续，选择概率与其超图度成正比。我们证明，正是在原始采样方法会失败的情况下，这个过程很可能返回一个具有非常高影响力的节点。

If $k > 1$ ,we can combine these two approaches by returning a union of vertices selected according to each procedure. If $k = 1$ ,we must choose which approach to apply. However, in this case, there is a simple way to determine whether we have obtained enough samples that BuildSeedSet returns an acceptable solution: check whether the maximum degree in the hypergraph is sufficiently high.

如果 $k > 1$ ，我们可以通过返回根据每个过程选择的顶点的并集来结合这两种方法。如果 $k = 1$ ，我们必须选择应用哪种方法。然而，在这种情况下，有一种简单的方法来确定我们是否获得了足够的样本，使得BuildSeedSet返回一个可接受的解决方案：检查超图中的最大度是否足够高。

THEOREM 4.1. For any $\beta  < 1$ ,Algorithm 2 returns, with probability of at least $3/5$ ,a node with expected influence at least $\min \{ \frac{1}{4},\beta \}  \cdot  {OPT}$ . Its runtime is $O\left( {\beta \left( {n + m}\right) \log \left( n\right) }\right)$ .

定理4.1。对于任何 $\beta  < 1$ ，算法2以至少 $3/5$ 的概率返回一个预期影响力至少为 $\min \{ \frac{1}{4},\beta \}  \cdot  {OPT}$ 的节点。其运行时间为 $O\left( {\beta \left( {n + m}\right) \log \left( n\right) }\right)$ 。

Our proof of Theorem 4.1 proceeds via two cases, depending on whether $\mathcal{H}$ has many or few edges as a function of OPT. The precise number of edges we require involves the constant $C = {48} \cdot  {6}^{3}$ (from the definition of Algorithm 2), which we have not tried to optimize. We first show that,subject to $\mathcal{H}$ having many edges,set $S$ from line 5 or 8 (corresponding to $k > 1$ and $k = 1$ ,respectively) is likely to have high influence. This follows the analysis from Theorem 3.1.

我们对定理4.1的证明通过两种情况进行，具体取决于 $\mathcal{H}$ 作为最优解（OPT）的函数时边的数量是多还是少。我们所需的精确边数涉及常数 $C = {48} \cdot  {6}^{3}$ （来自算法2的定义），我们尚未尝试对其进行优化。我们首先证明，在 $\mathcal{H}$ 有许多边的条件下，第5行或第8行的集合 $S$ （分别对应于 $k > 1$ 和 $k = 1$ ）可能具有高影响力。这遵循定理3.1的分析。

<!-- Media -->

Algorithm 2 Runtime-Flexible Influence Maximization

算法2 运行时间灵活的影响力最大化

---

Define: $C = {48} \cdot  {6}^{3}$

定义： $C = {48} \cdot  {6}^{3}$

Require: Approximation parameter $\beta  < 1$ ,directed

要求：近似参数 $\beta  < 1$ ，有向

		weighted graph $\mathcal{G}$ .

		加权图 $\mathcal{G}$ 。

		$R \leftarrow  \beta  \cdot  {144C} \cdot  \left( {n + m}\right) \log \left( n\right)$

		$\mathcal{H} \leftarrow$ BuildHypergraph(R)

		$\mathcal{H} \leftarrow$ 构建超图(R)

	Choose $v \in  V$ with probability proportional to

	以与 $v \in  V$ 中的度成正比的概率选择 $v \in  V$

		degree in $\mathcal{H}$

		在 $\mathcal{H}$ 中的度

		if $k > 1$ then

			如果 $k > 1$ 则

			$S \leftarrow  \operatorname{BuildSeedSet}\left( {\mathcal{H},k - 1}\right)$

			return $S \cup  \{ v\}$

					返回 $S \cup  \{ v\}$

		else

			否则

			$S \leftarrow  \operatorname{BuildSeedSet}\left( {\mathcal{H},1}\right)$

			if $\mathop{\max }\limits_{u}\left\{  {{\deg }_{\mathcal{H}}\left( u\right) }\right\}   > {2C}\log n$ then return $S$

					如果 $\mathop{\max }\limits_{u}\left\{  {{\deg }_{\mathcal{H}}\left( u\right) }\right\}   > {2C}\log n$ 则返回 $S$

			else return $\{ v\}$

			否则返回 $\{ v\}$

---

<!-- Media -->

LEMMA 4.1. Suppose that $m\left( \mathcal{H}\right)  \geq  \frac{{Cn}\log \left( n\right) }{\mathrm{{OPT}}}$ . Then, with probability at least $1 - \frac{1}{n}$ ,set $S$ satisfies ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq$ $\frac{1}{4}\mathrm{{OPT}}$ ,with probability taken over randomness in $\mathcal{H}$ .

引理4.1. 假设 $m\left( \mathcal{H}\right)  \geq  \frac{{Cn}\log \left( n\right) }{\mathrm{{OPT}}}$ 。那么，至少以 $1 - \frac{1}{n}$ 的概率，集合 $S$ 满足 ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq$ $\frac{1}{4}\mathrm{{OPT}}$ ，这里的概率是关于 $\mathcal{H}$ 中的随机性而言的。

Proof. Recall $C = {48} \cdot  {6}^{3}$ . First suppose $k = 1$ ,so $S$ is as defined on line 8 . If we apply Lemma 3.2 with $\epsilon  = \frac{1}{6}$ , followed by the analysis of BuildSeedSet(H,k)from the proof of Theorem 3.1,we get that ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  \frac{1}{2}\mathrm{{OPT}}$ , as required. For $k > 1$ ,applying Lemma 3.2 with $\epsilon  = \frac{1}{6}$ yields instead that ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  \frac{1}{2}{\mathrm{{OPT}}}_{k - 1}$ ,where ${\mathrm{{OPT}}}_{k - 1}$ is the maximum influence over sets of size at most $k - 1$ . But now,by submodularity, $\frac{1}{2}{\mathrm{{OPT}}}_{k - 1} \geq$ $\frac{1}{2}\left( \frac{k - 1}{k}\right) \mathrm{{OPT}} \geq  \frac{1}{4}\mathrm{{OPT}}$ ,as required.

证明. 回顾 $C = {48} \cdot  {6}^{3}$ 。首先假设 $k = 1$ ，所以 $S$ 如第8行所定义。如果我们对 $\epsilon  = \frac{1}{6}$ 应用引理3.2，接着采用定理3.1证明中对BuildSeedSet(H,k)的分析，我们得到 ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  \frac{1}{2}\mathrm{{OPT}}$ ，符合要求。对于 $k > 1$ ，对 $\epsilon  = \frac{1}{6}$ 应用引理3.2则得到 ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( S\right) }\right\rbrack   \geq  \frac{1}{2}{\mathrm{{OPT}}}_{k - 1}$ ，其中 ${\mathrm{{OPT}}}_{k - 1}$ 是大小至多为 $k - 1$ 的集合上的最大影响力。但现在，根据次模性， $\frac{1}{2}{\mathrm{{OPT}}}_{k - 1} \geq$ $\frac{1}{2}\left( \frac{k - 1}{k}\right) \mathrm{{OPT}} \geq  \frac{1}{4}\mathrm{{OPT}}$ ，符合要求。

Note that $\beta$ does not appear explicitly in the statement of Lemma 4.1. The (implicit) role of $\beta$ in Lemma 4.1 is that as $\beta$ becomes small,Algorithm 2 uses fewer steps to construct hypergraph $\mathcal{H}$ and hence the condition of the lemma is less likely to be satisfied. We next show that if $m\left( \mathcal{H}\right)$ is small,then node $v$ from line 3 is likely to have high influence. This follows because, in a small number of edges, we do not expect to see many nodes with low influence. Since we see a large number of nodes in total, we conclude that most of them must have high influence.

注意， $\beta$ 在引理4.1的陈述中并未明确出现。 $\beta$ 在引理4.1中的（隐含）作用是，当 $\beta$ 变小时，算法2构建超图 $\mathcal{H}$ 所需的步骤更少，因此该引理的条件更不容易满足。接下来我们证明，如果 $m\left( \mathcal{H}\right)$ 很小，那么第3行中的节点 $v$ 很可能具有高影响力。这是因为，在少量的边中，我们预计不会看到很多影响力低的节点。由于我们总共看到了大量的节点，我们得出结论：它们中的大多数必然具有高影响力。

LEMMA 4.2. Suppose that $m\left( \mathcal{H}\right)  < \frac{{4Cn}\log \left( n\right) }{OPT}$ . Then, with probability at least $2/3$ ,node $v$ (from line 3 of ${Al}$ - gorithm 2) satisfies ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( v\right) }\right\rbrack   \geq  \beta  \cdot  \mathrm{{OPT}}$ ,with probability taken over randomness in $\mathcal{H}$ . Proof. Recall $C = \left( {{48} \cdot  {6}^{3}}\right)$ . Let random variable $X$ denote the number of times that a node with influence at most $\beta  \cdot  \mathrm{{OPT}}$ was added to a hyperedge of $\mathcal{H}$ . Since $\mathcal{H}$ has fewer than $\frac{{4Cn}\log \left( n\right) }{\mathrm{{OPT}}}$ edges,the expected value of $X$ is at most

引理4.2. 假设 $m\left( \mathcal{H}\right)  < \frac{{4Cn}\log \left( n\right) }{OPT}$ 。那么，至少以 $2/3$ 的概率，节点 $v$ （来自 ${Al}$ - 算法2的第3行）满足 ${\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( v\right) }\right\rbrack   \geq  \beta  \cdot  \mathrm{{OPT}}$ ，这里的概率是关于 $\mathcal{H}$ 中的随机性而言的。证明. 回顾 $C = \left( {{48} \cdot  {6}^{3}}\right)$ 。设随机变量 $X$ 表示影响力至多为 $\beta  \cdot  \mathrm{{OPT}}$ 的节点被添加到 $\mathcal{H}$ 的超边中的次数。由于 $\mathcal{H}$ 的边数少于 $\frac{{4Cn}\log \left( n\right) }{\mathrm{{OPT}}}$ ， $X$ 的期望值至多为

$$
E\left\lbrack  X\right\rbrack   \leq  \frac{{4Cn}\log \left( n\right) }{\mathrm{{OPT}}}\mathop{\sum }\limits_{{u \in  V}}\frac{1}{n}\min \left\{  {{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( u\right) }\right\rbrack  ,\beta  \cdot  \mathrm{{OPT}}}\right\}  
$$

$$
 \leq  {4C\beta n}\log \left( n\right) \text{.}
$$

Markov inequality then gives that $\Pr \lbrack X >$ ${24C\beta n}\log \left( n\right) \rbrack  < 1/6$ . Conditioning on this event, we have that at most ${24C\beta n}\log \left( n\right)$ of the nodes touched by BuildHypergraph have influence less than $\beta$ - OPT. Since at least $\left( {144C}\right) {\beta n}\log \left( n\right)$ nodes were touched in total,the probability that node $v$ from line 4 has influence less than $\beta  \cdot  \mathrm{{OPT}}$ is at most $1/6$ . The union bound then allows us to conclude that $v$ has $\mathbb{E}\left\lbrack  {I\left( v\right) }\right\rbrack   \geq  \beta  \cdot  \mathrm{{OPT}}$ with probability at least $1 - \left( {1/6 + 1/6}\right)  \geq  2/3$ .

马尔可夫不等式（Markov inequality）表明 $\Pr \lbrack X >$ ${24C\beta n}\log \left( n\right) \rbrack  < 1/6$ 。基于此事件进行条件分析，我们可知在构建超图（BuildHypergraph）过程中触及的节点里，影响力小于 $\beta$ - 最优解（OPT）的节点最多有 ${24C\beta n}\log \left( n\right)$ 个。由于总共至少触及了 $\left( {144C}\right) {\beta n}\log \left( n\right)$ 个节点，那么第 4 行中的节点 $v$ 影响力小于 $\beta  \cdot  \mathrm{{OPT}}$ 的概率至多为 $1/6$ 。通过联合界（union bound），我们可以得出节点 $v$ 具有 $\mathbb{E}\left\lbrack  {I\left( v\right) }\right\rbrack   \geq  \beta  \cdot  \mathrm{{OPT}}$ 的概率至少为 $1 - \left( {1/6 + 1/6}\right)  \geq  2/3$ 。

For the case $k = 1$ ,the algorithm chooses between returning $S$ and returning $\{ v\}$ ,based on the maximum degree in $\mathcal{H}$ . The following lemma motivates this choice. The proof follows from an application of concentration bounds: if a node is present in $O\left( {\log n}\right)$ hyperedges,then with high probability we have obtained enough samples to accurately estimate its influence.

对于 $k = 1$ 的情况，该算法会根据 $\mathcal{H}$ 中的最大度，在返回 $S$ 和返回 $\{ v\}$ 之间做出选择。以下引理说明了做出此选择的动机。该证明源于对集中界（concentration bounds）的应用：如果一个节点存在于 $O\left( {\log n}\right)$ 条超边中，那么我们很有可能已经获得了足够的样本，从而能够准确估计其影响力。

LEMMA 4.3. If $k = 1$ then the following is true with probability at least $1 - \frac{2}{n}$ . If $\mathop{\max }\limits_{u}\left\{  {{\deg }_{\mathcal{H}}\left( u\right) }\right\}   > {2C}\log n$ then $m\left( \mathcal{H}\right)  > \frac{{Cn}\log \left( n\right) }{OPT}$ . Otherwise, $m\left( \mathcal{H}\right)  < \frac{{4Cn}\log \left( n\right) }{OPT}$ .

引理 4.3：若 $k = 1$ ，则以下情况至少以 $1 - \frac{2}{n}$ 的概率成立。若 $\mathop{\max }\limits_{u}\left\{  {{\deg }_{\mathcal{H}}\left( u\right) }\right\}   > {2C}\log n$ ，则 $m\left( \mathcal{H}\right)  > \frac{{Cn}\log \left( n\right) }{OPT}$ ；否则， $m\left( \mathcal{H}\right)  < \frac{{4Cn}\log \left( n\right) }{OPT}$ 。

Proof. As in the proof of Lemma 3.2,we will think of $\mathcal{H}$ as being built incrementally edge by edge, and we will let ${D}_{w}^{J}$ denote the degree of $w$ in $\mathcal{H}$ after $J$ edges have been added. Then,for any fixed $J,{D}_{w}^{J}$ is precisely the sum of $J$ Bernoulli random variables,each with expectation $\frac{1}{n}{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( w\right) }\right\rbrack$ ,and hence $\mathbb{E}\left\lbrack  {D}_{w}^{J}\right\rbrack   = \frac{J}{n}{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( w\right) }\right\rbrack$ .

证明：与引理 3.2 的证明类似，我们将把 $\mathcal{H}$ 视为逐边递增构建而成的，并且用 ${D}_{w}^{J}$ 表示在添加了 $J$ 条边后， $w$ 在 $\mathcal{H}$ 中的度。那么，对于任何固定的 $J,{D}_{w}^{J}$ ，它恰好是 $J$ 个伯努利随机变量（Bernoulli random variables）之和，每个变量的期望为 $\frac{1}{n}{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( w\right) }\right\rbrack$ ，因此 $\mathbb{E}\left\lbrack  {D}_{w}^{J}\right\rbrack   = \frac{J}{n}{\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( w\right) }\right\rbrack$ 。

Let ${J}_{1}$ be the minimal $J$ such that $\mathbb{E}\left\lbrack  {D}_{w}^{{J}_{1}}\right\rbrack   >$ ${4C}\log n$ . Chernoff bounds (Lemma A.1) imply

设 ${J}_{1}$ 为满足 $\mathbb{E}\left\lbrack  {D}_{w}^{{J}_{1}}\right\rbrack   >$ ${4C}\log n$ 的最小 $J$ 。切尔诺夫界（Chernoff bounds，引理 A.1）表明

$$
\Pr \left\lbrack  {{D}_{w}^{{J}_{1}} \leq  {2C}\log n}\right\rbrack   < {e}^{-2\log n} = 1/{n}^{2}.
$$

Suppose that this event does not occur. Then for any $J \geq  {J}_{1}$ ,we have ${D}_{w}^{J} > {D}_{w}^{{J}_{1}} > {2C}\log n$ . In particular, if $\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   > {4C}\log n$ ,we must have $m\left( \mathcal{H}\right)  \geq  {J}_{1}$ ,and hence ${D}_{w}^{m\left( \mathcal{H}\right) } > {2C}\log n$ .

假设此事件未发生。那么对于任何 $J \geq  {J}_{1}$ ，我们有 ${D}_{w}^{J} > {D}_{w}^{{J}_{1}} > {2C}\log n$ 。特别地，如果 $\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   > {4C}\log n$ ，我们必然有 $m\left( \mathcal{H}\right)  \geq  {J}_{1}$ ，因此 ${D}_{w}^{m\left( \mathcal{H}\right) } > {2C}\log n$ 。

Let ${J}_{2}$ be the maximal $J$ such that $\mathbb{E}\left\lbrack  {D}_{w}^{{J}_{2}}\right\rbrack   <$ $C\log n$ . Chernoff bounds (Lemma A.1) again imply that

设 ${J}_{2}$ 为满足 $\mathbb{E}\left\lbrack  {D}_{w}^{{J}_{2}}\right\rbrack   <$ $C\log n$ 的最大 $J$。切尔诺夫界（引理 A.1）再次表明

$$
\Pr \left\lbrack  {{D}_{w}^{{J}_{2}} \geq  {2C}\log n}\right\rbrack   < {e}^{-2\log n} = 1/{n}^{2}.
$$

Suppose that this event does not occur. Then for any $J \leq  {J}_{2}$ ,we have ${D}_{w}^{J} < {D}_{w}^{{J}_{2}} < {2C}\log n$ . In particular, for any $w$ such that $\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   < C\log n$ ,we must have $m\left( \mathcal{H}\right)  \leq  {J}_{2}$ ,and hence ${D}_{w}^{m\left( \mathcal{H}\right) } < {2C}\log n$ .

假设此事件未发生。那么对于任意 $J \leq  {J}_{2}$，我们有 ${D}_{w}^{J} < {D}_{w}^{{J}_{2}} < {2C}\log n$。特别地，对于任意满足 $\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   < C\log n$ 的 $w$，我们必定有 $m\left( \mathcal{H}\right)  \leq  {J}_{2}$，因此有 ${D}_{w}^{m\left( \mathcal{H}\right) } < {2C}\log n$。

Taking the union bound over all $w$ ,we conclude that with probability at least $1 - 2/n$ ,only $w$ for which $\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   > C\log n$ ,and every $w$ with $\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   >$ ${4C}\log n$ ,will have ${D}_{w}^{m\left( \mathcal{H}\right) } \geq  {2C}\log n$ . We will condition on this event for the remainder of the proof.

对所有 $w$ 取联合界，我们得出，至少以 $1 - 2/n$ 的概率，只有满足 $\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   > C\log n$ 的 $w$，以及每个满足 $\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   >$ ${4C}\log n$ 的 $w$，才会有 ${D}_{w}^{m\left( \mathcal{H}\right) } \geq  {2C}\log n$。在证明的其余部分，我们将基于此事件进行条件分析。

Suppose that $\mathop{\max }\limits_{w}{D}_{w}^{m\left( \mathcal{H}\right) } < {2C}\log n$ . Then we have that $\mathop{\max }\limits_{w}\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   < {4C}\log n$ . Since $\mathop{\max }\limits_{w}\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   = \mathop{\max }\limits_{w}\frac{1}{n}m\left( \mathcal{H}\right) {\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( w\right) }\right\rbrack   = \frac{1}{n}m\left( \mathcal{H}\right) .$ ${OPT}$ ,we conclude $m\left( \mathcal{H}\right)  < \frac{{4Cn}\log \left( n\right) }{OPT}$ as required. Next suppose that $\mathop{\max }\limits_{w}{D}_{w}^{m\left( \mathcal{H}\right) } > {2C}\log n$ . We then have that $\mathop{\max }\limits_{w}\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   > C\log n$ . Since,again, $\mathop{\max }\limits_{w}\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   = \frac{1}{n}m\left( \mathcal{H}\right)  \cdot  {OPT}$ ,we conclude $m\left( \mathcal{H}\right)  >$ $\frac{{Cn}\log \left( n\right) }{OPT}$ as required.

假设 $\mathop{\max }\limits_{w}{D}_{w}^{m\left( \mathcal{H}\right) } < {2C}\log n$。那么我们有 $\mathop{\max }\limits_{w}\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   < {4C}\log n$。由于 $\mathop{\max }\limits_{w}\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   = \mathop{\max }\limits_{w}\frac{1}{n}m\left( \mathcal{H}\right) {\mathbb{E}}_{\mathcal{G}}\left\lbrack  {I\left( w\right) }\right\rbrack   = \frac{1}{n}m\left( \mathcal{H}\right) .$ ${OPT}$，我们得出所需的 $m\left( \mathcal{H}\right)  < \frac{{4Cn}\log \left( n\right) }{OPT}$。接下来假设 $\mathop{\max }\limits_{w}{D}_{w}^{m\left( \mathcal{H}\right) } > {2C}\log n$。那么我们有 $\mathop{\max }\limits_{w}\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   > C\log n$。同样，由于 $\mathop{\max }\limits_{w}\mathbb{E}\left\lbrack  {D}_{w}^{m\left( \mathcal{H}\right) }\right\rbrack   = \frac{1}{n}m\left( \mathcal{H}\right)  \cdot  {OPT}$，我们得出所需的 $m\left( \mathcal{H}\right)  >$ $\frac{{Cn}\log \left( n\right) }{OPT}$。

We are now ready to complete the proof of Theorem 4.1.

我们现在准备完成定理 4.1 的证明。

Proof. [Proof of Theorem 4.1] Lemma 4.1 and Lemma 4.2 imply that,with probability at least $2/3 - 1/{n}^{2} \geq$ $3/5$ (for $n \geq  5$ ),one of $S$ or $\{ v\}$ has influence at least $\min \{ \frac{1}{4},\beta \}  \cdot  {OPT}$ ,and therefore $S \cup  \{ v\}$ does as well. If $k > 1$ then we return $S \cup  \{ v\}$ and we are done. Otherwise,Lemma 4.3 implies that if we return set $S$ then the influence of $S$ is at least ${OPT}/4$ (by Lemma 4.1),and if we return set $v$ then the expected influence of $v$ is at least $\beta  \cdot  {OPT}$ (by Lemma 4.2). Thus,in all cases, we return a set of influence at least $\min \{ \frac{1}{4},\beta \}  \cdot  {OPT}$ . The required bound on the runtime of Algorithm 2 follows directly from the value of $R$ on line 1,as in the proof of Theorem 3.1.

证明。[定理4.1的证明] 引理4.1和引理4.2表明，至少以$2/3 - 1/{n}^{2} \geq$ $3/5$的概率（对于$n \geq  5$ ），$S$ 或$\{ v\}$ 中的一个的影响力至少为$\min \{ \frac{1}{4},\beta \}  \cdot  {OPT}$ ，因此$S \cup  \{ v\}$ 也是如此。如果$k > 1$ ，那么我们返回$S \cup  \{ v\}$ ，证明完成。否则，引理4.3表明，如果我们返回集合$S$ ，那么$S$ 的影响力至少为${OPT}/4$ （根据引理4.1），如果我们返回集合$v$ ，那么$v$ 的期望影响力至少为$\beta  \cdot  {OPT}$ （根据引理4.2）。因此，在所有情况下，我们返回的集合的影响力至少为$\min \{ \frac{1}{4},\beta \}  \cdot  {OPT}$ 。算法2运行时间的所需界限直接由第1行的$R$ 值得出，如同定理3.1的证明一样。

4.1 Dynamic Runtime Algorithm 2 assumes that the desired approximation factor, $\beta$ ,is provided as a parameter to the problem. We note that a slight modification to the algorithm removes the requirement that $\beta$ be specified in advance. That is,we obtain an algorithm that can be terminated without warning, say after $O\left( {\gamma  \cdot  \left( {n + m}\right) \log \left( n\right) }\right)$ steps for some $\gamma  \leq  1$ ,at which point it immediately returns a solution that is an $O\left( \gamma \right)$ approximation with probability at least $\frac{3}{5}$ . To achieve this,we execute Algorithm 2 as though $\beta  = 1$ , but then modify BuildHypergraph so that, for each $i \geq  1$ ,we pause the creation of hypergraph $\mathcal{H}$ after ${2}^{i}$ steps and complete the algorithm using the current hypergraph,which takes time at most $O\left( {2}^{i}\right)$ . Once this is done, we save the resulting solution and resume the creation of the hypergraph until the next power of 2. When the algorithm is terminated, we return the most recently-computed solution; this corresponds to a solution for a hypergraph built using at least half of the total steps taken by the algorithm at the time of termination. Theorem 4.1 then implies that this solution has approximation $O\left( \gamma \right)$ if termination occurs after $O\left( {\gamma  \cdot  \left( {n + m}\right) \log \left( n\right) }\right)$ steps.

4.1 动态运行时间 算法2假设所需的近似因子$\beta$ 作为问题的一个参数给出。我们注意到，对该算法进行轻微修改可以消除必须提前指定$\beta$ 的要求。也就是说，我们得到一个可以无预警终止的算法，例如在经过某个$\gamma  \leq  1$ 的$O\left( {\gamma  \cdot  \left( {n + m}\right) \log \left( n\right) }\right)$ 步之后，此时它会立即返回一个解，该解至少以$\frac{3}{5}$ 的概率是一个$O\left( \gamma \right)$ 近似解。为了实现这一点，我们就好像$\beta  = 1$ 那样执行算法2，但随后修改构建超图（BuildHypergraph）的过程，使得对于每个$i \geq  1$ ，我们在${2}^{i}$ 步之后暂停超图$\mathcal{H}$ 的创建，并使用当前的超图完成算法，这最多需要$O\left( {2}^{i}\right)$ 的时间。一旦完成这一步，我们保存得到的解，并继续创建超图，直到下一个2的幂次。当算法终止时，我们返回最近计算得到的解；这对应于使用算法在终止时所采取的总步数的至少一半构建的超图的一个解。然后定理4.1表明，如果在$O\left( {\gamma  \cdot  \left( {n + m}\right) \log \left( n\right) }\right)$ 步之后终止，这个解具有$O\left( \gamma \right)$ 近似性。

4.2 A Lower Bound We provide a lower bound on the time it takes for any algorithm, equipped with uniform node sampling,to compute a $\beta$ -approximation for the maximum expected influence problem under the adjacency list network representation. In particular, for any given budget $k$ ,at least $\Omega \left( {\beta n}\right)$ queries are required to obtain approximation factor $\beta$ with fixed probability.

4.2 一个下界 我们给出了在邻接表网络表示下，任何配备均匀节点采样的算法计算最大期望影响力问题的$\beta$ 近似解所需时间的一个下界。特别地，对于任何给定的预算$k$ ，至少需要$\Omega \left( {\beta n}\right)$ 次查询才能以固定概率获得近似因子$\beta$ 。

THEOREM 4.2. Let $0 < \epsilon  < \frac{1}{10e},\beta  \leq  1$ be given. Any randomized algorithm for the maximum influence problem that has runtime of $\frac{\beta \left( {m + n}\right) }{{24}\min \{ k,1/\beta \} }$ cannot return, with probability at least $1 - \frac{1}{e} - \epsilon$ ,a set of nodes with approximation ratio better than $\beta$ .

定理4.2。给定$0 < \epsilon  < \frac{1}{10e},\beta  \leq  1$ 。任何运行时间为$\frac{\beta \left( {m + n}\right) }{{24}\min \{ k,1/\beta \} }$ 的最大影响力问题的随机算法，都不能至少以$1 - \frac{1}{e} - \epsilon$ 的概率返回一个近似比优于$\beta$ 的节点集合。

Proof. Note first that for a graph consisting of $n$ singletons,an algorithm must return at least ${\beta k}$ nodes to obtain an approximation ratio of $\beta$ . Doing so in at most ${\beta }^{2}n/2$ queries requires that ${2\beta k} \leq  {\beta }^{2}n$ ,which implies ${2k}/\beta  \leq  n$ . We can therefore assume ${2k}/\beta  \leq  n$ .

证明。首先注意到，对于一个由 $n$ 个孤立顶点（singletons）组成的图，一个算法必须返回至少 ${\beta k}$ 个节点才能获得 $\beta$ 的近似比。要在最多 ${\beta }^{2}n/2$ 次查询内做到这一点，需要满足 ${2\beta k} \leq  {\beta }^{2}n$，这意味着 ${2k}/\beta  \leq  n$。因此，我们可以假设 ${2k}/\beta  \leq  n$。

The proof will use Yao's Minimax Principle for the performance of Las Vegas (LV) randomized algorithms on a family of inputs [39]. The lemma states that the least expected cost of deterministic LV algorithms on a distribution over a family inputs is a lower bound on the expected cost of the optimal randomized LV algorithm over that family of inputs. Define the cost of the algorithm as 0 if it returns a set nodes with approximation ratio better than $\beta$ and 1 otherwise. As the cost of an algorithm equals its probability of failure, we can think of it as a LV algorithm.

该证明将使用姚氏极小极大原理（Yao's Minimax Principle）来分析拉斯维加斯（Las Vegas，LV）随机算法在一族输入上的性能 [39]。该引理指出，确定性 LV 算法在一族输入的分布上的最小期望成本是最优随机 LV 算法在该族输入上的期望成本的下界。如果算法返回的节点集的近似比优于 $\beta$，则将算法的成本定义为 0，否则定义为 1。由于算法的成本等于其失败概率，我们可以将其视为一个 LV 算法。

Assume for notational simplicity that $\beta  = 1/T$ where $T$ is an integer. We will build a family of lower bound graphs,one for each value of $n$ (beginning from $n = 1 + T$ ); each graph will have $m \leq  n$ ,so it will suffice to demonstrate a lower bound of $\frac{n}{{12T}\min \{ k,T\} }$ .

为了符号表示的简便，假设 $\beta  = 1/T$，其中 $T$ 是一个整数。我们将构建一族下界图，对于 $n$ 的每个值（从 $n = 1 + T$ 开始）构建一个图；每个图将有 $m \leq  n$，因此证明 $\frac{n}{{12T}\min \{ k,T\} }$ 的下界就足够了。

We now consider the behavior of a deterministic algorithm $A$ with respect to the uniform distribution on the constructed family of inputs. For a given value $T$ the graph would be made from $k$ components of size ${2T}$ and $n - {2kT}$ singleton components (recall that ${2kT} =$ ${2k}/\beta  \leq  n)$ . If algorithm $A$ returns nodes from $\ell$ of the $k$ components of size ${2T}$ ,it achieves a total influence of $2\ell T + \left( {k - \ell }\right)$ . Thus,to attain approximation factor better than $\beta  = \frac{1}{T}$ ,we must have $2\ell T + \left( {k - \ell }\right)  \geq  \frac{1}{T}{2kT}$ , which implies $\ell  \geq  \frac{k}{{2T} - 1}$ for any $T > 1$ .

我们现在考虑确定性算法 $A$ 在构造的输入族上的均匀分布下的行为。对于给定的值 $T$，图将由 $k$ 个大小为 ${2T}$ 的组件和 $n - {2kT}$ 个孤立组件组成（回想 ${2kT} =$ ${2k}/\beta  \leq  n)$）。如果算法 $A$ 从 $k$ 个大小为 ${2T}$ 的组件中的 $\ell$ 个组件返回节点，它将获得 $2\ell T + \left( {k - \ell }\right)$ 的总影响力。因此，为了获得优于 $\beta  = \frac{1}{T}$ 的近似因子，我们必须有 $2\ell T + \left( {k - \ell }\right)  \geq  \frac{1}{T}{2kT}$，这意味着对于任何 $T > 1$ 都有 $\ell  \geq  \frac{k}{{2T} - 1}$。

Suppose $k > {12T}$ . The condition $\ell  \geq  \frac{k}{{2T} - 1}$ implies that at least $\frac{k}{{2T} - 1}$ of the large components must be queried by the algorithm, where each random query has probability $\frac{2kT}{n}$ of hitting a large component. If the algorithm makes fewer than $\frac{n}{{12}{T}^{2}}$ queries,then the expected number of components hit is $\frac{n}{{12}{T}^{2}} \cdot  \frac{2kT}{n} = \frac{k}{6T}$ . The Multiplicative Chernoff bound (Lemma A.1, part 3) then imply that the probability hitting more than $\frac{k}{2T}$ components is no more than ${e}^{-\frac{k}{6T} \cdot  2/3} \leq  \frac{1}{{e}^{4/3}} < 1 - \frac{1}{e} - \epsilon$ , a contradiction.

假设 $k > {12T}$。条件 $\ell  \geq  \frac{k}{{2T} - 1}$ 意味着算法必须查询至少 $\frac{k}{{2T} - 1}$ 个大组件，其中每次随机查询命中大组件的概率为 $\frac{2kT}{n}$。如果算法进行的查询次数少于 $\frac{n}{{12}{T}^{2}}$ 次，那么命中的组件的期望数量为 $\frac{n}{{12}{T}^{2}} \cdot  \frac{2kT}{n} = \frac{k}{6T}$。乘法切尔诺夫界（引理 A.1，第 3 部分）则意味着命中超过 $\frac{k}{2T}$ 个组件的概率不超过 ${e}^{-\frac{k}{6T} \cdot  2/3} \leq  \frac{1}{{e}^{4/3}} < 1 - \frac{1}{e} - \epsilon$，这是一个矛盾。

If $k \leq  {12T}$ then we need that $\ell  \geq  1$ ,which occurs only if the algorithm queries at least one of the ${kT}$ vertices in the large components. With $\frac{n}{2kT}$ queries,for $n$ large enough,this happens with probability smaller than $\frac{1}{e} - \epsilon$ ,a contradiction.

如果 $k \leq  {12T}$，那么我们需要 $\ell  \geq  1$，这仅当算法查询大组件中的 ${kT}$ 个顶点中的至少一个时才会发生。对于 $\frac{n}{2kT}$ 次查询，当 $n$ 足够大时，这种情况发生的概率小于 $\frac{1}{e} - \epsilon$，这是一个矛盾。

We conclude that,in all cases,at least $\frac{n}{{12T}\min \{ k,T\} }$ queries are necessary to obtain approximation factor better than $\beta  = \frac{1}{T}$ with probability at least $1 - \frac{1}{e} - \epsilon$ , as required.

我们得出结论，在所有情况下，为了以至少 $1 - \frac{1}{e} - \epsilon$ 的概率获得优于 $\beta  = \frac{1}{T}$ 的近似因子，至少需要 $\frac{n}{{12T}\min \{ k,T\} }$ 次查询，正如所要求的。

By Yao's Minimax Principle this gives a lower bound of $\Omega \left( \frac{nd}{{24T}\min \{ k,T\} }\right)$ on the expected performance of any randomized algorithm, on at least one of the inputs.

根据姚氏极小极大原理（Yao's Minimax Principle），这为任意随机算法在至少一个输入上的期望性能给出了一个下界$\Omega \left( \frac{nd}{{24T}\min \{ k,T\} }\right)$。

Finally, the construction can be modified to apply to non-sparse networks. For any $d \leq  n$ ,we can augment our graph by overlaying a $d$ -regular graph with exponentially small weight on each edge. This does not significantly impact the influence of any set, but increases the time to decide if a node is in a large component by a factor of $O\left( d\right)$ (as edges must be traversed until one with non-exponentially-small weight is found). Thus,for each $d \leq  n$ ,we have a lower bound of $\frac{nd}{{24T}\min \{ k,T\} }$ on the expected performance of $A$ on a distribution of networks with $m = {nd}$ edges.

最后，可以对该构造进行修改以应用于非稀疏网络。对于任意$d \leq  n$，我们可以通过在图上叠加一个每条边权重呈指数级小的$d$ - 正则图来扩充我们的图。这不会显著影响任何集合的影响力，但会使判断一个节点是否在一个大组件中的时间增加$O\left( d\right)$倍（因为必须遍历边，直到找到一条权重非指数级小的边）。因此，对于每个$d \leq  n$，我们在具有$m = {nd}$条边的网络分布上为$A$的期望性能给出了一个下界$\frac{nd}{{24T}\min \{ k,T\} }$。

Discussion: The lower bound construction of Theorem 4.2 is tailored to the query model considered in this paper. In particular, we do not assume that vertices are not sorted by degree, component size, etc. However, the construction can be easily modified to be robust to various changes in the model, by (for example) adding edges with small weight so that the exhibited network $\mathcal{G}$ becomes connected and/or regular.

讨论：定理4.2的下界构造是针对本文所考虑的查询模型量身定制的。特别地，我们不假设顶点按度、组件大小等进行排序。然而，通过（例如）添加权重较小的边，使得所展示的网络$\mathcal{G}$变得连通和/或正则，该构造可以很容易地修改以适应模型中的各种变化。

Acknowledgments We thank Elchanan Mossel for helpful discussions.

致谢 我们感谢埃尔查南·莫塞尔（Elchanan Mossel）提供的有益讨论。

## A Multiplicative Chernoff Bound

## 乘法切尔诺夫界（Multiplicative Chernoff Bound）

For reference, we now provide the statement of the Chernoff bounds used throughout this paper.

为了参考，我们现在给出本文中使用的切尔诺夫界（Chernoff bounds）的陈述。

LEMMA A.1. Let ${X}_{i}$ be $n$ i.i.d. Bernoulli random variables with expectation $\mu$ each. Define $X = \mathop{\sum }\limits_{{i = 1}}^{n}{X}_{i}$ . Then,

引理A.1。设${X}_{i}$为$n$个独立同分布（i.i.d.）的伯努利随机变量（Bernoulli random variables），每个变量的期望为$\mu$。定义$X = \mathop{\sum }\limits_{{i = 1}}^{n}{X}_{i}$。那么，

- For $0 < \lambda  < 1 : \Pr \left\lbrack  {X < \left( {1 - \lambda }\right) {\mu n}}\right\rbrack   <$ $\exp \left( {-{\mu n}{\lambda }^{2}/2}\right)$ .

- 对于$0 < \lambda  < 1 : \Pr \left\lbrack  {X < \left( {1 - \lambda }\right) {\mu n}}\right\rbrack   <$ $\exp \left( {-{\mu n}{\lambda }^{2}/2}\right)$。

- For $0 < \lambda  < 1 : \Pr \left\lbrack  {X > \left( {1 + \lambda }\right) {\mu n}}\right\rbrack   <$ $\exp \left( {-{\mu n}{\lambda }^{2}/4}\right)$ .

- 对于$0 < \lambda  < 1 : \Pr \left\lbrack  {X > \left( {1 + \lambda }\right) {\mu n}}\right\rbrack   <$ $\exp \left( {-{\mu n}{\lambda }^{2}/4}\right)$。

- For $\lambda  \geq  1 : \Pr \left\lbrack  {X > \left( {1 + \lambda }\right) {\mu n}}\right\rbrack   < \exp \left( {-{\mu n\lambda }/3}\right)$ .

- 对于$\lambda  \geq  1 : \Pr \left\lbrack  {X > \left( {1 + \lambda }\right) {\mu n}}\right\rbrack   < \exp \left( {-{\mu n\lambda }/3}\right)$。

## References

## 参考文献

[1] Andersen, R., Borgs, C., Chayes, J.,

[1] 安德森（Andersen），R.，博格斯（Borgs），C.，蔡斯（Chayes），J.，

HOPCRAFT, J., MIRROKNI, V. S., AND TENG, S.-H. 2007. Local computation of pagerank contributions. In ${WAW}$ . 150-165.

[2] Andersen, R., Chung, F., And Lang, K. 2006. Local graph partitioning using pagerank vectors. In FOCS. 475-486.

[3] Baksshy, E., Karrer, B., And Adamic, L. A. 2009. Social influence and the diffusion of user-created content. In ${ACMEC}$ . 325-334.

[4] Ben-ZwI, O., Hermelin, D., Lokshtanov, D., AND NEWMAN, I. 2011. Treewidth governs the complexity of target set selection. Discrete Opt. 8, 1, 87-96.

[5] Bharathi, S., Kempe, D., and Salek, M. 2007. Competitive influence maximization in social networks. In WINE. 306-311.

[6] Borgs, C., Brautbar, M., Chayes, J. T., and TENG, S.-H. 2012. A sublinear time algorithm for pagerank computations. In WAW. 41-53.

[7] BRAUTBAR, M. AND KEARNS, M. 2010. Local algorithms for finding interesting individuals in large networks. In ICS. 188-199.

[8] Brown, J. J. And Reingen, P. H. 1987. Social ties and word of mouth referral behavior. Journal of Consumer Research 14, 3, 350-362.

[9] Centola, D. and Macy, M. 2007. Complex contagions and the weakness of long ties. American Journal of Sociology 113, 3, 702-734.

[10] Cha, M., Mislove, A., and Gummadi, P. K. 2009. A measurement-driven analysis of information propagation in the flickr social network. In ${WWW}$ . 721-730.

[11] CHEN, N. 2008. On the approximability of influence in social networks. In ${SODA}$ . 1029-1037.

[12] Chen, W., Wang, C., And Wang, Y. 2010a. Scalable influence maximization for prevalent viral marketing in large-scale social networks. In ${KDD}$ . 1029-1038.

[13] Chen, W., Wang, Y., And Yang, S. 2009. Efficient influence maximization in social networks. In ${KDD}$ . 199-208.

[14] Chen, W., YuAn, Y., And Zhang, L. 2010b. Scalable influence maximization in social networks under the linear threshold model. In ICDM. 88-97.

[15] Dodds, P. AND WATTS, D. 2007. Universal behavior in a generalized model of contagion. Phys Rev Lett 92, 21, 218701.

[16] Domingos, P. and Richardson, M. 2001. Mining the network value of customers. In ${KDD}{.57} - {66}$ .

[17] Goldenserg, J., Lipai, B., and Mulle, E. 2001. Talk of the network: A complex systems look at the underlying process of word-of-mouth. Mark. Let., 221-223.

[18] Goldreich, O. 2010. Introduction to testing graph properties. In Property Testing. 105-141.

[19] Gomez-Robriguez, M., Leskovec, J., AND KRAUSE, A. 2012. Inferring networks of diffusion and influence. ${TKDD5},4,{21}$ .

[20] Goyal, S. and Kearns, M. 2012. Competitive contagion in networks. In STOC. 759-774.

[21] GRANOVETTER, M. 1978. Threshold models of collective behavior. American Journal of Sociology 83, 1420-1443.

[22] Jiang, Q., Song, G., Cong, G., Wang, Y., Si, W., AND XIE, K. 2011. Simulated annealing based influence maximization in social networks. In ${AAAI}$ .

[23] Kempe, D., Kleinberg, J., and Tardos, E. 2003. Maximizing the spread of influence through a social network. In ${KDD}$ . 137-146.

[24] Kempe, D., Kleinberg, J. M., and Tardos, É. 2005. Influential nodes in a diffusion model for social networks. In ICALP. 1127-1138.

[25] KIMURA, M. AND SAITO, K. 2006. Tractable models for information diffusion in social networks. In ${PKDD}$ . 259-271.

[26] LESKOVEC, J., ADAMIC, L. A., AND HUBERMAN, B. A. 2007a. The dynamics of viral marketing. TWEB 1, 1.

[27] LESKOVEC, J., KRAUSE, A., GUESTRIN, C., Faloutsos, C., VanBriesen, J. M., and Glance, N. S. 2007b. Cost-effective outbreak detection in networks. In ${KDD}$ . 420-429.

[28] LESKOVEC, J., MCGLOHON, M., FALOUTSOS, C., Glance, N. S., and Hurst, M. 2007c. Patterns of cascading behavior in large blog graphs. In ${SDM}$ .

[29] LIBEN-NOWELL, D. AND KLEINBERG, J. 2008. Tracing information flow on a global scale using internet chain-letter data. PNAS 105, 12, 4633-4638.

[30] Mathioudakis, M., Bonchi, F., Castillo, C., Gionis, A., AND Ukkonen, A. 2011. Sparsification of influence networks. In ${KDD}{.529} - {537}$ .

[31] MORRIS, S. 2000. Contagion. Review of Economic Studies 67, 57-78.

[32] MOSSEL, E. AND ROCH, S. 2007. On the sub-modularity of influence in social networks. In ${STOC}$ . 128-134.

[33] PELEG, D. 2002. Local majorities, coalitions and monopolies in graphs: a review. Theor. Comput. Sci. 282, 2, 231-257.

[34] Richardson, M. and Domingos, P. 2002. Mining knowledge-sharing sites for viral marketing. In ${KDD}$ . 61-70.

[35] ROGERS, E. 2003. Diffusion of Innovations 5th Ed. Free Press.

[36] Rubinfeld, R. and Shapira, A. 2011. Sublinear time algorithms. SIAM Journal on Discrete Math 25, ${1562} - {1588}$ .

[37] SPIELMAN, D. A. AND TENG, S.-H. 2004. Nearly-linear time algorithms for graph partitioning, graph sparsification, and solving linear systems. In STOC. 81-90.

[38] Wang, Y., Cong, G., Song, G., and Xie, K. 2010. Community-based greedy algorithm for mining top-k influential nodes in mobile social networks. In KDD. 1039-1048.

[39] Yao, A. C.-C. 1977. Probabilistic computations: Toward a unified measure of complexity (extended abstract). In ${FOCS}.{222} - {227}$ .