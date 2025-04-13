# Maximizing the Spread of Influence through a Social Network

# 通过社交网络最大化影响力传播

David Kempe ${}^{ * }$

大卫·肯普 ${}^{ * }$

Dept. of Computer Science Cornell University, Ithaca NY

康奈尔大学计算机科学系，纽约州伊萨卡市

kempe@cs.cornell.edu

Jon Kleinberg ${}^{ \dagger  }$

乔恩·克莱因伯格 ${}^{ \dagger  }$

Dept. of Computer Šcience

计算机科学系

Cornell University, Ithaca NY

康奈尔大学，纽约州伊萨卡市

kleinber@cs.cornell.edu

Eva Tardos ${}^{ \ddagger  }$

伊娃·塔尔多斯 ${}^{ \ddagger  }$

Dept. of Computer Science

计算机科学系

Cornell University, Ithaca NY

康奈尔大学，纽约州伊萨卡市

eva@cs.cornell.edu

## ABSTRACT

## 摘要

Models for the processes by which ideas and influence propagate through a social network have been studied in a number of domains, including the diffusion of medical and technological innovations, the sudden and widespread adoption of various strategies in game-theoretic settings, and the effects of "word of mouth" in the promotion of new products. Recently, motivated by the design of viral marketing strategies, Domingos and Richardson posed a fundamental algorithmic problem for such social network processes: if we can try to convince a subset of individuals to adopt a new product or innovation, and the goal is to trigger a large cascade of further adoptions, which set of individuals should we target?

关于思想和影响力在社交网络中传播过程的模型已经在多个领域得到研究，包括医疗和技术创新的扩散、博弈论环境中各种策略的突然广泛采用，以及“口碑”在新产品推广中的作用。最近，受病毒式营销战略设计的启发，多明戈斯和理查森针对此类社交网络过程提出了一个基本的算法问题：如果我们试图说服一部分人采用一种新产品或创新，并且目标是引发大规模的后续采用连锁反应，那么我们应该针对哪些人群呢？

We consider this problem in several of the most widely studied models in social network analysis. The optimization problem of selecting the most influential nodes is NP-hard here, and we provide the first provable approximation guarantees for efficient algorithms. Using an analysis framework based on submodular functions, we show that a natural greedy strategy obtains a solution that is provably within 63% of optimal for several classes of models; our framework suggests a general approach for reasoning about the performance guarantees of algorithms for these types of influence problems in social networks.

我们在社交网络分析中几个最广泛研究的模型中考虑了这个问题。在这里，选择最具影响力节点的优化问题是NP难问题，我们为高效算法提供了首个可证明的近似保证。使用基于次模函数的分析框架，我们表明一种自然的贪心策略在几类模型中能得到一个可证明达到最优解63%的解；我们的框架为推理社交网络中此类影响力问题算法的性能保证提供了一种通用方法。

We also provide computational experiments on large collaboration networks, showing that in addition to their provable guarantees, our approximation algorithms significantly out-perform node-selection heuristics based on the well-studied notions of degree centrality and distance centrality from the field of social networks.

我们还在大型合作网络上进行了计算实验，结果表明，除了具有可证明的保证外，我们的近似算法在性能上明显优于基于社交网络领域中经过充分研究的度中心性和距离中心性概念的节点选择启发式算法。

## Categories and Subject Descriptors

## 类别和主题描述符

F.2.2 [Analysis of Algorithms and Problem Complexity]: Nonnumerical Algorithms and Problems

F.2.2 [算法分析与问题复杂度]：非数值算法与问题

## Keywords

## 关键词

approximation algorithms, social networks, viral marketing, diffusion of innovations

近似算法、社交网络、病毒式营销、创新扩散

## 1. INTRODUCTION

## 1. 引言

A social network - the graph of relationships and interactions within a group of individuals - plays a fundamental role as a medium for the spread of information, ideas, and influence among its members. An idea or innovation will appear - for example, the use of cell phones among college students, the adoption of a new drug within the medical profession, or the rise of a political movement in an unstable society - and it can either die out quickly or make significant inroads into the population. If we want to understand the extent to which such ideas are adopted, it can be important to understand how the dynamics of adoption are likely to unfold within the underlying social network: the extent to which people are likely to be affected by decisions of their friends and colleagues, or the extent to which "word-of-mouth" effects will take hold. Such network diffusion processes have a long history of study in the social sciences. Some of the earliest systematic investigations focused on data pertaining to the adoption of medical and agricultural innovations in both developed and developing parts of the world $\left\lbrack  {8,{27},{29}}\right\rbrack$ ; in other contexts,research has investigated diffusion processes for "word-of-mouth" and "viral marketing" effects in the success of new products $\left\lbrack  {4,7,{10},{13},{14},{20},{26}}\right\rbrack$ , the sudden and widespread adoption of various strategies in game-theoretic settings $\left\lbrack  {6,{12},{21},{32},{33}}\right\rbrack$ ,and the problem of cascading failures in power systems $\left\lbrack  {2,3}\right\rbrack$ .

社交网络——一群个体之间的关系和互动图——作为信息、思想和影响力在其成员之间传播的媒介，起着基础性的作用。一种思想或创新将会出现——例如，大学生中使用手机、医疗行业采用新药，或者不稳定社会中政治运动的兴起——它要么迅速消亡，要么在人群中取得重大进展。如果我们想了解这些思想被采纳的程度，那么了解采纳动态在底层社交网络中可能如何展开就很重要：人们可能受其朋友和同事决策影响的程度，或者“口碑”效应将产生影响的程度。此类网络扩散过程在社会科学领域有着悠久的研究历史。一些最早的系统性研究聚焦于世界发达和发展中地区医疗和农业创新采纳的数据$\left\lbrack  {8,{27},{29}}\right\rbrack$；在其他情境中，研究调查了新产品成功过程中的“口碑”和“病毒式营销”效应的扩散过程$\left\lbrack  {4,7,{10},{13},{14},{20},{26}}\right\rbrack$、博弈论环境中各种策略的突然广泛采纳$\left\lbrack  {6,{12},{21},{32},{33}}\right\rbrack$，以及电力系统中的级联故障问题$\left\lbrack  {2,3}\right\rbrack$。

In recent work, motivated by applications to marketing, Domin-gos and Richardson posed a fundamental algorithmic problem for such systems $\left\lbrack  {{10},{26}}\right\rbrack$ . Suppose that we have data on a social network, with estimates for the extent to which individuals influence one another, and we would like to market a new product that we hope will be adopted by a large fraction of the network. The premise of viral marketing is that by initially targeting a few "influential" members of the network - say, giving them free samples of the product - we can trigger a cascade of influence by which friends will recommend the product to other friends, and many individuals will ultimately try it. But how should we choose the few key individuals to use for seeding this process? In [10, 26], this question was considered in a probabilistic model of interaction; heuristics were given for choosing customers with a large overall effect on the network, and methods were also developed to infer the influence data necessary for posing these types of problems.

在近期的研究中，受营销应用的启发，多明戈斯（Domingos）和理查森（Richardson）为这类系统提出了一个基本的算法问题$\left\lbrack  {{10},{26}}\right\rbrack$。假设我们有关于一个社交网络的数据，并且对个体之间相互影响的程度有估计，我们希望营销一款新产品，期望它能被网络中的很大一部分人采纳。病毒式营销的前提是，通过最初针对网络中的一些“有影响力”的成员——比如，给他们产品的免费样品——我们可以引发一系列影响，使得朋友会向其他朋友推荐该产品，最终许多人会尝试使用它。但是我们应该如何选择用于启动这个过程的几个关键个体呢？在文献[10, 26]中，这个问题是在一个概率交互模型中进行考虑的；给出了选择对网络有较大整体影响的客户的启发式方法，还开发了推断提出这类问题所需的影响数据的方法。

In this paper, we consider the issue of choosing influential sets of individuals as a problem in discrete optimization. The optimal solution is NP-hard for most models that have been studied, including the model of [10]. The framework proposed in [26], on the other hand, is based on a simple linear model where the solution to the optimization problem can be obtained by solving a system of linear equations. Here we focus on a collection of related, NP-hard models that have been extensively studied in the social networks community, and obtain the first provable approximation guarantees for efficient algorithms in a number of general cases. The generality of the models we consider lies between that of the polynomial-time solvable model of [26] and the very general model of [10], where the optimization problem cannot even be approximated to within a non-trivial factor.

在本文中，我们将选择有影响力的个体集合的问题视为一个离散优化问题。对于大多数已研究的模型，包括文献[10]中的模型，最优解是NP难的。另一方面，文献[26]中提出的框架基于一个简单的线性模型，其中优化问题的解可以通过求解一个线性方程组得到。在这里，我们关注社交网络领域广泛研究的一组相关的NP难模型，并在许多一般情况下为高效算法获得了首个可证明的近似保证。我们所考虑的模型的通用性介于文献[26]中可在多项式时间内求解的模型和文献[10]中非常通用的模型之间，在文献[10]的模型中，优化问题甚至无法在非平凡因子内进行近似。

---

<!-- Footnote -->

*Supported by an Intel Graduate Fellowship and an NSF Graduate Research Fellowship.

*由英特尔研究生奖学金和美国国家科学基金会（NSF）研究生研究奖学金资助。

${}^{ \dagger  }$ Supported in part by a David and Lucile Packard Foundation Fellowship and NSF ITR/IM Grant IIS-0081334.

${}^{ \dagger  }$ 部分由大卫与露西尔·帕卡德基金会（David and Lucile Packard Foundation）奖学金和美国国家科学基金会信息技术研究/信息管理（NSF ITR/IM）资助项目IIS - 0081334资助。

${}^{ \ddagger  }$ Supported in part by NSF ITR grant CCR-011337,and ONR grant N00014-98-1-0589.

${}^{ \ddagger  }$ 部分由美国国家科学基金会信息技术研究（NSF ITR）资助项目CCR - 011337和美国海军研究办公室（ONR）资助项目N00014 - 98 - 1 - 0589资助。

<!-- Footnote -->

---

We begin by departing somewhat from the Domingos-Richardson framework in the following sense: where their models are essentially descriptive, specifying a joint distribution over all nodes' behavior in a global sense, we focus on more operational models from mathematical sociology $\left\lbrack  {{15},{28}}\right\rbrack$ and interacting particle systems $\left\lbrack  {{11},{17}}\right\rbrack$ that explicitly represent the step-by-step dynamics of adoption. We show that approximation algorithms for maximizing the spread of influence in these models can be developed in a general framework based on submodular functions [9, 23]. We also provide computational experiments on large collaboration networks, showing that in addition to their provable guarantees, our algorithms significantly out-perform node-selection heuristics based on the well-studied notions of degree centrality and distance centrality [30] from the field of social network analysis.

我们首先在以下方面与多明戈斯 - 理查森（Domingos - Richardson）框架有所不同：他们的模型本质上是描述性的，从全局意义上指定了所有节点行为的联合分布，而我们关注的是来自数学社会学$\left\lbrack  {{15},{28}}\right\rbrack$和相互作用粒子系统$\left\lbrack  {{11},{17}}\right\rbrack$的更具操作性的模型，这些模型明确表示了采用过程的逐步动态。我们表明，在这些模型中最大化影响力传播的近似算法可以在基于次模函数的通用框架中开发[9, 23]。我们还在大型合作网络上进行了计算实验，结果表明，除了具有可证明的保证外，我们的算法在性能上显著优于基于社会网络分析领域中经过充分研究的度中心性和距离中心性概念[30]的节点选择启发式算法。

Two Basic Diffusion Models. In considering operational models for the spread of an idea or innovation through a social network $G$ ,represented by a directed graph,we will speak of each individual node as being either active (an adopter of the innovation) or inactive. We will focus on settings, guided by the motivation discussed above, in which each node's tendency to become active increases monotonically as more of its neighbors become active. Also, we will focus for now on the progressive case in which nodes can switch from being inactive to being active, but do not switch in the other direction; it turns out that this assumption can easily be lifted later. Thus, the process will look roughly as follows from the perspective of an initially inactive node $v$ : as time unfolds,more and more of $v$ ’s neighbors become active; at some point,this may cause $v$ to become active,and $v$ ’s decision may in turn trigger further decisions by nodes to which $v$ is connected.

两种基本扩散模型。在考虑通过有向图表示的社交网络$G$传播思想或创新的操作性模型时，我们将每个单独的节点视为活跃（创新的采用者）或不活跃。基于上述动机，我们将关注这样的情况：随着一个节点的更多邻居变得活跃，该节点变得活跃的倾向会单调增加。此外，目前我们将重点放在渐进情况上，即节点可以从不活跃状态转变为活跃状态，但不会反向转变；事实证明，这个假设稍后可以很容易地放宽。因此，从最初不活跃的节点$v$的角度来看，这个过程大致如下：随着时间的推移，$v$的越来越多的邻居变得活跃；在某个时刻，这可能会导致$v$变得活跃，而$v$的决策反过来又可能触发与$v$相连的节点做出进一步的决策。

Granovetter and Schelling were among the first to propose models that capture such a process; their approach was based on the use of node-specific thresholds [15, 28]. Many models of this flavor have since been investigated (see e.g. [5, 15, 18, 19, 21, 25, 28, 29, ${31},{32},{33}\rbrack )$ but the following Linear Threshold Model lies at the core of most subsequent generalizations. In this model,a node $v$ is influenced by each neighbor $w$ according to a weight ${b}_{v,w}$ such that $\sum {b}_{v,w} \leq  1$ . The dynamics of the process then proceed $w$ neighbor of $v$

格拉诺维特（Granovetter）和谢林（Schelling）是最早提出捕捉这种过程的模型的人之一；他们的方法基于使用特定节点的阈值[15, 28]。此后，许多这种类型的模型都得到了研究（例如，参见[5, 15, 18, 19, 21, 25, 28, 29, ${31},{32},{33}\rbrack )$，但以下线性阈值模型是大多数后续推广的核心。在这个模型中，节点$v$根据权重${b}_{v,w}$受到每个邻居$w$的影响，使得$\sum {b}_{v,w} \leq  1$。然后，该过程的动态按照以下方式进行$w$是$v$的邻居

as follows. Each node $v$ chooses a threshold ${\theta }_{v}$ uniformly at random from the interval $\left\lbrack  {0,1}\right\rbrack$ ; this represents the weighted fraction of $v$ ’s neighbors that must become active in order for $v$ to become active. Given a random choice of thresholds, and an initial set of active nodes ${A}_{0}$ (with all other nodes inactive),the diffusion process unfolds deterministically in discrete steps: in step $t$ ,all nodes that were active in step $t - 1$ remain active,and we activate any node $v$ for which the total weight of its active neighbors is at least ${\theta }_{v}$ :

如下。每个节点$v$从区间$\left\lbrack  {0,1}\right\rbrack$中均匀随机地选择一个阈值${\theta }_{v}$；这表示$v$的邻居中必须有多大的加权比例变得活跃，才能使$v$变得活跃。给定阈值的随机选择以及初始的活跃节点集${A}_{0}$（其他所有节点都不活跃），扩散过程以离散步骤确定性地展开：在步骤$t$中，所有在步骤$t - 1$中活跃的节点保持活跃，并且我们激活任何其活跃邻居的总权重至少为${\theta }_{v}$的节点$v$：

$$
\mathop{\sum }\limits_{{w\text{ active neighbor of }v}}{b}_{v,w} \geq  {\theta }_{v}.
$$

Thus,the thresholds ${\theta }_{v}$ intuitively represent the different latent tendencies of nodes to adopt the innovation when their neighbors do; the fact that these are randomly selected is intended to model our lack of knowledge of their values - we are in effect averaging over possible threshold values for all the nodes. (Another class of approaches hard-wires all thresholds at a known value like $1/2$ ; see for example work by Berger [5], Morris [21], and Peleg [25].)

因此，阈值${\theta }_{v}$直观地表示了节点在其邻居采用创新时采用该创新的不同潜在倾向；这些阈值是随机选择的，旨在模拟我们对其值的未知情况——实际上，我们是在对所有节点的可能阈值值进行平均。（另一类方法将所有阈值硬编码为已知值，如$1/2$；例如，参见伯杰（Berger）[5]、莫里斯（Morris）[21]和佩雷格（Peleg）[25]的工作。）

Based on work in interacting particle systems [11, 17] from probability theory, we can also consider dynamic cascade models for diffusion processes. The conceptually simplest model of this type is what one could call the Independent Cascade Model, investigated recently in the context of marketing by Goldenberg, Libai, and Muller [13, 14]. We again start with an initial set of active nodes ${A}_{0}$ ,and the process unfolds in discrete steps according to the following randomized rule. When node $v$ first becomes active in step $t$ ,it is given a single chance to activate each currently inactive neighbor $w$ ; it succeeds with a probability ${p}_{v,w} -$ a parameter of the system - independently of the history thus far. (If $w$ has multiple newly activated neighbors, their attempts are sequenced in an arbitrary order.) If $v$ succeeds,then $w$ will become active in step $t + 1$ ; but whether or not $v$ succeeds,it cannot make any further attempts to activate $w$ in subsequent rounds. Again,the process runs until no more activations are possible.

基于概率论中相互作用粒子系统的研究成果[11, 17]，我们还可以考虑用于扩散过程的动态级联模型。这类模型在概念上最简单的一种可称为独立级联模型（Independent Cascade Model），最近戈德堡（Goldenberg）、利拜（Libai）和米勒（Muller）在市场营销领域对其进行了研究[13, 14]。我们再次从一组初始活跃节点${A}_{0}$开始，该过程按照以下随机规则以离散步骤展开。当节点$v$在步骤$t$首次变为活跃节点时，它有一次机会激活每个当前不活跃的邻居节点$w$；激活成功的概率为${p}_{v,w} -$（这是系统的一个参数），且与到目前为止的历史情况无关。（如果$w$有多个新激活的邻居节点，它们的尝试顺序是任意的。）如果$v$激活成功，那么$w$将在步骤$t + 1$变为活跃节点；但无论$v$是否激活成功，在后续轮次中它都不能再尝试激活$w$。同样，该过程会一直运行，直到无法再进行激活为止。

The Linear Threshold and Independent Cascade Models are two of the most basic and widely-studied diffusion models, but of course many extensions can be considered. We will turn to this issue later in the paper, proposing a general framework that simultaneously includes both of these models as special cases. For the sake of concreteness in the introduction, we will discuss our results in terms of these two models in particular.

线性阈值模型（Linear Threshold Model）和独立级联模型是两种最基本且研究最为广泛的扩散模型，但当然可以考虑许多扩展情况。我们将在本文后面讨论这个问题，提出一个通用框架，将这两种模型同时作为特殊情况包含在内。为了在引言中表述得更具体，我们将特别针对这两种模型来讨论我们的研究结果。

Approximation Algorithms for Influence Maximization. We are now in a position to formally express the Domingos-Richardson style of optimization problem - choosing a good initial set of nodes to target - in the context of the above models. Both the Linear Threshold and Independent Cascade Models (as well as the generalizations to follow) involve an initial set of active nodes ${A}_{0}$ that start the diffusion process. We define the influence of a set of nodes $A$ ,denoted $\sigma \left( A\right)$ ,to be the expected number of active nodes at the end of the process,given that $A$ is this initial active set ${A}_{0}$ . The influence maximization problem asks,for a parameter $k$ ,to find a $k$ -node set of maximum influence. (When dealing with algorithms for this problem,we will say that the chosen set $A$ of $k$ initial active nodes has been targeted for activation by the algorithm.) For the models we consider, it is NP-hard to determine the optimum for influence maximization, as we will show later.

影响力最大化的近似算法。现在我们可以在上述模型的背景下，正式表述多明戈斯 - 理查森（Domingos - Richardson）风格的优化问题，即选择一组合适的初始节点作为目标。线性阈值模型和独立级联模型（以及后续的推广模型）都涉及一组初始活跃节点${A}_{0}$，它们启动扩散过程。我们将节点集$A$的影响力定义为（记为$\sigma \left( A\right)$），在给定$A$为初始活跃节点集${A}_{0}$的情况下，过程结束时活跃节点的期望数量。影响力最大化问题要求，对于参数$k$，找到一个具有最大影响力的$k$节点集。（在处理该问题的算法时，我们会说算法选择的$k$个初始活跃节点集$A$已被选定进行激活。）对于我们所考虑的模型，正如我们稍后将证明的，确定影响力最大化的最优解是NP难问题。

Our first main result is that the optimal solution for influence maximization can be efficiently approximated to within a factor of $\left( {1 - 1/e - \varepsilon }\right)$ ,in both the Linear Threshold and Independent Cascade models; here $e$ is the base of the natural logarithm and $\varepsilon$ is any positive real number. (Thus,this is a performance guarantee slightly better than ${63}\%$ .) The algorithm that achieves this performance guarantee is a natural greedy hill-climbing strategy related to the approach considered in [10], and so the main content of this result is the analysis framework needed for obtaining a provable performance guarantee, and the fairly surprising fact that hill-climbing is always within a factor of at least ${63}\%$ of optimal for this problem. We prove this result in Section 2 using techniques from the theory of submodular functions $\left\lbrack  {9,{23}}\right\rbrack$ ,which we describe in detail below, and which turn out to provide a natural context for reasoning about both models and algorithms for influence maximization.

我们的第一个主要结果是，在线性阈值模型和独立级联模型中，影响力最大化的最优解都可以被有效近似到$\left( {1 - 1/e - \varepsilon }\right)$的因子范围内；这里$e$是自然对数的底数，$\varepsilon$是任意正实数。（因此，这是一个略优于${63}\%$的性能保证。）实现这一性能保证的算法是一种自然的贪心爬山策略，与文献[10]中考虑的方法相关，因此这一结果的主要内容是获得可证明的性能保证所需的分析框架，以及一个相当令人惊讶的事实，即对于这个问题，爬山算法的结果总是至少达到最优解的${63}\%$倍。我们在第2节中使用子模函数理论$\left\lbrack  {9,{23}}\right\rbrack$的技术证明了这一结果，下面我们将详细描述该理论，事实证明它为推理影响力最大化的模型和算法提供了一个自然的背景。

In fact, this analysis framework allows us to design and prove guarantees for approximation algorithms in much richer and more realistic models of the processes by which we market to nodes. The deterministic activation of individual nodes is a highly simplified model; an issue also considered in $\left\lbrack  {{10},{26}}\right\rbrack$ is that we may in reality have a large number of different marketing actions available, each of which may influence nodes in different ways. The available budget can be divided arbitrarily between these actions. We show how to extend the analysis to this substantially more general framework. Our main result here is that a generalization of the hill-climbing algorithm still provides approximation guarantees arbitrarily close to $\left( {1 - 1/e}\right)$ .

事实上，这个分析框架使我们能够在更丰富、更现实的向节点进行营销的过程模型中设计近似算法并证明其保证。单个节点的确定性激活是一个高度简化的模型；文献$\left\lbrack  {{10},{26}}\right\rbrack$中也考虑到的一个问题是，实际上我们可能有大量不同的营销行动可供选择，每种行动可能以不同的方式影响节点。可用预算可以在这些行动之间任意分配。我们展示了如何将分析扩展到这个实质上更通用的框架。我们在这里的主要结果是，爬山算法的一种推广形式仍然能提供任意接近$\left( {1 - 1/e}\right)$的近似保证。

It is worth briefly considering the general issue of performance guarantees for algorithms in these settings. For both the Linear Threshold and the Independent Cascade models, the influence maximization problem is NP-complete, but it can be approximated well. In the linear model of Richardson and Domingos [26], on the other hand, both the propagation of influence as well as the effect of the initial targeting are linear. Initial marketing decisions here are thus limited in their effect on node activations; each node's probability of activation is obtained as a linear combination of the effect of targeting and the effect of the neighbors. In this fully linear model, the influence can be maximized by solving a system of linear equations. In contrast, we can show that general models like that of Domingos and Richardson [10], and even simple models that build in a fixed threshold (like $1/2$ ) at all nodes $\left\lbrack  {5,{21},{25}}\right\rbrack$ ,lead to influence maximization problems that cannot be approximated to within any non-trivial factor,assuming $\mathrm{P} \neq  \mathrm{{NP}}$ . Our analysis of approx-imability thus suggests a way of tracing out a more delicate boundary of tractability through the set of possible models, by helping to distinguish among those for which simple heuristics provide strong performance guarantees and those for which they can be arbitrarily far from optimal. This in turn can suggest the development of both more powerful algorithms, and the design of accurate models that simultaneously allow for tractable optimization.

值得简要考虑一下在这些设定下算法性能保证的一般性问题。对于线性阈值（Linear Threshold）模型和独立级联（Independent Cascade）模型而言，影响力最大化问题是NP完全问题，但它可以得到较好的近似解。另一方面，在理查森（Richardson）和多明戈斯（Domingos）[26]提出的线性模型中，影响力的传播以及初始目标设定的效果都是线性的。因此，这里的初始营销决策对节点激活的影响是有限的；每个节点的激活概率是目标设定效果和邻居节点效果的线性组合。在这个完全线性的模型中，可以通过求解一个线性方程组来实现影响力的最大化。相比之下，我们可以证明，像多明戈斯和理查森[10]提出的一般模型，甚至是在所有节点都设定固定阈值（如$1/2$ ）的简单模型$\left\lbrack  {5,{21},{25}}\right\rbrack$ ，都会导致影响力最大化问题无法在任何非平凡因子范围内得到近似解，假设$\mathrm{P} \neq  \mathrm{{NP}}$ 成立。因此，我们对近似性的分析为通过可能的模型集合描绘出更精细的可处理性边界提供了一种方法，有助于区分哪些模型可以通过简单的启发式方法提供强大的性能保证，哪些模型的启发式方法可能与最优解相差甚远。这反过来又可以为开发更强大的算法以及设计既能进行可处理优化又准确的模型提供思路。

Following the approximation and NP-hardness results, we describe in Section 3 the results of computational experiments with both the Linear Threshold and Independent Cascade Models, showing that the hill-climbing algorithm significantly out-performs strategies based on targeting high-degree or "central" nodes [30]. In Section 4 we then develop a general model of diffusion processes in social networks that simultaneously generalizes the Linear Threshold and Independent Cascade Models, as well as a number of other natural cases, and we show how to obtain approximation guarantees for a large sub-class of these models. In Sections 5 and 6, we also consider extensions of our approximation algorithms to models with more realistic scenarios in mind: more complex marketing actions as discussed above, and non-progressive processes, in which active nodes may become inactive in subsequent steps.

在得出近似性和NP难问题的结果之后，我们在第3节描述了对线性阈值模型和独立级联模型进行计算实验的结果，结果表明爬山算法的性能明显优于基于针对高度数节点或“中心”节点的策略[30]。在第4节中，我们开发了一个社交网络中扩散过程的通用模型，该模型同时推广了线性阈值模型和独立级联模型以及其他一些自然情况，并且展示了如何为这些模型的一个大类子模型获得近似保证。在第5节和第6节中，我们还考虑将近似算法扩展到更符合现实场景的模型：如上文讨论的更复杂的营销行动，以及非渐进式过程（在该过程中，活跃节点可能在后续步骤中变为非活跃节点）。

## 2. APPROXIMATIONGUARANTEES INTHE INDEPENDENT CASCADE AND LINEAR THRESHOLD MODELS

## 2. 独立级联模型和线性阈值模型中的近似保证

The overall approach. We begin by describing our strategy for proving approximation guarantees. Consider an arbitrary function $f\left( \cdot \right)$ that maps subsets of a finite ground set $U$ to non-negative real numbers. ${}^{1}$ We say that $f$ is submodular if it satisfies a natural "diminishing returns" property: the marginal gain from adding an element to a set $S$ is at least as high as the marginal gain from adding the same element to a superset of $S$ . Formally,a submodular function satisfies

总体方法。我们首先描述证明近似保证的策略。考虑一个任意函数$f\left( \cdot \right)$ ，它将有限基集$U$ 的子集映射到非负实数。${}^{1}$ 我们称$f$ 是次模函数，如果它满足一个自然的“收益递减”性质：向集合$S$ 中添加一个元素所带来的边际收益至少与向$S$ 的超集添加相同元素所带来的边际收益一样高。形式上，次模函数满足

$$
f\left( {S\cup \{ v\} }\right)  - f\left( S\right)  \geq  f\left( {T\cup \{ v\} }\right)  - f\left( T\right) ,
$$

for all elements $v$ and all pairs of sets $S \subseteq  T$ .

对于所有元素$v$ 和所有集合对$S \subseteq  T$ 。

Submodular functions have a number of very nice tractability properties; the one that is relevant to us here is the following. Suppose we have a function $f$ that is submodular,takes only nonnegative values, and is monotone in the sense that adding an element to a set cannot cause $f$ to decrease: $f\left( {S\cup \{ v\} }\right)  \geq  f\left( S\right)$ for all elements $v$ and sets $S$ . We wish to find a $k$ -element set $S$ for which $f\left( S\right)$ is maximized. This is an NP-hard optimization problem (it can be shown to contain the Hitting Set problem as a simple special case), but a result of Nemhauser, Wolsey, and Fisher $\left\lbrack  {9,{23}}\right\rbrack$ shows that the following greedy hill-climbing algorithm approximates the optimum to within a factor of $\left( {1 - 1/e}\right)$ (where $e$ is the base of the natural logarithm): start with the empty set, and repeatedly add an element that gives the maximum marginal gain.

次模函数具有许多非常好的可处理性性质；这里与我们相关的性质如下。假设我们有一个次模函数$f$ ，它只取非负值，并且具有单调性，即向集合中添加一个元素不会导致$f$ 的值减小：对于所有元素$v$ 和集合$S$ ，有$f\left( {S\cup \{ v\} }\right)  \geq  f\left( S\right)$ 。我们希望找到一个包含$k$ 个元素的集合$S$ ，使得$f\left( S\right)$ 达到最大值。这是一个NP难的优化问题（可以证明它包含命中集问题作为一个简单的特殊情况），但内姆豪瑟（Nemhauser）、沃尔西（Wolsey）和费舍尔（Fisher）$\left\lbrack  {9,{23}}\right\rbrack$ 的一个结果表明，以下贪心爬山算法可以在$\left( {1 - 1/e}\right)$ 因子范围内近似最优解（其中$e$ 是自然对数的底数）：从空集开始，反复添加能带来最大边际收益的元素。

THEOREM 2.1. $\left\lbrack  {9,{23}}\right\rbrack$ For a non-negative,monotone submod-ular function $f$ ,let $S$ be a set of size $k$ obtained by selecting elements one at a time, each time choosing an element that provides the largest marginal increase in the function value. Let ${S}^{ * }$ be a set that maximizes the value of $f$ over all $k$ -element sets. Then $f\left( S\right)  \geq  \left( {1 - 1/e}\right)  \cdot  f\left( {S}^{ * }\right)$ ; in other words, $S$ provides a $\left( {1 - 1/e}\right)$ - approximation.

定理2.1. $\left\lbrack  {9,{23}}\right\rbrack$ 对于一个非负的单调次模函数 $f$ ，设 $S$ 是一个通过逐个选择元素得到的大小为 $k$ 的集合，每次选择的元素能使函数值的边际增量最大。设 ${S}^{ * }$ 是在所有大小为 $k$ 的集合中使 $f$ 的值达到最大的集合。那么 $f\left( S\right)  \geq  \left( {1 - 1/e}\right)  \cdot  f\left( {S}^{ * }\right)$ ；换句话说， $S$ 提供了一个 $\left( {1 - 1/e}\right)$ -近似解。

Due to its generality, this result has found applications in a number of areas of discrete optimization (see e.g. [22]); the only direct use of it that we are aware of in the databases and data mining literature is in a context very different from ours, for the problem of selecting database views to materialize [16].

由于其通用性，这一结果已在离散优化的多个领域得到应用（例如，参见 [22]）；据我们所知，在数据库和数据挖掘文献中，对该结果的唯一直接应用是在与我们的场景非常不同的情境中，用于选择要物化的数据库视图的问题 [16]。

Our strategy will be to show that for the models we are considering,the resulting influence function $\sigma \left( \cdot \right)$ is submodular. A subtle difficulty lies in the fact that the result of Nemhauser et al. assumes that the greedy algorithm can evaluate the underlying function exactly,which may not be the case for the influence function $\sigma \left( A\right)$ . However, by simulating the diffusion process and sampling the resulting active sets, we are able to obtain arbitrarily close approximations to $\sigma \left( A\right)$ ,with high probability. Furthermore,one can extend the result of Nemhauser et al. to show that for any $\varepsilon  > 0$ ,there is a $\gamma  > 0$ such that by using $\left( {1 + \gamma }\right)$ -approximate values for the function to be optimized,we obtain a $\left( {1 - 1/e - \varepsilon }\right)$ -approximation.

我们的策略是证明对于我们所考虑的模型，所得到的影响函数 $\sigma \left( \cdot \right)$ 是次模的。一个微妙的困难在于，内姆豪泽（Nemhauser）等人的结果假设贪心算法可以精确评估底层函数，而对于影响函数 $\sigma \left( A\right)$ 来说可能并非如此。然而，通过模拟扩散过程并对得到的活跃集进行采样，我们能够以高概率获得与 $\sigma \left( A\right)$ 任意接近的近似值。此外，可以扩展内姆豪泽等人的结果，证明对于任意 $\varepsilon  > 0$ ，存在一个 $\gamma  > 0$ ，使得通过使用待优化函数的 $\left( {1 + \gamma }\right)$ -近似值，我们可以得到一个 $\left( {1 - 1/e - \varepsilon }\right)$ -近似解。

As mentioned in the introduction, we can extend this analysis to a general model with more complex marketing actions that can have a probabilistic effect on the initial activation of nodes. We show in Section 6 how, with a more careful hill-climbing algorithm and a generalization of Theorem 2.1, we can obtain comparable approximation guarantees in this setting.

如引言中所述，我们可以将这种分析扩展到一个更一般的模型，该模型具有更复杂的营销行动，这些行动可能对节点的初始激活产生概率性影响。我们将在第6节展示，通过更精细的爬山算法和定理2.1的推广，我们如何在这种情况下获得可比的近似保证。

A further extension is to assume that each node $v$ has an associated non-negative weight ${w}_{v}$ ,capturing how important it is that $v$ be activated in the final outcome. (For instance,if we are marketing textbooks to college teachers, then the weight could be the number of students in the teacher's class, resulting in a larger or smaller number of sales.) If we let $B$ denote the (random) set activated by the process with initial activation $A$ ,then we can define the weighted influence function ${\sigma }_{w}\left( A\right)$ to be the expected value over outcomes $B$ of the quantity $\mathop{\sum }\limits_{{v \in  B}}{w}_{v}$ . The influence function studied above is the special case obtained by setting ${w}_{v} = 1$ for all nodes $v$ . The objective function with weights is submodular whenever the unweighted version is, so we can still use the greedy algorithm for obtaining a $\left( {1 - 1/e - \varepsilon }\right)$ -approximation. Note,however, that a sampling algorithm to approximately choose the next element may need time that depends on the sizes of the weights.

进一步的扩展是假设每个节点 $v$ 都有一个相关的非负权重 ${w}_{v}$ ，该权重反映了在最终结果中激活 $v$ 的重要程度。（例如，如果我们向大学教师推销教科书，那么权重可以是该教师班级的学生人数，这会导致销售量或多或少。）如果我们用 $B$ 表示在初始激活 $A$ 的情况下该过程激活的（随机）集合，那么我们可以将加权影响函数 ${\sigma }_{w}\left( A\right)$ 定义为数量 $\mathop{\sum }\limits_{{v \in  B}}{w}_{v}$ 在所有结果 $B$ 上的期望值。上面研究的影响函数是通过对所有节点 $v$ 设置 ${w}_{v} = 1$ 得到的特殊情况。只要未加权版本的目标函数是次模的，带权重的目标函数也是次模的，因此我们仍然可以使用贪心算法来获得一个 $\left( {1 - 1/e - \varepsilon }\right)$ -近似解。然而，需要注意的是，一个用于近似选择下一个元素的采样算法可能需要依赖于权重大小的时间。

---

<!-- Footnote -->

${}^{1}$ Note that the influence function $\sigma \left( \cdot \right)$ defined above has this form; it maps each subset $A$ of the nodes of the social network to a real number denoting the expected size of the activated set if $A$ is targeted for initial activation.

${}^{1}$ 注意，上面定义的影响函数 $\sigma \left( \cdot \right)$ 具有这种形式；它将社交网络节点的每个子集 $A$ 映射到一个实数，该实数表示如果以 $A$ 为初始激活目标时激活集的期望大小。

<!-- Footnote -->

---

## Independent Cascade

## 独立级联模型

In view of the above discussion, an approximation guarantee for influence maximization in the Independent Cascade Model will be a consequence of the following

鉴于上述讨论，独立级联模型中影响最大化的近似保证将是以下内容的结果

THEOREM 2.2. For an arbitrary instance of the Independent Cascade Model,the resulting influence function $\sigma \left( \cdot \right)$ is submodu-lar.

定理2.2. 对于独立级联模型的一个实例，所得到的影响函数 $\sigma \left( \cdot \right)$ 是次模的。

In order to establish this result, we need to look, implicitly or explicitly,at the expression $\sigma \left( {A\cup \{ v\} }\right)  - \sigma \left( A\right)$ ,for arbitrary sets $A$ and elements $v$ . In other words,what increase do we get in the expected number of overall activations when we add $v$ to the set $A$ ? This increase is very difficult to analyze directly,because it is hard to work with quantities of the form $\sigma \left( A\right)$ . For example,the Independent Cascade process is underspecified, since we have not prescribed the order in which newly activated nodes in a given step $t$ will attempt to activate their neighbors. Thus,it is not initially obvious that the process is even well-defined, in the sense that it yields the same distribution over outcomes regardless of how we schedule the attempted activations.

为了证明这一结果，我们需要隐式或显式地考察表达式 $\sigma \left( {A\cup \{ v\} }\right)  - \sigma \left( A\right)$，其中 $A$ 为任意集合，$v$ 为任意元素。换句话说，当我们将 $v$ 添加到集合 $A$ 中时，总体激活次数的期望值会增加多少呢？这种增加很难直接分析，因为处理形如 $\sigma \left( A\right)$ 的量很困难。例如，独立级联过程的定义并不完备，因为我们没有规定在给定步骤 $t$ 中新激活的节点尝试激活其邻居节点的顺序。因此，从该过程无论我们如何安排激活尝试都能产生相同的结果分布这一意义上来说，该过程是否定义明确最初并不明显。

Our proof deals with these difficulties by formulating an equivalent view of the process, which makes it easier to see that there is an order-independent outcome, and which provides an alternate way to reason about the submodularity property.

我们的证明通过对该过程提出一种等价的观点来解决这些困难，这种观点使我们更容易看出存在与顺序无关的结果，并为推理次模性属性提供了另一种方法。

Consider a point in the cascade process when node $v$ has just become active,and it attempts to activate its neighbor $w$ ,succeeding with probability ${p}_{v,w}$ . We can view the outcome of this random event as being determined by flipping a coin of bias ${p}_{v,w}$ . From the point of view of the process, it clearly does not matter whether the coin was flipped at the moment that $v$ became active,or whether it was flipped at the very beginning of the whole process and is only being revealed now. Continuing this reasoning, we can in fact assume that for each pair of neighbors(v,w)in the graph,a coin of bias ${p}_{v,w}$ is flipped at the very beginning of the process (independently of the coins for all other pairs of neighbors), and the result is stored so that it can be later checked in the event that $v$ is activated while $w$ is still inactive.

考虑级联过程中的一个时刻，此时节点 $v$ 刚刚被激活，并尝试激活其邻居节点 $w$，激活成功的概率为 ${p}_{v,w}$。我们可以将这一随机事件的结果视为由抛掷一枚偏向概率为 ${p}_{v,w}$ 的硬币决定。从过程的角度来看，硬币是在节点 $v$ 被激活的那一刻抛掷，还是在整个过程开始时就抛掷而现在才揭晓结果，显然并不重要。继续这种推理，实际上我们可以假设，对于图中的每一对邻居节点 (v, w)，在过程开始时就抛掷一枚偏向概率为 ${p}_{v,w}$ 的硬币（与所有其他邻居节点对的硬币抛掷相互独立），并存储结果，以便在 $v$ 被激活而 $w$ 仍未被激活时进行检查。

With all the coins flipped in advance, the process can be viewed as follows. The edges in $G$ for which the coin flip indicated an activation will be successful are declared to be live; the remaining edges are declared to be blocked. If we fix the outcomes of the coin flips and then initially activate a set $A$ ,it is clear how to determine the full set of active nodes at the end of the cascade process:

如果预先抛掷所有硬币，该过程可以如下看待。对于 $G$ 中的边，如果硬币抛掷结果表明激活会成功，则将这些边标记为有效边；其余边标记为阻塞边。如果我们确定了硬币抛掷的结果，然后最初激活一个集合 $A$，那么很清楚如何确定级联过程结束时的所有激活节点集合：

CLAIM 2.3. A node $x$ ends up active if and only if there is a path from some node in $A$ to $x$ consisting entirely of live edges. (We will call such a path a live-edge path.)

命题 2.3：节点 $x$ 最终被激活，当且仅当存在一条从 $A$ 中的某个节点到 $x$ 的完全由有效边组成的路径。（我们将这样的路径称为有效边路径。）

Consider the probability space in which each sample point specifies one possible set of outcomes for all the coin flips on the edges. Let $X$ denote one sample point in this space,and define ${\sigma }_{X}\left( A\right)$ to be the total number of nodes activated by the process when $A$ is the set initially targeted,and $X$ is the set of outcomes of all coin flips on edges. Because we have fixed a choice for $X,{\sigma }_{X}\left( A\right)$ is in fact a deterministic quantity, and there is a natural way to express its value,as follows. Let $R\left( {v,X}\right)$ denote the set of all nodes that can be reached from $v$ on a path consisting entirely of live edges. By Claim 2.3, ${\sigma }_{X}\left( A\right)$ is the number of nodes that can be reached on live-edge paths from any node in $A$ ,and so it is equal to the cardinality of the union ${ \cup  }_{v \in  A}R\left( {v,X}\right)$ .

考虑一个概率空间，其中每个样本点指定了所有边上硬币抛掷的一种可能结果集。设 $X$ 表示该空间中的一个样本点，并定义 ${\sigma }_{X}\left( A\right)$ 为当 $A$ 是最初目标集合，且 $X$ 是所有边上硬币抛掷的结果集时，该过程激活的节点总数。因为我们已经确定了 $X,{\sigma }_{X}\left( A\right)$ 的选择，所以 ${\sigma }_{X}\left( A\right)$ 实际上是一个确定性的量，并且有一种自然的方式来表示其值，如下所示。设 $R\left( {v,X}\right)$ 表示可以通过完全由有效边组成的路径从 $v$ 到达的所有节点的集合。根据命题 2.3，${\sigma }_{X}\left( A\right)$ 是可以通过有效边路径从 $A$ 中的任何节点到达的节点数，因此它等于并集 ${ \cup  }_{v \in  A}R\left( {v,X}\right)$ 的基数。

Proof of Theorem 2.2. First, we claim that for each fixed outcome $X$ ,the function ${\sigma }_{X}\left( \cdot \right)$ is submodular. To see this,let $S$ and $T$ be two sets of nodes such that $S \subseteq  T$ ,and consider the quantity ${\sigma }_{X}\left( {S\cup \{ v\} }\right)  - {\sigma }_{X}\left( S\right)$ . This is the number of elements in $R\left( {v,X}\right)$ that are not already in the union ${ \cup  }_{u \in  S}R\left( {u,X}\right)$ ; it is at least as large as the number of elements in $R\left( {v,X}\right)$ that are not in the (bigger) union ${ \cup  }_{u \in  T}R\left( {u,X}\right)$ . It follows that ${\sigma }_{X}\left( {S\cup \{ v\} }\right)  - {\sigma }_{X}\left( S\right)  \geq$ ${\sigma }_{X}\left( {T\cup \{ v\} }\right)  - {\sigma }_{X}\left( T\right)$ ,which is the defining inequality for sub-modularity. Finally, we have

定理2.2的证明。首先，我们声称对于每个固定的结果 $X$ ，函数 ${\sigma }_{X}\left( \cdot \right)$ 是次模的。为了说明这一点，设 $S$ 和 $T$ 是两组节点，使得 $S \subseteq  T$ ，并考虑量 ${\sigma }_{X}\left( {S\cup \{ v\} }\right)  - {\sigma }_{X}\left( S\right)$ 。这是 $R\left( {v,X}\right)$ 中尚未在并集 ${ \cup  }_{u \in  S}R\left( {u,X}\right)$ 中的元素数量；它至少和 $R\left( {v,X}\right)$ 中不在（更大的）并集 ${ \cup  }_{u \in  T}R\left( {u,X}\right)$ 中的元素数量一样大。由此可得 ${\sigma }_{X}\left( {S\cup \{ v\} }\right)  - {\sigma }_{X}\left( S\right)  \geq$ ${\sigma }_{X}\left( {T\cup \{ v\} }\right)  - {\sigma }_{X}\left( T\right)$ ，这就是次模性的定义不等式。最后，我们有

$$
\sigma \left( A\right)  = \mathop{\sum }\limits_{{\text{outcomes }X}}\operatorname{Prob}\left\lbrack  X\right\rbrack   \cdot  {\sigma }_{X}\left( A\right) ,
$$

since the expected number of nodes activated is just the weighted average over all outcomes. But a non-negative linear combination of submodular functions is also submodular,and hence $\sigma \left( \cdot \right)$ is sub-modular, which concludes the proof.

因为被激活的节点的期望数量只是所有结果的加权平均值。但是次模函数的非负线性组合也是次模的，因此 $\sigma \left( \cdot \right)$ 是次模的，这就完成了证明。

Next we show the hardness of influence maximization.

接下来我们证明影响力最大化问题的难度。

THEOREM 2.4. The influence maximization problem is NP-hard for the Independent Cascade model.

定理2.4。对于独立级联模型（Independent Cascade model），影响力最大化问题是NP难的。

Proof. Consider an instance of the NP-complete Set Cover problem,defined by a collection of subsets ${S}_{1},{S}_{2},\ldots ,{S}_{m}$ of a ground set $U = \left\{  {{u}_{1},{u}_{2},\ldots ,{u}_{n}}\right\}$ ; we wish to know whether there exist $k$ of the subsets whose union is equal to $U$ . (We can assume that $k < n < m$ .) We show that this can be viewed as a special case of the influence maximization problem.

证明。考虑一个NP完全的集合覆盖问题（Set Cover problem）的实例，由一个基础集合 $U = \left\{  {{u}_{1},{u}_{2},\ldots ,{u}_{n}}\right\}$ 的子集集合 ${S}_{1},{S}_{2},\ldots ,{S}_{m}$ 定义；我们想知道是否存在 $k$ 个这样的子集，它们的并集等于 $U$ 。（我们可以假设 $k < n < m$ 。）我们证明这可以被看作是影响力最大化问题的一个特殊情况。

Given an arbitrary instance of the Set Cover problem, we define a corresponding directed bipartite graph with $n + m$ nodes: there is a node $i$ corresponding to each set ${S}_{i}$ ,a node $j$ corresponding to each element ${u}_{j}$ ,and a directed edge(i,j)with activation probability ${p}_{i,j} = 1$ whenever ${u}_{j} \in  {S}_{i}$ . The Set Cover problem is equivalent to deciding if there is a set $A$ of $k$ nodes in this graph with $\sigma \left( A\right)  \geq  n + k$ . Note that for the instance we have defined, activation is a deterministic process, as all probabilities are 0 or 1. Initially activating the $k$ nodes corresponding to sets in a Set Cover solution results in activating all $n$ nodes corresponding to the ground set $U$ ,and if any set $A$ of $k$ nodes has $\sigma \left( A\right)  \geq  n + k$ , then the Set Cover problem must be solvable.

给定集合覆盖问题的任意一个实例，我们定义一个对应的有 $n + m$ 个节点的有向二分图：有一个对应于每个集合 ${S}_{i}$ 的节点 $i$ ，一个对应于每个元素 ${u}_{j}$ 的节点 $j$ ，并且只要 ${u}_{j} \in  {S}_{i}$ ，就有一条激活概率为 ${p}_{i,j} = 1$ 的有向边 (i, j) 。集合覆盖问题等价于判断在这个图中是否存在一个由 $k$ 个节点组成的集合 $A$ ，使得 $\sigma \left( A\right)  \geq  n + k$ 。注意，对于我们定义的这个实例，激活是一个确定性过程，因为所有概率都是0或1。最初激活对应于集合覆盖问题解中的集合的 $k$ 个节点会导致激活对应于基础集合 $U$ 的所有 $n$ 个节点，并且如果任意由 $k$ 个节点组成的集合 $A$ 满足 $\sigma \left( A\right)  \geq  n + k$ ，那么集合覆盖问题一定是可解的。

## Linear Thresholds

## 线性阈值

We now prove an analogous result for the Linear Threshold Model.

我们现在为线性阈值模型（Linear Threshold Model）证明一个类似的结果。

THEOREM 2.5. For an arbitrary instance of the Linear Threshold Model,the resulting influence function $\sigma \left( \cdot \right)$ is submodular.

定理2.5。对于线性阈值模型的任意一个实例，得到的影响力函数 $\sigma \left( \cdot \right)$ 是次模的。

Proof. The analysis is a bit more intricate than in the proof of Theorem 2.2, but the overall argument has a similar structure. In the proof of Theorem 2.2, we constructed an equivalent process by initially resolving the outcomes of some random choices, considering each outcome in isolation, and then averaging over all outcomes. For the Linear Threshold Model, the simplest analogue would be to consider the behavior of the process after all node thresholds have been chosen. Unfortunately, for a fixed choice of thresholds, the number of activated nodes is not in general a submodular function of the targeted set; this fact necessitates a more subtle analysis.

证明。这个分析比定理2.2的证明要复杂一些，但整体论证结构类似。在定理2.2的证明中，我们通过最初确定一些随机选择的结果、单独考虑每个结果，然后对所有结果求平均，构造了一个等价的过程。对于线性阈值模型，最简单的类比是考虑在所有节点阈值都被选定之后这个过程的行为。不幸的是，对于固定的阈值选择，被激活的节点数量通常不是目标集合的次模函数；这一事实使得需要进行更细致的分析。

Recall that each node $v$ has an influence weight ${b}_{v,w} \geq  0$ from each of its neighbors $w$ ,subject to the constraint that $\mathop{\sum }\limits_{w}{b}_{v,w} \leq  1$ . (We can extend the notation by writing ${b}_{v,w} = 0$ when $w$ is not a neighbor of $v$ .) Suppose that $v$ picks at most one of its incoming edges at random,selecting the edge from $w$ with probability ${b}_{v,w}$ and selecting no edge with probability $1 - \mathop{\sum }\limits_{w}{b}_{v,w}$ . The selected edge is declared to be "live," and all other edges are declared to be "blocked." (Note the contrast with the proof of Theorem 2.2: there, we determined whether an edge was live independently of the decision for each other edge; here, we negatively correlate the decisions so that at most one live edge enters each node.)

回顾一下，每个节点 $v$ 都有来自其每个邻居节点 $w$ 的影响权重 ${b}_{v,w} \geq  0$，需满足约束条件 $\mathop{\sum }\limits_{w}{b}_{v,w} \leq  1$。（当 $w$ 不是 $v$ 的邻居节点时，我们可以通过写作 ${b}_{v,w} = 0$ 来扩展该表示法。）假设 $v$ 最多随机选择一条入边，以概率 ${b}_{v,w}$ 选择来自 $w$ 的边，以概率 $1 - \mathop{\sum }\limits_{w}{b}_{v,w}$ 不选择任何边。被选中的边被称为“活跃边”，其他所有边被称为“阻塞边”。（注意与定理 2.2 的证明形成对比：在那里，我们独立于其他每条边的决策来确定一条边是否活跃；在这里，我们使这些决策具有负相关性，使得每个节点最多有一条活跃边进入。）

The crux of the proof lies in establishing Claim 2.6 below, which asserts that the Linear Threshold model is equivalent to reachability via live-edge paths as defined above. Once that equivalence is established, submodularity follows exactly as in the proof of Theorem 2.2. We can define $R\left( {v,X}\right)$ as before to be the set of all nodes reachable from $v$ on live-edge paths,subject to a choice $X$ of live/blocked designations for all edges; it follows that ${\sigma }_{X}\left( A\right)$ is the cardinality of the union ${ \cup  }_{v \in  A}R\left( {v,X}\right)$ ,and hence a submodu-lar function of $A$ ; finally,the function $\sigma \left( \cdot \right)$ is a non-negative linear combination of the functions ${\sigma }_{X}\left( \cdot \right)$ and hence also submodular.

证明的关键在于确立下面的断言 2.6，该断言指出线性阈值模型等同于通过上述定义的活跃边路径的可达性。一旦确立了这种等价性，次模性就会像定理 2.2 的证明那样自然得出。我们可以像之前一样将 $R\left( {v,X}\right)$ 定义为在活跃边路径上从 $v$ 可达的所有节点的集合，这取决于对所有边的活跃/阻塞指定的选择 $X$；由此可知 ${\sigma }_{X}\left( A\right)$ 是并集 ${ \cup  }_{v \in  A}R\left( {v,X}\right)$ 的基数，因此是 $A$ 的一个次模函数；最后，函数 $\sigma \left( \cdot \right)$ 是函数 ${\sigma }_{X}\left( \cdot \right)$ 的非负线性组合，因此也是次模的。

CLAIM 2.6. For a given targeted set $A$ ,the following two distributions over sets of nodes are the same:

断言 2.6. 对于给定的目标集合 $A$，以下两种关于节点集合的分布是相同的：

(i) The distribution over active sets obtained by running the Linear Threshold process to completion starting from $A$ ; and

(i) 从 $A$ 开始运行线性阈值过程直至结束所得到的活跃集合的分布；以及

(ii) The distribution over sets reachable from $A$ via live-edge paths, under the random selection of live edges defined above.

(ii) 在上述活跃边的随机选择下，通过活跃边路径从 $A$ 可达的集合的分布。

Proof. We need to prove that reachability under our random choice of live and blocked edges defines a process equivalent to that of the Linear Threshold Model. To obtain intuition about this equivalence, it is useful to first analyze the special case in which the underlying graph $G$ is directed and acyclic. In this case,we can fix a topological ordering of the nodes ${v}_{1},{v}_{2},\ldots ,{v}_{n}$ (so that all edges go from earlier nodes to later nodes in the order), and build up the distribution of active sets by following this order. For each node ${v}_{i}$ , suppose we already have determined the distribution over active subsets of its neighbors. Then under the Linear Threshold process, the probability that ${v}_{i}$ will become active,given that a subset ${S}_{i}$ of its neighbors is active,is $\mathop{\sum }\limits_{{w \in  {S}_{i}}}{b}_{{v}_{i},w}$ . This is precisely the probability that the live incoming edge selected by ${v}_{i}$ lies in ${S}_{i}$ , and so inductively we see that the two processes define the same distribution over active sets.

证明。我们需要证明在我们对活跃边和阻塞边的随机选择下的可达性定义了一个等同于线性阈值模型的过程。为了直观理解这种等价性，首先分析基础图 $G$ 是有向无环图的特殊情况是很有用的。在这种情况下，我们可以确定节点 ${v}_{1},{v}_{2},\ldots ,{v}_{n}$ 的一个拓扑排序（使得所有边在该顺序中从较早的节点指向较晚的节点），并按照这个顺序构建活跃集合的分布。对于每个节点 ${v}_{i}$，假设我们已经确定了其邻居节点的活跃子集的分布。那么在线性阈值过程下，给定其邻居节点的一个子集 ${S}_{i}$ 是活跃的，${v}_{i}$ 变为活跃的概率是 $\mathop{\sum }\limits_{{w \in  {S}_{i}}}{b}_{{v}_{i},w}$。这恰好是 ${v}_{i}$ 选择的活跃入边位于 ${S}_{i}$ 中的概率，因此通过归纳我们可以看到这两个过程定义了关于活跃集合的相同分布。

To prove the claim generally,consider a graph $G$ that is not acyclic. It becomes trickier to show the equivalence, because there is no natural ordering of the nodes over which to perform induction. Instead, we argue by induction over the iterations of the Linear Threshold process. We define ${A}_{t}$ to be the set of active nodes at the end of iteration $t$ ,for $t = 0,1,2,\ldots$ (note that ${A}_{0}$ is the set initially targeted). If node $v$ has not become active by the end of iteration $t$ ,then the probability that it becomes active in iteration $t + 1$ is equal to the chance that the influence weights in ${A}_{t} \smallsetminus  {A}_{t - 1}$ push it over its threshold, given that its threshold was not exceeded already; this probability is $\frac{\mathop{\sum }\limits_{{u \in  {A}_{t} \smallsetminus  {A}_{t - 1}}}{b}_{v,u}}{1 - \mathop{\sum }\limits_{{u \in  {A}_{t - 1}}}{b}_{v,u}}$ .

为了一般性地证明该命题，考虑一个非无环的图$G$。证明等价性变得更加棘手，因为不存在对节点进行归纳的自然顺序。相反，我们通过对线性阈值过程的迭代进行归纳来论证。我们将${A}_{t}$定义为迭代$t$结束时的活跃节点集合，其中$t = 0,1,2,\ldots$（注意${A}_{0}$是初始目标集合）。如果节点$v$在迭代$t$结束时仍未变得活跃，那么它在迭代$t + 1$中变得活跃的概率等于在其阈值尚未被超过的情况下，${A}_{t} \smallsetminus  {A}_{t - 1}$中的影响权重将其推过阈值的概率；这个概率是$\frac{\mathop{\sum }\limits_{{u \in  {A}_{t} \smallsetminus  {A}_{t - 1}}}{b}_{v,u}}{1 - \mathop{\sum }\limits_{{u \in  {A}_{t - 1}}}{b}_{v,u}}$。

On the other hand, we can run the live-edge process by revealing the identities of the live edges gradually as follows. We start with the targeted set $A$ . For each node $v$ with at least one edge from the set $A$ ,we determine whether $v$ ’s live edge comes from $A$ . If so, then $v$ is reachable; but if not,we keep the source of $v$ ’s live edge unknown,subject to the condition that it comes from outside $A$ . Having now exposed a new set of reachable nodes ${A}_{1}^{\prime }$ in the first stage, we proceed to identify further reachable nodes by performing the same process on edges from ${A}_{1}^{\prime }$ ,and in this way produce sets ${A}_{2}^{\prime },{A}_{3}^{\prime },\ldots$ . If node $v$ has not been determined to be reachable by the end of stage $t$ ,then the probability that it is determined to be reachable in stage $t + 1$ is equal to the chance that its live edge comes from ${A}_{t} \smallsetminus  {A}_{t - 1}$ ,given that its live edge has not come from any of the earlier sets. But this is $\frac{\mathop{\sum }\limits_{{u \in  {A}_{t} \smallsetminus  {A}_{t - 1}}}{b}_{v,u}}{1 - \mathop{\sum }\limits_{{u \in  {A}_{t - 1}}}{b}_{v,u}}$ ,which is the same as in the Linear Threshold process of the previous paragraph. Thus, by induction over these stages, we see that the live-edge process produces the same distribution over active sets as the Linear Threshold process.

另一方面，我们可以通过逐步揭示活跃边的标识来运行活跃边过程，如下所示。我们从目标集合$A$开始。对于每个至少有一条来自集合$A$的边的节点$v$，我们确定$v$的活跃边是否来自$A$。如果是，那么$v$是可达的；但如果不是，我们保持$v$的活跃边的源未知，条件是它来自$A$之外。在第一阶段已经揭示了一组新的可达节点${A}_{1}^{\prime }$之后，我们通过对来自${A}_{1}^{\prime }$的边执行相同的过程来识别更多的可达节点，并以此方式生成集合${A}_{2}^{\prime },{A}_{3}^{\prime },\ldots$。如果节点$v$在阶段$t$结束时仍未被确定为可达，那么它在阶段$t + 1$被确定为可达的概率等于在其活跃边尚未来自任何早期集合的情况下，其活跃边来自${A}_{t} \smallsetminus  {A}_{t - 1}$的概率。但这就是$\frac{\mathop{\sum }\limits_{{u \in  {A}_{t} \smallsetminus  {A}_{t - 1}}}{b}_{v,u}}{1 - \mathop{\sum }\limits_{{u \in  {A}_{t - 1}}}{b}_{v,u}}$，与上一段中的线性阈值过程相同。因此，通过对这些阶段进行归纳，我们看到活跃边过程在活跃集合上产生的分布与线性阈值过程相同。

Influence maximization is hard in this model as well.

在这个模型中，影响力最大化问题同样困难。

THEOREM 2.7. The influence maximization problem is NP-hard for the Linear Threshold model.

定理2.7：对于线性阈值模型，影响力最大化问题是NP难的。

Proof. Consider an instance of the NP-complete Vertex Cover problem defined by an undirected $n$ -node graph $G = \left( {V,E}\right)$ and an integer $k$ ; we want to know if there is a set $S$ of $k$ nodes in $G$ so that every edge has at least one endpoint in $S$ . We show that this can be viewed as a special case of the influence maximization problem.

证明：考虑由一个无向$n$节点图$G = \left( {V,E}\right)$和一个整数$k$定义的NP完全顶点覆盖问题的一个实例；我们想知道在$G$中是否存在一个由$k$个节点组成的集合$S$，使得每条边至少有一个端点在$S$中。我们证明这可以被视为影响力最大化问题的一个特殊情况。

Given an instance of the Vertex Cover problem involving a graph $G$ ,we define a corresponding instance of the influence maximization problem by directing all edges of $G$ in both directions. If there is a vertex cover $S$ of size $k$ in $G$ ,then one can deterministically make $\sigma \left( A\right)  = n$ by targeting the nodes in the set $A = S$ ; conversely,this is the only way to get a set $A$ with $\sigma \left( A\right)  = n$ .

给定一个涉及图$G$的顶点覆盖问题的实例，我们通过将$G$的所有边都双向定向来定义一个相应的影响力最大化问题的实例。如果在$G$中存在一个大小为$k$的顶点覆盖$S$，那么可以通过针对集合$A = S$中的节点确定性地使$\sigma \left( A\right)  = n$；反之，这是获得一个满足$\sigma \left( A\right)  = n$的集合$A$的唯一方法。

In the proofs of both the approximation theorems in this section, we established submodularity by considering an equivalent process in which each node "hard-wired" certain of its incident edges as transmitting influence from neighbors. This turns out to be a proof technique that can be formulated in general terms, and directly applied to give approximability results for other models as well. We discuss this further in the context of the general framework presented in Section 4.

在本节两个近似定理的证明中，我们通过考虑一个等效过程来确立子模性，在这个过程中，每个节点将其某些关联边“硬连接”，以从邻居节点传递影响。事实证明，这是一种可以用通用术语表述的证明技术，并且也能直接应用于为其他模型给出可近似性结果。我们将在第4节介绍的通用框架背景下进一步讨论这一点。

## 3. EXPERIMENTS

## 3. 实验

In addition to obtaining worst-case guarantees on the performance of our approximation algorithm, we are interested in understanding its behavior in practice, and comparing its performance to other heuristics for identifying influential individuals. We find that our greedy algorithm achieves significant performance gains over several widely-used structural measures of influence in social networks [30].

除了获得我们的近似算法性能的最坏情况保证外，我们还对了解其在实际中的行为感兴趣，并将其性能与其他用于识别有影响力个体的启发式方法进行比较。我们发现，与社交网络中几种广泛使用的影响力结构度量方法相比 [30]，我们的贪心算法实现了显著的性能提升。

The Network Data. For evaluation, it is desirable to use a network dataset that exhibits many of the structural features of large-scale social networks. At the same time, we do not address the issue of inferring actual influence parameters from network observations (see e.g. [10, 26]). Thus, for our testbed, we employ a collaboration graph obtained from co-authorships in physics publications, with simple settings of the influence parameters. It has been argued extensively that co-authorship networks capture many of the key features of social networks more generally [24]. The co-authorship data was compiled from the complete list of papers in the high-energy physics theory section of the e-print arXiv (www.arxiv.org). ${}^{2}$

网络数据。为了进行评估，理想情况下应使用一个能展现大规模社交网络诸多结构特征的网络数据集。同时，我们不处理从网络观测中推断实际影响参数的问题（例如，参见 [10, 26]）。因此，对于我们的测试平台，我们采用了一个从物理学出版物的合著关系中获得的合作图，并对影响参数进行了简单设置。已有大量论述表明，合著网络更广泛地捕捉了社交网络的许多关键特征 [24]。合著数据是从预印本 arXiv（www.arxiv.org）高能物理理论部分的完整论文列表中整理而来的。${}^{2}$

The collaboration graph contains a node for each researcher who has at least one paper with co-author(s) in the arXiv database. For each paper with two or more authors, we inserted an edge for each pair of authors (single-author papers were ignored). Notice that this results in parallel edges when two researchers have co-authored multiple papers - we kept these parallel edges as they can be interpreted to indicate stronger social ties between the researchers involved. The resulting graph has 10748 nodes, and edges between about 53000 pairs of nodes.

合作图为 arXiv 数据库中至少有一篇合著论文的每位研究人员设置一个节点。对于每篇有两位或更多作者的论文，我们为每对作者插入一条边（单作者论文被忽略）。请注意，当两位研究人员合著多篇论文时，这会产生平行边——我们保留了这些平行边，因为它们可以被解释为表明相关研究人员之间的社会联系更强。得到的图有 10748 个节点，约 53000 对节点之间存在边。

---

<!-- Footnote -->

${}^{2}$ We also ran experiments on the co-authorship graphs induced by theoretical computer science papers. We do not report on the results here, as they are very similar to the ones for high-energy physics.

${}^{2}$ 我们还对理论计算机科学论文所诱导的合著图进行了实验。我们在此不报告这些结果，因为它们与高能物理学的结果非常相似。

<!-- Footnote -->

---

While processing the data, we corrected many common types of mistakes automatically or manually. In order to deal with aliasing problems at least partially, we abbreviated first names, and unified spellings for foreign characters. We believe that the resulting graph is a good approximation to the actual collaboration graph (the sheer volume of data prohibits a complete manual cleaning pass).

在处理数据时，我们自动或手动纠正了许多常见类型的错误。为了至少部分解决别名问题，我们缩写了名字，并统一了外文字符的拼写。我们相信，得到的图是对实际合作图的一个很好的近似（数据量太大，无法进行完整的手动清理）。

The Influence Models. We compared the algorithms in three different models of influence. In the linear threshold model, we treated the multiplicity of edges as weights. If nodes $u,v$ have ${c}_{u,v}$ parallel edges between them,and degrees ${d}_{u}$ and ${d}_{v}$ ,then the edge(u,v) has weight $\frac{{c}_{u,v}}{{d}_{v}}$ ,and the edge(v,u)has weight $\frac{{c}_{u,v}}{{d}_{u}}$ .

影响模型。我们在三种不同的影响模型中比较了这些算法。在线性阈值模型中，我们将边的重数视为权重。如果节点 $u,v$ 之间有 ${c}_{u,v}$ 条平行边，且度数分别为 ${d}_{u}$ 和 ${d}_{v}$，那么边 (u, v) 的权重为 $\frac{{c}_{u,v}}{{d}_{v}}$，边 (v, u) 的权重为 $\frac{{c}_{u,v}}{{d}_{u}}$。

In the independent cascade model, we assigned a uniform probability of $p$ to each edge of the graph,choosing $p$ to be $1\%$ and ${10}\%$ in separate trials. If nodes $u$ and $v$ have ${c}_{u,v}$ parallel edges,then we assume that for each of those ${c}_{u,v}$ edges, $u$ has a chance of $p$ to activate $v$ ,i.e. $u$ has a total probability of $1 - {\left( 1 - p\right) }^{{c}_{u,v}}$ of activating $v$ once it becomes active.

在独立级联模型中，我们为图的每条边分配了一个均匀概率 $p$，在不同的试验中分别选择 $p$ 为 $1\%$ 和 ${10}\%$。如果节点 $u$ 和 $v$ 之间有 ${c}_{u,v}$ 条平行边，那么我们假设对于这 ${c}_{u,v}$ 条边中的每一条，$u$ 有 $p$ 的概率激活 $v$，即 $u$ 一旦激活，就有 $1 - {\left( 1 - p\right) }^{{c}_{u,v}}$ 的总概率激活 $v$。

The independent cascade model with uniform probabilities $p$ on the edges has the property that high-degree nodes not only have a chance to influence many other nodes, but also to be influenced by them. Whether or not this is a desirable interpretation of the influence data is an application-specific issue. Motivated by this, we chose to also consider an alternative interpretation, where edges into high-degree nodes are assigned smaller probabilities. We study a special case of the Independent Cascade Model that we term "weighted cascade",in which each edge from node $u$ to $v$ is assigned probability $1/{d}_{v}$ of activating $v$ . The weighted cascade model resembles the linear threshold model in that the expected number of neighbors who would succeed in activating a node $v$ is 1 in both models.

边具有均匀概率 $p$ 的独立级联模型具有这样的性质：高度数节点不仅有机会影响许多其他节点，而且也有机会受到它们的影响。这是否是对影响数据的理想解释是一个特定应用的问题。受此启发，我们选择考虑另一种解释，即向高度数节点的边被分配较小的概率。我们研究了独立级联模型的一个特殊情况，我们称之为“加权级联”，其中从节点 $u$ 到 $v$ 的每条边被分配激活 $v$ 的概率为 $1/{d}_{v}$。加权级联模型与线性阈值模型相似，因为在这两个模型中，成功激活节点 $v$ 的邻居的期望数量都是 1。

The algorithms and implementation. We compare our greedy algorithm with heuristics based on nodes' degrees and centrality within the network, as well as the crude baseline of choosing random nodes to target. The degree and centrality-based heuristics are commonly used in the sociology literature as estimates of a node's influence [30].

算法与实现。我们将贪婪算法与基于节点在网络中的度（degree）和中心性（centrality）的启发式算法进行比较，同时也与随机选择目标节点的粗略基线方法进行比较。基于度和中心性的启发式算法在社会学文献中常被用作对节点影响力的估计 [30]。

The high-degree heuristic chooses nodes $v$ in order of decreasing degrees ${d}_{v}$ . Considering high-degree nodes as influential has long been a standard approach for social and other networks $\left\lbrack  {{30},1}\right\rbrack$ ,and is known in the sociology literature as "degree centrality".

高度启发式算法按节点度 ${d}_{v}$ 降序选择节点 $v$。长期以来，将高度节点视为有影响力的节点是社交网络和其他网络的标准方法 $\left\lbrack  {{30},1}\right\rbrack$，在社会学文献中这被称为“度中心性（degree centrality）”。

"Distance centrality" is another commonly used influence measure in sociology, building on the assumption that a node with short paths to other nodes in a network will have a higher chance of influencing them. Hence, we select nodes in order of increasing average distance to other nodes in the network. As the arXiv collaboration graph is not connected,we assigned a distance of $n -$ the number of nodes in the graph - for any pair of unconnected nodes. This value is significantly larger than any actual distance, and thus can be considered to play the role of an infinite distance. In particular, nodes in the largest connected component will have smallest average distance.

“距离中心性（distance centrality）”是社会学中另一种常用的影响力度量方法，其基于这样的假设：在网络中与其他节点路径较短的节点更有可能影响它们。因此，我们按节点到网络中其他节点的平均距离升序选择节点。由于 arXiv 合作图不是连通的，我们为任意一对不相连的节点分配距离 $n -$（图中节点的数量）。这个值明显大于任何实际距离，因此可以视为无穷大距离。特别是，最大连通分量中的节点平均距离最小。

Finally, we consider, as a baseline, the result of choosing nodes uniformly at random. Notice that because the optimization problem is NP-hard, and the collaboration graph is prohibitively large, we cannot compute the optimum value to verify the actual quality of approximations.

最后，作为基线，我们考虑随机均匀选择节点的结果。请注意，由于优化问题是 NP 难问题，且合作图非常大，我们无法计算最优值来验证近似解的实际质量。

Both in choosing the nodes to target with the greedy algorithm, and in evaluating the performance of the algorithms, we need to compute the value $\sigma \left( A\right)$ . It is an open question to compute this quantity exactly by an efficient method, but very good estimates can be obtained by simulating the random process. More specifically, we simulate the process 10000 times for each targeted set, re-choosing thresholds or edge outcomes pseudo-randomly from $\left\lbrack  {0,1}\right\rbrack$ every time. Previous runs indicate that the quality of approximation after 10000 iterations is comparable to that after 300000 or more iterations.

在使用贪婪算法选择目标节点以及评估算法性能时，我们都需要计算值 $\sigma \left( A\right)$。通过高效方法精确计算这个量仍是一个开放问题，但通过模拟随机过程可以得到很好的估计。更具体地说，对于每个目标集，我们模拟该过程 10000 次，每次从 $\left\lbrack  {0,1}\right\rbrack$ 中伪随机地重新选择阈值或边的结果。之前的运行结果表明，10000 次迭代后的近似质量与 300000 次或更多次迭代后的近似质量相当。

The results. Figure 1 shows the performance of the algorithms in the linear threshold model. The greedy algorithm outperforms the high-degree node heuristic by about ${18}\%$ ,and the central node heuristic by over ${40}\%$ . (As expected,choosing random nodes is not a good idea.) This shows that significantly better marketing results can be obtained by explicitly considering the dynamics of information in a network, rather than relying solely on structural properties of the graph.

结果。图 1 展示了线性阈值模型中各算法的性能。贪婪算法的表现比高度节点启发式算法好约 ${18}\%$，比中心节点启发式算法好超过 ${40}\%$。（正如预期的那样，随机选择节点不是一个好主意。）这表明，通过明确考虑网络中信息的动态变化，而不是仅仅依赖图的结构属性，可以获得显著更好的营销效果。

<!-- Media -->

<!-- figureText: 1200 greedy centra random 20 25 target set size 1000 800 600 200 5 -->

<img src="https://cdn.noedgeai.com/0195c910-bd4b-76f9-b113-bcf4539c1837_5.jpg?x=926&y=632&w=702&h=508&r=0"/>

Figure 1: Results for the linear threshold model

图 1：线性阈值模型的结果

<!-- Media -->

When investigating the reason why the high-degree and centrality heuristics do not perform as well, one sees that they ignore such network effects. In particular, neither of the heuristics incorporates the fact that many of the most central (or highest-degree) nodes may be clustered, so that targeting all of them is unnecessary. In fact, the uneven nature of these curves suggests that the network influence of many nodes is not accurately reflected by their degree or centrality.

在研究高度和中心性启发式算法表现不佳的原因时，我们发现它们忽略了这种网络效应。特别是，这两种启发式算法都没有考虑到许多最中心（或度最高）的节点可能聚集在一起，因此针对所有这些节点是不必要的。事实上，这些曲线的不均匀性表明，许多节点的网络影响力并不能通过它们的度或中心性准确反映出来。

Figure 2 shows the results for the weighted cascade model. Notice the striking similarity to the linear threshold model. The scale is slightly different (all values are about ${25}\%$ smaller),but the behavior is qualitatively the same, even with respect to the exact nodes whose network influence is not reflected accurately by their degree or centrality. The reason is that in expectation, each node is influenced by the same number of other nodes in both models (see Section 2), and the degrees are relatively concentrated around their expectation of 1 .

图 2 展示了加权级联模型的结果。注意到它与线性阈值模型有惊人的相似性。尺度略有不同（所有值大约小 ${25}\%$），但行为在性质上是相同的，甚至对于那些度或中心性不能准确反映其网络影响力的具体节点也是如此。原因是，在期望意义上，在这两个模型中每个节点受其他节点影响的数量相同（见第 2 节），并且度相对集中在期望值 1 附近。

The graph for the independent cascade model with probability $1\%$ ,given in Figure 3,seems very similar to the previous two at first glance. Notice, however, the very different scale: on average, each targeted node only activates three additional nodes. Hence, the network effects in the independent cascade model with very small probabilities are much weaker than in the other models. Several nodes have degrees well exceeding 100 , so the probabilities on their incoming edges are even smaller than $1\%$ in the weighted cascade model. This suggests that the network effects observed for the linear threshold and weighted cascade models rely heavily on low-degree nodes as multipliers, even though targeting high-degree nodes is a reasonable heuristic. Also notice that in the independent cascade model, the heuristic of choosing random nodes performs significantly better than in the previous two models.

图3所示的概率为$1\%$的独立级联模型图，乍一看与前两个模型非常相似。然而，请注意其尺度差异很大：平均而言，每个目标节点仅激活三个额外节点。因此，概率非常小的独立级联模型中的网络效应比其他模型弱得多。有几个节点的度远远超过100，因此在加权级联模型中，它们入边的概率甚至小于$1\%$。这表明，线性阈值模型和加权级联模型中观察到的网络效应在很大程度上依赖于低度节点作为倍增器，尽管以高度节点为目标是一种合理的启发式方法。还要注意，在独立级联模型中，随机选择节点的启发式方法的表现明显优于前两个模型。

<!-- Media -->

<!-- figureText: 700 greedy central random target set size 600 active set size 500 400 300 200 100 -->

<img src="https://cdn.noedgeai.com/0195c910-bd4b-76f9-b113-bcf4539c1837_6.jpg?x=155&y=162&w=701&h=506&r=0"/>

Figure 2: Results for the weighted cascade model

图2：加权级联模型的结果

<!-- figureText: 90 greedy central 15 25 30 target set size 60 active set size 50 40 30 20 10 -->

<img src="https://cdn.noedgeai.com/0195c910-bd4b-76f9-b113-bcf4539c1837_6.jpg?x=155&y=941&w=698&h=508&r=0"/>

Figure 3: Independent cascade model with probability $1\%$

图3：概率为$1\%$的独立级联模型

<!-- Media -->

The improvement in performance of the "random nodes" heuristic is even more pronounced for the independent cascade model with probabilities equal to ${10}\%$ ,depicted in Figure 4. In that model, it starts to outperform both the high-degree and the central nodes heuristics when more than 12 nodes are targeted. It is initially surprising that random targeting for this model should lead to more activations than centrality-based targeting, but in fact there is a natural underlying reason that we explore now.

如图4所示，对于概率等于${10}\%$的独立级联模型，“随机节点”启发式方法的性能提升更为明显。在该模型中，当目标节点超过12个时，它开始优于高度节点和中心节点启发式方法。起初，人们会惊讶于该模型的随机目标选择会比基于中心性的目标选择导致更多的激活，但实际上，我们现在来探讨一下其背后的自然原因。

The first targeted node, if chosen somewhat judiciously, will activate a large fraction of the network, in our case almost 25%. However, any additional nodes will only reach a small additional fraction of the network. In particular, other central or high-degree nodes are very likely to be activated by the initially chosen one, and thus have hardly any marginal gain. This explains the shapes of the curves for the high-degree and centrality heuristics, which leap up to about 2415 activated nodes, but make virtually no progress afterwards. The greedy algorithm, on the other hand, takes the effect of the first chosen node into account, and targets nodes with smaller marginal gain afterwards. Hence, its active set keeps growing, although at a much smaller slope than in other models.

如果第一个目标节点选择得比较明智，它将激活网络的很大一部分，在我们的例子中几乎达到25%。然而，任何额外的节点只能覆盖网络中很小的额外部分。特别是，其他中心节点或高度节点很可能会被最初选择的节点激活，因此几乎没有边际收益。这就解释了高度节点和中心性启发式方法曲线的形状，它们会跃升至约2415个激活节点，但之后几乎没有进展。另一方面，贪心算法考虑了第一个选择节点的影响，随后以边际收益较小的节点为目标。因此，其活跃集持续增长，尽管斜率比其他模型小得多。

<!-- Media -->

<!-- figureText: 3000 degree random 20 30 target set size 2500 2000 active set size 1000 500 0 10 -->

<img src="https://cdn.noedgeai.com/0195c910-bd4b-76f9-b113-bcf4539c1837_6.jpg?x=924&y=168&w=698&h=504&r=0"/>

Figure 4: Independent cascade model with probability ${10}\%$

图4：概率为${10}\%$的独立级联模型

<!-- Media -->

The random heuristic does not do as well initially as the other heuristics, but with sufficiently many attempts, it eventually hits some highly influential nodes and becomes competitive with the centrality-based node choices. Because it does not focus exclusively on central nodes, it eventually targets nodes with additional marginal gain, and surpasses the two centrality-based heuristics.

随机启发式方法起初的表现不如其他启发式方法，但经过足够多次的尝试，它最终会命中一些极具影响力的节点，并与基于中心性的节点选择方法竞争。由于它并不只专注于中心节点，最终会以具有额外边际收益的节点为目标，并超越两种基于中心性的启发式方法。

## 4. AGENERAL FRAMEWORK FOR INFLU- ENCE MAXIMIZATION

## 4. 影响力最大化的通用框架

General Threshold and Cascade Models. We have thus far been considering two specific, widely studied models for the diffusion of influence. We now propose a broader framework that simultaneously generalizes these two models, and allows us to explore the limits of models in which strong approximation guarantees can be obtained. Our general framework has equivalent formulations in terms of thresholds and cascades, thereby unifying these two views of diffusion through a social network.

通用阈值和级联模型。到目前为止，我们一直在考虑两种广泛研究的特定影响力扩散模型。现在，我们提出一个更广泛的框架，该框架同时推广了这两种模型，并使我们能够探索可获得强近似保证的模型的极限。我们的通用框架在阈值和级联方面有等价的表述，从而通过社交网络统一了这两种扩散观点。

- A general threshold model. We would like to be able to express the notion that a node $v$ ’s decision to become active can be based on an arbitrary monotone function of the set of neighbors of $v$ that are already active. Thus,associated with $v$ is a monotone threshold function ${f}_{v}$ that maps subsets of $v$ ’s neighbor set to real numbers in $\left\lbrack  {0,1}\right\rbrack$ ,subject to the condition that ${f}_{v}\left( \varnothing \right)  = 0$ . The diffusion process follows the general structure of the Linear Threshold Model. Each node $v$ initially chooses ${\theta }_{v}$ uniformly at random from the interval $\left\lbrack  {0,1}\right\rbrack$ . Now,however, $v$ becomes active in step $t$ if ${f}_{v}\left( S\right)  \geq  {\theta }_{v}$ ,where $S$ is the set of neighbors of $v$ that are active in step $t - 1$ . Note that the Linear Threshold Model is the special case in which each threshold function has the form ${f}_{v}\left( S\right)  = \mathop{\sum }\limits_{{u \in  S}}{b}_{v,u}$ for parameters ${b}_{v,u}$ such that $\mathop{\sum }\limits_{{u\text{ neighbor of }v}}{b}_{v,u} \leq  1$ .

- 通用阈值模型。我们希望能够表达这样一种概念，即节点$v$决定激活可以基于$v$的已激活邻居集的任意单调函数。因此，与$v$相关联的是一个单调阈值函数${f}_{v}$，它将$v$的邻居集的子集映射到$\left\lbrack  {0,1}\right\rbrack$中的实数，条件是${f}_{v}\left( \varnothing \right)  = 0$。扩散过程遵循线性阈值模型的一般结构。每个节点$v$最初从区间$\left\lbrack  {0,1}\right\rbrack$中均匀随机选择${\theta }_{v}$。然而，现在如果${f}_{v}\left( S\right)  \geq  {\theta }_{v}$，则节点$v$在步骤$t$激活，其中$S$是节点$v$在步骤$t - 1$激活的邻居集。请注意，线性阈值模型是一种特殊情况，其中每个阈值函数具有${f}_{v}\left( S\right)  = \mathop{\sum }\limits_{{u \in  S}}{b}_{v,u}$的形式，参数${b}_{v,u}$满足$\mathop{\sum }\limits_{{u\text{ neighbor of }v}}{b}_{v,u} \leq  1$。

- A general cascade model. We generalize the cascade model to allow the probability that $u$ succeeds in activating a neighbor $v$ to depend on the set of $v$ ’s neighbors that have already tried. Thus,we define an incremental function ${p}_{v}\left( {u,S}\right)  \in$ $\left\lbrack  {0,1}\right\rbrack$ ,where $S$ and $\{ u\}$ are disjoint subsets of $v$ ’s neighbor set. A general cascade process works by analogy with the independent case: in the general case,when $u$ attempts to activate $v$ ,it succeeds with probability ${p}_{v}\left( {u,S}\right)$ ,where $S$ is the set of neighbors that have already tried (and failed) to activate $v$ . The Independent Cascade Model is the special case where ${p}_{v}\left( {u,S}\right)$ is a constant ${p}_{u,v}$ ,independent of $S$ . We will only be interested in cascade models defined by incremental functions that are order-independent in the following sense: if neighbors ${u}_{1},{u}_{2},\ldots ,{u}_{\ell }$ try to activate $v$ ,then the probability that $v$ is activated at the end of these $\ell$ attempts does not depend on the order in which the attempts are made.

- 一个通用级联模型。我们对级联模型进行推广，使得节点 $u$ 成功激活邻居节点 $v$ 的概率取决于已经尝试激活 $v$ 的邻居节点集合。因此，我们定义一个增量函数 ${p}_{v}\left( {u,S}\right)  \in$ $\left\lbrack  {0,1}\right\rbrack$ ，其中 $S$ 和 $\{ u\}$ 是 $v$ 邻居集合的不相交子集。通用级联过程的工作方式与独立级联情况类似：在通用情况下，当 $u$ 尝试激活 $v$ 时，它以概率 ${p}_{v}\left( {u,S}\right)$ 成功，其中 $S$ 是已经尝试（但失败）激活 $v$ 的邻居节点集合。独立级联模型是 ${p}_{v}\left( {u,S}\right)$ 为常数 ${p}_{u,v}$ （与 $S$ 无关）的特殊情况。我们只关注由满足以下意义上的顺序无关的增量函数定义的级联模型：如果邻居节点 ${u}_{1},{u}_{2},\ldots ,{u}_{\ell }$ 尝试激活 $v$ ，那么在这 $\ell$ 次尝试结束时 $v$ 被激活的概率不取决于尝试的顺序。

These two models are equivalent, and we give a method to convert between them. First, consider an instance of the general threshold model with threshold functions ${f}_{v}$ . To define an equivalent cascade model, we need to understand the probability that an additional neighbor $u$ can activate $v$ ,given that the nodes in a set $S$ have already tried and failed. If the nodes in $S$ have failed,then node $v$ ’s threshold ${\theta }_{v}$ must be in the range ${\theta }_{v} \in  \left( {{f}_{v}\left( S\right) ,1}\right\rbrack$ . However, subject to this constraint, it is uniformly distributed.Thus, the probability that a neighbor $u \notin  S$ succeeds in activating $v$ ,given that the nodes in $S$ have failed,is

这两个模型是等价的，我们给出一种在它们之间进行转换的方法。首先，考虑一个具有阈值函数 ${f}_{v}$ 的通用阈值模型实例。为了定义一个等价的级联模型，我们需要理解在集合 $S$ 中的节点已经尝试并失败的情况下，额外的邻居节点 $u$ 能够激活 $v$ 的概率。如果集合 $S$ 中的节点失败了，那么节点 $v$ 的阈值 ${\theta }_{v}$ 必须在范围 ${\theta }_{v} \in  \left( {{f}_{v}\left( S\right) ,1}\right\rbrack$ 内。然而，在这个约束条件下，它是均匀分布的。因此，在集合 $S$ 中的节点失败的情况下，邻居节点 $u \notin  S$ 成功激活 $v$ 的概率是

$$
{p}_{v}\left( {u,S}\right)  = \frac{{f}_{v}\left( {S\cup \{ u\} }\right)  - {f}_{v}\left( S\right) }{1 - {f}_{v}\left( S\right) }.
$$

It is not difficult to show that the cascade process with these functions is equivalent to the original threshold process.

不难证明，具有这些函数的级联过程与原始阈值过程是等价的。

Conversely,consider a node $v$ in the cascade model,and a set $S = \left\{  {{u}_{1},\ldots ,{u}_{k}}\right\}$ of its neighbors. Assume that the nodes in $S$ try to activate $v$ in the order ${u}_{1},\ldots ,{u}_{k}$ ,and let ${S}_{i} = \left\{  {{u}_{1},\ldots ,{u}_{i}}\right\}$ . Then the probability that $v$ is not activated by this process is by definition $\mathop{\prod }\limits_{{i = 1}}^{k}\left( {1 - {p}_{v}\left( {{u}_{i},{S}_{i - 1}}\right) }\right)$ . Recall that we assumed that the order in which the ${u}_{i}$ try to activate $v$ does not affect their overall success probability. Hence,this value depends on the set $S$ only, and we can define ${f}_{v}\left( S\right)  = 1 - \mathop{\prod }\limits_{{i = 1}}^{k}\left( {1 - {p}_{v}\left( {{u}_{i},{S}_{i - 1}}\right) }\right)$ . Analogously, one can show that this instance of the threshold model is equivalent to the original cascade process.

反之，考虑级联模型中的一个节点 $v$ 及其邻居节点集合 $S = \left\{  {{u}_{1},\ldots ,{u}_{k}}\right\}$ 。假设集合 $S$ 中的节点按照顺序 ${u}_{1},\ldots ,{u}_{k}$ 尝试激活 $v$ ，并设 ${S}_{i} = \left\{  {{u}_{1},\ldots ,{u}_{i}}\right\}$ 。那么根据定义， $v$ 在此过程中未被激活的概率是 $\mathop{\prod }\limits_{{i = 1}}^{k}\left( {1 - {p}_{v}\left( {{u}_{i},{S}_{i - 1}}\right) }\right)$ 。回想一下，我们假设 ${u}_{i}$ 尝试激活 $v$ 的顺序不影响它们的总体成功概率。因此，这个值仅取决于集合 $S$ ，我们可以定义 ${f}_{v}\left( S\right)  = 1 - \mathop{\prod }\limits_{{i = 1}}^{k}\left( {1 - {p}_{v}\left( {{u}_{i},{S}_{i - 1}}\right) }\right)$ 。类似地，可以证明这个阈值模型实例与原始级联过程是等价的。

An Inapproximability Result. The general model proposed above includes large families of instances for which the influence function $\sigma \left( \cdot \right)$ is not submodular. Indeed,it may become NP-hard to approximate the optimization problem to within any non-trivial factor.

一个不可近似性结果。上述提出的通用模型包含大量实例族，对于这些实例，影响函数 $\sigma \left( \cdot \right)$ 不是次模的。实际上，将优化问题近似到任何非平凡因子内可能会成为 NP 难问题。

THEOREM 4.1. In general, it is NP-hard to approximate the influence maximization problem to within a factor of ${n}^{1 - \varepsilon }$ ,for any $\varepsilon  > 0$ .

定理 4.1。一般来说，对于任何 $\varepsilon  > 0$ ，将影响最大化问题近似到 ${n}^{1 - \varepsilon }$ 因子内是 NP 难的。

Proof. To prove this result, we reduce from the Set Cover problem. We start with the construction from the proof of Theorem 2.4,letting ${u}_{1},\ldots ,{u}_{n}$ denote the nodes corresponding to the $n$ elements; i.e. ${u}_{i}$ becomes active when at least one of the nodes corresponding to sets containing ${u}_{i}$ is active. Next,for an arbitrarily large constant $c$ ,we add $N = {n}^{c}$ more nodes ${x}_{1},\ldots ,{x}_{N}$ ; each ${x}_{j}$ is connected to all of the nodes ${u}_{i}$ ,and it becomes active only when all of the ${u}_{i}$ are.

证明。为了证明这一结果，我们从集合覆盖问题（Set Cover problem）进行归约。我们从定理2.4的证明构造开始，让${u}_{1},\ldots ,{u}_{n}$表示对应于$n$个元素的节点；即当对应于包含${u}_{i}$的集合的节点中至少有一个处于活跃状态时，${u}_{i}$变为活跃状态。接下来，对于任意大的常数$c$，我们再添加$N = {n}^{c}$个节点${x}_{1},\ldots ,{x}_{N}$；每个${x}_{j}$都与所有节点${u}_{i}$相连，并且只有当所有${u}_{i}$都处于活跃状态时，它才会变为活跃状态。

If there are at most $k$ sets that cover all elements,then activating the nodes corresponding to these $k$ sets will activate all of the nodes ${u}_{i}$ ,and thus also all of the ${x}_{j}$ . In total,at least $N + n + k$ nodes will be active. Conversely,if there is no set cover of size $k$ ,then no targeted set will activate all of the ${u}_{i}$ ,and hence none of the ${x}_{j}$ will become active (unless targeted). In particular,fewer than $n + k$ nodes are active in the end. If an algorithm could approximate the problem within ${n}^{1 - \varepsilon }$ for any $\varepsilon$ ,it could distinguish between the cases where $N + n + k$ nodes are active in the end,and where fewer than $n + k$ are. But this would solve the underlying instance of Set Cover,and therefore is impossible assuming $\mathrm{P} \neq  \mathrm{{NP}}$ .

如果存在至多$k$个集合可以覆盖所有元素，那么激活对应于这$k$个集合的节点将激活所有节点${u}_{i}$，从而也激活所有${x}_{j}$。总共至少有$N + n + k$个节点将处于活跃状态。相反，如果不存在大小为$k$的集合覆盖，那么没有目标集合能够激活所有${u}_{i}$，因此没有${x}_{j}$会变为活跃状态（除非被目标激活）。特别地，最终活跃的节点少于$n + k$个。如果一个算法能够对于任何$\varepsilon$在${n}^{1 - \varepsilon }$范围内近似该问题，那么它就能区分最终有$N + n + k$个节点活跃和最终活跃节点少于$n + k$个这两种情况。但这将解决底层的集合覆盖问题实例，因此在假设$\mathrm{P} \neq  \mathrm{{NP}}$的情况下是不可能的。

Note that our inapproximability result holds in a very simple model, in which each node is "hard-wired" with a fixed threshold.

请注意，我们的不可近似性结果在一个非常简单的模型中成立，在该模型中，每个节点都“硬连接”有一个固定的阈值。

Exploring the Boundaries of Approximability. Thus, the general threshold and cascade models are too broad to allow for nontrivial approximation guarantees in their full generality. At the same time, we have seen that the greedy algorithm achieves strong guarantees for some of the main special cases in the social networks literature. How far can we extend these approximability results?

探索可近似性的边界。因此，一般的阈值模型和级联模型过于宽泛，无法在其完全一般性上提供非平凡的近似保证。同时，我们已经看到，贪心算法在社交网络文献中的一些主要特殊情况下能够实现很强的保证。我们能将这些可近似性结果扩展到多远呢？

We can generalize the proof technique used in Theorems 2.2 and 2.5 to a model that is less general (and also less natural) than the general threshold and cascade models; however, it includes our special cases from Section 2, and every instance of this model will have a submodular influence function. The model is as follows.

我们可以将定理2.2和2.5中使用的证明技术推广到一个比一般阈值模型和级联模型更不通用（也更不自然）的模型；然而，它包含了我们在第2节中的特殊情况，并且该模型的每个实例都将有一个次模影响函数。该模型如下。

- The Triggering Model. Each node $v$ independently chooses a random "triggering set" ${T}_{v}$ according to some distribution over subsets of its neighbors. To start the process, we target a set $A$ for initial activation. After this initial iteration,an inactive node $v$ becomes active in step $t$ if it has a neighbor in its chosen triggering set ${T}_{v}$ that is active at time $t - 1$ . (Thus, $v$ ’s threshold has been replaced by a latent subset of ${T}_{v}$ of neighbors whose behavior actually affects $v$ .)

- 触发模型。每个节点$v$根据其邻居子集上的某个分布独立地选择一个随机“触发集”${T}_{v}$。为了启动这个过程，我们选择一个集合$A$进行初始激活。在这个初始迭代之后，如果一个非活跃节点$v$在其选择的触发集${T}_{v}$中有一个在时间$t - 1$处于活跃状态的邻居，那么它将在步骤$t$变为活跃状态。（因此，$v$的阈值已被${T}_{v}$中实际影响$v$行为的邻居的潜在子集所取代。）

It is useful to think of the triggering sets in terms of "live" and "blocked" edges: if node $u$ belongs to the triggering set ${T}_{v}$ of $v$ , then we declare the edge(u,v)to be live,and otherwise we declare it to be blocked. As in the proofs of Theorems 2.2 and 2.5, a node $v$ is activated in an instance of the Triggering Model if and only if there is a live-edge path from the initially targeted set $A$ to $v$ . Following the arguments in these proofs, we obtain the following

从“活跃”和“阻塞”边的角度来考虑触发集是很有用的：如果节点$u$属于$v$的触发集${T}_{v}$，那么我们称边(u,v)为活跃边，否则称其为阻塞边。与定理2.2和2.5的证明一样，在触发模型的一个实例中，当且仅当从初始目标集合$A$到$v$存在一条活跃边路径时，节点$v$才会被激活。遵循这些证明中的论证，我们得到以下结论

THEOREM 4.2. In every instance of the Triggering Model, the influence function $\sigma \left( \cdot \right)$ is submodular.

定理4.2。在触发模型的每个实例中，影响函数$\sigma \left( \cdot \right)$是次模的。

Beyond the Independent Cascade and Linear Threshold, there are other natural special cases of the Triggering Model. One example is the "Only-Listen-Once" Model. Here,each node $v$ has a parameter ${p}_{v}$ so that the first neighbor of $v$ to be activated causes $v$ to become active with probability ${p}_{v}$ ,and all subsequent attempts to activate $v$ deterministically fail. (In other words, $v$ only listens to the first neighbor that tries to activate it.) This process has an equivalent formulation in the Triggering Set Model, with an edge distribution defined as follows: for any node $v$ ,the triggering set ${T}_{v}$ is either the entire neighbor set of $v$ (with probability ${p}_{v}$ ),or the empty set otherwise. As a result, the influence function in the Only-Listen-Once Model is also submodular, and we can obtain a $\left( {1 - 1/e - \varepsilon }\right)$ -approximation here as well.

除了独立级联模型（Independent Cascade）和线性阈值模型（Linear Threshold）之外，触发模型（Triggering Model）还有其他自然的特殊情况。一个例子是“只听一次”模型（"Only-Listen-Once" Model）。在这里，每个节点 $v$ 都有一个参数 ${p}_{v}$，使得 $v$ 的第一个被激活的邻居节点会以概率 ${p}_{v}$ 激活 $v$，并且所有后续尝试激活 $v$ 的操作都将确定性地失败。（换句话说，$v$ 只听取第一个尝试激活它的邻居节点的信息。）这个过程在触发集模型（Triggering Set Model）中有一个等价的表述，其边分布定义如下：对于任何节点 $v$，触发集 ${T}_{v}$ 要么是 $v$ 的整个邻居节点集合（概率为 ${p}_{v}$），要么是空集。因此，“只听一次”模型中的影响函数也是次模的（submodular），我们在这里也可以得到一个 $\left( {1 - 1/e - \varepsilon }\right)$ -近似解。

However, we can show that there exist models with submodu-lar influence functions that do not have equivalent formulations in terms of triggering sets, so it makes sense to seek further models in which submodularity holds.

然而，我们可以证明，存在具有次模影响函数的模型，这些模型在触发集方面没有等价的表述，因此寻找次模性成立的更多模型是有意义的。

One tractable special case of the cascade model is based on the natural restriction that the probability of a node $u$ influencing $v$ is non-increasing as a function of the set of nodes that have previously tried to influence $v$ . In terms of the cascade model,this means that ${p}_{v}\left( {u,S}\right)  \geq  {p}_{v}\left( {u,T}\right)$ whenever $S \subseteq  T$ . We say that a process satisfying these conditions is an instance of the Decreasing Cascade Model. Although there are natural Decreasing Cascade instances that have no equivalent formulation in terms of triggering sets, we can show by a more intricate analysis that every instance of the Decreasing Cascade Model has a submodular influence function. We will include details of this proof in the full version of the paper. A Conjecture. Finally, we state an appealing conjecture that would include all the approximability results above as special cases.

级联模型（cascade model）的一个易处理的特殊情况基于这样一个自然的限制：节点 $u$ 影响 $v$ 的概率是之前尝试影响 $v$ 的节点集合的非增函数。就级联模型而言，这意味着当 $S \subseteq  T$ 时，${p}_{v}\left( {u,S}\right)  \geq  {p}_{v}\left( {u,T}\right)$ 成立。我们称满足这些条件的过程是递减级联模型（Decreasing Cascade Model）的一个实例。尽管存在一些自然的递减级联实例在触发集方面没有等价的表述，但我们可以通过更复杂的分析证明，递减级联模型的每个实例都有一个次模影响函数。我们将在论文的完整版本中包含这个证明的详细内容。一个猜想。最后，我们陈述一个有吸引力的猜想，该猜想将上述所有可近似性结果作为特殊情况包含在内。

CONJECTURE 4.3. Whenever the threshold functions ${f}_{v}$ at every node are monotone and submodular, the resulting influence function $\sigma \left( \cdot \right)$ is monotone and submodular as well.

猜想 4.3。只要每个节点的阈值函数 ${f}_{v}$ 是单调且次模的，那么由此产生的影响函数 $\sigma \left( \cdot \right)$ 也是单调且次模的。

It is not difficult to show that every instance of the Triggering Model has an equivalent formulation with submodular node thresholds. Every instance of the Decreasing Cascade Model has such an equivalent formulation as well; in fact, the Decreasing Cascade condition stands as a very natural special case of the conjecture, given that it too is based on a type of "diminishing returns." When translated into the language of threshold functions, we find that the Decreasing Cascade condition corresponds to the following natural requirement:

不难证明，触发模型的每个实例都有一个具有次模节点阈值的等价表述。递减级联模型的每个实例也有这样的等价表述；事实上，递减级联条件是该猜想的一个非常自然的特殊情况，因为它同样基于一种“收益递减”类型。当用阈值函数的语言来表述时，我们发现递减级联条件对应于以下自然要求：

$$
\frac{{f}_{v}\left( {S\cup \{ u\} }\right)  - {f}_{v}\left( S\right) }{1 - {f}_{v}\left( S\right) } \geq  \frac{{f}_{v}\left( {T\cup \{ u\} }\right)  - {f}_{v}\left( T\right) }{1 - {f}_{v}\left( T\right) },
$$

whenever $S \subseteq  T$ and $u \notin  T$ . This is in a sense a "normalized submodularity" property; it is stronger than submodularity, which would consist of the same inequality on just the numerators. (Note that by monotonicity, the denominator on the left is larger.)

当 $S \subseteq  T$ 且 $u \notin  T$ 时。从某种意义上说，这是一种“归一化次模性”属性；它比次模性更强，次模性仅要求分子满足相同的不等式。（注意，根据单调性，左边的分母更大。）

## 5. NON-PROGRESSIVE PROCESSES

## 5. 非渐进过程

We have thus far been concerned with the progressive case, in which nodes only go from inactivity to activity, but not vice versa. The non-progressive case, in which nodes can switch in both directions, can in fact be reduced to the progressive case.

到目前为止，我们一直关注渐进情况，即节点只会从非活动状态转变为活动状态，而不会相反。在非渐进情况下，节点可以在两个方向上切换，实际上可以将其简化为渐进情况。

The non-progressive threshold process is analogous to the progressive model,except that at each step $t$ ,each node $v$ chooses a new value ${\theta }_{v}^{\left( t\right) }$ uniformly at random from the interval $\left\lbrack  {0,1}\right\rbrack$ . Node $v$ will be active in step $t$ if ${f}_{v}\left( S\right)  \geq  {\theta }_{v}^{\left( t\right) }$ ,where $S$ is the set of neighbors of $v$ that are active in step $t - 1$ .

非渐进阈值过程类似于渐进模型，不同之处在于，在每一步 $t$，每个节点 $v$ 从区间 $\left\lbrack  {0,1}\right\rbrack$ 中均匀随机地选择一个新值 ${\theta }_{v}^{\left( t\right) }$。如果 ${f}_{v}\left( S\right)  \geq  {\theta }_{v}^{\left( t\right) }$，节点 $v$ 将在步骤 $t$ 中处于活动状态，其中 $S$ 是在步骤 $t - 1$ 中处于活动状态的 $v$ 的邻居节点集合。

From the perspective of influence maximization, we can ask the following question. Suppose we have a non-progressive model that is going to run for $\tau$ steps,and during this process,we are allowed to make up to $k$ interventions: for a particular node $v$ ,at a particular time $t \leq  \tau$ ,we can target $v$ for activation at time $t$ . ( $v$ itself may quickly de-activate, but we hope to create a large "ripple effect.") Which $k$ interventions should we perform? Simple examples show that to maximize influence, one should not necessarily perform all $k$ interventions at time 0 ; e.g., $G$ may not even have $k$ nodes.

从影响力最大化的角度来看，我们可以提出以下问题。假设我们有一个非渐进式模型，该模型将运行 $\tau$ 步，并且在此过程中，我们最多可以进行 $k$ 次干预：对于特定节点 $v$，在特定时间 $t \leq  \tau$，我们可以在时间 $t$ 对 $v$ 进行激活操作。（$v$ 本身可能会很快失活，但我们希望产生较大的“涟漪效应”。）我们应该进行哪 $k$ 次干预呢？简单的例子表明，为了最大化影响力，不一定非要在时间 0 进行所有 $k$ 次干预；例如，$G$ 可能甚至没有 $k$ 个节点。

Let $A$ be a set of $k$ interventions. The influence of these $k$ interventions $\sigma \left( A\right)$ is the sum,over all nodes $v$ ,of the number of time steps that $v$ is active. The influence maximization problem in the non-progressive threshold model is to find the $k$ interventions with maximum influence.

设 $A$ 为一组 $k$ 次干预。这 $k$ 次干预 $\sigma \left( A\right)$ 的影响力是所有节点 $v$ 处于活跃状态的时间步数之和。非渐进式阈值模型中的影响力最大化问题是找到具有最大影响力的 $k$ 次干预。

We can show that the non-progressive influence maximization problem reduces to the progressive case in a different graph. Given a graph $G = \left( {V,E}\right)$ and a time limit $\tau$ ,we build a layered graph ${G}^{\tau }$ on $\tau  \cdot  \left| V\right|$ nodes: there is a copy ${v}_{t}$ for each node $v$ in $G$ ,and each time-step $t \leq  \tau$ . We connect each node in this graph with its neighbors in $G$ indexed by the previous time step.

我们可以证明，非渐进式影响力最大化问题可以转化为另一个图中的渐进式情况。给定一个图 $G = \left( {V,E}\right)$ 和一个时间限制 $\tau$，我们在 $\tau  \cdot  \left| V\right|$ 个节点上构建一个分层图 ${G}^{\tau }$：对于 $G$ 中的每个节点 $v$ 以及每个时间步 $t \leq  \tau$，都有一个副本 ${v}_{t}$。我们将这个图中的每个节点与其在 $G$ 中由前一个时间步索引的邻居相连。

THEOREM 5.1. The non-progressive influence maximization problem on $G$ over a time horizon $\tau$ is equivalent to the progressive influence maximization problem on the layered graph ${G}^{\tau }$ . Node $v$ is active at time $t$ in the non-progressive process if and only if ${v}_{t}$ is activated in the progressive process.

定理 5.1。在时间范围 $\tau$ 内，图 $G$ 上的非渐进式影响力最大化问题等价于分层图 ${G}^{\tau }$ 上的渐进式影响力最大化问题。当且仅当在渐进式过程中 ${v}_{t}$ 被激活时，在非渐进式过程中节点 $v$ 在时间 $t$ 处于活跃状态。

Thus, models where we have approximation algorithms for the progressive case carry over. Theorem 5.1 also implies approximation results for certain non-progressive models used by Asavathi-ratham et al. to model cascading failures in power grids [2,3].

因此，对于渐进式情况有近似算法的模型同样适用。定理 5.1 还暗示了 Asavathi - ratham 等人用于模拟电网级联故障的某些非渐进式模型的近似结果 [2,3]。

Note that the non-progressive model discussed here differs from the model of Domingos and Richardson $\left\lbrack  {{10},{26}}\right\rbrack$ in two ways. We are concerned with the sum over all time steps $t \leq  \tau$ of the expected number of active nodes at time $t$ ,for a given a time limit $\tau$ , while $\left\lbrack  {{10},{26}}\right\rbrack$ study the limit of this process: the expected number of nodes active at time $t$ as $t$ goes to infinity. Further,we consider interventions for a particular node $v$ ,at a particular time $t \leq  \tau$ , while the interventions considered by $\left\lbrack  {{10},{26}}\right\rbrack$ permanently affect the activation probability function of the targeted nodes.

请注意，这里讨论的非渐进式模型与 Domingos 和 Richardson $\left\lbrack  {{10},{26}}\right\rbrack$ 的模型在两个方面有所不同。对于给定的时间限制 $\tau$，我们关注的是所有时间步 $t \leq  \tau$ 上时间 $t$ 时活跃节点的期望数量之和，而 $\left\lbrack  {{10},{26}}\right\rbrack$ 研究的是这个过程的极限：当 $t$ 趋于无穷大时，时间 $t$ 时活跃节点的期望数量。此外，我们考虑的是在特定时间 $t \leq  \tau$ 对特定节点 $v$ 进行干预，而 $\left\lbrack  {{10},{26}}\right\rbrack$ 考虑的干预会永久影响目标节点的激活概率函数。

## 6. GENERAL MARKETING STRATEGIES

## 6. 通用营销策略

In the formulation of the problem, we have so far assumed that for one unit of budget,we can deterministically target any node $v$ for activation. This is clearly a highly simplified view. In a more realistic scenario,we may have a number $m$ of different marketing actions ${M}_{i}$ available,each of which may affect some subset of nodes by increasing their probabilities of becoming active, without necessarily making them active deterministically. The more we spend on any one action the stronger its effect will be; however, different nodes may respond to marketing actions in different ways,

在问题的表述中，到目前为止我们假设用一单位预算，我们可以确定性地针对任何节点 $v$ 进行激活。这显然是一个高度简化的观点。在更现实的场景中，我们可能有 $m$ 种不同的营销行动 ${M}_{i}$ 可供选择，每种行动可能会通过增加某些节点的活跃概率来影响这些节点的子集，但不一定能确定性地使它们活跃。我们在任何一种行动上投入得越多，其效果就越强；然而，不同的节点可能会以不同的方式对营销行动做出反应。

In a general model,we choose investments ${x}_{i}$ into marketing actions ${M}_{i}$ ,such that the total investments do not exceed the budget. A marketing strategy is then an $m$ -dimensional vector $\mathbf{x}$ of investments. The probability that node $v$ will become active is determined by the strategy,and denoted by ${h}_{v}\left( \mathbf{x}\right)$ . We assume that this function is non-decreasing and satisfies the following "diminishing returns" property for all $\mathbf{x} \geq  \mathbf{y}$ and $\mathbf{a} \geq  \mathbf{0}$ (where we write $\mathbf{x} \geq  \mathbf{y}$ or $\mathbf{a} \geq  \mathbf{0}$ to denote that the inequalities hold in all coordinates):

在一般模型中，我们选择对营销行动 ${M}_{i}$ 进行投资 ${x}_{i}$，使得总投资不超过预算。那么，营销策略就是一个 $m$ 维的投资向量 $\mathbf{x}$。节点 $v$ 变为活跃状态的概率由该策略决定，用 ${h}_{v}\left( \mathbf{x}\right)$ 表示。我们假设该函数是非递减的，并且对于所有的 $\mathbf{x} \geq  \mathbf{y}$ 和 $\mathbf{a} \geq  \mathbf{0}$ 满足以下“收益递减”性质（我们用 $\mathbf{x} \geq  \mathbf{y}$ 或 $\mathbf{a} \geq  \mathbf{0}$ 表示不等式在所有坐标上都成立）：

$$
{h}_{v}\left( {\mathbf{x} + \mathbf{a}}\right)  - {h}_{v}\left( \mathbf{x}\right)  \leq  {h}_{v}\left( {\mathbf{y} + \mathbf{a}}\right)  - {h}_{v}\left( \mathbf{y}\right)  \tag{1}
$$

Intuitively, Inequality (1) states that any marketing action is more effective when the targeted individual is less "marketing-saturated" at that point.

直观地说，不等式 (1) 表明，当目标个体在该时刻的“营销饱和度”较低时，任何营销行动都更有效。

We are trying to maximize the expected size of the final active set. As a function of the marketing strategy $\mathbf{x}$ ,each node $v$ becomes active independently with probability ${h}_{v}\left( \mathbf{x}\right)$ ,resulting in a (random) set of initial active nodes $A$ . Given the initial set $A$ ,the expected size of the final active set is $\sigma \left( A\right)$ . The expected revenue of the marketing strategy $\mathbf{x}$ is therefore

我们试图最大化最终活跃集的期望规模。作为营销策略 $\mathbf{x}$ 的函数，每个节点 $v$ 以概率 ${h}_{v}\left( \mathbf{x}\right)$ 独立地变为活跃状态，从而产生一个（随机的）初始活跃节点集 $A$。给定初始集 $A$，最终活跃集的期望规模为 $\sigma \left( A\right)$。因此，营销策略 $\mathbf{x}$ 的期望收益为

$$
g\left( \mathbf{x}\right)  = \mathop{\sum }\limits_{{A \subseteq  V}}\sigma \left( A\right)  \cdot  \mathop{\prod }\limits_{{u \in  A}}{h}_{u}\left( \mathbf{x}\right)  \cdot  \mathop{\prod }\limits_{{v \notin  A}}\left( {1 - {h}_{v}\left( \mathbf{x}\right) }\right) .
$$

In order to (approximately) maximize $g$ ,we assume that we can evaluate the function at any point $\mathbf{x}$ approximately,and find a direction $i$ with approximately maximal gradient. Specifically,let ${\mathbf{e}}_{i}$ denote the unit vector along the ${i}^{\text{th }}$ coordinate axis,and $\delta$ be some constant. We assume that there exists some $\gamma  \leq  1$ such that we can find an $i$ with $g\left( {\mathbf{x} + \delta  \cdot  {\mathbf{e}}_{i}}\right)  - g\left( \mathbf{x}\right)  \geq  \gamma  \cdot  \left( {g\left( {\mathbf{x} + \delta  \cdot  {\mathbf{e}}_{j}}\right)  - g\left( \mathbf{x}\right) }\right)$ for each $j$ . We divide each unit of the total budget $k$ into equal parts of size $\delta$ . Starting with an all-0 investment,we perform an approximate gradient ascent,by repeatedly (a total of $\frac{k}{\delta }$ times) adding $\delta$ units of budget to the investment in the action ${M}_{i}$ that approximately maximizes the gradient.

为了（近似地）最大化 $g$，我们假设可以在任意点 $\mathbf{x}$ 近似地评估该函数，并找到一个具有近似最大梯度的方向 $i$。具体来说，设 ${\mathbf{e}}_{i}$ 表示沿 ${i}^{\text{th }}$ 坐标轴的单位向量，$\delta$ 为某个常数。我们假设存在某个 $\gamma  \leq  1$，使得对于每个 $j$，我们都能找到一个满足 $g\left( {\mathbf{x} + \delta  \cdot  {\mathbf{e}}_{i}}\right)  - g\left( \mathbf{x}\right)  \geq  \gamma  \cdot  \left( {g\left( {\mathbf{x} + \delta  \cdot  {\mathbf{e}}_{j}}\right)  - g\left( \mathbf{x}\right) }\right)$ 的 $i$。我们将总预算 $k$ 的每个单位划分为大小为 $\delta$ 的相等部分。从全为 0 的投资开始，我们执行近似梯度上升法，通过反复（总共 $\frac{k}{\delta }$ 次）将 $\delta$ 个单位的预算添加到近似最大化梯度的行动 ${M}_{i}$ 的投资中。

The proof that this algorithm gives a good approximation consists of two steps. First,we show that the function $g$ we are trying to optimize is non-negative, non-decreasing, and satisfies the "diminishing returns" condition (1). Second, we show that the hill-climbing algorithm gives a constant-factor approximation for any function $g$ with these properties. The latter part is captured by the following theorem.

该算法能给出良好近似的证明包括两个步骤。首先，我们证明我们试图优化的函数 $g$ 是非负的、非递减的，并且满足“收益递减”条件 (1)。其次，我们证明爬山算法对于任何具有这些性质的函数 $g$ 都能给出一个常数因子近似。后一部分由以下定理描述。

THEOREM 6.1. When the hill-climbing algorithm finishes with strategy $\mathbf{x}$ ,it guarantees that $g\left( \mathbf{x}\right)  \geq  \left( {1 - {e}^{-\frac{k \cdot  \gamma }{k + \delta  \cdot  n}}}\right)  \cdot  g\left( \widehat{\mathbf{x}}\right)$ ,where $\widehat{\mathbf{x}}$ denotes the optimal solution subject to $\mathop{\sum }\limits_{i}{\widehat{x}}_{i} \leq  k$ .

定理 6.1。当爬山算法以策略 $\mathbf{x}$ 结束时，它保证 $g\left( \mathbf{x}\right)  \geq  \left( {1 - {e}^{-\frac{k \cdot  \gamma }{k + \delta  \cdot  n}}}\right)  \cdot  g\left( \widehat{\mathbf{x}}\right)$，其中 $\widehat{\mathbf{x}}$ 表示受 $\mathop{\sum }\limits_{i}{\widehat{x}}_{i} \leq  k$ 约束的最优解。

The proof of this theorem builds on the analysis used by Nemhauser et al. [23], and we defer it to the full version of this paper.

该定理的证明基于 Nemhauser 等人 [23] 使用的分析方法，我们将其推迟到本文的完整版本中给出。

With Theorem 6.1 in hand,it remains to show that $g$ is nonnegative, monotone, and satisfies condition (1). The first two are clear, so we only sketch the proof of the third. Fix an arbitary ordering of vertices. We then use the fact that for any ${a}_{i},{b}_{i}$ ,

有了定理6.1，接下来只需证明$g$是非负的、单调的，并且满足条件(1)。前两个性质很明显，因此我们仅简述第三个性质的证明。固定顶点的任意一个排序。然后，我们利用对于任意${a}_{i},{b}_{i}$的事实

$$
\mathop{\prod }\limits_{i}{a}_{i} - \mathop{\prod }\limits_{i}{b}_{i} = \mathop{\sum }\limits_{i}\left( {{a}_{i} - {b}_{i}}\right)  \cdot  \mathop{\prod }\limits_{{j < i}}{a}_{j} \cdot  \mathop{\prod }\limits_{{j > i}}{b}_{j} \tag{2}
$$

and change the order of summation, to rewrite the difference

并改变求和顺序，以重写差值

$$
g\left( {\mathbf{x} + \mathbf{a}}\right)  - g\left( \mathbf{x}\right) 
$$

$$
 = \mathop{\sum }\limits_{u}\left( {\left( {{h}_{u}\left( {\mathbf{x} + \mathbf{a}}\right)  - {h}_{u}\left( \mathbf{x}\right) }\right)  \cdot  \mathop{\sum }\limits_{{A : u \notin  A}}\left( {\sigma \left( {A + u}\right)  - \sigma \left( A\right) }\right)  \cdot  }\right. 
$$

$$
\mathop{\prod }\limits_{{j < u,j \in  A}}{h}_{j}\left( {\mathbf{x} + \mathbf{a}}\right)  \cdot  \mathop{\prod }\limits_{{j < u,j \notin  A}}\left( {1 - {h}_{j}\left( {\mathbf{x} + \mathbf{a}}\right) }\right) .
$$

$$
\left. {\mathop{\prod }\limits_{{j > u,j \in  A}}{h}_{j}\left( \mathbf{x}\right)  \cdot  \mathop{\prod }\limits_{{j > u,j \notin  A}}\left( {1 - {h}_{j}\left( \mathbf{x}\right) }\right) }\right) \text{.}
$$

To show that this difference is non-increasing,we consider $\mathbf{y} \leq$ $\mathbf{x}$ . From the diminishing returns property of ${h}_{u}\left( \cdot \right)$ ,we obtain that ${h}_{u}\left( {\mathbf{x} + \mathbf{a}}\right)  - {h}_{u}\left( \mathbf{x}\right)  \leq  {h}_{u}\left( {\mathbf{y} + \mathbf{a}}\right)  - {h}_{u}\left( \mathbf{y}\right)$ . Then,applying again equation (2), changing the order of summation, and performing some tedious calculations,writing $\Delta \left( {v,\mathbf{x},\mathbf{y}}\right)  = {h}_{v}\left( {\mathbf{x} + \mathbf{a}}\right)  -$ ${h}_{v}\left( {\mathbf{y} + \mathbf{a}}\right)$ if $v < u$ ,and $\Delta \left( {v,\mathbf{x},\mathbf{y}}\right)  = {h}_{v}\left( \mathbf{x}\right)  - {h}_{v}\left( \mathbf{y}\right)$ if $v > u$ , we obtain that

为了证明这个差值是不增的，我们考虑$\mathbf{y} \leq$ $\mathbf{x}$。根据${h}_{u}\left( \cdot \right)$的收益递减性质，我们得到${h}_{u}\left( {\mathbf{x} + \mathbf{a}}\right)  - {h}_{u}\left( \mathbf{x}\right)  \leq  {h}_{u}\left( {\mathbf{y} + \mathbf{a}}\right)  - {h}_{u}\left( \mathbf{y}\right)$。然后，再次应用方程(2)，改变求和顺序，并进行一些繁琐的计算，若$v < u$，记$\Delta \left( {v,\mathbf{x},\mathbf{y}}\right)  = {h}_{v}\left( {\mathbf{x} + \mathbf{a}}\right)  -$ ${h}_{v}\left( {\mathbf{y} + \mathbf{a}}\right)$，若$v > u$，记$\Delta \left( {v,\mathbf{x},\mathbf{y}}\right)  = {h}_{v}\left( \mathbf{x}\right)  - {h}_{v}\left( \mathbf{y}\right)$，我们得到

$$
\left( {g\left( {\mathbf{x} + \mathbf{a}}\right)  - g\left( \mathbf{x}\right) }\right)  - \left( {g\left( {\mathbf{y} + \mathbf{a}}\right)  - g\left( \mathbf{y}\right) }\right) 
$$

$$
 \leq  \mathop{\sum }\limits_{{u,v : u \neq  v}}\left( {\left( {{h}_{u}\left( {\mathbf{y} + \mathbf{a}}\right)  - {h}_{u}\left( \mathbf{y}\right) }\right)  \cdot  \Delta \left( {v,\mathbf{x},\mathbf{y}}\right) .}\right. 
$$

$$
\mathop{\sum }\limits_{{A : u,v \notin  A}}\left( {\sigma \left( {A+\{ u,v\} }\right)  - \sigma \left( {A + v}\right)  - \sigma \left( {A + u}\right)  + \sigma \left( A\right) }\right) .
$$

$$
\mathop{\prod }\limits_{{j < \min \left( {u,v}\right) ,j \in  A}}{h}_{j}\left( {\mathbf{x} + \mathbf{a}}\right)  \cdot  \mathop{\prod }\limits_{{j < \min \left( {u,v}\right) ,j \notin  A}}\left( {1 - {h}_{j}\left( {\mathbf{x} + \mathbf{a}}\right) }\right) .
$$

$$
\mathop{\prod }\limits_{{u < j < v,j \in  A}}{h}_{j}\left( \mathbf{x}\right)  \cdot  \mathop{\prod }\limits_{{u < j < v,j \notin  A}}\left( {1 - {h}_{j}\left( \mathbf{x}\right) }\right) .
$$

$$
\mathop{\prod }\limits_{{v < j < u,j \in  A}}{h}_{j}\left( {\mathbf{y} + \mathbf{a}}\right)  \cdot  \mathop{\prod }\limits_{{v < j < u,j \notin  A}}\left( {1 - {h}_{j}\left( {\mathbf{y} + \mathbf{a}}\right) }\right) .
$$

$$
\left. {\mathop{\prod }\limits_{{j > \max \left( {u,v}\right) ,j \in  A}}{h}_{j}\left( \mathbf{y}\right)  \cdot  \mathop{\prod }\limits_{{j > \max \left( {u,v}\right) ,j \notin  A}}\left( {1 - {h}_{j}\left( \mathbf{y}\right) }\right) }\right) 
$$

In this expression, all terms are non-negative (by monotonicity of the ${h}_{v}\left( \cdot \right)$ ),with the exception of $\sigma \left( {A+\{ u,v\} }\right)  - \sigma \left( {A + u}\right)  -$ $\sigma \left( {A + v}\right)  + \sigma \left( A\right)$ ,which is non-positive because $\sigma$ is submodular. Hence,the above difference is always non-positive,so $g$ satisfies the diminishing returns condition (1).

在这个表达式中，除了$\sigma \left( {A+\{ u,v\} }\right)  - \sigma \left( {A + u}\right)  -$ $\sigma \left( {A + v}\right)  + \sigma \left( A\right)$之外，所有项都是非负的（由${h}_{v}\left( \cdot \right)$的单调性可知），而$\sigma \left( {A+\{ u,v\} }\right)  - \sigma \left( {A + u}\right)  -$ $\sigma \left( {A + v}\right)  + \sigma \left( A\right)$是非正的，因为$\sigma$是次模的。因此，上述差值总是非正的，所以$g$满足收益递减条件(1)。

## 7. REFERENCES

## 7. 参考文献

[1] R. Albert, H. Jeong, A. Barabasi. Error and attack tolerance of complex networks. Nature 406(2000), 378-382.

[2] C. Asavathiratham, S. Roy, B. Lesieutre, G. Verghese. The Influence Model. IEEE Control Systems, Dec. 2001.

[3] C. Asavathiratham. The Influence Model: A Tractable Representation for the Dynamics of Networked Markov Chains. Ph.D. Thesis, MIT 2000.

[4] F. Bass. A new product growth model for consumer durables. Management Science 15(1969), 215-227.

[5] E. Berger. Dynamic Monopolies of Constant Size. Journal of Combinatorial Theory Series B 83(2001), 191-200.

[6] L. Blume. The Statistical Mechanics of Strategic Interaction.

[6] L. 布卢姆（L. Blume）。战略互动的统计力学。

Games and Economic Behavior 5(1993), 387-424.

[7] J. Brown, P. Reinegen. Social ties and word-of-mouth referral behavior. Journal of Consumer Research 14:3(1987), 350-362.

[8] J. Coleman, H. Menzel, E. Katz. Medical Innovations: A Diffusion Study Bobbs Merrill, 1966.

[9] G. Cornuejols, M. Fisher, G. Nemhauser. Location of Bank Accounts to Optimize Float. Management Science, 23(1977).

[10] P. Domingos, M. Richardson. Mining the Network Value of Customers. Seventh International Conference on Knowledge Discovery and Data Mining, 2001.

[11] R. Durrett. Lecture Notes on Particle Systems and Percolation. Wadsworth Publishing, 1988.

[12] G. Ellison. Learning, Local Interaction, and Coordination. Econometrica 61:5(1993), 1047-1071.

[13] J. Goldenberg, B. Libai, E. Muller. Talk of the Network: A Complex Systems Look at the Underlying Process of Word-of-Mouth. Marketing Letters 12:3(2001), 211-223.

[14] J. Goldenberg, B. Libai, E. Muller. Using Complex Systems Analysis to Advance Marketing Theory Development. Academy of Marketing Science Review 2001.

[15] M. Granovetter. Threshold models of collective behavior. American Journal of Sociology 83(6):1420-1443, 1978.

[16] V. Harinarayan, A. Rajaraman, J. Ullman. Implementing Data Cubes Efficiently. Proc. ACM SIGMOD 1996.

[17] T.M. Liggett. Interacting Particle Systems. Springer, 1985.

[18] M. Macy. Chains of Cooperation: Threshold Effects in Collective Action. American Sociological Review 56(1991).

[19] M. Macy, R. Willer. From Factors to Actors: Computational Sociology and Agent-Based Modeling. Ann. Rev. Soc. 2002.

[20] V. Mahajan, E. Muller, F. Bass. New Product Diffusion Models in Marketing: A Review and Directions for Research. Journal of Marketing 54:1(1990) pp. 1-26.

[21] S. Morris. Contagion. Review of Economic Studies 67(2000).

[22] G. Nemhauser, L. Wolsey. Integer and Combinatorial Optimization. John Wiley, 1988. .

[23] G. Nemhauser, L. Wolsey, M. Fisher. An analysis of the approximations for maximizing submodular set functions. Mathematical Programming, 14(1978), 265-294.

[24] M. Newman. The structure of scientific collaboration networks. Proc. Natl. Acad. Sci. 98(2001).

[25] D. Peleg. Local Majority Voting, Small Coalitions, and Controlling Monopolies in Graphs: A Review. 3rd Colloq. on Structural Information and Communication, 1996.

[26] M. Richardson, P. Domingos. Mining Knowledge-Sharing Sites for Viral Marketing. Eighth Intl. Conf. on Knowledge Discovery and Data Mining, 2002.

[27] E. Rogers. Diffusion of innovations Free Press, 1995.

[28] T. Schelling. Micromotives and Macrobehavior. Norton, 1978.

[29] T. Valente. Network Models of the Diffusion of Innovations. Hampton Press, 1995.

[30] S. Wasserman, K. Faust. Social Network Analysis. Cambridge University Press, 1994.

[31] D. Watts. A Simple Model of Global Cascades in Random Networks. Proc. Natl. Acad. Sci. 99(2002), 5766-71.

[32] H. Peyton Young. The Diffusion of Innovations in Social Networks. Santa Fe Institute Working Paper 02-04-018(2002).

[33] H. Peyton Young. Individual Strategy and Social Structure: An Evolutionary Theory of Institutions. Princeton, 1998.