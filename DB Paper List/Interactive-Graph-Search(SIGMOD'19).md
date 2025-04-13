# Interactive Graph Search

# 交互式图搜索

Yufei Tao

陶宇飞

taoyf@cse.cuhk.edu.hk

Chinese University of Hong Kong

香港中文大学

Hong Kong, China

中国香港

Yuanbing Li

李远冰

yb-li16@mails.tsinghua.edu.cn

Tsinghua University

清华大学

Beijing, China

中国北京

Guoliang Li

李国良

liguoliang@tsinghua.edu.cn

Tsinghua University

清华大学

Beijing, China

中国北京

## ABSTRACT

## 摘要

We study interactive graph search (IGS), with the conceptual objective of departing from the conventional "top-down" strategy in searching a poly-hierarchy, a.k.a. a decision graph. In IGS, a machine assists a human in looking for a target node $z$ in an acyclic directed graph $G$ ,by repetitively asking questions. In each question,the machine picks a node $u$ in $G$ ,asks a human "is there a path from $u$ to $z$ ?",and takes a boolean answer from the human. The efficiency goal is to locate $z$ with as few questions as possible. We describe algorithms that solve the problem by asking a provably small number of questions, and establish lower bounds indicating that the algorithms are optimal up to a small additive factor. An experimental evaluation is presented to demonstrate the usefulness of our solutions in real-world scenarios.

我们研究交互式图搜索（IGS），其概念目标是摆脱在搜索多层次结构（也称为决策图）时传统的“自上而下”策略。在交互式图搜索中，机器通过反复提问来协助人类在有向无环图 $G$ 中寻找目标节点 $z$。在每个问题中，机器在 $G$ 中选择一个节点 $u$，询问人类“是否存在从 $u$ 到 $z$ 的路径？”，并从人类那里得到一个布尔值答案。效率目标是用尽可能少的问题来定位 $z$。我们描述了通过证明所需问题数量较少来解决该问题的算法，并建立了下界，表明这些算法在一个小的附加因子范围内是最优的。我们进行了实验评估，以证明我们的解决方案在现实场景中的实用性。

## CCS CONCEPTS

## 计算机协会分类系统概念

- Information systems $\rightarrow$ Collaborative search.

- 信息系统 $\rightarrow$ 协作搜索。

## KEYWORDS

## 关键词

Interactive Graph Search; Algorithms; Lower Bounds

交互式图搜索；算法；下界

## ACM Reference Format:

## ACM引用格式：

Yufei Tao, Yuanbing Li, and Guoliang Li. 2019. Interactive Graph Search. In 2019 International Conference on Management of Data (SIGMOD '19), June 30-July 5, 2019, Amsterdam, Netherlands. ACM, New York, NY, USA, 18 pages. https://doi.org/10.1145/3299869.3319885

陶宇飞、李远兵和李国良。2019年。交互式图搜索。见《2019年国际数据管理会议（SIGMOD '19）》，2019年6月30日至7月5日，荷兰阿姆斯特丹。美国纽约州纽约市美国计算机协会（ACM），18页。https://doi.org/10.1145/3299869.3319885

## 1 INTRODUCTION

## 1 引言

This paper considers a problem that we refer to as interactive graph search (IGS). It is concerned with the scenario where a human needs to explore a potentially massive poly-hierarchy - a.k.a. an acyclic directed graph (DAG) where each edge represents specialization - in order to locate the deepest node that best describes a certain concept. The DAG, typically, is stored at a remote server, and must be communicated to the human, with a unit cost charged on every node communicated. The algorithmic challenge is to devise a strategy to minimize the amount of interaction.

本文考虑了一个我们称之为交互式图搜索（IGS）的问题。它涉及这样一种场景：人类需要探索一个可能规模巨大的多层次结构（也就是一个有向无环图（DAG），其中每条边代表一种特化关系），以便找到最能描述某个特定概念的最深节点。通常，这个有向无环图存储在远程服务器上，必须将其传达给人类，并且每次传达一个节点都要收取单位成本。算法面临的挑战是设计一种策略来最小化交互量。

In Section 2, we will elaborate on the common patterns behind a class of applications that can be modeled as IGS, but for an immediate illustration here, let us examine a scenario from [15] where a machine summons a human's help to tag a picture according to a certain hierarchy. Figure 1a shows part of such a hierarchy, which is stored at the machine and is not known to the human. Interaction is initiated by the machine, which asks questions for the human to answer. Each question has the form: "is this (picture) an $\mathbf{x}$ ",where $\mathbf{x}$ is the name of a node. Here are some examples along a path in the hierarchy: "is this a car?", "is this a nissan?", and "is this a sentra?". Upon receiving a yes-answer to all of them, the machine can now place the tag sentra on the picture confidently.

在第2节中，我们将详细阐述一类可以建模为IGS的应用背后的常见模式，但为了在此处立即进行说明，让我们考察文献[15]中的一个场景，其中机器请求人类的帮助，根据某个层次结构对图片进行标注。图1a展示了这样一个层次结构的一部分，它存储在机器中，人类并不知晓。交互由机器发起，机器提出问题让人类回答。每个问题的形式为：“这（张图片）是$\mathbf{x}$吗”，其中$\mathbf{x}$是一个节点的名称。以下是沿着层次结构中的一条路径的一些示例问题：“这是一辆汽车吗？”“这是一辆日产汽车吗？”以及“这是一辆Sentra（阳光）汽车吗？”在得到所有问题的肯定回答后，机器现在可以自信地在图片上标注“Sentra（阳光）”。

The efficiency goal in the above scenario is to minimize the number of questions asked. More formally, one can think of the problem as a game between two players Alice and Bob. Initially,Bob secretly chooses a target node $z$ in the hierarchy. Alice’s job is to figure out which node is $z$ . There is an oracle that Alice can inquire repeatedly. Each time, she picks (at her will) a query node $q$ ,and asks the oracle: is there a (directed) path in the hierarchy from $q$ to the node chosen by Bob? Oracle reveals the answer (i.e., yes or no). So, what should be Alice’s strategy in order to locate $z$ with as few questions as possible?

上述场景中的效率目标是最小化提出的问题数量。更正式地说，我们可以将这个问题看作是两个玩家爱丽丝（Alice）和鲍勃（Bob）之间的一个游戏。最初，鲍勃在层次结构中秘密选择一个目标节点$z$。爱丽丝的任务是找出哪个节点是$z$。有一个神谕（oracle），爱丽丝可以反复向其询问。每次，她（按自己的意愿）选择一个查询节点$q$，并问神谕：在层次结构中是否存在从$q$到鲍勃所选节点的（有向）路径？神谕会给出答案（即“是”或“否”）。那么，为了用尽可能少的问题找到$z$，爱丽丝应该采取什么策略呢？

For our example, one sees that the machine plays the role of Alice. The target node $z$ is the final tag sentra that the machine should place on the picture. The human plays the role of oracle. Indeed, even though a human is not aware of the underlying hierarchy (let alone $z$ ), $\mathrm{s}$ /he can still correctly answer a question like "is this a car?" using her/his own knowledge and cognition power. A path exists in the hierarchy from $q = {car}$ to (the unknown) $z$ if and only if the human answers yes.

在我们的例子中，可以看到机器扮演了爱丽丝的角色。目标节点$z$是机器应该在图片上标注的最终标签“Sentra（阳光）”。人类扮演了神谕的角色。实际上，即使人类不知道底层的层次结构（更不用说$z$了），他/她仍然可以利用自己的知识和认知能力正确回答像“这是一辆汽车吗？”这样的问题。当且仅当人类回答“是”时，在层次结构中才存在从$q = {car}$到（未知的）$z$的路径。

While the above hierarchy is a tree, it can be a DAG in general. Figure 1b complements Figure 1a by showing another part of the hierarchy. When presented with the picture of a whale, a human will answer yes to both questions: "is

虽然上述层次结构是一棵树，但一般情况下它可以是一个有向无环图。图1b通过展示层次结构的另一部分对图1a进行了补充。当看到一张鲸鱼的图片时，人类会对以下两个问题都回答“是”：“这是

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.SIGMOD '19, June 30-July 5, 2019, Amsterdam, Netherlands

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且拷贝上要带有此通知和首页的完整引用信息。必须尊重本作品中除作者之外其他人拥有版权的组件。允许进行带引用的摘要。否则，如需复制、重新发布、上传到服务器或分发给列表，需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。SIGMOD '19，2019年6月30日至7月5日，荷兰阿姆斯特丹

© 2019 Copyright held by the owner/author(s). Publication rights licensed to ACM.

© 2019 版权归所有者/作者所有。出版权已授权给美国计算机协会（ACM）。

ACM ISBN 978-1-4503-5643-5/19/06...\$15.00

美国计算机协会（ACM）ISBN 978 - 1 - 4503 - 5643 - 5/19/06... 15.00美元

https://doi.org/10.1145/3299869.3319885

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: vehicle animal mammal oviparous aquatic terrestrial amphibious tiger whale (b) A DAG hierarchy car nissan honda mercedes maxima sentra (a) A tree hierarchy (reconstructed from [15]) -->

<img src="https://cdn.noedgeai.com/0195c91c-378f-77b0-8d6e-bb14508995e7_1.jpg?x=224&y=231&w=1318&h=258&r=0"/>

Figure 1: Example hierarchies for human-assisted graph search

图1：人类辅助图搜索的示例层次结构

<!-- Media -->

this a mammal?" and "is this an aquatic (animal)?" This is consistent with the fact that both mammal and aquatic have paths leading to the node whale in the hierarchy. Indeed, to tag the picture correctly, the machine can reach the node whale by asking questions along either path.

这是一种哺乳动物吗？”以及“这是一种水生（动物）吗？”这与层次结构中“哺乳动物”和“水生动物”都有路径通向“鲸鱼”节点这一事实相符。实际上，为了正确标注图片，机器可以沿着任意一条路径提出问题来到达“鲸鱼”节点。

Technical Challenges. The technical objectives of this work are two-fold:

技术挑战。这项工作的技术目标有两个方面：

- (Upper bound) Design an algorithm for Alice (i.e., the machine) to solve the problem with a small number of questions,regardless of Bob’s choice of $z$ .

- （上界）为爱丽丝（即机器）设计一种算法，使其能用较少的问题解决该问题，而无需考虑鲍勃对 $z$ 的选择。

- (Lower bound) Where is the limit of all algorithms? That is, how many questions must Alice ask in the worst case, no matter how smart she is?

- （下界）所有算法的极限在哪里？也就是说，无论爱丽丝多么聪明，在最坏的情况下她必须问多少个问题？

Our problem is online in nature, namely, the next question Alice asks depends on the previous answers of the oracle. Such interaction with the oracle is essential for keeping the total number of questions small. Opposite to this is the offline version, as was studied by Parameswaran et al. in [15] under the name human-assisted graph search, where no interactions are permitted. Instead, Alice must ask all her questions in one go. Once the answers are returned, she must then do her best to figure out where is $z$ . Suppose that Alice is constrained to ask no more than $t$ questions for some small $t \geq  1$ . In this case,she cannot guarantee finding $z$ even with all the $t$ answers collected; instead, all she can do is to narrow things down to a candidate set that must contain $z$ . The objective (of the offline version) is to choose the $t$ questions wisely to minimize the size of the candidate set.

我们的问题本质上是在线的，即爱丽丝提出的下一个问题取决于神谕的先前回答。与神谕的这种交互对于减少问题总数至关重要。与之相反的是离线版本，正如帕拉梅斯瓦兰（Parameswaran）等人在文献 [15] 中以人工辅助图搜索的名义所研究的那样，在该版本中不允许进行交互。相反，爱丽丝必须一次性提出所有问题。一旦收到答案，她就必须尽力找出 $z$ 的位置。假设爱丽丝被限制最多提出 $t$ 个问题，其中 $t \geq  1$ 是一个较小的值。在这种情况下，即使收集到了所有 $t$ 个答案，她也无法保证找到 $z$；相反，她所能做的只是将范围缩小到一个必定包含 $z$ 的候选集。（离线版本的）目标是明智地选择 $t$ 个问题，以最小化候选集的大小。

To illustrate,consider Figure 1a with $t = 3$ ; and assume that Alice asks the oracle: "is this an $\mathbf{x}$ ?",where $\mathbf{x} =$ nissan, honda, and mercedes, respectively. If the first question gets a yes answer,she is sure that $z$ must be in $\{$ nissan,maxima, sentra\}. This candidate set is her final knowledge because no more interaction with the oracle is allowed.

为了说明这一点，考虑图 1a 中的 $t = 3$；并假设爱丽丝问神谕：“这是一个 $\mathbf{x}$ 吗？”，其中 $\mathbf{x} =$ 分别为日产（Nissan）、本田（Honda）和梅赛德斯（Mercedes）。如果第一个问题得到肯定回答，她可以确定 $z$ 必定在 $\{$ 日产（Nissan）、千里马（Maxima）、阳光（Sentra）} 中。这个候选集就是她的最终信息，因为不允许再与神谕进行交互。

However, in the unlucky case where none of the questions returns yes,Alice has little information as to where is $z$ . To see this, just imagine an overall hierarchy combining Figures 1a and 1b. Alice does not even have a clue whether $z$ is a vehicle or an animal. Indeed, as shown in [15], the value of $t$ must be rather large - often at the same order as the size of the hierarchy - to guarantee a small candidate set.

然而，在不幸的情况下，即没有一个问题得到肯定回答时，爱丽丝对于 $z$ 的位置几乎没有任何信息。为了理解这一点，只需想象一个结合了图 1a 和图 1b 的整体层次结构。爱丽丝甚至不知道 $z$ 是车辆还是动物。实际上，正如文献 [15] 所示，$t$ 的值必须相当大——通常与层次结构的规模处于同一数量级——才能保证候选集较小。

This issue goes away in IGS. As explained next, typically only a small number of questions suffices to locate $z$ ,even in the worst case.

在交互式图搜索（IGS）中，这个问题就不存在了。正如接下来要解释的，通常只需少量问题就能定位 $z$，即使在最坏的情况下也是如此。

Our Contributions. Let us now return to IGS. If the hierarchy has $n$ nodes,the problem can be trivially solved with $n$ questions: simply ask a question on every node. A bit less trivial is to do so with at most $d \cdot  h$ questions - we will explain how in Section 2.2 - where $d$ is the maximum out-degree of a node,and $h$ is the length of the longest path in the hierarchy. Note that $h$ is at least $\left\lceil  {{\log }_{d}n}\right\rceil$ ,but can be as large as $n$ when the hierarchy is a single path.

我们的贡献。现在让我们回到交互式图搜索（IGS）。如果层次结构有 $n$ 个节点，那么用 $n$ 个问题就可以轻松解决该问题：只需对每个节点问一个问题。稍微复杂一点的是最多用 $d \cdot  h$ 个问题来解决——我们将在 2.2 节中解释如何实现——其中 $d$ 是节点的最大出度，$h$ 是层次结构中最长路径的长度。请注意，$h$ 至少为 $\left\lceil  {{\log }_{d}n}\right\rceil$，但当层次结构是单一路径时，$h$ 可以大到 $n$。

We show that the problem admits an algorithm with an alternative bound on the number of questions, and prove that the algorithm is nearly optimal:

我们证明了该问题存在一种算法，该算法对问题数量有另一种界，并证明该算法几乎是最优的：

- (Upper bound) We can find $z$ in a DAG with at most $\left\lceil  {{\log }_{2}h}\right\rceil  \left( {1 + \left\lfloor  {{\log }_{2}n}\right\rfloor  }\right)  + \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$ questions.

- （上界）我们可以在有向无环图（DAG）中最多用 $\left\lceil  {{\log }_{2}h}\right\rceil  \left( {1 + \left\lfloor  {{\log }_{2}n}\right\rfloor  }\right)  + \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$ 个问题找到 $z$。

- (Lower bound) Any algorithm must ask at least $\left( {d - 1}\right)  \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor$ questions in the worst case. In other words, the proposed algorithm is optimal up to a small additive factor.

- （下界）在最坏的情况下，任何算法至少必须问 $\left( {d - 1}\right)  \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor$ 个问题。换句话说，所提出的算法在一个小的加性因子范围内是最优的。

Our algorithm carefully decomposes the nodes of the input DAG hierarchy into disjoint subsets, where the nodes in each subset are connected by a path in the hierarchy. The decomposition allows us to navigate in the hierarchy through a series of binary searches on individual paths. This new technique is interesting in its own right, and is an outcome from the marriage of the white-path theorem and heavy-path decomposition (both will be explained in Section 3). In fact, the technique is - as we will show - powerful enough to settle near-optimally a more general variant of IGS where a human may need to answer multiple questions at a time.

我们的算法仔细地将输入有向无环图（DAG）层次结构的节点分解为不相交的子集，其中每个子集中的节点通过层次结构中的一条路径相连。这种分解使我们能够通过对各个路径进行一系列二分搜索来在层次结构中导航。这种新技术本身就很有趣，它是白路径定理和重路径分解（两者都将在第 3 节中解释）相结合的产物。事实上，正如我们将展示的，这种技术足够强大，能够近乎最优地解决交互式图搜索（IGS）的一个更一般的变体问题，在该变体中，人类可能需要一次性回答多个问题。

Paper Organization. The rest of the paper is organized as follows. Section 2 will formally define the IGS problem, give a baseline solution, and present a class of applications that are adequately modeled by IGS. Section 3 reviews some preliminary techniques needed in our discussion. Sections 4 and 5 will present our algorithms and prove their theoretical guarantees, focusing on tree and DAG hierarchies, respectively. Section 6 will describe how to extend our algorithms to solve a more general variant of the problem. Section 7 will experimentally evaluate the performance of the proposed solutions using real data. Section 8 will survey the previous work related to ours. Finally, Section 9 concludes the paper with a summary of findings.

论文结构。论文的其余部分组织如下。第2节将正式定义交互式图搜索（IGS）问题，给出一个基线解决方案，并介绍一类可以用IGS进行充分建模的应用。第3节回顾我们讨论所需的一些预备技术。第4节和第5节将分别针对树和有向无环图（DAG）层次结构，提出我们的算法并证明其理论保证。第6节将描述如何扩展我们的算法以解决该问题的更一般变体。第7节将使用真实数据对所提出解决方案的性能进行实验评估。第8节将综述与我们相关的先前工作。最后，第9节总结研究结果，结束本文。

## 2 INTERACTIVE GRAPH SEARCH

## 2 交互式图搜索

### 2.1 Problem Formulation

### 2.1 问题表述

Next, we will formally define the interactive graph search (IGS) problem studied in this paper. Some of the notions that already appeared in Section 1 will be repeated for the reader's convenience.

接下来，我们将正式定义本文研究的交互式图搜索（IGS）问题。为方便读者，第1节中已出现的一些概念将重复介绍。

We have a hierarchy,which is a connected DAG $G = \left( {V,E}\right)$ . Define a node $v \in  V$ as a root if it has an in-degree 0 (i.e., no incoming edges). We consider that $G$ has only one root - if this is not true,simply add a dummy vertex to $G$ with an outgoing edge to every original root. This dummy vertex has an out-degree equal to the number of roots in the original $G$ , and now serves as the only root of $G$ .

我们有一个层次结构，它是一个连通的有向无环图 $G = \left( {V,E}\right)$ 。如果一个节点 $v \in  V$ 的入度为0（即没有入边），则将其定义为根节点。我们假设 $G$ 只有一个根节点——如果不是这样，只需向 $G$ 中添加一个虚拟顶点，并从该虚拟顶点向每个原始根节点添加一条出边。这个虚拟顶点的出度等于原始 $G$ 中根节点的数量，现在它作为 $G$ 的唯一根节点。

An adversary chooses arbitrarily a target node $z \in  V$ . An algorithm’s goal is to identify which node is $z$ . There is an oracle that can answer questions. Formally, in each question, the algorithm specifies a query node $q \in  V$ ; and then the oracle returns a boolean answer denoted as reach(q):

对手任意选择一个目标节点 $z \in  V$ 。算法的目标是确定哪个节点是 $z$ 。有一个神谕可以回答问题。形式上，在每个问题中，算法指定一个查询节点 $q \in  V$ ；然后神谕返回一个布尔答案，记为 reach(q)：

- yes,if there is a (directed) path from $q$ to $z$ ;

- 如果从 $q$ 到 $z$ 存在一条（有向）路径，则为“是”；

- no, otherwise.

- 否则为“否”。

In other words,the answer reveals the reachability from $q$ to $z$ ,that is, $\operatorname{reach}\left( q\right)  = {yes}$ ,if and only if $z$ is reachable from $q$ , or equivalently, $q$ can reach $z$ . The algorithm is free to choose any query node in a question; and indeed, its choice in each question constitutes the core of the algorithm design.

换句话说，答案揭示了从 $q$ 到 $z$ 的可达性，即 $\operatorname{reach}\left( q\right)  = {yes}$ ，当且仅当 $z$ 可从 $q$ 到达，或者等价地， $q$ 可以到达 $z$ 。算法可以在一个问题中自由选择任何查询节点；实际上，它在每个问题中的选择构成了算法设计的核心。

The algorithm stops when it has figured out with no ambiguity where is $z$ . Its cost is defined as the number of questions it has asked.

当算法明确无误地确定 $z$ 的位置时，算法停止。其成本定义为它所提出的问题数量。

Throughout the paper,we set $n = \left| V\right|$ ,denote by $d$ the maximum out-degree of the nodes in $G$ ,and by $h$ the length of the longest path in $G$ . For instance,if $G$ is the DAG in Figure 2,then $n = {14},d = 3$ and $h = 5$ (the path from node 1 to node 14 is the longest in $G$ ).

在整篇论文中，我们设 $n = \left| V\right|$ ，用 $d$ 表示 $G$ 中节点的最大出度，用 $h$ 表示 $G$ 中最长路径的长度。例如，如果 $G$ 是图2中的有向无环图，则 $n = {14},d = 3$ 且 $h = 5$ （从节点1到节点14的路径是 $G$ 中最长的路径）。

### 2.2 A Baseline Top-Down Solution

### 2.2 一种基线的自顶向下解决方案

IGS has the following simple out-neighbor property:

IGS具有以下简单的出邻节点属性：

Proposition 1 (Out-Neighbor Property). Suppose that we already know reach $\left( u\right)  =$ yes for some node $u \in  V$ . Then:

命题1（出邻节点属性）。假设我们已经知道对于某个节点 $u \in  V$ ，reach $\left( u\right)  =$ 为“是”。那么：

- $u = z$ if and only if every out-neighbor $v$ of $u$ satisfies $\operatorname{reach}\left( v\right)  = {no}$ ;

- $u = z$ 当且仅当 $u$ 的每个出邻节点 $v$ 都满足 $\operatorname{reach}\left( v\right)  = {no}$ ；

- $u \neq  z$ if and only if $u$ has an out-neighbor $v$ satisfying $\operatorname{reach}\left( v\right)  =$ yes.

- $u \neq  z$ 当且仅当 $u$ 有一个出邻节点 $v$ 满足 $\operatorname{reach}\left( v\right)  =$ 为“是”。

Example. To illustrate,suppose that the input DAG $G$ is the graph in Figure 2. Assume that the target node $z$ is node 8 . Set $u$ to $z$ ; the first bullet says that no out-neighbors of $u$ can reach $z$ (that is rather trivial). Set instead $u$ to node 2; it is clear that reach $\left( {\text{node }2}\right)  =$ yes. The second bullet says that node 2 must have an out-neighbor that can reach $z$ . Indeed, in this case, both nodes 3 and 4 can be this out-neighbor.

示例。为了说明这一点，假设输入的有向无环图（DAG）$G$是图2中的图。假设目标节点$z$是节点8。将$u$设为$z$；第一条规则表明，$u$的任何出邻接点都无法到达$z$（这相当显然）。相反，将$u$设为节点2；显然，能到达$\left( {\text{node }2}\right)  =$，答案为是。第二条规则表明，节点2必须有一个能到达$z$的出邻接点。实际上，在这种情况下，节点3和节点4都可以作为这个出邻接点。

<!-- Media -->

<!-- figureText: ⑩ ③ -->

<img src="https://cdn.noedgeai.com/0195c91c-378f-77b0-8d6e-bb14508995e7_2.jpg?x=1121&y=238&w=325&h=322&r=0"/>

Figure 2: A DAG hierarchy

图2：有向无环图层次结构

<!-- Media -->

The property motivates a straightforward top-down algorithm for IGS. At the beginning,set $u$ to the root of $G$ . At each step,query the oracle on every out-neighbor of $u$ , until finding an out-neighbor $v$ with reach $\left( v\right)  =$ yes. If no such $v$ exists,we terminate by returning $u$ as the target node. Otherwise,we set $u$ to $v$ ,and repeat.

该性质为IGS（信息引导搜索）问题催生了一种直接的自顶向下算法。一开始，将$u$设为$G$的根节点。在每一步，对$u$的每个出邻接点向神谕（oracle）进行查询，直到找到一个能到达$\left( v\right)  =$（答案为是）的出邻接点$v$。如果不存在这样的$v$，我们就将$u$作为目标节点返回并终止算法。否则，我们将$u$设为$v$，然后重复上述过程。

Clearly,top-down asks at most $d$ questions at every $u$ . By moving from $u$ to $v$ ,it walks a step along a path in $G$ . Hence, the algorithm asks at most $d \cdot  h$ questions in total.

显然，自顶向下算法在每个$u$上最多询问$d$个问题。通过从$u$移动到$v$，它沿着$G$中的一条路径前进了一步。因此，该算法总共最多询问$d \cdot  h$个问题。

### 2.3 Applications

### 2.3 应用

The IGS problem offers an algorithmic framework for studying how to minimize interaction in applications where the objective is to locate the most specific node in a decision tree - or its extension decision graph [13] - that best fulfills an information need. Figure 1a illustrates a decision tree, while Figure 1b exemplifies a decision graph, which can be regarded as a decision tree but with identical subtrees merged. Conventionally, the top-down strategy in Section 2.2 has been the norm for exploring a decision tree/graph. Philosophically, the goal of IGS is to seek a way to beat that conventional wisdom.

IGS问题为研究如何在应用中最小化交互提供了一个算法框架，这些应用的目标是在决策树（或其扩展——决策图[13]）中找到最能满足信息需求的最具体节点。图1a展示了一个决策树，而图1b举例说明了一个决策图，决策图可以看作是合并了相同子树的决策树。传统上，2.2节中的自顶向下策略一直是探索决策树/图的标准方法。从理念上讲，IGS的目标是寻求一种方法来打破这种传统观念。

The concrete scenarios for such applications are versatile. What is described in Section 1 is known as image categorization in [15]. Next, we will describe several other applications. Our selection strives to achieve diversity: each application below is representative with distinct features.

此类应用的具体场景多种多样。第1节中描述的内容在文献[15]中被称为图像分类。接下来，我们将描述其他几个应用。我们的选择力求实现多样性：下面的每个应用都具有独特的特征，具有代表性。

Manual Curation. In the example with Figures 1a and 1b, the input hierarchy is fixed, with the goal being to find a node to fit a certain object (i.e., a picture). In manual curation, on the other hand, we want to extend a hierarchy by inserting a new node $\mathbf{x}$ ,e.g.,a new brand of nissan. This is effectively an instance of IGS, whose output is the node that should parent $x$ . In reality,many hierarchies (better known as taxonomies or categories) require this kind of periodic extensions; some examples are Wikipedia, web of concepts, ACM computing classification system, and so on.

手动编目。在图1a和图1b的示例中，输入的层次结构是固定的，目标是找到一个适合某个对象（即一张图片）的节点。另一方面，在手动编目中，我们希望通过插入一个新节点$\mathbf{x}$来扩展层次结构，例如，日产的一个新品牌。这实际上是IGS的一个实例，其输出是应该作为$x$父节点的节点。实际上，许多层次结构（更常见的叫法是分类法或类别）都需要这种定期扩展；一些例子包括维基百科、概念网络、ACM计算分类系统等等。

Relational Databases. Often times a user may need to search a database without being aware of the table schemata, ruling out the possibility to write an accurate SQL query to fetch the information targeted. This motivated faceted search $\left\lbrack  {{17},{20}}\right\rbrack$ ,where the system interacts with the user by asking increasingly refined questions that eventually lead to the data to be retrieved. These questions are selected during preprocessing, and are organized into a decision tree/graph, after which faceted search can be performed online by descending an appropriate path in the tree/graph. Our IGS algorithms nicely complement faceted search, which is exactly an instance of IGS.

关系数据库。很多时候，用户可能需要在不了解表结构的情况下搜索数据库，这就排除了编写准确的SQL查询来获取目标信息的可能性。这催生了分面搜索$\left\lbrack  {{17},{20}}\right\rbrack$，在分面搜索中，系统通过提出越来越精确的问题与用户进行交互，最终找到要检索的数据。这些问题是在预处理阶段选择的，并被组织成一个决策树/图，之后可以通过在树/图中沿着合适的路径向下搜索来在线执行分面搜索。我们的IGS算法很好地补充了分面搜索，分面搜索实际上就是IGS的一个实例。

A Commercial Site. Zingtree.com is the portal site of a company that specializes in helping organizations build a sophisticated decision tree/graph designed to facilitate one of the following services: technical support, call centers, customer care, retail, and medical and health. To provide, for example, technical support, an organization would rely on the decision tree/graph to interact with a customer, in order to diagnose the problem encountered by the customer and to suggest the corresponding remedy. This is a typical scenario of IGS. The algorithms in this paper can be integrated with any of those decision trees/graphs to reduce the amount of interaction demanded (which is crucial for the services aforementioned).

商业网站。Zingtree.com是一家专门帮助组织构建复杂决策树/图的公司的门户网站，这些决策树/图旨在促进以下服务之一：技术支持、呼叫中心、客户服务、零售以及医疗保健。例如，为了提供技术支持，一个组织会依靠决策树/图与客户进行交互，以诊断客户遇到的问题并提出相应的解决方案。这是IGS的一个典型场景。本文中的算法可以与任何这些决策树/图集成，以减少所需的交互量（这对于上述服务至关重要）。

## 3 PRELIMINARIES

## 3 预备知识

### 3.1 Heavy-Path Decomposition

### 3.1 重路径分解

In this subsection, we give a self-contained tutorial to the heavy-path decomposition technique [18]. Let $T$ be a tree of $n$ nodes which may not be balanced,i.e.,its height can be arbitrarily close to $n$ . The goal of heavy-path decomposition is to produce a balanced representation of $T$ .

在本小节中，我们将对重路径分解技术 [18] 进行独立的介绍。设 $T$ 是一棵包含 $n$ 个节点的树，该树可能不平衡，即其高度可以任意接近 $n$。重路径分解的目标是生成 $T$ 的一种平衡表示。

We need to be first familiar with the notions of "heavy edges" and "light edges". Consider $u$ to be an internal node in $T$ . Let $v$ be the child node of $u$ whose subtree has the largest ${\text{size}}^{1}$ (ties broken arbitrarily). The edge between $u$ and $v$ is said to be heavy,while the other out-going edges of $u$ are said to be light.

我们首先需要熟悉“重边”和“轻边”的概念。假设 $u$ 是 $T$ 中的一个内部节点。设 $v$ 是 $u$ 的子节点，且其对应的子树具有最大的 ${\text{size}}^{1}$（若有多个子树的 ${\text{size}}^{1}$ 相同，则任意选择一个）。$u$ 和 $v$ 之间的边称为重边，而 $u$ 的其他出边则称为轻边。

Example. Suppose that $T$ is the tree in Figure 3a (all the edges are pointing downwards). The subtree of node 4 has a size 6 , while that of node 5 has a size 5 . Set $u$ to node 2 . Among its three out-going edges, the one pointing to node 4 is heavy, while the other two are light - because node 4 is the child of $u$ with the largest subtree. In the figure,all the heavy edges are represented using white arrows, whereas the light ones have black arrows.

示例。假设 $T$ 是图 3a 中的树（所有边都指向下方）。节点 4 的子树大小为 6，而节点 5 的子树大小为 5。设 $u$ 为节点 2。在其三条出边中，指向节点 4 的边是重边，而另外两条是轻边，因为节点 4 是 $u$ 的子节点中拥有最大子树的节点。在图中，所有重边用白色箭头表示，而轻边用黑色箭头表示。

Now, concatenate heavy edges into maximal paths, i.e., no path can be extended with yet another heavy edge. Every resulting path is called a heavy path. Example (cont.). In Figure 3a,path "nodes $2 \rightarrow  4 \rightarrow  8$ ", which we abbreviate as(2,4,8)henceforth,is not a heavy path, because it can be extended with a white edge either in front or at the end. On the other hand,path(1,2,4,8,10)is a heavy path,and so is(5,9,12,14). Do not forget there are five more heavy paths: (3), (6), (7), (11), and (13); they are heavy paths of length 0 .

现在，将重边连接成最大路径，即没有路径可以再用另一条重边进行扩展。每个得到的路径都称为重路径。示例（续）。在图 3a 中，路径“节点 $2 \rightarrow  4 \rightarrow  8$”，我们此后将其缩写为 (2, 4, 8)，不是重路径，因为它可以在前端或末端用一条白色边进行扩展。另一方面，路径 (1, 2, 4, 8, 10) 是重路径，(5, 9, 12, 14) 也是重路径。不要忘记还有另外五条重路径：(3)、(6)、(7)、(11) 和 (13)；它们是长度为 0 的重路径。

<!-- Media -->

<!-- figureText: ① ${\pi }_{1}$ ⑤⑨⑫④ (b) The path-tree $\Pi$ ③ ⑤ ⑥ ⑩ ⑬ (a) A tree $T$ -->

<img src="https://cdn.noedgeai.com/0195c91c-378f-77b0-8d6e-bb14508995e7_3.jpg?x=978&y=236&w=615&h=376&r=0"/>

Figure 3: Heavy-path decomposition

图 3：重路径分解

<!-- figureText: (a) A DFS-tree ⑫ ⑭ (b) Colors when node 3 is discovered on the DAG of Figure 2 -->

<img src="https://cdn.noedgeai.com/0195c91c-378f-77b0-8d6e-bb14508995e7_3.jpg?x=967&y=677&w=639&h=402&r=0"/>

Figure 4: White-path theorem

图 4：白路径定理

<!-- Media -->

Every node appears in one and exactly one heavy path (observe this property from Figure 3a.) By viewing each heavy path as a whole,we can define a path tree $\Pi$ as follows:

每个节点恰好出现在一条重路径中（从图 3a 中可以观察到这一性质）。通过将每条重路径视为一个整体，我们可以按如下方式定义路径树 $\Pi$：

- Treat each heavy path as a "super-node", and make it a vertex in $\Pi$ .

- 将每条重路径视为一个“超级节点”，并使其成为 $\Pi$ 中的一个顶点。

- Given two heavy paths $\widehat{\pi }$ and $\pi$ ,add an edge in $\Pi$ from $\widehat{\pi }$ to $\pi$ if and only if a node of $\widehat{\pi }$ parents the first node of $\pi$ in $T$ .

- 给定两条重路径 $\widehat{\pi }$ 和 $\pi$，当且仅当在 $T$ 中 $\widehat{\pi }$ 的一个节点是 $\pi$ 的第一个节点的父节点时，在 $\Pi$ 中从 $\widehat{\pi }$ 到 $\pi$ 添加一条边。

Example (cont.). Figure 3b shows the path tree $\Pi$ for the $T$ in Figure 3a. II has 7 vertices ${\pi }_{1},{\pi }_{2},\ldots ,{\pi }_{7}$ ,corresponding to 7 heavy paths in $T$ ,respectively. There is an edge from ${\pi }_{1}$ to ${\pi }_{6}$ because node 2 of ${\pi }_{1}$ parents node 5 - the first node of ${\pi }_{6}$ - in $T$ . Likewise,an edge exists from ${\pi }_{6}$ to ${\pi }_{7}$ because node 9 parents node 13 in $T$ .

示例（续）。图 3b 展示了图 3a 中 $T$ 对应的路径树 $\Pi$。它有 7 个顶点 ${\pi }_{1},{\pi }_{2},\ldots ,{\pi }_{7}$，分别对应 $T$ 中的 7 条重路径。从 ${\pi }_{1}$ 到 ${\pi }_{6}$ 有一条边，因为在 $T$ 中 ${\pi }_{1}$ 的节点 2 是 ${\pi }_{6}$ 的第一个节点（节点 5）的父节点。同样，从 ${\pi }_{6}$ 到 ${\pi }_{7}$ 有一条边，因为在 $T$ 中节点 9 是节点 13 的父节点。

The path tree $\Pi$ in Figure 3b has 3 levels. It can be proved that $\Pi$ cannot be too tall in general:

图 3b 中的路径树 $\Pi$ 有 3 层。可以证明，一般情况下 $\Pi$ 不会太高：

LEMMA 1 ([18]). $\Pi$ has at most $1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$ levels.

引理 1 ([18])。$\Pi$ 最多有 $1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$ 层。

---

<!-- Footnote -->

${}^{1}$ The size of a subtree is the number of nodes therein.

${}^{1}$ 子树的大小是其中节点的数量。

<!-- Footnote -->

---

### 3.2 DFS and White-Path Theorem

### 3.2 深度优先搜索与白路径定理

We devote this subsection to depth-first search (DFS), which will play an essential role in our IGS solutions. The DFS algorithm is "deceptively simple", and is endowed with numerous interesting properties. Our main goal is to review the white-path theorem: the famous theorem that explains why DFS is the key to solving a long list of non-trivial problems, e.g., cycle detection, topological sort, finding strongly-connected components, etc. The IGS problem will be a new member on the list, as we will show in this paper.

我们在本小节中介绍深度优先搜索（DFS），它将在我们的 IGS 解决方案中发挥重要作用。DFS 算法“看似简单”，却具有许多有趣的性质。我们的主要目标是回顾白路径定理：这个著名的定理解释了为什么 DFS 是解决一系列非平凡问题（如环检测、拓扑排序、寻找强连通分量等）的关键。正如我们将在本文中展示的，IGS 问题将成为这个列表中的新成员。

DFS. We will only be concerned with a connected DAG $G =$ (V,E)that has a single root $r$ . DFS traverses $G$ by resorting to a stack and a vertex coloring scheme:

深度优先搜索（DFS）。我们仅关注一个连通的有向无环图（DAG） $G =$ (V,E)，它有一个单一的根节点 $r$ 。深度优先搜索通过借助一个栈和顶点着色方案来遍历 $G$ ：

- White: a vertex has never been pushed into the stack.

- 白色：一个顶点从未被压入栈中。

- Gray: a vertex is currently in the stack.

- 灰色：一个顶点当前在栈中。

- Black: a vertex has been popped out of the stack.

- 黑色：一个顶点已从栈中弹出。

In the outset,the stack contains only $r$ (we always start DFS from the root in this work). Accordingly, all the vertices are colored white,except $r$ ,which is colored gray. The algorithm then proceeds as follows:

一开始，栈中仅包含 $r$ （在这项工作中，我们总是从根节点开始进行深度优先搜索）。因此，除了 $r$ 被染成灰色外，所有顶点都被染成白色。然后算法按如下步骤进行：

1. while stack not empty

1. 当栈不为空时

2. $\;u \leftarrow$ the vertex at the top of the stack

2. $\;u \leftarrow$ 栈顶的顶点

3. if $u$ has any white out-neighbor $v$

3. 如果 $u$ 有任何白色的出邻接点 $v$

4. push $v$ into the stack,and color it gray

4. 将 $v$ 压入栈中，并将其染成灰色

else

否则

5. pop $u$ out of the stack,and color it black

5. 将 $u$ 从栈中弹出，并将其染成黑色

At the moment right before $v$ turns gray at Line $4 -$ namely, after it is found at Line 3 as a white out-neighbor of $u -$ we say that $v$ is discovered,and $u$ is its finder. Every node other than $r$ is discovered once and exactly once in the algorithm.

在第 $4 -$ 行 $v$ 变为灰色之前的那一刻，即，在第3行发现它是 $u -$ 的白色出邻接点之后，我们称 $v$ 被发现， $u$ 是它的发现者。在算法中，除 $r$ 之外的每个节点都恰好被发现一次。

Example. Let $G$ be the DAG in Figure 2. Suppose that,at Line 3,we adopt the policy that the out-neighbors of $u$ be picked in ascending order of node id. At the beginning, the stack has only node 1 . Node 2 is discovered next, with node 1 as the finder. In turn, node 2 is the finder for node 3 , which is the finder for node 8 , which is the finder of node 10 . At this moment,the stack has (from bottom to top): nodes1,2,3,8, 10 ; these five nodes are in gray, while the other nodes are still white. Node 10 is then popped out, and turns black. Node 8 currently tops the stack, with only one white out-neighbor: node 11. Hence, node 11 is discovered next, making the stack: nodes1,2,3,8,11. We omit the rest of the execution.

示例。设$G$为图2中的有向无环图（DAG）。假设在第3行，我们采用按节点ID升序选取$u$的出邻接点的策略。开始时，栈中只有节点1。接下来发现节点2，节点1是其发现者。接着，节点2是节点3的发现者，节点3是节点8的发现者，节点8是节点10的发现者。此时，栈中（从底到顶）有：节点1、2、3、8、10；这五个节点为灰色，而其他节点仍为白色。然后节点10出栈并变为黑色。当前栈顶节点为节点8，它只有一个白色出邻接点：节点11。因此，接下来发现节点11，栈变为：节点1、2、3、8、11。我们省略其余的执行过程。

DFS-Tree. The traversal order of DFS defines a DFS-tree $T$ as follows:

深度优先搜索树（DFS - Tree）。深度优先搜索（DFS）的遍历顺序定义了一棵深度优先搜索树$T$，如下所示：

- The set of vertices of $T$ is just $V$ .

- $T$的顶点集就是$V$。

- $T$ is rooted at $r$ .

- $T$以$r$为根。

- If a node $u$ is the finder of a node $v,u$ parents $v$ in $T$ .

- 如果节点$u$是节点$v,u$的发现者，则在$T$中$v$是其父节点。

Example (cont.). Figure 4a gives the DFS-tree for the execution of DFS illustrated earlier. It is worth mentioning that every path emanating from the root represents the content of the stack at some point of the algorithm. For example, the path(1,2,3,8,11)is the content of the stack right after node 11 was discovered (as shown earlier).

示例（续）。图4a给出了前面所说明的深度优先搜索执行过程的深度优先搜索树。值得一提的是，从根节点出发的每条路径都代表了算法在某一时刻栈的内容。例如，路径(1, 2, 3, 8, 11)就是在发现节点11之后栈的内容（如前面所示）。

Based on the DFS-tree $T$ ,every edge $\left( {u,v}\right)  \in  E$ can be classified into one of the three categories below:

基于深度优先搜索树$T$，每条边$\left( {u,v}\right)  \in  E$可以分为以下三类之一：

- Tree edge: $u$ is the parent of $v$ in $T$ .

- 树边：在$T$中，$u$是$v$的父节点。

- Forward edge: $u$ is a proper ancestor of $v$ in $T$ .

- 前向边：在$T$中，$u$是$v$的真祖先节点。

- Cross edge: neither $u$ nor $v$ is an ancestor of the other.

- 交叉边：$u$和$v$都不是对方的祖先节点。

Example (cont.). In Figure 2 (which is reproduced in Figure $4\mathrm{\;b}$ ),edge(1,3)is a forward edge with respect to the DFS-tree of Figure $4\mathrm{a},\left( {4,8}\right) ,\left( {9,{13}}\right)$ are cross edges,while the other edges are tree edges.

示例（续）。在图2（图$4\mathrm{\;b}$中重现）中，边(1, 3)相对于图$4\mathrm{a},\left( {4,8}\right) ,\left( {9,{13}}\right)$的深度优先搜索树是前向边，而其他边是树边。

White-Path Theorem. The theorem points out a crucial property of DFS. Consider the moment when a node $u$ is just discovered (it is about to be pushed into the stack). Suppose that there is a white path - namely a path where all the vertices are white - starting from $u$ and ending at another vertex $v$ . In other words,the vertices on this path have not been discovered yet. It is guaranteed that the algorithm must be able to discover $v$ while $u$ is still in the stack.

白色路径定理。该定理指出了深度优先搜索的一个关键性质。考虑节点$u$刚被发现（即将被压入栈中）的时刻。假设存在一条白色路径——即所有顶点都是白色的路径——从$u$开始并终止于另一个顶点$v$。换句话说，这条路径上的顶点尚未被发现。可以保证，在$u$仍在栈中的时候，算法一定能够发现$v$。

Example (cont.). Figure 4b shows the color state when node 3 is discovered in our earlier execution of DFS. At this moment, node 3 has white paths to nodes8,10,11,13. Then,for sure, before node 13 turns black (i.e., while it still remains in the stack), DFS will definitely have discovered all those 4 nodes.

示例（续）。图4b展示了在我们前面执行深度优先搜索发现节点3时的颜色状态。此时，节点3到节点8、10、11、13有白色路径。那么，肯定地，在节点13变为黑色之前（即它仍在栈中时），深度优先搜索肯定会发现所有这4个节点。

The white-path theorem states the above property formally by resorting to the DFS-tree.

白色路径定理通过借助深度优先搜索树正式阐述了上述性质。

THEOREM 1 (WHITE-PATH THEOREM [1]). In the DFS-tree, a node $u$ is a proper ancestor of a node $v$ if and only if the following is true: when $u$ is discovered,there is a white path from $u$ to $v$ .

定理1（白色路径定理[1]）。在深度优先搜索树中，节点$u$是节点$v$的真祖先节点，当且仅当以下情况成立：当$u$被发现时，存在一条从$u$到$v$的白色路径。

Example (cont.). Indeed, nodes 8, 10, 11, 13 are the only proper descendants of node 3 in the DFS-tree of Figure 4a.

示例（续）。实际上，在图4a的深度优先搜索树中，节点8、10、11、13是节点3仅有的真后代节点。

## 4 ALGORITHMS FOR TREES

## 4 树的算法

Let us "warm up" by dealing with a special version of IGS. Recall that the underlying hierarchy is a DAG $G$ . In this section, we will focus on the case where $G$ is a tree. This allows us to present some of our techniques (particularly, those related to heavy-path decomposition) without the other details needed to cope with DAGs. Since we will be concerned only with a tree hierarchy,we will denote the hierarchy as $T$ (rather than $G$ ). Let $r$ be the root of $T$ . Every edge in $T$ is directed away from $r$ . As defined in Section 2.1,we denote by $d$ the maximum out-degree of a node in $T$ ,and by $h$ the length of the longest (directed) path.

让我们通过处理IGS（迭代图搜索，Iterative Graph Search）的一个特殊版本来“热热身”。回顾一下，底层层次结构是一个有向无环图（DAG，Directed Acyclic Graph）$G$。在本节中，我们将关注$G$是树的情况。这使我们能够展示我们的一些技术（特别是与重路径分解相关的技术），而无需处理有向无环图所需的其他细节。由于我们只关注树层次结构，我们将该层次结构表示为$T$（而不是$G$）。设$r$为$T$的根节点。$T$中的每条边都背离$r$。如第2.1节所定义，我们用$d$表示$T$中节点的最大出度，用$h$表示最长（有向）路径的长度。

### 4.1 The First Algorithm

### 4.1 第一种算法

In the extreme case where $T$ is a single path of length $h$ , it is trivial to find the target node $z$ by binary search in at most $\left\lceil  {{\log }_{2}h}\right\rceil$ questions. What makes binary search work is monotonicity. In general,on any directed path $\pi$ ,we always have two monotone properties:

在极端情况下，即$T$是长度为$h$的单一路径时，通过二分查找，最多用$\left\lceil  {{\log }_{2}h}\right\rceil$个问题就能轻松找到目标节点$z$。二分查找有效的原因是单调性。一般来说，在任何有向路径$\pi$上，我们总是有两个单调性质：

- If reach $\left( u\right)  = {yes}$ for a node $u$ on $\pi$ ,then reach(v)must also be yes for any node $v$ before $u$ on $\pi$ .

- 如果对于$\pi$上的节点$u$，reach $\left( u\right)  = {yes}$为真，那么对于$\pi$上$u$之前的任何节点$v$，reach(v)也一定为真。

- If reach $\left( u\right)  = {no}$ for a node $u$ on $\pi$ ,then reach(v)must also be no for any node $v$ after $u$ on $\pi$ .

- 如果对于$\pi$上的节点$u$，reach $\left( u\right)  = {no}$为假，那么对于$\pi$上$u$之后的任何节点$v$，reach(v)也一定为假。

How to exploit the monotonicity on a general tree hierarchy $T$ ? This is where heavy-tree path decomposition comes in. First,perform such a decomposition on $T$ ,and obtain a path tree $\Pi$ ,in the way introduced in Section 3.1. Then,we can carry out the search by interleaving between $T$ and $\Pi$ . The algorithm, named interleave, is formally described as follows:

如何在一般的树层次结构$T$上利用这种单调性呢？这就是重树路径分解发挥作用的地方。首先，按照第3.1节介绍的方法对$T$进行这样的分解，得到路径树$\Pi$。然后，我们可以通过在$T$和$\Pi$之间交替进行搜索。这个名为interleave的算法正式描述如下：

<!-- Media -->

algorithm interleave

算法interleave

---

1. $\pi  \leftarrow$ the root (super-node) of $\Pi / \star  \pi$ is a path in ${T}^{ * }/$

1. $\pi  \leftarrow$ $\Pi / \star  \pi$的根（超级节点）是${T}^{ * }/$中的一条路径

2. repeat

2. 重复

3. /* navigate in $\Pi$ */

3. /* 在$\Pi$中导航 */

		binary search $\pi$ to find the last node $u$

		对$\pi$进行二分查找，找到最后一个节点$u$

		with reach $\left( u\right)  =$ yes

		其reach $\left( u\right)  =$为真

4. /* navigate in ${T}^{ * }$ /

4. /* 在${T}^{ * }$中导航 */

		find a child $v$ of $u$ in $T$ with reach $\left( v\right)  =$ yes

		在$T$中找到$u$的一个子节点$v$，其reach $\left( v\right)  =$为真

		(note that $v$ cannot be in $\pi$ )

		（注意$v$不能在$\pi$中）

5. if $v$ does not exist then return $u$

5. 如果 $v$ 不存在，则返回 $u$

		else

		否则

6. $\;\pi  \leftarrow$ the (only) super-node in $\Pi$ containing $v$

6. $\;\pi  \leftarrow$ 是 $\Pi$ 中包含 $v$ 的（唯一）超级节点

			${/}^{ * }\pi$ is a path in $T$ ,and $v$ must be the first node

			${/}^{ * }\pi$ 是 $T$ 中的一条路径，并且 $v$ 必须是该路径中的第一个节点

			in this path */

			在这条路径中 */

---

<!-- Media -->

Example. To illustrate the algorithm,assume that $T$ is the tree in Figure 3a,whose decomposition tree $\Pi$ is in Figure 3b. Suppose that the adversary has secretly decided the target node $z$ to be node 9 .

示例。为了说明该算法，假设 $T$ 是图 3a 中的树，其分解树 $\Pi$ 如图 3b 所示。假设对手已秘密选定目标节点 $z$ 为节点 9。

Interleave starts by looking at the root ${\pi }_{1}$ of $\Pi$ . At Line 3,it performs binary search on ${\pi }_{1}$ to find node 2,which is the last node on ${\pi }_{1}$ that can reach $z$ . Then,the algorithm jumps to node 2 in $T$ ,and examines its child nodes 3 and 5 (child node 4 can be left out because it is in ${\pi }_{1}$ ,and hence,has already been considered in the binary search on ${\pi }_{1}$ ). After issuing a question on each node,we find that reach $\left( {\text{node }5}\right)  =$ yes; thus, $v =$ node 5 at Line 4 . At Line 6,Interleave switches back to $\Pi$ ,and identifies the path ${\pi }_{6}$ ,i.e.,the super-node in $\Pi$ covering node 5 .

Interleave 算法从查看 $\Pi$ 的根节点 ${\pi }_{1}$ 开始。在第 3 行，它对 ${\pi }_{1}$ 进行二分查找，以找到节点 2，该节点是 ${\pi }_{1}$ 上能够到达 $z$ 的最后一个节点。然后，算法跳转到 $T$ 中的节点 2，并检查其子节点 3 和 5（子节点 4 可以忽略，因为它在 ${\pi }_{1}$ 中，因此在对 ${\pi }_{1}$ 的二分查找中已经考虑过）。在对每个节点进行询问后，我们发现 reach $\left( {\text{node }5}\right)  =$ 为是；因此，在第 4 行选择节点 5。在第 6 行，Interleave 算法切换回 $\Pi$，并确定路径 ${\pi }_{6}$，即 $\Pi$ 中覆盖节点 5 的超级节点。

Continuing, Interleave (at Line 3) performs binary search on ${\pi }_{6}$ ,which finds node 9 as the last node in ${\pi }_{6}$ that can reach $z$ (note: at this moment,the algorithm still does not know that $z$ is just node 9). At Line 4,it turns back to $T$ to inspect the child node 13 of node 9 (child node 12 can be left out). As reach(node 13) $= {no}$ ,now we can conclude that $z$ is node 9 . The algorithm finishes here.

继续，Interleave 算法（在第 3 行）对 ${\pi }_{6}$ 进行二分查找，找到节点 9 作为 ${\pi }_{6}$ 中能够到达 $z$ 的最后一个节点（注意：此时，算法仍然不知道 $z$ 就是节点 9）。在第 4 行，它返回 $T$ 以检查节点 9 的子节点 13（子节点 12 可以忽略）。由于 reach(节点 13) $= {no}$，现在我们可以得出结论，$z$ 是节点 9。算法到此结束。

Next, we analyze the number of questions asked by interleave in the worst case. Call Lines 3-7 an iteration. Since $\pi$ has a length of at most $h$ ,the binary search at Line 3 requires at most $\left\lceil  {{\log }_{2}h}\right\rceil$ questions. Line 4 obviously requires no more than $d$ questions because $u$ can have at most $d$ child nodes. This caps the number of questions per each iteration at $d + \left\lceil  {{\log }_{2}h}\right\rceil$ .

接下来，我们分析 Interleave 算法在最坏情况下提出的询问次数。将第 3 - 7 行称为一次迭代。由于 $\pi$ 的长度最多为 $h$，第 3 行的二分查找最多需要 $\left\lceil  {{\log }_{2}h}\right\rceil$ 次询问。第 4 行显然最多需要 $d$ 次询问，因为 $u$ 最多可以有 $d$ 个子节点。这将每次迭代的询问次数上限设定为 $d + \left\lceil  {{\log }_{2}h}\right\rceil$。

How many iterations are needed? The crucial observation is that, every time we come to Line 4, we have descended one level of $\Pi$ . By Lemma 1, $\Pi$ has at most $1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$ levels. Hence,the number of questions is $O\left( {\log n \cdot  \log h + d \cdot  \log n}\right)$ . We do not need to be bothered by the hidden constants here because the result will be improved very shortly.

需要多少次迭代呢？关键的观察结果是，每次我们到达第 4 行时，我们在 $\Pi$ 中下降了一层。根据引理 1，$\Pi$ 最多有 $1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$ 层。因此，询问次数为 $O\left( {\log n \cdot  \log h + d \cdot  \log n}\right)$。我们在这里不需要考虑隐藏常数，因为结果很快就会得到改进。

### 4.2 Improving the Cost

### 4.2 降低成本

Next, we reduce the number of questions of interleave by making a small modification to the algorithm.

接下来，我们通过对算法进行小的修改来减少 Interleave 算法的询问次数。

At Line 4,interleave finds a child node $v$ of $u$ in $T$ with $\operatorname{reach}\left( v\right)  =$ yes. We did so with $d$ questions by querying the children of $u$ in an arbitrary order. Now,we apply a particular ordering:

在第 4 行，Interleave 算法在 $T$ 中找到 $u$ 的一个子节点 $v$，且 reach $\operatorname{reach}\left( v\right)  =$ 为是。我们通过以任意顺序查询 $u$ 的子节点，用 $d$ 次询问做到了这一点。现在，我们采用一种特定的顺序：

query the child nodes $v$ of $u -$ but ignoring the child node in $\pi  -$ in non-ascending order of the subtree size of $v$ .

按 $v$ 子树大小的非升序查询 $u -$ 的子节点 $v$，但忽略 $\pi  -$ 中的子节点。

As soon as a child node returns reach $\left( v\right)  = {yes}$ ,we stop and proceed to Line 5 . We refer to the modified algorithm as ordered-interleave.

一旦某个子节点返回 reach $\left( v\right)  = {yes}$ 为是，我们就停止并进入第 5 行。我们将修改后的算法称为有序 Interleave 算法。

Example. For an illustration, consider again the execution of interleave traced out earlier. Recall the moment after node 2 is found in ${\pi }_{1}$ through binary search. As explained before, it suffices to consider child nodes 3 and 5 of node 2 (because node 4 is in ${\pi }_{1}$ ). While interleave inspects the child nodes in an arbitrary order, ordered-interleave processes node 5 first, because it has a larger subtree than node 3 .

示例。为了说明这一点，我们再次考虑之前追踪的交错（interleave）执行过程。回顾一下，在通过二分查找在${\pi }_{1}$中找到节点2之后的时刻。如前所述，只需考虑节点2的子节点3和5即可（因为节点4在${\pi }_{1}$中）。虽然交错（interleave）以任意顺序检查子节点，但有序交错（ordered - interleave）会先处理节点5，因为它的子树比节点3的子树大。

The modification provably reduces the worst-case cost:

可以证明，这种修改降低了最坏情况下的成本：

LEMMA 2. Ordered-interleave asks at most $\left\lceil  {{\log }_{2}h}\right\rceil$ . $\left( {1 + \left\lfloor  {{\log }_{2}n}\right\rfloor  }\right)  + \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$ questions.

引理2。有序交错（ordered - interleave）最多询问$\left\lceil  {{\log }_{2}h}\right\rceil$。$\left( {1 + \left\lfloor  {{\log }_{2}n}\right\rfloor  }\right)  + \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$个问题。

Proof. See Appendix A.

证明。见附录A。

### 4.3 A Lower Bound

### 4.3 下界

We finish the section by proving a lower bound on the number of questions needed to perform IGS on a tree hierarchy.

在本节的最后，我们将证明在树层次结构上执行IGS所需问题数量的下界。

LEMMA 3. Given a tree hierarchy, any algorithm must ask at least $\left( {d - 1}\right)  \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor$ questions in the worst case.

引理3。给定一个树层次结构，任何算法在最坏情况下至少要询问$\left( {d - 1}\right)  \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor$个问题。

Proof. See Appendix B.

证明。见附录B。

The lower bound matches the upper bound in Lemma 2 up to a small additive factor.

该下界与引理2中的上界相差一个小的加法因子。

## 5 ALGORITHMS FOR DAGS

## 5 有向无环图（DAG）的算法

We are now ready to attack the IGS problem in its general form: the input hierarchy is a DAG $G = \left( {V,E}\right)$ . Our algorithm, in essence, reduces the problem to that on a special DFS-tree of $G -$ which we name the heavy-path DFS-tree - that integrates features of both DFS-tree and heavy-path decomposition.

现在我们准备解决一般形式的IGS问题：输入的层次结构是一个有向无环图（DAG）$G = \left( {V,E}\right)$。本质上，我们的算法将该问题简化为在$G -$的一个特殊深度优先搜索树（DFS - tree）上的问题，我们将其命名为重路径深度优先搜索树（heavy - path DFS - tree），它结合了深度优先搜索树（DFS - tree）和重路径分解（heavy - path decomposition）的特点。

### 5.1 The Heavy-Path DFS-Tree

### 5.1 重路径深度优先搜索树

Consider performing DFS on $G$ starting from the root. Recall that,at each step,the algorithm takes the top node $u$ of the stack,and looks for a white out-neighbor $v$ of $u$ to visit next (as shown at Line 3 of the pseudocode in Section 3.2). Normally,any out-neighbor $v$ would suffice. We,however, will insist on choosing the most "out-reaching" $v$ .

考虑从根节点开始对$G$进行深度优先搜索（DFS）。回顾一下，在每一步中，算法取出栈顶节点$u$，并寻找$u$的一个白色出邻接点$v$作为下一个要访问的节点（如第3.2节伪代码的第3行所示）。通常情况下，任何出邻接点$v$都可以。然而，我们将坚持选择最“向外延伸”的$v$。

Formally,let $S$ be the set of white out-neighbors of $u$ at this moment. For each $v \in  S$ ,we count the number - denoted as $\operatorname{count}\left( v\right)  -$ of nodes that $v$ can reach via white paths. Then,the node $v$ to visit next is the one in $S$ with the largest $\operatorname{count}\left( v\right)$ ,breaking ties arbitrarily.

形式上，设$S$是此时$u$的白色出邻接点集合。对于每个$v \in  S$，我们计算$v$通过白色路径可以到达的节点数量，记为$\operatorname{count}\left( v\right)  -$。然后，下一个要访问的节点$v$是$S$中$\operatorname{count}\left( v\right)$值最大的节点，若有多个节点值相同则任意选择。

Example. To illustrate this, let us consider Figure 4b again. Remember that, at this moment, the stack contains (from bottom to top): nodes 1 and 2 . These two nodes are gray, while the other nodes are still white. Since node 2 tops the stack, we need to decide which of its white out-neighbors should be visited next. From Figure 4b, one can see that node 3 can reach five nodes via white paths at this moment: nodes 3,8,10,11,13; hence,count(node 3) = 5 . On the other hand, count(node 4) = 6 because node 4 can reach six nodes via white paths: nodes4,6,7,8,10,11. Finally,count(node 5) is also 5 . Therefore, the node to visit next is node 4 .

示例。为了说明这一点，我们再次考虑图4b。请记住，此时栈中（从下到上）包含节点1和2。这两个节点是灰色的，而其他节点仍然是白色的。由于节点2在栈顶，我们需要决定它的哪个白色出邻接点应该被下一个访问。从图4b中可以看出，此时节点3通过白色路径可以到达五个节点：节点3、8、10、11、13；因此，count(节点3) = 5。另一方面，count(节点4) = 6，因为节点4通过白色路径可以到达六个节点：节点4、6、7、8、10、11。最后，count(节点5)也是5。因此，下一个要访问的节点是节点4。

As another illustration, Figure 5a shows the color state when node 4 is popped out of the stack. At this moment, the stack once again contains (bottom to top): nodes 1 and 2, which are in gray. Nodes 4, 6, 7, 8, 10, and 11 are currently black (they have been pushed and then popped from the stack). Which out-neighbor of node 2 to pick this time? Now, node 2 has only two white out-neighbors: nodes 3 and 5. Notice that count(node 3) has decreased to 2: node 3 can reach only itself and node 13 via white paths. Since count(node 5) = 4, next DFS visits node 5.

作为另一个示例，图5a显示了节点4从栈中弹出时的颜色状态。此时，栈中再次（从下到上）包含节点1和2，它们是灰色的。节点4、6、7、8、10和11目前是黑色的（它们已经被压入栈然后弹出）。这次应该选择节点2的哪个出邻接点呢？现在，节点2只有两个白色出邻接点：节点3和5。注意，count(节点3)已经减少到2：节点3通过白色路径只能到达它自己和节点13。由于count(节点5) = 4，下一次深度优先搜索（DFS）将访问节点5。

Define $T$ as the DFS-tree corresponding to running DFS in the way explained above. We "regulate" $T$ by arranging its nodes as follows:

将 $T$ 定义为按照上述方式运行深度优先搜索（DFS）所对应的深度优先搜索树。我们通过以下方式排列其节点来“规范” $T$：

At each internal node $u$ of $T$ ,arrange its child nodes from left to right in the order that those child nodes are discovered.

在 $T$ 的每个内部节点 $u$ 处，按照这些子节点被发现的顺序从左到右排列其子节点。

The resulting $T$ has a nice property: a pre-order traversal of $T$ enumerates the nodes in the same order as they were discovered in DFS. Example (cont.). Using the ordering strategy introduced earlier, DFS discovers the nodes in this sequence:1,2,4,8,10, 11,6,7,5,9,12,14,13,3. Figure $5\mathrm{\;b}$ shows the corresponding DFS-tree. Note that a pre-order traversal of the tree gives precisely the same node sequence.

得到的 $T$ 具有一个很好的性质：对 $T$ 进行前序遍历所枚举的节点顺序与它们在深度优先搜索中被发现的顺序相同。示例（续）。使用前面介绍的排序策略，深度优先搜索按此序列发现节点：1,2,4,8,10, 11,6,7,5,9,12,14,13,3。图 $5\mathrm{\;b}$ 展示了相应的深度优先搜索树。注意，对该树进行前序遍历恰好得到相同的节点序列。

<!-- Media -->

<!-- figureText: ⑤ ⑩ ⑫ 10 pops out node 4 -->

<img src="https://cdn.noedgeai.com/0195c91c-378f-77b0-8d6e-bb14508995e7_6.jpg?x=961&y=235&w=646&h=406&r=0"/>

Figure 5: Computing the heavy-path DFS-tree

图 5：计算重路径深度优先搜索树

<!-- Media -->

The lemma below explains why we refer to this $T$ as the heavy-path DFS-tree:

下面的引理解释了为什么我们将这个 $T$ 称为重路径深度优先搜索树：

LEMMA 4. Consider any internal node $u$ . Let ${v}_{1},{v}_{2}$ be child nodes of $u$ such that ${v}_{1}$ is on the left of ${v}_{2}$ . Then,the subtree of ${v}_{1}$ in $T$ is at least as large as that of ${v}_{2}$ .

引理 4。考虑任意内部节点 $u$。设 ${v}_{1},{v}_{2}$ 是 $u$ 的子节点，使得 ${v}_{1}$ 在 ${v}_{2}$ 的左侧。那么，在 $T$ 中 ${v}_{1}$ 的子树至少与 ${v}_{2}$ 的子树一样大。

Proof. See Appendix C.

证明。见附录 C。

Example (cont.). Observe that, in Figure 5b, the child nodes of each internal node have been automatically ordered from left to right in non-ascending order of subtree size.

示例（续）。观察到，在图 5b 中，每个内部节点的子节点已自动按照子树大小的非升序从左到右排序。

### 5.2 The Algorithm

### 5.2 算法

Our algorithm for performing IGS on a DAG - named DFS-interleave - can be formally described as:

我们在有向无环图（DAG）上执行交错图搜索（IGS）的算法——名为深度优先搜索交错（DFS - interleave）——可以正式描述为：

algorithm DFS-interleave

算法 深度优先搜索交错

/* $T$ is the heavy-path DFS-tree */

/* $T$ 是重路径深度优先搜索树 */

1. $\widehat{u} \leftarrow$ the root $r$

1. $\widehat{u} \leftarrow$ 根节点 $r$

2. repeat

2. 重复执行

${/}^{ * }$ invariant: $z$ is reachable from $\widehat{u}{}^{ * }$ /

${/}^{ * }$ 不变式：$z$ 可从 $\widehat{u}{}^{ * }$ 到达 /

3. $\;\pi  \leftarrow$ the leftmost $\widehat{u}$ -to-leaf path of $T$

3. $\;\pi  \leftarrow$ $T$ 的最左侧 $\widehat{u}$ 到叶节点的路径

4. binary search $\pi$ to find the last node $u$ with reach $\left( u\right)  =$ yes

4. 二分查找 $\pi$ 以找到可达性 $\left( u\right)  =$ 为“是”的最后一个节点 $u$

5. find the leftmost child $v$ of $u$ in $T$ with $\operatorname{reach}\left( v\right)  =$ yes

5. 在 $T$ 中找到 $u$ 的最左子节点 $v$，其可达性 $\operatorname{reach}\left( v\right)  =$ 为“是”

${\mu }^{ * }$ note that $v$ cannot be in $\pi {}^{ * }$ /

${\mu }^{ * }$ 注意，$v$ 不能在 $\pi {}^{ * }$ 中 /

6. if $v$ does not exist then return $u$

6. 如果 $v$ 不存在，则返回 $u$

7. else $\widehat{u} \leftarrow  v$

7. 否则 $\widehat{u} \leftarrow  v$

Example. We illustrate the algorithm by setting $G$ to the graph in Figure 5a,whose heavy-path DFS-tree $T$ is shown in Figure 5b. Suppose that the adversary has secretly chosen the target node $z$ to be node 9 .

示例。我们通过将 $G$ 设置为图 5a 中的图来说明该算法，其重路径深度优先搜索树 $T$ 如图 5b 所示。假设对手已秘密选择目标节点 $z$ 为节点 9。

DFS-interleave first identifies at Line 3 the leftmost root-to-leaf path of $T : \pi  = \left( {1,2,4,8,{10}}\right)$ . The algorithm performs binary search on $\pi$ to find node 2,which is the last node on $\pi$ that can reach $z$ . At Line 4,we find node 5,which is the leftmost child of node 2 that can reach $z -$ this requires only one question: since node 5 is on the left of node 3 , we test reach(node 5) first, which turns out to be yes, and thus, removes the need to test node 3 (note: node 4 does not need to be considered because it is on $\pi$ ).

深度优先搜索交错算法（DFS - interleave）首先在第 3 行识别 $T : \pi  = \left( {1,2,4,8,{10}}\right)$ 的最左根到叶路径。该算法在 $\pi$ 上进行二分查找以找到节点 2，它是 $\pi$ 上能够到达 $z$ 的最后一个节点。在第 4 行，我们找到节点 5，它是节点 2 的最左子节点且能够到达 $z -$，这只需要一个问题：由于节点 5 在节点 3 的左侧，我们首先测试可达性（节点 5），结果为“是”，因此无需测试节点 3（注意：节点 4 无需考虑，因为它在 $\pi$ 上）。

Now the execution goes back to Line 3 , where we set $\pi  = \left( {5,9,{12},{14}}\right)$ . The binary search at Line 4 finds node 9 . Then, we test reach(node 13), which is no. The algorithm terminates here by returning node 9 .

现在执行回到第 3 行，我们在那里设置 $\pi  = \left( {5,9,{12},{14}}\right)$。第 4 行的二分查找找到节点 9。然后，我们测试可达性（节点 13），结果为“否”。算法在此处返回节点 9 并终止。

The correctness proof of DFS-interleave is somewhat technical, and can be found in Appendix D.

深度优先搜索交错算法（DFS - interleave）的正确性证明有些技术性，可在附录 D 中找到。

### 5.3 Analysis

### 5.3 分析

Interestingly, the cost analysis of DFS-interleave is completely the same as that of ordered-interleave. To see why, let us first observe:

有趣的是，深度优先搜索交错算法（DFS - interleave）的成本分析与有序交错算法（ordered - interleave）的完全相同。为了明白原因，让我们首先观察：

COROLLARY 1. Consider any internal node $u$ of $T$ ,and a child node $v$ of $u$ in $T$ . The edge(u,v)is heavy if and only if $v$ is the leftmost child of $u$ .

推论 1。考虑 $T$ 的任何内部节点 $u$，以及 $T$ 中 $u$ 的子节点 $v$。当且仅当 $v$ 是 $u$ 的最左子节点时，边 (u, v) 为重边。

Proof. Immediately from Lemma 4. $\square$

证明。直接由引理 4 得出。$\square$

The next lemma may come as a pleasant surprise:

下一个引理可能会带来惊喜：

COROLLARY 2. The path $\pi$ identified at Line 3 (in any iteration) must be a heavy path in $T$ .

推论 2。在第 3 行（在任何迭代中）识别出的路径 $\pi$ 必须是 $T$ 中的重路径。

Proof. Immediately from Corollary 1 and the definition of heavy path.

证明。直接由推论 1 和重路径的定义得出。

Imagine that we perform a path-decomposition of $T$ to obtain its corresponding path tree $\Pi$ . Equipped with Corollary 2 , it is easy to verify that DFS-interleave asks exactly the same questions as running ordered-interleave on $\Pi$ . Therefore, the upper bound in Lemma 2 holds directly on DFS-interleave.

假设我们对$T$进行路径分解，以得到其对应的路径树$\Pi$。借助推论2，很容易验证深度优先搜索交错（DFS - interleave）所提出的问题与在$\Pi$上运行有序交错（ordered - interleave）时提出的问题完全相同。因此，引理2中的上界直接适用于深度优先搜索交错。

Example (cont.). Let us look at the example shown in Figure 5 one more time. One can verify that a heavy-path decomposition of the tree in Figure $5\mathrm{\;b}$ gives precisely the path tree in Figure 3b. Indeed, the running of DFS-interleave follows the same steps as running ordered-interleave on Figure 3b.

示例（续）。让我们再看一次图 5 所示的示例。可以验证，图 $5\mathrm{\;b}$ 中树的重路径分解（heavy-path decomposition）恰好得到图 3b 中的路径树。实际上，DFS 交错（DFS-interleave）的运行步骤与对图 3b 运行有序交错（ordered-interleave）的步骤相同。

Finally, the lower bound in Lemma 3 still holds on general DAG hierarchies because a tree is a DAG. With this, we have arrived at the first main result of this paper.

最后，由于树是有向无环图（DAG），引理 3 中的下界在一般的有向无环图层次结构上仍然成立。至此，我们得到了本文的第一个主要结果。

THEOREM 2. Both the following statements are true about the IGS problem:

定理 2。关于 IGS 问题，以下两个陈述均为真：

- DFS-interleave asks at most $\left\lceil  {{\log }_{2}h}\right\rceil   \cdot  \left( {1 + \left\lfloor  {{\log }_{2}n}\right\rfloor  }\right)  +$ $\left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$ questions.

- DFS 交错最多询问 $\left\lceil  {{\log }_{2}h}\right\rceil   \cdot  \left( {1 + \left\lfloor  {{\log }_{2}n}\right\rfloor  }\right)  +$ $\left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$ 个问题。

- Any algorithm must ask at least $\left( {d - 1}\right)  \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor$ questions in the worst case.

- 在最坏情况下，任何算法至少必须询问 $\left( {d - 1}\right)  \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor$ 个问题。

## 6 EXTENSIONS

## 6 扩展

In our IGS problem so far, in each question, we can ask the oracle to resolve the reachability of only one node. In practice, an algorithm may invite a human to resolve the reachability of multiple nodes at a time. Next, we show that our algorithms can be extended easily to these scenarios.

到目前为止，在我们的 IGS 问题中，在每个问题中，我们只能要求神谕（oracle）解析一个节点的可达性。在实践中，算法可能会让人类一次性解析多个节点的可达性。接下来，我们将展示我们的算法可以轻松扩展到这些场景。

The $k$ -IGS Problem. Let us start by extending the problem definition. As before,we have a DAG hierarchy $G = \left( {V,E}\right)$ with a single root; and an adversary secretly chooses a target node $z$ . An algorithm’s goal is still to find $z$ by resorting to an oracle.

$k$ -IGS 问题。让我们从扩展问题定义开始。和之前一样，我们有一个具有单个根节点的有向无环图层次结构 $G = \left( {V,E}\right)$；并且对手秘密选择一个目标节点 $z$。算法的目标仍然是通过借助神谕来找到 $z$。

The oracle,however,is $k$ -times more powerful,where $k \geq  1$ is an integer. Formally,in a $k$ -question,a query specifies a query set $Q$ of nodes ${q}_{1},{q}_{2},\ldots ,{q}_{k}$ in $V$ . The oracle returns $k$ boolean values ${b}_{1},{b}_{2},\ldots ,{b}_{k}$ ,where ${b}_{i} = 1\left( {i \in  \left\lbrack  {1,k}\right\rbrack  }\right)$ if $z$ is reachable from ${q}_{i}$ (i.e., $\operatorname{reach}\left( {q}_{i}\right)  =$ yes),and ${b}_{i} = 0$ , otherwise. Accordingly, the cost of the algorithm is defined as the number of $k$ -questions issued. We refer to this problem as $k$ -IGS,which captures IGS as a special case with $k = 1$ .

然而，神谕的能力增强了 $k$ 倍，其中 $k \geq  1$ 是一个整数。形式上，在一个 $k$ -问题中，一个查询指定了 $V$ 中节点 ${q}_{1},{q}_{2},\ldots ,{q}_{k}$ 的查询集 $Q$。神谕返回 $k$ 个布尔值 ${b}_{1},{b}_{2},\ldots ,{b}_{k}$，其中如果 $z$ 可以从 ${q}_{i}$ 到达（即，$\operatorname{reach}\left( {q}_{i}\right)  =$ 是），则 ${b}_{i} = 1\left( {i \in  \left\lbrack  {1,k}\right\rbrack  }\right)$ 为真，否则 ${b}_{i} = 0$ 为假。相应地，算法的成本定义为发出的 $k$ -问题的数量。我们将这个问题称为 $k$ -IGS，当 $k = 1$ 时，它将 IGS 作为一个特殊情况包含在内。

Top-Down. The out-neighbor property in Proposition 1 still holds. Recall that the top-down algorithm works by repeating the following step: given a node $u$ with reach $\left( u\right)  =$ yes, find an out-neighbor $v$ of $u$ with reach $\left( v\right)  =$ yes (if $v$ exists). In IGS,this step required cost $d$ ,where $d$ is the largest number of out-neighbors that $u$ may have. In $k$ -IGS,the step can be implemented using $\lceil d/k\rceil k$ -questions,where each question includes $k$ out-neighbors of $u$ in the query set. Therefore, top-down entails a cost of $\lceil d/k\rceil  \cdot  h$ ,where $h$ is the length of the longest path in $G$ .

自顶向下。命题 1 中的出邻接点属性仍然成立。回顾一下，自顶向下算法的工作方式是重复以下步骤：给定一个可达性 $\left( u\right)  =$ 为“是”的节点 $u$，找到 $u$ 的一个可达性 $\left( v\right)  =$ 为“是”的出邻接点 $v$（如果 $v$ 存在）。在 IGS 中，这一步需要成本 $d$，其中 $d$ 是 $u$ 可能拥有的最大出邻接点数量。在 $k$ -IGS 中，这一步可以使用 $\lceil d/k\rceil k$ -问题来实现，其中每个问题在查询集中包含 $u$ 的 $k$ 个出邻接点。因此，自顶向下算法的成本为 $\lceil d/k\rceil  \cdot  h$，其中 $h$ 是 $G$ 中最长路径的长度。

Ordered-Interleave. Next, we will show how the proposed algorithms can be adapted to $k$ -IGS,starting with ordered-interleave designed for tree hierarchies.

有序交错。接下来，我们将展示如何将所提出的算法应用于 $k$ -IGS，从为树层次结构设计的有序交错算法开始。

The algorithm still runs in the way described by the pseu-docode in Section 4.1, with Line 4 implemented using the ordering idea of Section 4.2. There are only two differences:

该算法仍然按照第 4.1 节中的伪代码描述的方式运行，第 4 行使用第 4.2 节中的排序思想实现。只有两个不同之处：

- At Line 3,the binary search is replaced by $k$ -ary search (which works in a fashion similar to searching for a key in a B-tree). Specifically, suppose that the path $\pi$ is the sequence of nodes ${u}_{1},{u}_{2},\ldots ,{u}_{x}$ ; the objective is to find the largest $y \in  \left\lbrack  {1,x}\right\rbrack$ such that $\operatorname{reach}\left( {u}_{y}\right)  = {yes}$ . We inquire the oracle with a query set $Q = \left\{  {{u}_{x/k},{u}_{{2x}/k},\ldots ,{u}_{x}}\right\}$ . If $i$ is the largest integer satisfying reach $\left( {u}_{i \cdot  x/k}\right)  = {yes}$ ,we know that $y$ must be $\lbrack i \cdot  x/k,\left( {i + 1}\right) x/k)$ ,which is then searched recursively. For simplicity,we have assumed $x$ to be a multiple of $k$ , because it is trivial to extend the strategy to arbitrary $x$ ,so that $y$ can be determined in $\left\lceil  {{\log }_{k}x}\right\rceil   \leq  \left\lceil  {{\log }_{k}h}\right\rceil$ $k$ -questions.

- 在第3行，二分查找被$k$ - 叉查找（其工作方式类似于在B - 树中查找键）所取代。具体来说，假设路径$\pi$是节点序列${u}_{1},{u}_{2},\ldots ,{u}_{x}$；目标是找到最大的$y \in  \left\lbrack  {1,x}\right\rbrack$，使得$\operatorname{reach}\left( {u}_{y}\right)  = {yes}$成立。我们用查询集$Q = \left\{  {{u}_{x/k},{u}_{{2x}/k},\ldots ,{u}_{x}}\right\}$向神谕（oracle）进行查询。如果$i$是满足可达性$\left( {u}_{i \cdot  x/k}\right)  = {yes}$的最大整数，我们知道$y$必定是$\lbrack i \cdot  x/k,\left( {i + 1}\right) x/k)$，然后对其进行递归查找。为简单起见，我们假设$x$是$k$的倍数，因为将该策略扩展到任意的$x$是很容易的，这样就可以通过$\left\lceil  {{\log }_{k}x}\right\rceil   \leq  \left\lceil  {{\log }_{k}h}\right\rceil$个$k$ - 查询来确定$y$。

- At Line 4,we query $k$ child nodes of $u$ in one $k$ - question.

- 在第4行，我们在一个$k$ - 查询中查询$u$的$k$个子节点。

LEMMA 5. Ordered-interleave with the above adaptation makes at most $\left( {1 + \left\lceil  {{\log }_{k}h}\right\rceil  }\right) \left( {1 + \left\lfloor  {{\log }_{2}n}\right\rfloor  }\right)  + \frac{d - 1}{k}\left\lceil  {{\log }_{d}n}\right\rceil$ $k$ -questions to solve $k$ -IGS on a tree hierarchy.

引理5. 采用上述调整后的有序交错（Ordered - interleave）方法在树层次结构上解决$k$ - IGS问题时，最多进行$\left( {1 + \left\lceil  {{\log }_{k}h}\right\rceil  }\right) \left( {1 + \left\lfloor  {{\log }_{2}n}\right\rfloor  }\right)  + \frac{d - 1}{k}\left\lceil  {{\log }_{d}n}\right\rceil$个$k$ - 查询。

Proof. See Appendix E.

证明. 见附录E。

DFS-Interleave. Consider now $k$ -IGS on a general DAG hierarchy $G$ . Section 5 proved that DFS-interleave can be regarded as running ordered-interleave on the heavy-path DFS-tree of $G$ . The same proof holds here. In other words,DFS-interleave settles $k$ -IGS with at most the cost given in Lemma 5.

深度优先搜索交错（DFS - Interleave）。现在考虑一般有向无环图（DAG）层次结构$G$上的$k$ - IGS问题。第5节证明了深度优先搜索交错可以看作是在$G$的重路径深度优先搜索树（heavy - path DFS - tree）上运行有序交错。这里同样的证明也成立。换句话说，深度优先搜索交错解决$k$ - IGS问题的成本最多为引理5中给出的成本。

In general,any lower bound $L$ on the worst-case cost of IGS immediately implies a lower bound $L/k$ on $k$ -IGS. This is because any $k$ -IGS algorithm $A$ with cost $U$ implies an IGS algorithm with cost $U \cdot  k$ ,by implementing each $k$ -question issued by $A$ with $k$ questions to the oracle of IGS. The above discussion brings us to our general result:

一般来说，IGS问题最坏情况下成本的任何下界$L$都直接意味着$k$ - IGS问题的下界$L/k$。这是因为任何成本为$U$的$k$ - IGS算法$A$都意味着存在一个成本为$U \cdot  k$的IGS算法，方法是将$A$发出的每个$k$ - 查询用向IGS神谕提出的$k$个查询来实现。上述讨论得出了我们的一般结果：

THEOREM 3. Both the following statements are true about the $k$ -IGS problem:

定理3. 关于$k$ - IGS问题，以下两个陈述均为真：

- DFS-interleave asks at most $\left( {1 + \left\lceil  {{\log }_{k}h}\right\rceil  }\right) (1 +$ $\left. \left\lfloor  {{\log }_{2}n}\right\rfloor  \right)  + \frac{d - 1}{k} \cdot  \left\lceil  {{\log }_{d}n}\right\rceil  k$ -questions.

- 深度优先搜索交错最多进行$\left( {1 + \left\lceil  {{\log }_{k}h}\right\rceil  }\right) (1 +$个$\left. \left\lfloor  {{\log }_{2}n}\right\rfloor  \right)  + \frac{d - 1}{k} \cdot  \left\lceil  {{\log }_{d}n}\right\rceil  k$ - 查询。

- Any algorithm must ask at least $\frac{d - 1}{k} \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor  k$ - questions in the worst case.

- 任何算法在最坏情况下至少要进行$\frac{d - 1}{k} \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor  k$个查询。

## 7 EXPERIMENTS

## 7 实验

### 7.1 IGS under Reliable Oracles

### 7.1 可靠神谕下的IGS问题

The experiments of this subsection were designed to study the characteristics of IGS algorithms under the algorithmic framework proposed in Section 2.1. In particular, the oracle is reliable, i.e., it does not make mistakes. Applications with such oracles are the primary beneficiaries of our solutions.

本小节的实验旨在研究第2.1节提出的算法框架下IGS算法的特性。特别地，神谕是可靠的，即它不会出错。具有此类神谕的应用是我们解决方案的主要受益者。

Data. We deployed two datasets:

数据. 我们使用了两个数据集：

- Amazon: this is a tree that represents the product hierarchy at Amazon. The tree was obtained from the file metadata.json.gz downloadable at ${jm}$ - cauley.ucsd.edu/data/amazon/links.html [7]. The file contains a record for each product sold at Amazon. The record has a field named categories, which specifies the nodes on the path from the root to the product's category. For example, here is what the path for a book on US history looks like: $\lbrack  *$ ,Books,History,Americas, United States] (where $*$ means the root). We reconstructed the product hierarchy as the trie on all these paths. The resulting tree has 29,240 nodes.

- 亚马逊（Amazon）：这是一棵代表亚马逊产品层级结构的树。该树是从可在 ${jm}$ - cauley.ucsd.edu/data/amazon/links.html [7] 下载的 metadata.json.gz 文件中获取的。该文件包含亚马逊所售每种产品的记录。记录中有一个名为“categories”（类别）的字段，它指定了从根节点到产品类别的路径上的节点。例如，一本关于美国历史的书籍的路径如下：$\lbrack  *$ ,书籍（Books）,历史（History）,美洲（Americas）,美国（United States）]（其中 $*$ 表示根节点）。我们将所有这些路径构建成字典树（trie）来重构产品层级结构。最终得到的树有 29,240 个节点。

<!-- Media -->

depth average out-degree

深度 平均出度

<table><tr><td>depth</td><td>average out-degree</td></tr><tr><td>0</td><td>8</td></tr><tr><td>1</td><td>83</td></tr><tr><td>2</td><td>3.4</td></tr><tr><td>3</td><td>2.2</td></tr><tr><td>4</td><td>1.4</td></tr><tr><td>5</td><td>0.87</td></tr><tr><td>6</td><td>0.71</td></tr><tr><td>7</td><td>0.59</td></tr><tr><td>8</td><td>0.54</td></tr><tr><td>9</td><td>0.48</td></tr><tr><td>10</td><td>0.69</td></tr><tr><td>11</td><td>0.44</td></tr><tr><td>12</td><td>0</td></tr><tr><td colspan="2">(b) ImageNet</td></tr></table>

<table><tbody><tr><td>深度</td><td>平均出度</td></tr><tr><td>0</td><td>8</td></tr><tr><td>1</td><td>83</td></tr><tr><td>2</td><td>3.4</td></tr><tr><td>3</td><td>2.2</td></tr><tr><td>4</td><td>1.4</td></tr><tr><td>5</td><td>0.87</td></tr><tr><td>6</td><td>0.71</td></tr><tr><td>7</td><td>0.59</td></tr><tr><td>8</td><td>0.54</td></tr><tr><td>9</td><td>0.48</td></tr><tr><td>10</td><td>0.69</td></tr><tr><td>11</td><td>0.44</td></tr><tr><td>12</td><td>0</td></tr><tr><td colspan="2">(b) 图像网（ImageNet）</td></tr></tbody></table>

0 84

1 11

2 4.6

3 2.4

4 0.97

5 0.33

6 0.17

0.13

0.11

(a) Amazon

(a) 亚马逊（Amazon）

Table 1: Out-degree statistics

表 1：出度统计

<!-- Media -->

- ImageNet: this is a DAG that represents the organization of a collection of images according to WordNet. The DAG was obtained from www.image-net.org/api/xml/structure_released.xml. Each synset tag in the XML document represents a node, whose id is given in the wnid attribute of the tag. The out-neighbors of the node are explicitly given inside the tag. We retained all the nodes, except the one with wnid = "fa11misc" because this node contains miscellaneous images that do not conform to WordNet. The DAG has 27,714 nodes.

- ImageNet（图像网）：这是一个有向无环图（DAG），它根据 WordNet（词网）对一组图像的组织方式进行表示。该有向无环图从 www.image-net.org/api/xml/structure_released.xml 获取。XML 文档中的每个同义词集标签代表一个节点，其 ID 在标签的 wnid 属性中给出。节点的出邻节点在标签内明确给出。我们保留了所有节点，但排除了 wnid = "fa11misc" 的节点，因为该节点包含不符合 WordNet 的杂项图像。该有向无环图有 27,714 个节点。

For each dataset, Table 1 shows the average out-degree of the nodes at each depth. Recall that, in general, the depth of a node in a DAG (with a single root) is the length of the shortest path from the root to the node (this definition applies to a tree as well). It is clear from the table that, for both datasets, nodes closer to the root tend to have more out-neighbors. Note that an average out-degree can be less than 1 at a depth where there are many leaves (a leaf in a DAG is a node with out-degree 0 ).

对于每个数据集，表 1 显示了每个深度的节点的平均出度。回想一下，一般来说，有向无环图（具有单个根节点）中节点的深度是从根节点到该节点的最短路径的长度（此定义也适用于树）。从表中可以清楚地看出，对于这两个数据集，离根节点较近的节点往往有更多的出邻节点。请注意，在有许多叶子节点的深度处，平均出度可能小于 1（有向无环图中的叶子节点是出度为 0 的节点）。

Competing Algorithms. Our objective is to evaluate the usefulness of the proposed DFS-interleave algorithm, using the top-down algorithm (see Section 3) as a benchmark. As explained in Section 5.3, on a tree, DFS-interleave degenerates into the ordered-interleave algorithm in Section 4.2. So one can think conveniently that the competition was between ordered-interleave and top-down on Amazon, but between DFS-interleave and top-down on ImageNet. We left out the interleave algorithm in Section 4.1 because it served as a stepping stone towards ordered-interleave.

竞争算法。我们的目标是评估所提出的深度优先搜索交错（DFS - interleave）算法的实用性，以自顶向下算法（见第 3 节）作为基准。如第 5.3 节所述，在树上，深度优先搜索交错算法退化为第 4.2 节中的有序交错算法。因此，可以方便地认为，在亚马逊数据集上是有序交错算法和自顶向下算法之间的竞争，而在 ImageNet 数据集上是深度优先搜索交错算法和自顶向下算法之间的竞争。我们排除了第 4.1 节中的交错算法，因为它是迈向有序交错算法的垫脚石。

A remark about top-down is in order. Recall that at each node $u$ ,the algorithm examines its out-neighbors $v$ in turn, until finding the first one with reach $\left( v\right)  = 1$ . This,however, means that the algorithm's performance is highly sensitive to how the out-neighbors are ordered. To avoid "pinning" the algorithm to any particular ordering, we adopted the implementation that the out-neighbors of $u$ were examined based on a random permutation. As such, top-down became a randomized algorithm. Every measurement reported in our experiments was averaged from 10 runs of this algorithm.

有必要对自顶向下算法做一个说明。回想一下，在每个节点 $u$ 处，该算法依次检查其出邻节点 $v$，直到找到第一个可达的节点 $\left( v\right)  = 1$。然而，这意味着该算法的性能对出邻节点的排序方式非常敏感。为了避免将算法“固定”在任何特定的排序上，我们采用的实现方式是基于随机排列来检查 $u$ 的出邻节点。因此，自顶向下算法变成了一个随机算法。我们实验中报告的每个测量值都是该算法运行 10 次的平均值。

Workload. Recall that,in IGS or $k$ -IGS,an adversary specifies a target node. Different target nodes define different instances of the problem. We considered all the possible instances defined by every single leaf in the underlying hierarchy. All these instances together constituted a workload. The workloads on Amazon and ImageNet had 24,329 and 21,427 instances, respectively (these are the number of leaves in each dataset).

工作负载。回想一下，在 IGS 或 $k$ - IGS 中，对手指定一个目标节点。不同的目标节点定义了问题的不同实例。我们考虑了底层层次结构中每个叶子节点定义的所有可能实例。所有这些实例共同构成了一个工作负载。亚马逊和 ImageNet 上的工作负载分别有 24,329 个和 21,427 个实例（这些是每个数据集中叶子节点的数量）。

Metrics. The primary metric for assessing an algorithm was its cost,i.e.,number of questions or $k$ -questions issued.

指标。评估算法的主要指标是其成本，即发出的问题或 $k$ - 问题的数量。

We also used another metric - candidate set size (CSS) - to measure an algorithm's progressiveness. Specifically, at any moment during an algorithm's execution, the CSS is the number of leaf nodes that the algorithm still cannot rule out (i.e., every such a leaf could still be the target node). For sure, the CSS monotonically decreases as the algorithm runs; and the algorithm cannot stop until the CSS has dropped to 1. Ideally, we would like the algorithm to reduce CSS substantially with just a few questions $k$ -questions. Indeed, in practice, one may even choose to terminate an algorithm manually once the CSS has become sufficiently small.

我们还使用了另一个指标——候选集大小（CSS）——来衡量算法的渐进性。具体来说，在算法执行的任何时刻，CSS 是算法仍然无法排除的叶子节点的数量（即每个这样的叶子节点仍然可能是目标节点）。当然，随着算法的运行，CSS 单调递减；并且在 CSS 降至 1 之前，算法不能停止。理想情况下，我们希望算法只需几个问题 $k$ - 问题就能大幅减小 CSS。实际上，一旦 CSS 变得足够小，甚至可以手动终止算法。

Machine and Coding. In all the experiments, CPU computation was carried out on a machine equipped with an Intel Core i7-4870HQ CPU at 2.5GHz, and 16 GB of memory. All our implementations were programed in Python.

机器和编码。在所有实验中，CPU 计算在配备 2.5GHz Intel Core i7 - 4870HQ CPU 和 16GB 内存的机器上进行。我们所有的实现都用 Python 编程。

Results on IGS. Let us start with Amazon. The first experiment aims to evaluate the efficiency of ordered-interleave and top-down when the target node was placed at various depths. For this purpose, we used each algorithm to run a workload. For each depth value $d$ ,we calculated the average cost of the algorithm on all the instances defined by the leaves of depth $d$ . The results are presented in Figure 6a.

IGS 上的结果。让我们从亚马逊数据集开始。第一个实验旨在评估当目标节点位于不同深度时，有序交错算法和自顶向下算法的效率。为此，我们使用每个算法运行一个工作负载。对于每个深度值 $d$，我们计算算法在由深度为 $d$ 的叶子节点定义的所有实例上的平均成本。结果如图 6a 所示。

When $d = 1$ (namely the target node is directly below the root), top-down was better because in this case the binary searches performed by ordered-interleave offer little help, and thus, do not pay off. However, ordered-interleave started to outperform top-down as soon as $d$ increased to 2 ; and the gap between the two algorithms was fairly significant for all the other depth values. In general, the binary searches of ordered-interleave are more effective when the target node lies deeper in the tree - because a single binary search can skip multiple levels, which would need to be "plowed through" by top-down.

当 $d = 1$（即目标节点直接位于根节点下方）时，自上而下（top-down）算法表现更好，因为在这种情况下，有序交错（ordered-interleave）算法执行的二分查找帮助不大，因此无法体现出优势。然而，一旦 $d$ 增加到 2，有序交错算法就开始优于自上而下算法；并且对于所有其他深度值，这两种算法之间的差距相当显著。一般来说，当目标节点位于树的更深层时，有序交错算法的二分查找更有效——因为一次二分查找可以跳过多个层级，而自上而下算法则需要“逐层遍历”这些层级。

It is worth pointing out that the cost of top-down does not need to grow with the depth - note the "surge" in its cost at $d = 3$ and the "dip" at $d = 4$ . In general,this algorithm is sensitive to how many children are owned by the nodes on the path from the root to the target leaf (we will delve into this issue later in Figure 6c). Indeed, in Amazon, many depth-3 leaves gather under large-fanout ancestors that do not have leaves of depth 4 or more. This is the reason behind the aforementioned surge and dip.

值得指出的是，自上而下算法的成本并不一定随深度增加而增长——注意其成本在 $d = 3$ 处的“激增”和在 $d = 4$ 处的“骤降”。一般来说，该算法对从根节点到目标叶节点路径上的节点所拥有的子节点数量很敏感（我们将在图 6c 中深入探讨这个问题）。实际上，在亚马逊的数据结构中，许多深度为 3 的叶节点聚集在扇出较大的祖先节点下，而这些祖先节点没有深度为 4 或更深的叶节点。这就是上述成本激增和骤降的原因。

To demonstrate the progressiveness of each algorithm, we designed an experiment as follows. For each instance in a workload, we generated an array CSS that had (conceptually) an infinite length,such that $\operatorname{CSS}\left\lbrack  i\right\rbrack  \left( {i \geq  1}\right)$ was set to the CSS at the moment right after the algorithm had entailed a cost of $i$ . After the algorithm had terminated at some cost - say $c$ - we set ${CSS}\left\lbrack  i\right\rbrack   = 1$ for every $i \geq  c$ . In this way, ${CSS}\left\lbrack  i\right\rbrack$ bore an intuitive meaning: a cost budget of $i \geq  1$ guaranteed a CSS equal to $\operatorname{CSS}\left\lbrack  i\right\rbrack$ . For the workload as a whole,we calculated an array $\overline{CSS}$ to average out the CSS-arrays of all the instances; that is,for each $i \geq  1,\overline{CSS}\left\lbrack  i\right\rbrack$ was the average ${CSS}\left\lbrack  i\right\rbrack$ of all the instances in the workload.

为了展示每种算法的渐进性，我们设计了如下实验。对于工作负载中的每个实例，我们生成一个（概念上）长度无限的数组 CSS，使得 $\operatorname{CSS}\left\lbrack  i\right\rbrack  \left( {i \geq  1}\right)$ 被设置为算法产生成本 $i$ 之后那一刻的 CSS 值。在算法以某个成本（例如 $c$）终止后，我们为每个 $i \geq  c$ 设置 ${CSS}\left\lbrack  i\right\rbrack   = 1$。通过这种方式，${CSS}\left\lbrack  i\right\rbrack$ 具有直观的含义：成本预算为 $i \geq  1$ 可保证 CSS 等于 $\operatorname{CSS}\left\lbrack  i\right\rbrack$。对于整个工作负载，我们计算一个数组 $\overline{CSS}$ 来对所有实例的 CSS 数组求平均值；也就是说，对于每个 $i \geq  1,\overline{CSS}\left\lbrack  i\right\rbrack$，它是工作负载中所有实例的 ${CSS}\left\lbrack  i\right\rbrack$ 的平均值。

Figure 6b plots the $\overline{CSS}$ array for ordered-interleave and top-down. Note that the y-axis is in log scale. It is evident that ordered-interleave was significantly faster in reducing CSS. In particular, the average CSS was below 10 in less than 40 questions, while at this cost top-down still had an average CSS close to 10,000 .

图 6b 绘制了有序交错算法和自上而下算法的 $\overline{CSS}$ 数组。请注意，y 轴采用对数刻度。显然，有序交错算法在降低 CSS 方面明显更快。具体而言，在不到 40 个问题的情况下，平均 CSS 低于 10，而在这个成本下，自上而下算法的平均 CSS 仍接近 10000。

The next experiment aims to provide a "zoom-in" into the cost of an algorithm on individual instances in a workload. Towards the purpose, let us define the sum of out-degrees of ancestors (SODA) of a node $u$ as the total out-degree of all the proper ancestors of $u$ . For Amazon,the SODA values of all the leaves fell in the range $\left\lbrack  {{84},{399}}\right\rbrack$ . We cut the range into 20 intervals of the same length. For each algorithm, we measured 20 costs,one for each interval $I$ . Specifically,the measurement on $I$ was the algorithm’s average cost on all the instances that were defined by the leaves with SODA values in $I$ . By putting these 20 averages together,we acquired a cost distribution of the algorithm over the SODA spectrum.

下一个实验旨在详细分析工作负载中单个实例上算法的成本。为此，我们将节点 $u$ 的祖先节点出度之和（SODA）定义为 $u$ 所有真祖先节点的总出度。对于亚马逊的数据结构，所有叶节点的 SODA 值落在范围 $\left\lbrack  {{84},{399}}\right\rbrack$ 内。我们将该范围划分为 20 个等长的区间。对于每种算法，我们测量 20 个成本，每个区间 $I$ 对应一个成本。具体来说，对 $I$ 的测量是算法在所有由 SODA 值在 $I$ 范围内的叶节点定义的实例上的平均成本。通过将这 20 个平均值汇总，我们得到了算法在 SODA 范围内的成本分布。

Figure 6c compares the obtained cost distributions of ordered-interleave and of top-down. The former algorithm consistently outperformed the latter in all intervals, often by large factors. Note that there were no results for the SODA range from 212 to 308, because no leaves have SODA values in that range. Also, observe that the cost of top-down exhibited a clear ascending trend as the SODA value increased.

图 6c 比较了有序交错算法和自上而下算法所得到的成本分布。在前述的所有区间中，前者算法始终优于后者，且差距往往很大。请注意，SODA 范围从 212 到 308 没有结果，因为没有叶节点的 SODA 值在该范围内。此外，观察到自上而下算法的成本随着 SODA 值的增加呈现出明显的上升趋势。

We repeated the same experiment on ImageNet, using DFS-interleave and top-down as the competitors. Figure 7 presents the results, which were obtained in the same manner as those of Figure 6. A bit extra explanation is needed regarding the SODA of a node $u$ in a DAG. Let us define a proper ancestor of $u$ as a node $v$ that has a path reaching $u$ . With this notion, SODA becomes well defined also for a DAG, thus allowing us to generate Figure $7\mathrm{c}$ in the way Figure $6\mathrm{c}$ was produced.

我们在ImageNet（图像网）上重复了相同的实验，使用深度优先搜索交错法（DFS - interleave）和自顶向下法（top - down）作为对比方法。图7展示了实验结果，其获取方式与图6相同。关于有向无环图（DAG）中节点$u$的祖先节点度总和（SODA），需要额外解释一下。我们将$u$的合适祖先节点定义为有路径可达$u$的节点$v$。有了这个概念，祖先节点度总和（SODA）对于有向无环图（DAG）也有了明确的定义，这样我们就可以按照生成图$6\mathrm{c}$的方式生成图$7\mathrm{c}$。

<!-- Media -->

<!-- figureText: number of questions 120 ordered-interleave ${10}^{5}$ ordered-interleave top-down average CSS ${10}^{4}$ ${10}^{3}$ ${10}^{2}$ 10 20 40 60 80 100 120 140 160 180 240 number of questions (b) Progressiveness (c) Cost vs. sum of out-degrees of ancestors Figure 6: IGS on Amazon (i.e., $k = 1$ ) ${10}^{5}$ DFS-interleave top-down average CSS ${10}^{4}$ ${10}^{3}$ ${10}^{2}$ 10 30 60 90 120 1 180 210 240 number of questions (b) Progressiveness (398,435 (472,509 (509,546) (546,583) (583,620) (620,657 (657,694 (694,731 sum of out-degrees of ancestors (c) Cost vs. sum of out-degrees of ancestors top-down 100 80 60 40 20 node depth (a) Cost vs. node depth 200 ordered-interleave top-down number of questions 150 100 50 0 number of questions 200 DFS-interleave 180 top-down 160 140 120 100 80 60 40 20 2 3 7 8 9 10 11 12 node depth (a) Cost vs. node depth DFS-interleave number of questions top-down 200 150 100 0 [28,65] (65,102] (102,139) (139,176 (176,213] (213,250 (250,287) (287,324 (324,36 -->

<img src="https://cdn.noedgeai.com/0195c91c-378f-77b0-8d6e-bb14508995e7_10.jpg?x=262&y=280&w=1255&h=1394&r=0"/>

Figure 7: IGS on ImageNet (i.e., $k = 1$ )

图7：ImageNet（图像网）上的IGS（即$k = 1$）

<!-- Media -->

Two comments are worth noting. First, in Figure 7a, there were no results at depth 1 , because ImageNet has no leaves at this depth. Second, the cost of top-down initially increased with SODA, but the trend of increasing disappeared after SODA had become large enough. This can be explained by the fact that, in a DAG, there can be multiple paths from the root to the target leaf; and a large SODA can be caused by an abundance of such paths, which actually makes it more likely for top-down to find a relatively short way to get to the target. Other than the above, the overall observations are similar to those on Amazon. Notice that the performance advantages of our solution were even more prominent on ImageNet.

有两点值得注意。首先，在图7a中，深度为1时没有结果，因为ImageNet（图像网）在这个深度没有叶子节点。其次，自顶向下法（top - down）的成本最初随祖先节点度总和（SODA）的增加而增加，但当祖先节点度总和（SODA）变得足够大后，这种增加趋势就消失了。这可以解释为，在有向无环图（DAG）中，从根节点到目标叶子节点可能有多条路径；而较大的祖先节点度总和（SODA）可能是由大量这样的路径导致的，这实际上使自顶向下法（top - down）更有可能找到一条相对较短的路径到达目标。除此之外，总体观察结果与在亚马逊数据集上的结果相似。值得注意的是，我们的解决方案在ImageNet（图像网）上的性能优势更加明显。

A final remark concerns the CPU efficiency. The most computation-intensive step is the preparation of the heavy-path DFS-tree in Section 5.1. But even this step took no more than 10 seconds on both datasets. The CPU delays in the other steps were all unnoticeable. The same was true in all the other experiments to be reported in this paper.

最后一点关于CPU效率。计算量最大的步骤是第5.1节中重路径深度优先搜索树（heavy - path DFS - tree）的准备工作。但即使是这一步，在两个数据集上花费的时间都不超过10秒。其他步骤的CPU延迟都可以忽略不计。本文后续报告的所有其他实验也是如此。

Results on $k$ -IGS. The experiment results on $k$ -IGS were in general similar to those of the experiments presented earlier (for $k = 1$ ),and can be found in Appendix F.

$k$ - IGS的实验结果。$k$ - IGS的实验结果总体上与之前（针对$k = 1$）的实验结果相似，具体结果可在附录F中找到。

### 7.2 IGS on Crowdsourcing

### 7.2 众包场景下的IGS

The results in Section 7.1 are representative of what one would expect in applications (such as those in Section 2.3) where the oracle is reliable. In this subsection, we will inspect the usefulness of DFS-interleave in crowdsourcing scenarios, where questions are answered by human workers that could err, thus generating "noise" that may prevent an algorithm from returning a correct answer. As our algorithmic framework in Section 2.1 does not explicitly take mistakes into account, DFS-interleave is not tailored for crowdsourcing. Nevertheless, the subsequent evaluation aims to make three points. First, DFS-interleave was resilient to random noise (i.e., mistakes due to carelessness) such that it achieved good accuracy even in its current form. Second, the main difficulty in crowdsourcing seemed to stem from the system noise caused by humans' lack of knowledge about the objects of concern. Third, top-down was much more susceptible to noise simply because it needed to issue more questions.

7.1节中的结果代表了在神谕（oracle）可靠的应用场景（如2.3节中的场景）中人们的预期结果。在本小节中，我们将考察深度优先搜索交错法（DFS - interleave）在众包场景中的实用性。在众包场景中，问题由可能出错的人类工作者回答，从而产生“噪声”，这可能会阻碍算法返回正确答案。由于我们在2.1节中的算法框架没有明确考虑错误情况，深度优先搜索交错法（DFS - interleave）并非专门为众包场景设计。尽管如此，后续评估旨在说明三点。首先，深度优先搜索交错法（DFS - interleave）对随机噪声（即由于粗心导致的错误）具有鲁棒性，即使以当前形式也能达到较好的准确率。其次，众包中的主要困难似乎源于人类对相关对象缺乏了解而导致的系统噪声。第三，自顶向下法（top - down）更容易受到噪声影响，仅仅是因为它需要提出更多问题。

Data. We again took the Amazon dataset, but in its DAG form. Recall the tree hierarchy generated in Section 7.1, i.e., the product category tree at Amazon. We observed that, to some products $p$ ,the source file metadata.json.gz attached multiple categories cat, each corresponding to a leaf node in the category tree. By inserting $p$ as a new leaf and adding an edge from node cat to $p$ ,we converted the tree hierarchy into a DAG.

数据。我们再次使用亚马逊数据集，但采用其有向无环图（DAG）形式。回顾7.1节中生成的树层次结构，即亚马逊的产品类别树。我们观察到，对于某些产品$p$，源文件metadata.json.gz附加了多个类别cat，每个类别对应类别树中的一个叶子节点。通过将$p$作为新的叶子节点插入，并从节点cat添加一条边到$p$，我们将树层次结构转换为有向无环图（DAG）。

Two issues, however, arose. First, some products had missing values in the "title" field, and therefore, could not be posted as informative queries on a crowdsourcing platform (this will be further clarified later). Second, the file contained over 9 million products, such that some nodes in the DAG ended up with unrealistically huge out-degrees. We remedied the issues as follows. We cleansed the dataset by discarding the products with empty title fields. This still left over five million products such that the second issue still existed. We sorted those products by id, and then picked 25,000 products evenly from the sorted list (i.e.,picking the $i \cdot  \lfloor x/{25000}\rfloor$ -th for each $i \in  \left\lbrack  {1,{25000}}\right\rbrack$ ,where $x$ was the number of products after cleansing). Then, we created a DAG in the way described earlier,by ranging $p$ over the 25,000 products,and removed "dead" category nodes with no product descendants. This yielded a DAG hierarchy with 33,573 nodes in total. Crowd. The DAG thus generated allowed us to evaluate the two competing IGS algorithms (i.e., DFS-interleave and top-down) on the commercial crowdsourcing platform of figure-eight.com, in a scenario where one would like to leverage the crowd to automatically assign pictures to products. Specifically,given a product $z$ (e.g.,a US history book),we defined its metainfo as the combination of (i) a picture ${}^{2}$ of $z$ , (ii) the title of $z$ ,and (iii) the detailed description of $z$ (if such description existed in the source file). In a question posted on figure-eight.com,we provided the metainfo of $z$ ,and asked: "does this product belong to the following category?" Each category in the question is a node in the DAG (e.g., [*, Books, Comics],in which case the correct answer for " $z = \mathrm{a}$ history book" should be no). Interestingly, $z$ itself offered the ground truth such that we could directly compare $z$ to the output of the algorithm to see if it was correct.

然而，出现了两个问题。首先，一些产品的“标题”字段存在缺失值，因此无法作为信息丰富的查询发布到众包平台上（这一点将在后面进一步说明）。其次，该文件包含超过900万种产品，导致有向无环图（DAG）中的一些节点最终具有不切实际的巨大出度。我们按以下方式解决了这些问题。我们通过丢弃标题字段为空的产品来清理数据集。即便如此，仍剩下超过500万种产品，因此第二个问题仍然存在。我们按ID对这些产品进行排序，然后从排序后的列表中均匀选取25,000种产品（即，对于每个$i \in  \left\lbrack  {1,{25000}}\right\rbrack$选取第$i \cdot  \lfloor x/{25000}\rfloor$个产品，其中$x$是清理后产品的数量）。然后，我们按照前面描述的方式创建了一个有向无环图，让$p$遍历这25,000种产品，并移除没有产品后代的“死”类别节点。这样得到了一个总共有33,573个节点的有向无环图层次结构。众包。这样生成的有向无环图使我们能够在figure-eight.com的商业众包平台上评估两种相互竞争的IGS算法（即深度优先搜索交错算法和自顶向下算法），场景是有人希望利用众包将图片自动分配给产品。具体来说，给定一个产品$z$（例如，一本美国历史书），我们将其元信息定义为以下内容的组合：（i）$z$的一张图片${}^{2}$；（ii）$z$的标题；（iii）$z$的详细描述（如果源文件中存在此类描述）。在发布到figure-eight.com的一个问题中，我们提供了$z$的元信息，并询问：“该产品是否属于以下类别？”问题中的每个类别都是有向无环图中的一个节点（例如，[*, 书籍, 漫画]，在这种情况下，“$z = \mathrm{a}$历史书”的正确答案应该是否）。有趣的是，$z$本身提供了真实情况，这样我们就可以直接将$z$与算法的输出进行比较，以查看其是否正确。

Two standard measures were taken to ensure good quality for the answers collected. First, every human worker had to pass a so-called gold standard test where s/he was given a list of questions and must correctly resolve ${85}\%$ to be qualified. Second, for the same question, confidence-guided repeats and majority-taking were applied: more answers were solicited until either a maximum of 9 answers had been returned or a confidence at least 0.7 had been reached from the majority of at least 5 answers ${}^{3}$ .

采取了两项标准措施来确保所收集答案的质量良好。首先，每个人工工作者必须通过所谓的黄金标准测试，在测试中会给他们一份问题列表，他们必须正确解答${85}\%$个问题才能合格。其次，对于同一个问题，采用了置信度引导的重复回答和多数表决方法：持续征集更多答案，直到最多返回9个答案，或者从至少5个答案的多数中达到至少0.7的置信度${}^{3}$。

Workload. As in Section 7.1, each leaf (a.k.a. product) in the DAG defines an instance of IGS. We generated a workload of 50 instances scattered evenly in the spectrum of SODA values defined in Section 7.1. Specifically, the leaves of the DAG had SODA values in the range $\left\lbrack  {{40},{2255}}\right\rbrack$ . We partitioned the range into 10 equi-length intervals. For each interval, we sorted the the leaves covered in that interval by SODA value, and picked 5 leaves evenly from the sorted list. This gave 50 leaves, a.k.a. instances, in total, which constituted the workload.

工作量。如第7.1节所述，有向无环图中的每个叶子节点（也就是产品）定义了一个IGS实例。我们生成了一个包含50个实例的工作量，这些实例均匀分布在第7.1节定义的SODA值范围内。具体来说，有向无环图的叶子节点的SODA值范围是$\left\lbrack  {{40},{2255}}\right\rbrack$。我们将该范围划分为10个等长的区间。对于每个区间，我们按SODA值对该区间内的叶子节点进行排序，并从排序后的列表中均匀选取5个叶子节点。这样总共得到50个叶子节点，也就是实例，它们构成了工作量。

Metrics. For each instance in the workload, we compared top-down and DFS-interleave using two metrics: (i) whether the algorithm correctly solved the instance, and (ii) if so, what was the crowd cost, defined as the total number of answers collected throughout the algorithm (if the same question received $x$ answers,the crowd cost increased by $x$ ). For fairness,both algorithms were executed on the same DAG, where the out-neighbors of each node were ordered randomly.

指标。对于工作量中的每个实例，我们使用两个指标来比较自顶向下算法和深度优先搜索交错算法：（i）算法是否正确解决了该实例；（ii）如果是，众包成本是多少，众包成本定义为整个算法过程中收集的答案总数（如果同一个问题收到$x$个答案，众包成本就增加$x$）。为了公平起见，两种算法都在同一个有向无环图上执行，其中每个节点的出邻接点是随机排序的。

Results. Table 2 details the performance of top-down and DFS-interleave on each of the 50 instances in the workload. Recall that five instances were created from each of the 10 intervals that partition the SODA spectrum. Those 10 intervals are listed in the second column of the table. The instances from the same interval form a group. Columns 3-5 concern only top-down. Specifically, for each instance, Column 3 gives the crowd cost of top-down, but only if the algorithm managed to resolve the instance; an incorrect output of the algorithm is indicated by the sign "-". For each group,(i) the percentage in Column 4 is calculated as $x/5$ , where $x$ is the number of instances in the group that were correctly resolved by top-down, while (ii) the number in Column 5 is the average crowd cost of the algorithm on those $x$ instances. Columns 6-8 depict DFS-interleave in the same manner.

结果。表2详细列出了自顶向下（top-down）算法和深度优先搜索交错（DFS-interleave）算法在工作负载中的50个实例上的性能。请记住，SODA频谱被划分为10个区间，每个区间生成了5个实例。这10个区间列在表的第二列。来自同一区间的实例构成一个组。第3 - 5列仅涉及自顶向下算法。具体而言，对于每个实例，第3列给出了自顶向下算法的众包成本，但仅当该算法成功解决该实例时才给出；算法输出错误用符号“ - ”表示。对于每个组，（i）第4列的百分比计算方式为$x/5$ ，其中$x$ 是自顶向下算法正确解决的该组实例数量，（ii）第5列的数字是该算法在这$x$ 个实例上的平均众包成本。第6 - 8列以相同方式描述了深度优先搜索交错算法。

---

<!-- Footnote -->

${}^{2}$ The source file included a picture URL for each of the 25,000 products.

${}^{2}$ 源文件为25000种产品中的每一种都包含了一个图片URL。

${}^{3}$ Such a functionality was directly available at figure-eight.com.

${}^{3}$ 这种功能在figure-eight.com网站上可直接使用。

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td rowspan="2">id</td><td rowspan="2">SODA range</td><td colspan="3">top-down</td><td colspan="3">DFS-interleave</td></tr><tr><td>crowd cost</td><td>success rate</td><td>avg crowd cost on successful instances</td><td>crowd cost</td><td>success rate</td><td>avg crowd cost on successful instances</td></tr><tr><td rowspan="5">1 2 3 4 5</td><td rowspan="5">$\left\lbrack  {{40},{261}}\right\rbrack$</td><td>-</td><td rowspan="5">40%</td><td rowspan="5">707</td><td>-</td><td rowspan="5">40%</td><td rowspan="5">131</td></tr><tr><td>-</td><td>-</td></tr><tr><td>-</td><td>188</td></tr><tr><td>390</td><td>-</td></tr><tr><td>1023</td><td>74</td></tr><tr><td rowspan="5">6 7 8 9 10</td><td rowspan="5">$\left\lbrack  {{262},{483}}\right\rbrack$</td><td>509</td><td rowspan="5">60%</td><td rowspan="5">991</td><td>90</td><td rowspan="5">80%</td><td rowspan="5">410</td></tr><tr><td>1061</td><td>185</td></tr><tr><td>-</td><td>-</td></tr><tr><td>1403</td><td>1276</td></tr><tr><td>-</td><td>90</td></tr><tr><td rowspan="5">11 12 13 14 15</td><td rowspan="5">$\left\lbrack  {{484},{705}}\right\rbrack$</td><td>-</td><td rowspan="5">0%</td><td rowspan="5">-</td><td>457</td><td rowspan="5">80%</td><td rowspan="5">477</td></tr><tr><td>-</td><td>885</td></tr><tr><td>-</td><td>480</td></tr><tr><td>-</td><td>85</td></tr><tr><td>-</td><td>-</td></tr><tr><td rowspan="5">16 17 18 19 20</td><td rowspan="5">$\left\lbrack  {{706},{927}}\right\rbrack$</td><td>-</td><td rowspan="5">20%</td><td rowspan="5">674</td><td>830</td><td rowspan="5">60%</td><td rowspan="5">374</td></tr><tr><td>-</td><td>69</td></tr><tr><td>-</td><td>-</td></tr><tr><td>-</td><td>225</td></tr><tr><td>674</td><td>-</td></tr><tr><td rowspan="5">21 22 23 24 25</td><td rowspan="5">[928, 1149]</td><td>-</td><td rowspan="5">0%</td><td rowspan="5">-</td><td>4248</td><td rowspan="5">100%</td><td rowspan="5">2410</td></tr><tr><td>-</td><td>1484</td></tr><tr><td>-</td><td>4606</td></tr><tr><td>-</td><td>1350</td></tr><tr><td>-</td><td>362</td></tr><tr><td>26</td><td rowspan="5">$\left\lbrack  {{1150},{1371}}\right\rbrack$</td><td>-</td><td rowspan="5">0%</td><td rowspan="5">-</td><td>346</td><td rowspan="5">100%</td><td rowspan="5">762</td></tr><tr><td>27</td><td>-</td><td>545</td></tr><tr><td>28</td><td>-</td><td>806</td></tr><tr><td>29</td><td>-</td><td>1769</td></tr><tr><td>30</td><td>-</td><td>344</td></tr><tr><td rowspan="5">31 32 33 34 35</td><td rowspan="5">$\left\lbrack  {{1372},{1593}}\right\rbrack$</td><td>729</td><td rowspan="5">80%</td><td rowspan="5">701</td><td>-</td><td rowspan="5">60%</td><td rowspan="5">161</td></tr><tr><td>514</td><td>165</td></tr><tr><td>-</td><td>-</td></tr><tr><td>1021</td><td>128</td></tr><tr><td>537</td><td>189</td></tr><tr><td rowspan="5">36 37 38 39 40</td><td rowspan="5">$\left\lbrack  {{1594},{1815}}\right\rbrack$</td><td>1296</td><td rowspan="5">40%</td><td rowspan="5">864</td><td>409</td><td rowspan="5">80%</td><td rowspan="5">425</td></tr><tr><td>-</td><td>520</td></tr><tr><td>-</td><td>680</td></tr><tr><td>432</td><td>90</td></tr><tr><td>-</td><td>-</td></tr><tr><td rowspan="5">41 42 43 44 45</td><td rowspan="5">$\left\lbrack  {{1816},{2037}}\right\rbrack$</td><td>1122</td><td rowspan="5">60%</td><td rowspan="5">1527</td><td>187</td><td rowspan="5">80%</td><td rowspan="5">520</td></tr><tr><td>1712</td><td>790</td></tr><tr><td>-</td><td>-</td></tr><tr><td>-</td><td>258</td></tr><tr><td>1748</td><td>845</td></tr><tr><td rowspan="5">46 47 48 49 50</td><td rowspan="5">$\left\lbrack  {{2038},{2259}}\right\rbrack$</td><td>2002</td><td rowspan="5">60%</td><td rowspan="5">1649</td><td>1095</td><td rowspan="5">80%</td><td rowspan="5">915</td></tr><tr><td>-</td><td>1355</td></tr><tr><td>-</td><td>745</td></tr><tr><td>1566</td><td>-</td></tr><tr><td>1378</td><td>465</td></tr></table>

<table><tbody><tr><td rowspan="2">编号</td><td rowspan="2">苏打范围（SODA range）</td><td colspan="3">自顶向下</td><td colspan="3">深度优先搜索交错（DFS-interleave）</td></tr><tr><td>群体成本</td><td>成功率</td><td>成功实例的平均群体成本</td><td>群体成本</td><td>成功率</td><td>成功实例的平均群体成本</td></tr><tr><td rowspan="5">1 2 3 4 5</td><td rowspan="5">$\left\lbrack  {{40},{261}}\right\rbrack$</td><td>-</td><td rowspan="5">40%</td><td rowspan="5">707</td><td>-</td><td rowspan="5">40%</td><td rowspan="5">131</td></tr><tr><td>-</td><td>-</td></tr><tr><td>-</td><td>188</td></tr><tr><td>390</td><td>-</td></tr><tr><td>1023</td><td>74</td></tr><tr><td rowspan="5">6 7 8 9 10</td><td rowspan="5">$\left\lbrack  {{262},{483}}\right\rbrack$</td><td>509</td><td rowspan="5">60%</td><td rowspan="5">991</td><td>90</td><td rowspan="5">80%</td><td rowspan="5">410</td></tr><tr><td>1061</td><td>185</td></tr><tr><td>-</td><td>-</td></tr><tr><td>1403</td><td>1276</td></tr><tr><td>-</td><td>90</td></tr><tr><td rowspan="5">11 12 13 14 15</td><td rowspan="5">$\left\lbrack  {{484},{705}}\right\rbrack$</td><td>-</td><td rowspan="5">0%</td><td rowspan="5">-</td><td>457</td><td rowspan="5">80%</td><td rowspan="5">477</td></tr><tr><td>-</td><td>885</td></tr><tr><td>-</td><td>480</td></tr><tr><td>-</td><td>85</td></tr><tr><td>-</td><td>-</td></tr><tr><td rowspan="5">16 17 18 19 20</td><td rowspan="5">$\left\lbrack  {{706},{927}}\right\rbrack$</td><td>-</td><td rowspan="5">20%</td><td rowspan="5">674</td><td>830</td><td rowspan="5">60%</td><td rowspan="5">374</td></tr><tr><td>-</td><td>69</td></tr><tr><td>-</td><td>-</td></tr><tr><td>-</td><td>225</td></tr><tr><td>674</td><td>-</td></tr><tr><td rowspan="5">21 22 23 24 25</td><td rowspan="5">[928, 1149]</td><td>-</td><td rowspan="5">0%</td><td rowspan="5">-</td><td>4248</td><td rowspan="5">100%</td><td rowspan="5">2410</td></tr><tr><td>-</td><td>1484</td></tr><tr><td>-</td><td>4606</td></tr><tr><td>-</td><td>1350</td></tr><tr><td>-</td><td>362</td></tr><tr><td>26</td><td rowspan="5">$\left\lbrack  {{1150},{1371}}\right\rbrack$</td><td>-</td><td rowspan="5">0%</td><td rowspan="5">-</td><td>346</td><td rowspan="5">100%</td><td rowspan="5">762</td></tr><tr><td>27</td><td>-</td><td>545</td></tr><tr><td>28</td><td>-</td><td>806</td></tr><tr><td>29</td><td>-</td><td>1769</td></tr><tr><td>30</td><td>-</td><td>344</td></tr><tr><td rowspan="5">31 32 33 34 35</td><td rowspan="5">$\left\lbrack  {{1372},{1593}}\right\rbrack$</td><td>729</td><td rowspan="5">80%</td><td rowspan="5">701</td><td>-</td><td rowspan="5">60%</td><td rowspan="5">161</td></tr><tr><td>514</td><td>165</td></tr><tr><td>-</td><td>-</td></tr><tr><td>1021</td><td>128</td></tr><tr><td>537</td><td>189</td></tr><tr><td rowspan="5">36 37 38 39 40</td><td rowspan="5">$\left\lbrack  {{1594},{1815}}\right\rbrack$</td><td>1296</td><td rowspan="5">40%</td><td rowspan="5">864</td><td>409</td><td rowspan="5">80%</td><td rowspan="5">425</td></tr><tr><td>-</td><td>520</td></tr><tr><td>-</td><td>680</td></tr><tr><td>432</td><td>90</td></tr><tr><td>-</td><td>-</td></tr><tr><td rowspan="5">41 42 43 44 45</td><td rowspan="5">$\left\lbrack  {{1816},{2037}}\right\rbrack$</td><td>1122</td><td rowspan="5">60%</td><td rowspan="5">1527</td><td>187</td><td rowspan="5">80%</td><td rowspan="5">520</td></tr><tr><td>1712</td><td>790</td></tr><tr><td>-</td><td>-</td></tr><tr><td>-</td><td>258</td></tr><tr><td>1748</td><td>845</td></tr><tr><td rowspan="5">46 47 48 49 50</td><td rowspan="5">$\left\lbrack  {{2038},{2259}}\right\rbrack$</td><td>2002</td><td rowspan="5">60%</td><td rowspan="5">1649</td><td>1095</td><td rowspan="5">80%</td><td rowspan="5">915</td></tr><tr><td>-</td><td>1355</td></tr><tr><td>-</td><td>745</td></tr><tr><td>1566</td><td>-</td></tr><tr><td>1378</td><td>465</td></tr></tbody></table>

Table 2: Results of the workload on crowdsourcing

表2：众包工作量的结果

<!-- Media -->

DFS-interleave successfully resolved 38 instances, striking an overall success rate of ${76}\%$ . In contrast,top-down managed with only 18 instances, settling with an overall success rate of only ${36}\%$ . To explain such a vast difference, first note that since every question has a chance of triggering a fatal mistake, the overall failure probability increases with the number of questions. The gain in accuracy achieved by DFS-interleave, therefore, can be attributed to the fact that it necessitated much fewer questions than top-down, as can be clearly seen from the table. It is worth pointing out that the crowd-cost comparison between the two algorithms is reminiscent of the patterns observed earlier in Section 7.1.

深度优先搜索交错算法（DFS - interleave）成功解决了38个实例，总体成功率达到${76}\%$。相比之下，自顶向下算法仅解决了18个实例，总体成功率仅为${36}\%$。为解释这种巨大差异，首先要注意到，由于每个问题都有可能引发致命错误，因此总体失败概率会随问题数量的增加而上升。从表中可以明显看出，深度优先搜索交错算法所需的问题数量比自顶向下算法少得多，因此其在准确性上的提升可归因于此。值得指出的是，这两种算法在众包成本方面的比较让人想起了7.1节中早些时候观察到的模式。

We delved into each of the 12 instances that DFS-interleave failed to resolve. It turned out that none of those cases were due to "careless" mistakes by the human workers. Indeed, even though such kind of mistakes did happen, their influence was essentially eliminated by the quality control measures adopted. In other words, random noise hardly played any roles in the outcome of the algorithm. This, at least in retrospect, was not surprising, and essentially confirmed the effectiveness of quality control at a modern crowdsourcing site such as figure-eight.com.

我们深入研究了深度优先搜索交错算法未能解决的12个实例。结果发现，这些情况都不是由于人工工作者的“粗心”错误造成的。实际上，尽管这类错误确实发生过，但通过采取的质量控制措施，其影响基本被消除了。换句话说，随机噪声对算法结果几乎没有影响。至少事后看来，这并不奇怪，并且从本质上证实了像figure - eight.com这样的现代众包平台质量控制的有效性。

So, what was the cause behind the mistakes made by DFS-interleave? Next, we gave three most representative causes. Interestingly, none of these causes was really the fault of the human workers. Phrased differently, those causes correspond to system noise that is difficult to deal with, and therefore would be persistent regardless of the IGS algorithm applied.

那么，深度优先搜索交错算法出错的原因是什么呢？接下来，我们给出三个最具代表性的原因。有趣的是，这些原因都并非人工工作者的真正过错。换句话说，这些原因对应着难以处理的系统噪声，因此无论应用哪种交互式图搜索（IGS）算法，这些问题都会持续存在。

Cause 1: incomplete ground truth. One mistaken instance of DFS-interleave is on a product with the title "Crocs Women’s Nadia Boot". ${}^{4}$ The product is a type of footwear that extends almost to laps. The ground truth places it under the category [*, Clothing, Shoes & Jewelry, Women, Shoes, Fashion Sneakers]. However, all the workers classified the product into the category ${\lbrack }^{ * }$ ,Clothing,Shoes &Jewelry,Women, Shoes, Boots], which also appears reasonable, and could have been added to the ground truth.

原因1：不完整的真实标签。深度优先搜索交错算法出错的一个实例涉及一款名为“卡骆驰（Crocs）女士娜迪亚靴子（Nadia Boot）”的产品。${}^{4}$该产品是一种几乎到大腿的鞋类。真实标签将其归为[*, 服装、鞋类与珠宝, 女士, 鞋类, 时尚运动鞋]类别。然而，所有工作者都将该产品归为${\lbrack }^{ * }$, 服装、鞋类与珠宝, 女士, 鞋类, 靴子]类别，这似乎也合理，并且本可以添加到真实标签中。

Cause 2: questionable ground truth. Another instance mistaken by DFS-interleave is on a product with the title "Naturalizer Women’s Lennox Pump". ${}^{5}$ The product is a pair of high-heels suitable even for business meetings. The ground truth, however, dictates that it should be under the category [*, Clothing, Shoes & Jewelry', Women, Shoes, Sandals], which no workers were able to discern.

原因2：存疑的真实标签。深度优先搜索交错算法出错的另一个实例涉及一款名为“娜然蒂诗（Naturalizer）女士伦诺克斯浅口鞋（Lennox Pump）”的产品。${}^{5}$该产品是一双甚至适合商务会议穿着的高跟鞋。然而，真实标签规定它应归为[*, 服装、鞋类与珠宝, 女士, 鞋类, 凉鞋]类别，没有一个工作者能识别出这种分类。

Cause 3: subjective judgments. Our last example is on a product titled "Kenneth Cole New York 'Modern Ombre' Blue Green Ombre Resin Linear Earrings". ${}^{6}$ By the ground truth, it is under the category [*, Clothing, Shoes & Jewelry, Women, Jewelry, Fashion]. In contrast, many workers chose [*, Clothing, Shoes & Jewelry, Women, Jewelry, Fine]. Note that the subtle difference is about whether the jewelry piece is "fine" or "fashion". This appears to be a rather subjective as one can see from the image at the URL provided earlier in the footnote.

原因3：主观判断。我们的最后一个例子是一款名为“肯尼思·柯尔（Kenneth Cole）纽约‘现代渐变色’蓝绿色渐变色树脂线性耳环”的产品。${}^{6}$根据真实标签，它属于[*, 服装、鞋类与珠宝, 女士, 珠宝, 时尚]类别。相比之下，许多工作者选择了[*, 服装、鞋类与珠宝, 女士, 珠宝, 高级]类别。请注意，细微的差别在于这件珠宝是“高级”还是“时尚”。从脚注中早些时候提供的URL图片可以看出，这似乎是一个相当主观的判断。

## 8 RELATED WORK

## 8 相关工作

Most relevant to our paper is the work of [15], which as mentioned in Section 1 introduced the offline counterpart of IGS (under the name human-assisted graph search). The solutions in [15] were designed for the scenario where the algorithm must ask all the questions altogether. The number of questions generated by those solutions is huge: often at the same magnitude as the number of nodes in the input hierarchy. As explained in Section 1, the main advantage of IGS (owing to the possibility of interaction) is that the number of questions can be reduced dramatically.

与我们的论文最相关的是文献[15]的工作，如第1节所述，该文献引入了交互式图搜索（IGS）的离线版本（名为人工辅助图搜索）。文献[15]中的解决方案是为算法必须一次性提出所有问题的场景设计的。这些解决方案生成的问题数量巨大：通常与输入层次结构中的节点数量处于同一数量级。如第1节所述，交互式图搜索（IGS）的主要优势（由于存在交互的可能性）在于可以大幅减少问题数量。

At a higher level, our work is somewhat related to human-based computation (HBC). The fundamental rationale behind this area is that some tasks are inherently easy for humans, as opposed to the old computing philosophy that "computation is a job of machines". HBC algorithms aim at engaging both humans and machines so that they can work collaboratively to solve a problem effectively and/or efficiently. In recent years, considerable attention has been devoted to crowdsourc-ing, which is a large-scaled form of HBC that involves a huge number of human workers. A significant amount of work has been carried out on studying crowdsourcing algorithms (see representative works $\left\lbrack  {2 - 4,6,8,{10},{11},{14},{19}}\right\rbrack$ ) and on developing crowdsourcing systems (see representative works $\left\lbrack  {5,9,{12},{16}}\right\rbrack  )$ .

从更宏观的层面来看，我们的工作与基于人类的计算（HBC）有一定关联。这一领域的基本原理是，与“计算是机器的工作”这一旧计算理念相反，有些任务对人类来说本质上很容易。基于人类的计算（HBC）算法旨在让人类和机器都参与进来，以便它们能够协作有效地和/或高效地解决问题。近年来，众包受到了相当多的关注，它是一种大规模的基于人类的计算形式，涉及大量的人工工作者。已经开展了大量关于研究众包算法（见代表性文献$\left\lbrack  {2 - 4,6,8,{10},{11},{14},{19}}\right\rbrack$）和开发众包系统（见代表性文献$\left\lbrack  {5,9,{12},{16}}\right\rbrack  )$）的工作。

Two remarks are in order about interpreting our work as a form of HBC. First, there is not much "computation" by the traditional yardstick of HBC: all an IGS algorithm does is to figure out the node that a human has in mind. The challenge lies in how to utilize reachability to identify that node as quickly as possible. Second, our algorithms are designed for an authoritative oracle that never errs. This implies opportunities for improving those algorithms in terms of effectiveness on a crowdsourcing platform. In fact, some crowdsourcing-specific issues have been experimentally identified in Section 7.2. Integrating our algorithms with remedies to those issues would make a promising direction for future work.

关于将我们的工作解读为一种人类辅助计算（HBC）形式，有两点需要说明。首先，按照传统的人类辅助计算标准，这里并没有太多“计算”：交互式图搜索（IGS）算法所做的只是找出人类心中所想的节点。挑战在于如何利用可达性尽可能快地识别出该节点。其次，我们的算法是为一个从不犯错的权威神谕设计的。这意味着在众包平台上有改进这些算法有效性的机会。事实上，在第7.2节中已经通过实验确定了一些特定于众包的问题。将我们的算法与解决这些问题的方法相结合，将是未来工作的一个有前景的方向。

## 9 CONCLUSIONS

## 9 结论

Conventionally, people are used to searching a decision tree/graph in the straightforward top-down fashion. This paper aims to show that there can be alternative strategies achieving better efficiency than that traditional wisdom. To allow for a rigorous algorithmic study, we introduced the the interactive graph search problem. Here, the input is a directed acyclic graph $G$ . Given an initially unknown vertex $z$ in $G$ ,the objective is to eventually locate $z$ by asking reachability questions: each question specifies a query node $q$ and obtains a boolean answer as to whether $z$ is reachable from $q$ . We have described algorithms which solve variants of the problem using a provably small number of questions, and established a nearly matching lower bound. We have also presented an experimental evaluation to demonstrate the efficiency and usefulness of the proposed solutions in real world scenarios.

传统上，人们习惯以直接的自上而下的方式搜索决策树/图。本文旨在表明，可能存在比传统方法更高效的替代策略。为了进行严格的算法研究，我们引入了交互式图搜索问题。这里，输入是一个有向无环图 $G$ 。给定 $G$ 中一个初始未知的顶点 $z$ ，目标是通过询问可达性问题最终定位 $z$ ：每个问题指定一个查询节点 $q$ ，并获得一个关于 $z$ 是否可从 $q$ 到达的布尔答案。我们描述了使用可证明的少量问题解决该问题变体的算法，并建立了一个近乎匹配的下界。我们还进行了实验评估，以证明所提出的解决方案在现实场景中的效率和实用性。

---

<!-- Footnote -->

${}^{4}$ Image at ecx.images-amazon.com/images/I/41z0zj%2BVhzL._SY395_.jpg.

${}^{4}$ 图片位于 ecx.images-amazon.com/images/I/41z0zj%2BVhzL._SY395_.jpg。

${}^{5}$ Image at ecx.images-amazon.com/images/I/41lVbFn%2B2lL._SX395_.jpg

${}^{5}$ 图片位于 ecx.images-amazon.com/images/I/41lVbFn%2B2lL._SX395_.jpg

${}^{6}$ Image at ecx.images-amazon.com/images/I/31S4Sgi-HoL_SY300_.jpg.

${}^{6}$ 图片位于 ecx.images-amazon.com/images/I/31S4Sgi-HoL_SY300_.jpg。

<!-- Footnote -->

---

## ACKNOWLEDGEMENTS

## 致谢

The research of Yufei Tao was partially supported by a direct grant (Project Number: 4055079) from CUHK and by a Faculty Research Award from Google. The research of Guolinag Li was supported by the 973 Program of China (2015CB358700), NSF of China (61632016, 61521002, 61661166012), Huawei, and TAL education.

陶宇飞的研究部分得到了香港中文大学的直接资助（项目编号：4055079）和谷歌的教师研究奖的支持。李国良的研究得到了中国973计划（2015CB358700）、国家自然科学基金（61632016、61521002、61661166012）、华为和好未来教育的支持。

## REFERENCES

## 参考文献

[1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. 2001. Introduction to Algorithms, Second Edition. The MIT Press.

[2] Susan B. Davidson, Sanjeev Khanna, Tova Milo, and Sudeepa Roy. 2013. Using the crowd for top-k and group-by queries.. In ICDT. 225-236.

[3] Eyal Dushkin and Tova Milo. 2018. Top-k Sorting Under Partial Order Information. In SIGMOD. 1007-1019.

[4] Ju Fan, Guoliang Li, Beng Chin Ooi, Kian-Lee Tan, and Jianhua Feng. 2015. iCrowd: An Adaptive Crowdsourcing Framework. In SIGMOD. 1015-1030.

[5] Michael J. Franklin, Donald Kossmann, Tim Kraska, Sukriti Ramesh, and Reynold Xin. 2011. CrowdDB: answering queries with crowd-sourcing. In SIGMOD. 61-72.

[6] Stephen Guo, Aditya G. Parameswaran, and Hector Garcia-Molina. 2012. So who won?: dynamic max discovery with the crowd. In SIG-MOD. 385-396.

[7] Ruining He and Julian McAuley. 2016. Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering. In ${WWW}$ . 507-517.

[8] Chien-Ju Ho, Shahin Jabbari, and Jennifer Wortman Vaughan. 2013. Adaptive Task Assignment for Crowdsourced Classification. In ICML. 534-542.

[9] Guoliang Li, Chengliang Chai, Ju Fan, Xueping Weng, Jian Li, Yudian Zheng, Yuanbing Li, Xiang Yu, Xiaohang Zhang, and Haitao Yuan. 2017. CDB: Optimizing Queries with Crowd-Based Selections and Joins. In SIGMOD. 1463-1478.

[10] Xuan Liu, Meiyu Lu, Beng Chin Ooi, Yanyan Shen, Sai Wu, and Meihui Zhang. 2012. CDAS: A Crowdsourcing Data Analytics System. PVLDB 5, 10 (2012), 1040-1051.

[11] Adam Marcus, David R. Karger, Samuel Madden, Rob Miller, and Se-woong Oh. 2012. Counting with the Crowd. PVLDB 6, 2 (2012), 109- 120.

[12] Adam Marcus, Eugene Wu, Samuel Madden, and Robert C. Miller. 2011. Crowdsourced Databases: Query Processing with People. In CIDR. 211-214.

[13] Jonathan J Oliver. 1993. Decision Graphs - An Extension of Decision Trees. In Int. Conf. Artificial Intelligence and Statistics. 343-350.

[14] Aditya G. Parameswaran, Hector Garcia-Molina, Hyunjung Park, Neoklis Polyzotis, Aditya Ramesh, and Jennifer Widom. 2012. Crowd-Screen: algorithms for filtering data with humans. In SIGMOD. 361- 372.

[15] Aditya G. Parameswaran, Anish Das Sarma, Hector Garcia-Molina, Neoklis Polyzotis, and Jennifer Widom. 2011. Human-assisted graph search: it's okay to ask questions. PVLDB 4, 5 (2011), 267-278.

[16] Hyunjung Park, Richard Pang, Aditya G. Parameswaran, Hector Garcia-Molina, Neoklis Polyzotis, and Jennifer Widom. 2012. Deco: A System for Declarative Crowdsourcing. PVLDB 5, 12 (2012), 1990-1993.

[17] Senjuti Basu Roy, Haidong Wang, Gautam Das, Ullas Nambiar, and Mukesh K. Mohania. 2008. Minimum-effort driven dynamic faceted search in structured databases. In CIKM. 13-22.

[18] Daniel Dominic Sleator and Robert Endre Tarjan. 1983. A Data Structure for Dynamic Trees. JCSS 26, 3 (1983), 362-391.

[19] Vasilis Verroios, Hector Garcia-Molina, and Yannis Papakonstanti-nou. 2017. Waldo: An Adaptive Human Interface for Crowd Entity Resolution. In SIGMOD. 1133-1148.

[19] Vasilis Verroios、Hector Garcia - Molina和Yannis Papakonstantinou。2017年。Waldo：用于人群实体解析的自适应人机界面。发表于SIGMOD会议。1133 - 1148。

[20] Ka-Ping Yee, Kirsten Swearingen, Kevin Li, and Marti A. Hearst. 2003. Faceted metadata for image search and browsing. In CHI. 401-408.

[20] Ka - Ping Yee、Kirsten Swearingen、Kevin Li和Marti A. Hearst。2003年。用于图像搜索和浏览的分面元数据。发表于CHI会议。401 - 408。

## A PROOF OF LEMMA 2

## 引理2的证明

Let $x$ be the number of iterations performed by ordered-interleave. For each $i \in  \left\lbrack  {1,x}\right\rbrack$ ,we denote by ${d}_{i} \leq  d - 1$ the number of questions issued at Line 4 in the $i$ -th iteration. Equivalently, ${d}_{i}$ is the number of child nodes of $u$ that are queried at Line 4 .

设 $x$ 为有序交错（ordered - interleave）执行的迭代次数。对于每个 $i \in  \left\lbrack  {1,x}\right\rbrack$ ，我们用 ${d}_{i} \leq  d - 1$ 表示在第 $i$ 次迭代的第4行发出的问题数量。等价地， ${d}_{i}$ 是在第4行查询的 $u$ 的子节点数量。

As in interleave, the binary search at Line 3 asks at most $\left\lceil  {{\log }_{2}h}\right\rceil$ questions for each iteration. It thus follows that ordered-interleave entails a cost at most

与交错（interleave）一样，第3行的二分查找每次迭代最多询问 $\left\lceil  {{\log }_{2}h}\right\rceil$ 个问题。因此，有序交错的成本最多为

$$
\left\lceil  {{\log }_{2}h}\right\rceil   \cdot  x + \mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}
$$

Since $\Pi$ has at most $1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$ levels (Lemma 1),obviously $x \leq  1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$ . It suffices to prove an upper bound for $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ .

由于 $\Pi$ 最多有 $1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$ 层（引理1），显然 $x \leq  1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$ 。只需证明 $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ 的上界即可。

For each $i \in  \left\lbrack  {1,x}\right\rbrack$ ,let ${v}_{i}$ be the "child $v$ " identified at Line 4 in the $i$ -th iteration. Denote by ${n}_{i}$ the subtree size of ${v}_{i}$ . Specially,define ${v}_{0}$ as the root $r$ of $T$ ,and ${n}_{0} = n$ .

对于每个 $i \in  \left\lbrack  {1,x}\right\rbrack$ ，设 ${v}_{i}$ 为在第 $i$ 次迭代的第4行确定的“子 $v$ ”。用 ${n}_{i}$ 表示 ${v}_{i}$ 的子树大小。特别地，将 ${v}_{0}$ 定义为 $T$ 的根 $r$ ，且 ${n}_{0} = n$ 。

Lemma 6. ${n}_{i} \leq  {n}_{i - 1}/\left( {{d}_{i} + 1}\right)$ .

引理6. ${n}_{i} \leq  {n}_{i - 1}/\left( {{d}_{i} + 1}\right)$ 。

Proof. In executing the $i$ -th iteration,Line 3 binary searches a path $\pi$ to identify a node $u$ in $\pi$ . Since ${v}_{i - 1}$ is the first node on $\pi$ ,we know that $u$ must be in the subtree of ${v}_{i - 1}$ . Hence,the subtree size of $u$ is at most ${n}_{i - 1}$ .

证明。在执行第 $i$ 次迭代时，第3行通过二分查找一条路径 $\pi$ 来确定 $\pi$ 中的一个节点 $u$ 。由于 ${v}_{i - 1}$ 是 $\pi$ 上的第一个节点，我们知道 $u$ 一定在 ${v}_{i - 1}$ 的子树中。因此，$u$ 的子树大小至多为 ${n}_{i - 1}$ 。

At Line 4 (applying the modification in Section 4.2 that searches the child nodes of $u$ in non-ascending order of subtree size),since ${d}_{i}$ child nodes of $u$ were queried until ${v}_{i}$ is found, $u$ has at least ${d}_{i}$ child nodes whose subtrees are as large as that of ${v}_{i}$ (counting also the child node of $u$ in $\pi$ ). The lemma then follows.

在第4行（应用4.2节中的修改，即按子树大小非升序搜索 $u$ 的子节点），由于在找到 ${v}_{i}$ 之前查询了 $u$ 的 ${d}_{i}$ 个子节点，$u$ 至少有 ${d}_{i}$ 个子节点，其所在子树与 ${v}_{i}$ 的子树一样大（也包括 $\pi$ 中 $u$ 的子节点）。引理得证。

$$
\text{LEMMA 7.}\left( {{d}_{1} + 1}\right) \left( {{d}_{2} + 1}\right) \ldots \left( {{d}_{x} + 1}\right)  \leq  n\text{.}
$$

Proof. By Lemma 6, we know that

证明。根据引理6，我们知道

$$
{n}_{x} \leq  \frac{n}{\left( {{d}_{1} + 1}\right) \left( {{d}_{2} + 1}\right) \ldots \left( {{d}_{x} + 1}\right) }.
$$

The lemma then follows from ${n}_{x} \geq  1$ .

然后由 ${n}_{x} \geq  1$ 可得该引理。

Now it remains to upper bound $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ subject to the above condition. The lemma below provides such an upper bound, which will complete the proof.

现在，在上述条件下，还需要对 $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ 进行上界估计。下面的引理给出了这样的上界，从而完成证明。

LEMMA 8. $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i} \leq  \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$ .

引理8. $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i} \leq  \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$ 。

Proof. We prove that the lemma holds even if ${d}_{1},\ldots ,{d}_{x}$ are non-negative real values. Without loss of generality, assume ${d}_{1} \geq  {d}_{2} \geq  \ldots  \geq  {d}_{x}$ .

证明。我们证明即使 ${d}_{1},\ldots ,{d}_{x}$ 是非负实数值，该引理仍然成立。不失一般性，假设 ${d}_{1} \geq  {d}_{2} \geq  \ldots  \geq  {d}_{x}$ 。

Claim: Fix the value of $x$ . To maximize $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ subject to Lemma 7, the best strategy is to set

命题：固定 $x$ 的值。为了在满足引理7的条件下最大化 $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ ，最佳策略是设置

- variables ${d}_{1},\ldots ,{d}_{i}$ to $d - 1$ for some $i$ ;

- 对于某个 $i$ ，将变量 ${d}_{1},\ldots ,{d}_{i}$ 设置为 $d - 1$ ；

- optionally ${d}_{i + 1}$ to a value greater than 0 but less than $d - 1$

- 可选地，将 ${d}_{i + 1}$ 设置为大于0但小于 $d - 1$ 的值

- and the remaining variables all to 0 .

- 其余变量都设置为0。

Proof of the claim: Suppose that $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ is maximized when ${d}_{i - 1}$ and ${d}_{i}$ are both greater than 0 but less than $d - 1$ , for some $i \geq  2$ . Let $s = \left( {{d}_{i - 1} + 1}\right) \left( {{d}_{i} + 1}\right)$ . Clearly, $1 < s < {d}^{2}$ .

命题证明：假设当对于某个 $i \geq  2$ ，${d}_{i - 1}$ 和 ${d}_{i}$ 都大于0但小于 $d - 1$ 时，$\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ 取得最大值。设 $s = \left( {{d}_{i - 1} + 1}\right) \left( {{d}_{i} + 1}\right)$ 。显然，$1 < s < {d}^{2}$ 。

- If $d < s < {d}^{2}$ ,we set ${d}_{i - 1}$ to $d - 1$ ,and ${d}_{i}$ to $\frac{s}{d} - 1$ . The new values still satisfy Lemma 7, but increase $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ ,contradicting the fact that $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ is already maximized.

- 如果 $d < s < {d}^{2}$ ，我们将 ${d}_{i - 1}$ 设置为 $d - 1$ ，并将 ${d}_{i}$ 设置为 $\frac{s}{d} - 1$ 。新的值仍然满足引理7，但会增大 $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ ，这与 $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ 已经是最大值相矛盾。

- If $s \leq  d$ ,we set ${d}_{i - 1}$ to $s - 1$ ,and ${d}_{i}$ to 0 . The new values still satisfy Lemma 7,but increase $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ ,i.e., contradiction. QED

- 如果 $s \leq  d$ ，我们将 ${d}_{i - 1}$ 设置为 $s - 1$ ，并将 ${d}_{i}$ 设置为0。新的值仍然满足引理7，但会增大 $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ ，即产生矛盾。证毕

Now we vary $x$ . When $x \leq  \left\lceil  {{\log }_{d}n}\right\rceil$ ,by the above claim $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i} \leq  \left( {d - 1}\right) x \leq  \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil  .$

现在我们改变 $x$。当 $x \leq  \left\lceil  {{\log }_{d}n}\right\rceil$ 时，根据上述断言 $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i} \leq  \left( {d - 1}\right) x \leq  \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil  .$

When $x \geq  \left\lceil  {{\log }_{d}n}\right\rceil   + 1$ ,by the above claim $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ is maximized by setting

当 $x \geq  \left\lceil  {{\log }_{d}n}\right\rceil   + 1$ 时，根据上述断言，通过设置 $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}$ 可使其最大化

- ${d}_{i}$ to $d - 1$ for $1 \leq  i \leq  \left\lfloor  {{\log }_{d}n}\right\rfloor$ ;

- 对于 $1 \leq  i \leq  \left\lfloor  {{\log }_{d}n}\right\rfloor$，将 ${d}_{i}$ 设置为 $d - 1$；

- (only if $n$ is not a power of $d$ ) ${d}_{1 + \left\lfloor  {{\log }_{d}n}\right\rfloor  }$ to a value larger than 0 and less than $d - 1$ ;

- （仅当 $n$ 不是 $d$ 的幂时）将 ${d}_{1 + \left\lfloor  {{\log }_{d}n}\right\rfloor  }$ 设置为大于 0 且小于 $d - 1$ 的值；

- and the remaining variables to 0 .

- 并将其余变量设置为 0。

It thus follows that $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i} \leq  \left( {d - 1}\right) x \leq  \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$ .

由此可得 $\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i} \leq  \left( {d - 1}\right) x \leq  \left( {d - 1}\right)  \cdot  \left\lceil  {{\log }_{d}n}\right\rceil$。

## B PROOF OF LEMMA 3

## B 引理 3 的证明

The hard hierarchy is simply a perfect $d$ -ary tree $T$ with $h + 1$ levels,where $h = \left\lfloor  {{\log }_{d}n}\right\rfloor$ .

困难层次结构简单来说就是一个完美的 $d$ 叉树 $T$，有 $h + 1$ 层，其中 $h = \left\lfloor  {{\log }_{d}n}\right\rfloor$。

We let the adversary - Bob - play the role of oracle. He does not choose the target node $z$ at the beginning. Instead, he observes how to algorithm runs, and gradually shrinks the set of nodes where he could place $z$ ,without violating any of the answers he (as the oracle) has given to the algorithm's questions so far. He will execute a strategy of $h$ rounds,where in each round he forces the algorithm to ask at least $d - 1$ questions.

我们让对手——鲍勃——扮演神谕的角色。他在开始时并不选择目标节点 $z$。相反，他观察算法的运行情况，并逐渐缩小他可以放置 $z$ 的节点集合，同时不违反他（作为神谕）到目前为止对算法问题给出的任何答案。他将执行一个 $h$ 轮的策略，在每一轮中，他迫使算法至少询问 $d - 1$ 个问题。

Bob's strategy adheres to the following invariant: at the beginning of a round,he has chosen a node $u$ ,and made up his mind to place $z$ in the subtree of $u$ eventually (for round $1,u$ is simply the root of $T$ ). At the end of the round,he will descend into a child node of $u$ ,and set $u$ to that child node for the next round.

鲍勃的策略遵循以下不变性：在一轮开始时，他选择了一个节点 $u$，并决定最终将 $z$ 放置在 $u$ 的子树中（对于第 $1,u$ 轮，$u$ 就是 $T$ 的根节点）。在这一轮结束时，他将下降到 $u$ 的一个子节点，并将该子节点设置为下一轮的 $u$。

Now we explain the details of Bob's actions in a round. Let $S$ be the set of $d$ child nodes of $u$ . Suppose that Alice asks a question with query node $q$ . Bob answers the question as follows:

现在我们解释鲍勃在一轮中的具体行动。设 $S$ 是 $u$ 的 $d$ 个子节点的集合。假设爱丽丝用查询节点 $q$ 提出一个问题。鲍勃按如下方式回答问题：

- If $q$ is not in the subtree of $u$ ,he returns reach $\left( q\right)  = {no}$ .

- 如果 $q$ 不在 $u$ 的子树中，他返回可达 $\left( q\right)  = {no}$。

- If $q = u$ ,he returns $\operatorname{reach}\left( q\right)  = {yes}$ .

- 如果 $q = u$，他返回 $\operatorname{reach}\left( q\right)  = {yes}$。

- If $q$ is in the subtree of a child node $v$ of $u$ ,he returns $\operatorname{reach}\left( q\right)  =$ no. Furthermore,if $v$ is still in $S$ ,he removes $v$ from $S$ . The round finishes when $\left| S\right|$ has decreased to 1 . In this case, he sets $u$ to the only node left in $S$ . With this $u$ ,the next round starts.

- 如果 $q$ 在 $u$ 的一个子节点 $v$ 的子树中，他返回 $\operatorname{reach}\left( q\right)  =$ 否。此外，如果 $v$ 仍在 $S$ 中，他将 $v$ 从 $S$ 中移除。当 $\left| S\right|$ 减少到 1 时，这一轮结束。在这种情况下，他将 $u$ 设置为 $S$ 中剩下的唯一节点。以这个 $u$，下一轮开始。

<!-- Media -->

<!-- figureText: Upre -->

<img src="https://cdn.noedgeai.com/0195c91c-378f-77b0-8d6e-bb14508995e7_15.jpg?x=1115&y=235&w=334&h=310&r=0"/>

Figure 8: Proof of Lemma 9

图 8：引理 9 的证明

<!-- Media -->

After $h$ rounds, $u$ is a leaf node in $T$ . Bob then chooses $u$ as the target node $z$ .

经过 $h$ 轮后，$u$ 是 $T$ 中的一个叶节点。然后鲍勃选择 $u$ 作为目标节点 $z$。

In each round,Alice needs to ask at least $d - 1$ questions in order to shrink $\left| S\right|$ from $d$ to 1 . Therefore,in the whole process,Alice must ask at least $\left( {d - 1}\right)  \cdot  h = \left( {d - 1}\right)  \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor$ questions.

在每一轮中，爱丽丝需要至少询问$d - 1$个问题，以便将$\left| S\right|$从$d$缩小到1。因此，在整个过程中，爱丽丝必须至少询问$\left( {d - 1}\right)  \cdot  h = \left( {d - 1}\right)  \cdot  \left\lfloor  {{\log }_{d}n}\right\rfloor$个问题。

## C PROOF OF LEMMA 4

## C 引理4的证明

Consider the moment when ${v}_{1}$ was discovered. By our ordering strategy,count $\left( {v}_{1}\right)  \geq  \operatorname{count}\left( {v}_{2}\right)$ at that moment. By the white-path theorem (Theorem 1), we know that the subtree of ${v}_{1}$ of $T$ has a size equal to precisely count $\left( {v}_{1}\right)$ .

考虑发现${v}_{1}$的时刻。根据我们的排序策略，计算该时刻的$\left( {v}_{1}\right)  \geq  \operatorname{count}\left( {v}_{2}\right)$。根据白路径定理（定理1），我们知道$T$中${v}_{1}$的子树大小恰好等于计数$\left( {v}_{1}\right)$。

What is the subtree size of ${v}_{2}$ ? By the white-path theorem, it is exactly the value of $\operatorname{count}\left( {v}_{2}\right)$ at the moment when ${v}_{2}$ was discovered,which is after the discovery of ${v}_{1}$ . As $\operatorname{count}\left( {v}_{2}\right)$ cannot increase during the algorithm,we conclude that the subtree size of ${v}_{1}$ is at least that of ${v}_{2}$ .

${v}_{2}$的子树大小是多少？根据白路径定理，它恰好是发现${v}_{2}$时$\operatorname{count}\left( {v}_{2}\right)$的值，而发现${v}_{2}$是在发现${v}_{1}$之后。由于在算法执行过程中$\operatorname{count}\left( {v}_{2}\right)$不会增加，我们得出结论：${v}_{1}$的子树大小至少与${v}_{2}$的子树大小相同。

## D CORRECTNESS OF DFS-INTERLEAVE

## D DFS - 交错算法的正确性

As before,let $G$ be the input DAG hierarchy, $V$ the set of vertices in $G$ ,and $T$ the heavy-path DFS-tree decided in Section 5.1. Given a node $u \in  V$ ,define $P\left( u\right)$ as the set of nodes ${u}^{\prime } \in  V$ satisfying:

和之前一样，设$G$为输入的有向无环图（DAG）层次结构，$V$为$G$中的顶点集，$T$为在5.1节中确定的重路径深度优先搜索（DFS）树。给定一个节点$u \in  V$，将$P\left( u\right)$定义为满足以下条件的节点${u}^{\prime } \in  V$的集合：

- ${u}^{\prime }$ was discovered earlier than $u$ in the DFS described in Section 5.1, and

- 在5.1节描述的深度优先搜索中，${u}^{\prime }$比$u$先被发现，并且

- ${u}^{\prime }$ is not an ancestor of $u$ in $T$ .

- 在$T$中，${u}^{\prime }$不是$u$的祖先节点。

Example. Consider again the example shown in Figure 5. $P\left( \text{node 7}\right)$ ,for instance,consists of nodes8,10,11,6. As another example, $P\left( {\text{node }{13}}\right)  = \{ 4,8,{10},{11},6,7,{12},{14}\}$ .

示例。再次考虑图5所示的示例。例如，$P\left( \text{node 7}\right)$由节点8、10、11、6组成。再举一个例子，$P\left( {\text{node }{13}}\right)  = \{ 4,8,{10},{11},6,7,{12},{14}\}$。

LEMMA 9. Consider any node u obtained at Line 4 of DFS-interleave. None of the nodes in $P\left( u\right)$ can reach the target node $z$ .

引理9。考虑在DFS - 交错算法的第4行得到的任意节点u。$P\left( u\right)$中的任何节点都无法到达目标节点$z$。

Proof. Call node $u$ a pivot node. Also,let us refer to Lines 3-7 as an iteration. We will prove the lemma by induction on the number of iterations.

证明。称节点$u$为枢轴节点。此外，我们将第3 - 7行称为一次迭代。我们将通过对迭代次数进行归纳来证明该引理。

<!-- Media -->

<!-- figureText: 50 ${10}^{5}$ ordered-interleave top-down average CSS ${10}^{4}$ ${10}^{3}$ ${10}^{2}$ 10 10 15 20 25 30 35 40 45 50 number of k-questions (b) Progressiveness (228,244] (276,292 (292,308) (308,324) (324,340) (340,356 (356,372 (372,388 (388,404) sum of out-degrees of ancestors (c) Cost vs. sum of out-degrees of ancestors Figure 9: $k$ -IGS on Amazon with $k = 5$ ${10}^{5}$ DFS-interleave top-down average CSS ${10}^{4}$ ${10}^{3}$ ${10}^{2}$ 10 10 20 30 40 50 60 70 80 number of k-questions (b) Progressiveness (398,435 (472,509 (509,546] (546,583 (583,620) (620,657) (657,694 (694,731) sum of out-degrees of ancestors (c) Cost vs. sum of out-degrees of ancestors number of k-questions ordered-interleave top-down 30 20 10 0 1 2 4 5 6 7 8 9 node depth (a) Cost vs. node depth 45 ordered-interleave number of k-questions top-down 25 5 (116,132 (132,148 (148,164) (164,180] (180,196 (196,212) (212,228) 45 number of k -questions DFS-interleave top-down 30 25 20 15 10 2 8 9 10 node depth (a) Cost vs. node depth 60 DFS-interleave number of k-questions top-down [28,65] (65,102] (102,139) (139,176 (176,213] (213,250) (250,287] (287,324) (324,361 (361,398) -->

<img src="https://cdn.noedgeai.com/0195c91c-378f-77b0-8d6e-bb14508995e7_16.jpg?x=262&y=281&w=1254&h=1378&r=0"/>

Figure 10: $k$ -IGS on ImageNet with $k = 5$

图10：在ImageNet数据集上使用$k = 5$的$k$ - IGS算法

<!-- Media -->

First Iteration. In this case $P\left( u\right)  = \varnothing$ noticing that $\pi$ is the leftmost root-to-leaf path of $T$ . The claim obviously holds.

第一次迭代。在这种情况下$P\left( u\right)  = \varnothing$，注意到$\pi$是$T$中最左侧的根到叶路径。该命题显然成立。

Inductive Step: Iteration $i \geq  2$ . Let ${u}_{pre}$ be the pivot node obtained in Iteration $i - 1$ . The node $\widehat{u}$ at Line 3 of the $i$ -th iteration (which is also the node $v$ at Line 5 of Iteration $i - 1$ ) is a child node of ${u}_{pre}$ . The pivot node of this iteration is on the leftmost $\widehat{u}$ -to-leaf path $\pi$ of $T$ . See Figure 8 for an illustration,where ${v}_{1},\ldots ,{v}_{x}\left( {x \geq  0}\right)$ are the child nodes of ${u}_{\text{pre }}$ to the left of $\widehat{u}$ .

归纳步骤：第$i \geq  2$次迭代。设${u}_{pre}$为在第$i - 1$次迭代中得到的枢轴节点。第$i$次迭代第3行的节点$\widehat{u}$（它也是第$i - 1$次迭代第5行的节点$v$）是${u}_{pre}$的子节点。本次迭代的枢轴节点位于$T$中最左侧的从$\widehat{u}$到叶的路径$\pi$上。如图8所示，${v}_{1},\ldots ,{v}_{x}\left( {x \geq  0}\right)$是${u}_{\text{pre }}$中位于$\widehat{u}$左侧的子节点。

What are the nodes in $P\left( u\right)  \smallsetminus  P\left( {u}_{\text{pre }}\right)$ ? Remember that a pre-order traversal of $T$ enumerates the nodes exactly in the order they were discovered in DFS. Therefore, $P\left( u\right)  \smallsetminus  P\left( {u}_{\text{pre }}\right)$ is exactly the set of nodes that are in the subtrees of ${v}_{1},\ldots ,{v}_{x}$ as shown in the figure.

$P\left( u\right)  \smallsetminus  P\left( {u}_{\text{pre }}\right)$ 中的节点是什么？请记住，$T$ 的前序遍历恰好按照深度优先搜索（DFS）中发现节点的顺序枚举这些节点。因此，$P\left( u\right)  \smallsetminus  P\left( {u}_{\text{pre }}\right)$ 恰好是图中所示 ${v}_{1},\ldots ,{v}_{x}$ 的子树中的节点集合。

By the way our algorithm runs,we know that $\widehat{u}$ is the leftmost child $v$ of ${u}_{\text{pre }}$ with reach $\left( v\right)  =$ yes. It thus follows that none of the nodes ${v}_{1},\ldots ,{v}_{x}$ can reach the target node $z$ ; and therefore, neither can any of their descendants.

根据我们算法的运行方式，我们知道 $\widehat{u}$ 是 ${u}_{\text{pre }}$ 具有可达性 $\left( v\right)  =$ 为“是”的最左子节点 $v$。由此可知，节点 ${v}_{1},\ldots ,{v}_{x}$ 中没有一个可以到达目标节点 $z$；因此，它们的任何后代节点也都不能到达。

<!-- Media -->

<!-- figureText: number of k-questions ordered-interleave number of k-questions 140 DFS-interleave 120 top-down 100 80 60 40 20 1 2 9 10 k (b) ImageNet top-down 1 2 3 6 7 8 9 10 (a) Amazon -->

<img src="https://cdn.noedgeai.com/0195c91c-378f-77b0-8d6e-bb14508995e7_17.jpg?x=292&y=297&w=1189&h=347&r=0"/>

Figure 11: $k$ -IGS: Cost vs. $k$

图 11：$k$ -IGS：成本与 $k$ 的关系

<!-- Media -->

By the inductive assumption,none of the nodes in $P\left( {u}_{pre}\right)$ can reach $z$ . We thus conclude that no nodes in $P\left( u\right)$ can reach $z$ ,completing the proof.

根据归纳假设，$P\left( {u}_{pre}\right)$ 中的任何节点都不能到达 $z$。因此，我们得出结论，$P\left( u\right)$ 中的任何节点都不能到达 $z$，从而完成证明。

We now return to the correctness of DFS-interleave. It suffices to show that when Line 4 finds no child $v$ of $u$ satisfying reach $\left( v\right)  =$ yes, $u$ must be the target node (and hence, is correctly returned at Line 5).

现在我们回到深度优先搜索交错算法（DFS - interleave）的正确性上。只需证明当第 4 行没有找到 $u$ 的子节点 $v$ 满足可达性 $\left( v\right)  =$ 为“是”时，$u$ 必定是目标节点（因此，在第 5 行正确返回）。

Assume that this is not true,i.e., $u$ is not the target node $z$ . Then,since $\operatorname{reach}\left( u\right)  =$ yes,the out-neighbor property (Proposition 1) tells us that $u$ must have an out-neighbor ${u}^{\prime }$ that can reach $z$ . Consider the edge $\left( {u,{u}^{\prime }}\right)$ . As discussed in Section 3.2,every edge in $G$ can be classified as a tree edge,a forward edge,or a cross edge. We know that $\left( {u,{u}^{\prime }}\right)$ is not a tree edge; otherwise,the algorithm would have found ${u}^{\prime }$ as a child of $u$ in $T$ . It cannot be a forward edge either,because in that case,still the algorithm would have found a child $v$ of $u$ with reach $\left( v\right)  =$ yes. Hence, $\left( {u,{u}^{\prime }}\right)$ must be a cross edge.

假设这不是真的，即 $u$ 不是目标节点 $z$。那么，由于 $\operatorname{reach}\left( u\right)  =$ 为“是”，出邻接点性质（命题 1）告诉我们，$u$ 必定有一个出邻接点 ${u}^{\prime }$ 可以到达 $z$。考虑边 $\left( {u,{u}^{\prime }}\right)$。如第 3.2 节所讨论的，$G$ 中的每条边都可以分类为树边、前向边或交叉边。我们知道 $\left( {u,{u}^{\prime }}\right)$ 不是树边；否则，算法会在 $T$ 中找到 ${u}^{\prime }$ 作为 $u$ 的子节点。它也不可能是前向边，因为在那种情况下，算法仍然会找到 $u$ 的一个子节点 $v$ 满足可达性 $\left( v\right)  =$ 为“是”。因此，$\left( {u,{u}^{\prime }}\right)$ 必定是交叉边。

Since ${u}^{\prime }$ is not a descendant of $u$ in $T$ ,the white-path theorem (Theorem 1) tells us that ${u}^{\prime }$ was discovered before $u$ . Furthermore, ${u}^{\prime }$ cannot be an ancestor of $u$ in $T$ (there can be no cycles). Therefore, ${u}^{\prime } \in  P\left( u\right)$ . However,Lemma 9 asserts that the target node $z$ cannot be reachable from ${u}^{\prime }$ ,giving a contradiction.

由于在 $T$ 中 ${u}^{\prime }$ 不是 $u$ 的后代节点，白色路径定理（定理 1）告诉我们，${u}^{\prime }$ 是在 $u$ 之前被发现的。此外，在 $T$ 中 ${u}^{\prime }$ 不可能是 $u$ 的祖先节点（不能有环）。因此，${u}^{\prime } \in  P\left( u\right)$。然而，引理 9 断言目标节点 $z$ 不能从 ${u}^{\prime }$ 到达，这产生了矛盾。

## E PROOF OF LEMMA 5

## E 引理 5 的证明

Similar to Appendix A, define

与附录 A 类似，定义

- $x$ as the number of iterations performed by ordered-interleave;

- $x$ 为有序交错算法（ordered - interleave）执行的迭代次数；

- for each $i \in  \left\lbrack  {1,x}\right\rbrack  ,{d}_{i} \leq  d - 1$ as the number of child nodes of $u$ that are queried at Line 4 in the $i$ -th iteration. It thus follows that ordered-interleave entails a cost at most

- 对于每个 $i \in  \left\lbrack  {1,x}\right\rbrack  ,{d}_{i} \leq  d - 1$，为在第 $i$ 次迭代的第 4 行查询的 $u$ 的子节点数量。由此可知，有序交错算法的成本至多为

$$
\left\lceil  {{\log }_{k}h}\right\rceil   \cdot  x + \mathop{\sum }\limits_{{i = 1}}^{x}\left\lceil  {{d}_{i}/k}\right\rceil  
$$

$$
 \leq  \left\lceil  {{\log }_{k}h}\right\rceil   \cdot  x + \mathop{\sum }\limits_{{i = 1}}^{x}\left( {1 + {d}_{i}/k}\right) 
$$

$$
 \leq  \left( {1 + \left\lceil  {{\log }_{k}h}\right\rceil  }\right)  \cdot  x + \frac{1}{k}\mathop{\sum }\limits_{{i = 1}}^{x}{d}_{i}
$$

By $x \leq  1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$ (Lemma 1) and Lemma 8,we complete the proof.

根据 $x \leq  1 + \left\lfloor  {{\log }_{2}n}\right\rfloor$（引理 1）和引理 8，我们完成证明。

## F EXPERIMENTS ON $k$ -IGS

## F 关于 $k$ -IGS 的实验

We now proceed to the $k$ -IGS problem in Section 6. Remember that an oracle in this problem is more powerful, in the sense that each time it can reveal the reachability of $k$ nodes to the target node.

现在我们将在第6节中讨论$k$ -IGS问题。请记住，此问题中的神谕（oracle）功能更强大，因为它每次都能揭示$k$个节点到目标节点的可达性。

Repeating the experiments of Figures 6 and 7 but setting $k = 5$ ,we obtained the results in Figures 9 and 10 for Amazon and ImageNet, respectively. The behavior of all algorithms and their relative superiority were very similar to what was observed in Section 7.1.

重复图6和图7的实验，但设置$k = 5$，我们分别得到了亚马逊（Amazon）和ImageNet数据集在图9和图10中的结果。所有算法的表现及其相对优势与7.1节中观察到的非常相似。

The last experiment inspected the influence of $k$ on the algorithms' cost. Focusing on ImageNet, Figure 11a plots the average (per-instance) cost of ordered-interleave in processing a workload as $k$ grew from 1 to 10,and also the same for top-down. Turning to ImageNet, Figure 11b presents the corresponding results with respect to DFS-interleave and top-down. As expected, all algorithms had their costs improved continuously as $k$ got higher,confirming our theoretical analysis.

最后一个实验考察了$k$对算法成本的影响。聚焦于ImageNet数据集，图11a绘制了随着$k$从1增加到10，有序交错（ordered - interleave）算法处理工作负载的平均（每个实例）成本，同时也绘制了自顶向下（top - down）算法的相同情况。再看ImageNet数据集，图11b展示了深度优先搜索交错（DFS - interleave）算法和自顶向下算法的相应结果。正如预期的那样，随着$k$的增大，所有算法的成本都持续降低，这证实了我们的理论分析。