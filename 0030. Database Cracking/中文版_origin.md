
# 0. Abstract

本文研究了当 $r \leq n$ 的值事先未知时，如何以在线方式（即只有在前一个查询结果就绪后才给出下一个查询）最小化回答 $n$ 元素上 $r$ 查询的总成本。传统的索引在回答查询之前，首先要在 $n$ 元素上建立一个完整的索引，这可能并不适合，因为索引的构建时间通常为 $\Omega(n \log n)$ - 可能会成为性能瓶颈。与此相反，对于许多问题来说，在[1, n]$中的每$r，$r$查询的总成本都有一个$\Omega(n \log (1+r))$的下限。匹配这个下限是延迟数据结构（DDS）的主要目标，在系统界也被称为数据库破解。对于各类问题，我们提出了将传统索引转换为 DDS 算法的通用还原方法，这些算法可以在 $r$ 的较长范围内匹配下限。对于一个可分解的问题，如果一个数据结构可以在 $O(n \log n)$ 时间内建立，并且有 $Q(n)$ 查询搜索时间，那么我们的还原就会产生一种算法，它可以在所有 $r \leq \frac{n \log n}{Q(n)}$时间内以 $O(n \log (1+r))$ 运行，其中上界 $\frac{n \log n}{Q(n)}$在温和的约束条件下是渐近最佳的。特别是，如果$Q(n)=O(\log n)$，那么$O(n \log (1+r))$时间保证扩展到所有的$r \leq n$，我们可以用它来优化解决大量的DDS问题。我们的结果可以推广到一类 “频谱可索引问题”，它包含可分解问题。

## 1 Introduction

传统索引在开始回答查询之前，首先会在整个数据集上创建一个索引。例如，在$O(n \log n)$时间内，在一个由$n$实值组成的（未排序的）集合$S$上建立一棵二叉搜索树（BST）后，我们可以在$O(\log n)$时间内使用这棵树回答每个前置查询${ }^{1}$。然而，当数据集 $S$ 只被搜索少量次数时，这种模式就会出现问题。最极端的情况是，如果只需要回答一个查询，最好的方法是全面扫描 $S$，这只需要 $O(n)$ 时间。更一般地说，如果我们要以在线方式回答 $r$ 查询，即在输出前一个查询的结果后再给出下一个查询，那么有可能付出 $O(n\log(1+r))$的总成本[19]。当 $r \ll n$ 时（例如，$r=$ polylog $n$ 或 $2^{\sqrt{\log n}}$，$\text{cost}为 O(n \log (1+r))$ 打破了$\Omega(n \log n)$的障碍，表明排序是不必要的。

---

在许多大数据应用中都会出现类似上述情况，即收集了大量数据，但只进行少量查询。这种现象推动了系统界以数据库破解为名的一系列研究；参见 [12,13,16-18,23,29,31,32] 及其中的参考文献。在这些环境中，问题的大小 $n$ 是巨大的，因此即使排序也被认为是昂贵的，应尽可能避免。此外，无法预测需要支持多少查询，即前面提到的整数 $r$。数据库破解不会立即构建完整的索引，而是在每次查询时只计算构建索引的一部分。如果 $r$ 最终达到某个阈值，则会创建完整的索引，但更有可能的情况是，$r$ 会停在某个明显低于该阈值的值上。数据库破解的挑战在于确保 “平稳性”：随着 $r$ 的增长，迄今为止所有 $r$ 查询的总成本应尽可能缓慢地增长。

---

在理论领域，1988 年，Karp、Motwani 和 Raghavan [19] 以延迟数据结构（DDS）为名，对数据库破解的核心数据结构问题进行了研究。他们解释了如何在事先不知道 $r$ 的情况下，以 $O(n \log (1+r))$ 的总成本回答 $n$ 元素上 $r \in[1, n]$ 的查询。对于每一个 $r \in[1, n]$，他们证明了这些问题的匹配下限为 $\Omega(n \log (1+r))$ 。通过还原，同样的下界在许多其他 DDS 问题上也被证明成立。

---

这项工作探讨了以下主题：如何设计一种通用还原法，在给定（传统）数据结构的情况下，将其自动转化为高效的 DDS 算法？这种还原可以大大简化 DDS 算法的设计，并为传统索引和 DDS 之间错综复杂的联系提供新的启示。

---

**数学约定。** 我们用 $\mathbb{N}^{+}$ 表示正整数集合。对于任意整数 $x \geq 1$，令 $[x]$ 表示集合 ${1,2,\ldots,x}$。所有对数默认以 2 为底。定义 $\log (x)=\log (1+x)$，其中 $x \geq 1$。

### 1.1 Problem Definitions

本小节将形式化待研究的问题。设 $S$ 是一个从域 $\mathbb{D}$ 中抽取的包含 $n$ 个元素的集合，称为数据集（dataset）。设 $\mathbb{Q}$ 是一个（可能是无限的）集合，其中的元素称为谓词（predicates）。给定一个谓词 $q \in \mathbb{Q}$，在 $S$ 上发起的查询返回一个答案，记为 $\text{Ans}_{q}(S)$。我们假设，对于任意 $q \in \mathbb{Q}$，答案 $\text{Ans}_{q}(S)$ 可以用 $O(1)$ 个字表示，并能在 $O(n)$ 时间内计算得到（这本质上意味着查询可以通过暴力算法，如穷举扫描，来完成）。

---

**延迟数据结构化 (Deferred Data Structuring, DDS)。**现在我们形式化定义 DDS 问题。最初，算法 $\mathcal{A}$ 接收到一个以数组形式存储的数据集 $S$，其中的元素是随机排列的。对手首先选择一个谓词 $q_{1}$ 作为第一个查询，并向 $\mathcal{A}$ 请求答案 $\text{ANS}_{q_{1}}(S)$。之后，迭代地，对于每个 $i \geq 2$，在获得第 $(i-1)$ 次查询的答案后，对手可以决定终止整个过程，或者选择下一个查询的谓词 $q_{i}$ 并向 $\mathcal{A}$ 请求答案 $\text{ANS}_{q_{i}}(S)$。对手被允许观察算法 $\mathcal{A}$ 的执行过程，因此可以选择对 $\mathcal{A}$ 来说“最不利”的 $q_{i}$。

---

若算法 $\mathcal{A}$ 对于 $t$ 次查询能保证运行时间为 $\text{Time}(n, r)$，则表示对于任意 $r \leq t$，前 $r$ 次查询的总成本至多为 $\text{Time}(n, r)$。我们将集中研究 $t \leq n$ 的情况，因为这是数据库分裂（database cracking）中重要的场景。

---

**最佳性连贯性 (Streak of Optimality)。** 尽管上述设置是延迟数据结构（DDS）领域的“标准”做法，但从数据库裂解（database cracking）的角度来看，有必要从一个全新的视角进行研究。由于数据库裂解主要在数据集接受较少查询的情况下才显得有用，因此设计特别适合少量查询情况下高效的DDS算法是至关重要的。

---

为了形式化“特别高效”的概念，我们利用这样一个事实：对于许多DDS问题（包括本文所涉及的问题），存在难度界限（hardness barriers），规定了在相关的计算模型下，$\text{Time}(n, r)=\Omega(n \log r)$ 对于任意 $r \in [1, n]$ 成立。基于此，我们定义，若一个算法 $\mathcal{A}$ 的运行时间满足 $\text{Time}(n, r)=O(n \log r)$ 且对所有 $r \leq \text{LogStreak}(n)$ 成立，则称其保证了一个 $\log(r)$ 连续性（$\log(r)$-streak）为 $\text{LogStreak}(n)$。

---

最差的 $\log(r)$ 连续性保证是 $\text{LogStreak}(n)=O(1)$ ——这是显而易见的，因为任何查询都可以在 $O(n)$ 时间内回答。理想情况下，我们希望 $\text{LogStreak}(n)=n$，但如本文后续所论证，这并不总是可能的。然而，在实际应用中，如果某个算法能够确保 $\text{LogStreak}(n)=\Omega\left(n^{\epsilon}\right)$（其中 $\epsilon>0$ 是某个常数），则已经足够实用，因为 $\Omega\left(n^{\epsilon}\right)$ 个查询可能已经超出了数据库裂解的吸引力范围。

---

**问题类别 (Classes of Problems)。** 上述定义框架可以根据数据域 $\mathbb{D}$、谓词域 $\mathbb{Q}$ 和结构 $\mathcal{T}$ 的不同，专门化为各种问题实例。接下来，我们介绍两个与本文讨论相关的问题类别：

- 如果对于任意不相交集合 $S_{1}, S_{2} \subseteq \mathbb{D}$ 和任意谓词 $q \in \mathbb{Q}$，可以通过 $\text{ANs}_{q}\left(S_{1}\right)$ 和 $\text{ANs}_{q}\left(S_{2}\right)$ 在常数时间内推导出 $\text{ANs}_{q}\left(S_{1} \cup S_{2}\right)$，那么我们称一个问题实例是可分解的（decomposable）。
- 我们定义一个问题实例为 $(B(n), Q(n))$ 光谱可索引（spectrum indexable），如果它对任意数据集 $S \subseteq \mathbb{D}$ 满足以下性质：对于任意整数 $s \in [|S|]$，可以在 $O(|S| \cdot B(s))$ 时间内构建一个关于 $S$ 的数据结构，该数据结构能够在 $O\left(\frac{|S|}{s} \cdot Q(s)\right)$ 时间内回答任意查询。术语“光谱可索引”用来反映该问题实例在参数 $s$ 的整个范围内，根据函数 $B(n)$ 和 $Q(n)$ 的定义，能够构建一个“良好”的索引结构的能力。

---

关于上述定义，有两个重要的观察点：

- $(B(n), Q(n))$ 光谱可索引性意味着我们可以在任意数据集 $S \subseteq \mathbb{D}$ 上用 $O(|S| \cdot B(|S|))$ 时间构建一个数据结构，以在 $O(Q(|S|))$ 时间内回答一个查询（为此，只需将 $s=|S|$ 即可）。
- 考虑任何具有以下性质的可分解问题实例：对于任意数据集 $S \subseteq \mathbb{D}$，我们可以用 $O(|S| \cdot B(|S|))$ 时间构建一个数据结构 $\mathcal{T}$，以在 $O(Q(|S|))$ 时间内回答查询。那么，该问题实例必定是 $(B(n), Q(n))$ 光谱可索引的。要证明这一点，给定一个整数 $s \in [|S|]$，将 $S$ 任意划分为 $m=\lceil|S| / s\rceil$ 个不相交子集 $S_{1}, S_{2}, \ldots, S_{m}$，其中除 $S_{m}$ 外的所有子集大小为 $s$。对于每个 $i \in [m]$，用 $O\left(\left|S_{i}\right| \cdot B(s)\right)$ 时间创建一个结构 $\mathcal{T}\left(S_{i}\right)$；创建所有 $m$ 个结构的总时间为 $O(m \cdot s \cdot B(s))=O(|S| \cdot B(s))$。为了回答一个查询 $q$，只需搜索每个 $\mathcal{T}\left(S_{i}\right)$ 以在 $O(Q(s))$ 时间内得到 $\text{Ans}_{q}\left(S_{i}\right)$，然后在 $O(m)$ 时间内将 $\text{Ans}_{q}\left(S_{1}\right), \text{Ans}_{q}\left(S_{2}\right), \ldots, \text{Ans}_{q}\left(S_{m}\right)$ 合并为 $\text{ANs}_{q}(S)$。因此，总的查询时间为 $O(m \cdot Q(s))$。


### 1.2 Related Work

Motwani 和 Raghavan 在一篇会议论文 [24] 中引入了延迟数据结构（DDS）的概念，该论文后来与 Karp 合著并合并到一篇期刊文章 [19] 中。在 [19] 中，他们为以下 DDS 问题设计了算法：

- **前驱搜索**（Predecessor search）：$S$ 包含 $n$ 个实数值，每个查询给定一个任意值 $q$，并返回 $q$ 在 $S$ 中的前驱。
- **半平面覆盖**（Halfplane containment）：$S$ 包含 $\mathbb{R}^{2}$ 中的 $n$ 个半平面，每个查询给定一个任意点 $q \in \mathbb{R}^{2}$，并返回 $q$ 是否被 $S$ 中的所有半平面覆盖。
- **凸包覆盖**（Convex hull containment）：$S$ 包含 $\mathbb{R}^{2}$ 中的 $n$ 个点，每个查询给定一个任意点 $q \in \mathbb{R}^{2}$，并返回 $q$ 是否被 $S$ 的凸包覆盖。
- **二维线性规划**（2D linear programming）：$S$ 包含 $\mathbb{R}^{2}$ 中的 $n$ 个半平面，每个查询给定一个二维向量 $\boldsymbol{u}$，并返回在所有 $n$ 个半平面的交集中最大化点积 $\boldsymbol{u} \cdot \boldsymbol{p}$ 的点 $p$。
- **正交范围计数**（Orthogonal range counting）：$S$ 包含 $\mathbb{R}^{d}$ 中的 $n$ 个点，维度 $d$ 是一个固定常数，每个查询给定一个任意的 $d$ 维矩形 $q$（即一个轴对齐的形如 $\left[x_{1}, y_{1}\right] \times\left[x_{2}, y_{2}\right] \times \ldots \times\left[x_{d}, y_{d}\right]$ 的盒子），并返回 $S$ 中被 $q$ 覆盖的点的数量。

---

对于前四个问题，Karp、Motwani 和 Raghavan 提出了能够实现 $\text{Time}(n, r) = O(n \log r)$ 的算法，其中 $r \leq n$。对于正交范围计数问题，他们提出了两种算法：第一种算法保证 $\text{Time}(n, r) = O\left(n \log^{d} r\right)$，其中 $r \leq n$；而另一种算法保证 $\text{Time}(n, r) = O\left(n \log n + n \log^{d-1} r\right)$，其中 $r \leq n$。对于所有这些问题，他们证明，在比较模型和/或代数模型下，任何算法的运行时间都必须满足 $\text{Time}(n, r) = \Omega(n \log r)$，其中 $r \in [1, n]$。

---

Aggarwal 和 Raghavan [1] 提出了一个针对最近邻搜索的延迟数据结构（DDS）算法，其运行时间为 $\text{Time}(n, r) = O(n \log r)$，其中 $r \leq n$。在最近邻搜索中，$S$ 包含 $\mathbb{R}^{2}$ 中的 $n$ 个点，每个查询给定一个任意点 $q \in \mathbb{R}^{2}$，返回 $S$ 中距离 $q$ 最近的点。在代数模型下，这一运行时间对于所有 $r \leq n$ 是最优的。

---

关于范围中位数问题（range median）在延迟数据结构（DDS）中的应用，可以讲述一个“成功故事”。在该问题的离线版本中，给定一个包含 $n$ 个实数值的集合 $S$，其存储在一个（未排序的）数组 $A$ 中。对于任意 $1 \leq x \leq y \leq n$，令 $A[x: y]$ 表示元素集合 ${A[x], A[x+1], \ldots, A[y]}$。此外，还给定 $r$ 对整数对 $\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{r}, y_{r}\right)$，其中每对满足 $1 \leq x_{i} \leq y_{i} \leq n$ 且 $i \in [r]$。目标是对每个 $i \in [r]$ 找到集合 $A\left[x_{i}: y_{i}\right]$ 的中位数。在文献 [15] 中，Har-Peled 和 Muthukrishnan 解释了如何在 $O(n \log r + r \log n \cdot \log r)$ 时间内解决这个问题，并证明了在比较模型下的一个下界 $\Omega(n \log r)$，其中 $r \leq n$。在文献 [11]（另见文献 [5]）中，Gfeller 和 Sanders 研究了该问题的 DDS 版本，其中 $S$ 的定义如前，查询则给定任意一对 $(x, y)$，满足 $1 \leq x \leq y \leq n$，并返回 $A[x: y]$ 的中位数。他们设计了一种算法，在所有 $r \leq n$ 的情况下实现了 $\text{Time}(n, r) = O(n \log r)$。显然，任何 DDS 算法都可以用于解决离线问题，并达到 $\text{Time}(n, r)$ 的运行时间。因此，Gfeller 和 Sanders 的算法改进了文献 [15] 的离线解决方案，并且在比较模型下对于 DDS 是最优的。

---

与我们的工作更远相关的是文献 [8]，其中 Ching、Mehlhorn 和 Smid 考虑了一个动态延迟数据结构（DDS）问题。在该问题中，除了需要处理查询外，算法还需支持对数据集 $S$ 的更新。在另一个方向上，Barbay 等人在文献 [2] 中研究了前驱搜索问题的 DDS 版本，但他们的算法分析使用了更细粒度的参数“间隙”（gaps），而不仅仅依赖于 $n$ 和 $r$。这一方向也被扩展到动态 DDS，可以参考最近的研究 [26, 27]。

---

如前所述，DDS 在系统研究领域中以“数据库裂解”（database cracking）的名义得到了广泛研究。此类研究的重点是设计高效的启发式方法，以加速各种实际场景中的查询工作负载，而不是建立强有力的理论保证。有兴趣的读者可以参考文献 [12, 13, 16-18, 23, 31, 32] 作为入门点了解相关文献。

### 1.3 Our Results

我们的主要结果是一个具有以下保证的通用归约：

**定理 1**. 假设 $B(n)$ 和 $Q(n)$ 是非递减函数，并满足 $B(n) = O(\log n)$ 且 $Q(n) = O\left(n^{1-\epsilon}\right)$，其中 $\epsilon > 0$ 是任意小的常数。每个 $(B(n), Q(n))$ 光谱可索引问题都允许一个延迟数据结构（DDS）算法，其具有如下性质：
$$
\begin{equation*}
\operatorname{LoGStreak}(n)=\min \left\{n, \frac{c \cdot n \log n}{Q(n)}\right\} \tag{1}
\end{equation*}
$$

其中 $c$ 是任意大的常数。也就是说，该算法在所有 $r \leq \min \left\{n, \frac{c \cdot n \log n}{Q(n)}\right\}$ 的情况下，实现了 $\text{Time}(n, r) = O(n \cdot \log r)$。

---

在温和的约束条件下，我们在第3节中论证，公式 (1) 中的连贯性界限是最优的。作为一个重要的特殊情况，当 $Q(n) = O(\log n)$ 时，通过我们的归约生成的 DDS 算法实现了 $\text{LogStreak}(n) = n$，即对于所有 $r \leq n$，$\text{Time}(n, r) = O(n \cdot \log r)$。对于许多对数据库系统至关重要的可分解问题，已经存在具有 $O(n \log n)$ 构建时间和 $O(\log n)$ 查询时间的数据结构（例如，前驱搜索和最近邻搜索）。由于此类问题必须是 $(\log n, \log n)$ 光谱可索引的（如第1.1节所述），定理1可以直接生成这些问题的 DDS 版本的优秀解决方案，而这些解决方案通常在比较/代数模型下是最优的。第4.1节将列举这些问题的选集。

---

定理1对数据库裂解（database cracking）来说是个令人欣慰的消息：即使查询时间较慢的数据结构也能在裂解中发挥作用！例如，对于二维空间中的正交范围计数问题（定义见第1.2节），kd树的构建时间为 $O(n \log n)$，其查询时间为 $Q(n) = O(\sqrt{n})$。定理1表明，这种结构可以在 $O(n \log r)$ 时间内回答任何 $r = O(\sqrt{n} \log n)$ 的查询，这在实际数据库裂解场景中可能已经绰绰有余了。这很好地体现了我们在第1.1节中提到的研究“特别高效处理小 $r$ 的 DDS 算法”的动机。

---

定理1对 DDS 算法的设计具有启发意义——我们应研究底层问题的光谱可索引性到底如何。对这一方向的探索可能相当有趣，我们将在第4.2节中以半平面覆盖、凸包覆盖、范围中位数以及二维线性规划（定义见第1.2节）为例进行展示。我们可以证明，对于每一个问题，都存在适当选择的函数 $Q(n)$，使其是 $(\log n, Q(n))$ 光谱可索引的，然后利用定理1以 $\text{Time}(n, r) = O(n \log r)$ 的时间解决这些问题，适用于所有 $r \leq n$。

---

那么，对于构建时间为 $\omega(n \log n)$ 的结构（等价于 $B(n) = \omega(\log n)$）会发生什么呢？第5节将证明，在温和的约束条件下，无法通过通用归约利用此类结构获得 $\text{LogStreak}(n) = \omega(1)$ 的 DDS 算法。换句话说，这些算法只能在 $r = O(1)$ 的平凡情景中实现 $\text{Time}(n, r) = O(n \log r)$。尽管如此，如果接受形式为“$\text{Time}(n, r) \leq n \log^{O(1)} r$，其中 $r \leq n$”的保证，我们的归约（基于定理1）可以扩展以生成这样的算法，只要 $B(n)$ 和 $Q(n)$ 是 $\log^{O(1)} n$ 的函数，如第5节所讨论。

## 2 Warm Up: Predecessor Search

前驱搜索是延迟数据结构（DDS）和数据库裂解社区中最受关注的问题。本节将回顾文献 [19] 中针对该问题实现 $\text{Time}(n, r) = O(n \log r)$ 的两种方法。这些方法构成了几乎所有现有 DDS 算法的基础。

---

**自底向上的方法（Bottom-Up）**。回忆一下，数据集 $S$ 包含 $n$ 个实数值。我们不失一般性地假设 $n$ 是 2 的幂。在任意时刻，集合 $S$ 被任意划分为称为运行（runs）的不相交子集，每个运行的大小为 $s = 2^{i}$（其中 $i \geq 0$）。每个运行是有序的，并存储在一个数组中。初始时，$s = 1$，即每个运行包含 $S$ 的一个单独元素。随着时间推移，运行的大小 $s$ 单调增加。当需要将运行大小从 $2^{i}$ 增加到 $2^{j}$（其中 $j > i$）时，算法会进行一次全面调整（overhaul），以生成新的运行。一个大小为 $2^{j}$ 的运行可以通过在 $O\left(2^{j} \cdot (j-i)\right)$ 时间内合并 $2^{j-i}$ 个大小为 $2^{i}$ 的运行获得，因此全面调整的时间复杂度为 $O(n \cdot (j-i))$。因此，如果当前运行大小为 $s$，生成历史上所有运行的总成本为 $O(n \log s)$。

---

一个前驱查询只需通过对每个运行进行二分搜索即可完成。然而，为了控制成本，算法确保在处理第 $i$ 个查询（$i \geq 1$）之前，当前运行大小 $s$ 至少为 $i \log i$。如果不满足这一条件，则触发一次全面调整，将 $s$ 增加到至少 $i \log i$ 的最接近的 2 的幂。此后，查询的成本为 $O\left(\frac{n}{s} \log s\right) = O\left(\frac{n}{i \log i} \cdot \log (i \log i)\right) = O(n / i)$。将这些成本累加到 $r$ 次查询，总成本为 $O\left(n \sum_{i=1}^{r} \frac{1}{i}\right) = O(n \log r)$。由于最终的运行大小 $s$ 为 $O(r \log r)$，可以得出结论，该算法在 $O(n \log r)$ 时间内处理了 $r$ 次查询。需要注意的是，这只在 $r \log r \leq n$ 时成立（最大运行大小为 $n$）。然而，当 $r$ 达到 $\lceil n / \log n \rceil$ 时，算法可以用 $O(n \log n) = O(n \log r)$ 时间对整个 $S$ 进行排序，并在之后的每次查询中以 $O(\log n) = O(\log r)$ 的时间完成。因此，该算法在所有 $r \leq n$ 的情况下实现了 $\text{Time}(n, r) = O(n \log r)$。

---

**自顶向下的方法（Top-Down）**。此方法模仿了在 $S$ 上构建二叉搜索树（BST）$\mathcal{T}$ 的以下策略：(i) 找到 $S$ 的中位数，将 $S$ 按中位数分为 $S_{1}$ 和 $S_{2}$；(ii) 将中位数存储为根节点的键值，然后分别在 $S_{1}$（对应左子树）和 $S_{2}$（对应右子树）上递归构建左右子树。与直接完全构建不同，该算法在查询处理过程中根据需求逐步构建 $\mathcal{T}$。

---

一开始，只有根节点存在且处于“未展开”模式。通常，一个未展开节点 $u$ 还没有子节点，但与子集 $S_{u} \subseteq S$ 相关联，子集中包含应存储在其子树中的元素。查询的处理方式与普通 BST 类似，通过遍历 $\mathcal{T}$ 的根到叶路径 $\pi$ 来回答查询。主要区别在于，当搜索到 $\pi$ 上的一个未展开节点 $u$ 时，算法必须首先展开它。展开 $u$ 的过程包括：为 $u$ 创建两个子节点，根据 $u$ 的键值将 $S_{u}$ 按中位数分成两部分，并将每部分与一个子节点关联。之后，$u$ 被标记为已展开，其子节点处于未展开模式。展开操作的时间复杂度为 $O\left(\left|S_{u}\right|\right)$（根据 [4]，找到集合的中位数需要线性时间）。

---

经过 $r$ 次查询后，BST $\mathcal{T}$ 是部分构建的，因为仅有查询过程中遍历的 $r$ 条根到叶路径上的节点被展开。前 $\log r$ 层的节点的总展开成本为 $O(n \log r)$。对于每条根到叶路径 $\pi$，第 $\log r$ 层的节点 $u$ 的展开成本为 $O(n / r)$，这主导了 $\pi$ 上 $u$ 的所有后代节点的总展开成本。因此，除了前 $\log r$ 层的节点，$\mathcal{T}$ 中所有其他节点的总展开成本为 $r \cdot O\left(\frac{n}{r}\right) = O(n)$。因此，该算法在所有 $r \leq n$ 的情况下实现了 $\text{Time}(n, r) = O(n \log r)$。

## 3 Reductions from Data Structures to DDS Algorithms

第3.1节将扩展上一节回顾的自底向上方法，形成一种通用归约。只要满足一个关键要求——线性可合并性（linear mergeability，将在稍后定义），该归约即可生成具有较大 $\log(r)$ 连贯性界限的 DDS 算法。尽管该归约将在定理1所述的最终归约（第3.2节中介绍）中被取代，但其讨论具有以下意义：(i) 加深读者对该方法优势与局限性的理解；(ii) 阐明在缺乏线性可合并性时引入新思想的必要性。最后，第3.3节将确立一个难度结果，证明定理1中的连贯性界限已无法显著改进。

### 3.1 The First Reduction: Generalizing the Bottom-Up Approach

本小节将聚焦于可分解问题。对于任意数据集 $S \subseteq \mathbb{D}$，我们假设存在一个数据结构 $\mathcal{T}(S)$，能够在 $O(Q(n))$ 时间内回答任意查询。此外，该结构是线性可合并的，即对于任意不相交的 $S_{1}, S_{2} \subseteq \mathbb{D}$，可以在 $O\left(\left|S_{1}\right|+\left|S_{2}\right|\right)$ 时间内通过 $\mathcal{T}\left(S_{1}\right)$ 和 $\mathcal{T}\left(S_{2}\right)$ 构建出 $S_{1} \cup S_{2}$ 的数据结构。这意味着 $\mathcal{T}(S)$ 可以在 $O(n \log n)$ 时间内构建。

---

**引理2（Lemma 2）。** 对于存在线性可合并结构且查询时间为 $Q(n)=O\left(n^{1-\epsilon}\right)$（其中 $\epsilon>0$ 是一个常数）的可分解问题，我们可以设计一个 DDS 算法以保证 $\text{LogStreak}(n)=\min \left\{n, \frac{c \cdot n \log n}{Q(n)}\right\}$，其中 $c$ 是任意大的常数。

---

**证明（Proof）。** 我们不失一般性地假设 $n$ 是 2 的幂。如同自底向上的方法一样，在任意时刻，我们将 $S$ 任意划分为运行（runs），每个运行的大小为 $s = 2^{i}$（其中 $i \geq 0$）。对于每个运行，构建其内部元素的数据结构。初始时，运行大小 $s = 1$。每次将 $s$ 从 $2^{i}$ 增长到某个值 $2^{j}$（其中 $j > i$）时，会进行一次全面调整（overhaul）以构建大小为 $2^{j}$ 的运行。根据线性可合并性，一个大小为 $2^{j}$ 的运行的数据结构可以通过合并 $2^{j-i}$ 个大小为 $2^{i}$ 的运行的数据结构在 $O\left(2^{j} \cdot (j-i)\right)$ 时间内构建完成。通过与第2节类似的分析，如果当前运行大小为 $s$，生成历史上所有出现过的运行的数据结构的总成本为 $O(n \log s)$。

---

对于一个谓词 $q$ 的查询，通过在每个运行的数据结构中搜索并将所有运行的结果合并为 $\text{Ans}_{q}(S)$ 来完成查询。查询成本为 $O\left(\frac{n}{s} \cdot Q(s)\right)$。我们要求，在回答第 $i$ 个查询（$i \geq 1$）之前，运行大小 $s$ 必须满足：
$$
\frac{Q(s)}{s} \leq \frac{1}{i} \tag{2}
$$
如果不满足此条件，我们会启动一次全面调整，将 $s$ 增加到满足条件 (2) 的最小的 2 的幂。这样可以保证第 $i$ 个查询的成本为 $O(n / i)$。因此，处理 $r$ 次查询的总成本为 $O(n \log r)$。

---

接下来需要估计全面调整的成本。根据条件 (2)，最终运行大小 $s$ 必须满足以下条件中最小的 2 的幂：

- $s \leq n$（运行大小不能超过 $n$），以及
- $s / Q(s) \geq r$（由条件 (2) 导出）。

由于 $Q(s) = O\left(s^{1-\epsilon}\right)$，我们知道 $Q(s) \leq \alpha \cdot s^{1-\epsilon}$，其中 $\alpha > 0$ 是某个常数。因此，$s / Q(s) \geq s^{\epsilon} / \alpha$。为了找到 $s$ 的上界，我们需要 $s$ 满足 $s^{\epsilon} / \alpha \geq r$（这比 $s / Q(s) \geq r$ 更严格），即 $s \geq (\alpha \cdot r)^{1 / \epsilon}$。因此，如果 $(\alpha \cdot r)^{1 / \epsilon} \leq n$，我们可以得出 $s = O\left(r^{1 / \epsilon}\right)$，从而 $O(\log s) = O(\log r)$。在这种情况下，所有的全面调整总共需要 $O(n \log r)$ 时间。

---

上述策略在 $(\alpha \cdot r)^{1 / \epsilon} > n$ 的情况下不再适用。然而，在这种情况下，$r > n^{\epsilon} / \alpha$，也就是说，$r$ 已经是 $n$ 的一个多项式函数。这启发了以下的暴力策略：当 $r$ 达到 $\left\lceil n^{\epsilon} / \alpha\right\rceil$ 时（我们称之为临界点 snapping point），直接在整个 $S$ 上构建一个数据结构，耗时 $O(n \log n) = O(n \log r)$，并使用该数据结构来回答之后的每次查询，每次查询耗时 $O(Q(n))$，直到 $r = \min \left\{n, \frac{c \cdot n \log n}{Q(n)}\right\}$。在临界点之后的所有查询总成本为 $O(n \log n) = O(n \log r)$。因此，我们得到了一个算法，它能够保证对所有 $r \leq \min \left\{n, \frac{c \cdot n \log n}{Q(n)}\right\}$ 满足 $\text{Time}(n, r) = O(n \log r)$。

---

上述归约关键依赖于结构 $\mathcal{T}$ 的线性可合并性。如果没有这一特性，构建大小为 $2^{i}$ 的运行的全面调整成本将变为 $\Omega\left(n \cdot \log \left(2^{i}\right)\right)$，此时所有全面调整的总成本将达到 $\Omega\left(n \cdot \log^{2} r\right)$。接下来，我们将介绍另一种归约方法，可以削减一个 $\log r$ 因子。

### 3.2 The Second Reduction: No Linear Mergeability

我们现在将在第3.1节中取消对线性可合并性的要求，并确立定理1。回忆一下，基础问题是 $(B(n), Q(n))$ 光谱可索引的，其中 $B(n) = O(\log n)$ 且 $Q(n) = O\left(n^{1-\epsilon}\right)$，$\epsilon > 0$ 是某个常数。目标是设计一个算法，在所有 $r \leq \min \left\{n, \frac{c \cdot n \log n}{Q(n)}\right\}$ 的情况下满足 $\text{Time}(n, r) = O(n \log r)$，其中 $c > 0$ 是任意常数。

---

假设 $n$ 是 2 的幂（不失一般性）。我们的算法分为若干阶段执行。在第 $i$ 阶段（$i \geq 1$）开始时，我们设定 $s = 2^{2^{i}}$ 并构建一个数据结构 $\mathcal{T}$——根据 $(B(n), Q(n))$ 光谱可索引性，在 $S$ 上以 $O(n \cdot B(s))$ 时间构建的数据结构。该结构能够在 $O\left(\frac{n}{s} \cdot Q(s)\right)$ 时间内回答任意查询。第 $i$ 阶段在回答 $\lceil s / Q(s) \rceil$ 次查询后结束。这些查询的总成本为：

$$
\left\lceil\frac{s}{Q(s)}\right\rceil \cdot O\left(\frac{n}{s} \cdot Q(s)\right)=O(n)
$$

---

- 从上述分析可以看出，第 $i$ 阶段的总计算时间为 $O(n \cdot B(s)) = O\left(n \cdot 2^{i}\right)$。由于 $n \cdot 2^{i}$ 在 $i$ 增加 1 时会翻倍，回答 $r$ 次查询的总成本为 $O\left(n \cdot 2^{h^{*}}\right)$，其中 $h^{*}$ 是所需的阶段数。具体而言，$h^{*}$ 的值是满足以下两个条件的最小整数 $h \geq 1$：
  - 条件 C1：$2^{2^{h}} \leq n$（$s$ 的值不能超过 $n$）；
  - 条件 C2：$\sum_{i=1}^{h}\left\lceil\frac{2^{2^{i}}}{Q\left(2^{2^{i}}\right)}\right\rceil \geq r$（$h$ 个阶段能够回答的查询数必须至少为 $r$）。

---

我们的目标是找到 $h$ 的上界，为此，我们用更严格的条件替代条件 $\mathbf{C2}$：$2^{2^{h}} / Q\left(2^{2^{h}}\right) \geq r$。由于 $Q(n) = O\left(n^{1-\epsilon}\right)$，我们知道 $Q(n) \leq \alpha \cdot n^{1-\epsilon}$，其中 $\alpha > 0$ 是某个常数。因此，我们将条件 $\mathbf{C2}$ 进一步改写为更严格的不等式：

$$
\begin{equation*}
\frac{2^{2^{h}}}{\alpha \cdot\left(2^{2^{h}}\right)^{1-\epsilon}}=\frac{\left(2^{2^{h}}\right)^{\epsilon}}{\alpha} \geq r \quad \Leftrightarrow \quad 2^{2^{h}} \geq(\alpha \cdot r)^{1 / \epsilon} \tag{3}
\end{equation*}
$$

令 $H$ 表示满足 (3) 的最小整数 $h \geq 1$，即：

$$
\begin{equation*}
2^{2^{H-1}}<(\alpha \cdot r)^{1 / \epsilon} \Leftrightarrow 2^{2^{H}}<(\alpha \cdot r)^{2 / \epsilon} \tag{4}
\end{equation*}
$$

当 $2^{2^{H}} \leq n$ 时，上述推导保证了所需阶段数 $h^{*}$ 最多为 $H$（不等式 $2^{2^{H}} \leq n$ 意味着 $2^{2^{h^{*}}} \leq n$，从而满足条件 $\mathbf{C1}$）。在这种情况下，所有阶段的总成本为 $O\left(n \cdot 2^{h^{*}}\right) = O\left(n \cdot 2^{H}\right)$，根据 (4)，这是 $O(n \log r)$。

---

上述推导在 $2^{2^{H}} > n$ 的情况下不再适用。然而，当这种情况发生时，根据 (4)，我们可以得出 $(\alpha \cdot r)^{2 / \epsilon} > 2^{2^{H}} > n$，从而得到 $r > n^{\epsilon / 2} / \alpha$。一旦 $r$ 达到 $\left\lceil n^{\epsilon / 2} / \alpha\right\rceil$（称为临界点 snapping point），我们就直接在整个 $S$ 上构建一个数据结构 $\mathcal{T}$，耗时 $O(n \log n) = O(n \log r)$，并使用该结构回答之后的所有查询，每次查询耗时 $O(Q(n))$，直到 $r=\min \left\{n, \frac{c \cdot n \log n}{Q(n)}\right\}$。在临界点之后的查询总成本为 $O(n \log n) = O(n \log r)$。因此，我们得到了一个算法，能够保证对所有 $r \leq \min \left\{n, \frac{c \cdot n \log n}{Q(n)}\right\}$ 满足 $\text{Time}(n, r) = O(n \log r)$，从而完成了定理1的证明。

### 3.3 Tightness of the Streak Bound

本节将解释为什么定理1中的连贯性界限 $\Omega\left(\frac{n \log n}{Q(n)}\right)$ 对于具有“合理”行为的归约算法来说在渐近上是最优的。

---

**对具有受限结构的可分解问题的黑箱归约（Black-box Reductions on Decomposable Problems with Restricted Structures）。** 为了证明难度结果，我们可以任意专门化问题类别，因此我们仅考虑可分解问题。归约算法 $\mathcal{A}$ 需要能够处理任意 $(B(n), Q(n))$ 光谱可索引的可分解问题。如第1.1节所述，这意味着存在一个数据结构 $\mathcal{T}$，可以在任何 $S \subseteq \mathbb{D}$ 上以 $O(|S| \cdot B(|S|))$ 时间构建，并能够以 $O(Q(|S|))$ 时间在 $S$ 上回答任意查询。

---

只要 $(B(n), Q(n))$ 光谱可索引性被保留，我们可以限制 $\mathcal{T}$ 的功能性，使得归约算法 $\mathcal{A}$ 的实现变得困难。具体而言，$\mathcal{T}$ 仅向 $\mathcal{A}$ 提供以下“服务”：

- $\mathcal{A}$ 可以在任意子集 $S^{\prime} \subseteq S$ 上创建一个结构 $\mathcal{T}\left(S^{\prime}\right)$；
- 给定一个谓词 $q \in \mathbb{Q}$，$\mathcal{A}$ 可以使用 $\mathcal{T}\left(S^{\prime}\right)$ 在 $S^{\prime}$ 上找到答案 $\text{Ans}_{q}\left(S^{\prime}\right)$；
- 给定一个谓词 $q$，在 $\mathcal{A}$ 已经获得了关于不相交子集 $S_{1}^{\prime}, S_{2}^{\prime} \subseteq S$ 的 $\text{Ans}_{q}\left(S_{1}^{\prime}\right)$ 和 $\text{Ans}_{q}\left(S_{2}^{\prime}\right)$ 后，可以在常数时间内将这些答案合并为 $\text{Ans}_{q}\left(S_{1}^{\prime} \cup S_{2}^{\prime}\right)$（合并算法由 $\mathcal{T}$ 提供）。

---

尽管功能有限，$\mathcal{T}$ 仍然使得问题 $(B(n), Q(n))$ 光谱可索引，因为该问题是可分解的；参见第1.1节中的解释。

---

到目前为止，我们没有对算法 $\mathcal{A}$ 的行为进行任何限制，但现在我们准备对其进行限制。为了回答一个查询，算法 $\mathcal{A}$ 需要搜索一系列子集（包括零个）上的结构，假设为 $\mathcal{T}\left(S_{1}^{\prime}\right), \mathcal{T}\left(S_{2}^{\prime}\right), \ldots, \mathcal{T}\left(S_{t}^{\prime}\right)$。算法可以选择任意 $t \geq 0$ 和任意的 $S_{1}^{\prime}, S_{2}^{\prime}, \ldots, S_{t}^{\prime}$（它们不需要是互不相交的）。此外，$\mathcal{A}$ 还允许检查另一个子集 $S_{\text{scan}}^{\prime} \subseteq S$，并支付 $\Omega\left(\left|S_{\text{scan}}^{\prime}\right|\right)$ 的时间。所有这些操作后，算法必须确保：

$$
\begin{equation*}
S_{\text {scan }}^{\prime} \cup S_{1}^{\prime} \cup S_{2}^{\prime} \cup \ldots \cup S_{t}^{\prime}=S \tag{5}
\end{equation*}
$$

上述约束是自然的，因为否则 $S_{\text{scan}}^{\prime} \cup S_{1}^{\prime} \cup S_{2}^{\prime} \cup \ldots \cup S_{t}^{\prime}$ 中至少缺少 $S$ 的一个元素。如果算法 $\mathcal{A}$ “敢于”在这种情况下返回查询答案，它必须已经获得了底层问题的某些特殊属性。在本研究中，我们关注的是通用归约，这些归约不依赖于问题特定的属性。

---

我们将遵循上述要求的归约算法称为黑箱归约。引理2和定理1中的算法属于黑箱类。

---

**定理1的紧密性。** 我们将证明，任何黑箱归约在 $O(n \log r)$ 时间内最多只能回答 $r = O\left(\frac{n \log n}{Q(n)}\right)$ 个查询，对于任意满足以下条件的函数 $Q(n): \mathbb{N}^{+} \rightarrow \mathbb{N}^{+}$：

- 满足 $Q(n) = O(n)$，并且 $Q(n) = \Theta(Q(c \cdot n))$ 对于任意常数 $c > 0$；
- 是次加法的，即 $Q(x + y) \leq Q(x) + Q(y)$ 对于任何 $x, y \geq 1$ 成立。

这将确认定理1在黑箱类中的紧密性。

---

我们将构造一个可分解的问题及其相应的数据结构，这些在现实中毫无意义，但在数学上却是合理的。数据集 $S$ 由 $n$ 个任意元素组成；给定任何谓词，对 $S$ 的查询始终返回 $|S|$（元素和谓词的具体形式无关）。每当被要求“构建”一个数据结构时，我们故意浪费 $|S| \log |S|$ 的时间，然后简单地返回 $S$ 的一个任意排列（存储在数组中）。每当被要求“回答”一个查询时，我们故意浪费 $Q(|S|)$ 时间，然后返回 $|S|$。显然，这个问题是可分解的。

---

我们认为，任何黑箱归约算法 $\mathcal{A}$ 在回答每个查询时都需要 $\Omega(Q(n))$ 时间。考虑一个任意查询，假设 $\mathcal{A}$ 通过搜索结构 $\mathcal{T}\left(S_{1}^{\prime}\right), \ldots, \mathcal{T}\left(S_{t}^{\prime}\right)$（对于某个 $t \geq 0$）并扫描 $S_{\text{scan}}^{\prime}$ 来处理该查询。根据我们结构的设计，查询的成本至少为：
$$
\begin{aligned}
\Omega\left(\left|S_{\mathrm{scan}}^{\prime}\right|\right) & +\sum_{i \in[t]} Q\left(\left|S_i^{\prime}\right|\right) \geq \Omega\left(\left|S_{\mathrm{scan}}^{\prime}\right|\right)+Q\left(\sum_{i \in[t]}\left|S_i^{\prime}\right|\right) \quad \text { (by sub-additivity) } \\
& =\Omega\left(Q\left(\left|S_{\mathrm{scan}}^{\prime}\right|\right)+Q\left(\sum_{i \in[t]}\left|S_i^{\prime}\right|\right)\right) \quad(\text { by } Q(n)=O(n), Q(n)=\Theta(Q(c n))) \\
& =\Omega\left(Q\left(\left|S_{\mathrm{scan}}^{\prime}\right|+\sum_{i \in[t]}\left|S_i^{\prime}\right|\right)\right) \quad \text { (by sub-additivity) } \\
& =\Omega(Q(n)) . \quad(\text { by }(5))
\end{aligned}
$$

---

根据 $\log(r)$-streak 的定义，算法 $\mathcal{A}$ 必须在总成本为 $O(n \log r)$ 的情况下处理 $r = \text{LogStreak}(n)$ 个查询，而显然该成本不能超过 $O(n \log n)$（记住 $r \leq n$）。因此可以推断，$\mathcal{A}$ 最多只能处理 $O\left(\frac{n \log n}{Q(n)}\right)$ 个查询。

## 4 New DDS Algorithms for Concrete Problems

我们现在利用定理 1 来为具体的 DDS 问题开发算法，重点讨论第 4.1 节中的可分解问题和第 4.2 节中的非可分解问题。

### 4.1 Applications to Decomposable Problems

如前所述，如果一个可分解问题具有一个数据结构 $\mathcal{T}$，可以在 $O(n \log n)$ 时间内构建（即 $B(n)=O(\log n)$），并且支持在 $Q(n)=O(\log n)$ 时间内查询，那么定理 1 直接为该问题提供了一个 DDS 算法，保证其时间复杂度为 $\text{TimE}(n, r)=O(n \log r)$，适用于所有 $r \leq n$。下面列举了一个部分问题列表，这些问题对数据库系统具有重要性，而之前并没有已知的具有相同保证的算法。

- **二维正交范围计数**。参见第 1.2 节中的问题定义。结构 $\mathcal{T}$ 可以是一个持久的“聚合”二叉搜索树（BST）[30]。
- **矩形上的正交范围计数**。数据集 $S$ 是 $n$ 个二维矩形（即轴平行的矩形）在 $\mathbb{R}^2$ 中的集合。给定一个任意的二维矩形 $q$，查询返回与 $q$ 相交的 $S$ 中矩形的数量。该问题可以简化为对前述问题（正交范围计数点问题）的四个查询[33]。结构 $\mathcal{T}$ 再次可以是一个持久的聚合 BST [30]。
- **点位置问题**。数据集 $S$ 是由 $n$ 条线段定义的平面划分，每条线段与其相邻的两个面的 id 相关联。给定一个任意的点 $q \in \mathbb{R}^2$，查询返回包含点 $q$ 的面，实际上就是找到位于 $q$ 上方的线段（即从 $q$ 向上射出的射线“首先碰到”的线段）。结构 $\mathcal{T}$ 可以是一个持久的 BST [28] 或 Kirkpatrick 结构 [20]。
- **$k=O(1)$ 最近邻搜索（二维）**。数据集 $S$ 是 $\mathbb{R}^2$ 中的 $n$ 个点。固定一个常数整数 $k \geq 1$。给定一个点 $q \in \mathbb{R}^2$，查询返回 $S$ 中距离 $q$ 最近的 $k$ 个点。结构 $\mathcal{T}$ 可以是建立在 order-$k$ Voronoi 图上的点位置结构 [20,28]（order-$k$ Voronoi 图可以在 $O(n \log n)$ 时间内计算得到 [6]）。对于 $k=1$，已经找到了一个 DDS 算法，其 $\text{TimE}(n, r)=O(n \log r)$，适用于所有 $r \leq n$ [1]。然而，文献 [1] 的算法严重依赖于能够在线性时间内合并两个（order-1）Voronoi 图，因此无法轻易扩展到更大的 $k$ 值。
- **度量空间中的近似最近邻搜索**。数据集 $S$ 包含 $n$ 个在具有常数倍增维度的度量空间中的对象（这包括任何具有常数维度的欧几里得空间）。设 $\text{dist}\left(o_1, o_2\right)$ 表示空间中两个对象 $o_1$ 和 $o_2$ 之间的距离。给定空间中的任意对象 $q$，查询返回一个对象 $o \in S$，使得对所有 $o' \in S$，$\text{dist}(o, q) \leq (1+\epsilon) \cdot \text{dist}\left(o', q\right)$，其中 $\epsilon$ 是一个常数。为实现我们的目的，可以在 [14, 22] 中找到相应的结构 $\mathcal{T}$。

---

对于 $\mathbb{R}^d$ 中的正交范围计数问题，其中 $d \geq 3$ 为固定常数（如第 1.2 节所定义），可以应用定理 1 来获得一个稍显不寻常的结果。已有研究[3]表明，可以在 $O(n \log n)$ 时间内构建一个数据结构 $\mathcal{T}$，该结构能够在 $Q(n)=O\left(n^{\epsilon}\right)$ 时间内回答查询，其中常数 $\epsilon>0$ 可以选择得非常小。因此，定理 1 提供了一个 DDS 算法，其时间复杂度为 $\text{Time}(n, r)=O(n \log r)$，适用于所有 $r \leq n^{1-\epsilon}$。由于 $\epsilon$ 可以非常接近 0，$\log (r)$-streak 上界 $\log \text{Streak}(n)=n^{1-\epsilon}$ 比最大值 $n$ 小的程度仅为一个次多项式因子。如果能够为所有常数维度解决此间隙问题，值得做出一个重要的备注。如果能够发现一个 DDS 算法，其 $\text{LogStreak}(n)=n$，那么该算法也将解决以下离线版本问题，并且时间复杂度为 $O(n \log n)$：给定一个包含 $n$ 个点的集合 $P$ 和一个包含 $n$ 个 $d$-矩形的集合 $Q$，目标是报告每个 $d$-矩形 $q \in Q$ 中覆盖的点的数量。这个离线问题已经被广泛研究，但据我们所知，最速的算法仍然需要 $O(n \log^{\Theta(d)} n)$ 时间。

### 4.2 Non-Decomposable but Spectrum-Indexable Problems

本小节将利用定理 1 来处理一些不可分解的问题（至少在表面上看不出来是不可分解的）。关键是要证明该问题对于合适的 $Q(n)$ 是 $(\log n, Q(n))$ 谱索引可索引的。这本身是一个有趣的话题，接下来我们将通过开发新的 DDS 算法，针对半平面包含、凸包包含、范围中位数和二维线性规划问题，提供 $\text{TimE}(n, r)=O(n \log r)$ 的算法，适用于所有 $r \leq n$。这些问题的原始算法 $[11,19]$ 都是采用第 2 节中回顾的自顶向下方法设计的。我们的算法与这些算法形成对比，展示了定理 1 如何促进 DDS 算法的设计。

---

**半平面包含问题。** 通过使用几何对偶性 [7]，该问题可以转换为以下等价形式 [19]：

- 通过凸包的直线。给定 $\mathbb{R}^{2}$ 中的 $n$ 个点集合 $S$，对于任何一条直线 $l$，查询确定 $l$ 是否与 $S$ 的凸包 $\mathrm{CH}(S)$ 相交。

我们将集中讨论上述问题。

---

假设现在我们给定了一个点 $q$，它位于直线 $l$ 上，且在 $\mathrm{CH}(S)$ 之外。从点 $q$ 出发，我们可以射出两条切线射线，每条射线都与 $\mathrm{CH}(S)$ 相切，但不进入 $\mathrm{CH}(S)$ 的内部。在图 1(a) 中，$S$ 是黑色点的集合，第一条射线穿过点 $p_{1} \in S$，第二条射线穿过点 $p_{2} \in S$。这两条射线形成一个“楔形”区域，将 $\mathrm{CH}(S)$ 包围在其中（注意楔形的角度小于 $180^{\circ}$）；我们将其称为点 $q$ 在 $\mathrm{CH}(S)$ 上的“楔形”。如果直线 $l$ 穿过 $\mathrm{CH}(S)$，则当且仅当 $l$ 穿过这个楔形区域。如果 $\mathrm{CH}(S)$ 的顶点按顺时针顺序存储，那么这两条切线射线可以在 $O(\log n)$ 时间内找到 [25]。

---

我们将证明“通过凸包的直线”问题是 $(\log n, \log n)$ 谱索引可索引的。取任意一个整数 $s \in[1, n]$ 并设 $m=\lceil n / s\rceil$。将 $S$ 随意分成 $S_{1}, S_{2}, \ldots, S_{m}$，使得对于 $i \in [m-1]$，$\left|S_{i}\right|=s$，并且 $\left|S_{m}\right|=n-s(m-1)$。为了构建结构 $\mathcal{T}$，我们用 $O(s \log s)$ 时间计算每个 $S_{i}$ 的凸包 $\mathrm{CH}(S_{i})$，并按顺时针顺序存储其顶点。该结构的构建时间是 $O(n \log s)$。现在我们来看如何回答一个查询，给定直线 $l$。假设再次给定一个位于 $l$ 上且在 $\mathrm{CH}(S)$ 之外的点 $q$。对于每个 $i \in [m]$，我们计算 $q$ 在 $\mathrm{CH}(S_{i})$ 上的楔形，所需时间为 $O(\log s)$。从这 $m$ 个楔形中，我们可以简单地在 $O(m)$ 时间内获得 $q$ 在 $\mathrm{CH}(S)$ 上的楔形（我们将在稍后讨论“凸包包含”问题时处理一个更一般的问题）。现在，可以很容易地确定 $l$ 是否与 $\mathrm{CH}(S)$ 相交。目前查询的时间为 $O(m \log s)$。

---

剩下的任务是解释如何找到 $q$。如果我们已经知道 $S$ 的最小轴对齐外包矩形 $\text{MBR}(S)$，则可以在 $O(1)$ 时间内找到 $q$。注意，$\text{MBR}(S)$ 必须包含 $\mathrm{CH}(S)$；见图 1(b)。显然，$\text{MBR}(S)$ 可以在 $O(m)$ 时间内从 $\text{MBR}(S_{1}), \text{MBR}(S_{2}), \ldots, \text{MBR}(S_{m})$ 中获得，而每个 $\text{MBR}(S_{i})(i \in[m])$ 可以在构建结构 $\mathcal{T}$ 时以 $O(s)$ 时间计算。因此，我们得出结论，该问题是 $(\log n, \log n)$ 谱索引可索引的，并且可以应用定理 1。

---

**凸包包含问题。** 在这个问题中，$S$ 是 $\mathbb{R}^{2}$ 中的一组 $n$ 个点。给定任意一点 $q \in \mathbb{R}^{2}$，查询确定点 $q$ 是否被 $\mathrm{CH}(S)$ 包含。我们将证明该问题是 $(\log n, \log n)$ 谱索引可索引的。

---

取任意一个整数 $s \in[1, n]$，并设 $m=\lceil n / s\rceil$。将 $S$ 随意分成 $S_{1}, S_{2}, \ldots, S_{m}$，使得对于 $i \in [m-1]$，$\left|S_{i}\right|=s$，并且 $\left|S_{m}\right|=n-s(m-1)$。为了构建结构 $\mathcal{T}$，我们计算每个 $S_{i}$ 的凸包 $\mathrm{CH}(S_{i})$，并按顺时针顺序存储其顶点；这将花费 $O(n \log s)$ 时间，如前所述。现在来看如何回答给定点 $q$ 的查询。对于每个 $i \in [m]$，我们可以在 $O(\log s)$ 时间内检查 $q$ 是否被 $\mathrm{CH}(S_{i})$ 包含 [25]。如果对于某个 $i$ 答案是“是”，则点 $q$ 必定被 $\mathrm{CH}(S)$ 包含，查询结束。接下来的讨论假设 $q$ 对于所有 $i$ 都不在 $\mathrm{CH}(S_{i})$ 内。我们将在 $O(m \log s)$ 时间内，计算每个 $i \in [m]$ 上的 $q$ 在 $\mathrm{CH}(S_{i})$ 上的楔形，如前一个问题中所述。

---

接下来，我们需要从这 $m$ 个楔形中确定 $q$ 是否被 $\mathrm{CH}(S)$ 包含。这可以重新建模为以下问题。以点 $q$ 为圆心，画一个任意半径的圆。对于每个楔形，其两条边界射线与圆相交，形成小于 $180^{\circ}$ 的弧。设这些弧为 $a_{1}, a_{2}, \ldots, a_{m}$，并定义 $a^{*}$ 为覆盖所有这些弧的最小圆弧。图 1(c) 显示了一个例子，其中 $m=3$。弧 $a_{1}$ 由点 A 和 B 所对，弧 $a_{2}$ 由点 C 和 D 所对，弧 $a_{3}$ 由点 E 和 F 所对。这里，最小的弧 $a^{*}$ 顺时针从点 A 到 F。关键是，$q$ 是否被 $\mathrm{CH}(S)$ 包含，当且仅当 $a^{*}$ 跨越至少 $180^{\circ}$ 时（这就是图 1(c) 中的情况）——注意，如果 $q$ 在 $\mathrm{CH}(S)$ 外部，则 $a^{*}$ 必须由 $q$ 在 $\mathrm{CH}(S)$ 上的楔形所对，且该弧小于 $180^{\circ}$。因此，目标是确定 $a^{*}$ 是否至少为 $180^{\circ}$。

---

我们可以在 $O(m)$ 时间内实现这一目标（即使这 $m$ 个弧是以任意顺序给出的）。为此，我们逐个处理这些弧，保持当前覆盖已处理弧的最小弧 $a^{*}$，并在发现 $a^{*}$ 已经至少为 $180^{\circ}$ 时停止算法。具体地，对于 $i=1$，直接将 $a^{*}$ 设为 $a_{1}$。然后，依次给定下一个弧 $a_{i}(i \geq 2)$，常数时间内检查是否有小于 $180^{\circ}$ 的弧可以覆盖 $a_{i}$ 和 $a^{*}$。如果可以，则将 $a^{*}$ 更新为该弧；否则，停止算法。例如，在图 1(c) 中，当处理完 $a_{2}$ 后，我们保持的 $a^{*}$ 从 A 到 D 顺时针。当处理 $a_{3}$ 时，算法意识到 $a^{*}$ 必须至少为 $180^{\circ}$，因此终止。

---

我们因此得出结论，凸包包含问题是 $(\log n, \log n)$ 谱索引可索引的，并且可以应用定理 1。

**区间中位数问题。** 在这个问题中，$S$ 是存储在数组 $A$ 中的 $n$ 个实数值。给定一个任意的整数对 $(x, y)$，满足 $1 \leq x \leq y \leq n$，查询返回数组 $A[x: y]$ 的中位数。我们将证明该问题是 $(\log n, \log n)$ 谱索引可索引的。设定任意整数 $s \leq n$，并令 $m=\lceil n / s\rceil$。对于每个 $i \leq m-1$，定义 $S_{i}=A[(i-1) s+1: i \cdot s]$，并对于 $S_{m}=A[(m-1) s+1: n]$。接下来，我们假设 $s \leq \sqrt{n}$；否则，直接使用 $O(n \log n)=O(n \log s)$ 时间在整个 $S$ 上构建一个 [11] 中的结构，该结构能够在 $O(\log n)=O(\log s)$ 时间内回答 $S$ 上的任何查询。

---

为了构建结构 $\mathcal{T}$，对于每个 $i \in[m]$，将 $S_{i}$ 按升序存储，但每个元素需要与其在 $A$ 中的原始位置索引关联。显然，$\mathcal{T}$ 可以在 $O(n \log s)$ 时间内构建。让我们看看如何回答查询 $(x, y)$。首先，确定 $a, b \in[m]$ 使得 $A[x] \in S_{a}$ 且 $A[y] \in S_{b}$，这可以通过在 $O(m)$ 时间内简单地完成。扫描 $S_{a}$ 并识别子集 $S_{a}^{\prime}=S_{a} \cap A[x: y]$（对于 $S_{a}$ 中的每个元素，检查其在 $A$ 中的原始索引是否落在区间 $[x, y]$ 中）。由于 $S_{a}$ 已经排序，我们可以在 $O(s)$ 时间内以排序的顺序生成 $S_{a}^{\prime}$。以相同的方式，在 $O(s)$ 时间内计算 $S_{b}^{\prime}=S_{b} \cap A[x: y]$。此时，$A[x: y]$ 中的所有元素已经被划分到 $b-a+1$ 个排序数组中：$S_{a}^{\prime}, S_{a+1}, S_{a+2}, \ldots, S_{b-1}, S_{b}^{\prime}$。现在的目标是从这些数组的并集中找到第 $\lfloor(y-x+1) / 2\rfloor$ 小的元素。Frederickson 和 Johnson [10] 描述了一种从排序数组的并集中选择给定排名元素的算法。在我们的场景中，该算法运行时间为 $O\left(m \log \frac{n}{m}\right)=O(m \log s)$。总的查询时间为 $O(s+m \log s)=O(m \log s)$，因为 $s \leq \sqrt{n}$。

---

我们现在得出结论，区间中位数问题是 $(\log n, \log n)$ 谱索引可索引的，并且可以应用定理 1。

**二维线性规划问题。** 通过借助几何对偶性 [7]，该问题可以转换为 [19] 中的 "通过凸包的直线"（我们已经解决了这个问题）以及以下问题：

- 从凸包外射出的射线。这里，$S$ 是一个包含 $n$ 个点的集合，且 $\mathrm{CH}(S)$ 覆盖了原点。给定从原点出发的任意射线 $q$，查询返回的是 $\mathrm{CH}(S)$ 的边 $e_{\text{exit}}$，从该边射线 $q$ 离开 $\mathrm{CH}(S)$。

---

我们将专注于解决上述问题。在继续之前，让我们陈述关于该问题的两个事实：

- 给定任意射线 $q$，可以使用 [21] 中的算法在 $O(n)$ 时间内找到 $e_{\text{exit}}$；我们将其称为基本算法。
- 我们可以在 $O(n \log n)$ 时间内构建一个结构来在 $O(\log n)$ 时间内回答任何查询。首先计算 $\mathrm{CH}(S)$，然后通过连接原点与所有顶点的线段来“划分”它；参见图 1(d)。查询射线 $q$ 可以通过查找该射线经过的划分中的三角形来得到答案。我们将其称为基本结构。

---

与之前讨论的所有问题不同，目前我们无法证明“射线离开凸包”问题是 $(\log n, \log n)$ 谱索引可索引的。然而，凭借定理 1，我们并不需要这样做！只需证明该问题是 $\left(\log n, n^{c}\right)$ 谱索引可索引的，其中 $c$ 是任意小于 1 的正常数。定理 1 允许我们在 $O(n \log r)$ 时间内回答 $r \leq n^{1-c} \log n$ 个查询。当 $r$ 达到 $n^{1-c}$ 时，我们可以构建基本结构，在 $O(n \log n) = O(n \log r)$ 时间内完成构建，并且每个后续查询可以在 $O(\log n) = O(\log r)$ 时间内得到答案。这使我们能够在所有 $r \leq n$ 的情况下实现 $\text{Time}(n, r)=O(n \log r)$。

---

我们将证明“射线离开凸包”问题是 $(\log n, \sqrt{n})$ 谱索引可索引的。我们将通过借助 [19] 中的一个结果来实现这一目的，在该结果中，Karp、Motwani 和 Raghavan 使用自顶向下的方法构建了一个具有以下特性的二叉树 $\mathcal{T}$：

- 如果节点 $u$ 在树的第 $\ell$ 层，那么 $u$ 与 $S$ 中的一个包含 $n / 2^{\ell}$ 个点的集合 $S(u)$ 相关联。
- 树的前 $\ell$ 层可以在 $O(n \cdot \ell)$ 时间内构建。
- 通过遍历最多一个从根到叶的路径来回答查询。如果搜索过程降至节点 $u$，则可以通过在 $S(u)$ 上运行基本算法 [21] 来在 $O(|S(u)|)$ 时间内找到目标边 $e_{\text{exit}}$。

---

回到我们的场景，设定任意整数 $s \leq n$。在 $O(n \log \sqrt{s}) = O(n \log s)$ 时间内，在 $S$ 上构建树的前 $1 + \lceil \log \sqrt{s} \rceil$ 层。为了回答一个查询，如果目标边 $e_{\text{exit}}$ 尚未找到，我们沿着树的路径下降到层数为 $\lceil \log \sqrt{s} \rceil$ 的节点 $u$。节点 $u$ 关联的集合 $|S(u)|$ 最多包含 $n / 2^{\log \sqrt{s}} = n / \sqrt{s}$ 个点。因此，我们可以在 $O(n / \sqrt{s}) = O\left(\frac{n}{s} \cdot \sqrt{s}\right)$ 时间内在 $S(u)$ 上运行基本算法找到 $e_{\text{exit}}$。因此，“射线离开凸包”问题是 $(\log n, \sqrt{n})$ 谱索引可索引的。

---

我们在本节末尾作如下备注，正如上述讨论所揭示的，关于自顶向下方法与我们归约之间的内在联系。从本质上讲，我们是逐步构建 [19] 中的结构：第 $i$ 次（$i \geq 1$）“时期”（在第 3.2 节的证明中）重建了 $\mathcal{T}$ 的前 $2^{2^{i}}$ 层。这与 [19] 中的做法不同，在 [19] 中，节点是按照“首次接触时扩展”的方式进行构建，如第 2 节所述。事实上，所有基于自顶向下方法设计的现有 DDS 算法，都可以通过“谱索引可索引性”的桥梁，像我们在“射线离开凸包”问题中展示的那样，被封装进我们的归约框架中。

## 5 DDS Using Structures with $\omega(n \log n)$ Construction Time

到目前为止，我们的讨论主要集中在满足 $B(n) = O(\log n)$ 的数据结构上。在本节中，我们首先展示这一条件对于黑箱归约算法保证非常数 $\log (r)$-streak 上界的必要性。其次，我们提出定理 1 的一个扩展版本，该版本允许使用 $\max {B(n), Q(n)}=\text{polylog} n$ 的数据结构来设计一个在所有 $r \leq n$ 的情况下具有良好 $\text{TimE}(n, r)$ 性能的 DDS 算法。

---

**对于 $\boldsymbol{B}(\boldsymbol{n})=\boldsymbol{\omega}(\log \boldsymbol{n})$ 的常数 streak 上界。** 我们后续的难度论证需要 $n \cdot B(n)$ 是一个凸函数。考虑任意一个黑箱归约算法 $\mathcal{A}$。回忆一下，$\mathcal{A}$ 需要能够处理任何具有受限数据结构的可分解问题（读者可以在继续之前回顾第 3.3 节）。假设 $\mathcal{A}$ 能够对所有此类问题保证 LogStreak $(n)$ 的 $\log (r)$-streak 上界，即 $\mathcal{A}$ 能够始终在 $O(n \log r)$ 时间内回答所有 $r \leq \text{LogStreak}(n)$ 的查询。我们将证明，如果 $B(n)=\omega(\log n)$，那么 LogStreak $(n)$ 必须是 $O(1)$。

---

类似于第 3.3 节，我们将构造一个可分解问题及其伴随的数据结构。数据集 $S$ 包含 $n$ 个任意元素；给定任何谓词，对 $S$ 的查询始终返回 $|S|$。每当被要求在 $S$ 上构建数据结构 $\mathcal{T}(S)$ 时，我们故意浪费 $|S| \cdot B(|S|)$ 的时间，然后输出 $S$ 的一个任意排列（存储在数组中）。每当被要求回答一个查询时，我们立即以常数时间返回 $|S|$。换句话说，函数 $Q(n)$ 被固定为 1。

---

从现在起，我们将 $r$ 固定为算法 $\mathcal{A}$ 在给定我们构造的数据结构时能够保证的 $\text{LogStreak}(n)$ 的值。我们假设 $r=\omega(1)$；否则 $\text{LogStreak}(n)=O(1)$，我们的论证已经完成。由于 $\mathcal{A}$ 在 $O(n \log r)$ 时间内回答了 $r$ 个查询，其中至少一个查询的成本必须是 $O\left(\frac{n \log r}{r}\right)$。接下来的论证将集中于这个特定的查询。

---

回忆第 3.3 节的内容，为了回答这个查询，算法 $\mathcal{A}$ 需要搜索若干（包括 0）结构 $\mathcal{T}\left(S_{1}^{\prime}\right), \mathcal{T}\left(S_{2}^{\prime}\right), \ldots, \mathcal{T}\left(S_{t}^{\prime}\right)$ 并扫描一个子集 $S_{\text{scan}}^{\prime} \subseteq S$。由于 $\mathcal{A}$ 需要支付 $\Omega\left(\left|S_{\text{scan}}^{\prime}\right|\right)$ 的成本来扫描 $S_{\text{scan}}^{\prime}$，因此必须满足 $\left|S_{\text{scan}}^{\prime}\right| \leq \alpha \cdot \frac{n \log r}{r}$，其中常数 $\alpha>0$。我们考虑足够大的 $r$（我们知道 $r=\omega(1)$），使得 $\alpha \cdot \frac{n \log r}{r} \leq n / 2$。由于 $\mathcal{A}$ 必须满足约束条件 (5)，我们可以断言：

$$
\begin{equation*}
\left|S_{1}^{\prime} \cup S_{2}^{\prime} \cup \ldots \cup S_{t}^{\prime}\right| \geq|S|-\left|S_{\text {scan }}^{\prime}\right| \geq n / 2 \tag{6}
\end{equation*}
$$

这意味着 $t \geq 1$。由于算法 $\mathcal{A}$ 在搜索每个结构 $\mathcal{T}\left(S_{i}^{\prime}\right)(i \in[t])$ 时需要花费常数时间，我们可以得出 $t=O((n / r) \cdot \log r)$。

---

然而，根据我们设计 $\mathcal{T}$ 的方式，构造 $\mathcal{T}\left(S_{1}^{\prime}\right), \mathcal{T}\left(S_{2}^{\prime}\right), \ldots, \mathcal{T}\left(S_{t}^{\prime}\right)$ 的总成本为：
$$
\begin{equation*}
\sum_{i \in[t]}\left|S_{i}^{\prime}\right| \cdot B\left(\left|S_{i}^{\prime}\right|\right) \tag{8}
\end{equation*}
$$

设 $\lambda = \sum_{i \in[t]} \left|S_{i}^{\prime}\right|$；由 (6) 可知 $\lambda \geq n / 2$。由于 $n \cdot B(n)$ 是凸函数且 $B(n)$ 是单调不减的，我们知道当所有 $\left|S_{i}^{\prime}\right| = \lambda / t$ 时，式 (8) 取最小值。因此：

(8) $\geq \lambda \cdot B(\lambda / t) \geq \frac{n}{2} \cdot B\left(\frac{n}{2 t}\right)$.

由 (7) 可知 $n /(2 t)=\Omega(r / \log r)$。由于 $B(n)=\omega(\log n)$，基本的渐近分析表明，$B(n /(2 t))$ 必然是 $\omega(\log r)$。

---

我们现在得出结论，式 (9)，因此式 (8)，必然是 $\omega(n \log r)$。但这与算法 $\mathcal{A}$ 能够在 $O(n \log r)$ 时间内处理 $r$ 个查询相矛盾。因此，我们关于 $r = \omega(1)$ 的假设必须是错误的。

**对于 $\boldsymbol{B}(\boldsymbol{n}) = \boldsymbol{\omega}(\log \boldsymbol{n})$ 的 DDS 算法。** 构建时间为 $\omega(n \log n)$ 的数据结构在 DDS 中仍然有用，只要我们不执着于在 $O(n \log r)$ 时间内回答 $r$ 个查询。为了形式化这一点，我们对定理 1 背后的技术进行了修改，从而得出另一个具有以下保证的通用归约。

- **定理 3。** 假设 $B(n)$ 和 $Q(n)$ 是非减函数，且都满足 $O\left(\log ^{\gamma} n\right)$，其中 $\gamma \geq 1$ 是常数。每个 $(B(n), Q(n))$ 谱索引可索引问题都有一个 DDS 算法，其时间复杂度为 $\text{TimE}(n, r) = O\left(n \log ^{\gamma} r\right)$，适用于所有 $r \leq n$。

---

该证明与第 3.2 节中所述的内容类似，移至附录 A。上述定理的一个有趣应用是 $\mathbb{R}^{d}$ 中的正交范围计数/报告问题，其中 $d$ 是至少为 3 的固定常数（参见第 1.2 节中的问题定义）。增强了分数级联的范围树 [9]，可以在 $O\left(n \log ^{d-1} n\right)$ 时间内构建，且能够在 $O\left(\log ^{d-1} n\right)$ 时间内回答计数/报告查询。该问题是可分解的，因此是 $\left(\log ^{d-1} n, \log ^{d-1} n\right)$ 谱索引可索引的。定理 3 直接给出了一个 DDS 算法，其时间复杂度为 $\text{TimE}(n, r) = O\left(n \log ^{d-1} r\right)$，适用于所有 $r \leq n$。这严格改进了第 1.2 节中提到的 [19] 的结果。

## A Proof of Theorem 3

们的算法按“时期”执行。在第 $i$ 次 $(i \geq 1)$ 时期的开始，我们设定 $s=2^{2^{i}}$，并在 $S$ 上创建一个结构 $\mathcal{T}$——这是由 $(B(n), Q(n))$ 谱索引可索引性所保证的结构，其构建时间为 $O(n \cdot B(s))=O\left(n \cdot 2^{i \cdot \gamma}\right)$。该结构可以在 $O\left(\frac{n}{s} \cdot Q(s)\right)$ 时间内回答任何查询。第 $i$ 次时期在回答了 $s$ 个查询后结束。这些查询的总成本为：

$$
s \cdot O\left(\frac{n}{s} \cdot Q(s)\right)=O(n \cdot Q(s))=O\left(n \cdot 2^{i \cdot \gamma}\right)
$$

---

因此，第 $i$ 次时期的总计算时间为 $O\left(n \cdot 2^{i \cdot \gamma}\right)$。由于 $\gamma \geq 1$，我们知道当 $i$ 增加 1 时，$n \cdot 2^{i \cdot \gamma}$ 至少翻倍。因此，所有时期的总成本在渐近上由 $O\left(n \cdot 2^{h^{*} \cdot \gamma}\right)$ 主导，其中 $h^{*}$ 是所需时期的数量。具体而言，$h^{*}$ 的值是满足以下两个条件的最小整数 $h \geq 1$：

- 条件 C1：$2^{2^{h}} \leq n$（$s$ 的值不能超过 $n$）；
- 条件 C2：$\sum_{i=1}^{h} 2^{2^{i}} \geq r$（$h$ 个时期能够回答的查询数必须至少为 $r$）。

---

令 $H$ 表示满足 $2^{2^{H}} \geq r$ 的最小整数 $h \geq 1$。这意味着：
$$
\begin{equation*}
2^{2^{H-1}}<r \Leftrightarrow 2^{2^{H}}<r^{2} . \tag{10}
\end{equation*}
$$

当 $2^{2^{H}} \leq n$ 时，上述论证保证时期数量 $h^{*}$ 最多为 $H$（不等式 $2^{2^{H}} \leq n$ 意味着 $2^{2^{h^{*}}} \leq n$，符合条件 C1）。在这种情况下，所有时期的总成本为 $O\left(n \cdot 2^{h^{*} \cdot \gamma}\right)=O\left(n \cdot 2^{H \cdot \gamma}\right)=O\left(n \log ^{\gamma} r\right)$。

---

如果 $2^{2^{H}}>n$，上述论证将不成立。然而，此时从 (10) 可知 $r^{2}>2^{2^{H}}>n$，这导致 $r>\sqrt{n}$。一旦 $r$ 达到 $\lceil\sqrt{n}\rceil$（称为“断点”），我们将在整个 $S$ 上创建一个结构 $\mathcal{T}$，其构建时间为 $O\left(n \log ^{\gamma} n\right)$，并在 $O(Q(n))=O\left(\log ^{\gamma} n\right)$ 时间内回答每个后续查询，直到 $r=n$。断点之后的查询总成本为 $O\left(n \log ^{\gamma} n\right)=O\left(n \log ^{\gamma} r\right)$。因此，我们得到了一个算法，保证 $\text{TimE}(n, r)=O\left(n \log ^{\gamma} r\right)$，适用于所有 $r \leq n$，完成了定理 3 的证明。

