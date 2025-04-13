# Optimal Sampling from Sliding Windows

# 滑动窗口的最优采样

Vladimir Braverman*

弗拉基米尔·布拉弗曼（Vladimir Braverman）*

University of California Los Angeles

加利福尼亚大学洛杉矶分校

vova@cs.ucla.edu

Rafail Ostrovsky ${}^{ \dagger  }$

拉菲尔·奥斯特罗夫斯基（Rafail Ostrovsky） ${}^{ \dagger  }$

University of California Los

加利福尼亚大学

Angeles

洛杉矶分校

rafail@cs.ucla.edu

Carlo Zaniolo

卡罗·扎尼奥洛（Carlo Zaniolo）

University of California Los

加利福尼亚大学

Angeles

洛杉矶分校

zaniolo@cs.ucla.edu

## Abstract

## 摘要

A sliding windows model is an important case of the streaming model, where only the most "recent" elements remain active and the rest are discarded in a stream. The sliding windows model is important for many applications (see, e.g., Babcock, Babu, Datar, Motwani and Widom (PODS 02); and Datar, Gionis, Indyk and Motwani (SODA 02)). There are two equally important types of the sliding windows model - windows with fixed size, (e.g., where items arrive one at a time,and only the most recent $n$ items remain active for some fixed parameter $n$ ),and bursty windows (e.g., where many items can arrive in "bursts" at a single step and where only items from the last $t$ steps remain active,again for some fixed parameter $t$ ).

滑动窗口模型是流数据模型的一个重要实例，在流数据中，只有最“近期”的元素保持活跃，其余元素则被丢弃。滑动窗口模型在许多应用中都很重要（例如，参见巴布科克（Babcock）、巴布（Babu）、达塔尔（Datar）、莫特瓦尼（Motwani）和维德姆（Widom）（PODS 02）；以及达塔尔（Datar）、吉奥尼斯（Gionis）、因迪克（Indyk）和莫特瓦尼（Motwani）（SODA 02））。滑动窗口模型有两种同样重要的类型——固定大小的窗口（例如，元素逐个到达，对于某个固定参数 $n$，只有最近的 $n$ 个元素保持活跃）和突发窗口（例如，在某一步可能会有大量元素“突发”到达，对于某个固定参数 $t$，只有最后 $t$ 步的元素保持活跃）。

Random sampling is a fundamental tool for data streams, as numerous algorithms operate on the sampled data instead of on the entire stream. Effective sampling from sliding windows is a nontrivial problem, as elements eventually expire. In fact, the deletions are implicit; i.e., it is not possible to identify deleted elements without storing the entire window. The implicit nature of deletions on sliding windows does not allow the existing methods (even those that support explicit deletions, e.g., Cormode, Muthukrishnan and Rozenbaum (VLDB 05); Frahling, Indyk and Sohler (SOCG 05)) to be directly "translated" to the sliding windows model. One trivial approach to overcoming the problem of implicit deletions is that of over-sampling. When $k$ samples are required,the over-sampling method maintains ${k}^{\prime } > k$ samples in the hope that at least $k$ samples are not expired. The obvious disadvantages of this method are twofold:

随机采样是处理数据流的基本工具，因为许多算法是对采样数据而非整个数据流进行操作。从滑动窗口中进行有效采样是一个非平凡的问题，因为元素最终会过期。实际上，删除操作是隐式的；也就是说，如果不存储整个窗口，就无法识别已删除的元素。滑动窗口上删除操作的隐式性质使得现有的方法（即使是那些支持显式删除的方法，例如，科尔莫德（Cormode）、穆图克里什南（Muthukrishnan）和罗森鲍姆（Rozenbaum）（VLDB 05）；弗拉林（Frahling）、因迪克（Indyk）和索勒（Sohler）（SOCG 05））无法直接“转换”到滑动窗口模型。解决隐式删除问题的一种简单方法是过采样。当需要 $k$ 个样本时，过采样方法会保留 ${k}^{\prime } > k$ 个样本，希望至少有 $k$ 个样本未过期。这种方法有两个明显的缺点：

(a) It introduces additional costs and thus decreases the performance; and

(a) 它会引入额外的成本，从而降低性能；并且

(b) The memory bounds are not deterministic, which is atypical for streaming algorithms (where even small probabil-

(b) 内存界限不是确定性的，这在流算法中是不常见的（对于足够大的数据流，即使是小概率事件最终也可能发生）。

ity events may eventually happen for a stream that is big enough).

Babcock, Datar and Motwani (SODA 02), were the first to stress the importance of improvements to over-sampling. They formally introduced the problem of sampling from sliding windows and improved the over-sampling method for sampling with replacement. Their elegant solutions for sampling with replacement are optimal in expectation,and thus resolve disadvantage(a)mentioned above. Unfortunately, the randomized bounds do not resolve disadvantage (b) above. Interestingly, all algorithms that employ the ideas of Babcock, Datar and Motwani have the same central problem of having to deal with randomized complexity (see e.g., Datar and Muthukrishnan (ESA 02); Chakrabarti, Cormode and McGregor (SODA 07)). Further, the proposed solutions of Babcock, Datar and Motwani for sampling without replacement are based on the criticized over-sampling method and thus do not solve problem (a). Therefore, the question of whether we can solve sampling on sliding windows optimally (i.e., resolving both disadvantages) is implicit in the paper of Babcock, Datar and Motwani and has remained open for all variants of the problem.

巴布科克（Babcock）、达塔尔（Datar）和莫特瓦尼（Motwani）（SODA 02）首次强调了改进过采样方法的重要性。他们正式提出了从滑动窗口中采样的问题，并改进了有放回采样的过采样方法。他们针对有放回采样提出的精妙解决方案在期望意义上是最优的，从而解决了上述缺点 (a)。不幸的是，随机界限并没有解决上述缺点 (b)。有趣的是，所有采用巴布科克（Babcock）、达塔尔（Datar）和莫特瓦尼（Motwani）思想的算法都存在一个共同的核心问题，即必须处理随机复杂度（例如，参见达塔尔（Datar）和穆图克里什南（Muthukrishnan）（ESA 02）；查克拉巴蒂（Chakrabarti）、科尔莫德（Cormode）和麦格雷戈（McGregor）（SODA 07））。此外，巴布科克（Babcock）、达塔尔（Datar）和莫特瓦尼（Motwani）针对无放回采样提出的解决方案是基于饱受批评的过采样方法，因此没有解决问题 (a)。因此，我们是否能够最优地解决滑动窗口采样问题（即解决上述两个缺点）这一问题在巴布科克（Babcock）、达塔尔（Datar）和莫特瓦尼（Motwani）的论文中是隐含的，并且对于该问题的所有变体而言，这个问题仍然悬而未决。

In this paper we answer these questions affirmatively and provide optimal sampling schemas for all variants of the problem, i.e., sampling with or without replacement from fixed or bursty windows. Specifically, for fixed-size windows, we provide optimal solutions that require $O\left( k\right)$ memory; for bursty windows,we show algorithms that require $O\left( {k\log n}\right)$ ,which is optimal since it matches the lower bound by Gemulla and Lehner (SIGMOD 08). In contrast to the work of of Babcock, Datar and Motwani, our solutions have deterministic bounds. Thus, we prove a perhaps somewhat surprising fact: the memory complexity of the sampling-based algorithm for all variants of the sliding windows model is comparable with that of streaming models (i.e., without the sliding windows). This is the first result of this type, since all previous "translations" of sampling-based algorithms to sliding windows incur randomized memory guarantees only.

在本文中，我们肯定地回答了这些问题，并为该问题的所有变体提供了最优采样方案，即从固定窗口或突发窗口中有放回或无放回地采样。具体而言，对于固定大小的窗口，我们提供了需要$O\left( k\right)$内存的最优解决方案；对于突发窗口，我们展示了需要$O\left( {k\log n}\right)$的算法，这是最优的，因为它与Gemulla和Lehner（SIGMOD 08）给出的下界相匹配。与Babcock、Datar和Motwani的工作相比，我们的解决方案具有确定性的界限。因此，我们证明了一个可能有些令人惊讶的事实：基于采样的算法在滑动窗口模型所有变体中的内存复杂度与流模型（即没有滑动窗口）中的内存复杂度相当。这是此类的首个结果，因为之前所有将基于采样的算法“转换”到滑动窗口的工作仅能保证随机化的内存使用。

## Categories and Subject Descriptors

## 类别与主题描述符

F. 2 [ANALYSIS OF ALGORITHMS AND PROBLEM COM-

F. 2 [算法分析与问题复杂度（ANALYSIS OF ALGORITHMS AND PROBLEM COM-

PLEXITY]: Miscellaneous

复杂度）]: 其他

## Keywords

## 关键词

Data Streams, Sliding Windows, Random Sampling

数据流、滑动窗口、随机采样

## General Terms

## 通用术语

Algorithms, Theory

算法、理论

---

<!-- Footnote -->

*Supported in part by NSF grant 0830803

*部分由美国国家科学基金会（NSF）资助，项目编号0830803

${}^{ \dagger  }$ Department of Computer Science and Department of Mathematics, UCLA, Los Angeles, CA 90095, USA. Supported in part by IBM Faculty Award, Xerox Innovation Group Award, NSF grants 0430254, 0716835, 0716389, 0830803 and U.C. MICRO grant.

${}^{ \dagger  }$ 美国加利福尼亚州洛杉矶市加州大学洛杉矶分校（UCLA）计算机科学系和数学系，邮编90095。部分由IBM教师奖、施乐创新集团奖、美国国家科学基金会资助（项目编号0430254、0716835、0716389、0830803）以及加州大学微项目资助。

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. To copy otherwise, to republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee.

允许个人或课堂使用免费制作本作品全部或部分的数字或硬拷贝，前提是这些拷贝不用于盈利或商业目的，并且所有拷贝都带有此声明和第一页的完整引用信息。否则，若要进行复制、重新发布、上传到服务器或分发给列表，需要事先获得特定许可和/或支付费用。

PODS'09, June 29-July 2, 2009, Providence, Rhode Island, USA. Copyright 2009 ACM 978-1-60558-553-6 /09/06 ...\$10.00.

2009年ACM数据库系统原理研讨会（PODS'09），2009年6月29日至7月2日，美国罗德岛州普罗维登斯市。版权所有2009美国计算机协会，ISBN 978 - 1 - 60558 - 553 - 6 / 09 / 06 ... 10.00美元。

<!-- Footnote -->

---

## 1. INTRODUCTION

## 1. 引言

Random sampling and sliding windows are two fundamental concepts for data streams. Sampling is a very natural way to summarize data properties with sublinear space; indeed, it is a key component of many streaming algorithms and techniques. Just to mention a few, the relevant papers include Aggarwal [2]; Alon, Duffield, Lund and Thorup [3]; Alon, Matias and Szegedy [4]; Babcock, Babu, Datar, Motwani and Widom [8]; Babcock, Datar and Motwani [10]; Bar-Yossef [13]; Bar-Yossef, Kumar and Sivaku-mar [17]; Buriol, Frahling, Leonardi, Marchetti-Spaccamela and Sohler [20]; Chakrabarti, Cormode and McGregor [21]; Chaudhuri and Mishra [26]; Chaudhuri, Motwani and Narasayya [27]; Cohen [29]; Cohen and Kaplan [30]; Cormode, Muthukrishnan and Rozenbaum [32]; Dasgupta, Drineas, Harb, Kumar and Mahoney [35]; Datar and Muthukrishnan [37]; Duffield, Lund and Thorup [38]; Frahling, Indyk and Sohler [43]; Gandhi, Suri and Welzl [46]; Gemulla [47]; Gemulla and Lehner [48]; Gibbons and Matias [49]; Guha, Meyerson, Mishra, Motwani and O'Callaghan [54]; Haas [55]; Kolonko and Wäsch [59]; Li [62]; Palmer and Falout-sos [67]; Szegedy [70]; and Vitter [72]; These papers illustrate the vitality of effective sampling methods for data streams. Among other methods, uniform random sampling is the most general and well-understood. Most applications maintain multiple samples using two popular methods: namely, sampling with replacement and sampling without replacement. The former method assumes independence among samples; the latter forbids repetitions. While sampling without replacement preserves more information, sampling with replacement is sometimes preferred due to its simplicity; thus both schemas are important for applications.

随机采样和滑动窗口是数据流的两个基本概念。采样是一种用亚线性空间总结数据属性的非常自然的方法；实际上，它是许多流算法和技术的关键组成部分。仅举几例，相关论文包括Aggarwal [2]；Alon、Duffield、Lund和Thorup [3]；Alon、Matias和Szegedy [4]；Babcock、Babu、Datar、Motwani和Widom [8]；Babcock、Datar和Motwani [10]；Bar - Yossef [13]；Bar - Yossef、Kumar和Sivakumar [17]；Buriol、Frahling、Leonardi、Marchetti - Spaccamela和Sohler [20]；Chakrabarti、Cormode和McGregor [21]；Chaudhuri和Mishra [26]；Chaudhuri、Motwani和Narasayya [27]；Cohen [29]；Cohen和Kaplan [30]；Cormode、Muthukrishnan和Rozenbaum [32]；Dasgupta、Drineas、Harb、Kumar和Mahoney [35]；Datar和Muthukrishnan [37]；Duffield、Lund和Thorup [38]；Frahling、Indyk和Sohler [43]；Gandhi、Suri和Welzl [46]；Gemulla [47]；Gemulla和Lehner [48]；Gibbons和Matias [49]；Guha、Meyerson、Mishra、Motwani和O'Callaghan [54]；Haas [55]；Kolonko和Wäsch [59]；Li [62]；Palmer和Faloutsos [67]；Szegedy [70]；以及Vitter [72]；这些论文展示了有效采样方法在数据流中的重要性。在众多方法中，均匀随机采样是最通用且被广泛理解的。大多数应用使用两种流行的方法维护多个样本：即有放回采样和无放回采样。前者假设样本之间相互独立；后者禁止重复。虽然无放回采样保留了更多信息，但有放回采样有时因其简单性而更受青睐；因此，这两种方案对应用都很重要。

The concept of sliding windows expresses the importance of recent data for applications. In this model, analysis is restricted to the most recent portion of the stream; the outdated elements must not be considered. The significance of sliding windows has been emphasized from the very beginning of data stream research. We cite, e.g., the influential paper of Babcock, Babu, Datar, Motwani and Widom [8]:

滑动窗口的概念体现了近期数据对应用的重要性。在这个模型中，分析仅限于数据流的最新部分；不能考虑过时的元素。从数据流研究一开始，滑动窗口的重要性就得到了强调。例如，我们引用Babcock、Babu、Datar、Motwani和Widom具有影响力的论文[8]：

Imposing sliding windows on data streams is a natural method for approximation that has several attractive properties. It is well-defined and easily understood: the semantics of the approximation are clear, so that users of the system can be confident that they understand what is given up in producing the approximate answer. It is deterministic, so there is no danger that unfortunate random choices will produce a bad approximation. Most importantly, it emphasizes recent data, which in the majority of real-world applications is more important and relevant than old data: if one is trying in real-time to make sense of network traffic patterns, or phone call or transaction records, or scientific sensor data, then in general insights based on the recent past will be more informative and useful than insights based on stale data.

在数据流上设置滑动窗口是一种自然的近似方法，它具有几个吸引人的特性。它定义明确且易于理解：近似的语义清晰，因此系统用户可以确信他们明白在生成近似答案时放弃了什么。它是确定性的，因此不存在因不幸的随机选择而产生糟糕近似结果的风险。最重要的是，它强调近期数据，在大多数实际应用中，近期数据比旧数据更重要、更相关：如果有人试图实时理解网络流量模式、电话或交易记录，或科学传感器数据，那么一般来说，基于近期数据的见解比基于陈旧数据的见解更有信息价值和实用性。

The importance of the sliding windows model is well illustrated by the considerable amount of relevant papers in both theory and database communities. A small sample subset of relevant papers includes the work of Arasu, Babcock, Babu, Cieslewicz, Datar, Ito, Motwani, Srivastava and Widom [5]; Arasu and Manku [6]; Ayad and Naughton [7]; Babcock, Babu, Datar, Motwani and Thomas [9]; Babcock, Babu, Datar, Motwani and Widom [8]; Babcock, Datar and Motwani [10, 11]; Babcock, Datar, Motwani and O'Callaghan [12]; Das, Gehrke and Riedewald [34]; Datar, Gionis, Indyk and Motwani [36]; Datar and Motwani, Chapter 8, [1]; Datar and Muthukrishnan [37]; Feigenbaum, Kannan and Zhang [39]; Gibbons and Tirthapura [50]; Golab, DeHaan, Demaine, Lopez-Ortiz and Munro [51]; Golab and Özsu [52]; Lee and Ting [61]; Li, Maier, Tufte, Papadimos and Tucker [64]; and Tatbul and Zdonik [71].

滑动窗口模型的重要性在理论界和数据库界的大量相关论文中得到了充分体现。相关论文的一小部分示例包括阿拉苏（Arasu）、巴布科克（Babcock）、巴布（Babu）、切斯莱维茨（Cieslewicz）、达塔尔（Datar）、伊藤（Ito）、莫特瓦尼（Motwani）、斯里瓦斯塔瓦（Srivastava）和维德姆（Widom）的工作 [5]；阿拉苏和曼库（Manku） [6]；阿亚德（Ayad）和诺顿（Naughton） [7]；巴布科克、巴布、达塔尔、莫特瓦尼和托马斯（Thomas） [9]；巴布科克、巴布、达塔尔、莫特瓦尼和维德姆 [8]；巴布科克、达塔尔和莫特瓦尼 [10, 11]；巴布科克、达塔尔、莫特瓦尼和奥卡拉汉（O'Callaghan） [12]；达斯（Das）、格尔克（Gehrke）和里德瓦尔德（Riedewald） [34]；达塔尔、吉奥尼斯（Gionis）、因迪克（Indyk）和莫特瓦尼 [36]；达塔尔和莫特瓦尼，第 8 章， [1]；达塔尔和穆图克里什南（Muthukrishnan） [37]；费根鲍姆（Feigenbaum）、坎南（Kannan）和张（Zhang） [39]；吉本斯（Gibbons）和蒂尔塔普拉（Tirthapura） [50]；戈拉布（Golab）、德哈恩（DeHaan）、德梅因（Demaine）、洛佩斯 - 奥尔蒂斯（Lopez - Ortiz）和芒罗（Munro） [51]；戈拉布和厄兹苏（Özsu） [52]；李（Lee）和廷（Ting） [61]；李、迈尔（Maier）、图夫特（Tufte）、帕帕迪莫斯（Papadimos）和塔克（Tucker） [64]；以及塔图尔（Tatbul）和兹多尼克（Zdonik） [71]。

Two types of sliding windows are widely recognized. Fixed-size (or sequence-based) windows define a fixed amount of the most recent elements to be active. For instance an application may restrict an analysis to the last trillion of the elements. Fixed-size windows are important for applications where the arrival rate of the data is fixed (but still extremely fast), such as sensors or stock market measurements. In bursty (or timestamp-based) windows the validity of an element is defined by an additional parameter such as a timestamp. For instance, an application may restrict an analysis to elements that arrived within the last hour. Bursty windows are important for applications with asynchronous data arrivals, such as networking or database applications. The importance of both concepts raises two natural questions of optimal sampling from sliding windows:

两种类型的滑动窗口被广泛认可。固定大小（或基于序列）的窗口定义了最近的固定数量的元素为活跃元素。例如，一个应用程序可能将分析限制在最后一万亿个元素上。固定大小的窗口对于数据到达率固定（但仍然极快）的应用程序很重要，例如传感器或股票市场测量。在突发（或基于时间戳）的窗口中，元素的有效性由一个额外的参数（如时间戳）定义。例如，一个应用程序可能将分析限制在最后一小时内到达的元素上。突发窗口对于具有异步数据到达的应用程序很重要，例如网络或数据库应用程序。这两个概念的重要性引出了关于从滑动窗口进行最优采样的两个自然问题：

QUESTION 1.1. Is it possible to maintain a uniform random sampling from sliding windows using provably optimal memory bounds?

问题 1.1：是否可以使用可证明的最优内存界限从滑动窗口中维持均匀随机采样？

QUESTION 1.2. Is sampling from sliding windows algorithmically harder than sampling from the entire stream?

问题 1.2：从滑动窗口采样在算法上是否比从整个流采样更难？

In this paper, perhaps somewhat surprisingly, we definitively answer both questions. Informally, what we show is that it is possible to "translate" (with optimal deterministic memory bounds for all sampling-based algorithms) sampling with and without replacement on the entire stream to sampling with or without replacement in all variants of the sliding windows model. We state precise results in Theorems 2.1, 2.2, 3.6, 4.4 and 5.1.

在本文中，或许有些令人惊讶的是，我们明确地回答了这两个问题。简单来说，我们所展示的是，可以（对于所有基于采样的算法使用最优确定性内存界限）将在整个流上的有放回和无放回采样“转换”为在滑动窗口模型的所有变体中的有放回或无放回采样。我们在定理 2.1、2.2、3.6、4.4 和 5.1 中陈述了精确的结果。

### 1.1 Discussion and Related Work

### 1.1 讨论与相关工作

In spite of their apparent simplicity, both Questions 1.1 and 1.2 have no trivial solution. Indeed, the sliding windows model implies eventual deletions of samples; thus, none of the well-known methods for insertion-only streams (such as the reservoir method [72]) are applicable. Moreover, the deletions are implicit, i.e., they are not triggered by an explicit user's request. Thus, the algorithms for streams with explicit deletions (such as $\left\lbrack  {{32},{43}}\right\rbrack$ ) do not help. Let us illustrate the inherent difficulty of sampling from sliding windows with the following example. Consider the problem of maintaining a single sample from a window of the last ${2}^{50}$ elements. Assume that the 100-th element is picked as an initial sample. Eventually, the $\left( {{2}^{50} + {100}}\right)$ -th element arrives,in which case the sample is outdated. But at this time, the data has already been passed and cannot be sampled.

尽管问题 1.1 和 1.2 看似简单，但都没有简单的解决方案。实际上，滑动窗口模型意味着样本最终会被删除；因此，所有已知的仅适用于插入流的方法（如蓄水池方法 [72]）都不适用。此外，删除是隐式的，即它们不是由用户的显式请求触发的。因此，适用于显式删除流的算法（如 $\left\lbrack  {{32},{43}}\right\rbrack$）也无济于事。让我们用以下示例来说明从滑动窗口采样的内在困难。考虑从最后 ${2}^{50}$ 个元素的窗口中维持单个样本的问题。假设第 100 个元素被选为初始样本。最终，第 $\left( {{2}^{50} + {100}}\right)$ 个元素到达，此时样本已过时。但此时，数据已经流过，无法再进行采样。

Babcock, Datar and Motwani [10] were the first to address the problem of sampling from sliding windows. They criticized "naive" solutions and stressed the importance of further improvements; we repeat their important arguments here. Periodic sampling is a folklore method for fixed-size windows. This method replaces an expired sample with the newly arrived element. While this method gives worst-case optimal memory bounds, the predictability of samples is not acceptable for many applications (see also [48, 55]). Another obvious approach is over-sampling using the Bernoulli method. If $k$ samples are required,then each element is picked w.p. $O\left( \frac{k\log n}{n}\right)$ ,independently,in the hope that at least $k$ samples are not expired. This method has two drawbacks: first, the expected memory is $O\left( {k\log n}\right)$ ,where $n$ is a window size; and second,with small but positive probability, the sample size will be smaller than $k$ .

巴布科克（Babcock）、达塔尔（Datar）和莫特瓦尼（Motwani）[10]是最早解决从滑动窗口中采样问题的人。他们批评了“简单”的解决方案，并强调了进一步改进的重要性；我们在此重复他们的重要观点。定期采样是一种针对固定大小窗口的常见方法。该方法用新到达的元素替换过期的样本。虽然这种方法在最坏情况下能给出最优的内存界限，但样本的可预测性在许多应用中是不可接受的（另见[48, 55]）。另一种明显的方法是使用伯努利方法进行过采样。如果需要$k$个样本，那么每个元素以概率$O\left( \frac{k\log n}{n}\right)$被独立选取，期望至少有$k$个样本未过期。这种方法有两个缺点：首先，期望内存为$O\left( {k\log n}\right)$，其中$n$是窗口大小；其次，存在一个小但为正的概率，使得样本大小小于$k$。

The key idea of the elegant algorithms in [10] is a "successors list"; in fact, this idea has been used in almost all subsequent papers. The successors list method suggests backing up a sample with a list of active successors. When a sample expires, the next successor in the list becomes a sample; thus a sample is available at any moment. Based on this idea, Babcock, Datar and Motwani built solutions for sampling with replacement. For sequence-based windows of size $n$ ,their chain sampling algorithm picks a successor among $n$ future elements,and stores it as it arrives. They show that such schema has an expected memory bound of $O\left( k\right)$ and with high probability will not exceed $O\left( {k\log n}\right)$ . For timestamp-based windows, their priority sampling method associates a priority with every new element. A priority is a random number from(0,1); a sample is an element $p$ with highest priority; and a sample’s successor is an element with the highest priority among all elements that arrived after $p$ . Priority sampling requires $O\left( {k\log n}\right)$ memory words in expectation and with high probability. There is a lower bound of $\Omega \left( {k\log n}\right)$ for timestamp-based windows that was shown in [48]. Thus, the methods of Babcock, Datar and Motwani are optimal in expectation. However, the inherent problem of the replacement method is that the size of the successors list is itself a random variable; thus this method cannot provide worst-case optimal bounds. Moreover, Babcock, Datar and Motwani suggested over-sampling as a solution for sampling without replacement; thus the problem of further improvements is implicitly present in their paper. In his excellent survey, Haas [55] gave a further detailed discussion of their solutions.

[10]中优雅算法的关键思想是“后继列表”；事实上，这一思想几乎在所有后续论文中都有应用。后继列表方法建议用一个活跃后继列表来备份样本。当一个样本过期时，列表中的下一个后继成为样本；因此，任何时刻都有样本可用。基于这一思想，巴布科克、达塔尔和莫特瓦尼构建了有放回采样的解决方案。对于大小为$n$的基于序列的窗口，他们的链式采样算法从$n$个未来元素中选取一个后继，并在其到达时存储。他们表明，这种方案的期望内存界限为$O\left( k\right)$，并且以高概率不会超过$O\left( {k\log n}\right)$。对于基于时间戳的窗口，他们的优先级采样方法为每个新元素关联一个优先级。优先级是一个来自(0, 1)的随机数；样本是具有最高优先级的元素$p$；样本的后继是在$p$之后到达的所有元素中具有最高优先级的元素。优先级采样在期望和高概率情况下需要$O\left( {k\log n}\right)$个内存字。[48]中给出了基于时间戳的窗口的一个下界$\Omega \left( {k\log n}\right)$。因此，巴布科克、达塔尔和莫特瓦尼的方法在期望上是最优的。然而，有放回采样方法的固有问题是后继列表的大小本身是一个随机变量；因此，这种方法无法提供最坏情况下的最优界限。此外，巴布科克、达塔尔和莫特瓦尼建议使用过采样作为无放回采样的解决方案；因此，他们的论文中隐含着进一步改进的问题。哈斯（Haas）[55]在他出色的综述中对他们的解决方案进行了进一步详细的讨论。

Zhang, Li, Yu, Wang and Jiang [73] provide an adaptation of reservoir sampling to sliding windows. However, their approach implies storing the window in memory; thus it is applicable only for small windows. In an important work, Gemulla and Lehner [48] addressed the question of sampling without replacement for timestamp-based windows. They suggest a natural extension of priority sampling by maintaining a list of elements with $k$ -highest priorities. This gives an expected $O\left( {k\log n}\right)$ solution. However,their memory bounds are still randomized. Gemulla [47] and Gemulla and Lehner [48] recently addressed the problem of random sampling from timestamp-based windows with a bounded memory. This setting is different from the original problem of Babcock, Datar and Motwani [10]. Namely, it introduces additional uncertainty in the following sense: there is no guarantee that a sample is available at any moment. They provide a "local" lower bound on the success probability that depends on the window's data. However, there is no "global" lower bound; as Gemulla [47] states in his thesis:

张（Zhang）、李（Li）、余（Yu）、王（Wang）和江（Jiang）[73]对蓄水池采样进行了改进以适用于滑动窗口。然而，他们的方法意味着要将窗口存储在内存中；因此，它仅适用于小窗口。在一项重要的工作中，格穆拉（Gemulla）和莱纳（Lehner）[48]解决了基于时间戳的窗口的无放回采样问题。他们建议通过维护一个具有$k$个最高优先级的元素列表来自然扩展优先级采样。这给出了一个期望为$O\left( {k\log n}\right)$的解决方案。然而，他们的内存界限仍然是随机的。格穆拉[47]以及格穆拉和莱纳[48]最近解决了在有界内存下从基于时间戳的窗口中进行随机采样的问题。这种设置与巴布科克、达塔尔和莫特瓦尼[10]的原始问题不同。即，它在以下意义上引入了额外的不确定性：不能保证任何时刻都有样本可用。他们给出了一个依赖于窗口数据的成功概率的“局部”下界。然而，没有“全局”下界；正如格穆拉[47]在他的论文中所说：

we cannot guarantee a global lower bound other than 0 that holds at any arbitrary time without a-priori knowledge of the data stream.

在没有先验了解数据流的情况下，我们无法保证在任意时刻都成立的非零全局下界。

Thus, the problem of optimal sampling for the entire life-span of the stream from sliding windows remained an open problem for all versions until today. We stress that, while this problem is important in its own right, it also has further implication for many other problems. Indeed, uniform random sampling is a key tool for many streaming problems (see, e.g., [66]). "Translations" to sliding windows using previous methods introduce randomized complexity instead of deterministic memory bounds (see, e.g., [21]).

因此，直到今天，从滑动窗口中对数据流的整个生命周期进行最优采样的问题对于所有版本来说仍然是一个未解决的问题。我们强调，虽然这个问题本身很重要，但它对许多其他问题也有进一步的影响。实际上，均匀随机采样是许多流数据问题的关键工具（例如，见[66]）。使用以前的方法“转换”到滑动窗口会引入随机复杂度，而不是确定性的内存界限（例如，见[21]）。

### 1.2 Our Contribution

### 1.2 我们的贡献

In this paper we answer affirmatively to Questions 1.1 and 1.2 for all variants of the problem, i.e., for sampling with and without replacement from fixed-size or bursty windows. Our solutions have provable optimal memory guarantees and are stated precisely in Theorems 2.1,2.2,3.6,4.4 and 5.1. In particular,we give $O\left( k\right)$ bounds for fixed-size windows (for sampling with or without replacement) and $O\left( {k\log n}\right)$ bounds for bursty windows (for sampling with or without replacement). This is a strict improvement over previous methods that closes the gap between randomized and deterministic complexity, an important fact in its own right. Furthermore, we prove a perhaps somewhat surprising result: that the memory complexity of the sampling-based algorithm for all variants of the sliding windows model is comparable with that of streaming models (i.e., without the sliding windows). This is the first result of this type, since all previous "translations" of sampling-based algorithms to sliding windows incur randomized memory guarantees only.

在本文中，我们对该问题的所有变体（即从固定大小窗口或突发窗口中有放回和无放回抽样）的问题1.1和1.2给出了肯定的答案。我们的解决方案具有可证明的最优内存保证，并在定理2.1、2.2、3.6、4.4和5.1中进行了精确阐述。特别是，我们给出了固定大小窗口（有放回或无放回抽样）的$O\left( k\right)$界，以及突发窗口（有放回或无放回抽样）的$O\left( {k\log n}\right)$界。这是对先前方法的严格改进，缩小了随机复杂度和确定性复杂度之间的差距，这本身就是一个重要的事实。此外，我们证明了一个可能有些令人惊讶的结果：基于抽样的算法在滑动窗口模型所有变体下的内存复杂度与流模型（即没有滑动窗口）的内存复杂度相当。这是此类的首个结果，因为之前所有将基于抽样的算法“转换”到滑动窗口的方法仅能保证随机内存。

Finally, this paper introduces surprisingly simple (yet novel) techniques that are different from all previous approaches. In particular, we reduce the problem of sampling without replacement to the problem of sampling with replacement for all variants of the sliding windows model. This may be of independent interest, since the former method is a more general case then the latter; thus our paper also proves equivalence for sliding windows, as discussed in the next section.

最后，本文引入了令人惊讶的简单（但新颖）的技术，这些技术与之前的所有方法都不同。特别是，我们将滑动窗口模型所有变体下的无放回抽样问题简化为有放回抽样问题。这可能具有独立的研究价值，因为前者方法是后者的更一般情况；因此，正如在下一节中所讨论的，我们的论文也证明了滑动窗口的等价性。

### 1.3 High-Level Ideas of Our Approach

### 1.3 我们方法的高层思路

We start by describing four key ideas: equivalent-width partitions, covering decomposition, generating implicit events, and black-box reduction. Also, we outline our approach by giving high-level descriptions of the most important steps.

我们首先描述四个关键思路：等宽划分、覆盖分解、生成隐式事件和黑盒归约。此外，我们通过对最重要的步骤进行高层描述来概述我们的方法。

Equivalent-Width Partitions. Our methods for the sequence-based windows are based on a surprisingly simple (yet novel) idea of equivalent-width partitions: consider sets $A,B,C$ such that $C \subseteq$ $B \subseteq  A \cup  C$ and $A \cap  C = \varnothing$ and $\left| B\right|  = \left| A\right|$ . Our goal is to obtain a sample from $B$ ,given samples from $A$ and $C$ . We use the following rule: if a sample from $A$ belongs to $B$ ,then we assign it to be a sample from $B$ ; otherwise,we assign a sample from $C$ to be a sample from $B$ . The direct computations show the correctness of this schema,i.e.,the result is always a uniform sample from $B$ .

等宽划分。我们针对基于序列的窗口的方法基于一个令人惊讶的简单（但新颖）的等宽划分思路：考虑集合$A,B,C$，使得$C \subseteq$ $B \subseteq  A \cup  C$且$A \cap  C = \varnothing$且$\left| B\right|  = \left| A\right|$ 。我们的目标是在给定来自$A$和$C$的样本的情况下，从$B$中获取一个样本。我们使用以下规则：如果来自$A$的样本属于$B$，那么我们将其指定为来自$B$的样本；否则，我们将来自$C$的样本指定为来自$B$的样本。直接计算表明该方案的正确性，即结果始终是来自$B$的均匀样本。

As a next step, observe that the above idea can be applied to the sliding windows model. We partition (logically) the entire stream into disjoint intervals (we call them buckets) of size $n$ ,where $n$ is the size of the window. For each bucket we maintain a random sample using any one-pass algorithm (e.g., the reservoir sampling method). If the window coincides with the most recent bucket, then our task is easy; we assign this bucket's sample to be the output. Otherwise, the window intersects the two most recent buckets. It must be the case that the most recent bucket is "partial"; i.e., not all elements have arrived yet. But this case matches precisely our key idea: the most recent bucket corresponds to $C$ ,our window corresponds to $B$ and the second-most recent bucket corresponds to $A$ . We thus can apply the above rule and obtain our sample using only samples from the two buckets. Since we need only these samples, the optimality of our schema is straightforward.

作为下一步，观察到上述思路可以应用于滑动窗口模型。我们（逻辑上）将整个流划分为不相交的区间（我们称之为桶），每个桶的大小为$n$，其中$n$是窗口的大小。对于每个桶，我们使用任何单遍算法（例如，蓄水池抽样方法）维护一个随机样本。如果窗口与最近的桶重合，那么我们的任务很简单；我们将这个桶的样本指定为输出。否则，窗口与最近的两个桶相交。最近的桶必然是“部分的”，即并非所有元素都已到达。但这种情况恰好符合我们的关键思路：最近的桶对应于$C$，我们的窗口对应于$B$，第二近的桶对应于$A$。因此，我们可以应用上述规则，仅使用两个桶的样本获得我们的样本。由于我们只需要这些样本，我们方案的最优性是显而易见的。

The above idea can be generalized to the sampling without replacement. Indeed,we show that,given $k$ -samples without replacement from $A$ and $C$ ,we take the portion of $A$ ’s sample that belongs to $B$ and complete it with the random portion from $C$ ’s sample. We show that the result is a $k$ -sample without replacement from $B$ . As before, we apply this idea to sliding windows; the detailed proofs can be found in the main body of this paper.

上述思路可以推广到无放回抽样。实际上，我们表明，在给定来自$A$和$C$的无放回$k$样本的情况下，我们取$A$样本中属于$B$的部分，并从$C$的样本中随机选取一部分来补充它。我们证明结果是来自$B$的无放回$k$样本。和之前一样，我们将这个思路应用于滑动窗口；详细的证明可以在本文的主体部分找到。

## Covering Decomposition and Generating Implicit Events.

## 覆盖分解和生成隐式事件。

For timestamp-based windows, the size of the window is unknown; moreover, it was shown (see, e.g., [36]) that the size of the window cannot be computed precisely with sublinear memory. This negative result is a key difference between timestamp-based windows and all other models, such as insertion-only streams and streams with explicit deletions (the turnstile model). In fact, this negative result is one of the main reasons for the randomized bounds in previous solutions. Indeed, it is not clear at all how to obtain uniformity if even the size of the sampled domain is unknown.

对于基于时间戳的窗口，窗口的大小是未知的；此外，已有研究表明（例如，参见[36]），无法使用亚线性内存精确计算窗口的大小。这一负面结果是基于时间戳的窗口与所有其他模型（如仅插入流和具有显式删除的流（旋转门模型））之间的关键区别。事实上，这一负面结果是先前解决方案中采用随机界限的主要原因之一。的确，如果连采样域的大小都未知，根本不清楚如何实现均匀性。

Our key observation is that it is possible to sample from a window without even a knowledge of its size. As before, consider disjoint sets $A,B,C$ such that $C \subseteq  B \subseteq  A \cup  C$ and $A \cap  C = \varnothing$ . In the current scenario we do not assume that $\left| A\right|  = \left| B\right|$ and still obtain samples from $B$ . We show that if $\left| A\right|  \leq  \left| B\right|$ ,and it is possible to generate random events w.p. $\frac{\left| A\right| }{\left| B\right| }$ ,then it is possible to "combine" the samples from $A$ and $C$ into a sample from $B$ . The new rule is a generalization of our above ideas. We assign the sample from $A$ to be a sample from $B$ if the $A$ ’s sample belongs to $B$ (for technical reasons,we decrease the probability of this event by $\frac{\left| A\right| }{\left| B\right| }$ multiplicative factor). Otherwise,we assign the sample from $C$ to be the sample from $B$ . We show that this rule gives a uniform sample from $B$ .

我们的关键观察结果是，即使不知道窗口的大小，也有可能从窗口中进行采样。和之前一样，考虑不相交的集合$A,B,C$，使得$C \subseteq  B \subseteq  A \cup  C$且$A \cap  C = \varnothing$。在当前场景中，我们不假设$\left| A\right|  = \left| B\right|$，仍然可以从$B$中获取样本。我们证明了，如果$\left| A\right|  \leq  \left| B\right|$，并且有可能以概率$\frac{\left| A\right| }{\left| B\right| }$生成随机事件，那么就有可能将来自$A$和$C$的样本“组合”成来自$B$的样本。新规则是我们上述想法的推广。如果$A$的样本属于$B$，我们就将来自$A$的样本指定为来自$B$的样本（出于技术原因，我们将此事件的概率降低$\frac{\left| A\right| }{\left| B\right| }$倍）。否则，我们将来自$C$的样本指定为来自$B$的样本。我们证明了该规则能从$B$中得到均匀样本。

To apply this idea to sliding windows, we need to overcome two problems. First,we must be able to maintain such an $A$ and $C$ (as before we associate $B$ with our window). This task is nontrivial, since the size of the window is unknown. Our second key idea is a novel covering decomposition structure. Using this structure, we are able to maintain such an $A$ and $C$ at any moment. In fact,we obtain an important property that $\left| A\right|  \leq  \left| C\right|$ . This structure can be seen as a modification and generalization of the smooth histogram [19] method.

要将这一想法应用于滑动窗口，我们需要克服两个问题。首先，我们必须能够维护这样的$A$和$C$（和之前一样，我们将$B$与我们的窗口关联起来）。这项任务并非易事，因为窗口的大小是未知的。我们的第二个关键想法是一种新颖的覆盖分解结构。利用这种结构，我们能够在任何时刻维护这样的$A$和$C$。事实上，我们得到了一个重要性质，即$\left| A\right|  \leq  \left| C\right|$。这种结构可以看作是平滑直方图[19]方法的修改和推广。

Second,we need the ability to generate events w.p. $\frac{\left| A\right| }{\left| B\right| }$ which is still an unknown probability since $\left| B\right|$ is the size of our window. Our third key idea is a novel technique that we call generating implicit events. At the heart of our technique lies the idea of gradually decreasing the probabilities, starting from 1 , until we achieve the desired probability of $\left| A\right| /\left| B\right|$ . In particular,we show that it is possible to generate a non-uniform distribution over the elements of $A$ ,where the probability of picking an element is a function of the element's timestamp (or index). The function is constructed in such a way that the probability of picking an element among the last $i$ elements of $A$ is equal to $\frac{i}{\left| C\right|  + i}$ . That is,the probability of picking an expired element is $\frac{\left| C\right| }{\left| B\right| }$ . Since $\left| A\right|  \leq  \left| C\right|$ and since we know the values of $\left| A\right|$ and $\left| C\right|$ ,it is possible to generate events w.p. $\frac{\left| A\right| }{\left| B\right| }$ . The details can be found in the main body of this paper.

其次，我们需要能够以概率$\frac{\left| A\right| }{\left| B\right| }$生成事件，由于$\left| B\right|$是我们窗口的大小，这仍然是一个未知概率。我们的第三个关键想法是一种我们称之为生成隐式事件的新颖技术。我们技术的核心思想是从1开始逐渐降低概率，直到达到所需的概率$\left| A\right| /\left| B\right|$。特别地，我们证明了可以在$A$的元素上生成非均匀分布，其中选取一个元素的概率是该元素时间戳（或索引）的函数。该函数的构造方式使得在$A$的最后$i$个元素中选取一个元素的概率等于$\frac{i}{\left| C\right|  + i}$。也就是说，选取一个过期元素的概率是$\frac{\left| C\right| }{\left| B\right| }$。由于$\left| A\right|  \leq  \left| C\right|$，并且我们知道$\left| A\right|$和$\left| C\right|$的值，因此有可能以概率$\frac{\left| A\right| }{\left| B\right| }$生成事件。具体细节可在本文主体部分找到。

Black-Box Reduction. We show that a $k$ -sample without replacement may be generated from $k$ independent samples, ${R}_{0},\ldots ,{R}_{k - 1}$ . We apply our fourth key idea, a black-box reduction from sampling without replacement to sampling with replacement. The novelty of our approach is based on sampling from different domains; in fact, ${R}_{i}$ samples all but $i$ last active elements. Such samples can be generated if,in addition,we store the last $k$ elements.

黑盒归约。我们表明，可以从 $k$ 个独立样本 ${R}_{0},\ldots ,{R}_{k - 1}$ 中生成一个无放回的 $k$ 样本。我们应用第四个关键思想，即从无放回抽样到有放回抽样的黑盒归约。我们方法的新颖之处在于从不同的域进行抽样；实际上，${R}_{i}$ 对除最后 $i$ 个活跃元素之外的所有元素进行抽样。此外，如果我们存储最后 $k$ 个元素，就可以生成这样的样本。

Independence of Disjoint Windows. Our algorithms generate independent samples for non-overlapping windows. The independency follows from the nice property of the reservoir algorithm (that we use to generate samples in the buckets). Let ${R}_{1}$ be a sample generated for the bucket $B$ ,upon arrival of $i$ elements of $B$ . Let ${R}_{2}$ be a fraction of the final sample (i.e., the sample when the last element of $B$ arrives) that belongs to the last $\left| B\right|  - i$ elements. The reservoir algorithm implies that ${R}_{1}$ and ${R}_{2}$ are independent. Since the rest of the buckets contain independent samples as well, we conclude that our algorithms are independent for non-overlapping windows.

不相交窗口的独立性。我们的算法为不重叠的窗口生成独立样本。这种独立性源于蓄水池算法（我们用它在桶中生成样本）的良好性质。设 ${R}_{1}$ 是在桶 $B$ 的 $i$ 个元素到达时为该桶生成的样本。设 ${R}_{2}$ 是最终样本（即当桶 $B$ 的最后一个元素到达时的样本）中属于最后 $\left| B\right|  - i$ 个元素的部分。蓄水池算法表明 ${R}_{1}$ 和 ${R}_{2}$ 是相互独立的。由于其余桶也包含独立样本，我们得出结论，我们的算法对于不重叠的窗口是独立的。

### 1.4 Roadmap and Notations

### 1.4 路线图和符号说明

We use the following notations throughout our paper. We denote by $D$ a stream and by ${p}_{i},i \geq  0$ its $i$ -th element. For $0 \leq  x < y$ we define $\left\lbrack  {x,y}\right\rbrack   = \{ i,x \leq  i \leq  y\}$ . Finally,bucket $B\left( {x,y}\right)$ is the set of all stream elements between ${p}_{x}$ and ${p}_{y - 1} : B\left( {x,y}\right)  = \left\{  {{p}_{i},i \in  }\right.$ $\left\lbrack  {x,y - 1}\right\rbrack  \}$ .

我们在整篇论文中使用以下符号。我们用 $D$ 表示一个流，用 ${p}_{i},i \geq  0$ 表示它的第 $i$ 个元素。对于 $0 \leq  x < y$，我们定义 $\left\lbrack  {x,y}\right\rbrack   = \{ i,x \leq  i \leq  y\}$。最后，桶 $B\left( {x,y}\right)$ 是位于 ${p}_{x}$ 和 ${p}_{y - 1} : B\left( {x,y}\right)  = \left\{  {{p}_{i},i \in  }\right.$ $\left\lbrack  {x,y - 1}\right\rbrack  \}$ 之间的所有流元素的集合。

Our bounds are expressed in memory words; that is we assume that a single memory word is sufficient to store a stream element or its index or a timestamp.

我们的界限用内存字来表示；也就是说，我们假设一个内存字足以存储一个流元素、其索引或一个时间戳。

Section 2 presents sampling for sequence-based windows, with and without replacement. Sections 3 and 4 are devoted to sampling for timestamp-based windows, with and without replacement. Section 5 outlines possible applications for our approach. Due to the lack of space, some proofs are omitted and will be included in the full version of the paper.

第 2 节介绍基于序列的窗口的抽样，包括有放回和无放回抽样。第 3 节和第 4 节致力于基于时间戳的窗口的抽样，同样包括有放回和无放回抽样。第 5 节概述了我们方法的可能应用。由于篇幅限制，一些证明省略了，将包含在论文的完整版本中。

## 2. EQUIVALENT-WIDTH PARTITIONS AND SAMPLING FOR SEQUENCE-BASED WIN- DOWS

## 2. 等宽分区与基于序列的窗口抽样

### 2.1 Sampling With Replacement

### 2.1 有放回抽样

Let $n$ be the predefined size of a window. We say that a bucket is active if all its elements have arrived and at least one element is non-expired. We say that a bucket is partial if not all of its elements have arrived. We show below how to create a single random sample. To create a $k$ -random sample,we repeat the procedure $k$ times, independently.

设 $n$ 是窗口的预定义大小。如果一个桶的所有元素都已到达且至少有一个元素未过期，我们称该桶为活跃桶。如果一个桶的并非所有元素都已到达，我们称该桶为部分桶。下面我们将展示如何创建单个随机样本。要创建一个 $k$ 随机样本，我们独立地重复该过程 $k$ 次。

We divide $D$ into buckets $B\left( {{in},\left( {i + 1}\right) n}\right) ,i = 0,1,\ldots$ At any point in time, we have exactly one active bucket and at most one partial bucket. For every such bucket $B$ ,we independently generate a single sample, using the reservoir algorithm [72]. We denote this sample by ${X}_{B}$ .

我们将 $D$ 划分为桶 $B\left( {{in},\left( {i + 1}\right) n}\right) ,i = 0,1,\ldots$。在任何时间点，我们恰好有一个活跃桶和至多一个部分桶。对于每个这样的桶 $B$，我们使用蓄水池算法 [72] 独立地生成一个样本。我们用 ${X}_{B}$ 表示这个样本。

Let $B$ be a partial bucket and $C \subseteq  B$ be the set of all arrived elements. The properties of the reservoir algorithm imply that ${X}_{B}$ is a random sample of $C$ .

设 $B$ 是一个部分桶，$C \subseteq  B$ 是所有已到达元素的集合。蓄水池算法的性质表明，${X}_{B}$ 是 $C$ 的一个随机样本。

Below,we construct a random sample $Z$ of all non-expired elements. Let $U$ be the active bucket. If there is no partial bucket,then $U$ contains only all non-expired elements. Therefore, $Z = {X}_{U}$ is a valid sample. Otherwise,let $V$ be the partial bucket. Let ${U}_{e} = \{ x$ : $x \in  U,x$ is expired $\} ,{U}_{a} = \{ x : x \in  U,x$ is non-expired $\} ,{V}_{a} =$ $\{ x : x \in  V,x$ arrived $\}$ .

下面，我们构建所有未过期元素的随机样本$Z$。设$U$为活动桶。如果没有部分桶，那么$U$仅包含所有未过期元素。因此，$Z = {X}_{U}$是一个有效的样本。否则，设$V$为部分桶。设${U}_{e} = \{ x$：$x \in  U,x$已过期 $\} ,{U}_{a} = \{ x : x \in  U,x$未过期 $\} ,{V}_{a} =$ $\{ x : x \in  V,x$已到达 $\}$。

Note that $\left| {V}_{a}\right|  = \left| {U}_{e}\right|$ and let $s = \left| {V}_{a}\right|$ . Also,note that our window is ${U}_{a} \cup  {V}_{a}$ and ${X}_{V}$ is a random sample of ${V}_{a}$ . The random sample $Z$ is constructed as follows. If ${X}_{U}$ is not expired,we put $Z = {X}_{U}$ ,otherwise $Z = {X}_{V}$ . To prove the correctness,let $p$ be a non-expired element. If $p \in  {U}_{a}$ ,then $P\left( {Z = p}\right)  = P\left( {{X}_{U} = }\right.$ $p) = \frac{1}{n}$ . If $p \in  {V}_{a}$ ,then

注意$\left| {V}_{a}\right|  = \left| {U}_{e}\right|$，并设$s = \left| {V}_{a}\right|$。此外，注意我们的窗口是${U}_{a} \cup  {V}_{a}$，并且${X}_{V}$是${V}_{a}$的随机样本。随机样本$Z$的构建方式如下。如果${X}_{U}$未过期，我们放入$Z = {X}_{U}$，否则放入$Z = {X}_{V}$。为了证明其正确性，设$p$为一个未过期元素。如果$p \in  {U}_{a}$，那么$P\left( {Z = p}\right)  = P\left( {{X}_{U} = }\right.$ $p) = \frac{1}{n}$。如果$p \in  {V}_{a}$，那么

$$
P\left( {Z = p}\right)  = P\left( {{X}_{U} \in  {U}_{e},{X}_{V} = p}\right)  = 
$$

$$
P\left( {{X}_{U} \in  {U}_{e}}\right) P\left( {{X}_{V} = p}\right)  = \frac{s}{n}\frac{1}{s} = \frac{1}{n}.
$$

Therefore, $Z$ is a valid random sample. We need to store only samples of active or partial buckets. Since the number of such buckets is at most two and the reservoir algorithm requires $\Theta \left( 1\right)$ memory, the total memory of our algorithm for $k$ -sample is $\Theta \left( k\right)$ . Thus,

因此，$Z$是一个有效的随机样本。我们只需存储活动桶或部分桶的样本。由于此类桶的数量最多为两个，并且蓄水池算法需要$\Theta \left( 1\right)$的内存，我们的算法用于$k$ - 样本的总内存为$\Theta \left( k\right)$。因此，

THEOREM 2.1. It is possible to maintain $k$ -sampling with replacement for sequence-based windows using $O\left( k\right)$ memory words.

定理2.1。使用$O\left( k\right)$个内存字可以为基于序列的窗口维护有放回的$k$ - 抽样。

### 2.2 Sampling Without Replacement

### 2.2 无放回抽样

We can generalize the idea above to provide a $k$ -random sample without replacement. In this section $k$ -sample means $k$ -random sampling without replacement.

我们可以将上述想法进行推广，以提供无放回的$k$ - 随机样本。在本节中，$k$ - 样本指的是无放回的$k$ - 随机抽样。

We use the same buckets $B\left( {{in},\left( {i + 1}\right) n}\right) ,i = 0,1,\ldots$ . For every such bucket $B$ ,we independently generate a $k$ -sample ${X}_{B}$ , using the reservoir algorithm.

我们使用相同的桶$B\left( {{in},\left( {i + 1}\right) n}\right) ,i = 0,1,\ldots$。对于每个这样的桶$B$，我们使用蓄水池算法独立地生成一个$k$ - 样本${X}_{B}$。

Let $B$ be a partial bucket and $C \subseteq  B$ be the set of all arrived elements. The properties of the reservoir algorithm imply that either ${X}_{B} = C$ ,if $\left| C\right|  < k$ ,or ${X}_{B}$ is a $k$ -sample of $C$ . In both cases,we can generate an $i$ -sample of $C$ using ${X}_{B}$ only,for any $0 < i \leq  \min \left( {k,\left| C\right| }\right)$ .

设$B$为部分桶，$C \subseteq  B$为所有已到达元素的集合。蓄水池算法的性质表明，如果$\left| C\right|  < k$，则${X}_{B} = C$，或者${X}_{B}$是$C$的一个$k$ - 样本。在这两种情况下，对于任何$0 < i \leq  \min \left( {k,\left| C\right| }\right)$，我们都可以仅使用${X}_{B}$生成$C$的一个$i$ - 样本。

Our algorithm is as follows. Let $U$ be the active bucket. If there is no partial bucket,then $U$ contains only all active elements. Therefore,we can put $Z = {X}_{U}$ . Otherwise,let $V$ be the partial bucket. We define ${U}_{e},{U}_{a},{V}_{a},s$ as before and construct $Z$ as follows. If all elements of ${X}_{U}$ are not expired, $Z = {X}_{U}$ . Otherwise, let $i$ be the number of expired elements, $i = \left| {{U}_{e} \cap  {X}_{U}}\right|$ . As we mentioned before,we can generate an $i$ -sample of ${V}_{a}$ from ${X}_{V}$ , since $i \leq  \min \left( {k,s}\right)$ . We denote this sample as ${X}_{V}^{i}$ and put

我们的算法如下。设$U$为活动桶。如果没有部分桶，那么$U$仅包含所有活动元素。因此，我们可以放入$Z = {X}_{U}$。否则，设$V$为部分桶。我们如前定义${U}_{e},{U}_{a},{V}_{a},s$，并按如下方式构建$Z$。如果${X}_{U}$的所有元素都未过期，则$Z = {X}_{U}$。否则，设$i$为过期元素的数量，$i = \left| {{U}_{e} \cap  {X}_{U}}\right|$。如我们之前所述，由于$i \leq  \min \left( {k,s}\right)$，我们可以从${X}_{V}$生成${V}_{a}$的一个$i$ - 样本。我们将这个样本记为${X}_{V}^{i}$并放入

$$
Z = \left( {{X}_{U} \cap  {U}_{a}}\right)  \cup  {X}_{V}^{i}.
$$

We will prove now that $Z$ is a valid random sample. Let $Q =$ $\left\{  {{p}_{{j}_{1}},\ldots ,{p}_{{j}_{k}}}\right\}$ be a fixed set of $k$ non-expired elements such that ${j}_{1} < {j}_{2} < \ldots  < {j}_{k}$ . Let $i = \left| {Q \cap  {V}_{A}}\right|$ ,so $\left\{  {{p}_{{j}_{1}},\ldots ,{p}_{{j}_{k - i}}}\right\}   \subseteq  {U}_{a}$ and $\left\{  {{p}_{{j}_{k - i + 1}},\ldots ,{p}_{{j}_{k}}}\right\}   \subseteq  {V}_{a}$ . If $i = 0$ ,then $Q \subseteq  U$ and

我们现在将证明 $Z$ 是一个有效的随机样本。设 $Q =$ $\left\{  {{p}_{{j}_{1}},\ldots ,{p}_{{j}_{k}}}\right\}$ 是一组固定的 $k$ 个未过期元素，使得 ${j}_{1} < {j}_{2} < \ldots  < {j}_{k}$ 。设 $i = \left| {Q \cap  {V}_{A}}\right|$ ，因此 $\left\{  {{p}_{{j}_{1}},\ldots ,{p}_{{j}_{k - i}}}\right\}   \subseteq  {U}_{a}$ 且 $\left\{  {{p}_{{j}_{k - i + 1}},\ldots ,{p}_{{j}_{k}}}\right\}   \subseteq  {V}_{a}$ 。如果 $i = 0$ ，那么 $Q \subseteq  U$ 且

$$
P\left( {Z = Q}\right)  = P\left( {{X}_{U} = Q}\right)  = \frac{1}{\left( \begin{array}{l} n \\  k \end{array}\right) }.
$$

Otherwise,by independency of ${X}_{U}$ and ${X}_{V}^{i}$

否则，由于 ${X}_{U}$ 和 ${X}_{V}^{i}$ 相互独立

$$
P\left( {Z = Q}\right)  = 
$$

$$
P\left( {\left| {{X}_{U} \cap  {U}_{e}}\right|  = i,\left\{  {{p}_{{j}_{1}},\ldots ,{p}_{{j}_{k - i}}}\right\}   \subseteq  {X}_{U}}\right. \text{,}
$$

$$
{X}_{V}^{i} = \left\{  {{p}_{{j}_{k - i + 1}},\ldots ,{p}_{{j}_{k}}}\right\}  ) = 
$$

$$
P\left( {\left| {{X}_{U} \cap  {U}_{e}}\right|  = i,\left\{  {{p}_{{j}_{1}},\ldots ,{p}_{{j}_{k - i}}}\right\}   \subseteq  {X}_{U}}\right)  * 
$$

$$
P\left( {{X}_{V}^{i} = \left\{  {{p}_{{j}_{k - i + 1}},\ldots ,{p}_{{j}_{k}}}\right\}  }\right)  = 
$$

$$
\frac{\left( \begin{array}{l} s \\  i \end{array}\right) }{\left( \begin{array}{l} n \\  k \end{array}\right) } * \frac{1}{\left( \begin{array}{l} s \\  i \end{array}\right) } = \frac{1}{\left( \begin{array}{l} n \\  k \end{array}\right) }
$$

Therefore, $Z$ is a valid random sample of non-expired elements. Note that we store only samples of active or partial buckets. Since the number of such buckets is at most two and the reservoir algorithm requires $O\left( k\right)$ memory,the total memory of our algorithm is $O\left( k\right)$ . Thus,

因此， $Z$ 是未过期元素的一个有效随机样本。注意，我们仅存储活动桶或部分桶的样本。由于此类桶的数量最多为两个，并且蓄水池算法需要 $O\left( k\right)$ 的内存，因此我们算法的总内存为 $O\left( k\right)$ 。因此，

THEOREM 2.2. It is possible to maintain $k$ -sampling without replacement for sequence-based windows using $O\left( k\right)$ memory words.

定理 2.2：使用 $O\left( k\right)$ 个内存字，可以对基于序列的窗口进行无放回的 $k$ -抽样。

## 3. SAMPLING WITH REPLACEMENT FOR TIMESTAMP-BASED WINDOWS

## 3. 基于时间戳的窗口的有放回抽样

Let $n = n\left( t\right)$ be the number of non-expired elements. For each element $p$ ,timestamp $T\left( p\right)$ represents the moment of $p$ ’s entrance. For a window with (predefined) parameter ${t}_{0},p$ is active at time $t$ if $t - T\left( p\right)  < {t}_{0}$ . We show below how to create a single random sample. To create a $k$ -random sample,we repeat the procedure $k$ times, independently.

设 $n = n\left( t\right)$ 为未过期元素的数量。对于每个元素 $p$ ，时间戳 $T\left( p\right)$ 表示 $p$ 进入的时刻。对于一个具有（预定义）参数 ${t}_{0},p$ 的窗口，如果 $t - T\left( p\right)  < {t}_{0}$ ，则该窗口在时间 $t$ 处于活动状态。我们下面将展示如何创建单个随机样本。要创建一个 $k$ -随机样本，我们独立地重复该过程 $k$ 次。

### 3.1 Notations

### 3.1 符号说明

A bucket structure ${BS}\left( {x,y}\right)$ is a set

桶结构 ${BS}\left( {x,y}\right)$ 是一个集合

$$
\left\{  {{p}_{x},x,y,T\left( x\right) ,{R}_{x,y},{Q}_{x,y},r,q}\right\}  
$$

where $T\left( x\right)$ is a timestamp of ${p}_{x},{R}_{x,y}$ and ${Q}_{x,y}$ are independent random samples from $B\left( {x,y}\right)$ and $r,q$ are indexes of the picked (for random samples) elements. We denote by $N\left( t\right)$ the size of $D$ at the moment $t$ and by $l\left( t\right)$ the index of the earliest active element. Note that $N\left( t\right)  \leq  N\left( {t + 1}\right) ,l\left( t\right)  \leq  l\left( {t + 1}\right)$ and $T\left( {p}_{i}\right)  \leq  T\left( {p}_{i + 1}\right)$ .

其中 $T\left( x\right)$ 是 ${p}_{x},{R}_{x,y}$ 的时间戳， ${Q}_{x,y}$ 是来自 $B\left( {x,y}\right)$ 的独立随机样本， $r,q$ 是被选取（用于随机样本）元素的索引。我们用 $N\left( t\right)$ 表示时刻 $t$ 时 $D$ 的大小，用 $l\left( t\right)$ 表示最早活动元素的索引。注意， $N\left( t\right)  \leq  N\left( {t + 1}\right) ,l\left( t\right)  \leq  l\left( {t + 1}\right)$ 且 $T\left( {p}_{i}\right)  \leq  T\left( {p}_{i + 1}\right)$ 。

### 3.2 Covering Decomposition

### 3.2 覆盖分解

Let $a \leq  b$ be two indexes. A covering decomposition of a bucket $B\left( {a,b}\right) ,\zeta \left( {a,b}\right)$ ,is an ordered set of bucket structures with independent samples inductively defined below.

设 $a \leq  b$ 为两个索引。桶 $B\left( {a,b}\right) ,\zeta \left( {a,b}\right)$ 的覆盖分解是一个有序的桶结构集合，其独立样本定义如下。

$$
\zeta \left( {b,b}\right)  \mathrel{\text{:=}} {BS}\left( {b,b + 1}\right) ,
$$

and for $a < b$ ,

并且对于 $a < b$ ，

$$
\zeta \left( {a,b}\right)  \mathrel{\text{:=}} \langle {BS}\left( {a,c}\right) ,\zeta \left( {c,b}\right) \rangle ,
$$

where $c = a + {2}^{\lfloor \log \left( {b + 1 - a}\right) \rfloor  - 1}$ . Note that

其中 $c = a + {2}^{\lfloor \log \left( {b + 1 - a}\right) \rfloor  - 1}$ 。注意

$$
\left| {\zeta \left( {a,b}\right) }\right|  = O\left( {\log \left( {b - a}\right) }\right) ,
$$

so $\zeta \left( {a,b}\right)$ uses $O\left( {\log \left( {b - a}\right) }\right)$ memory.

因此 $\zeta \left( {a,b}\right)$ 使用 $O\left( {\log \left( {b - a}\right) }\right)$ 的内存。

Given ${p}_{b + 1}$ ,we inductively define an operator $\operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right)$ as follows.

给定 ${p}_{b + 1}$ ，我们如下归纳定义一个运算符 $\operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right)$ 。

$$
\operatorname{Incr}\left( {\zeta \left( {b,b}\right) }\right)  \mathrel{\text{:=}} \langle {BS}\left( {b,b + 1}\right) ,{BS}\left( {b + 1,b + 2}\right) \rangle .
$$

For $a < b$ ,we put

对于 $a < b$ ，我们令

$$
\operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right)  \mathrel{\text{:=}} \langle {BS}\left( {a,v}\right) ,\operatorname{Incr}\left( {\zeta \left( {v,b}\right) }\right) \rangle ,
$$

where $v$ is defined below.

其中 $v$ 的定义如下。

If $\lfloor \log \left( {b + 2 - a}\right) \rfloor  = \lfloor \log \left( {b + 1 - a}\right) \rfloor$ ,then we put $v = c$ , where ${BS}\left( {a,c}\right)$ is the first bucket structure of $\zeta \left( {a,b}\right)$ . Otherwise, we put $v = d$ ,where ${BS}\left( {c,d}\right)$ is the second bucket structure of $\zeta \left( {a,b}\right)$ . (Note that $\zeta \left( {a,b}\right)$ contains at least two buckets for $a < b$ .)

如果 $\lfloor \log \left( {b + 2 - a}\right) \rfloor  = \lfloor \log \left( {b + 1 - a}\right) \rfloor$ 成立，则我们令 $v = c$，其中 ${BS}\left( {a,c}\right)$ 是 $\zeta \left( {a,b}\right)$ 的第一个桶结构（bucket structure）。否则，我们令 $v = d$，其中 ${BS}\left( {c,d}\right)$ 是 $\zeta \left( {a,b}\right)$ 的第二个桶结构。（注意，对于 $a < b$，$\zeta \left( {a,b}\right)$ 至少包含两个桶。）

We show how to construct ${BS}\left( {a,d}\right)$ from ${BS}\left( {a,c}\right)$ and ${BS}\left( {c,d}\right)$ . We have in this case $\lfloor \log \left( {b + 2 - a}\right) \rfloor  = \lfloor \log \left( {b + 1 - a}\right) \rfloor  + 1$ , and therefore $b + 1 - a = {2}^{i} - 1$ for some $i \geq  2$ . Thus $c - a =$ ${2}^{\left\lfloor  {\log \left( {{2}^{i} - 1}\right) }\right\rfloor   - 1} = {2}^{i - 2}$ and

我们展示如何从 ${BS}\left( {a,c}\right)$ 和 ${BS}\left( {c,d}\right)$ 构造 ${BS}\left( {a,d}\right)$。在这种情况下，我们有 $\lfloor \log \left( {b + 2 - a}\right) \rfloor  = \lfloor \log \left( {b + 1 - a}\right) \rfloor  + 1$，因此对于某个 $i \geq  2$ 有 $b + 1 - a = {2}^{i} - 1$。于是 $c - a =$ ${2}^{\left\lfloor  {\log \left( {{2}^{i} - 1}\right) }\right\rfloor   - 1} = {2}^{i - 2}$ 且

$$
\lfloor \log \left( {b + 1 - c}\right) \rfloor  = \lfloor \log \left( {b + 1 - a - \left( {c - a}\right) }\right) \rfloor  = 
$$

$$
\left\lfloor  {\log \left( {{2}^{i} - {2}^{i - 2} - 1}\right) }\right\rfloor   = i - 1.
$$

Thus $d - c = {2}^{\lfloor \log \left( {b + 1 - c}\right) \rfloor  - 1} = {2}^{i - 2} = c - a$ . Now we can create ${BS}\left( {a,v}\right)$ by unifying ${BS}\left( {a,c}\right)$ and ${BS}\left( {c,d}\right)  : {BS}\left( {a,v}\right)  =$ $\left\{  {{p}_{a},d - a,{R}_{a,d},{Q}_{a,d},{r}^{\prime },{q}^{\prime }}\right\}$ . We put ${R}_{a,d} = {R}_{a,c}$ with probability $\frac{1}{2}$ and ${R}_{a,d} = {R}_{c,d}$ otherwise. Since $d - c = c - a$ ,and ${R}_{c,d},{\widetilde{R}}_{a,c}$ are distributed uniformly,we conclude that ${R}_{a,d}$ is distributed uniformly as well. ${Q}_{a,d}$ is defined similarly and ${r}^{\prime },{q}^{\prime }$ are indexes of the chosen samples. Finally, the new samples are independent of the rest of $\zeta$ ’s samples. Note also that $\operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right)$ requires $O\left( {\log \left( {b - a}\right) }\right)$ operations.

因此 $d - c = {2}^{\lfloor \log \left( {b + 1 - c}\right) \rfloor  - 1} = {2}^{i - 2} = c - a$ 。现在我们可以通过合并 ${BS}\left( {a,c}\right)$ 和 ${BS}\left( {c,d}\right)  : {BS}\left( {a,v}\right)  =$ $\left\{  {{p}_{a},d - a,{R}_{a,d},{Q}_{a,d},{r}^{\prime },{q}^{\prime }}\right\}$ 来创建 ${BS}\left( {a,v}\right)$ 。我们以概率 $\frac{1}{2}$ 放入 ${R}_{a,d} = {R}_{a,c}$ ，否则放入 ${R}_{a,d} = {R}_{c,d}$ 。由于 $d - c = c - a$ 和 ${R}_{c,d},{\widetilde{R}}_{a,c}$ 是均匀分布的，我们得出 ${R}_{a,d}$ 也是均匀分布的。 ${Q}_{a,d}$ 的定义类似， ${r}^{\prime },{q}^{\prime }$ 是所选样本的索引。最后，新样本与 $\zeta$ 的其余样本相互独立。另请注意， $\operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right)$ 需要 $O\left( {\log \left( {b - a}\right) }\right)$ 次操作。

LEMMA 3.1. For any $a$ and $b,\operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right)  = \zeta \left( {a,b + 1}\right)$ .

引理3.1。对于任意 $a$ 和 $b,\operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right)  = \zeta \left( {a,b + 1}\right)$ 。

Proof. We prove the lemma by induction on $b - a$ . If $a = b$ then,since $b + 1 = b + {2}^{\left\lfloor  {\log \left( {\left( {b + 1}\right)  + 1 - b}\right) }\right\rfloor   - 1}$ ,we have,by definition of $\zeta \left( {b,b + 1}\right)$ ,

证明。我们通过对 $b - a$ 进行归纳来证明该引理。如果 $a = b$ ，那么，由于 $b + 1 = b + {2}^{\left\lfloor  {\log \left( {\left( {b + 1}\right)  + 1 - b}\right) }\right\rfloor   - 1}$ ，根据 $\zeta \left( {b,b + 1}\right)$ 的定义，我们有

$$
\zeta \left( {b,b + 1}\right)  = \langle {BS}\left( {b,b + 1}\right) ,\zeta \left( {b + 1,b + 1}\right) \rangle  = 
$$

$$
\langle {BS}\left( {b,b + 1}\right) ,{BS}\left( {b + 1,b + 2}\right) \rangle  = \operatorname{Incr}\left( {\zeta \left( {b,b}\right) }\right) .
$$

We assume that the lemma is correct for $b - a < h$ and prove it for $b - a = h$ . Let ${BS}\left( {a,v}\right)$ be the first bucket of $\operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right)$ . Let ${BS}\left( {a,c}\right)$ be the first bucket of $\zeta \left( {a,b}\right)$ . By definition,if $\lfloor \log (b +$ $2 - a)\rbrack  = \left\lfloor  {\log \left( {b + 1 - a}\right) }\right\rfloor$ then $v = c$ . We have

我们假设该引理对于 $b - a < h$ 是正确的，并证明它对于 $b - a = h$ 也成立。设 ${BS}\left( {a,v}\right)$ 是 $\operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right)$ 的第一个桶。设 ${BS}\left( {a,c}\right)$ 是 $\zeta \left( {a,b}\right)$ 的第一个桶。根据定义，如果 $\lfloor \log (b +$ $2 - a)\rbrack  = \left\lfloor  {\log \left( {b + 1 - a}\right) }\right\rfloor$ ，那么 $v = c$ 。我们有

$$
v = c = a + {2}^{\lfloor \log \left( {b + 1 - a}\right) \rfloor  - 1} = a + {2}^{\lfloor \log \left( {b + 2 - a}\right) \rfloor  - 1}.
$$

Otherwise,let ${BS}\left( {c,d}\right)$ be the second bucket of $\zeta \left( {a,b}\right)$ . We have from above $\lfloor \log \left( {b + 2 - a}\right) \rfloor  = \lfloor \log \left( {b + 1 - a}\right) \rfloor  + 1,d - c = c - a$ and $v = d$ . Thus

否则，设 ${BS}\left( {c,d}\right)$ 是 $\zeta \left( {a,b}\right)$ 的第二个桶。由上述内容我们有 $\lfloor \log \left( {b + 2 - a}\right) \rfloor  = \lfloor \log \left( {b + 1 - a}\right) \rfloor  + 1,d - c = c - a$ 和 $v = d$ 。因此

$$
v = d = {2c} - a = 2\left( {a + {2}^{\lfloor \log \left( {b + 1 - a}\right) \rfloor  - 1}}\right)  - a = 
$$

$$
a + {2}^{\lfloor \log \left( {b + 1 - a}\right) \rfloor } = a + {2}^{\lfloor \log \left( {b + 2 - a}\right) \rfloor  - 1}.
$$

In both cases $v = a + {2}^{\left\lfloor  {\log \left( {\left( {b + 1}\right)  + 1 - a}\right) }\right\rfloor   - 1}$ and,by definition of $\zeta$

在这两种情况下 $v = a + {2}^{\left\lfloor  {\log \left( {\left( {b + 1}\right)  + 1 - a}\right) }\right\rfloor   - 1}$ ，并且根据 $\zeta$ 的定义

$$
\zeta \left( {a,b + 1}\right)  = \langle {BS}\left( {a,v}\right) ,\zeta \left( {v,b + 1}\right) \rangle .
$$

By induction,since $b - v < h$ ,we have $\operatorname{Incr}\left( {\zeta \left( {v,b}\right) }\right)  = \zeta (v,b +$ 1). Thus

通过归纳法，由于 $b - v < h$ ，我们有 $\operatorname{Incr}\left( {\zeta \left( {v,b}\right) }\right)  = \zeta (v,b +$ 1)。因此

$$
\zeta \left( {a,b + 1}\right)  = \langle {BS}\left( {a,v}\right) ,\zeta \left( {v,b + 1}\right) \rangle  = 
$$

$$
\langle {BS}\left( {a,v}\right) ,\operatorname{Incr}\left( {\zeta \left( {v,b}\right) }\right) \rangle  = \operatorname{Incr}\left( {\zeta \left( {a,b}\right) }\right) .
$$

LEMMA 3.2. For any $t$ with a positive number of active elements, we are able to maintain one of the following:

引理3.2。对于任何具有正数量活跃元素的 $t$ ，我们能够维持以下情况之一：

1. $\zeta \left( {l\left( t\right) ,N\left( t\right) }\right)$ ,

or

或者

2. ${BS}\left( {{y}_{t},{z}_{t}}\right) ,\zeta \left( {{z}_{t},N\left( t\right) }\right)$ ,

where ${y}_{t} < l\left( t\right)  \leq  {z}_{t},{z}_{t} - {y}_{t} \leq  N\left( t\right)  + 1 - {z}_{t}$ and all random samples are independent.

其中 ${y}_{t} < l\left( t\right)  \leq  {z}_{t},{z}_{t} - {y}_{t} \leq  N\left( t\right)  + 1 - {z}_{t}$ 且所有随机样本相互独立。

### 3.3 Generating Implicit Events

### 3.3 生成隐式事件

We use the following notations for this section. Let ${B}_{1} = B\left( {a,b}\right)$ and ${B}_{2} = B\left( {b,N\left( t\right)  + 1}\right)$ be two buckets such that ${p}_{a}$ is expired, ${p}_{b}$ is active and $\left| {B}_{1}\right|  \leq  \left| {B}_{2}\right|$ . Let $B{S}_{1}$ and $B{S}_{2}$ be corresponding bucket structures,with independent random samples ${R}_{1},{Q}_{1}$ and ${R}_{2},{Q}_{2}$ . We put $\alpha  = b - a$ and $\beta  = N\left( t\right)  + 1 - b$ . Let $\gamma$ be the (unknown) number of non-expired elements inside ${B}_{1}$ ,so $n = \beta  + \gamma$ . We stress that $\alpha ,\beta$ are known and $\gamma$ is unknown.

我们在本节使用以下符号。设 ${B}_{1} = B\left( {a,b}\right)$ 和 ${B}_{2} = B\left( {b,N\left( t\right)  + 1}\right)$ 为两个桶，使得 ${p}_{a}$ 已过期，${p}_{b}$ 处于活动状态且 $\left| {B}_{1}\right|  \leq  \left| {B}_{2}\right|$ 。设 $B{S}_{1}$ 和 $B{S}_{2}$ 为相应的桶结构，具有独立的随机样本 ${R}_{1},{Q}_{1}$ 和 ${R}_{2},{Q}_{2}$ 。我们令 $\alpha  = b - a$ 和 $\beta  = N\left( t\right)  + 1 - b$ 。设 $\gamma$ 为 ${B}_{1}$ 内未过期元素的（未知）数量，因此 $n = \beta  + \gamma$ 。我们强调 $\alpha ,\beta$ 是已知的，而 $\gamma$ 是未知的。

LEMMA 3.3. It is possible to generate a random sample $Y =$ $Y\left( {Q}_{1}\right)$ of ${B}_{1}$ ,with the following distribution:

引理 3.3。可以生成 ${B}_{1}$ 的一个随机样本 $Y =$ $Y\left( {Q}_{1}\right)$ ，其分布如下：

$$
P\left( {Y = {p}_{b - i}}\right)  = \frac{\beta }{\left( {\beta  + i}\right) \left( {\beta  + i - 1}\right) },\;0 < i < \alpha ,
$$

$$
P\left( {Y = {p}_{a}}\right)  = \frac{\beta }{\beta  + \alpha  - 1}.
$$

$Y$ is independent of ${R}_{1},{R}_{2},{Q}_{2}$ and can be generated within constant memory and time,using ${Q}_{1}$ .

$Y$ 与 ${R}_{1},{R}_{2},{Q}_{2}$ 相互独立，并且可以使用 ${Q}_{1}$ 在恒定的内存和时间内生成。

Proof. Let ${\left\{  {H}_{j}\right\}  }_{j = 1}^{\alpha  - 1}$ be a set of zero-one independent random variables such that

证明。设 ${\left\{  {H}_{j}\right\}  }_{j = 1}^{\alpha  - 1}$ 为一组零 - 一独立随机变量，使得

$$
P\left( {{H}_{j} = 1}\right)  = \frac{\alpha \beta }{\left( {\beta  + j}\right) \left( {\beta  + j - 1}\right) }.
$$

Let $D = {B}_{1} \times  \{ 0,1{\} }^{\alpha  - 1}$ and $Z$ be the random vector with values from $D,Z = \left\langle  {{Q}_{1},{H}_{1},\ldots ,{H}_{\alpha  - 1}}\right\rangle$ . Let ${\left\{  {A}_{i}\right\}  }_{i = 1}^{\alpha }$ be a set of subsets of $D$ :

设 $D = {B}_{1} \times  \{ 0,1{\} }^{\alpha  - 1}$ 和 $Z$ 为取值来自 $D,Z = \left\langle  {{Q}_{1},{H}_{1},\ldots ,{H}_{\alpha  - 1}}\right\rangle$ 的随机向量。设 ${\left\{  {A}_{i}\right\}  }_{i = 1}^{\alpha }$ 为 $D$ 的子集的集合：

$$
{A}_{i} = \left\{  \left\langle  {{q}_{b - i},{a}_{1},\ldots ,{a}_{i - 1},1,{a}_{i + 1},\ldots ,{a}_{\alpha  - 1}}\right\rangle  \right. 
$$

$$
{a}_{j} \in  \{ 0,1\} ,j \neq  i\} \text{.}
$$

Finally we define $Y$ as follows

最后，我们如下定义 $Y$

$$
Y = \left\{  \begin{array}{ll} {q}_{b - i}, & \text{ if }Z \in  {A}_{i},1 \leq  i < \alpha , \\  {q}_{a}, & \text{ otherwise } \end{array}\right. 
$$

Since ${Q}_{1}$ is independent of ${R}_{1},{R}_{2},{Q}_{2},Y$ is independent of them as well. We have

由于 ${Q}_{1}$ 与 ${R}_{1},{R}_{2},{Q}_{2},Y$ 相互独立，${R}_{1},{R}_{2},{Q}_{2},Y$ 也与它们相互独立。我们有

$$
P\left( {Y = {p}_{b - i}}\right)  = P\left( {Z \in  {A}_{i}}\right)  = 
$$

$$
P\left( {{Q}_{1} = {q}_{b - i},{H}_{i} = 1,{H}_{j} \in  \{ 0,1\} \text{ for }j \neq  i}\right)  = 
$$

$$
P\left( {{Q}_{1} = {q}_{b - i}}\right) P\left( {{H}_{i} = 1}\right) P\left( {{H}_{j} \in  \{ 0,1\} \text{ for }j \neq  i}\right)  = 
$$

$$
P\left( {{Q}_{1} = {q}_{b - i}}\right) P\left( {{H}_{i} = 1}\right)  = 
$$

$$
\frac{1}{\alpha }\frac{\alpha \beta }{\left( {\beta  + i}\right) \left( {\beta  + i - 1}\right) } = \frac{\beta }{\left( {\beta  + i}\right) \left( {\beta  + i - 1}\right) }.
$$

Also,

此外，

$$
P\left( {Y = {p}_{a}}\right)  = 1 - \mathop{\sum }\limits_{{i = 1}}^{{\alpha  - 1}}P\left( {Y = {p}_{b - i}}\right)  = 
$$

$$
1 - \mathop{\sum }\limits_{{i = 1}}^{{\alpha  - 1}}\frac{\beta }{\left( {\beta  + i}\right) \left( {\beta  + i - 1}\right) } = 
$$

$$
1 - \beta \mathop{\sum }\limits_{{i = 1}}^{{\alpha  - 1}}\left( {\frac{1}{\beta  + i - 1} - \frac{1}{\beta  + i}}\right)  = 
$$

$$
1 - \beta \left( {\frac{1}{\beta } - \frac{1}{\beta  + \alpha  - 1}}\right)  = \frac{\beta }{\beta  + \alpha  - 1}.
$$

By definition of ${A}_{i}$ ,the value of $Y$ is uniquely defined by ${Q}_{1}$ and exactly one $H$ . Therefore,the generation of the whole vector $Z$ is not necessary. Instead,we can calculate $Y$ by the following simple procedure. Once we know the index of ${Q}_{1}$ ’s value,we generate the corresponding ${H}_{i}$ and calculate the value of $Y$ . We can omit the generation of other $H\mathrm{\;s}$ ,and therefore we need constant time and memory.

根据 ${A}_{i}$ 的定义，$Y$ 的值由 ${Q}_{1}$ 和恰好一个 $H$ 唯一确定。因此，不需要生成整个向量 $Z$ 。相反，我们可以通过以下简单过程计算 $Y$ 。一旦我们知道 ${Q}_{1}$ 的值的索引，我们就生成相应的 ${H}_{i}$ 并计算 $Y$ 的值。我们可以省略其他 $H\mathrm{\;s}$ 的生成，因此我们只需要恒定的时间和内存。

LEMMA 3.4. It is possible to generate a zero-one random variable $X$ such that $P\left( {X = 1}\right)  = \frac{\alpha }{\beta  + \gamma }.X$ is independent of ${R}_{1},{R}_{2},{Q}_{2}$ and can be generated using constant time and memory.

引理 3.4。可以生成一个零 - 一随机变量 $X$ ，使得 $P\left( {X = 1}\right)  = \frac{\alpha }{\beta  + \gamma }.X$ 与 ${R}_{1},{R}_{2},{Q}_{2}$ 相互独立，并且可以在恒定的时间和内存内生成。

Proof. Since $\gamma$ is unknown,it cannot be generated by flipping a coin; a slightly more complicated procedure is required.

证明。由于 $\gamma$ 是未知的，不能通过抛硬币来生成它；需要一个稍微复杂一些的过程。

Let $Y\left( {Q}_{1}\right)$ be the random variable from Lemma 3.3. We have

设 $Y\left( {Q}_{1}\right)$ 为引理 3.3 中的随机变量。我们有

$$
P\left( {\mathrm{Y}\text{ is not expired }}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\gamma }P\left( {Y = {q}_{b - i}}\right)  = 
$$

$$
\mathop{\sum }\limits_{{i = 1}}^{\gamma }\frac{\beta }{\left( {\beta  + i}\right) \left( {\beta  + i - 1}\right) } = 
$$

$$
\beta \mathop{\sum }\limits_{{i = 1}}^{\gamma }\left( {\frac{1}{\beta  + i - 1} - \frac{1}{\beta  + i}}\right)  = 
$$

$$
\beta \left( {\frac{1}{\beta } - \frac{1}{\beta  + \gamma }}\right)  = \frac{\gamma }{\beta  + \gamma }.
$$

Therefore $P\left( {\mathrm{\;Y}\text{is expired}}\right)  = \frac{\beta }{\beta  + \gamma }$ .

因此 $P\left( {\mathrm{\;Y}\text{is expired}}\right)  = \frac{\beta }{\beta  + \gamma }$ 。

Let $S$ be a zero-one variable,independent of ${R}_{1},{R}_{2},{Q}_{2},Y$ such that

设 $S$ 为一个零 - 一变量，与 ${R}_{1},{R}_{2},{Q}_{2},Y$ 相互独立，使得

$$
P\left( {S = 1}\right)  = \frac{\alpha }{\beta }.
$$

We put

我们放置

$$
X = \left\{  \begin{array}{ll} 1, & \text{ if }Y\text{ is expired AND }S = 1, \\  0, & \text{ otherwise. } \end{array}\right. 
$$

We have

我们有

$$
P\left( {X = 1}\right)  = P\left( {Y\text{ is expired,}S = 1}\right)  = 
$$

$$
P\left( {Y\text{ is expired }}\right) P\left( {S = 1}\right)  = \frac{\beta }{\beta  + \gamma }\frac{\alpha }{\beta } = \frac{\alpha }{\beta  + \gamma }.
$$

Since $Y$ and $S$ are independent of ${R}_{1},{R}_{2},{Q}_{2},X$ is independent of them as well. Since we can determine if $Y$ is expired within constant time, we need a constant amount of time and memory.

由于 $Y$ 和 $S$ 与 ${R}_{1},{R}_{2},{Q}_{2},X$ 无关，因此 ${R}_{1},{R}_{2},{Q}_{2},X$ 也与它们无关。由于我们可以在常量时间内确定 $Y$ 是否过期，所以我们需要常量的时间和内存。

LEMMA 3.5. It is possible to construct a random sample $V$ of all non-expired elements using only the data of $B{S}_{1},B{S}_{2}$ and constant time and memory.

引理 3.5。仅使用 $B{S}_{1},B{S}_{2}$ 的数据以及常量的时间和内存，就有可能构造出所有未过期元素的随机样本 $V$。

Proof. Our goal is to generate a random variable $V$ that chooses a non-expired element w.p. $\frac{1}{\beta  + \gamma }$ . Let $X$ be the random variable generated in Lemma 3.4. We define $V$ as follows.

证明。我们的目标是生成一个随机变量 $V$，它以概率 $\frac{1}{\beta  + \gamma }$ 选择一个未过期的元素。设 $X$ 是引理 3.4 中生成的随机变量。我们将 $V$ 定义如下。

$$
V = \left\{  \begin{array}{ll} {R}_{1}, & {R}_{1}\text{ is not expired AND }X = 1, \\  {R}_{2}, & \text{ otherwise. } \end{array}\right. 
$$

Let $p$ be a non-expired element. If $p \in  {B}_{1}$ ,then since $X$ is independent of ${R}_{1}$ ,we have

设 $p$ 是一个未过期的元素。如果 $p \in  {B}_{1}$，那么由于 $X$ 与 ${R}_{1}$ 无关，我们有

$$
P\left( {V = p}\right)  = P\left( {{R}_{1} = p,X = 1}\right)  = 
$$

$$
P\left( {{R}_{1} = p}\right) P\left( {X = 1}\right)  = \frac{1}{\alpha }\frac{\alpha }{\beta  + \gamma } = \frac{1}{\beta  + \gamma } = \frac{1}{n}.
$$

If $p \in  {B}_{2}$ ,then

如果 $p \in  {B}_{2}$，那么

$$
P\left( {V = p}\right)  = 
$$

$\left( {1 - P\left( {{R}_{1}\text{ is not expired }}\right) P\left( {X = 1}\right) )P\left( {{R}_{2} = p}\right)  = }\right.$

$$
\left( {1 - \frac{\gamma }{\alpha }\frac{\alpha }{\beta  + \gamma }}\right) \frac{1}{\beta } = \frac{1}{\beta  + \gamma } = \frac{1}{n}.
$$

### 3.4 Main Results

### 3.4 主要结果

THEOREM 3.6. We can maintain a random sample over all non-expired elements using $\Theta \left( {\log n}\right)$ memory.

定理 3.6。我们可以使用 $\Theta \left( {\log n}\right)$ 的内存来维护所有未过期元素的随机样本。

Proof. By using Lemma 3.2, we are able to maintain one of two cases. If case 1 occurs, we can combine random variables of all bucket structures with appropriate probabilities and get a random sample of all non-expired elements. If case 2 occurs, we use notations of Section 3.3,interpret the first bucket as ${B}_{1}$ and combine buckets of covering decomposition to generate samples from ${B}_{2}$ . Properties of the second case imply $\left| {B}_{1}\right|  \leq  \left| {B}_{2}\right|$ and therefore, by using Lemma 3.5, we are able to produce a random sample as well. All procedures described in the lemmas require $\Theta \left( {\log n}\right)$ memory. Therefore, the theorem is correct.

证明。通过使用引理 3.2，我们能够处理两种情况之一。如果情况 1 发生，我们可以将所有桶结构的随机变量以适当的概率组合起来，得到所有未过期元素的随机样本。如果情况 2 发生，我们使用第 3.3 节的符号，将第一个桶解释为 ${B}_{1}$，并将覆盖分解的桶组合起来以从 ${B}_{2}$ 中生成样本。第二种情况的性质意味着 $\left| {B}_{1}\right|  \leq  \left| {B}_{2}\right|$，因此，通过使用引理 3.5，我们也能够生成一个随机样本。引理中描述的所有过程都需要 $\Theta \left( {\log n}\right)$ 的内存。因此，该定理是正确的。

LEMMA 3.7. The memory usage of maintaining a random sample within a timestamp-based window has a lower bound $\Omega \left( {\log \left( n\right) }\right)$ .

引理 3.7。在基于时间戳的窗口内维护随机样本的内存使用有一个下界 $\Omega \left( {\log \left( n\right) }\right)$。

Proof. Let $D$ be a stream with the following property. For timestamp $i,0 \leq  i \leq  2{t}_{0}$ ,we have ${2}^{2{t}_{0} - i}$ elements and for $i >$ $2{t}_{0}$ ,we have exactly one element per timestamp.

证明。设 $D$ 是一个具有以下性质的流。对于时间戳 $i,0 \leq  i \leq  2{t}_{0}$，我们有 ${2}^{2{t}_{0} - i}$ 个元素，对于 $i >$ $2{t}_{0}$，每个时间戳恰好有一个元素。

For timestamp $0 \leq  i \leq  {t}_{0}$ ,the probability of choosing $p$ with $T\left( p\right)  = i$ at the moment ${t}_{0} + i - 1$ is

对于时间戳 $0 \leq  i \leq  {t}_{0}$，在时刻 ${t}_{0} + i - 1$ 用 $T\left( p\right)  = i$ 选择 $p$ 的概率是

$$
\frac{{2}^{2{t}_{0} - i}}{\mathop{\sum }\limits_{{j = i}}^{{i + {t}_{0} - 1}}{2}^{2{t}_{0} - j}} = \frac{{2}^{2{t}_{0} - i}}{{2}^{{t}_{0} - i + 1}\mathop{\sum }\limits_{{j = 0}}^{{{t}_{0} - 1}}{2}^{{t}_{0} - j - 1}} = 
$$

$$
\frac{{2}^{{t}_{0} - 1}}{\mathop{\sum }\limits_{{j = 0}}^{{{t}_{0} - 1}}{2}^{j}} = \frac{{2}^{{t}_{0} - 1}}{{2}^{{t}_{0}} - 1} > \frac{1}{2}.
$$

Therefore, the expected number of distinct timestamps that will be picked between moments ${t}_{0} - 1$ and $2{t}_{0} - 1$ is at least $\mathop{\sum }\limits_{{i = {t}_{0} - 1}}^{{2{t}_{0} - 1}}\frac{1}{2} =$ $\frac{{t}_{0} + 1}{2}$ . So,with a positive probability we need to keep in memory at least $\frac{{t}_{0}}{2}$ distinct elements at the moment ${t}_{0}$ . The number of active elements $n$ at this moment is at least ${2}^{{t}_{0}}$ . Therefore the memory usage at this moment is $\Omega \left( {\log n}\right)$ ,with positive probability. We can conclude that $\log \left( n\right)$ is a lower bound for memory usage.

因此，在时刻 ${t}_{0} - 1$ 和 $2{t}_{0} - 1$ 之间将被选中的不同时间戳的期望数量至少为 $\mathop{\sum }\limits_{{i = {t}_{0} - 1}}^{{2{t}_{0} - 1}}\frac{1}{2} =$ $\frac{{t}_{0} + 1}{2}$。所以，以正概率，我们在时刻 ${t}_{0}$ 需要在内存中保留至少 $\frac{{t}_{0}}{2}$ 个不同的元素。此时活动元素 $n$ 的数量至少为 ${2}^{{t}_{0}}$。因此，此时的内存使用量为 $\Omega \left( {\log n}\right)$，且概率为正。我们可以得出结论，$\log \left( n\right)$ 是内存使用的下界。

## 4. BLACK-BOX REDUCTION

## 4. 黑盒归约

In this section, we present black-box reduction from sampling without replacement to sampling with replacement. As a result, we obtain an optimal algorithm for sampling without replacement for timestamp-based windows. Informally, the idea is as follows. We maintain $k$ independent random samples ${R}_{0},\ldots ,{R}_{k - 1}$ of active elements, using the algorithm from Section 3. The difference between these samples and the $k$ -sample with replacement is that ${R}_{i}$ samples all active elements except the last $i$ . This can be done using $O\left( {k + k\log n}\right)$ memory. Finally,a $k$ -sample without replacement can be generated using ${R}_{0},\ldots ,{R}_{k - 1}$ only.

在本节中，我们介绍了从无放回抽样到有放回抽样的黑盒归约。因此，我们得到了一种基于时间戳窗口的无放回抽样的最优算法。简单来说，其思路如下。我们使用第3节中的算法维护$k$个独立的活动元素随机样本${R}_{0},\ldots ,{R}_{k - 1}$。这些样本与有放回的$k$样本的区别在于，${R}_{i}$对除最后$i$个元素之外的所有活动元素进行抽样。这可以使用$O\left( {k + k\log n}\right)$的内存来完成。最后，仅使用${R}_{0},\ldots ,{R}_{k - 1}$就可以生成一个无放回的$k$样本。

Let us describe the algorithm in detail. First,we construct ${R}_{i}$ . To do this,we maintain an auxiliary array with the last $i$ elements. We repeat all procedures in Section 3,but we "delay" the last $i$ elements. An element is added to covering decomposition only when more then $i$ elements arrive after it. We prove the following variant of Lemma 3.2.

让我们详细描述该算法。首先，我们构造${R}_{i}$。为此，我们维护一个包含最后$i$个元素的辅助数组。我们重复第3节中的所有步骤，但我们“延迟”处理最后$i$个元素。只有当一个元素之后有超过$i$个元素到达时，才将其添加到覆盖分解中。我们证明引理3.2的以下变体。

LEMMA 4.1. Let $0 < i \leq  k$ . For any $t$ with more then $i$ active elements, we are able to maintain one of the following:

引理4.1。设$0 < i \leq  k$。对于任何具有超过$i$个活动元素的$t$，我们能够维护以下情况之一：

1. $\zeta \left( {l\left( t\right) ,N\left( t\right)  - i}\right)$ ,

or

或者

2. ${BS}\left( {{y}_{t},{z}_{t}}\right) ,\zeta \left( {{z}_{t},N\left( t\right)  - i}\right)$ ,

where ${y}_{t} < l\left( t\right)  \leq  {z}_{t}$ and ${z}_{t} - {y}_{t} \leq  N\left( t\right)  + 1 - i - {z}_{t}$ and all random samples of the bucket structures are independent.

其中${y}_{t} < l\left( t\right)  \leq  {z}_{t}$和${z}_{t} - {y}_{t} \leq  N\left( t\right)  + 1 - i - {z}_{t}$，并且桶结构的所有随机样本都是独立的。

The rest of the procedure remains the same. Note that we can use the same array for every $i$ ,and therefore we can construct ${R}_{0},\ldots ,{R}_{k - 1}$ using $\Theta \left( {k + k\log n}\right)$ memory.

其余步骤保持不变。注意，我们可以为每个$i$使用相同的数组，因此我们可以使用$\Theta \left( {k + k\log n}\right)$的内存构造${R}_{0},\ldots ,{R}_{k - 1}$。

In the reminder of this section,we show how ${R}_{0},\ldots ,{R}_{k - 1}$ can be used to generate a $k$ -sample without replacement. We denote by ${R}_{i}^{j}$ a $i$ -random sample without replacement from $\left\lbrack  {1,j}\right\rbrack$ .

在本节的剩余部分，我们展示如何使用${R}_{0},\ldots ,{R}_{k - 1}$生成一个无放回的$k$样本。我们用${R}_{i}^{j}$表示从$\left\lbrack  {1,j}\right\rbrack$中无放回抽取的$i$随机样本。

LEMMA 4.2. ${R}_{a + 1}^{b + 1}$ can be generated using independent ${R}_{a}^{b}$ , ${R}_{1}^{b + 1}$ samples only.

引理4.2。仅使用独立的${R}_{a}^{b}$、${R}_{1}^{b + 1}$样本就可以生成${R}_{a + 1}^{b + 1}$。

PROOF. The algorithm is as follows.

证明。算法如下。

$$
{R}_{a + 1}^{b + 1} = \left\{  \begin{array}{ll} {R}_{a}^{b} \cup  \{ b + 1\} , & \text{ if }{R}_{1}^{b + 1} \in  {R}_{a}^{b}, \\  {R}_{a}^{b} \cup  {R}_{1}^{b + 1}, & \text{ otherwise }. \end{array}\right. 
$$

Let $X = \left\{  {{x}_{1},\ldots ,{x}_{a + 1}}\right\}$ be a set of points from $\left\lbrack  {1,b + 1}\right\rbrack$ ,such that ${x}_{1} < {x}_{2} < \cdots  < {x}_{a} < {x}_{a + 1}$ .

设$X = \left\{  {{x}_{1},\ldots ,{x}_{a + 1}}\right\}$是来自$\left\lbrack  {1,b + 1}\right\rbrack$的点集，使得${x}_{1} < {x}_{2} < \cdots  < {x}_{a} < {x}_{a + 1}$。

If ${x}_{a + 1} < b + 1$ ,then we have

如果${x}_{a + 1} < b + 1$，那么我们有

$$
P\left( {{R}_{a + 1}^{b + 1} = X}\right)  = P\left( {\mathop{\bigcup }\limits_{{j = 1}}^{{a + 1}}\left( {{R}_{1}^{b + 1} = {x}_{j} \cap  {R}_{a}^{b} = X \smallsetminus  \left\{  {x}_{j}\right\}  }\right) }\right)  = 
$$

$$
\mathop{\sum }\limits_{{j = 1}}^{{a + 1}}P\left( {{R}_{1}^{b + 1} = {x}_{j}}\right) P\left( {{R}_{a}^{b} = X \smallsetminus  \left\{  {x}_{j}\right\}  }\right)  = 
$$

$$
\left( {a + 1}\right) \frac{1}{b + 1}\frac{1}{\left( \begin{array}{l} b \\  a \end{array}\right) } = \frac{1}{\left( \begin{array}{l} b + 1 \\  a + 1 \end{array}\right) }.
$$

Otherwise,

否则

$$
P\left( {{R}_{a + 1}^{b + 1} = X}\right)  = P\left( {{R}_{a}^{b} = X\smallsetminus \{ b + 1\} ,{R}_{1}^{b + 1} \in  X}\right)  = 
$$

$$
\frac{1}{\left( \begin{array}{l} b \\  a \end{array}\right) }\frac{a + 1}{b + 1} = \frac{1}{\left( \begin{array}{l} b + 1 \\  a + 1 \end{array}\right) }.
$$

LEMMA 4.3. ${R}_{k}^{n}$ can be generated using only independent samples ${R}_{1}^{n},{R}_{1}^{n - 1},\ldots ,{R}_{1}^{n - k + 1}$ .

引理4.3。仅使用独立样本${R}_{1}^{n},{R}_{1}^{n - 1},\ldots ,{R}_{1}^{n - k + 1}$就可以生成${R}_{k}^{n}$。

Proof. By using Lemma 4.2,we can generate ${R}_{2}^{n - k + 2}$ using ${R}_{1}^{n - k + 1}$ and ${R}_{1}^{n - k + 2}$ . We can repeat this procedure and generate ${R}_{j}^{n - k + j},2 \leq  j \leq  k$ ,using ${R}_{j - 1}^{n - k + j - 1}$ (that we already constructed by induction) and ${R}_{1}^{n - k + j}$ . For $j = k$ we have ${R}_{k}^{n}$ .

证明。通过使用引理4.2，我们可以使用${R}_{1}^{n - k + 1}$和${R}_{1}^{n - k + 2}$生成${R}_{2}^{n - k + 2}$。我们可以重复这个过程，并使用${R}_{j - 1}^{n - k + j - 1}$（我们已经通过归纳法构造出来）和${R}_{1}^{n - k + j}$生成${R}_{j}^{n - k + j},2 \leq  j \leq  k$。对于$j = k$，我们有${R}_{k}^{n}$。

By using Lemma 4.3,we can generate a $k$ -sample without replacement using only ${R}_{0},\ldots ,{R}_{k - 1}$ . Thus,we have proved

通过使用引理4.3，我们仅使用${R}_{0},\ldots ,{R}_{k - 1}$就可以生成一个无放回的$k$样本。因此，我们证明了

THEOREM 4.4. It is possible to maintain $k$ -sampling without replacement for timestamp-based windows using $O\left( {k\log n}\right)$ memory words.

定理4.4。使用$O\left( {k\log n}\right)$个存储字，可以为基于时间戳的窗口维护无放回的$k$采样。

## 5. APPLICATIONS

## 5. 应用

Consider that algorithm $\Lambda$ is sampling-based,i.e.,it operates on a uniformly chosen subset of $D$ instead of the whole stream. Such an algorithm can be immediately transformed to sliding windows by replacing the underlying sampling method with our algorithms. We obtain the following general result and illustrate it with the examples below.

考虑到算法$\Lambda$是基于采样的，即它对$D$的一个均匀选择的子集进行操作，而不是对整个数据流进行操作。通过用我们的算法替换底层的采样方法，这样的算法可以立即转换为滑动窗口算法。我们得到以下一般结果，并通过下面的例子进行说明。

THEOREM 5.1. For the sampling-based algorithm $\Lambda$ that solves problem $P$ ,there exists an algorithm ${\Lambda }^{\prime }$ that solves $P$ on sliding windows. The memory guarantees are preserved for sequence-based windows and have a multiplicative overhead of $\log n$ for timestamp-based windows.

定理5.1。对于解决问题$P$的基于采样的算法$\Lambda$，存在一个算法${\Lambda }^{\prime }$可以在滑动窗口上解决$P$。对于基于序列的窗口，存储保证得以保留；对于基于时间戳的窗口，有一个$\log n$的乘法开销。

Frequency moment is a fundamental problem in data stream processing. Given a stream of elements,such that ${p}_{j} \in  \left\lbrack  m\right\rbrack$ ,the frequency ${x}_{i}$ of each $i \in  \left\lbrack  m\right\rbrack$ is defined as $\left| \left\{  {j \mid  {p}_{j} = i}\right\}  \right|$ and the $k$ -th frequency moment is defined as ${F}_{k} = \mathop{\sum }\limits_{{i = 1}}^{m}{x}_{i}^{k}$ . The first algorithm for frequency moments for $k > 2$ was proposed in the seminal paper of Alon, Matias and Szegedy [4]. They present an algorithm that uses $O\left( {m}^{1 - \frac{1}{k}}\right)$ memory. Numerous improvements to lower and upper bounds have been reported, including the works of Bar-Yossef, Jayram, Kumar and Sivakumar [14], Chakrabarti, Khot and Sun [23], Coppersmith and Kumar [33], and Ganguly[44]. Finally, Indyk and Woodruff [57] and later Bhuvanagiri, Ganguly, Kesh and Saha [18] presented algorithms that use $\widetilde{O}\left( {m}^{1 - \frac{2}{k}}\right)$ memory and are optimal. The algorithm of Alon, Matias and Szegedy [4] is sampling-based, thus we can adapt it to sliding windows using our methods. The memory usage is not optimal, however this is the first algorithm for frequency moments over sliding windows that works for all $k$ . Recently Braverman and Ostrovsky [19] adapted the algorithm from [18] to sliding windows, producing a memory-optimal algorithm that uses $\widetilde{O}\left( {m}^{1 - \frac{2}{k}}\right)$ . However,it involves ${k}^{k}$ multiplicative overhead,making it infeasible for large $k$ ; thus these results generally cannot be compared. We have

频率矩是数据流处理中的一个基本问题。给定一个元素流，使得${p}_{j} \in  \left\lbrack  m\right\rbrack$，每个$i \in  \left\lbrack  m\right\rbrack$的频率${x}_{i}$定义为$\left| \left\{  {j \mid  {p}_{j} = i}\right\}  \right|$，第$k$阶频率矩定义为${F}_{k} = \mathop{\sum }\limits_{{i = 1}}^{m}{x}_{i}^{k}$。关于$k > 2$的频率矩的第一个算法是由阿隆（Alon）、马蒂亚斯（Matias）和塞格迪（Szegedy）在开创性论文[4]中提出的。他们提出了一个使用$O\left( {m}^{1 - \frac{1}{k}}\right)$存储的算法。已经有许多关于上下界改进的报道，包括巴 - 约塞夫（Bar - Yossef）、杰伊拉姆（Jayram）、库马尔（Kumar）和西瓦库马尔（Sivakumar）[14]、查克拉巴蒂（Chakrabarti）、霍特（Khot）和孙（Sun）[23]、库珀史密斯（Coppersmith）和库马尔（Kumar）[33]以及甘古利（Ganguly）[44]的工作。最后，因迪克（Indyk）和伍德拉夫（Woodruff）[57]以及后来的布瓦纳吉里（Bhuvanagiri）、甘古利（Ganguly）、凯什（Kesh）和萨哈（Saha）[18]提出了使用$\widetilde{O}\left( {m}^{1 - \frac{2}{k}}\right)$存储的最优算法。阿隆（Alon）、马蒂亚斯（Matias）和塞格迪（Szegedy）[4]的算法是基于采样的，因此我们可以使用我们的方法将其应用于滑动窗口。存储使用不是最优的，然而这是第一个适用于所有$k$的滑动窗口频率矩算法。最近，布拉弗曼（Braverman）和奥斯特罗夫斯基（Ostrovsky）[19]将[18]中的算法应用于滑动窗口，产生了一个使用$\widetilde{O}\left( {m}^{1 - \frac{2}{k}}\right)$的存储最优算法。然而，它涉及${k}^{k}$的乘法开销，使得对于大的$k$不可行；因此这些结果通常无法比较。我们有

COROLLARY 5.2. For any $k > 2$ ,there exists an algorithm that maintains an $\left( {\epsilon ,\delta }\right)$ -approximation of the $k$ -th frequency moment over sliding windows using $\widetilde{O}\left( {m}^{1 - \frac{1}{k}}\right)$ bits.

推论5.2。对于任何$k > 2$，存在一个算法，使用$\widetilde{O}\left( {m}^{1 - \frac{1}{k}}\right)$位来维护滑动窗口上第$k$阶频率矩的$\left( {\epsilon ,\delta }\right)$近似值。

Recently, numerous graph problems were addressed in the streaming environment. Stream elements represent edges of the graph, given in arbitrary order (we refer readers to [20] for a detailed explanation of the model). One of the fundamental graph problems is estimating a number of small cliques in a graph, in particular the number of triangles. Effective solutions were proposed by Jowhari and Ghodsi [58], Bar-Yosseff, Kumar and Sivakumar [15] and Bu-riol, Frahling, Leonardi, Marchetti-Spaccamela and Sohler [20]. The last paper presented an $\left( {\epsilon ,\delta }\right)$ -approximation algorithm that uses $O\left( {1 + \frac{\log \left| E\right| }{\left| E\right| }\frac{1}{{\epsilon }^{2}}\frac{\left| {T}_{1}\right|  + 2\left| {T}_{2}\right|  + 3\left| {T}_{3}\right| }{\left| {T}_{3}\right| }\log \frac{2}{\delta }}\right)$ memory ([20],Theorem 2) that is the best result so far. Here, $\left| {T}_{i}\right|$ represents the number of node-triplets having $i$ edges in the induced sub-graph. The algorithm is applied on a random sample collected using the reservoir method. By replacing the reservoir sampling with our algorithms, we obtain the following result.

最近，众多图问题在流环境中得到了解决。流元素表示图的边，以任意顺序给出（我们建议读者参考[20]以详细了解该模型）。基本的图问题之一是估计图中小团的数量，特别是三角形的数量。乔哈里（Jowhari）和戈德西（Ghodsi）[58]、巴 - 约塞夫（Bar - Yosseff）、库马尔（Kumar）和西瓦库马尔（Sivakumar）[15]以及布里奥尔（Buriol）、弗拉林（Frahling）、莱昂纳迪（Leonardi）、马尔凯蒂 - 斯帕卡梅拉（Marchetti - Spaccamela）和索勒（Sohler）[20]提出了有效的解决方案。最后一篇论文提出了一种使用$O\left( {1 + \frac{\log \left| E\right| }{\left| E\right| }\frac{1}{{\epsilon }^{2}}\frac{\left| {T}_{1}\right|  + 2\left| {T}_{2}\right|  + 3\left| {T}_{3}\right| }{\left| {T}_{3}\right| }\log \frac{2}{\delta }}\right)$内存的$\left( {\epsilon ,\delta }\right)$近似算法（[20]，定理2），这是目前为止的最佳结果。这里，$\left| {T}_{i}\right|$表示在诱导子图中具有$i$条边的节点三元组的数量。该算法应用于使用蓄水池方法收集的随机样本。通过用我们的算法替换蓄水池采样，我们得到以下结果。

COROLLARY 5.3. There exists an algorithm that maintains an $\left( {\epsilon ,\delta }\right)$ -approximation of the number of triangles over sliding windows. For sequence-based windows it uses

推论5.3：存在一种算法，可在滑动窗口上维持三角形数量的$\left( {\epsilon ,\delta }\right)$近似值。对于基于序列的窗口，它使用

$$
O\left( {1 + \frac{\log \left| {E}_{W}\right| }{\left| {E}_{W}\right| }\frac{1}{{\epsilon }^{2}}\frac{\left| {T}_{1}\right|  + 2\left| {T}_{2}\right|  + 3\left| {T}_{3}\right| }{\left| {T}_{3}\right| }\log \frac{2}{\delta }}\right) 
$$

memory bits,where ${E}_{W}$ is the set of active edges. Timestamp-based windows adds a multiplicative factor of $\log n$ .

内存位，其中${E}_{W}$是活动边的集合。基于时间戳的窗口会增加一个$\log n$的乘法因子。

Following [20], our method is also applicable for incidence streams, where all edges of the same vertex come together.

遵循文献[20]，我们的方法也适用于关联流，即同一顶点的所有边都一起出现的情况。

The entropy of a stream is defined as $H =  - \mathop{\sum }\limits_{{i = 1}}^{m}\frac{{x}_{i}}{N}\log \frac{{x}_{i}}{N}$ , where ${x}_{i}$ is as above. The entropy norm is defined as ${F}_{H} =$ $\mathop{\sum }\limits_{{i = 1}}^{m}{x}_{i}\log {x}_{i}$ . Effective solutions for entropy and entropy norm estimations were recently reported by Guha, McGregor and Venkata-subramanian [53]; Chakrabarti, Do Ba and Muthukrishnan [22]; Harvey, Nelson and Onak [56]; Chakrabarti, Cormode and McGregor [21]; and Zhao, Lall, Ogihara, Spatscheck, Wang and Xu [74].

流的熵定义为$H =  - \mathop{\sum }\limits_{{i = 1}}^{m}\frac{{x}_{i}}{N}\log \frac{{x}_{i}}{N}$，其中${x}_{i}$如上所述。熵范数定义为${F}_{H} =$ $\mathop{\sum }\limits_{{i = 1}}^{m}{x}_{i}\log {x}_{i}$。古哈（Guha）、麦格雷戈（McGregor）和文卡特 - 苏布拉马尼亚姆（Venkata - subramanian）[53]；查克拉巴蒂（Chakrabarti）、多巴（Do Ba）和穆图克里什南（Muthukrishnan）[22]；哈维（Harvey）、纳尔逊（Nelson）和奥纳克（Onak）[56]；查克拉巴蒂（Chakrabarti）、科尔莫德（Cormode）和麦格雷戈（McGregor）[21]；以及赵（Zhao）、拉尔（Lall）、荻原（Ogihara）、斯帕茨切克（Spatscheck）、王（Wang）和徐（Xu）[74]最近报告了熵和熵范数估计的有效解决方案。

The paper of Chakrabarti, Cormode and McGregor presents an algorithm that is based on a variation of reservoir sampling. The algorithm maintains entropy using $O\left( {{\epsilon }^{-2}\log {\delta }^{-1}}\right)$ that is nearly optimal. The authors also considered the sliding window model and used a variant of priority sampling [10] to obtain the approximation. Thus, the worst-case memory guarantees are not preserved for sliding windows. By replacing priority sampling with our methods, we obtain

查克拉巴蒂（Chakrabarti）、科尔莫德（Cormode）和麦格雷戈（McGregor）的论文提出了一种基于蓄水池采样变体的算法。该算法使用$O\left( {{\epsilon }^{-2}\log {\delta }^{-1}}\right)$来维持熵，这几乎是最优的。作者还考虑了滑动窗口模型，并使用优先级采样的一种变体[10]来获得近似值。因此，滑动窗口的最坏情况内存保证无法保留。通过用我们的方法替换优先级采样，我们得到

COROLLARY 5.4. There exists an algorithm that maintains an $\left( {\epsilon ,\delta }\right)$ -approximation of entropy on sliding windows using $O\left( {{\epsilon }^{-2}\log {\delta }^{-1}\log n}\right)$ memory bits.

推论5.4：存在一种算法，可使用$O\left( {{\epsilon }^{-2}\log {\delta }^{-1}\log n}\right)$内存位在滑动窗口上维持熵的$\left( {\epsilon ,\delta }\right)$近似值。

Moreover, our methods can be used with the algorithm from [22] to obtain $\widetilde{O}\left( 1\right)$ memory for large values of the entropy norm. This algorithm is based on reservoir sampling and thus can be straightforwardly implemented in sliding windows. As a result, we build the first solutions with provable memory guarantees on sliding windows.

此外，我们的方法可以与文献[22]中的算法结合使用，以在熵范数较大时获得$\widetilde{O}\left( 1\right)$内存。该算法基于蓄水池采样，因此可以直接在滑动窗口中实现。结果，我们构建了第一个在滑动窗口上具有可证明内存保证的解决方案。

Our algorithms can be naturally extended to some biased functions. Biased sampling [2] is non-uniform, giving larger probabilities for more recent elements. The distribution is defined by a biased function. We can apply our methods to implement step biased functions, maintaining samples over each window with different lengths and combining the samples with corresponding probabilities. Our algorithm can extend the ideas of Feigenbaum, Kannan, Strauss and Viswanathan [42] for testing and spot-checking to sliding windows. Finally, we can apply our tools to the algorithm of Procopiuc and Procopiuc for density estimation [69], since it is based on the reservoir algorithm as well.

我们的算法可以自然地扩展到一些有偏函数。有偏采样[2]是非均匀的，为最近的元素赋予更大的概率。这种分布由有偏函数定义。我们可以应用我们的方法来实现阶梯有偏函数，在不同长度的每个窗口上维持样本，并将样本与相应的概率相结合。我们的算法可以将费根鲍姆（Feigenbaum）、坎南（Kannan）、施特劳斯（Strauss）和维斯瓦纳坦（Viswanathan）[42]用于测试和抽查的思想扩展到滑动窗口。最后，我们可以将我们的工具应用于普罗科皮乌克（Procopiuc）和普罗科皮乌克（Procopiuc）用于密度估计的算法[69]，因为它同样基于蓄水池算法。

## 6. REFERENCES

## 6. 参考文献

[1] C. Aggarwal (editor), Data Streams: Models and Algorithms, Springer Verlag, 2007.

[2] C. Aggarwal, "On biased reservoir sampling in the presence of stream evolution", Proceedings of the 32nd international conference on Very large data bases, pp. 607-618, 2006.

[3] N. Alon, N. Duffield, C. Lund, M. Thorup, "Estimating arbitrary subset sums with few probes," Proceedings of the twenty-fourth ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems, pp. 317-325, 2005.

[4] N. Alon, Y. Matias, M.Szegedy, "The space complexity of approximating the frequency moments," Proceedings of the twenty-eighth annual ACM symposium on Theory of computing, pp. 20-29, 1996.

[5] A. Arasu, B. Babcock, S. Babu, J. Cieslewicz, M. Datar, K. Ito, R. Motwani, U. Srivastava, J. Widom, "STREAM: The Stanford Data Stream Management System," Book Chapter, "Data-Stream Management: Processing High-Speed Data Streams", Springer-Verlag, 2005.

[6] A. Arasu, G. S. Manku, "Approximate counts and quantiles over sliding windows," Proceedings of the twenty-third ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems, 2004.

[7] A. M. Ayad, J. F. Naughton, "Static optimization of conjunctive queries with sliding windows over infinite streams," Proceedings of the 2004 ACM SIGMOD international conference on Management of data, 2004.

[8] B. Babcock, S. Babu, M. Datar, R. Motwani, J. Widom, "Models and issues in data stream systems", Proceedings of the twenty-first ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems, 2002.

[9] B. Babcock, S. Babu, M. Datar, R. Motwani, D. Thomas, "Operator scheduling in data stream systems", The VLDB Journal ÂU The International Journal on Very Large Data Bases, v.13 n.4, pp.333-353, 2004.

[10] B. Babcock, M. Datar, R. Motwani, "Sampling from a moving window over streaming data", Proceedings of the thirteenth annual ACM-SIAM symposium on Discrete algorithms, pp.633-634, 2002.

[11] B. Babcock, M. Datar, R. Motwani, "Load Shedding for Aggregation Queries over Data Streams", Proceedings of the 20th International Conference on Data Engineering, 2004.

[12] B. Babcock, M. Datar, R. Motwani, L. O'Callaghan, "Maintaining variance and k-medians over data stream windows", Proceedings of the twenty-second ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems, pp.234-243, 2003.

[13] Z. Bar-Yossef, "Sampling lower bounds via information theory", STOC, 2003.

[14] Z. Bar-Yossef, T. S. Jayram, R. Kumar, D. Sivakumar, "An Information Statistics Approach to Data Stream and Communication Complexity", Proceedings of the 43rd Symposium on Foundations of Computer Science, pp. 209-218, 2002.

[15] Z. Bar-Yosseff, R. Kumar, D. Sivakumar, "Reductions in streaming algorithms, with an application to counting triangles in graphs", Proceedings of the thirteenth annual

ACM-SIAM symposium on Discrete algorithms, pp.623-632,

ACM - SIAM离散算法研讨会，第623 - 632页

2002.

[16] Z. Bar-Yossef, T. S. Jayram, R. Kumar, D. Sivakumar, L. Trevisan, "Counting Distinct Elements in a Data Stream", Proceedings of the 6th International Workshop on Randomization and Approximation Techniques, pp.1-10, 2002.

[17] Z. Bar-Yossef, R. Kumar, D. Sivakumar, "Sampling algorithms: lower bounds and applications", STOC, 2001.

[18] L. Bhuvanagiri, S. Ganguly, D. Kesh, C. Saha, "Simpler algorithm for estimating frequency moments of data streams", Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm, pp.708-713, 2006.

[19] V. Braverman, R. Ostrovsky, "Smooth histograms on stream windows", Proceedings of the 48th Symposium on Foundations of Computer Science, 2007.

[20] L. S. Buriol, G. Frahling, S. Leonardi, A. Marchetti-Spaccamela, C. Sohler, "Counting triangles in data streams", Proceedings of the twenty-fifth ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems, pp.253-262, 2006.

[21] A. Chakrabarti, G. Cormode, A. McGregor, "A near-optimal algorithm for computing the entropy of a stream". In Proceedings of ACM-SIAM Symposium on Discrete Algorithms, 2007.

[22] A. Chakrabarti, K. Do Ba, S. Muthukrishnan, "Estimating Entropy and Entropy Norm on Data Streams", In Proceedings of the 23rd International Symposium on Theoretical Aspects of Computer Science, 2006.

[23] A. Chakrabarti, S. Khot, X. Sun, "Near-optimal lower bounds on the multi-party communication complexity of set-disjointness", Proceedings of the 18th Annual IEEE Conference on Computational Complexity, 2003.

[24] K. L. Chang, R. Kannan, "The space complexity of pass-efficient algorithms for clustering", in ACM-SIAM Symposium on Discrete Algorithms, 2006, pp. 1157-ÅÜ1166.

[25] M. Charikar, C. Chekuri, T. Feder, R. Motwani, "Incremental clustering and dynamic information retrieval", ${SIAMJ}$ . Comput., 33 (2004), pp. 1417ÅÜ-1440.

[26] K. Chaudhuri, N. Mishra, "When Random Sampling Preserves Privacy", CRYPTO, 2006.

[27] S. Chaudhuri, R. Motwani, V. Narasayya, "On random sampling over joins", Proceedings of the 1999 ACM SIGMOD international conference on Management of data, pp.263-274, 1999.

[28] Y. Chi, H. Wang, P. S. Yu, R. R. Muntz, "Moment: Maintaining Closed Frequent Itemsets over a Stream Sliding Window", Fourth IEEE International Conference on Data Mining (ICDM'04), pp. 59-66, 2004.

[29] E. Cohen, "Size-estimation framework with applications to transitive closure and reachability," Journal of Computer and System Sciences, v.55 n.3, pp.441-453, 1997.

[30] E. Cohen, H. Kaplan, "Summarizing data using bottom-k sketches,", Proceedings of the twenty-sixth annual ACM symposium on Principles of distributed computing, 2007.

[31] G. Cormode, M. Datar, P. Indyk, S. Muthukrishnan, "Comparing Data Streams Using Hamming Norms (How to Zero In)", IEEE Transactions on Knowledge and Data Engineering, v. 15 n.3, pp.529-540, 2003.

[32] G. Cormode, S. Muthukrishnan, I. Rozenbaum, "Summarizing and mining inverse distributions on data streams via dynamic inverse sampling", Proceedings of the 31st international conference on Very large data bases, 2005.

[33] D. Coppersmith, R. Kumar, "An improved data stream algorithm for frequency moments", Proceedings of the fifteenth annual ACM-SIAM symposium on Discrete algorithms, pp.151-156, 2004.

[34] A. Das, J. Gehrke, M. Riedewald, "Semantic Approximation of Data Stream Joins", IEEE Transactions on Knowledge and Data Engineering, v. 17 n.1, pp.44-59, 2005.

[35] A. Dasgupta, P. Drineas, B. Harb, R. Kumar, M. W.

Mahoney,"Sampling algorithms and coresets for ${l}_{p}$

马奥尼（Mahoney），“${l}_{p}$的采样算法和核心集”

regression", SODA, 2008.

[36] M. Datar, A. Gionis, P. Indyk, R. Motwani, "Maintaining stream statistics over sliding windows: (extended abstract)", Proceedings of the thirteenth annual ACM-SIAM symposium on Discrete algorithms, pp.635-644, 2002.

[37] M. Datar, S. Muthukrishnan, "Estimating Rarity and Similarity over Data Stream Windows", Proceedings of the 10th Annual European Symposium on Algorithms, pp.323-334, 2002.

[38] N. Duffield, C. Lund, M. Thorup, "Flow sampling under hard resource constraints", ACM SIGMETRICS Performance Evaluation Review, v.32 n.1, 2004.

[39] J. Feigenbaum, S. Kannan, and J. Zhang, "Computing diameter in the streaming and sliding-window models", Algorithmica, 41:25-41, 2005.

[40] J. Feigenbaum, S. Kannan, A. McGregor, S. Suri, J. Zhang, "Graph distances in the streaming model: the value of space", SODA, 2005.

[41] J. Feigenbaum, S. Kannan, A. McGregor, S. Suri, J. Zhang, "On graph problems in a semi-streaming model", Theor. Comput. Sci., 2005.

[42] J. Feigenbaum, S. Kannan, M. Strauss, M. Viswanathan, "Testing and Spot-Checking of Data Streams", Algorithmica, 34(1): 67-80, 2002.

[43] G. Frahling, P. Indyk, C. Sohler, "Sampling in dynamic data streams and applications", Proceedings of the twenty-first annual symposium on Computational geometry, 2005.

[44] S. Ganguly. "Estimating Frequency Moments of Update Streams using Random Linear Combinations". Proceedings of the 8th International Workshop on Randomized Algorithms, pp. 369-Ü380, 2004.

[45] S. Ganguly, "Counting distinct items over update streams", Theoretical Computer Science, pp.211-222, 2007.

[46] S. Gandhi, S. Suri, E. Welzl, "Catching elephants with mice: sparse sampling for monitoring sensor networks", SenSys, 2007.

[47] R. Gemulla, "Sampling Algorithms for Evolving Datasets", PhD Dissertation.

[48] R. Gemulla and W. Lehner, "Sampling time-based sliding windows in bounded space", In Proc. of the 2008 ACM SIGMOD Intl. Conf. on Management of Data, pp. 379-392.

[49] P. B. Gibbons, Y. Matias, "New sampling-based summary statistics for improving approximate query answers", Proceedings of the 1998 ACM SIGMOD international conference on Management of data, pp.331-342, 1998.

[50] P. B. Gibbons, S. Tirthapura, "Distributed streams algorithms for sliding windows", Proceedings of the fourteenth annual ACM symposium on Parallel algorithms and architectures, pp.10-13, 2002.

[51] L. Golab, D. DeHaan, E. D. Demaine, A. Lopez-Ortiz, J. I. Munro, "Identifying frequent items in sliding windows over on-line packet streams", Proceedings of the 3rd ACM SIGCOMM conference on Internet measurement, 2003.

[52] L. Golab , M. T. Özsu, "Processing sliding window multi-joins in continuous queries over data streams", Proceedings of the 29th international conference on Very large data bases, pp.500-511, 2003.

[53] S. Guha, A. McGregor, S. Venkatasubramanian, "Streaming and sublinear approximation of entropy and information distances", Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm, pp.733-742, 2006.

[54] S. Guha, A. Meyerson, N. Mishra, R. Motwani, L. O'Callaghan, "Clustering Data Streams: Theory and Practice", IEEE Trans. on Knowledge and Data Engineering, vol. 15, 2003.

[55] P. J. Haas, "Data stream sampling: Basic techniques and results", In M. Garofalakis, J. Gehrke, and R. Rastogi (Eds.), Data Stream Management: Processing High Speed Data Streams, Springer.

[56] N. Harvey, J. Nelson, K. Onak, "Sketching and Streaming

[56] N. 哈维（Harvey）、J. 尼尔森（Nelson）、K. 奥纳克（Onak），“草图与流处理”

Entropy via Approximation Theory", The 49th Annual Symposium on Foundations of Computer Science (FOCS 2008).

[57] P. Indyk, D. Woodruff, "Optimal approximations of the frequency moments of data streams", Proceedings of the thirty-seventh annual ACM symposium on Theory of computing, pp.202-208, 2005.

[58] H. Jowhari, M. Ghodsi, "New streaming algorithms for counting triangles in graphs", Proceedings of the 11th COCOON, pp. 710-716, 2005.

[59] M. Kolonko, D. Wäsch, "Sequential reservoir sampling with a nonuniform distribution", v.32, i.2, pp.257-273, 2006.

[60] L. K. Lee, H. F. Ting, "Frequency counting and aggregation: A simpler and more efficient deterministic scheme for finding frequent items over sliding windows", Proceedings of the twenty-fifth ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems (PODS '06), pp. 290-297, 2006.

[61] L. K. Lee, H. F. Ting, "Maintaining significant stream statistics over sliding windows", Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm, pp.724-732, 2006.

[62] K. Li, "Reservoir-sampling algorithms of time complexity $O\left( {n\left( {1 + \log \left( {N/n}\right) }\right) }\right)$ ",ACM Transactions on Mathematical Software (TOMS), v.20 n.4, pp.481-493, Dec. 1994.

[63] J. Li, D. Maier, K. Tufte, V. Papadimos, P. A. Tucker, "Semantics and Evaluation Techniques for Window Aggregates in Data Streams", SIGMOD, 2005.

[64] J. Li, D. Maier, K. Tufte, V. Papadimos, P. A. Tucker, "No pane, no gain: efficient evaluation of sliding-window aggregates over data streams", ACM SIGMOD Record, v.34 n.1,2005.

[65] G. S. Manku, R. Motwani, "Approximate frequency counts over data streams". In Proceedings of the 28th International Conference on Very Large Data Bases, pp.356-357, 2002.

[66] S. Muthukrishnan, "Data Streams: Algorithms And Applications" Foundations and Trends in Theoretical Computer Science, Volume 1, Issue 2.

[67] C. R. Palmer, C. Faloutsos, "Density biased sampling: an improved method for data mining and clustering", Proceedings of the 2000 ACM SIGMOD international conference on Management of data, pp.82-92, 2000

[68] V. Paxson, G. Almes, J. Mahdavi, M. Mathis, "Framework for IP performance metrics", RFC 2330, 1998.

[69] C. Procopiuc, O. Procopiuc, "Density Estimation for Spatial Data Streams", Proceedings of the 9th International Symposium on Spatial and Temporal Databases, pp.109-126, 2005.

[70] M. Szegedy, "The DLT priority sampling is essentially optimal", Proceedings of the thirty-eighth annual ACM symposium on Theory of computing, pp.150-158, 2006.

[71] N. Tatbul, S. Zdonik, "Window-aware load shedding for aggregation queries over data streams", Proceedings of the 32nd international conference on Very large data bases, 2006.

[72] J. S. Vitter, "Random sampling with a reservoir", ACM Transactions on Mathematical Software (TOMS), v.11 n.1, pp.37-57, 1985.

[73] L. Zhang, Z. Li, M. Yu, Y. Wang, Y. Jiang, "Random sampling algorithms for sliding windows over data streams", Proc. of the 11th Joint International Computer Conference, pp. 572-575, 2005.

[74] H. Zhao, A. Lall, M. Ogihara, O. Spatscheck, J. Wang, J. Xu, "A data streaming algorithm for estimating entropies of od flows", Proceedings of the 7th ACM SIGCOMM conference on Internet measurement, 2007.