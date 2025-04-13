# Multidimensional Online Tracking

# 多维在线跟踪

KE YI and QIN ZHANG, Hong Kong University of Science and Technology

易可（KE YI）和张秦（QIN ZHANG），香港科技大学

We propose and study a new class of online problems, which we call online tracking. Suppose an observer, say Alice,observes a multivalued function $f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$ over time in an online fashion,that is,she only sees $f\left( t\right)$ for $t \leq  {t}_{\text{now }}$ where ${t}_{\text{now }}$ is the current time. She would like to keep a tracker,say Bob,informed of the current value of $f$ at all times. Under this setting,Alice could send new values of $f$ to Bob from time to time,so that the current value of $f$ is always within a distance of $\Delta$ to the last value received by Bob. We give competitive online algorithms whose communication costs are compared with the optimal offline algorithm that knows the entire $f$ in advance. We also consider variations of the problem where Alice is allowed to send predictions to Bob, to further reduce communication for well-behaved functions. These online tracking problems have a variety of application, ranging from sensor monitoring, location-based services, to publish/subscribe systems.

我们提出并研究了一类新的在线问题，我们称之为在线跟踪。假设一位观察者，比如爱丽丝（Alice），以在线方式随时间观察一个多值函数 $f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$，也就是说，她仅能看到 $f\left( t\right)$（其中 $t \leq  {t}_{\text{now }}$，${t}_{\text{now }}$ 为当前时间）。她希望让一个跟踪器，比如鲍勃（Bob），随时了解 $f$ 的当前值。在这种设定下，爱丽丝可以不时地将 $f$ 的新值发送给鲍勃，使得 $f$ 的当前值与鲍勃最后收到的值之间的距离始终在 $\Delta$ 以内。我们给出了具有竞争力的在线算法，其通信成本与预先知晓整个 $f$ 的最优离线算法进行比较。我们还考虑了该问题的变体，即允许爱丽丝向鲍勃发送预测信息，以进一步减少行为良好的函数的通信量。这些在线跟踪问题有多种应用，范围从传感器监测、基于位置的服务到发布/订阅系统。

Categories and Subject Descriptors: F.2.2 [Analysis of Algorithms and Problem Complexity]: Nonnumerical Algorithms and Problems

分类与主题描述符：F.2.2 [算法分析与问题复杂度]：非数值算法与问题

General Terms: Algorithms, Theory

通用术语：算法、理论

Additional Key Words and Phrases: Online tracking

其他关键词和短语：在线跟踪

## ACM Reference Format:

## ACM 引用格式：

Yi, K. and Zhang, Q. 2012. Multidimensional online tracking. ACM Trans. Algor. 8, 2, Article 12 (April 2012), 16 pages.

易可（Yi, K.）和张秦（Zhang, Q.）2012 年。多维在线跟踪。《ACM 算法汇刊》8 卷 2 期，文章编号 12（2012 年 4 月），16 页。

DOI $= {10.1145}/{2151171.2151175}$ http://doi.acm.org/10.1145/2151171.2151175

DOI $= {10.1145}/{2151171.2151175}$ http://doi.acm.org/10.1145/2151171.2151175

## 1. INTRODUCTION

## 1. 引言

Let Alice be an observer who observes a function $f\left( t\right)$ in an online fashion over time. She would like to keep a tracker, Bob, informed of the current function value within some predefined error. What is the best strategy that Alice could adopt so that the total communication is minimized? This is the general problem that we study in this article. For concreteness, consider the simplest case, where the function takes integer values at each time step ${1f} : {\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$ ,and we require an absolute error of at most $\Delta$ . The natural solution to the problem is to first communicate $f\left( 0\right)$ to Bob; then every time $f\left( t\right)$ has changed by more than $\Delta$ since the last communication,Alice updates Bob with the current $\bar{f}\left( t\right)$ . Interestingly,this natural algorithm for this seemingly simple problem has an unbounded competitive ratio compared with the optimal. Consider the case where $f\left( t\right)$ starts at $f\left( 0\right)  = 0$ and then oscillates between 0 and ${2\Delta }$ . Then this

假设爱丽丝（Alice）是一位以在线方式随时间观察函数 $f\left( t\right)$ 的观察者。她希望让一个跟踪器鲍勃（Bob）在某个预定义的误差范围内了解函数的当前值。爱丽丝可以采用的使总通信量最小化的最佳策略是什么？这是我们在本文中研究的一般问题。为了具体说明，考虑最简单的情况，即函数在每个时间步 ${1f} : {\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$ 取整数值，并且我们要求绝对误差至多为 $\Delta$。该问题的自然解决方案是首先将 $f\left( 0\right)$ 传达给鲍勃；然后每当自上次通信以来 $f\left( t\right)$ 的变化超过 $\Delta$ 时，爱丽丝就用当前的 $\bar{f}\left( t\right)$ 更新鲍勃。有趣的是，对于这个看似简单的问题，这种自然算法与最优算法相比具有无界的竞争比。考虑 $f\left( t\right)$ 从 $f\left( 0\right)  = 0$ 开始，然后在 0 和 ${2\Delta }$ 之间振荡的情况。那么这个

${}^{1}$ We use ${\mathbb{Z}}^{ + }$ to denote the domain of all nonnegative integers in this article. algorithm will communicate an infinite number of times while the optimal solution only needs one message: $f\left( 0\right)  = \Delta$ .

${}^{1}$ 在本文中，我们使用 ${\mathbb{Z}}^{ + }$ 表示所有非负整数的定义域。该算法将进行无限次通信，而最优解决方案只需要一条消息：$f\left( 0\right)  = \Delta$。

---

<!-- Footnote -->

K. Yi is supported by an RPC grant from HKUST and a Google Faculty Research Award. Q. Zhang (corresponding author) was supported by Hong Kong CERG Grant 613507.

易可（K. Yi）得到了香港科技大学的研究资助委员会（RPC）资助和谷歌教职研究奖的支持。张秦（Q. Zhang，通讯作者）得到了香港研究资助局（CERG）613507 号资助。

A preliminary version of the article appeared in Proceedings of the ACM-SIAM Symposium on Discrete Algorithms (SODA) 2009.

本文的一个初步版本发表在 2009 年 ACM - SIAM 离散算法研讨会（SODA）会议录中。

Authors' address: K. Yi and Q. Zhang, HKUST, Clear Water Bay, Hong Kong, China; email: \{yike, qinzhang\}@cse.ust.hk.

作者地址：易可（K. Yi）和张秦（Q. Zhang），香港科技大学，中国香港清水湾；电子邮件：\{yike, qinzhang\}@cse.ust.hk。

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies show this notice on the first page or initial screen of a display along with the full citation. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, to republish, to post on servers, to redistribute to lists, or to use any component of this work in other works requires prior specific permission and/or a fee. Permissions may be requested from Publications Dept., ACM, Inc., 2 Penn Plaza, Suite 701, New York, NY 10121-0701 USA, fax +1 (212) 869-0481, or permissions@acm.org.

允许个人或课堂使用本作品的部分或全部内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，且在显示的第一页或初始屏幕上要同时显示此声明以及完整的引用信息。必须尊重本作品中除美国计算机协会（ACM）之外其他所有者的版权。允许在注明出处的情况下进行摘要引用。若要以其他方式复制、重新发布、发布到服务器、分发给列表，或在其他作品中使用本作品的任何部分，则需要事先获得特定许可和/或支付费用。许可申请可发送至美国计算机协会（ACM）出版部，地址为美国纽约州纽约市宾夕法尼亚广场2号701室，邮编10121 - 0701，传真 +1 (212) 869 - 0481，或发送邮件至permissions@acm.org。

(C) 2012 ACM 1549-6325/2012/04-ART12 \$10.00

(C) 2012 美国计算机协会（ACM） 1549 - 6325/2012/04 - ART12 10.00美元

DOI 10.1145/2151171.2151175 http://doi.acm.org/10.1145/2151171.2151175

数字对象标识符（DOI） 10.1145/2151171.2151175 http://doi.acm.org/10.1145/2151171.2151175

<!-- Footnote -->

---

This example shows that even in its simplest instantiation, the online tracking problem will require some nontrivial solutions. Indeed,in Section 2 we give an $O\left( {\log \Delta }\right)$ - competitive algorithm for the preceding problem. The competitive ratio is also tight. Formally, we define the general problem considered in this article as follows. Let $f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$ be a function observed by Alice over time. At the current time ${t}_{\text{now }}$ , Alice only sees all function values for $f\left( t\right) ,t \leq  {t}_{\text{now }}$ . Then she decides if she wants to communicate to Bob,and if so,a pair $\left( {{t}_{\text{now }},g\left( {t}_{\text{now }}\right) }\right)$ is to be sent. Note that $g\left( {t}_{\text{now }}\right)$ is not necessarily equal to $f\left( {t}_{now}\right)$ . The only constraint is that at any ${t}_{now}$ ,if Alice does not communicate,then we must have $\begin{Vmatrix}{f\left( {t}_{\text{now }}\right)  - g\left( {t}_{\text{last }}\right) }\end{Vmatrix} \leq  \Delta$ ,where ${t}_{\text{last }}$ is the last time Bob got informed,and $\Delta  > 0$ is some predefined error parameter. Unless stated otherwise, $\parallel  \cdot  \parallel$ denotes the ${\ell }_{2}$ norm throughout the article. We are mostly interested in the total communication (also referred to as the cost) incurred throughout time, that is, the total number of messages sent by the algorithm. We will analyze the performance of an algorithm in terms of its competitive ratio: the worst-case ratio between the cost of the online algorithm and the cost of the best offline algorithm that knows the entire $f$ in advance. Note that the offline problem is to approximate $f$ with the minimum number of horizontal segments and with error at most $\Delta$ ,which can be easily computed by a greedy algorithm.

此示例表明，即使在最简单的实例中，在线跟踪问题也需要一些非平凡的解决方案。实际上，在第2节中，我们针对上述问题给出了一个$O\left( {\log \Delta }\right)$ - 竞争算法。竞争比也是紧的。形式上，我们将本文所考虑的一般问题定义如下。设$f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$是爱丽丝（Alice）随时间观察到的一个函数。在当前时间${t}_{\text{now }}$，爱丽丝仅能看到$f\left( t\right) ,t \leq  {t}_{\text{now }}$的所有函数值。然后她决定是否要与鲍勃（Bob）进行通信，如果要通信，则需发送一个对$\left( {{t}_{\text{now }},g\left( {t}_{\text{now }}\right) }\right)$。请注意，$g\left( {t}_{\text{now }}\right)$不一定等于$f\left( {t}_{now}\right)$。唯一的约束条件是，在任何${t}_{now}$时刻，如果爱丽丝不进行通信，那么我们必须有$\begin{Vmatrix}{f\left( {t}_{\text{now }}\right)  - g\left( {t}_{\text{last }}\right) }\end{Vmatrix} \leq  \Delta$，其中${t}_{\text{last }}$是鲍勃最后一次得到通知的时间，$\Delta  > 0$是某个预定义的误差参数。除非另有说明，本文通篇用$\parallel  \cdot  \parallel$表示${\ell }_{2}$范数。我们主要关注的是整个时间段内产生的总通信量（也称为成本），即算法发送的消息总数。我们将根据算法的竞争比来分析其性能：在线算法的成本与事先知晓整个$f$的最佳离线算法的成本之间的最坏情况比率。请注意，离线问题是用最少数量的水平线段来近似$f$，且误差至多为$\Delta$，这可以通过贪心算法轻松计算得出。

Motivations. Online tracking problems naturally arise in a variety of applications whenever the observer and the tracker are separate entities and the communication between them is expensive. For example, wireless sensors [Yao and Gehrke 2003] are now widely deployed to collect many different kinds of measurements in the physical world, for example, temperature, humidity, and oxygen level. These small and cheap devices can be easily deployed, but face strict power constraints. It is often costly or even impossible to replace them, so it is essential to develop energy-efficient algorithms for their prolonged functioning. It is well known that among all the factors, wireless transmission of data is the biggest source of battery drain [Pottie and Kaiser 2000]. Therefore, it is very important to minimize the amount of communication back to the tracker, while guaranteeing that the tracker always maintains an approximate measurement monitored by the sensor. This directly corresponds to the one-dimensional version of our problem mentioned at the beginning of the article.

动机。只要观察者和跟踪器是独立的实体，并且它们之间的通信成本较高，在线跟踪问题就会自然地出现在各种应用中。例如，无线传感器[Yao和Gehrke 2003]现在被广泛部署用于收集物理世界中的许多不同类型的测量数据，例如温度、湿度和氧气水平。这些小型且廉价的设备易于部署，但面临严格的功率限制。更换它们通常成本高昂，甚至是不可能的，因此开发节能算法以确保它们长时间运行至关重要。众所周知，在所有因素中，无线数据传输是电池电量消耗的最大来源[Pottie和Kaiser 2000]。因此，在保证跟踪器始终能维持传感器所监测的近似测量值的同时，尽量减少与跟踪器之间的通信量非常重要。这直接对应于本文开头提到的我们问题的一维版本。

Our study is also motivated by the increasing popularity of location-based services [Schiller and Voisard 2004]. Nowadays, many mobile devices, such as cell phones and PDAs, are equipped with GPS. It is common for the service provider to keep track of the user's approximate location, and provide many location-based services, for instance finding the nearest business (ATMs or restaurants), receiving traffic alerts, and so on. This case corresponds to the two-dimensional version of our problem. Here approximation is often necessary not just for reducing communication, but also due to privacy concerns [Beresford and Stajano 2003]. For carriers, location-based services provide added value by enabling dynamic resource tracking (e.g., tracking taxis and service people). Similar to sensors, power consumption is the biggest concern for these mobile devices, and both the user the service provider have incentives to reduce communication while being able to track the locations dynamically.

我们的研究也受到基于位置的服务日益普及的推动[席勒和沃伊萨尔，2004年]。如今，许多移动设备，如手机和个人数字助理（PDA），都配备了全球定位系统（GPS）。服务提供商跟踪用户的大致位置并提供许多基于位置的服务是很常见的，例如查找最近的商家（自动取款机或餐厅）、接收交通警报等等。这种情况对应于我们问题的二维版本。在这里，近似处理通常不仅是为了减少通信量，也是出于隐私考虑[贝雷斯福德和斯塔亚诺，2003年]。对于运营商来说，基于位置的服务通过实现动态资源跟踪（例如，跟踪出租车和服务人员）增加了附加值。与传感器类似，功耗是这些移动设备最大的担忧，用户和服务提供商都有动力在能够动态跟踪位置的同时减少通信量。

Finally, our problem also finds applications in the so called publish / subscribe systems [Diao et al. 2004; Chandramouli et al. 2007]. Traditionally, users poll data from service providers for information; but this has been considered to be very communication-inefficient. In a pub/sub system, users register their queries at the server, and the server pushes updated results to the users according to their registered queries as new data arrives. Unlike the two previous applications, here we have one observer (the server) and many trackers (the users). Although energy is not a concern here, bad decisions by the online tracking algorithm still have severe consequences, since the messages need to be forwarded to a potentially larger number of users, consuming a lot of network bandwidth. Depending on the nature of the query, the function being tracked could take values from a high-dimensional space. For instance, a set of items from a universe $U$ corresponds to a $\{ 0,1\}$ -vector in $\left| U\right|$ dimensions.

最后，我们的问题在所谓的发布/订阅系统中也有应用[刁等人，2004年；钱德拉穆利等人，2007年]。传统上，用户从服务提供商那里轮询数据以获取信息；但这被认为是非常低效的通信方式。在发布/订阅系统中，用户在服务器上注册他们的查询，当新数据到达时，服务器根据用户注册的查询将更新的结果推送给用户。与前两个应用不同，这里我们有一个观察者（服务器）和许多跟踪器（用户）。虽然这里不考虑能源问题，但在线跟踪算法的错误决策仍然会产生严重后果，因为消息需要转发给可能更多的用户，消耗大量的网络带宽。根据查询的性质，被跟踪的函数可能取自高维空间的值。例如，来自一个全集$U$的一组项目对应于$\left| U\right|$维空间中的一个$\{ 0,1\}$向量。

Related work. Although our problem is easily stated and finds many applications, to the best of our knowledge it has not been studied before in the theory community. Some related models include online algorithms, communication complexity, data streams, and the distributed tracking model.

相关工作。虽然我们的问题很容易表述且有许多应用，但据我们所知，理论界之前尚未对其进行研究。一些相关模型包括在线算法、通信复杂性、数据流和分布式跟踪模型。

Our problem generally falls in the realm of online algorithms, and as with all online algorithms, we analyze the performance of our algorithms in terms of competitive ratios.

我们的问题总体上属于在线算法的范畴，与所有在线算法一样，我们根据竞争比来分析我们算法的性能。

In communication complexity [Yao 1979],Alice has $x$ and Bob has $y$ ,and the goal is to compute some function $f\left( {x,y}\right)$ by communicating the minimum number of bits between them. There are two major differences between communication complexity and online tracking. First, in online tracking, only Alice sees the input, Bob just wants to keep track of it. Second,in communication complexity both inputs $x$ and $y$ are given in advance, and the goal is to study the worst-case communication between $x$ and $y$ ; while in online tracking, the inputs arrive in an online fashion, and we focus on the competitive ratio. It is easy to see that the worst-case total communication bound for our problems is meaningless,since the function $f$ could change drastically at each time step.

在通信复杂性[姚，1979年]中，爱丽丝（Alice）拥有$x$，鲍勃（Bob）拥有$y$，目标是通过他们之间最少的比特通信来计算某个函数$f\left( {x,y}\right)$。通信复杂性和在线跟踪有两个主要区别。首先，在在线跟踪中，只有爱丽丝能看到输入，鲍勃只是想跟踪它。其次，在通信复杂性中，两个输入$x$和$y$都是预先给定的，目标是研究$x$和$y$之间的最坏情况通信；而在在线跟踪中，输入以在线方式到达，我们关注的是竞争比。很容易看出，我们问题的最坏情况总通信界限是没有意义的，因为函数$f$在每个时间步都可能发生巨大变化。

In data streams [Alon et al. 1999], the inputs arrive online, and the goal is to track some function over the inputs received so far. In this aspect it is similar to our problem. However, the focus in streaming algorithms is to minimize the space used by the algorithm, not communication. The memory contents could change rapidly, so simply sending out the memory contents could lead to high communication costs.

在数据流[阿隆等人，1999年]中，输入以在线方式到达，目标是跟踪到目前为止接收到的输入上的某个函数。在这方面，它与我们的问题类似。然而，流算法的重点是最小化算法使用的空间，而不是通信量。内存内容可能会快速变化，因此简单地发送内存内容可能会导致高昂的通信成本。

In distributed tracking [Cormode et al. 2008; Keralapura et al. 2006; Cormode et al. 2005; Cormode and Garofalakis 2005; Olston et al. 2001; Davis et al. 2006], the inputs are distributed among multiple sites and arrive online. There is a coordinator who wants to keep track of some function over the union of the inputs received by all sites up until ${t}_{now}$ . So in some sense our problem is the special version of distributed tracking, where there is only one site. However, most work in this area is heuristic-based with only two exceptions to the best of our knowledge. Cormode et al. [2008] consider monotone functions and study worst-case costs. But when the function is not monotone, the worst-case bounds are trivial. In this article, we allow functions to change arbitrarily and use competitive analysis to avoid meaningless worst-case bounds. Davis et al. [2006] propose online algorithms for distributed tracking functions at multiple sites where site $i$ is allowed an error of ${\Delta }_{i}$ ,and the total error $\mathop{\sum }\limits_{i}{\Delta }_{i}$ is fixed. However, in their model, both the online and offline algorithms can only communicate when the error ${\Delta }_{i}$ allocated to some site $i$ is violated,and the site can only send in the current value of the function, that is, exactly what the naive algorithm that we described at the beginning is doing. As such, the problem is only meaningful for two or more sites where the online algorithm needs to decide how to allocate the total error to the sites. When there is only one site, there is nothing controlled by the algorithm. In our problem, we allow both the online and offline algorithms to send in any function value and at any time,as long as the error bound $\Delta$ is satisfied.

在分布式跟踪中 [科尔莫德（Cormode）等人，2008年；凯拉拉普拉（Keralapura）等人，2006年；科尔莫德（Cormode）等人，2005年；科尔莫德（Cormode）和加罗法拉克基斯（Garofalakis），2005年；奥尔斯顿（Olston）等人，2001年；戴维斯（Davis）等人，2006年]，输入数据分布在多个站点并在线到达。有一个协调器，它希望跟踪直到 ${t}_{now}$ 时刻所有站点接收到的输入数据的并集上的某个函数。因此，从某种意义上说，我们的问题是分布式跟踪的特殊版本，即只有一个站点的情况。然而，据我们所知，该领域的大多数工作都是基于启发式方法的，只有两个例外。科尔莫德（Cormode）等人 [2008年] 考虑单调函数并研究最坏情况下的成本。但当函数不是单调函数时，最坏情况的边界是无意义的。在本文中，我们允许函数任意变化，并使用竞争分析来避免无意义的最坏情况边界。戴维斯（Davis）等人 [2006年] 为多个站点的分布式跟踪函数提出了在线算法，其中站点 $i$ 允许有 ${\Delta }_{i}$ 的误差，并且总误差 $\mathop{\sum }\limits_{i}{\Delta }_{i}$ 是固定的。然而，在他们的模型中，在线算法和离线算法只有在分配给某个站点 $i$ 的误差 ${\Delta }_{i}$ 被突破时才能进行通信，并且该站点只能发送函数的当前值，这正是我们在开头描述的简单算法所做的。因此，这个问题仅在有两个或更多站点时才有意义，因为在线算法需要决定如何将总误差分配给各个站点。当只有一个站点时，算法没有任何可控制的内容。在我们的问题中，只要满足误差边界 $\Delta$，我们允许在线算法和离线算法在任何时间发送任何函数值。

Finally, it should be noted that similar problems have been studied in the database community [Madden et al. 2005; Keralapura et al. 2006; Deshpande et al. 2004; Olston et al. 2001]. However, all the techniques proposed there are based on heuristics with no theoretical guarantees.

最后，应该指出的是，数据库领域已经研究过类似的问题 [马登（Madden）等人，2005年；凯拉拉普拉（Keralapura）等人，2006年；德什潘德（Deshpande）等人，2004年；奥尔斯顿（Olston）等人，2001年]。然而，那里提出的所有技术都是基于启发式方法的，没有理论保证。

<!-- Media -->

Table I. Summary of Results for Online Tracking $T$ is the length of the tracking period.

表一. 在线跟踪结果总结 $T$ 是跟踪周期的长度。

<table><tr><td/><td colspan="2">$\beta  = 1$</td><td colspan="2">$\beta  = 1 + \epsilon$</td></tr><tr><td>problem</td><td>$\alpha  -$ competitive</td><td>running time</td><td>$\alpha  -$ competitive</td><td>running time</td></tr><tr><td>1-dim</td><td>$\Theta \left( {\log \Delta }\right)$</td><td>O(1)</td><td>✓</td><td>✓</td></tr><tr><td>$d$ -dim</td><td>$O\left( {{d}^{2}\log \left( {d\Delta }\right) }\right)$</td><td>$\operatorname{poly}\left( {d,\log \Delta }\right)$</td><td>$O\left( {d\log \left( {d/\epsilon }\right) }\right)$</td><td>$\operatorname{poly}\left( {d,\log \left( {1/\epsilon }\right) }\right)$</td></tr><tr><td>1-dim prediction</td><td>$O\left( {\log \left( {\Delta T}\right) }\right)$</td><td>$\operatorname{poly}\left( {\Delta ,T}\right)$</td><td>✓</td><td>/</td></tr></table>

<table><tbody><tr><td></td><td colspan="2">$\beta  = 1$</td><td colspan="2">$\beta  = 1 + \epsilon$</td></tr><tr><td>问题</td><td>$\alpha  -$ 竞争性的</td><td>运行时间</td><td>$\alpha  -$ 竞争性的</td><td>运行时间</td></tr><tr><td>一维</td><td>$\Theta \left( {\log \Delta }\right)$</td><td>O(1)</td><td>✓</td><td>✓</td></tr><tr><td>$d$ 维</td><td>$O\left( {{d}^{2}\log \left( {d\Delta }\right) }\right)$</td><td>$\operatorname{poly}\left( {d,\log \Delta }\right)$</td><td>$O\left( {d\log \left( {d/\epsilon }\right) }\right)$</td><td>$\operatorname{poly}\left( {d,\log \left( {1/\epsilon }\right) }\right)$</td></tr><tr><td>一维预测</td><td>$O\left( {\log \left( {\Delta T}\right) }\right)$</td><td>$\operatorname{poly}\left( {\Delta ,T}\right)$</td><td>✓</td><td>/</td></tr></tbody></table>

<!-- Media -->

Our results. In Section 2 we first give an $O\left( {\log \Delta }\right)$ -competitive algorithm for tracking an integer-valued function. We show that the algorithm is optimal by proving a matching lower bound on the competitive ratio. Our lower bound argument also implies that any real-valued function cannot be tracked with a bounded competitive ratio. This justifies our study being confined with integer-valued functions. In Section 3 we extend our algorithms to $d$ dimensions for arbitrary $d$ . Here we consider the more general $\left( {\alpha ,\beta }\right)$ -competitive algorithms. An online algorithm is $\left( {\alpha ,\beta }\right)$ -competitive if its cost is $\alpha  \cdot$ OPT while allowing an error of $\beta  \cdot  \Delta$ ,where OPT is the cost of the optimal offline algorithm allowing error $\Delta$ . We first give a simple algorithm using the Tukey median of a set of points, and then propose improved algorithms based on volume-cutting, a technique used in many convex optimization algorithms. This results in algorithms with a competitive ratio of $O\left( {{d}^{2}\log \left( {d\Delta }\right) }\right)$ for $\beta  = 1$ and $O\left( {d\log \left( {d/\epsilon }\right) }\right)$ for $\beta  = 1 + \epsilon$ , respectively. The algorithms also have running times polynomial in $d$ .

我们的研究成果。在第2节中，我们首先给出一种用于跟踪整数值函数的$O\left( {\log \Delta }\right)$ - 竞争算法。通过证明竞争比的匹配下界，我们表明该算法是最优的。我们的下界论证还表明，任何实值函数都无法以有界的竞争比进行跟踪。这证明了我们将研究局限于整数值函数是合理的。在第3节中，我们将算法扩展到任意$d$维的情况。在这里，我们考虑更一般的$\left( {\alpha ,\beta }\right)$ - 竞争算法。如果一个在线算法的成本是$\alpha  \cdot$倍的最优离线算法成本（OPT），同时允许误差为$\beta  \cdot  \Delta$，则称该在线算法为$\left( {\alpha ,\beta }\right)$ - 竞争算法，其中OPT是允许误差为$\Delta$的最优离线算法的成本。我们首先使用一组点的图基中位数（Tukey median）给出一个简单的算法，然后基于体积切割（volume - cutting，一种在许多凸优化算法中使用的技术）提出改进的算法。这分别得到了在$\beta  = 1$时竞争比为$O\left( {{d}^{2}\log \left( {d\Delta }\right) }\right)$，在$\beta  = 1 + \epsilon$时竞争比为$O\left( {d\log \left( {d/\epsilon }\right) }\right)$的算法。这些算法的运行时间关于$d$是多项式的。

In Section 4 we further extend our model by considering tracking with predictions. More precisely,Alice tries to predict the future trend of the function $f$ based on history, and then sends the prediction to Bob, for example a linearly increasing trend. If the actual function values do not deviate from the prediction by more than $\Delta$ ,no communication is necessary. The previous tracking problem can be seen as a special case of this more general framework,in which we always predict $f\left( t\right)$ to be $g\left( {t}_{\text{last }}\right)$ . In general,we could use a family $\mathcal{F}$ of prediction functions (e.g.,linear functions),which could greatly reduce the total communication when $f$ can be approximated well by a small number of functions in $\mathcal{F}$ (note that the offline algorithm also uses $\mathcal{F}$ to approximate $f$ ). In this article we only consider the most natural case of linear functions, but we believe that our technique can be extended to more general prediction functions (e.g. polynomial functions with bounded degrees). Our results are summarized in Table I.

在第4节中，我们通过考虑带预测的跟踪来进一步扩展我们的模型。更确切地说，爱丽丝（Alice）试图根据历史数据预测函数$f$的未来趋势，然后将预测结果发送给鲍勃（Bob），例如线性增长趋势。如果实际函数值与预测值的偏差不超过$\Delta$，则无需进行通信。之前的跟踪问题可以看作是这个更一般框架的一个特殊情况，在这种情况下，我们总是预测$f\left( t\right)$为$g\left( {t}_{\text{last }}\right)$。一般来说，我们可以使用一族预测函数$\mathcal{F}$（例如线性函数），当$f$可以由$\mathcal{F}$中的少数几个函数很好地近似时，这可以大大减少总的通信量（注意，离线算法也使用$\mathcal{F}$来近似$f$）。在本文中，我们只考虑最自然的线性函数情况，但我们相信我们的技术可以扩展到更一般的预测函数（例如有界次数的多项式函数）。我们的研究结果总结在表I中。

Finally,we comment that our study in this article focuses only on the ${\ell }_{2}$ metric; the online tracking problem could in general be posed in any metric space, which could potentially lead to other interesting techniques and results. 2. ONLINE TRACKING IN ONE DIMENSION

最后，我们说明本文的研究仅聚焦于${\ell }_{2}$度量；一般来说，在线跟踪问题可以在任何度量空间中提出，这可能会产生其他有趣的技术和结果。2. 一维在线跟踪

In this section,we consider the online tracking problem for functions in the form of $f$ : ${\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$ . The algorithm for the one-dimensional case mainly serves as an illustration, which lays down the general framework for the more advanced algorithms in higher dimensions. For simplicity we assume for now that $\Delta$ is an integer; the assumption will be removed in later sections.

在本节中，我们考虑形式为$f$ : ${\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$的函数的在线跟踪问题。一维情况下的算法主要用于说明问题，它为更高维的更高级算法奠定了一般框架。为了简单起见，我们目前假设$\Delta$是一个整数；该假设将在后续章节中去除。

An $O\left( {\log \Delta }\right)$ -competitive algorithm. Let OPT be the cost of the optimal offline algorithm. The basic idea of the algorithm actually originates from the motivating example at the beginning of the article: when $f$ oscillates within a range of ${2\Delta }$ ,then OPT is constant. Thus, our algorithm tries to guess what value the optimal algorithm has sent, using a binary search. Our algorithm proceeds in rounds; the procedure for each round is outlined in Algorithm 1.

一种$O\left( {\log \Delta }\right)$ - 竞争算法。设OPT为最优离线算法的成本。该算法的基本思想实际上源于文章开头的激励示例：当$f$在${2\Delta }$的范围内振荡时，OPT是常数。因此，我们的算法尝试使用二分查找来猜测最优算法发送的值。我们的算法按轮次进行；每一轮的过程在算法1中概述。

Algorithm 1 is correct since at any ${t}_{\text{now }}$ ,if $f\left( t\right)$ deviates more than $\Delta$ from $g\left( {t}_{\text{last }}\right)$ ,we always update $S$ so that all elements in $S$ are within $\Delta$ of $f\left( {t}_{\text{now }}\right)$ . It is also easy to see that Algorithm 1 can be implemented in $O\left( 1\right)$ time per time step. In the following,we show that its competitive ratio is $O\left( {\log \Delta }\right)$ .

算法1是正确的，因为在任何${t}_{\text{now }}$时刻，如果$f\left( t\right)$与$g\left( {t}_{\text{last }}\right)$的偏差超过$\Delta$，我们总会更新$S$，使得$S$中的所有元素与$f\left( {t}_{\text{now }}\right)$的偏差在$\Delta$以内。不难看出，算法1在每个时间步可以在$O\left( 1\right)$时间内实现。接下来，我们将证明其竞争比为$O\left( {\log \Delta }\right)$。

<!-- Media -->

ALGORITHM 1: One round of 1D tracking

算法1：一维跟踪的一轮操作

---

let $S = \left\lbrack  {f\left( {t}_{\text{now }}\right)  - \Delta ,f\left( {t}_{\text{now }}\right)  + \Delta }\right\rbrack   \cap  \mathbb{Z}$ ;

令$S = \left\lbrack  {f\left( {t}_{\text{now }}\right)  - \Delta ,f\left( {t}_{\text{now }}\right)  + \Delta }\right\rbrack   \cap  \mathbb{Z}$；

while $S \neq  \varnothing$ do

当$S \neq  \varnothing$时执行以下操作

	let $g\left( {t}_{\text{now }}\right)$ be the median of $S$ ;

	令$g\left( {t}_{\text{now }}\right)$为$S$的中位数；

	send $g\left( {t}_{\text{now }}\right)$ to Bob;

	将$g\left( {t}_{\text{now }}\right)$发送给鲍勃；

	wait until $\begin{Vmatrix}{f\left( {t}_{\text{now }}\right)  - g\left( {t}_{\text{last }}\right) }\end{Vmatrix} > \Delta$ ;

	等待直到$\begin{Vmatrix}{f\left( {t}_{\text{now }}\right)  - g\left( {t}_{\text{last }}\right) }\end{Vmatrix} > \Delta$；

	$S \leftarrow  S \cap  \left\lbrack  {f\left( {t}_{\text{now }}\right)  - \Delta ,f\left( {t}_{\text{now }}\right)  + \Delta }\right\rbrack  ;$

---

<!-- Media -->

We will proceed by showing that in each round,the offline optimal algorithm ${\mathcal{A}}_{\mathrm{{OPT}}}$ must send at least one message,while Algorithm 1 sends $O\left( {\log \Delta }\right)$ messages,which will lead to the claimed competitive ratio. The latter simply follows from the fact that the cardinality of $S$ reduces by at least half in each iteration in the while loop,so we only argue for the former. For convenience, we define a round to include its starting time (when $S$ is initialized) and ending time (when $S = \varnothing$ ). Thus,a message sent at a joint point will be counted twice, but that will not affect the competitive ratio by more than a factor of 2 . Suppose the last function value sent by ${\mathcal{A}}_{\mathrm{{OPT}}}$ in the previous round is $y$ . Note that if ${\mathcal{A}}_{\text{OPT }}$ has not sent any message by ${t}_{\text{now }}$ ,then we must have $y \in  S$ at that time,since $S$ is a superset of $\mathop{\bigcap }\limits_{t}\left\lbrack  {f\left( t\right)  - \Delta ,f\left( t\right)  + \Delta }\right\rbrack   \cap  \mathbb{Z}$ ,where the intersection is taken over all $t$ up to ${t}_{\text{now }}$ in this round. In the end, $S = \varnothing$ ,so ${\mathcal{A}}_{\text{OPT }}$ must have sent a new function value other than $y$ .

我们将通过证明在每一轮中，离线最优算法${\mathcal{A}}_{\mathrm{{OPT}}}$必须至少发送一条消息，而算法1发送$O\left( {\log \Delta }\right)$条消息，从而得出所声称的竞争比。后者很容易理解，因为在while循环的每次迭代中，$S$的基数至少减少一半，所以我们只论证前者。为方便起见，我们将一轮定义为包括其开始时间（即初始化$S$的时间）和结束时间（即$S = \varnothing$的时间）。因此，在连接点发送的消息会被计算两次，但这对竞争比的影响不会超过2倍。假设${\mathcal{A}}_{\mathrm{{OPT}}}$在上一轮发送的最后一个函数值是$y$。注意，如果到${t}_{\text{now }}$时刻${\mathcal{A}}_{\text{OPT }}$还没有发送任何消息，那么在该时刻我们必然有$y \in  S$，因为$S$是$\mathop{\bigcap }\limits_{t}\left\lbrack  {f\left( t\right)  - \Delta ,f\left( t\right)  + \Delta }\right\rbrack   \cap  \mathbb{Z}$的超集，这里的交集是对本轮中直到${t}_{\text{now }}$的所有$t$取的。最后，$S = \varnothing$，所以${\mathcal{A}}_{\text{OPT }}$必须发送一个不同于$y$的新函数值。

THEOREM 1. There is an $O\left( {\log \Delta }\right)$ -competitive online algorithm to track any function $f : {\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$

定理1. 存在一个$O\left( {\log \Delta }\right)$ - 竞争的在线算法来跟踪任何函数$f : {\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$

Lower bound on the competitive ratio. We now show that the $O\left( {\log \Delta }\right)$ competitive ratio is optimal. We will construct an adversary under which any deterministic online algorithm ${\mathcal{A}}_{\text{SOL }}$ has to send at least $\Omega \left( {\log \Delta  \cdot  \mathrm{{OPT}}}\right)$ messages,while the optimal offline algorithm ${\mathcal{A}}_{\text{OPT }}$ only needs to send OPT messages.

竞争比的下界。我们现在证明$O\left( {\log \Delta }\right)$竞争比是最优的。我们将构造一个对手策略，在该策略下，任何确定性在线算法${\mathcal{A}}_{\text{SOL }}$必须至少发送$\Omega \left( {\log \Delta  \cdot  \mathrm{{OPT}}}\right)$条消息，而最优离线算法${\mathcal{A}}_{\text{OPT }}$只需要发送OPT条消息。

The adversary (call her Carole) also divides the whole tracking period into rounds. We will show that in each round,Carole could manipulate the value of $f$ so that ${\mathcal{A}}_{\mathrm{{SOL}}}$ has to communicate $\Omega \left( {\log \Delta }\right)$ times,while ${\mathcal{A}}_{\text{OPT }}$ just needs one message. During each round, Carole maintains a set $S$ of possible values so that for any $y \in  S$ ,if ${\mathcal{A}}_{\text{OPT }}$ communicates $y$ at the beginning of this round,it does not need any further communication in this round. The round terminates when $S$ contains less than 3 elements. More precisely, $S$ is initialized to $\left\lbrack  {y - \Delta ,y + \Delta }\right\rbrack   \cap  \mathbb{Z}$ where $y$ is some function value at a distance of at least ${2\Delta } + 1$ from any function value used in the previous round; as time goes on, $S$ is maintained as $\mathop{\bigcap }\limits_{t}\left\lbrack  {f\left( t\right)  - \Delta ,f\left( t\right)  + \Delta }\right\rbrack   \cap  \mathbb{Z}$ ,where the intersection is taken over all time up to ${t}_{\text{now }}$ . Carole uses the following strategy to change the value of $f$ . If ${\mathcal{A}}_{\text{S0L }}$ announces some function value greater than the median of $S$ ,decrease $f$ until ${\mathcal{A}}_{\mathrm{{SOL}}}$ sends out the next message; otherwise increase $f$ until ${\mathcal{A}}_{\text{SOL }}$ sends out the next message. Let ${n}_{i}$ be the number of elements left in $S$ after the $i$ -th triggering of ${\mathcal{A}}_{\mathrm{{SOL}}}$ ,and initially, ${n}_{0} = {2\Delta } + 1$ . It is not difficult to see that ${n}_{i + 1} \geq  \lceil \left( {{n}_{i} - 3}\right) /2\rceil$ ,so it takes $\Omega \left( {\log \Delta }\right)$ iterations for $\left| S\right|$ to be a constant. When $S$ contains less than 3 elements,Carole terminates the round and starts a new one. By the definition of $S,{\mathcal{A}}_{\text{OPT }}$ could send an element in $S$ at the beginning of the round, which is a valid approximation for all function values in this round.

对手（称她为卡罗尔）也将整个跟踪周期划分为若干轮。我们将证明，在每一轮中，卡罗尔可以操纵$f$的值，使得${\mathcal{A}}_{\mathrm{{SOL}}}$必须进行$\Omega \left( {\log \Delta }\right)$次通信，而${\mathcal{A}}_{\text{OPT }}$只需要发送一条消息。在每一轮中，卡罗尔维护一个可能值的集合$S$，使得对于任何$y \in  S$，如果${\mathcal{A}}_{\text{OPT }}$在本轮开始时发送$y$，那么在本轮中就不需要进一步的通信。当$S$中的元素少于3个时，本轮结束。更准确地说，$S$初始化为$\left\lbrack  {y - \Delta ,y + \Delta }\right\rbrack   \cap  \mathbb{Z}$，其中$y$是某个函数值，且与上一轮中使用的任何函数值的距离至少为${2\Delta } + 1$；随着时间的推移，$S$被维护为$\mathop{\bigcap }\limits_{t}\left\lbrack  {f\left( t\right)  - \Delta ,f\left( t\right)  + \Delta }\right\rbrack   \cap  \mathbb{Z}$，其中交集是对直到${t}_{\text{now }}$的所有时间取的。卡罗尔使用以下策略来改变$f$的值。如果${\mathcal{A}}_{\text{S0L }}$宣布某个大于$S$中位数的函数值，则减小$f$，直到${\mathcal{A}}_{\mathrm{{SOL}}}$发出下一条消息；否则，增大$f$，直到${\mathcal{A}}_{\text{SOL }}$发出下一条消息。设${n}_{i}$为${\mathcal{A}}_{\mathrm{{SOL}}}$第$i$次触发后$S$中剩余的元素数量，初始时，${n}_{0} = {2\Delta } + 1$。不难看出${n}_{i + 1} \geq  \lceil \left( {{n}_{i} - 3}\right) /2\rceil$，因此$\left| S\right|$变为常数需要$\Omega \left( {\log \Delta }\right)$次迭代。当$S$中的元素少于3个时，卡罗尔结束本轮并开始新的一轮。根据$S,{\mathcal{A}}_{\text{OPT }}$的定义，它可以在本轮开始时发送$S$中的一个元素，这是本轮中所有函数值的有效近似。

THEOREM 2. To track a function $f : {\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$ ,any online algorithm has to send $\Omega \left( {\log \Delta  \cdot  \mathtt{{OPT}}}\right)$ messages in the worst case,where $\mathtt{{OPT}}$ is the number of messages needed by the optimal offline algorithm.

定理2. 为了跟踪一个函数$f : {\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$，在最坏情况下，任何在线算法都必须发送$\Omega \left( {\log \Delta  \cdot  \mathtt{{OPT}}}\right)$条消息，其中$\mathtt{{OPT}}$是最优离线算法所需的消息数量。

Remark. The argument above also implies that,if $f$ takes values from the domain of reals (or any dense set),the competitive ratio is unbounded,since $S$ always contains infinitely many elements.

注：上述论证还表明，如果$f$的取值来自实数域（或任何稠密集合），则竞争比是无界的，因为$S$总是包含无限多个元素。

## 3. ONLINE TRACKING IN $D$ DIMENSIONS

## 3. $D$维空间中的在线跟踪

In this section we extend our algorithm to higher dimensions, i.e., tracking functions $f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$ for arbitrary $d$ . From now on we will consider the more general $\left( {\alpha ,\beta }\right)$ - competitive algorithms.

在本节中，我们将算法扩展到更高维度，即跟踪任意$d$的函数$f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$。从现在起，我们将考虑更一般的$\left( {\alpha ,\beta }\right)$ - 竞争算法。

Our algorithm actually follows the same framework as in the one-dimensional case. We still divide the whole tracking period into rounds,and show that ${\mathcal{A}}_{\mathrm{{OPT}}}$ must communicate once in each round,while our algorithm communicates at most,say, $k$ times, and then the competitive ratio would be bounded by $k$ . The algorithm for each round is also similar to Algorithm 1. At the beginning of each round (say at time $t = {t}_{\text{start }}$ ), we initialize a set $S = {S}_{0}$ containing all the possible points that might be sent by ${\mathcal{A}}_{\mathrm{{OPT}}}$ in its last communication. In each iteration in the while loop, we first pick a "median" from $S$ and send it to Bob. When $f$ deviates from $g\left( {t}_{\text{last }}\right)$ by more than ${\beta \Delta }$ ,we cut $S$ as $S \leftarrow  S \cap  \operatorname{Ball}\left( {f\left( {t}_{\text{now }}\right) ,\Delta }\right)$ where $\operatorname{Ball}\left( {p,r}\right)$ represents the closed ball centered at $p$ with radius $r$ in ${\mathbb{R}}^{d}$ . This way, $S$ is always a superset of ${S}_{0} \cap  \left( {\mathop{\bigcap }\limits_{{{t}_{\text{start }} < t < {t}_{\text{now }}}}\operatorname{Ball}\left( {f\left( t\right) ,\Delta }\right) }\right)$ . When $S$ becomes empty,we can terminate the round,knowing that ${\mathcal{A}}_{\mathrm{{OPT}}}$ must have sent a new message. Thus,the only remaining issues are how to construct ${S}_{0}$ and how to choose the median so that $S$ will become empty after a small number of cuts.

我们的算法实际上遵循与一维情况相同的框架。我们仍然将整个跟踪周期划分为若干轮，并表明${\mathcal{A}}_{\mathrm{{OPT}}}$必须在每一轮中进行一次通信，而我们的算法最多通信，比如说，$k$次，那么竞争比将被$k$所界定。每一轮的算法也与算法1类似。在每一轮开始时（比如说在时间$t = {t}_{\text{start }}$），我们初始化一个集合$S = {S}_{0}$，其中包含${\mathcal{A}}_{\mathrm{{OPT}}}$在上一次通信中可能发送的所有可能的点。在while循环的每次迭代中，我们首先从$S$中选取一个“中位数”并将其发送给鲍勃。当$f$与$g\left( {t}_{\text{last }}\right)$的偏差超过${\beta \Delta }$时，我们将$S$切割为$S \leftarrow  S \cap  \operatorname{Ball}\left( {f\left( {t}_{\text{now }}\right) ,\Delta }\right)$，其中$\operatorname{Ball}\left( {p,r}\right)$表示在${\mathbb{R}}^{d}$中以$p$为中心、半径为$r$的闭球。这样，$S$始终是${S}_{0} \cap  \left( {\mathop{\bigcap }\limits_{{{t}_{\text{start }} < t < {t}_{\text{now }}}}\operatorname{Ball}\left( {f\left( t\right) ,\Delta }\right) }\right)$的超集。当$S$变为空集时，我们可以终止这一轮，因为我们知道${\mathcal{A}}_{\mathrm{{OPT}}}$一定发送了一条新消息。因此，仅剩下的问题是如何构造${S}_{0}$以及如何选择中位数，以便在进行少量切割后$S$会变为空集。

Note that for $\beta  = 2$ ,the problem is trivial,since ${S}_{0} \subset  \operatorname{Ball}\left( {f\left( {t}_{\text{start }}\right) ,\Delta }\right)$ . The algorithm simply needs to send $g\left( {t}_{\text{start }}\right)  = f\left( {t}_{\text{start }}\right)$ ,and then the first cut will use a ball centered at a distance of more than ${2\Delta }$ away from $f\left( {t}_{\text{start }}\right)$ . So by the triangle inequality,the round always terminates after just one cut,yielding an $\left( {O\left( 1\right) ,2}\right)$ -competitive algorithm. Thus in the remainder of the article,we are only interested in $\beta  = 1$ or $\beta  = 1 + \epsilon$ for any small $\epsilon  > 0$ .

注意，对于$\beta  = 2$，这个问题很简单，因为${S}_{0} \subset  \operatorname{Ball}\left( {f\left( {t}_{\text{start }}\right) ,\Delta }\right)$。该算法只需发送$g\left( {t}_{\text{start }}\right)  = f\left( {t}_{\text{start }}\right)$，然后第一次切割将使用一个中心与$f\left( {t}_{\text{start }}\right)$的距离超过${2\Delta }$的球。因此，根据三角不等式，这一轮总是在仅进行一次切割后就终止，从而得到一个$\left( {O\left( 1\right) ,2}\right)$竞争算法。因此，在本文的其余部分，我们只关注对于任何小的$\epsilon  > 0$的$\beta  = 1$或$\beta  = 1 + \epsilon$。

### 3.1. Algorithms by Tukey Medians

### 3.1. 基于图基中位数（Tukey Medians）的算法

In this section we consider the case $\beta  = 1$ . We start by fixing the set ${S}_{0}$ . Let ${C}_{l}(2 \leq$ $l \leq  d + 1$ ) be the collection of centers of the smallest enclosing balls of every $l$ points in Ball $\left( {f\left( {t}_{\text{start }}\right) ,{2\Delta }}\right)  \cap  {\mathbb{Z}}^{d}$ . At the beginning of the current round,we initialize ${S}_{0}$ to be ${S}_{0} = {C}_{2} \cup  {C}_{3}\ldots  \cup  {C}_{d + 1}$ . The following lemma is justification that ${S}_{0}$ is sufficient for our purpose.

在本节中，我们考虑$\beta  = 1$的情况。我们首先固定集合${S}_{0}$。设${C}_{l}(2 \leq$（$l \leq  d + 1$）是球$\left( {f\left( {t}_{\text{start }}\right) ,{2\Delta }}\right)  \cap  {\mathbb{Z}}^{d}$中每$l$个点的最小包围球的中心的集合。在当前轮开始时，我们将${S}_{0}$初始化为${S}_{0} = {C}_{2} \cup  {C}_{3}\ldots  \cup  {C}_{d + 1}$。以下引理证明了${S}_{0}$对于我们的目的是足够的。

LEMMA 1. If $S$ becomes empty at some time step,then the optimal offline algorithm must have communicated once in the current round.

引理1. 如果$S$在某个时间步变为空集，那么最优离线算法在当前轮中必定进行了一次通信。

Proof. Suppose that the optimal offline algorithm ${\mathcal{A}}_{\text{OPT }}$ does not send any message in the current round when $S$ becomes empty. Let $s$ be the point sent by ${\mathcal{A}}_{\mathrm{{OPT}}}$ in its last communication and ${q}_{1},{q}_{2},\ldots ,{q}_{m}$ ,be all the distinct points taken by the function $f$ in the current round. It is easy to see that if ${\mathcal{A}}_{0\mathrm{{PT}}}$ keeps silent in the current round,we have $\begin{Vmatrix}{s - {q}_{i}}\end{Vmatrix} \leq  \Delta$ for all $1 \leq  i \leq  m$ . If $m = 1,S$ cannot be empty. So we have $m \geq  2$ .

证明。假设最优离线算法 ${\mathcal{A}}_{\text{OPT }}$ 在 $S$ 变为空集的当前轮次中不发送任何消息。设 $s$ 为 ${\mathcal{A}}_{\mathrm{{OPT}}}$ 在其最后一次通信中发送的点，${q}_{1},{q}_{2},\ldots ,{q}_{m}$ 为函数 $f$ 在当前轮次中所取的所有不同的点。不难看出，如果 ${\mathcal{A}}_{0\mathrm{{PT}}}$ 在当前轮次中保持沉默，那么对于所有的 $1 \leq  i \leq  m$ ，我们有 $\begin{Vmatrix}{s - {q}_{i}}\end{Vmatrix} \leq  \Delta$ 。如果 $m = 1,S$ 不可能为空集。所以我们有 $m \geq  2$ 。

Let $B$ be the smallest enclosing ball with center $o$ containing all the ${q}_{i}\left( {1 \leq  i \leq  m}\right)$ . It is not difficult to see that $o \in  {S}_{0}$ ,for the following reason (see also Figure 1). Let $X$ be the set of smallest enclosing balls of a set of integer points in ${\mathbb{Z}}^{d}$ each of which is within a distance of at most ${2\Delta }$ from $f\left( {t}_{\text{start }}\right)$ . Then ${S}_{0}$ is actually the set of centers of balls in $X$ . If $o \notin  {S}_{0}$ ,then at least one ${q}_{j}\left( {1 \leq  i \leq  m}\right)$ should be at a distance more than ${2\Delta }$ from $f\left( {t}_{\text{start }}\right)$ . Since $\begin{Vmatrix}{s - {q}_{j}}\end{Vmatrix} \leq  \Delta$ ,we have $\begin{Vmatrix}{s - f\left( {t}_{\text{start }}\right) }\end{Vmatrix} > \Delta$ ,which means that ${\mathcal{A}}_{\text{OPT }}$ must have communicated once in the current round, which is a contradiction.

设 $B$ 是以 $o$ 为圆心、包含所有 ${q}_{i}\left( {1 \leq  i \leq  m}\right)$ 的最小包围球。不难看出 $o \in  {S}_{0}$ ，原因如下（另见图 1）。设 $X$ 是 ${\mathbb{Z}}^{d}$ 中一组整数点的最小包围球的集合，其中每个点与 $f\left( {t}_{\text{start }}\right)$ 的距离至多为 ${2\Delta }$ 。那么 ${S}_{0}$ 实际上就是 $X$ 中球的圆心的集合。如果 $o \notin  {S}_{0}$ ，那么至少有一个 ${q}_{j}\left( {1 \leq  i \leq  m}\right)$ 与 $f\left( {t}_{\text{start }}\right)$ 的距离应大于 ${2\Delta }$ 。由于 $\begin{Vmatrix}{s - {q}_{j}}\end{Vmatrix} \leq  \Delta$ ，我们有 $\begin{Vmatrix}{s - f\left( {t}_{\text{start }}\right) }\end{Vmatrix} > \Delta$ ，这意味着 ${\mathcal{A}}_{\text{OPT }}$ 必定在当前轮次中进行了一次通信，这是矛盾的。

<!-- Media -->

<!-- figureText: ${q}_{2}$ ${q}_{3}$ $f\left( {t}_{\text{start }}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c904-cf12-7bd7-9ed3-4fa404f6957a_6.jpg?x=636&y=336&w=510&h=373&r=0"/>

Fig. 1.

图 1。

<!-- Media -->

Since $\begin{Vmatrix}{s - {q}_{i}}\end{Vmatrix} \leq  \Delta$ for all $1 \leq  i \leq  m$ ,and $B$ is the smallest enclosing ball containing all ${q}_{i}\left( {1 \leq  i \leq  m}\right)$ with center $o$ ,we have $\begin{Vmatrix}{o - {q}_{i}}\end{Vmatrix} \leq  \Delta$ for all $1 \leq  i \leq  m$ . Thus $o$ must still survive at the current time step,which means that $S$ is not empty-a contradiction.

由于对于所有的 $1 \leq  i \leq  m$ 都有 $\begin{Vmatrix}{s - {q}_{i}}\end{Vmatrix} \leq  \Delta$ ，并且 $B$ 是以 $o$ 为圆心、包含所有 ${q}_{i}\left( {1 \leq  i \leq  m}\right)$ 的最小包围球，所以对于所有的 $1 \leq  i \leq  m$ ，我们有 $\begin{Vmatrix}{o - {q}_{i}}\end{Vmatrix} \leq  \Delta$ 。因此，$o$ 在当前时间步必定仍然存在，这意味着 $S$ 不为空集——这是矛盾的。

The rest of our task is to choose a good median so that the cardinality of $S$ would decrease by some fraction after each triggering of communication. Before proceeding, we need the following concepts.

我们接下来的任务是选择一个合适的中位数，使得每次触发通信后，$S$ 的基数会减少一定比例。在继续之前，我们需要以下概念。

Definition 1 (Location depth). Let $S$ be a set of points in ${\mathbb{R}}^{d}$ . The location depth of a point $q \in  {\mathbb{R}}^{d}$ with respect to $S$ is the minimum number of points of $S$ lying in a closed halfspace containing $q$ .

定义 1（位置深度）。设 $S$ 是 ${\mathbb{R}}^{d}$ 中的一个点集。点 $q \in  {\mathbb{R}}^{d}$ 相对于 $S$ 的位置深度是包含 $q$ 的闭半空间中 $S$ 的最少点数。

The following observation is a direct consequence of Helly's Theorem [Matousek 2002].

以下观察结果是赫利定理（Helly's Theorem）[马托塞克（Matousek）2002 年]的直接推论。

Observation 1. Given a set $S$ in ${\mathbb{R}}^{d}$ ,there always exists a point $q \in  {\mathbb{R}}^{d}$ having location depth at least $\left| S\right| /\left( {d + 1}\right)$ with respect to $S$ . The point with maximum depth is usually called the Tukey median.

观察 1。给定 ${\mathbb{R}}^{d}$ 中的一个集合 $S$ ，总是存在一个点 $q \in  {\mathbb{R}}^{d}$ ，它相对于 $S$ 的位置深度至少为 $\left| S\right| /\left( {d + 1}\right)$ 。深度最大的点通常称为图基中位数（Tukey median）。

The algorithm for the ${\mathbb{R}}^{d}$ case maintains rounds similarly as the one-dimensional case. We just pick the Tukey median to send in each triggering of communication. Since whenever $\begin{Vmatrix}{f\left( {t}_{\text{now }}\right)  - g\left( {t}_{\text{last }}\right) }\end{Vmatrix} > \Delta$ ,Ball $\left( {f\left( {t}_{\text{now }}\right) ,\Delta }\right)$ is strictly contained in a halfspace bounded by a hyperplane passing through $g\left( {t}_{\text{last }}\right)$ ,the cardinality of $S$ decreases by a factor of at least $1/\left( {d + 1}\right)$ . Thus,the algorithm sends ${\log }_{1 + \frac{1}{d}}\left| {S}_{0}\right|  = O\left( {d\log \left| {S}_{0}\right| }\right)$ messages in each round.

对于${\mathbb{R}}^{d}$的情况，该算法维护轮次的方式与一维情况类似。我们只需在每次触发通信时选择图基中位数（Tukey median）进行发送。由于只要$\begin{Vmatrix}{f\left( {t}_{\text{now }}\right)  - g\left( {t}_{\text{last }}\right) }\end{Vmatrix} > \Delta$成立，球$\left( {f\left( {t}_{\text{now }}\right) ,\Delta }\right)$就严格包含在一个由经过$g\left( {t}_{\text{last }}\right)$的超平面所界定的半空间内，$S$的基数至少会减少$1/\left( {d + 1}\right)$倍。因此，该算法在每一轮中发送${\log }_{1 + \frac{1}{d}}\left| {S}_{0}\right|  = O\left( {d\log \left| {S}_{0}\right| }\right)$条消息。

Remember that initially, ${S}_{0} = {C}_{2} \cup  {C}_{3} \cup  \ldots {C}_{d + 1}$ ,and ${C}_{j}\left( {2 \leq  j \leq  d + 1}\right)$ is the collection of centers of the smallest enclosing balls of every $j$ points in $\operatorname{Ball}\left( {f\left( {t}_{\text{start }}\right) ,{2\Delta }}\right)  \cap  {\mathbb{Z}}^{d}$ , whose cardinality is at most $\left( \begin{matrix} {\left( \left\lfloor  4\Delta \right\rfloor   + 1\right) }^{d} \\  j \end{matrix}\right)$ . Therefore ${S}_{0}$ contains at most

请记住，初始时，${S}_{0} = {C}_{2} \cup  {C}_{3} \cup  \ldots {C}_{d + 1}$，并且${C}_{j}\left( {2 \leq  j \leq  d + 1}\right)$是$\operatorname{Ball}\left( {f\left( {t}_{\text{start }}\right) ,{2\Delta }}\right)  \cap  {\mathbb{Z}}^{d}$中每$j$个点的最小包围球的中心集合，其基数至多为$\left( \begin{matrix} {\left( \left\lfloor  4\Delta \right\rfloor   + 1\right) }^{d} \\  j \end{matrix}\right)$。因此，${S}_{0}$至多包含

$$
\mathop{\sum }\limits_{{l = 0}}^{d}\left( \begin{matrix} {\left( \lfloor 4\Delta \rfloor  + 1\right) }^{d} \\  l + 1 \end{matrix}\right)  = O\left( {d{\left( \frac{e{\left( \lfloor 4\Delta \rfloor  + 1\right) }^{d}}{d + 1}\right) }^{d + 1}}\right) 
$$

points. Therefore, we have

个点。因此，我们有

THEOREM 3. There is an $O\left( {{d}^{3}\log \Delta }\right)$ -competitive online algorithm that tracks any function $f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$ .

定理3. 存在一种$O\left( {{d}^{3}\log \Delta }\right)$竞争的在线算法，可跟踪任何函数$f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$。

Running time. To find the Tukey median exactly requires $S$ to be explicitly maintained,which has size exponential in $d$ . Clarkson et al. [1993] proposed fast algorithms to compute an approximate Tukey median (a point with location depth $\Omega \left( {n/{d}^{2}}\right)$ ) via random sampling,but it seems difficult to sample from $S$ when $S$ is only implicitly maintained. We get around this problem with a new approach, presented in the next subsection,which also improves the competitive ratio by roughly a $d$ factor.

运行时间。要精确找到图基中位数（Tukey median），需要显式维护$S$，其规模在$d$上呈指数级。克拉克森（Clarkson）等人[1993]提出了通过随机采样来计算近似图基中位数（位置深度为$\Omega \left( {n/{d}^{2}}\right)$的点）的快速算法，但当$S$仅被隐式维护时，似乎很难从$S$中进行采样。我们通过下一小节介绍的一种新方法解决了这个问题，该方法还将竞争比大致提高了$d$倍。

<!-- Media -->

ALGORITHM 2: One round of $d$ -dimensional tracking via volume-cutting

算法2：通过体积切割进行$d$维跟踪的一轮

---

let $P = \operatorname{Ball}\left( {f\left( {t}_{\text{now }}\right) ,{\beta \Delta }}\right)$ ;

令$P = \operatorname{Ball}\left( {f\left( {t}_{\text{now }}\right) ,{\beta \Delta }}\right)$；

while $\left( {{\omega }_{\max }\left( P\right)  \geq  {\epsilon \Delta }}\right)$ do

当$\left( {{\omega }_{\max }\left( P\right)  \geq  {\epsilon \Delta }}\right)$时

	let $g\left( {t}_{\text{now }}\right)$ be the centroid of $P$ ;

	令$g\left( {t}_{\text{now }}\right)$为$P$的质心；

	send $g\left( {t}_{\text{now }}\right)$ to Bob;

	将$g\left( {t}_{\text{now }}\right)$发送给鲍勃；

	wait until $\begin{Vmatrix}{f\left( {t}_{\text{now }}\right)  - g\left( {t}_{\text{last }}\right) }\end{Vmatrix} > {\beta \Delta }$ ;

	等待直到$\begin{Vmatrix}{f\left( {t}_{\text{now }}\right)  - g\left( {t}_{\text{last }}\right) }\end{Vmatrix} > {\beta \Delta }$；

	$P \leftarrow  P \cap  \operatorname{Ball}\left( {f\left( {t}_{\text{now }}\right) ,{\beta \Delta }}\right)$ ;

---

<!-- Media -->

### 3.2. Algorithms by Volume-Cutting

### 3.2. 通过体积切割的算法

In this section,we first consider $\beta  = 1 + \epsilon$ ,and then show that we can set $\epsilon$ small enough to obtain an algorithm for the $\beta  = 1$ case. Before proceeding to the new algorithm,note that an $O\left( {{d}^{2}\log \left( {d/\epsilon }\right) ,1 + \epsilon }\right)$ -competitive algorithm can be obtained by slightly modifying the algorithm in the previous section. However, as discussed earlier, this algorithm has running time exponential in $d$ . In this section we propose algorithms with polynomial running time (w.r.t. both $d$ and $\Delta$ ) and also improved competitive ratios. These new algorithms use a volume-cutting technique, which is similar in spirits to many convex optimization algorithms.

在本节中，我们首先考虑$\beta  = 1 + \epsilon$，然后证明我们可以将$\epsilon$设置得足够小，从而得到适用于$\beta  = 1$情况的算法。在介绍新算法之前，请注意，通过对前一节中的算法进行轻微修改，可以得到一个$O\left( {{d}^{2}\log \left( {d/\epsilon }\right) ,1 + \epsilon }\right)$ - 竞争算法。然而，如前文所述，该算法的运行时间在$d$上是指数级的。在本节中，我们提出了具有多项式运行时间（关于$d$和$\Delta$）且竞争比有所改进的算法。这些新算法采用了体积切割技术，其原理与许多凸优化算法类似。

##### 3.2.1.The case with $\beta  = 1 + \epsilon$

##### 3.2.1. $\beta  = 1 + \epsilon$的情况

Definition 2 (Directional Width). For a set $P$ of points in ${\mathbb{R}}^{d}$ ,and a unit direction $\mu$ ,the directional widths of $P$ in direction $\mu$ is ${\omega }_{\mu }\left( P\right)  = \mathop{\max }\limits_{{p \in  P}}\langle \mu ,p\rangle  - \mathop{\min }\limits_{{p \in  P}}\langle \mu ,p\rangle$ , where $\langle \mu ,p\rangle$ is the standard inner product.

定义2（方向宽度）。对于${\mathbb{R}}^{d}$中的点集$P$和单位方向$\mu$，$P$在方向$\mu$上的方向宽度为${\omega }_{\mu }\left( P\right)  = \mathop{\max }\limits_{{p \in  P}}\langle \mu ,p\rangle  - \mathop{\min }\limits_{{p \in  P}}\langle \mu ,p\rangle$，其中$\langle \mu ,p\rangle$是标准内积。

The centroid of $P$ is the intersection of hyperplanes that divide $P$ into two parts of equal moments. Let ${\omega }_{\max }\left( P\right) ,{\omega }_{\min }\left( P\right)$ be the maximum and minimum directional widths of $P$ ,respectively. Our volume-cutting algorithm also proceeds in rounds; the procedure for each round is outlined in Algorithm 2.

$P$的质心是将$P$分成两个矩相等部分的超平面的交点。设${\omega }_{\max }\left( P\right) ,{\omega }_{\min }\left( P\right)$分别为$P$的最大和最小方向宽度。我们的体积切割算法也是分轮进行的；每一轮的过程概述于算法2中。

There are two differences between Algorithm 1 and 2. First, we now do not maintain the set $S$ -instead we maintain $P$ as the intersection of a collection of balls. Note that $P$ could be maintained efficiently since the number of intersecting balls is polynomial in $d$ and $\log \Delta$ as we will show later. Second,instead of sending the median of $P \cap  S$ , we send the centroid of $P$ to Bob. The correctness of the algorithm is obvious,since any point in $P$ is within a distance of ${\beta \Delta }$ to $f\left( {t}_{\text{now }}\right)$ . As for the competitive ratio,it is easy to see that $P$ always contains $S$ . Thus when $P$ contains no point in ${S}_{0}$ ,we can safely terminate the round,knowing that ${\mathcal{A}}_{\mathrm{{OPT}}}$ must have sent a message. However,we cannot simply repeat the algorithm and wait until $P$ is empty,since it may never be. Instead we will stop the round when the maximum width of $P$ is small enough (we will show later how to conduct this test efficiently), and then argue that when this happens, $S$ must be empty.

算法1和算法2有两个区别。首先，我们现在不维护集合$S$，而是维护$P$作为一组球的交集。请注意，由于相交球的数量在$d$和$\log \Delta$上是多项式的（我们稍后会证明），因此可以高效地维护$P$。其次，我们不是发送$P \cap  S$的中位数，而是将$P$的质心发送给鲍勃。该算法的正确性是显而易见的，因为$P$中的任何点到$f\left( {t}_{\text{now }}\right)$的距离都在${\beta \Delta }$以内。至于竞争比，很容易看出$P$始终包含$S$。因此，当$P$中不包含${S}_{0}$中的任何点时，我们可以安全地终止这一轮，因为我们知道${\mathcal{A}}_{\mathrm{{OPT}}}$一定发送了消息。然而，我们不能简单地重复该算法并等待$P$为空，因为它可能永远不会为空。相反，当$P$的最大宽度足够小时，我们将停止这一轮（我们稍后会说明如何高效地进行此测试），然后证明当这种情况发生时，$S$一定为空。

We need the following result, proved by Grunbaum [1960].

我们需要格伦鲍姆（Grunbaum）[1960]证明的以下结果。

LEMMA 2 ([GRUNBAUM 1960]). For a convex set $P$ in ${\mathbb{R}}^{d}$ ,any halfspace that contains the centroid of $P$ also contains at least $1/e$ of the volume of $P$ .

引理2（[格伦鲍姆（GRUNBAUM）1960]）。对于${\mathbb{R}}^{d}$中的凸集$P$，任何包含$P$质心的半空间也至少包含$P$体积的$1/e$。

<!-- Media -->

<!-- figureText: ${B}_{o}$ cap with diameter $\leq  \lambda$ (c) ${D}_{i}$ $B$ (a) ${o}_{1}$ (b) -->

<img src="https://cdn.noedgeai.com/0195c904-cf12-7bd7-9ed3-4fa404f6957a_8.jpg?x=374&y=341&w=1041&h=633&r=0"/>

Fig. 2. The relation between ${\omega }_{\min }\left( P\right)$ and ${\omega }_{\max }\left( P\right)$ .

图2. ${\omega }_{\min }\left( P\right)$和${\omega }_{\max }\left( P\right)$之间的关系。

<!-- Media -->

Since Ball $\left( {f\left( {t}_{\text{now }}\right) ,{\beta \Delta }}\right)$ is contained in a halfspace not containing the centroid of $P$ , every time a communication is triggered in Algorithm 2, we have

由于球$\left( {f\left( {t}_{\text{now }}\right) ,{\beta \Delta }}\right)$包含在一个不包含$P$质心的半空间中，因此在算法2中每次触发通信时，我们有

$$
\frac{\operatorname{vol}\left( {P \cap  \operatorname{Ball}\left( {f\left( {t}_{\text{now }}\right) ,{\beta \Delta }}\right) }\right) }{\operatorname{vol}\left( P\right) } < 1 - \frac{1}{e},
$$

that is,the volume of the convex set $P$ containing $S$ will be decreased by a constant factor.

也就是说，包含$S$的凸集$P$的体积将减少一个常数因子。

The rest of our task is to bound the iterations in each round. At first glance, the number of iterations could be endless,since $P$ might be cut into thinner and thinner slices. Fortunately, we can show that such a situation will not happen, by making use of the fact that we are cutting $P$ using a series of balls with radii that are not too large.

我们任务的其余部分是对每一轮的迭代次数进行界定。乍一看，迭代次数可能是无穷的，因为$P$可能会被切割成越来越薄的切片。幸运的是，我们可以利用这样一个事实，即我们使用一系列半径不是太大的球来切割$P$，从而证明这种情况不会发生。

LEMMA 3. If ${\omega }_{\max }\left( P\right)  \geq  {\epsilon \Delta }$ ,then ${\omega }_{\min }\left( P\right)  = \Omega \left( {{\epsilon }^{2}\Delta }\right)$ .

引理3. 如果${\omega }_{\max }\left( P\right)  \geq  {\epsilon \Delta }$，那么${\omega }_{\min }\left( P\right)  = \Omega \left( {{\epsilon }^{2}\Delta }\right)$。

Before proving Lemma 3, we first show the following facts.

在证明引理3之前，我们先证明以下事实。

LEMMA 4. Let $H$ be any supporting hyperplane of $P$ at $p \in  \partial P$ ,that is, $H$ contains $p$ and $P$ is contained in one of the two halfspaces bounded by $H$ . Then there is a ball $B$ with radius ${\beta \Delta }$ such that $H$ is tangent to $B$ at $p$ and $B$ contains $P$ .

引理4. 设$H$是$P$在$p \in  \partial P$处的任意一个支撑超平面，即$H$包含$p$且$P$包含在由$H$所界定的两个半空间之一中。那么存在一个半径为${\beta \Delta }$的球$B$，使得$H$在$p$处与$B$相切，并且$B$包含$P$。

Proof. It is easy to see that there is a unique ball $B$ with center ${o}_{B}$ and radius ${\beta \Delta }$ such that $H$ is tangent to $B$ at $p$ and $B$ is on the same side of $H$ as $P$ . We show that $P$ must be contained in $B$ . If this were not the case,there must be a point $q \in  P$ such that $q \notin  B$ . Let $J$ be the two-dimensional plane spanned by ${o}_{B},p$ and $q$ ,intersecting $H$ at line $l$ ; Figure 2(a) shows the situation on $J$ . Suppose $P$ is the intersection of ${B}_{i},i = 1,\ldots ,m$ . Clearly,the intersection between $J$ and any ${B}_{i}$ is a disk ${D}_{i}$ containing $p$ and $q$ . Simple planar geometry shows that $\partial {D}_{i}$ must intersect $l$ at two points,since the radius of ${D}_{i}$ is no more than ${\beta \Delta }$ and $\begin{Vmatrix}{{o}_{B}q}\end{Vmatrix} > {\beta \Delta }$ . Let ${u}_{i},{v}_{i}$ be the two intersection points between $\partial {D}_{i}$ and $l$ ,and ${p}^{\prime }$ be the projection of $p$ on line $l$ . It is also easy to see that one of ${u}_{i},{v}_{i}$ ,say ${v}_{i}$ ,is different than $p$ and lies at the same side of $p$ as ${q}^{\prime }$ . Therefore, the intersection of all such disks ${D}_{i}\left( {1 \leq  i \leq  m}\right)$ must contain a segment $\overline{p{v}_{j}}$ ,where ${v}_{j}$ is the closest point to $p$ among all the points ${v}_{i},i = 1,\ldots ,m$ ,which means that $P$ ,the intersection of all the ${B}_{i}$ ,must lie on both sides of $H$ -a contradiction.

证明。容易看出，存在一个唯一的以${o}_{B}$为圆心、半径为${\beta \Delta }$的球$B$，使得$H$在$p$处与$B$相切，并且$B$与$P$位于$H$的同一侧。我们证明$P$必定包含在$B$中。如果不是这样，那么必定存在一个点$q \in  P$，使得$q \notin  B$。设$J$是由${o}_{B},p$和$q$所张成的二维平面，它与$H$相交于直线$l$；图2(a)展示了$J$上的情况。假设$P$是${B}_{i},i = 1,\ldots ,m$的交点。显然，$J$与任何${B}_{i}$的交集是一个包含$p$和$q$的圆盘${D}_{i}$。简单的平面几何知识表明，$\partial {D}_{i}$必定与$l$相交于两点，因为${D}_{i}$的半径不超过${\beta \Delta }$且$\begin{Vmatrix}{{o}_{B}q}\end{Vmatrix} > {\beta \Delta }$。设${u}_{i},{v}_{i}$是$\partial {D}_{i}$与$l$的两个交点，${p}^{\prime }$是$p$在直线$l$上的投影。同样容易看出，${u}_{i},{v}_{i}$中的一个，比如${v}_{i}$，与$p$不同，并且与${q}^{\prime }$位于$p$的同一侧。因此，所有这样的圆盘${D}_{i}\left( {1 \leq  i \leq  m}\right)$的交集必定包含一个线段$\overline{p{v}_{j}}$，其中${v}_{j}$是所有点${v}_{i},i = 1,\ldots ,m$中离$p$最近的点，这意味着所有${B}_{i}$的交集$P$必定位于$H$的两侧——这是一个矛盾。

LEMMA 5. Let $M$ be the intersection of two balls of radius $r$ in ${R}^{d}$ . If ${\omega }_{\max }\left( M\right)  \geq  {\epsilon r}$ , then ${\omega }_{\min }\left( M\right)  = \Omega \left( {{\epsilon }^{2}r}\right)$ .

引理5. 设$M$是${R}^{d}$中两个半径为$r$的球的交集。如果${\omega }_{\max }\left( M\right)  \geq  {\epsilon r}$，那么${\omega }_{\min }\left( M\right)  = \Omega \left( {{\epsilon }^{2}r}\right)$。

Proof. Let ${B}_{1},{B}_{2}$ be two balls whose intersection is $M$ and let ${o}_{1},{o}_{2}$ be their centers, respectively. Let ${S}_{1},{S}_{2}$ be the boundary of ${B}_{1}$ and ${B}_{2}$ . It is clear that the intersection of ${S}_{1}$ and ${S}_{2}$ is a(d - 2)-dimensional sphere $S$ . Let $p$ be an arbitrary point on $S$ ,and let $J$ be the two-dimensional plane passing through $p,{o}_{1},{o}_{2}$ ,and intersecting $S$ at another point $q$ . It is easy to see that $\parallel {pq}\parallel$ is equal to the maximum width of $M$ ,and ${\omega }_{\min }\left( M\right)$ is equal to $2\left( {r - \sqrt{{r}^{2} - {\left( \parallel pq\parallel /2\right) }^{2}}}\right)$ ; see Figure 2(b). Thus if $\parallel {pq}\parallel  \geq  {\epsilon r}$ ,then ${\omega }_{\min }\left( M\right)  = \Omega \left( {{\epsilon }^{2}r}\right) .$

证明。设${B}_{1},{B}_{2}$为两个球，它们的交集为$M$，并设${o}_{1},{o}_{2}$分别为它们的球心。设${S}_{1},{S}_{2}$为${B}_{1}$和${B}_{2}$的边界。显然，${S}_{1}$和${S}_{2}$的交集是一个(d - 2)维球面$S$。设$p$为$S$上的任意一点，并设$J$为过$p,{o}_{1},{o}_{2}$且与$S$相交于另一点$q$的二维平面。容易看出，$\parallel {pq}\parallel$等于$M$的最大宽度，且${\omega }_{\min }\left( M\right)$等于$2\left( {r - \sqrt{{r}^{2} - {\left( \parallel pq\parallel /2\right) }^{2}}}\right)$；见图2(b)。因此，如果$\parallel {pq}\parallel  \geq  {\epsilon r}$，那么${\omega }_{\min }\left( M\right)  = \Omega \left( {{\epsilon }^{2}r}\right) .$

Proof. (Lemma 3) Let $Q$ be a polytope inscribed in $P$ such that the diameter of every cap formed by the intersection of $P$ and a halfspace bounded by the hyperplane containing a face of $\partial Q$ is no more than $\lambda$ ; see Figure 2(c). Let $\mu$ be the direction in which the directional width of $Q$ is minimized. Let ${H}_{p}$ and ${H}_{q}$ be the two parallel supporting hyperplanes of $Q$ orthogonal to $\mu$ . Let $p,q$ be two points on $Q \cap  {H}_{p}$ and $Q \cap  {H}_{q}$ ,respectively,so that $\overline{pq}$ is in the direction of $\mu$ . Such two points must exist, since $\dot{Q}$ is a polytope. Let ${H}_{x}$ and ${H}_{y}$ be the two hyperplanes parallel to ${H}_{p}$ and ${H}_{q}$ and support $P$ at $x$ and $y$ ,respectively. Suppose the line $\overline{pq}$ intersects ${H}_{x}$ and ${H}_{y}$ at ${x}^{\prime }$ and ${y}^{\prime }$ ,respectively.

证明。（引理3）设$Q$为内接于$P$的多面体，使得由$P$与包含$\partial Q$一个面的超平面所界定的半空间的交集形成的每个球冠的直径不超过$\lambda$；见图2(c)。设$\mu$为使$Q$的方向宽度最小化的方向。设${H}_{p}$和${H}_{q}$为与$\mu$正交的$Q$的两个平行支撑超平面。设$p,q$分别为$Q \cap  {H}_{p}$和$Q \cap  {H}_{q}$上的两点，使得$\overline{pq}$沿$\mu$的方向。这样的两点必定存在，因为$\dot{Q}$是一个多面体。设${H}_{x}$和${H}_{y}$为分别平行于${H}_{p}$和${H}_{q}$且在$x$和$y$处支撑$P$的两个超平面。假设直线$\overline{pq}$分别与${H}_{x}$和${H}_{y}$相交于${x}^{\prime }$和${y}^{\prime }$。

From Lemma 4 we know that there is a ball ${B}_{{o}_{1}^{\prime }}$ centered at ${o}_{1}^{\prime }$ with radius ${\beta \Delta }$ containing $P$ and tangent to ${H}_{x}$ . Pick ${o}_{1}$ on the line $\overline{pq}$ between $p$ and $q$ such that $\begin{Vmatrix}{{o}_{1}{x}^{\prime }}\end{Vmatrix} = \begin{Vmatrix}{{o}_{1}^{\prime }x}\end{Vmatrix}$ . By triangle inequality it is easy to see that the ball ${B}_{{o}_{1}}$ centered at ${o}_{1}$ with radius $\left( {{\beta \Delta } + \lambda }\right)$ must contain ${B}_{{o}_{1}^{\prime }}$ and thereby contain the convex set $P$ . Similarly, there is another ball ${B}_{{o}_{2}}$ entered at ${o}_{2}\left( {{o}_{2} \in  \overline{pq}}\right)$ with radius $\left( {{\beta \Delta } + \lambda }\right)$ containing $P$ if we consider $y,{y}^{\prime }$ instead of $x,{x}^{\prime }$ . Let ${p}^{\prime } = \partial {B}_{{o}_{1}} \cap  \overline{pq}$ and ${q}^{\prime } = \partial {B}_{{o}_{2}} \cap  \overline{pq}$ . We have

从引理4我们知道，存在一个以${o}_{1}^{\prime }$为中心、半径为${\beta \Delta }$的球${B}_{{o}_{1}^{\prime }}$，它包含$P$且与${H}_{x}$相切。在$p$和$q$之间的直线$\overline{pq}$上选取${o}_{1}$，使得$\begin{Vmatrix}{{o}_{1}{x}^{\prime }}\end{Vmatrix} = \begin{Vmatrix}{{o}_{1}^{\prime }x}\end{Vmatrix}$。根据三角不等式，很容易看出以${o}_{1}$为中心、半径为$\left( {{\beta \Delta } + \lambda }\right)$的球${B}_{{o}_{1}}$必定包含${B}_{{o}_{1}^{\prime }}$，从而包含凸集$P$。类似地，如果我们考虑$y,{y}^{\prime }$而非$x,{x}^{\prime }$，则存在另一个以${o}_{2}\left( {{o}_{2} \in  \overline{pq}}\right)$为中心、半径为$\left( {{\beta \Delta } + \lambda }\right)$的球${B}_{{o}_{2}}$包含$P$。令${p}^{\prime } = \partial {B}_{{o}_{1}} \cap  \overline{pq}$和${q}^{\prime } = \partial {B}_{{o}_{2}} \cap  \overline{pq}$。我们有

$$
\parallel {pq}\parallel  \leq  {\omega }_{\min }\left( P\right)  \leq  \begin{Vmatrix}{{p}^{\prime }{q}^{\prime }}\end{Vmatrix}.
$$

Note that $\begin{Vmatrix}{{p}^{\prime }{q}^{\prime }}\end{Vmatrix}$ is the minimum width of $M = {B}_{{o}_{1}} \cap  {B}_{{o}_{2}}$ . By Lemma 5,we know that ${\omega }_{\max }\left( M\right)  = O\left( {\begin{Vmatrix}{{p}^{\prime }{q}^{\prime }}\end{Vmatrix}/\epsilon }\right)$ ,provided that ${\omega }_{\max }\left( M\right)  \geq  {\epsilon \Delta }$ . Finally,if we choose $\lambda$ sufficiently small,we have ${\omega }_{\min }\left( P\right)  \geq  \Omega \left( \begin{Vmatrix}{{p}^{\prime }{q}^{\prime }}\end{Vmatrix}\right)  \geq  \Omega \left( {\epsilon  \cdot  {\omega }_{\max }\left( M\right) }\right)  \geq  \Omega \left( {\epsilon  \cdot  {\omega }_{\max }\left( P\right) }\right)$ .

注意到$\begin{Vmatrix}{{p}^{\prime }{q}^{\prime }}\end{Vmatrix}$是$M = {B}_{{o}_{1}} \cap  {B}_{{o}_{2}}$的最小宽度。根据引理5，我们知道，若${\omega }_{\max }\left( M\right)  \geq  {\epsilon \Delta }$成立，则${\omega }_{\max }\left( M\right)  = O\left( {\begin{Vmatrix}{{p}^{\prime }{q}^{\prime }}\end{Vmatrix}/\epsilon }\right)$。最后，如果我们选择足够小的$\lambda$，我们有${\omega }_{\min }\left( P\right)  \geq  \Omega \left( \begin{Vmatrix}{{p}^{\prime }{q}^{\prime }}\end{Vmatrix}\right)  \geq  \Omega \left( {\epsilon  \cdot  {\omega }_{\max }\left( M\right) }\right)  \geq  \Omega \left( {\epsilon  \cdot  {\omega }_{\max }\left( P\right) }\right)$。

The lower bound on the minimum width implies a lower bound on the volume of $P$ . Formally, we have

最小宽度的下界意味着$P$体积的下界。形式上，我们有

LEMMA 6. Let $K$ be a convex set in ${\mathbb{R}}^{d}$ . If ${\omega }_{\mu }\left( K\right)  \geq  r$ for all $\mu  \in  {S}^{d - 1}$ ,then $\operatorname{vol}\left( K\right)  \geq$ ${r}^{d}/d!$ .

引理6。设$K$是${\mathbb{R}}^{d}$中的一个凸集。如果对于所有的$\mu  \in  {S}^{d - 1}$都有${\omega }_{\mu }\left( K\right)  \geq  r$，那么$\operatorname{vol}\left( K\right)  \geq$ ${r}^{d}/d!$。

Proof. Since all of the directional width of $K$ is larger than $r$ ,the diameter of $K$ must also be larger than $r$ . Let ${p}_{1},{q}_{1} \in  K$ be the two points with the largest distance in $K$ ,and let ${\mu }_{1} = \overrightarrow{{p}_{1}{q}_{1}}$ . We then pick a direction ${\mu }_{2}$ that is orthogonal to ${\mu }_{1}$ . Let ${p}_{2},{q}_{2} \in  K$ be two extreme points in ${\mu }_{2}$ ,connect ${p}_{2},{q}_{2}$ to ${p}_{1},{q}_{1}$ ,respectively,forming a convex quadrilateral ${Q}_{2}$ in ${\mathbb{R}}^{2}$ (with basis ${\mu }_{1},{\mu }_{2}$ ). We keep on doing this,that is,we pick a third direction ${\mu }_{3}$ that is orthogonal to both ${\mu }_{1}$ and ${\mu }_{2}$ ,find two extreme points ${p}_{3},{q}_{3}$ in this direction,and then connect ${p}_{3}$ and ${q}_{3}$ to all ${p}_{i}$ and ${q}_{i}\left( {1 \leq  i \leq  2}\right)$ ,forming a convex polytope ${Q}_{3}$ in ${\mathbb{R}}^{3}$ (with basis ${\mu }_{1},{\mu }_{2}$ and ${\mu }_{3}$ ),and so on. After $d$ steps,we obtain a convex polytope ${Q}_{d}$ in ${\mathbb{R}}^{d}$ whose volume must be no smaller than ${r}^{d}/d!$ . Therefore,the volume of $K$ in ${\mathbb{R}}^{d}$ must also be no smaller than ${r}^{d}/d!$ ,since ${Q}_{d}$ is contained in $K$ .

证明。由于$K$的所有方向宽度都大于$r$，那么$K$的直径也必定大于$r$。设${p}_{1},{q}_{1} \in  K$为$K$中距离最大的两个点，并设${\mu }_{1} = \overrightarrow{{p}_{1}{q}_{1}}$。然后我们选取一个与${\mu }_{1}$正交的方向${\mu }_{2}$。设${p}_{2},{q}_{2} \in  K$为${\mu }_{2}$方向上的两个端点，分别连接${p}_{2},{q}_{2}$和${p}_{1},{q}_{1}$，在${\mathbb{R}}^{2}$中形成一个凸四边形${Q}_{2}$（以${\mu }_{1},{\mu }_{2}$为底边）。我们继续这样操作，即选取一个与${\mu }_{1}$和${\mu }_{2}$都正交的第三个方向${\mu }_{3}$，找到该方向上的两个端点${p}_{3},{q}_{3}$，然后将${p}_{3}$和${q}_{3}$连接到所有的${p}_{i}$和${q}_{i}\left( {1 \leq  i \leq  2}\right)$，在${\mathbb{R}}^{3}$中形成一个凸多面体${Q}_{3}$（以${\mu }_{1},{\mu }_{2}$和${\mu }_{3}$为底边），依此类推。经过$d$步后，我们在${\mathbb{R}}^{d}$中得到一个凸多面体${Q}_{d}$，其体积必定不小于${r}^{d}/d!$。因此，由于${Q}_{d}$包含在$K$中，$K$在${\mathbb{R}}^{d}$中的体积也必定不小于${r}^{d}/d!$。

By Lemma 3,we know that as long as ${\omega }_{\max }\left( P\right)  \geq  {\epsilon \Delta }$ ,the width of $P$ in all directions is at least $c \cdot  {\epsilon }^{2}\Delta$ for some constant $c$ ,which means that the volume of $P$ is at least ${\left( c \cdot  {\epsilon }^{2}\Delta \right) }^{d}/d$ ! by Lemma 6 . Then by Lemma 2,as well as the fact that at the beginning of the round, $\operatorname{vol}\left( P\right)  \leq  {\left( 4\Delta \right) }^{d}$ ,we know that after at most

根据引理3，我们知道只要${\omega }_{\max }\left( P\right)  \geq  {\epsilon \Delta }$，对于某个常数$c$，$P$在所有方向上的宽度至少为$c \cdot  {\epsilon }^{2}\Delta$，这意味着根据引理6，$P$的体积至少为${\left( c \cdot  {\epsilon }^{2}\Delta \right) }^{d}/d$！然后根据引理2，以及在这一轮开始时$\operatorname{vol}\left( P\right)  \leq  {\left( 4\Delta \right) }^{d}$这一事实，我们知道在算法2中最多进行

$$
\log \frac{{\left( 4\Delta \right) }^{d}}{{\left( c \cdot  {\epsilon }^{2}\Delta \right) }^{d}/d!} = O\left( {d\log \frac{d}{\epsilon }}\right)  \tag{1}
$$

triggerings of communication in Algorithm 2, ${\omega }_{\max }\left( P\right)$ will be less than ${\epsilon \Delta }$ . At this moment,consider the ${P}^{\prime }$ obtained by replacing all the balls in Algorithm 2 by balls with radius $\Delta$ . By triangle inequality,we know that ${P}^{\prime } = \varnothing$ . Recall that ${\mathcal{A}}_{\text{OPT }}$ is only allowed an error of $\Delta$ . Therefore, ${\mathcal{A}}_{\text{OPT }}$ must have already sent a message,since ${P}^{\prime } = \varnothing$ .

通信触发后，${\omega }_{\max }\left( P\right)$将小于${\epsilon \Delta }$。此时，考虑将算法2中的所有球替换为半径为$\Delta$的球后得到的${P}^{\prime }$。根据三角形不等式，我们知道${P}^{\prime } = \varnothing$。回想一下，${\mathcal{A}}_{\text{OPT }}$只允许有$\Delta$的误差。因此，由于${P}^{\prime } = \varnothing$，${\mathcal{A}}_{\text{OPT }}$必定已经发送了一条消息。

Running time. Generally, it is hard to compute the centroid of a convex body; see Rademacher [2007]. However, Bertsimas and Vempala [2004] showed that there is a randomized algorithm that computes an approximate centroid of a convex body given by a separation oracle. Formally, they proved the following.

运行时间。一般来说，计算凸体的质心是很困难的；参见拉德马赫（Rademacher）[2007]。然而，贝尔西马斯（Bertsimas）和文帕拉（Vempala）[2004]表明，存在一种随机算法，它可以根据分离 oracle 计算凸体的近似质心。正式地，他们证明了以下内容。

LEMMA 7 ([BERTSIMAS AND VEMPALA 2004]). Let $K$ be a convex body in ${R}^{d}$ given by a separation oracle,and a point in a ball of radius $\Delta$ that contains $K$ . If ${\omega }_{\min }\left( K\right)  \geq  r$ ,then there is a randomized algorithm with running time $\operatorname{poly}\left( {d,\log \left( \frac{\Delta }{r}\right) }\right)$ that computes,with high probability,the approximate centroid $z$ of a convex set $K$ such that any halfspace that contains $z$ also contains at least $1/3$ of the volume of $K$ .

引理7（[贝尔西马斯（Bertsimas）和文帕拉（Vempala），2004年]）。设$K$是${R}^{d}$中的一个凸体，由一个分离 oracle 给出，并且设一个点位于半径为$\Delta$且包含$K$的球内。如果${\omega }_{\min }\left( K\right)  \geq  r$，那么存在一个运行时间为$\operatorname{poly}\left( {d,\log \left( \frac{\Delta }{r}\right) }\right)$的随机算法，该算法以高概率计算出凸集$K$的近似质心$z$，使得任何包含$z$的半空间也至少包含$K$体积的$1/3$。

In our case,since $P$ is the intersection of $O\left( {d\log \frac{d}{\epsilon }}\right)$ balls,we can simply implement the separation oracle by checking each of these balls one by one. Moreover, $f\left( {t}_{\text{start }}\right)$ could be used as the starting point $p$ required by Lemma 7 . We set $r = c \cdot  {\epsilon }^{2}\Delta$ ,thus computing the approximate centroid could be done in time poly $\left( {d,\log \left( \frac{1}{\epsilon }\right) }\right)$ . If the algorithm of Lemma 7 fails,then with high probability, ${\omega }_{\max }\left( P\right)  < {\epsilon \Delta }$ . This fact,together with the discussion after Lemma 6,provides us a way to avoid monitoring the maximum width of $P$ at the beginning of each iteration in Algorithm 2 (which is expensive). More precisely, we slightly modify Algorithm 2 as follows.

在我们的情况中，由于$P$是$O\left( {d\log \frac{d}{\epsilon }}\right)$个球的交集，我们可以通过逐个检查这些球来简单地实现分离 oracle。此外，$f\left( {t}_{\text{start }}\right)$可以用作引理7所需的起始点$p$。我们设$r = c \cdot  {\epsilon }^{2}\Delta$，因此可以在多项式时间 poly $\left( {d,\log \left( \frac{1}{\epsilon }\right) }\right)$内计算近似质心。如果引理7的算法失败，那么以高概率有${\omega }_{\max }\left( P\right)  < {\epsilon \Delta }$。这一事实，再结合引理6之后的讨论，为我们提供了一种方法来避免在算法2的每次迭代开始时监测$P$的最大宽度（这是代价高昂的）。更准确地说，我们对算法2进行如下轻微修改。

(1) Line $2 \rightarrow$ while the number of iterations in the current round is no more than (1) $\mathbf{{do}}$

(1) 第$2 \rightarrow$行：当当前轮次的迭代次数不超过(1) $\mathbf{{do}}$

(2) Line $3 \rightarrow$ compute the approximate centroid of $P$ using the algorithm of Lemma 7 and assign it to $g\left( {t}_{\text{now }}\right)$ ; if the algorithm of Lemma 7 fails,terminate the current round;

(2) 第$3 \rightarrow$行：使用引理7的算法计算$P$的近似质心并将其赋值给$g\left( {t}_{\text{now }}\right)$；如果引理7的算法失败，则终止当前轮次；

THEOREM 4. There is an $\left( {O\left( {d\log \left( {d/\epsilon }\right) }\right) ,1 + \epsilon }\right)$ -competitive online algorithm to track any function $f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$ . The algorithm runs in time $\operatorname{poly}\left( {d,\log \frac{1}{\epsilon }}\right)$ at every time step.

定理4。存在一个$\left( {O\left( {d\log \left( {d/\epsilon }\right) }\right) ,1 + \epsilon }\right)$ - 竞争的在线算法来跟踪任何函数$f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$。该算法在每个时间步的运行时间为$\operatorname{poly}\left( {d,\log \frac{1}{\epsilon }}\right)$。

Remark. Note that the algorithm proposed in this section also works for tracking real-valued functions $f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{R}}^{d}$ .

注记。注意，本节提出的算法也适用于跟踪实值函数$f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{R}}^{d}$。

3.2.2. The case with $\beta  = 1$ . Recall that in Section 3.1,we showed that considering ${S}_{0} = {C}_{2} \cup  {C}_{3}\ldots  \cup  {C}_{d + 1}$ is enough. Since ${S}_{0}$ is the collection of points that are centers of the smallest enclosing balls of at most $d + 1$ points in ${\mathbb{Z}}^{d}$ ,the following fact can be established.

3.2.2. $\beta  = 1$的情况。回顾在3.1节中，我们表明考虑${S}_{0} = {C}_{2} \cup  {C}_{3}\ldots  \cup  {C}_{d + 1}$就足够了。由于${S}_{0}$是${\mathbb{Z}}^{d}$中至多$d + 1$个点的最小包围球的中心的点集，因此可以建立以下事实。

LEMMA 8. For any points $s = \left( {{x}_{1},\ldots ,{x}_{d}}\right)$ in ${S}_{0},{x}_{i}\left( {1 \leq  i \leq  d}\right)$ are fractions in the form of $\frac{y}{z}$ where $y,z$ are integers and $\left| z\right|  \leq  d!{\left( {16}{\Delta }^{2}d\right) }^{d}$ .

引理8。对于${S}_{0},{x}_{i}\left( {1 \leq  i \leq  d}\right)$中的任何点$s = \left( {{x}_{1},\ldots ,{x}_{d}}\right)$，它们都是$\frac{y}{z}$形式的分数，其中$y,z$是整数且$\left| z\right|  \leq  d!{\left( {16}{\Delta }^{2}d\right) }^{d}$。

Proof. For any $s \in  {S}_{0}$ ,let $B$ be one of the minimum enclosing balls of some points in ${\mathbb{Z}}^{d}$ centered at $s$ . Assume that there are $k\left( {2 \leq  k \leq  d + 1}\right)$ integer points ${q}_{1},{q}_{2},\ldots ,{q}_{k}$ lying on the boundary of $B$ . If $k = d + 1$ ,we can compute $s$ by solving a linear system of $d$ equations $\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{2}s}\end{Vmatrix},\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{3}s}\end{Vmatrix},\ldots ,\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{d + 1}s}\end{Vmatrix}$ ,or we can write them as $A{s}^{T} = b$ ,where $A = \left( {a}_{ij}\right)  \in  {\mathbb{Z}}^{d \times  d}$ and $b \in  {\mathbb{Z}}^{d \times  1}$ . Each coefficient ${a}_{ij}\left( {1 \leq  i,j \leq  d}\right)$ is an integer in the range of $\left\lbrack  {-{8\Delta },{8\Delta }}\right\rbrack$ and $b$ is a vector of integers. Thus by Cramer’s rule,

证明。对于任意$s \in  {S}_{0}$，设$B$是以$s$为圆心、包含${\mathbb{Z}}^{d}$中某些点的最小包围球之一。假设在$B$的边界上有$k\left( {2 \leq  k \leq  d + 1}\right)$个整点${q}_{1},{q}_{2},\ldots ,{q}_{k}$。如果$k = d + 1$，我们可以通过求解一个包含$d$个方程$\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{2}s}\end{Vmatrix},\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{3}s}\end{Vmatrix},\ldots ,\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{d + 1}s}\end{Vmatrix}$的线性方程组来计算$s$，或者我们可以将它们写成$A{s}^{T} = b$的形式，其中$A = \left( {a}_{ij}\right)  \in  {\mathbb{Z}}^{d \times  d}$且$b \in  {\mathbb{Z}}^{d \times  1}$。每个系数${a}_{ij}\left( {1 \leq  i,j \leq  d}\right)$是一个取值范围在$\left\lbrack  {-{8\Delta },{8\Delta }}\right\rbrack$内的整数，并且$b$是一个整数向量。因此，根据克莱姆法则，

$$
{x}_{i} = \frac{\det {A}_{i}}{\det A}
$$

where ${A}_{i}$ is the matrix formed by replacing the $i$ -th column of $A$ by $b$ . It is easy to see that $\left| {\det A}\right|  \leq  d!{\left( 8\Delta \right) }^{d}$ . If $2 \leq  k \leq  d$ ,then $s$ must be in the(k - 1)-dimensional subspace determined by ${q}_{1},{q}_{2},\ldots ,{q}_{k}$ . Hence,we can write $s = {\alpha }_{1}{q}_{1} + {\alpha }_{2}{q}_{2} + \cdots  + {\alpha }_{k}{q}_{k}$ ,where $\mathop{\sum }\limits_{{i = 1}}^{k}{\alpha }_{i} = 1$ . Together with $\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{2}s}\end{Vmatrix},\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{3}s}\end{Vmatrix},\ldots ,\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{k}s}\end{Vmatrix}$ ,we have a linear system of $k$ equalities. By similar arguments as before,we can show that each ${\alpha }_{i}\left( {1 \leq  i \leq  k}\right)$ is a fraction of form $\frac{{y}_{i}}{z}\left( {{y}_{i},z \in  \mathbb{Z}}\right)$ and $\left| z\right|$ is no more than $d!{\left( {16}{\Delta }^{2}d\right) }^{d}$ . Since $s = \mathop{\sum }\limits_{{i = 1}}^{k}{\alpha }_{i}{p}_{i}$ ,we know that each coordinate of $s$ is in the form of $\frac{y}{z}$ ,where $y$ and $z$ are integers and $\left| z\right|  \leq  d!{\left( {16}{\Delta }^{2}d\right) }^{d}$ .

其中${A}_{i}$是将$A$的第$i$列替换为$b$后形成的矩阵。很容易看出$\left| {\det A}\right|  \leq  d!{\left( 8\Delta \right) }^{d}$。如果$2 \leq  k \leq  d$，那么$s$必定位于由${q}_{1},{q}_{2},\ldots ,{q}_{k}$所确定的(k - 1)维子空间中。因此，我们可以写成$s = {\alpha }_{1}{q}_{1} + {\alpha }_{2}{q}_{2} + \cdots  + {\alpha }_{k}{q}_{k}$的形式，其中$\mathop{\sum }\limits_{{i = 1}}^{k}{\alpha }_{i} = 1$。结合$\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{2}s}\end{Vmatrix},\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{3}s}\end{Vmatrix},\ldots ,\begin{Vmatrix}{{q}_{1}s}\end{Vmatrix} = \begin{Vmatrix}{{q}_{k}s}\end{Vmatrix}$，我们得到一个包含$k$个等式的线性方程组。通过与之前类似的论证，我们可以证明每个${\alpha }_{i}\left( {1 \leq  i \leq  k}\right)$是$\frac{{y}_{i}}{z}\left( {{y}_{i},z \in  \mathbb{Z}}\right)$形式的分数，并且$\left| z\right|$不超过$d!{\left( {16}{\Delta }^{2}d\right) }^{d}$。由于$s = \mathop{\sum }\limits_{{i = 1}}^{k}{\alpha }_{i}{p}_{i}$，我们知道$s$的每个坐标都具有$\frac{y}{z}$的形式，其中$y$和$z$是整数且$\left| z\right|  \leq  d!{\left( {16}{\Delta }^{2}d\right) }^{d}$。

By this observation,we know that the distance between any two points in ${S}_{0}$ is at least $1/{\left( d!{\left( {16}{\Delta }^{2}d\right) }^{d}\right) }^{2}$ . Therefore,by setting $\epsilon  = 1/{8\Delta }{\left( d!{\left( {16}{\Delta }^{2}d\right) }^{d}\right) }^{2}$ ,we know that once ${\omega }_{\max }\left( P\right)  < {\epsilon \Delta }$ ,there is at most one point of ${S}_{0}$ in $P$ . The rest of our job is to find such a point if it exists. Once the point is found, we just send it to Bob, and the round will terminate as soon as $f\left( {t}_{\text{now }}\right)$ gets $\Delta$ away from this point. However,directly computing such a point might be expensive. Instead we use an indirect way to find the last surviving point.

通过这一观察，我们知道${S}_{0}$中任意两点之间的距离至少为$1/{\left( d!{\left( {16}{\Delta }^{2}d\right) }^{d}\right) }^{2}$。因此，通过设置$\epsilon  = 1/{8\Delta }{\left( d!{\left( {16}{\Delta }^{2}d\right) }^{d}\right) }^{2}$，我们知道一旦${\omega }_{\max }\left( P\right)  < {\epsilon \Delta }$成立，$P$中最多只有${S}_{0}$中的一个点。我们接下来的工作就是找出这个点（如果它存在的话）。一旦找到这个点，我们就将其发送给鲍勃，并且当$f\left( {t}_{\text{now }}\right)$与这个点的距离达到$\Delta$时，这一轮就会终止。然而，直接计算这样一个点的代价可能很高。相反，我们采用一种间接的方法来找到最后剩下的点。

We say a number $x$ is good if $x = \frac{y}{z}$ with $y,z \in  \mathbb{Z}$ and $\left| z\right|  \leq  d!{\left( {16}{\Delta }^{2}d\right) }^{d}$ . A point $s$ is good if all of its coordinates are good. The basic idea is that if we can successfully compute the centroid $p$ of $P$ ,we can snap $p$ to its nearest good point, $s$ . If there is a point ${s}^{\prime } \in  {S}_{0}$ inside $P$ ,then we must have ${s}^{\prime } = s$ . Thus if $s \notin  P$ ,we simply terminate the current round; otherwise $s$ must be the last point of ${S}_{0}$ in $P$ . The difficulty is that ${\omega }_{\min }\left( P\right)$ could be very small,so that Lemma 7 could not be applied directly. To avoid such a situation,we expand $P$ slightly by increasing all of the balls’ radii from $\Delta$ to $\left( {1 + \epsilon }\right) \Delta$ . Denote by ${P}^{\prime }$ the intersection of these enlarged balls. The observation is that by our choice of $\epsilon$ ,if there is a point ${s}^{\prime }$ of ${S}_{0}$ in $P$ ,then ${s}^{\prime }$ is still the only point of ${S}_{0}$ in ${P}^{\prime }$ . Now we can apply the algorithm of Lemma 7 on ${P}^{\prime }$ ,with $r = c \cdot  {\epsilon }^{2}\Delta$ . If the algorithm fails,we know that $P$ must be empty. Otherwise we obtain a point $p \in  {P}^{\prime }$ . Finally,we find $s$ by rounding each coordinate of $p$ to its nearest good number,and check if $s \in  P$ . The rounding could be done in polynomial time according to a theorem by Khintchine (cf. Korte and Vygen [2007, Chapter 4]).

我们称一个数$x$是“良好的”（good），如果满足$x = \frac{y}{z}$，其中$y,z \in  \mathbb{Z}$且$\left| z\right|  \leq  d!{\left( {16}{\Delta }^{2}d\right) }^{d}$。如果一个点$s$的所有坐标都是良好的，那么这个点就是良好的。基本思路是，如果我们能够成功计算出$P$的质心$p$，我们就可以将$p$调整到与其最近的良好点$s$。如果$P$内部存在一个点${s}^{\prime } \in  {S}_{0}$，那么我们必然有${s}^{\prime } = s$。因此，如果$s \notin  P$成立，我们就直接终止当前轮次；否则，$s$必定是${S}_{0}$在$P$中的最后一个点。困难在于${\omega }_{\min }\left( P\right)$可能非常小，以至于不能直接应用引理7。为了避免这种情况，我们通过将所有球的半径从$\Delta$增加到$\left( {1 + \epsilon }\right) \Delta$来稍微扩大$P$。用${P}^{\prime }$表示这些扩大后的球的交集。观察发现，根据我们对$\epsilon$的选择，如果$P$中存在${S}_{0}$的一个点${s}^{\prime }$，那么${s}^{\prime }$仍然是${S}_{0}$在${P}^{\prime }$中的唯一一点。现在我们可以对${P}^{\prime }$应用引理7的算法，其中$r = c \cdot  {\epsilon }^{2}\Delta$。如果算法失败，我们就知道$P$必定为空。否则，我们会得到一个点$p \in  {P}^{\prime }$。最后，我们通过将$p$的每个坐标四舍五入到最近的良好数来找到$s$，并检查$s \in  P$是否成立。根据欣钦（Khintchine）的一个定理（参见Korte和Vygen [2007，第4章]），这种四舍五入可以在多项式时间内完成。

By the choice of $\epsilon$ and Theorem 4,we obtain the following.

根据对$\epsilon$的选择和定理4，我们可以得到以下结论。

THEOREM 5. There is an $O\left( {{d}^{2}\log \left( {d\Delta }\right) }\right)$ -competitive online algorithm to track any function $f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$ . The algorithm runs in time $\operatorname{poly}\left( {d,\log \Delta }\right)$ at every time step.

定理5. 存在一种$O\left( {{d}^{2}\log \left( {d\Delta }\right) }\right)$ - 竞争的在线算法来跟踪任何函数$f : {\mathbb{Z}}^{ + } \rightarrow  {\mathbb{Z}}^{d}$。该算法在每个时间步的运行时间为$\operatorname{poly}\left( {d,\log \Delta }\right)$。

### 3.3. Online Tracking a Dynamic Set

### 3.3. 在线跟踪动态集合

One of the main applications of online tracking in high dimensions is tracking a dynamic set. Formally,we want to track the function $f : {\mathbb{Z}}^{ + } \rightarrow  {2}^{U}$ ,where $U$ is a finite universe consisting of $d$ items. We can represent each set $X \in  {2}^{U}$ as a $\{ 0,1\}$ -vector in ${\mathbb{R}}^{d}$ ,and define the difference between two sets, $X$ and $Y$ ,to be the ${l}_{2}$ distance between the corresponding vectors in ${\mathbb{R}}^{d}$ (note that the Hamming distance between two sets is just the square of their ${l}_{2}$ distance). Ideally,Alice should send out subsets of $U$ to approximate $f\left( {t}_{\text{now }}\right)$ ,but applying our previous algorithms would send out vectors with fractional coordinates. Unfortunately, if we insist that Alice always sends a set, that is, a $\{ 0,1\}$ -vector to Bob,the competitive ratio would be exponentially large in $\Delta$ ,even if we allow a relatively larger $\beta$ ,as shown in the next theorem.

高维在线追踪的主要应用之一是追踪动态集合。形式上，我们想要追踪函数 $f : {\mathbb{Z}}^{ + } \rightarrow  {2}^{U}$ ，其中 $U$ 是一个由 $d$ 个元素组成的有限全域。我们可以将每个集合 $X \in  {2}^{U}$ 表示为 ${\mathbb{R}}^{d}$ 中的一个 $\{ 0,1\}$ 维向量，并将两个集合 $X$ 和 $Y$ 之间的差异定义为 ${\mathbb{R}}^{d}$ 中对应向量的 ${l}_{2}$ 距离（注意，两个集合之间的汉明距离恰好是它们 ${l}_{2}$ 距离的平方）。理想情况下，爱丽丝（Alice）应该发送 $U$ 的子集来近似 $f\left( {t}_{\text{now }}\right)$ ，但应用我们之前的算法会发送带有分数坐标的向量。不幸的是，如果我们坚持让爱丽丝总是向鲍勃（Bob）发送一个集合，即一个 $\{ 0,1\}$ 维向量，那么即使我们允许相对较大的 $\beta$ ，竞争比在 $\Delta$ 上也会呈指数级增长，如下一个定理所示。

THEOREM 6. Suppose that there is an $\left( {\alpha ,\beta }\right)$ -competitive algorithm for online tracking $f : {\mathbb{Z}}^{ + } \rightarrow  {2}^{U}$ and $\left| U\right|  > {\left( \beta \Delta \right) }^{2}$ ,if the algorithm can only send subsets of $U$ ,then $\alpha  = {2}^{\Omega \left( {\Delta }^{2}\right) }$ for any constant $\beta  < {19}/{18}$ .

定理 6. 假设存在一个用于在线追踪 $\left( {\alpha ,\beta }\right)$ 和 $f : {\mathbb{Z}}^{ + } \rightarrow  {2}^{U}$ 的 $\left| U\right|  > {\left( \beta \Delta \right) }^{2}$ 竞争算法，如果该算法只能发送 $U$ 的子集，那么对于任何常数 $\beta  < {19}/{18}$ ，有 $\alpha  = {2}^{\Omega \left( {\Delta }^{2}\right) }$ 。

Proof. Without lose of generality,let $H = \{ 0,1{\} }^{d}$ ,where $d$ is chosen to be ${\left( \beta \Delta \right) }^{2} + 1$ . Similar to the proof of theorem 2, we just need to show that the adversary can manipulate $f\left( t\right)$ so that a round will have at least $\alpha$ iterations.

证明. 不失一般性，设 $H = \{ 0,1{\} }^{d}$ ，其中 $d$ 被选为 ${\left( \beta \Delta \right) }^{2} + 1$ 。与定理 2 的证明类似，我们只需要证明对手可以操纵 $f\left( t\right)$ ，使得一轮至少有 $\alpha$ 次迭代。

Let ${S}_{0}$ be the set of possible vertices sent by ${\mathcal{A}}_{\text{OPT }}$ in its last communication,that is, all the vertices within distance $\Delta$ from $f\left( {t}_{\text{start }}\right)$ . The cardinality of ${S}_{0}$ is

设 ${S}_{0}$ 是 ${\mathcal{A}}_{\text{OPT }}$ 在其最后一次通信中发送的可能顶点的集合，即距离 $f\left( {t}_{\text{start }}\right)$ 不超过 $\Delta$ 的所有顶点。 ${S}_{0}$ 的基数为

$$
\left| {S}_{0}\right|  = \mathop{\sum }\limits_{{k = 1}}^{{\Delta }^{2}}\left( \begin{array}{l} d \\  k \end{array}\right)  = \Omega \left( {2}^{{\Delta }^{2}}\right) .
$$

The adversary,Carole,sets $S = {S}_{0}$ at the beginning of each round and then manipulates the value of the function $f$ according to the online algorithm ${\mathcal{A}}_{\text{SOL }}$ ,as follows. Whenever ${\mathcal{A}}_{\text{SOL }}$ sends $\mathbf{v} \in  H$ ,Carole changes $f$ to $\mathbf{u} = \mathbf{1} - \mathbf{v}$ ,that is,flipping all the coordinates of v. Since $\parallel \mathbf{v},\mathbf{u}\parallel  > {\beta \Delta },{\mathcal{A}}_{\mathrm{{SOL}}}$ has to communicate again. Every time Carole uses a value $\mathbf{u}$ for $f,S$ is cut as $S \leftarrow  S \cap  \operatorname{Ball}\left( {\mathbf{u},\Delta }\right)$ . So $S$ loses at most (let $\epsilon  = \beta  - 1 < 1/{18}$ )

对手卡罗尔（Carole）在每一轮开始时设置 $S = {S}_{0}$ ，然后根据在线算法 ${\mathcal{A}}_{\text{SOL }}$ 操纵函数 $f$ 的值，具体如下。每当 ${\mathcal{A}}_{\text{SOL }}$ 发送 $\mathbf{v} \in  H$ 时，卡罗尔将 $f$ 更改为 $\mathbf{u} = \mathbf{1} - \mathbf{v}$ ，即翻转向量 v 的所有坐标。由于 $\parallel \mathbf{v},\mathbf{u}\parallel  > {\beta \Delta },{\mathcal{A}}_{\mathrm{{SOL}}}$ 必须再次进行通信。每次卡罗尔使用一个值 $\mathbf{u}$ 时， $f,S$ 被削减为 $S \leftarrow  S \cap  \operatorname{Ball}\left( {\mathbf{u},\Delta }\right)$ 。因此 $S$ 最多损失（设 $\epsilon  = \beta  - 1 < 1/{18}$ ）

$$
\left| {H - \operatorname{Ball}\left( {\mathbf{u},\Delta }\right) }\right|  = \mathop{\sum }\limits_{{k = {\Delta }^{2} + 1}}^{d}\left( \begin{array}{l} d \\  k \end{array}\right)  \leq  \left( \begin{matrix} 2{\Delta }^{2} \\  {3\epsilon }{\Delta }^{2} \end{matrix}\right)  \leq  {\left( e/\epsilon \right) }^{{3\epsilon }{\Delta }^{2}}
$$

elements. Therefore, ${\mathcal{A}}_{\text{SOL }}$ will communicate at least

个元素。因此，在 ${\mathcal{A}}_{\text{SOL }}$ 变为空集之前， ${\mathcal{A}}_{\text{SOL }}$ 至少会进行

$$
\Omega \left( \frac{{2}^{{\Delta }^{2}}}{{\left( e/\epsilon \right) }^{{3\epsilon }{\Delta }^{2}}}\right)  = \Omega \left( {c}^{{\Delta }^{2}}\right) \;\left( {c > 1\text{ when }\epsilon  < 1/{18}}\right) 
$$

times before $S$ becomes empty.

次通信。

Therefore, to avoid an exponentially large competitive ratio, we have to allow the algorithm to send vectors with fractional coordinates. We can use the previously developed algorithms to guarantee that the ${l}_{2}$ distance between $f\left( {t}_{\text{now }}\right)$ and the fractional vector $g\left( {t}_{\text{last }}\right)$ sent by our algorithm is no more than $\Delta$ . If in some applications it is unnatural to report to the client, a vector with fractional values when the underlying function being tracked is a set,the tracker could convert the vector to a set $Y$ by probabilistically rounding every coordinate of $g\left( {t}_{\text{last }}\right)$ . It can be easily shown that the expected distance between $Y$ and $f\left( {t}_{\text{now }}\right)$ is no more that $\Delta$ .

因此，为了避免出现指数级大的竞争比，我们必须允许算法发送具有分数坐标的向量。我们可以使用之前开发的算法来保证$f\left( {t}_{\text{now }}\right)$与我们的算法发送的分数向量$g\left( {t}_{\text{last }}\right)$之间的${l}_{2}$距离不超过$\Delta$。如果在某些应用中，当被跟踪的底层函数是一个集合时，向客户端报告一个具有分数值的向量不太自然，跟踪器可以通过对$g\left( {t}_{\text{last }}\right)$的每个坐标进行概率性取整，将该向量转换为一个集合$Y$。可以很容易地证明，$Y$和$f\left( {t}_{\text{now }}\right)$之间的期望距离不超过$\Delta$。

## 4. ONLINE TRACKING WITH PREDICTIONS

## 4. 带预测的在线跟踪

In this section, we further generalize our model by considering predictions. We assume that Alice tries to predict the future trend of the function based on history, and then sends the prediction to Bob. If the actual function values do not deviate from the prediction by more than $\Delta$ ,no communication is necessary. One can imagine that when $f$ is well-behaved,using good predictions could greatly reduce communications. Indeed, the same approach has been taken in many heuristics in practice [Keralapura et al. 2006; Cormode et al. 2005; Cormode and Garofalakis 2005]. In this article we only consider the case where the algorithms (both the online and the offline) use linear functions as predictions,and for $d = 1$ ; the technique can be extended to more general prediction functions and high dimensions.

在本节中，我们通过考虑预测来进一步推广我们的模型。我们假设爱丽丝（Alice）试图根据历史情况预测函数的未来趋势，然后将预测结果发送给鲍勃（Bob）。如果实际函数值与预测值的偏差不超过$\Delta$，则无需进行通信。可以想象，当$f$表现良好时，使用良好的预测可以大大减少通信量。实际上，在许多实际的启发式方法中都采用了相同的方法[Keralapura等人，2006年；Cormode等人，2005年；Cormode和Garofalakis，2005年]。在本文中，我们仅考虑算法（包括在线算法和离线算法）使用线性函数作为预测的情况，对于$d = 1$；该技术可以扩展到更一般的预测函数和高维情况。

In one dimension, the offline problem is to approximate a function by a small number of straight line segments. O'Rourke [1981] gave a linear-time algorithm to compute the optimal solution. His algorithm is online but in the sense that the algorithm scans $f$ only once,and the partial solution computed so far is optimal for the portion of $f$ that has been scanned. However, the partial solution could keep changing at each time step as $f$ is observed. While in our problem,we need to make an immediate decision on what to communicate at each time step whenever $f$ deviates more than $\Delta$ from the prediction previously sent.

在一维情况下，离线问题是用少量的直线段来近似一个函数。奥罗克（O'Rourke）[1981]给出了一种线性时间算法来计算最优解。他的算法是在线算法，其含义是该算法仅对$f$扫描一次，并且到目前为止所计算出的部分解对于已扫描的$f$部分是最优的。然而，随着对$f$的观测，部分解可能在每个时间步都会发生变化。而在我们的问题中，每当$f$与之前发送的预测值的偏差超过$\Delta$时，我们需要在每个时间步立即决定要传达的内容。

<!-- Media -->

<!-- figureText: ${q}_{1}$ ${q}_{1}$ ${q}_{1} = \frac{{t}_{1}}{{t}_{2}}\left( {f\left( {t}_{2}\right)  + \Delta  - {q}_{0}}\right)$ $= \frac{{t}_{1}}{{t}_{2}}\left( {f\left( {t}_{2}\right)  - \Delta  - {q}_{0}}\right)$ $f\left( 0\right)  + \Delta$ ${t}_{1}$ ${t}_{2}$ ${t}_{3}$ (c) $P$ ${q}_{1} = f\left( {t}_{1}\right)  + \Delta$ $g\left( {t}_{1}\right)  \bullet$ ${q}_{1} = f\left( {t}_{1}\right)  - \Delta$ $f\left( 0\right)  - \Delta$ $f\left( 0\right)  + \Delta$ ${q}_{0}$ $f\left( 0\right)  - \Delta$ (a) (b) -->

<img src="https://cdn.noedgeai.com/0195c904-cf12-7bd7-9ed3-4fa404f6957a_13.jpg?x=379&y=342&w=1025&h=296&r=0"/>

Fig. 3. (a, b) Cutting in the parametric space. (c) Considering a small set of lines is enough.

图3. （a，b）在参数空间中进行切割。（c）考虑一小部分直线就足够了。

<!-- Media -->

Our algorithm with line predictions still follows the general framework outlined in Section 2. At the beginning of each round (assuming ${t}_{\text{start }} = 0$ ),we just send $f\left( 0\right)$ to Bob,and predict $f$ to be $f\left( 0\right)$ . Let ${t}_{1}$ be the time of the first triggering. We parameterize the lines by ${q}_{0},{q}_{1}$ ,meaning that the line $\left( {{q}_{0},{q}_{1}}\right)$ passes through $\left( {0,{q}_{0}}\right)$ and $\left( {{t}_{1},{q}_{1}}\right)$ . We call the $\left( {{q}_{0},{q}_{1}}\right)$ -space the parametric space,thus any line sent out by the algorithm is a point in the parametric space. Let $P$ be the region in the parametric space consisting of all the points that are valid $\Delta$ -approximations of $f\left( 0\right)$ and $f\left( {t}_{1}\right)$ ,which is a square (Figure 3(a)). We will pick a point $g\left( {t}_{1}\right)$ in $P$ and send it to Bob. Suppose at time ${t}_{2},g\left( {t}_{1}\right)$ fails to approximate $f\left( {t}_{2}\right)$ . Let $Q$ be the region in the parametric space consisting of all the valid $\Delta$ -approximations of $f\left( 0\right)$ and $f\left( {t}_{2}\right)$ ,which can be shown to be a parallelogram (Figure 3(b)). We update $P \leftarrow  P \cap  Q$ ,and then iterate the procedure. It is easy to see that if ${\mathcal{A}}_{\text{OPT }}$ does not need any further communication in the current round,its last message must lie inside $P$ .

我们带有直线预测的算法仍然遵循第2节中概述的一般框架。在每一轮开始时（假设${t}_{\text{start }} = 0$），我们只需将$f\left( 0\right)$发送给鲍勃（Bob），并预测$f$为$f\left( 0\right)$。设${t}_{1}$为首次触发的时间。我们用${q}_{0},{q}_{1}$对直线进行参数化，这意味着直线$\left( {{q}_{0},{q}_{1}}\right)$经过$\left( {0,{q}_{0}}\right)$和$\left( {{t}_{1},{q}_{1}}\right)$。我们将$\left( {{q}_{0},{q}_{1}}\right)$空间称为参数空间，因此算法发出的任何直线都是参数空间中的一个点。设$P$为参数空间中由所有对$f\left( 0\right)$和$f\left( {t}_{1}\right)$进行有效$\Delta$近似的点组成的区域，它是一个正方形（图3（a））。我们将在$P$中选取一个点$g\left( {t}_{1}\right)$并将其发送给鲍勃。假设在时间${t}_{2},g\left( {t}_{1}\right)$时，无法对$f\left( {t}_{2}\right)$进行近似。设$Q$为参数空间中由所有对$f\left( 0\right)$和$f\left( {t}_{2}\right)$进行有效$\Delta$近似的点组成的区域，可以证明它是一个平行四边形（图3（b））。我们更新$P \leftarrow  P \cap  Q$，然后重复该过程。很容易看出，如果${\mathcal{A}}_{\text{OPT }}$在当前轮次中不需要进一步的通信，那么它的最后一条消息必定位于$P$内。

The major task is to choose the initial set $S = {S}_{0}$ at the beginning of the round. After that, the algorithm is similar to that in Section 3.1, that is, at every triggering we update $S \leftarrow  S \cap  P$ and send the Tukey median of $S$ . The analysis also follows the same line. Let $M = \{ \left( {t,y}\right)  \mid  t \in  \left\lbrack  T\right\rbrack  ,y \in  \{ \mathbb{Z} + \Delta \}  \cup  \{ \mathbb{Z} - \Delta \} \}$ ,where $\{ \mathbb{Z} + \Delta \}$ denotes the set $\{ x \mid  x = y + \Delta ,y \in  \mathbb{Z}\}$ ,and similarly $\{ \mathbb{Z} - \Delta \}$ . Let $\mathcal{L}$ be the collection of lines passing through two points in $M$ . Let $X$ be the collection of intersection points between line $t = 0$ and lines in $\mathcal{L}$ ,and $Y$ be the collection of intersection points between line $t = {t}_{1}$ and lines in $\mathcal{L}$ . We choose ${S}_{0}$ to be $\left\{  {\left( {{q}_{0},{q}_{1}}\right)  \mid  {q}_{0} \in  X,{q}_{1} \in  Y}\right\}   \cap  P(P$ is the first square we get).

主要任务是在每一轮开始时选择初始集合$S = {S}_{0}$。此后，该算法与3.1节中的算法类似，即每次触发时，我们更新$S \leftarrow  S \cap  P$并发送$S$的Tukey中位数（图基中位数）。分析过程也遵循相同的思路。设$M = \{ \left( {t,y}\right)  \mid  t \in  \left\lbrack  T\right\rbrack  ,y \in  \{ \mathbb{Z} + \Delta \}  \cup  \{ \mathbb{Z} - \Delta \} \}$，其中$\{ \mathbb{Z} + \Delta \}$表示集合$\{ x \mid  x = y + \Delta ,y \in  \mathbb{Z}\}$，同理$\{ \mathbb{Z} - \Delta \}$。设$\mathcal{L}$为经过$M$中两点的直线集合。设$X$为直线$t = 0$与$\mathcal{L}$中直线的交点集合，$Y$为直线$t = {t}_{1}$与$\mathcal{L}$中直线的交点集合。我们选择${S}_{0}$为$\left\{  {\left( {{q}_{0},{q}_{1}}\right)  \mid  {q}_{0} \in  X,{q}_{1} \in  Y}\right\}   \cap  P(P$（这是我们得到的第一个正方形）。

We argue that considering only the points (lines) in ${S}_{0}$ is sufficient for our purpose. In particular,we can show that if ${\mathcal{A}}_{\mathrm{{OPT}}}$ keeps silent in the current round,there must be some surviving point (line) in ${S}_{0}$ . Consider the original function space (Figure 3(c)). Let $l$ be the line chosen by ${\mathcal{A}}_{\mathrm{{OPT}}}$ in its last communication. Suppose that ${\mathcal{A}}_{\mathrm{{OPT}}}$ has not made any communication in the current round, $l$ must intersect with all the line segments $\left( {\left( {t,f\left( t\right)  - \Delta }\right) ,\left( {t,f\left( t\right)  + \Delta }\right) }\right)$ ,for ${t}_{\text{start }} \leq  t \leq  {t}_{\text{now }}$ . We can always rotate and translate $l$ so that it passes through two points in $M$ ,and it still intersects with all line segments (line ${l}^{\prime }$ in Figure 3(c)). Therefore, ${l}^{\prime }$ must still survive at the current time.

我们认为，仅考虑${S}_{0}$中的点（直线）就足以满足我们的目的。特别地，我们可以证明，如果${\mathcal{A}}_{\mathrm{{OPT}}}$在当前轮次中保持沉默，那么${S}_{0}$中必定存在某个存活点（直线）。考虑原始函数空间（图3(c)）。设$l$为${\mathcal{A}}_{\mathrm{{OPT}}}$在其最后一次通信中选择的直线。假设${\mathcal{A}}_{\mathrm{{OPT}}}$在当前轮次中未进行任何通信，那么对于${t}_{\text{start }} \leq  t \leq  {t}_{\text{now }}$，$l$必定与所有线段$\left( {\left( {t,f\left( t\right)  - \Delta }\right) ,\left( {t,f\left( t\right)  + \Delta }\right) }\right)$相交。我们始终可以对$l$进行旋转和平移，使其经过$M$中的两点，并且它仍然与所有线段相交（图3(c)中的直线${l}^{\prime }$）。因此，${l}^{\prime }$在当前时刻必定仍然存活。

Finally we bound the cardinality of ${S}_{0}$ .

最后，我们对${S}_{0}$的基数进行界定。

LEMMA 9. $\left| {S}_{0}\right|  = O\left( {{\Delta }^{2}{T}^{6}}\right)$ ,where $T$ is the length of the tracking period.

引理9. $\left| {S}_{0}\right|  = O\left( {{\Delta }^{2}{T}^{6}}\right)$，其中$T$是跟踪周期的长度。

Proof. If a line $\left( {{q}_{0},{q}_{1}}\right)$ passes two points $\left( {{t}_{i},f\left( {t}_{i}\right)  \pm  \Delta }\right) ,\left( {{t}_{j},f\left( {t}_{j}\right)  \pm  \Delta }\right) \left( {{t}_{i},{t}_{j} \in  {\mathbb{Z}}^{ + },0 \leq  }\right.$ $\left. {{t}_{i} < {t}_{j} \leq  T}\right)$ in $M$ ,then

证明。如果一条直线$\left( {{q}_{0},{q}_{1}}\right)$经过$M$中的两点$\left( {{t}_{i},f\left( {t}_{i}\right)  \pm  \Delta }\right) ,\left( {{t}_{j},f\left( {t}_{j}\right)  \pm  \Delta }\right) \left( {{t}_{i},{t}_{j} \in  {\mathbb{Z}}^{ + },0 \leq  }\right.$ $\left. {{t}_{i} < {t}_{j} \leq  T}\right)$，那么

$$
{q}_{0} = \left( {f\left( {t}_{i}\right)  \pm  \Delta }\right)  - \frac{\left( {f\left( {t}_{i}\right)  \pm  \Delta }\right)  - \left( {f\left( {t}_{j}\right)  \pm  \Delta }\right) }{{t}_{i} - {t}_{j}}{t}_{i}, \tag{2}
$$

$$
{q}_{1} = \left( {f\left( {t}_{i}\right)  \pm  \Delta }\right)  - \frac{\left( {f\left( {t}_{i}\right)  \pm  \Delta }\right)  - \left( {f\left( {t}_{j}\right)  \pm  \Delta }\right) }{{t}_{i} - {t}_{j}}\left( {{t}_{i} - {t}_{1}}\right) . \tag{3}
$$

The number of possible choices of ${q}_{0}$ is $O\left( {\Delta {T}^{3}}\right)$ ,and so is that for ${q}_{1}$ . Thus cardinality of ${S}_{0}$ is at most $O\left( {{\Delta }^{2}{T}^{6}}\right)$ .

${q}_{0}$的可能选择数量为$O\left( {\Delta {T}^{3}}\right)$，${q}_{1}$的可能选择数量也是如此。因此，${S}_{0}$的基数至多为$O\left( {{\Delta }^{2}{T}^{6}}\right)$。

Therefore, $S$ will become empty after at most $O\left( {\log \left( {\Delta T}\right) }\right)$ iterations.

因此，$S$ 最多经过 $O\left( {\log \left( {\Delta T}\right) }\right)$ 次迭代后将变为空集。

THEOREM 7. There is an $O\left( {\log \left( {\Delta T}\right) }\right)$ -competitive online algorithm to track any function $f : {\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$ with line predictions,where $T$ is the length of the tracking period.

定理 7. 存在一种 $O\left( {\log \left( {\Delta T}\right) }\right)$ -竞争的在线算法，用于通过线性预测跟踪任何函数 $f : {\mathbb{Z}}^{ + } \rightarrow  \mathbb{Z}$，其中 $T$ 是跟踪周期的长度。

The preceding algorithm assumes that $T$ is given in advance in order to initialize ${S}_{0}$ . If $T$ is not known,we can use the following squaring trick to keep the competitive ratio. We start with $T$ being set to $\Delta$ . Whenever ${t}_{\text{now }}$ reaches $T$ and the current round has not yet finished,we restart the round with $T \leftarrow  \Delta {T}^{2}$ . It can be easily shown that the number of iterations in a round is still at most $O\left( {\log \left( {\Delta T}\right) }\right)$ .

上述算法假设预先给定 $T$ 以便初始化 ${S}_{0}$。如果 $T$ 未知，我们可以使用以下平方技巧来保持竞争比。我们从将 $T$ 设置为 $\Delta$ 开始。每当 ${t}_{\text{now }}$ 达到 $T$ 且当前轮次尚未结束时，我们以 $T \leftarrow  \Delta {T}^{2}$ 重新开始该轮次。可以很容易地证明，一轮中的迭代次数仍然最多为 $O\left( {\log \left( {\Delta T}\right) }\right)$。

## 5. OPEN PROBLEMS

## 5. 开放性问题

As mentioned in the related work, the problem studied in this article is a special case of the distributed tracking framework where there is only one site. It would be nice to generalize our techniques to multiple sites. Second,in the $d$ -dimensional case,if we consider the number of bits (instead of number of messages) the algorithms have sent, the competitive ratios of our current algorithms will increase by roughly a factor of $d$ . Thus we want to ask whether we can do better by a subset of the coordinates instead of a whole vector in ${\mathbb{R}}^{d}$ . Finally,it is also interesting to consider online tracking problems in other metric spaces.

如相关工作中所述，本文研究的问题是分布式跟踪框架中只有一个站点的特殊情况。将我们的技术推广到多个站点会很不错。其次，在 $d$ 维情况下，如果我们考虑算法发送的比特数（而不是消息数），我们当前算法的竞争比将大致增加 $d$ 倍。因此，我们想问是否可以通过 ${\mathbb{R}}^{d}$ 中的坐标子集而不是整个向量来做得更好。最后，考虑其他度量空间中的在线跟踪问题也很有趣。

## ACKNOWLEDGMENTS

## 致谢

We would like to thank Siu-Wing Cheng, Mordecai Golin, Jiongxin Jin, and Yajun Wang for fruitful discussions on various aspects of this problem.

我们要感谢郑绍远（Siu - Wing Cheng）、莫迪凯·戈林（Mordecai Golin）、金炯新（Jiongxin Jin）和王亚俊（Yajun Wang）就该问题的各个方面进行了富有成效的讨论。

## REFERENCES

## 参考文献

ALON, N., MATIAS, Y., AND SZEGEDY, M. 1999. The space complexity of approximating the frequency moments. J. Comput. Syst. Sci. 58, 137-147.

BERESFORD, A. AND STAJANO, F. 2003. Location privacy in pervasive computing. IEEE Pervasive Comput. 2, 1. BERTSIMAS, D. AND VEMPALA, S. 2004. Solving convex programs by random walks. J. ACM 51, 4, 540-556.

Chanoramouli, B., Phillips, J. M., AND Yang, J. 2007. Value-based notification conditions in large-scale publish/subscribe systems. In Proceedings of the International Conference on Very Large Databases.

Clarkson, K. L., Eppstein, D., Miller, G. L., Sturtivant, C., AND Teng, S. 1993. Approximating center points with iterated radon points. In Proceedings of the Annual Symposium on Computational Geometry.

Cormode, G. AND GAROFALAKIS, M. 2005. Sketching streams through the net: Distributed approximate query tracking. In Proceedings of the International Conference on Very Large Databases.

Cormode, G., Garofalakis, M., Mutthurkshnan, S., and Rastogr, R. 2005. Holistic aggregates in a networked world: Distributed tracking of approximate quantiles. In Proceedings of the ACM SIGMOD International Conference on Management of Data.

Cormode, G., MutthukRISHNAN, S., AND YI, K. 2008. Algorithms for distributed functional monitoring. In Proceedings of the ACM-SIAM Symposium on Discrete Algorithms.

Davis, S., Edmonds, J., AND IMPAGLIAZZ01, R. 2006. Online algorithms to minimize resource reallocations and network communication. In Proceedings of the International Workshop on Approximation Algorithms for Combinatorial Optimization (APPROX).

Desнрапре, A., Guestrın, C., Madden, S. R., Hellersтеɪɪ, J. M., and Hong, W. 2004. Model-driven data acquisition in sensor networks. In Proceedings of the International Conference on Very Large Databases.

DIAO, Y., RIZVI, S., AND FRANKLIN, M. J. 2004. Towards an Internet-scale XML dissemination service. In Proceedings of the International Conference on Very Large Databases.

GRUNBAUM, B. 1960. Partitions of mass-distributions and of convex bodies by hyperplanes. Pacific J. Math. 10,4.

KERALAPURA, R., CORMODE, G., AND RAMAMIRTHAM, J. 2006. Communication-efficient distributed monitoring of thresholded counts. In Proceedings of ACM SIGMOD International Conference on Management of Data.

Korre, B. AND VYGEN, J. 2007. Combinatorial Optimization: Theory and Algorithms, 4th Ed. Springer-Verlag. Madden, S., Franklin, M., Hellerstein, J., and Hong, W. 2005. TinyDB: an acquisitional query processing system for sensor networks. ACM Trans. Datab. Syst. 30, 1, 122-173.

Matoussek, J. 2002. Lectures on Discrete Geometry. Springer-Verlag, New York.

OLSTON, C., Loo, B. T., AND WIDOM, J. 2001. Adaptive precision setting for cached approximate values. In Proceedings of the ACM SIGMOD International Conference on Management of Data.

O'Rourke, J. 1981. An on-line algorithm for fitting straight lines between data ranges. Comm. ACM 24, 9.

Pottte, G. and Kaiser, W. 2000. Wireless integrated network sensors. Comm. ACM 43, 5, 51-58.

RADEMACHER, L. A. 2007. Approximating the centroid is hard. In Proceedings of the Annual Symposium on Computational Geometry.

Schiller, J. H. AND VOISARD, A. 2004. Location-Based Services. Morgan Kaufmann Publishers.

Yao, A. C. 1979. Some complexity questions related to distributive computing. In Proceedings of the ACM Symposium on Theory of Computing.

Yao, Y. AND GEHRKE, J. 2003. Query processing for sensor networks. In Proceedings of the Conference on Innovative Data Systems Research.