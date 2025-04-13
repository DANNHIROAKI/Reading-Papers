# Indexing the Positions of Continuously Moving Objects

# 连续移动对象位置的索引

Simonas Saltenis† Christian S. Jensen† Scott T. Leutenegger‡ Mario A. Lopez‡ † Department of Computer Science, Aalborg University, Denmark

西蒙纳斯·萨尔泰尼斯† 克里斯蒂安·S·延森† 斯科特·T·洛滕内格‡ 马里奥·A·洛佩斯‡ † 丹麦奥尔堡大学计算机科学系

$\ddagger$ Department of Mathematics and Computer Science,University of Denver,Colorado,USA

$\ddagger$ 美国科罗拉多州丹佛大学数学与计算机科学系

## Abstract

## 摘要

The coming years will witness dramatic advances in wireless communications as well as positioning technologies. As a result, tracking the changing positions of objects capable of continuous movement is becoming increasingly feasible and necessary. The present paper proposes a novel, ${\mathrm{R}}^{ * }$ -tree based indexing technique that supports the efficient querying of the current and projected future positions of such moving objects. The technique is capable of indexing objects moving in one-, two-, and three-dimensional space. Update algorithms enable the index to accommodate a dynamic data set, where objects may appear and disappear, and where changes occur in the anticipated positions of existing objects. A comprehensive performance study is reported.

未来几年，无线通信和定位技术将取得巨大进展。因此，跟踪能够连续移动的对象的位置变化变得越来越可行和必要。本文提出了一种基于 ${\mathrm{R}}^{ * }$ -树的新型索引技术，该技术支持对这类移动对象的当前和预计未来位置进行高效查询。该技术能够对在一维、二维和三维空间中移动的对象进行索引。更新算法使索引能够适应动态数据集，在该数据集中，对象可能出现和消失，并且现有对象的预期位置可能发生变化。本文还报告了一项全面的性能研究。

## 1 Introduction

## 1 引言

The rapid and continued advances in positioning systems, e.g., GPS, wireless communication technologies, and electronics in general promise to render it increasingly feasible to track and record the changing positions of objects capable of continuous movement.

定位系统（如全球定位系统）、无线通信技术以及电子技术的快速持续发展，使得跟踪和记录能够连续移动的对象的位置变化变得越来越可行。

In a recent interview with Danish newspaper Børsen, Michael Hawley from MIT's Media Lab described how he was online when he ran the Boston Marathon this year [19]. Prior to the race, he swallowed several capsules, which in conjunction with other hardware enabled the monitoring of his position, body temperature, and pulse during the race. This scenario demonstrates the potential for putting bodies, and, more generally, objects that move, online. Achieving this may enable a multitude of applications. It becomes possible to detect the signs of an impending medical emergency in a person early and warn the person or alert a medical service. It becomes possible to have equipment recognize its user; and the equipment may alert its owner in the case of unauthorized use or theft.

在最近接受丹麦报纸《商报》采访时，麻省理工学院媒体实验室的迈克尔·霍利描述了他今年参加波士顿马拉松比赛时如何保持在线状态 [19]。比赛前，他吞下了几颗胶囊，这些胶囊与其他硬件设备一起，使他在比赛过程中能够监测自己的位置、体温和脉搏。这个场景展示了将人体以及更广泛意义上的移动对象连接到网络的潜力。实现这一点可能会催生众多应用。例如，可以提前检测到一个人即将出现的医疗紧急情况的迹象，并向本人发出警告或通知医疗服务机构。设备可以识别其用户；如果设备被未经授权使用或被盗，它可以向所有者发出警报。

Industry leaders in the mobile phone market expect more than 500 million mobile phone users by year 2002 (compared to 300 million Internet users) and 1 billion by year 2004, and they expect mobile phones to evolve into wireless Internet terminals $\left\lbrack  {{14},{25}}\right\rbrack$ . Rendering such terminals location aware may substantially improve the quality of the services offered to them $\left\lbrack  {{12},{25}}\right\rbrack$ . In addition, the cost of providing location awareness is expected to be relatively low. These factors combine to promise the presence of substantial numbers of location aware, on-line objects capable of continuous movement.

手机市场的行业领袖预计，到 2002 年，手机用户将超过 5 亿（相比之下，互联网用户为 3 亿），到 2004 年将达到 10 亿，并且他们预计手机将发展成为无线互联网终端 $\left\lbrack  {{14},{25}}\right\rbrack$。使这些终端具备位置感知能力可能会显著提高为用户提供的服务质量 $\left\lbrack  {{12},{25}}\right\rbrack$。此外，提供位置感知功能的成本预计相对较低。这些因素共同预示着将出现大量具备位置感知能力、能够连续移动的在线对象。

Applications such as process monitoring do not depend on positioning technologies. In these, the position of a moving point object could for example be a pair of temperature and pressure values. Yet other applications include vehicle navigation, tracking, and monitoring, where the positions of air, sea, or land-based equipment such as airplanes, fishing boats and freighters, and cars and trucks are of interest. It is diverse applications such as these that warrant the study of the indexing of objects that move.

像过程监控这样的应用并不依赖于定位技术。在这些应用中，移动点对象的位置可以是例如温度和压力值的一对数值。其他应用还包括车辆导航、跟踪和监控，其中飞机、渔船和货船、汽车和卡车等空中、海上或陆地设备的位置是关注的重点。正是这类多样化的应用促使我们研究移动对象的索引问题。

Continuous movement poses new challenges to database technology. In conventional databases, data is assumed to remain constant unless it is explicitly modified. Capturing continuous movement with this assumption would entail either performing very frequent updates or recording outdated, inaccurate data, neither of which are attractive alternatives.

连续移动给数据库技术带来了新的挑战。在传统数据库中，除非数据被显式修改，否则假定数据保持不变。基于这种假设来捕捉连续移动，要么需要非常频繁地进行更新，要么会记录过时、不准确的数据，这两种选择都不太理想。

A different tack must be adopted. The continuous movement should be captured directly, so that the mere advance of time does not necessitate explicit updates [27]. Put differently, rather than storing simple positions, functions of time that express the objects' positions should be stored. Then updates are necessary only when the parameters of the functions change. We use one linear function per object, with the parameters of a function being the position and velocity vector of the object at the time the function is reported to the database.

必须采取不同的方法。应该直接捕捉连续移动，这样仅仅时间的推移就不需要进行显式更新 [27]。换句话说，应该存储表达对象位置的时间函数，而不是简单的位置。这样，只有当函数的参数发生变化时才需要进行更新。我们为每个对象使用一个线性函数，函数的参数是对象在函数被报告给数据库时的位置和速度向量。

Two different, although related, indexing problems must be solved in order to support applications involving continuous movement. One problem is the indexing of the current and anticipated future positions of moving objects. The other problem is the indexing of the histories, or trajectories, of the positions of moving objects. We focus on the former problem. One approach to solving the latter problem (while simultaneously solving the first) is to render the solution to the first problem partially persistent $\left\lbrack  {6,{15}}\right\rbrack$ .

为了支持涉及连续移动的应用，必须解决两个不同但相关的索引问题。一个问题是对移动对象的当前和预期未来位置进行索引。另一个问题是对移动对象位置的历史或轨迹进行索引。我们专注于前一个问题。解决后一个问题（同时解决前一个问题）的一种方法是使前一个问题的解决方案具有部分持久性 $\left\lbrack  {6,{15}}\right\rbrack$。

---

<!-- Footnote -->

Permission to make digital or hard copies of part or all of this work or personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. To copy otherwise, to republish, to post on servers, or to redistribute to lists, requires prior specific permission and/or a fee.

允许个人或课堂使用本作品的部分或全部内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且必须在首页上标明此声明和完整的引用信息。否则，如需复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。

MOD 2000, Dallas, TX USA

2000 年移动对象数据库研讨会，美国得克萨斯州达拉斯

© ACM 2000 1-58113-218-2/00/05 . . .\$5.00

© 美国计算机协会 2000 1 - 58113 - 218 - 2/00/05 ... 5 美元

<!-- Footnote -->

---

We propose an indexing technique, the time-parameterized R-tree (the TPR-tree, for short), which efficiently indexes the current and anticipated future positions of moving point objects (or "moving points," for short). The technique naturally extends the ${\mathrm{R}}^{ * }$ -tree [5].

我们提出了一种索引技术，即时间参数化R树（简称TPR树），它能有效地对移动点对象（简称“移动点”）的当前位置和预期未来位置进行索引。该技术自然地扩展了${\mathrm{R}}^{ * }$树[5]。

Several distinctions may be made among the possible approaches to the indexing of the future linear trajectories of moving points. First, approaches may differ according to the space that they index. Assuming the objects move in $d$ -dimensional space $\left( {d = 1,2,3}\right)$ ,their future trajectories may be indexed as lines in $\left( {d + 1}\right)$ -dimensional space [26]. As an alternative, one may map the trajectories to points in a higher-dimensional space which are then indexed [13]. Queries must subsequently also be transformed to counter the data transformation. Yet another alternative is to index data in its native, $d$ -dimensional space,which is possible by parameterizing the index structure using velocity vectors and thus enabling the index to be "viewed" as of any future time. The TPR-tree adopts this latter alternative. This absence of transformations yields a quite intuitive indexing technique.

在对移动点的未来线性轨迹进行索引的可能方法中，可以做出几种区分。首先，不同方法所索引的空间可能不同。假设对象在$d$维空间$\left( {d = 1,2,3}\right)$中移动，它们的未来轨迹可以作为$\left( {d + 1}\right)$维空间中的线进行索引[26]。另一种选择是，将轨迹映射到更高维空间中的点，然后对这些点进行索引[13]。随后，查询也必须进行相应转换以应对数据转换。还有一种选择是在其原生的$d$维空间中对数据进行索引，这可以通过使用速度向量对索引结构进行参数化来实现，从而使索引能够在任何未来时间“查看”。TPR树采用了后一种选择。这种无需转换的方式产生了一种非常直观的索引技术。

A second distinction is whether the index partitions the data (e.g., as do R-trees) or the embedding space (e.g., as do Quadtrees). When indexing the data in its native space, an index based on data partitioning seems to be more suitable. On the other hand, if trajectories are indexed as lines in $\left( {d + 1}\right)$ -dimensional space,a data partitioning access method that does not employ clipping may introduce substantial overlap.

第二个区别是索引是对数据进行分区（例如R树）还是对嵌入空间进行分区（例如四叉树）。当在原生空间中对数据进行索引时，基于数据分区的索引似乎更合适。另一方面，如果将轨迹作为$\left( {d + 1}\right)$维空间中的线进行索引，不采用裁剪的基于数据分区的访问方法可能会引入大量重叠。

Third, indices may differ in the degrees of data replication they entail. Replication may improve query performance, but may also adversely affect update performance. The TPR-tree does not employ replication.

第三，索引在数据复制程度上可能有所不同。复制可以提高查询性能，但也可能对更新性能产生不利影响。TPR树不采用数据复制。

Fourth, we may distinguish approaches according to whether or not they require periodic index rebuilding. Some approaches (e.g., [26]) employ individual indices that are only functional for a certain time period. In these approaches, a new index must be provided before its predecessor is no longer functional. Other approaches may employ an index that in principle remains functional indefinitely [13], but which may be optimized for some specific time horizon and perhaps deteriorates as time progresses. The TPR-tree belongs to this latter category.

第四，我们可以根据方法是否需要定期重建索引来进行区分。一些方法（例如[26]）使用仅在特定时间段内有效的单个索引。在这些方法中，必须在旧索引失效之前提供新的索引。其他方法可能使用原则上可以无限期保持有效的索引[13]，但该索引可能针对某个特定的时间范围进行了优化，并且可能会随着时间的推移而性能下降。TPR树属于后一类。

In the TPR-tree, the bounding rectangles in the tree are functions of time, as are the moving points being indexed. Intuitively, the bounding rectangles are capable of continuously following the enclosed data points or other rectangles as these move. Like the R-trees, the new index is capable of indexing points in one-, two-, and three-dimensional space. In addition, the principles at play in the new index are extendible to non-point objects.

在TPR树中，树中的边界矩形是时间的函数，被索引的移动点也是如此。直观地说，边界矩形能够随着所包含的数据点或其他矩形的移动而持续跟踪它们。与R树一样，新的索引能够对一维、二维和三维空间中的点进行索引。此外，新索引所遵循的原则可以扩展到非点对象。

The next section presents the problem being addressed, by describing the data to be indexed, the queries to be supported, and problem parameters. In addition, related research is covered. Section 3 describes the tree structure and algorithms. It is assumed that the reader has some familiarity with the ${\mathrm{R}}^{ * }$ -tree. To ease the exposition,one-dimensional data is generally assumed,and the general $n$ -dimensional case is only considered when the inclusion of additional dimensions introduces new issues. Section 4 reports on performance experiments, and Section 5 summarizes and offers research directions.

下一节通过描述要索引的数据、要支持的查询以及问题参数来阐述所解决的问题。此外，还会涵盖相关研究。第3节描述树结构和算法。假设读者对${\mathrm{R}}^{ * }$树有一定的了解。为了便于阐述，通常假设数据是一维的，只有当引入额外维度会带来新问题时，才会考虑一般的$n$维情况。第4节报告性能实验结果，第5节进行总结并提出研究方向。

## 2 Problem Statement and Related Work

## 2 问题陈述与相关工作

We describe the data being indexed, the queries being supported, the problem parameters, and related work in turn.

我们依次描述要索引的数据、要支持的查询、问题参数以及相关工作。

### 2.1 Problem Setting

### 2.1 问题设定

An object’s position at some time $t$ is given by $\bar{x}\left( t\right)  =$ $\left( {{x}_{1}\left( t\right) ,{x}_{2}\left( t\right) ,\ldots ,{x}_{d}\left( t\right) }\right)$ ,where it is assumed that the times $t$ are not before the current time. This position is modeled as a linear function of time, which is specified by two parameters. The first is a position for the object at some specified time ${t}_{ref},\bar{x}\left( {t}_{ref}\right)$ ,which we term the reference position. The second parameter is a velocity vector for the object, $\bar{v} = \left( {{v}_{1},{v}_{2},\ldots ,{v}_{d}}\right)$ . Thus, $\bar{x}\left( t\right)  = \bar{x}\left( {t}_{\text{ref }}\right)  + \bar{v}(t -$ $\left. {t}_{ref}\right)$ . An object’s movement is observed at some time, ${t}_{obs}$ . The first parameter, $\bar{x}\left( {t}_{\text{ref }}\right)$ ,may be the object’s position at this time, or it may be the position that the object would have at some other, chosen reference time, given the velocity vector $\bar{v}$ observed at ${t}_{obs}$ and the position $\bar{x}\left( {t}_{obs}\right)$ observed at ${t}_{obs}$ .

某个物体在某个时刻$t$的位置由$\bar{x}\left( t\right)  =$ $\left( {{x}_{1}\left( t\right) ,{x}_{2}\left( t\right) ,\ldots ,{x}_{d}\left( t\right) }\right)$给出，其中假设时刻$t$不早于当前时间。该位置被建模为时间的线性函数，由两个参数指定。第一个参数是物体在某个指定时刻${t}_{ref},\bar{x}\left( {t}_{ref}\right)$的位置，我们称之为参考位置。第二个参数是物体的速度向量$\bar{v} = \left( {{v}_{1},{v}_{2},\ldots ,{v}_{d}}\right)$。因此，$\bar{x}\left( t\right)  = \bar{x}\left( {t}_{\text{ref }}\right)  + \bar{v}(t -$ $\left. {t}_{ref}\right)$。在某个时刻${t}_{obs}$观察到物体的运动。第一个参数$\bar{x}\left( {t}_{\text{ref }}\right)$可以是物体在此时刻的位置，也可以是在给定时刻${t}_{obs}$观察到的速度向量$\bar{v}$和时刻${t}_{obs}$观察到的位置$\bar{x}\left( {t}_{obs}\right)$的情况下，物体在其他选定参考时刻的位置。

Modeling the positions of moving objects as functions of time not only enables us to make tentative future predictions, but also solves the problem of the frequent updates that would otherwise be required to approximate continuous movement in a traditional setting. For example, objects may report their positions and velocity vectors when their actual positions deviate from what they have previously reported by some threshold. The choice of the update frequency then depends on the type of movement, the desired accuracy, and the technical limitations $\left\lbrack  {{28},{20},{17}}\right\rbrack$ .

将移动物体的位置建模为时间的函数，不仅使我们能够对未来进行初步预测，还解决了在传统环境中近似连续运动时原本需要频繁更新的问题。例如，当物体的实际位置与之前报告的位置偏差超过某个阈值时，物体可以报告其位置和速度向量。更新频率的选择则取决于运动类型、所需精度和技术限制$\left\lbrack  {{28},{20},{17}}\right\rbrack$。

As will be illustrated in the following and explained in Section 3, the reference position and the velocity are used not only when recording the future trajectories of moving points, but also for representing the coordinates of the bounding rectangles in the index as functions of time.

正如下面将说明并在第3节解释的那样，参考位置和速度不仅在记录移动点的未来轨迹时使用，还用于将索引中边界矩形的坐标表示为时间的函数。

As an example, consider Figure 1. The top left diagram shows the positions and velocity vectors of 7 point objects at time 0 .

例如，考虑图1。左上方的图显示了7个点对象在时刻0的位置和速度向量。

Assume we create an R-tree at time 0 . The top right diagram shows one possible assignment of the objects to minimum bounding rectangles (MBRs) assuming a maximum of three objects per node. Previous work has shown that attempting to minimize the quantities known as overlap, dead space, and perimeter leads to an index with good query performance $\left\lbrack  {{11},{18}}\right\rbrack$ ,and so the chosen assignment appears to be well chosen. However, although it is good for queries at the present time, the movement of the objects may adversely affect this assignment.

假设我们在时刻0创建一个R树。右上方的图显示了在每个节点最多包含三个对象的假设下，对象到最小边界矩形（MBR）的一种可能分配方式。先前的研究表明，尝试最小化重叠、空白空间和周长这些量会得到一个查询性能良好的索引$\left\lbrack  {{11},{18}}\right\rbrack$，因此所选择的分配方式似乎是合理的。然而，尽管这种分配方式在当前时间对查询有利，但对象的移动可能会对这种分配产生不利影响。

<!-- Media -->

<!-- figureText: $\left| \begin{matrix} 1 & {2}^{ * } \\   & 3 \end{matrix}\right|$ -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_2.jpg?x=148&y=138&w=647&h=649&r=0"/>

Figure 1: Moving Points and Resulting Leaf-Level MBRs

图1：移动点和生成的叶级最小边界矩形

<!-- Media -->

The bottom left diagram shows the locations of the objects and the MBRs at time 3, assuming that MBRs grow to stay valid. The grown MBRs adversely affect query performance; and as time increases, the MBRs will continue to grow, leading to further deterioration. Even though the objects belonging to the same MBR (e.g., objects 4 and 5) were originally close, the different directions of their movement cause their positions to diverge rapidly and hence the MBRs to grow.

左下方的图显示了假设最小边界矩形（MBR）扩展以保持有效性时，时刻3物体的位置和MBR。扩展后的MBR对查询性能产生不利影响；并且随着时间的增加，MBR将继续扩展，导致性能进一步下降。即使属于同一个MBR的物体（例如，物体4和5）最初距离很近，但它们不同的运动方向会导致它们的位置迅速偏离，从而使MBR扩展。

From the perspective of queries at time 3 , it would have been better to assign objects to MBRs as illustrated by the bottom right diagram. Note that at time 0 , this assignment will yield worse query performance than the original assignment. Thus, the assignment of objects to MBRs must take into consideration when most queries will arrive.

从时刻3的查询角度来看，按照右下方的图所示将对象分配到MBR会更好。请注意，在时刻0，这种分配方式的查询性能会比最初的分配方式差。因此，将对象分配到MBR时必须考虑大多数查询将在何时到达。

The MBRs in this example illustrate the kind of time-parameterized bounding rectangles supported by the TPR-tree. The algorithms presented in Section 3, which are responsible for the assignment of objects to bounding rectangles and thus control the structure and quality of the index, attempt to take observations such as those illustrated by this example into consideration.

此示例中的MBR说明了TPR树所支持的那种时间参数化边界矩形。第3节中介绍的算法负责将对象分配到边界矩形，从而控制索引的结构和质量，这些算法会尝试考虑此示例所说明的这类观察结果。

### 2.2 Query Types

### 2.2 查询类型

The queries supported by the index retrieve all points with positions within specified regions. We distinguish between three kinds, based on the regions they specify. In the sequel, a $d$ -dimensional rectangle $R$ is specified by its $d$ projections $\left\lbrack  {{a}_{1}^{ \vdash  },{a}_{1}^{ \dashv  }}\right\rbrack  ,\ldots \left\lbrack  {{a}_{d}^{ \vdash  },{a}_{d}^{ \dashv  }}\right\rbrack  ,{a}_{j}^{ \vdash  } \leq  {a}_{j}^{ \dashv  }$ ,into the $d$ coordinate axes.

该索引支持的查询可检索出位置位于指定区域内的所有点。根据指定的区域，我们将其分为三种类型。接下来，一个$d$维矩形$R$由其在$d$个坐标轴上的投影$\left\lbrack  {{a}_{1}^{ \vdash  },{a}_{1}^{ \dashv  }}\right\rbrack  ,\ldots \left\lbrack  {{a}_{d}^{ \vdash  },{a}_{d}^{ \dashv  }}\right\rbrack  ,{a}_{j}^{ \vdash  } \leq  {a}_{j}^{ \dashv  }$来确定。

Let $R,{R}_{1}$ ,and ${R}_{2}$ be three $d$ -dimensional rectangles and $t$ , ${t}^{ \vdash  } < {t}^{ \dashv  }$ ,three time values that are not less than the current time.

设$R,{R}_{1}$、${R}_{2}$为三个$d$维矩形，$t$、${t}^{ \vdash  } < {t}^{ \dashv  }$为三个不小于当前时间的时间值。

Type 1 timeslice query: $Q = \left( {R,t}\right)$ specifies a hyper-rectangle $R$ located at time point $t$ .

类型1时间片查询：$Q = \left( {R,t}\right)$指定了一个位于时间点$t$的超矩形$R$。

Type 2 window query: $Q = \left( {R,{t}^{ \vdash  },{t}^{ \dashv  }}\right)$ specifies a hyper-rectangle $R$ that covers the interval $\left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack$ . Stated differently, this query retrieves points with trajectories in $\left( {\bar{x},t}\right)$ -space crossing the $\left( {d + 1}\right)$ -dimensional hyper-rectangle $\left( {\left\lbrack  {{a}_{1}^{ \vdash  },{a}_{1}^{ \dashv  }}\right\rbrack  ,\left\lbrack  {{a}_{2}^{ \vdash  },{a}_{2}^{ \dashv  }}\right\rbrack  ,\ldots ,\left\lbrack  {{a}_{d}^{ \vdash  },{a}_{d}^{ \dashv  }}\right\rbrack  ,\left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack  }\right)$ .

类型2窗口查询：$Q = \left( {R,{t}^{ \vdash  },{t}^{ \dashv  }}\right)$指定了一个覆盖区间$\left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack$的超矩形$R$。换句话说，此查询检索的是在$\left( {\bar{x},t}\right)$空间中轨迹穿过$\left( {d + 1}\right)$维超矩形$\left( {\left\lbrack  {{a}_{1}^{ \vdash  },{a}_{1}^{ \dashv  }}\right\rbrack  ,\left\lbrack  {{a}_{2}^{ \vdash  },{a}_{2}^{ \dashv  }}\right\rbrack  ,\ldots ,\left\lbrack  {{a}_{d}^{ \vdash  },{a}_{d}^{ \dashv  }}\right\rbrack  ,\left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack  }\right)$的点。

Type 3 moving query: $Q = \left( {{R}_{1},{R}_{2},{t}^{ \vdash  },{t}^{ \dashv  }}\right)$ specifies the $\left( {d + 1}\right)$ -dimensional trapezoid obtained by connecting ${R}_{1}$ at time ${t}^{ \vdash  }$ to ${R}_{2}$ at time ${t}^{ \dashv  }$ .

类型3移动查询：$Q = \left( {{R}_{1},{R}_{2},{t}^{ \vdash  },{t}^{ \dashv  }}\right)$指定了通过连接时间${t}^{ \vdash  }$时的${R}_{1}$和时间${t}^{ \dashv  }$时的${R}_{2}$所得到的$\left( {d + 1}\right)$维梯形。

The second type of query generalizes the first, and is itself a special case of the third type. To illustrate the query types, consider the one-dimensional data set in Figure 2, which represents temperatures measured at different locations. Here,queries ${Q0}$ and ${Q1}$ are timeslice queries, ${Q2}$ is a window query,and ${Q3}$ is a moving query.

第二种类型的查询是第一种的推广，而它本身又是第三种类型的一个特例。为了说明这些查询类型，考虑图2中的一维数据集，它表示在不同位置测量的温度。这里，查询${Q0}$和${Q1}$是时间片查询，${Q2}$是窗口查询，${Q3}$是移动查询。

<!-- Media -->

<!-- figureText: value 01 Q2 Q3 3 4 time 40 o2 30 Q1 10 Q0 0 o1 -10 -20 -30 o3 -40 1 2 -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_2.jpg?x=874&y=853&w=669&h=535&r=0"/>

Figure 2: Query Examples for One-Dimensional Data

图2：一维数据的查询示例

<!-- Media -->

Let $\operatorname{iss}\left( Q\right)$ denote the time when a query $Q$ is issued. The two parameters, reference position and velocity vector, of an object as seen by a query $Q$ depend on $\operatorname{iss}\left( Q\right)$ ,because objects update their parameters as time goes. Consider object ${o1}$ : its movement is described by one trajectory for queries with $\operatorname{iss}\left( Q\right)  < 1$ ,another trajectory for queries with $1 \leq  \operatorname{iss}\left( Q\right)  < 3$ ,and a third trajectory for queries with $3 \leq  \operatorname{iss}\left( Q\right)$ . For example,the answer to query ${Q1}$ is ${o1}$ , if $\operatorname{iss}\left( {Q1}\right)  < 1$ ,and no object qualifies for this query if iss $\left( {Q1}\right)  \geq  1$ .

令$\operatorname{iss}\left( Q\right)$表示发出查询$Q$的时间。查询$Q$所观察到的对象的两个参数，即参考位置和速度向量，取决于$\operatorname{iss}\left( Q\right)$，因为对象会随着时间更新其参数。考虑对象${o1}$：对于$\operatorname{iss}\left( Q\right)  < 1$的查询，其运动由一条轨迹描述；对于$1 \leq  \operatorname{iss}\left( Q\right)  < 3$的查询，由另一条轨迹描述；对于$3 \leq  \operatorname{iss}\left( Q\right)$的查询，由第三条轨迹描述。例如，如果$\operatorname{iss}\left( {Q1}\right)  < 1$，查询${Q1}$的答案是${o1}$；如果$\left( {Q1}\right)  \geq  1$，则没有对象符合该查询条件。

This example illustrates that queries far in the future are likely to be of little value, because the positions as predicted at query time become less and less accurate as queries move into the future, and because updates not known at query time may occur. Therefore, real-world applications may be expected to issue queries that are concentrated in some limited time window extending from the current time.

这个例子说明，远期的查询可能价值不大，因为随着查询时间向未来推移，在查询时预测的位置会变得越来越不准确，而且在查询时未知的更新可能会发生。因此，可以预期现实世界的应用程序会发出集中在从当前时间开始的某个有限时间窗口内的查询。

### 2.3 Problem Parameters

### 2.3 问题参数

The values of three problem parameters affect the indexing problem and the qualities of a TPR-tree. Figure 3 illustrates these parameters, which will be used throughout the paper.

三个问题参数的值会影响索引问题和TPR树的性能。图3展示了这些参数，本文将始终使用这些参数。

- Querying window (W): how far queries can "look" into the future. Thus, $\operatorname{iss}\left( Q\right)  \leq  t \leq  \operatorname{iss}\left( Q\right)  + \mathrm{W}$ ,for Type 1 queries,and $\operatorname{iss}\left( Q\right)  \leq  {t}^{ \vdash  } \leq  {t}^{ \dashv  } \leq  \operatorname{iss}\left( Q\right)  + \mathrm{W}$ for queries of Types 2 and 3.

- 查询窗口（W）：查询能够“展望”未来的时间范围。因此，对于1类查询为$\operatorname{iss}\left( Q\right)  \leq  t \leq  \operatorname{iss}\left( Q\right)  + \mathrm{W}$，对于2类和3类查询为$\operatorname{iss}\left( Q\right)  \leq  {t}^{ \vdash  } \leq  {t}^{ \dashv  } \leq  \operatorname{iss}\left( Q\right)  + \mathrm{W}$。

- Index usage time (U): the time interval during which an index will be used for querying. Thus, ${t}_{l} \leq  {iss}\left( Q\right)  \leq$ ${t}_{l} + \mathrm{U}$ ,where ${t}_{l}$ is the time when index is created/loaded.

- 索引使用时间（U）：索引用于查询的时间间隔。因此，${t}_{l} \leq  {iss}\left( Q\right)  \leq$ ${t}_{l} + \mathrm{U}$，其中${t}_{l}$是索引创建/加载的时间。

- Time horizon(H): the length of the time interval from which the times $t,{t}^{ \vdash  }$ ,and ${t}^{ \dashv  }$ specified in queries are drawn. The time horizon for an index is the index usage time plus the querying window.

- 时间范围（H）：查询中指定的时间$t,{t}^{ \vdash  }$和${t}^{ \dashv  }$所取自的时间间隔的长度。索引的时间范围是索引使用时间加上查询窗口。

<!-- Media -->

<!-- figureText: $\mathrm{H} = \mathrm{U} + \mathrm{W}$ W ${t}^{-1}$ iss(Q) U -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_3.jpg?x=159&y=872&w=640&h=192&r=0"/>

Figure 3: Time Horizon H, Index Usage Time U, and Querying Window W

图3：时间范围H、索引使用时间U和查询窗口W

<!-- Media -->

Thus, a newly created index must support queries that reach $H$ time units into the future. While the utility of parameter $\mathrm{U}$ (and $\mathrm{H}$ ) is more clearcut for static data sets and bulkloading, we shall see in Section 3.4 that this parameter is also useful in a dynamic setting where index updates are allowed. Specifically, although a TPR-tree is functional at all times after its creation, using different values for parameter $\mathrm{U}$ during insertions affects the search properties of the tree.

因此，新创建的索引必须支持能查询到未来$H$个时间单位的查询。虽然参数$\mathrm{U}$（和$\mathrm{H}$）对于静态数据集和批量加载的作用更为明确，但我们将在3.4节中看到，这个参数在允许索引更新的动态环境中也很有用。具体来说，虽然TPR树在创建后的所有时间都能正常工作，但在插入操作中使用不同的参数$\mathrm{U}$值会影响树的搜索性能。

### 2.4 Previous Work

### 2.4 相关工作

Related work on the indexing of the current and future positions of moving objects has concentrated mostly on points moving in one-dimensional space.

关于移动对象当前和未来位置索引的相关工作大多集中在一维空间中移动的点上。

Tayeb et al. [26] use PMR-Quadtrees [22] for indexing the future linear trajectories of one-dimensional moving point objects as line segments in(x,t)-space. The segments span the time interval that starts at the current time and extends $\mathrm{H}$ time units into the future. A tree expires after $\mathrm{U}$ time units, and a new tree must be made available for querying. This approach introduces substantial data replication in the index- a line segment is usually stored in several nodes.

Tayeb等人[26]使用PMR四叉树[22]将一维移动点对象的未来线性轨迹作为(x,t)空间中的线段进行索引。这些线段跨越从当前时间开始并延伸到未来$\mathrm{H}$个时间单位的时间间隔。一棵树在$\mathrm{U}$个时间单位后过期，必须提供一棵新树用于查询。这种方法在索引中引入了大量的数据复制——一条线段通常存储在多个节点中。

Kollios et al. [13] employ the dual data transformation where a line $x = x\left( {t}_{ref}\right)  + v\left( {t - {t}_{ref}}\right)$ is transformed to the point $\left( {x\left( {t}_{\text{ref }}\right) ,v}\right)$ ,enabling the use of regular spatial indices. It is argued that indices based on Kd-trees are well suited for this problem because these best accommodate the shapes of the (transformed) queries on the data. Kollios et al. suggest, but do not investigate in any detail, how this approach may be extended to two and higher dimensions. Kollios et al. also propose two other methods that achieve better query performance at the cost of data replication. These methods do not seem to apply to more than one dimension.

科利奥斯（Kollios）等人 [13] 采用了双重数据转换方法，即将直线 $x = x\left( {t}_{ref}\right)  + v\left( {t - {t}_{ref}}\right)$ 转换为点 $\left( {x\left( {t}_{\text{ref }}\right) ,v}\right)$，从而能够使用常规的空间索引。有人认为，基于 Kd 树的索引非常适合解决这个问题，因为这些索引能最好地适应数据上（转换后的）查询的形状。科利奥斯等人提出了如何将这种方法扩展到二维及更高维度，但并未进行详细研究。科利奥斯等人还提出了另外两种方法，这些方法以数据复制为代价实现了更好的查询性能。这些方法似乎不适用于超过一维的情况。

Next, Kollios et al. provide theoretical lower bounds for this indexing problem, assuming a static data set and $\mathrm{H} = \infty$ . Allowing the index to use linear space,the types of queries discussed in Section 2 can be answered in $O\left( {{n}^{\left( {{2d} - 1}\right) /{2d}} + k}\right)$ time. Here $d$ is the number of dimensions of the space where the objects move, $n$ is the number of data blocks,and $k$ is the size in blocks of a query answer. To achieve this bound, an external memory version of partition trees may be used [1]. It is argued that, although having good asymptotic performance bounds, partition trees are not practical due to the large constant factors involved.

接下来，科利奥斯等人针对这个索引问题给出了理论下界，假设数据集是静态的且满足 $\mathrm{H} = \infty$。如果允许索引使用线性空间，第 2 节中讨论的查询类型可以在 $O\left( {{n}^{\left( {{2d} - 1}\right) /{2d}} + k}\right)$ 时间内得到解答。这里 $d$ 是对象移动空间的维度数，$n$ 是数据块的数量，$k$ 是查询答案的块大小。为了达到这个界限，可以使用分区树的外部内存版本 [1]。有人认为，尽管分区树具有良好的渐近性能界限，但由于涉及较大的常数因子，因此并不实用。

Basch et al. [4] propose so-called kinetic main-memory data structures for mobile objects. The idea is to schedule future events that update a data structure so that necessary invariants hold. Agarwal et al. [2] apply these ideas to external range trees [3]. Their approach may possibly be applicable to R-trees or time-parameterized R-trees where events would fix MBRs, although it is unclear how to contend with future queries that arrive in non-chronological order. Agarwal et al. address non-chronological queries using partial persistence techniques and also show how to combine kinetic range trees with partition trees to achieve a trade-off between the number of kinetic events and query performance.

巴施（Basch）等人 [4] 为移动对象提出了所谓的动态主存数据结构。其思路是安排未来的事件来更新数据结构，以确保必要的不变量成立。阿加瓦尔（Agarwal）等人 [2] 将这些思想应用于外部范围树 [3]。他们的方法可能适用于 R 树或时间参数化的 R 树，在这些树中，事件可以修正最小边界矩形（MBR），尽管目前尚不清楚如何处理非按时间顺序到达的未来查询。阿加瓦尔等人使用部分持久化技术处理非按时间顺序的查询，并展示了如何将动态范围树与分区树相结合，以在动态事件数量和查询性能之间取得平衡。

The problem of indexing moving points is related to the problem of indexing now-relative temporal data. The GR-tree [7] is an R-tree based index for now-relative bitemporal data. Combined valid and transaction time intervals with end-times related to the continuously progressing current time result in regions that grow, albeit in a restricted way. The idea in this index is to accommodate growing data regions by introducing bounding regions that also grow. Specifically, bounding regions are time-parameterized, and their extents are computed each time a query is asked.

移动点的索引问题与相对于当前时间的时态数据的索引问题相关。GR 树 [7] 是一种基于 R 树的、用于相对于当前时间的双时态数据的索引。将有效时间和事务时间间隔与不断推进的当前时间相关联的结束时间相结合，会形成以受限方式增长的区域。该索引的思路是通过引入同样会增长的边界区域来适应不断增长的数据区域。具体来说，边界区域是时间参数化的，并且每次进行查询时都会计算其范围。

The ${\mathrm{R}}^{\mathrm{{ST}}}$ -tree [24] is the spatiotemporal index that indexes the histories of the positions of objects. Positions are assumed to remain constant in-between explicit index updates, and their histories are captured by associating valid and transaction time intervals, which may be now-relative. with them. The continuity thus stems from the temporal aspects rather than the spatial, and the techniques employed in this index are more akin to those employed in the GR-tree than those employed here.

${\mathrm{R}}^{\mathrm{{ST}}}$ 树 [24] 是一种时空索引，用于对对象位置的历史信息进行索引。假设在显式索引更新之间，位置保持不变，并且通过关联有效时间和事务时间间隔（这些间隔可能是相对于当前时间的）来记录其历史信息。因此，这种连续性源于时间方面而非空间方面，并且该索引所采用的技术与 GR 树所采用的技术更为相似，而与本文所采用的技术不同。

Finally, Pfoser et al. [21] consider the separate, but related problem of indexing the past trajectories of moving points, which are represented as polylines (connected line segments).

最后，福瑟（Pfoser）等人 [21] 考虑了一个独立但相关的问题，即对移动点的过去轨迹进行索引，这些轨迹用折线（相连的线段）表示。

## 3 Structure and Algorithms

## 3 结构与算法

This section presents the structure and algorithms of the TPR-tree. The notion of a time-parameterized bounding rectangle is defined. It is shown how the tree is queried, and dynamic update algorithms are presented that tailor the tree to a specific time horizon $\mathrm{H}$ . In the following, we use the term bounding interval for a one-dimensional bounding rectangle and the term bounding rectangle for any $d$ -dimensional hyper-rectangle.

本节介绍 TPR 树的结构和算法。定义了时间参数化边界矩形的概念。展示了如何对树进行查询，并给出了动态更新算法，这些算法可使树适应特定的时间范围 $\mathrm{H}$。在下面的内容中，我们将一维的边界矩形称为边界区间，将任何 $d$ 维的超矩形称为边界矩形。

### 3.1 Index Structure

### 3.1 索引结构

The TPR-tree is a balanced, multi-way tree with the structure of an R-tree. Entries in leaf nodes are pairs of the position of a moving point and a pointer to the moving point, and entries in internal nodes are pairs of a pointer to a subtree and a rectangle that bounds the positions of all moving points or other bounding rectangles in that subtree.

TPR 树是一种平衡的多路树，其结构与 R 树类似。叶节点中的条目是移动点的位置与指向该移动点的指针的对，内部节点中的条目是指向子树的指针与一个矩形的对，该矩形界定了该子树中所有移动点或其他边界矩形的位置。

As suggested in Section 2, the position of a moving point is represented by a reference position and a corresponding velocity vector $- \left( {x,v}\right)$ in the one-dimensional case,where $x = x\left( {t}_{ref}\right)$ . We let ${t}_{ref}$ be equal to the index creation time, ${t}_{l}$ . Other possibilities include setting ${t}_{ref}$ to some constant value,e.g.,0,or using different ${t}_{ref}$ values in different nodes.

正如第2节所建议的，在一维情况下，一个移动点的位置由一个参考位置和一个相应的速度向量$- \left( {x,v}\right)$表示，其中$x = x\left( {t}_{ref}\right)$ 。我们令${t}_{ref}$等于索引创建时间，即${t}_{l}$ 。其他可能的做法包括将${t}_{ref}$设置为某个常数值，例如0，或者在不同的节点中使用不同的${t}_{ref}$值。

To bound a group of $d$ -dimensional moving points, $d$ - dimensional bounding rectangles are used that are also time-parameterized, i.e., their coordinates are functions of time. A time-parameterized bounding rectangle bounds all enclosed points or rectangles at all times not earlier than the current time.

为了界定一组$d$维移动点，使用了同样随时间参数化的$d$维边界矩形，即它们的坐标是时间的函数。一个随时间参数化的边界矩形在所有不早于当前时间的时刻都能界定所有被包含的点或矩形。

A tradeoff exists between how tightly a bounding rectangle bounds the enclosed moving points or rectangles across time and the storage needed to capture the bounding rectangle. It would be ideal to employ time-parameterized bounding rectangles that are always minimum, but the storage cost appears to be excessive. In the general case, doing so deteriorates to enumerating all the enclosed moving points or rectangles. This is exemplified by Figure 4, where a node consists of two one-dimensional points $A$ and $B$ moving towards each other. Each of these points plays the role of lower (resp. upper) bound of the minimum bounding interval at some time. Examples with this property may be constructed for any number of points.

在一个边界矩形随时间对被包含的移动点或矩形的界定紧密程度与捕获该边界矩形所需的存储空间之间存在权衡。使用始终最小的随时间参数化的边界矩形是理想的，但存储成本似乎过高。在一般情况下，这样做会退化为枚举所有被包含的移动点或矩形。图4举例说明了这一点，其中一个节点由两个一维点$A$和$B$相互靠近移动组成。在某些时刻，这些点中的每一个都充当最小边界区间的下界（或上界）。对于任意数量的点，都可以构造出具有这种性质的例子。

<!-- Media -->

<!-- figureText: $\mathrm{t} = 0$ t = 3 -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_4.jpg?x=153&y=1610&w=658&h=158&r=0"/>

Figure 4: Conservative (Dashed) Versus Always Minimum (Solid) Bounding Intervals

图4：保守（虚线）与始终最小（实线）边界区间

<!-- Media -->

Instead of using true, always minimum bounding rectangles, the TPR-tree employs what we term conservative bounding rectangles, which are minimum at some time point, but possibly (and most likely!) not at later times. In the one-dimensional case, the lower bound of a conservative interval is set to move with the minimum speed of the enclosed points, while the upper bound is set to move with the maximum speed of the enclosed points (speeds are negative or positive, depending on the direction). This ensures that conservative bounding intervals are indeed bounding for all times considered.

TPR树没有使用真正的、始终最小的边界矩形，而是采用了我们所称的保守边界矩形，这些矩形在某个时间点是最小的，但在后续时间可能（而且很可能！）不是最小的。在一维情况下，保守区间的下界被设置为以被包含点的最小速度移动，而上界被设置为以被包含点的最大速度移动（速度根据方向可为负或正）。这确保了保守边界区间在所有考虑的时间内确实能起到界定作用。

Figure 4 illustrates conservative bounding intervals. The left hand side of the conservative interval in the figure starts at the position of object A at time 0 and moves left at the speed of object B, and the right hand side of the interval starts at object B at time 0 and moves right at the speed of object A. Conservative bounding intervals never shrink. At best, when all of the enclosed points have the same velocity vector, a conservative bounding interval has constant size, although it may move.

图4展示了保守边界区间。图中保守区间的左侧从时间0时对象A的位置开始，并以对象B的速度向左移动，而区间的右侧从时间0时对象B的位置开始，并以对象A的速度向右移动。保守边界区间永远不会缩小。在最好的情况下，当所有被包含的点具有相同的速度向量时，保守边界区间的大小是恒定的，尽管它可能会移动。

Following the representation of moving points, we let ${t}_{\text{ref }} = {t}_{l}$ and capture a one-dimensional time-parameterized bounding interval $\left\lbrack  {{x}^{ \vdash  }\left( t\right) ,{x}^{ \dashv  }\left( t\right) }\right\rbrack   = \left\lbrack  {{x}^{ \vdash  }\left( {t}_{l}\right)  + {v}^{ \vdash  }(t - }\right.$ $\left. {\left. {t}_{l}\right) ,{x}^{ \dashv  }\left( {t}_{l}\right)  + {v}^{ \dashv  }\left( {t - {t}_{l}}\right) }\right\rbrack$ as $\left( {{x}^{ \vdash  },{x}^{ \dashv  },{v}^{ \vdash  },{v}^{ \dashv  }}\right)$ ,where

根据移动点的表示方法，我们令${t}_{\text{ref }} = {t}_{l}$并将一个一维随时间参数化的边界区间$\left\lbrack  {{x}^{ \vdash  }\left( t\right) ,{x}^{ \dashv  }\left( t\right) }\right\rbrack   = \left\lbrack  {{x}^{ \vdash  }\left( {t}_{l}\right)  + {v}^{ \vdash  }(t - }\right.$ $\left. {\left. {t}_{l}\right) ,{x}^{ \dashv  }\left( {t}_{l}\right)  + {v}^{ \dashv  }\left( {t - {t}_{l}}\right) }\right\rbrack$表示为$\left( {{x}^{ \vdash  },{x}^{ \dashv  },{v}^{ \vdash  },{v}^{ \dashv  }}\right)$ ，其中

$$
{x}^{ \vdash  } = {x}^{ \vdash  }\left( {t}_{l}\right)  = \mathop{\min }\limits_{i}\left\{  {{o}_{i} \cdot  {x}^{ \vdash  }\left( {t}_{l}\right) }\right\}  
$$

$$
{x}^{ \dashv  } = {x}^{ \dashv  }\left( {t}_{l}\right)  = \mathop{\max }\limits_{i}\left\{  \left( {{o}_{i}.{x}^{ \dashv  }\left( {t}_{l}\right) }\right. \right\}  
$$

$$
{v}^{ \vdash  } = \mathop{\min }\limits_{i}\left\{  {{o}_{i}.{v}^{ \vdash  }}\right\}  
$$

$$
{v}^{ \dashv  } = \mathop{\max }\limits_{i}\left\{  {{o}_{i} \cdot  {v}^{ \dashv  }}\right\}  
$$

Here,the ${o}_{i}$ range over the bounding intervals to be enclosed. If instead the bounding interval being defined is to bound moving points,the ${o}_{i}$ range over these points, ${o}_{i}.{x}^{ \vdash  }\left( {t}_{l}\right)$ and ${o}_{i}.{x}^{ \dashv  }\left( {t}_{l}\right)$ are replaced by ${o}_{i}.x\left( {t}_{l}\right)$ ,and ${o}_{i}.{v}^{ \vdash  }$ and ${o}_{i}.{v}^{ \dashv  }$ are replaced by ${o}_{i}.v$ .

这里，${o}_{i}$的取值范围是要被包含的边界区间。如果定义的边界区间是为了界定移动点，那么${o}_{i}$的取值范围是这些点，${o}_{i}.{x}^{ \vdash  }\left( {t}_{l}\right)$和${o}_{i}.{x}^{ \dashv  }\left( {t}_{l}\right)$被替换为${o}_{i}.x\left( {t}_{l}\right)$ ，并且${o}_{i}.{v}^{ \vdash  }$和${o}_{i}.{v}^{ \dashv  }$被替换为${o}_{i}.v$ 。

The rectangles defined above are termed load-time bounding rectangles and are bounding for all times not before ${t}_{l}$ . Because the rectangles never shrink, but may actually grow too much, it is desirable to be able to adjust them occasionally. Specifically, as the index is only queried for times greater or equal to the current time, it is possible and probably attractive to adjust the bounding rectangles every time any of the moving points or rectangles that they bound are updated. The following formulas specify the adjustments to the bounding rectangles that may be made during updates.

上述定义的矩形被称为加载时边界矩形（load-time bounding rectangles），它们在不早于${t}_{l}$的所有时刻都起到边界作用。由于这些矩形不会缩小，但实际上可能会过度增大，因此偶尔对其进行调整是很有必要的。具体而言，由于索引仅针对大于或等于当前时间的时刻进行查询，所以每当它们所界定的任何移动点或矩形被更新时，对边界矩形进行调整是可行的，而且可能很有吸引力。以下公式规定了在更新过程中对边界矩形可能进行的调整。

$$
{x}^{ \vdash  } = \mathop{\min }\limits_{i}\left\{  {{o}_{i}.{x}^{ \vdash  }\left( {t}_{upd}\right) }\right\}   - {v}^{ \vdash  }\left( {{t}_{upd} - {t}_{l}}\right) 
$$

$$
{x}^{ \dashv  } = \mathop{\max }\limits_{i}\left\{  {{o}_{i}.{x}^{ \dashv  }\left( {t}_{upd}\right) }\right\}   - {v}^{ \dashv  }\left( {{t}_{upd} - {t}_{l}}\right) 
$$

Here, ${t}_{upd}$ is the time of the update,and the formulas may be restricted to apply to the bounding of points rather than intervals, as before. Each formula involves five terms, which may differ by orders of magnitude. Special care must be taken to manage the rounding errors that may occur in the finite-precision floating-point arithmetic (e.g., IEEE standard 754) used for implementing the formulas [8].

这里，${t}_{upd}$是更新的时间，并且和之前一样，这些公式可能仅限于应用于点的边界，而非区间的边界。每个公式包含五项，它们的数量级可能不同。在实现这些公式时，必须特别注意处理有限精度浮点运算（例如，IEEE标准754）中可能出现的舍入误差[8]。

We call these rectangles update-time bounding rectangles. The two types of bounding rectangles are shown in Figure 5. The bold top and bottom lines capture the load-time, time-parameterized bounding interval for the four moving objects represented by the four lines. At time ${t}_{upd}$ ,a more narrow and thus better update-time bounding interval is introduced that is bounding from ${t}_{upd}$ and onwards.

我们将这些矩形称为更新时边界矩形（update-time bounding rectangles）。两种类型的边界矩形如图5所示。粗的上下线表示四个移动对象（由四条线表示）的加载时、时间参数化边界区间。在时间${t}_{upd}$，引入了一个更窄、因而更好的更新时边界区间，该区间从${t}_{upd}$及之后起到边界作用。

<!-- Media -->

<!-- figureText: 10 02 o3 04 -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_5.jpg?x=198&y=343&w=563&h=510&r=0"/>

Figure 5: Load-Time (Bold) and Update-Time (Dashed) Bounding Intervals for Four Moving Points

图5：四个移动点的加载时（粗线）和更新时（虚线）边界区间

<!-- Media -->

It is worth noticing that the sole use of load-time bounding rectangles corresponds to simply bounding the ${2d}$ - dimensional points that result from the dual transformation of the linear trajectories, as proposed by Kollios et al. [13]. Update-time bounding rectangles go beyond this approach.

值得注意的是，仅使用加载时边界矩形相当于简单地界定由线性轨迹的对偶变换得到的${2d}$维点，这是由科利奥斯（Kollios）等人[13]提出的。更新时边界矩形超越了这种方法。

### 3.2 Querying

### 3.2 查询

With the definition of bounding rectangles in place, we show how the three types of queries presented in Section 2 are answered using the TPR-tree.

在定义了边界矩形之后，我们将展示如何使用TPR树来回答第2节中提出的三种类型的查询。

Answering a timeslice query proceeds as for the regular R-tree, the only difference being that all bounding rectangles are computed for the time ${t}^{q}$ specified in the query before intersection is checked. Thus, a bounding interval specified by $\left( {{x}^{ \vdash  },{x}^{ \dashv  },{v}^{ \vdash  },{v}^{ \dashv  }}\right)$ satisfies a query $\left( {\left( \left\lbrack  {{a}^{ \vdash  },{a}^{ \dashv  }}\right\rbrack  \right) ,{t}^{q}}\right)$ if and only if ${a}^{ \vdash  } \leq  {x}^{ \dashv  } + {v}^{ \dashv  }\left( {{t}^{q} - {t}_{l}}\right)  \land  {a}^{ \dashv  } \geq  {x}^{ \vdash  } + {v}^{ \vdash  }\left( {{t}^{q} - {t}_{l}}\right)$ .

回答时间片查询的过程与常规R树相同，唯一的区别在于，在检查相交情况之前，所有边界矩形都要针对查询中指定的时间${t}^{q}$进行计算。因此，由$\left( {{x}^{ \vdash  },{x}^{ \dashv  },{v}^{ \vdash  },{v}^{ \dashv  }}\right)$指定的边界区间满足查询$\left( {\left( \left\lbrack  {{a}^{ \vdash  },{a}^{ \dashv  }}\right\rbrack  \right) ,{t}^{q}}\right)$，当且仅当${a}^{ \vdash  } \leq  {x}^{ \dashv  } + {v}^{ \dashv  }\left( {{t}^{q} - {t}_{l}}\right)  \land  {a}^{ \dashv  } \geq  {x}^{ \vdash  } + {v}^{ \vdash  }\left( {{t}^{q} - {t}_{l}}\right)$。

To answer window queries and moving queries, we need to be able to check if,in $\left( {\bar{x},t}\right)$ -space,the trapezoid of a query (cf. Figure 6) intersects with the trapezoid formed by the part of the trajectory of a bounding rectangle that is between the start and end times of the query. With one spatial dimension, this is relatively simple. For more dimensions, generic polyhedron-polyhedron intersection tests may be used [9], but due to the restricted nature of this problem, a simpler and more efficient algorithm may be devised.

为了回答窗口查询和移动查询，我们需要能够检查在$\left( {\bar{x},t}\right)$空间中，查询的梯形（参见图6）是否与由边界矩形轨迹在查询的开始时间和结束时间之间的部分所形成的梯形相交。在一维空间中，这相对简单。对于更多维度，可以使用通用的多面体 - 多面体相交测试[9]，但由于这个问题的特殊性，可以设计出一种更简单、更高效的算法。

Specifically,we provide an algorithm for checking if a $d$ - dimensional time-parameterized bounding rectangle $R$ given by parameters $\left( {{x}_{1}^{ \vdash  },{x}_{1}^{ \dashv  },{x}_{2}^{ \vdash  },{x}_{2}^{ \dashv  },\ldots ,{x}_{d}^{ \vdash  },{x}_{d}^{ \dashv  },{v}_{1}^{ \vdash  },{v}_{1}^{ \dashv  },{v}_{2}^{ \vdash  },{v}_{2}^{ \dashv  },}\right.$ $\ldots ,{v}_{d}^{ \vdash  },{v}_{d}^{ \dashv  }$ ) intersects a moving query $Q = \left( \left( \left\lbrack  {{a}_{1}^{ \vdash  },{a}_{1}^{ \dashv  }}\right\rbrack  \right. \right.$ , $\left\lbrack  {{a}_{2}^{ \vdash  },{a}_{2}^{ \dashv  }}\right\rbrack  ,\ldots ,\left\lbrack  {{a}_{d}^{ \vdash  },{a}_{d}^{ \dashv  }}\right\rbrack  ,\left\lbrack  {{w}_{1}^{ \vdash  },{w}_{1}^{ \dashv  }}\right\rbrack  ,\left\lbrack  {{w}_{2}^{ \vdash  },{w}_{2}^{ \dashv  }}\right\rbrack  ,\ldots ,\left\lbrack  {{w}_{d}^{ \vdash  },{w}_{d}^{ \dashv  }}\right\rbrack  )$ , ${t}^{ \vdash  },{t}^{ \dashv  })$ . This formulation of a moving query as a time-parameterized rectangle with starting and ending times is more convenient than the definition given in Section 2.2. The velocities $w$ are obtained by subtracting ${R}_{2}$ from ${R}_{1}$ in the earlier definition and then normalizing them with the length of interval $\left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack$ .

具体而言，我们提供了一种算法，用于检查由参数 $\left( {{x}_{1}^{ \vdash  },{x}_{1}^{ \dashv  },{x}_{2}^{ \vdash  },{x}_{2}^{ \dashv  },\ldots ,{x}_{d}^{ \vdash  },{x}_{d}^{ \dashv  },{v}_{1}^{ \vdash  },{v}_{1}^{ \dashv  },{v}_{2}^{ \vdash  },{v}_{2}^{ \dashv  },}\right.$ $\ldots ,{v}_{d}^{ \vdash  },{v}_{d}^{ \dashv  }$ 给出的 $d$ 维时间参数化边界矩形 $R$ 是否与移动查询 $Q = \left( \left( \left\lbrack  {{a}_{1}^{ \vdash  },{a}_{1}^{ \dashv  }}\right\rbrack  \right. \right.$、$\left\lbrack  {{a}_{2}^{ \vdash  },{a}_{2}^{ \dashv  }}\right\rbrack  ,\ldots ,\left\lbrack  {{a}_{d}^{ \vdash  },{a}_{d}^{ \dashv  }}\right\rbrack  ,\left\lbrack  {{w}_{1}^{ \vdash  },{w}_{1}^{ \dashv  }}\right\rbrack  ,\left\lbrack  {{w}_{2}^{ \vdash  },{w}_{2}^{ \dashv  }}\right\rbrack  ,\ldots ,\left\lbrack  {{w}_{d}^{ \vdash  },{w}_{d}^{ \dashv  }}\right\rbrack  )$、${t}^{ \vdash  },{t}^{ \dashv  })$ 相交。将移动查询表述为具有起始和结束时间的时间参数化矩形，比第2.2节中给出的定义更方便。速度 $w$ 是通过在早期定义中用 ${R}_{1}$ 减去 ${R}_{2}$，然后将其除以区间 $\left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack$ 的长度进行归一化得到的。

<!-- Media -->

<!-- figureText: ${\mathrm{X}}_{\mathrm{j}}$ Query t Bounding interval -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_5.jpg?x=924&y=141&w=568&h=506&r=0"/>

Figure 6: Intersection of a Bounding Interval and a Query

图6：边界区间与查询的交集

<!-- Media -->

The algorithm is based on the observation that for two moving rectangles to intersect, there has to be a time point when their extents intersect in each dimension. Thus, for each dimension $j\left( {j = 1,2,\ldots ,d}\right)$ ,the algorithm computes the time interval ${I}_{j} = \left\lbrack  {{t}_{j}^{ \vdash  },{t}_{j}^{ \dashv  }}\right\rbrack   \subset  \left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack$ when the extents of the rectangles intersect in that dimension. If $I = \mathop{\bigcap }\limits_{{j = 1}}^{d}{I}_{j} =$ $\varnothing$ ,the moving rectangles do not intersect and an empty result is returned; otherwise, the algorithm provides the time interval $I$ when the rectangles intersect. The intervals for each dimension are computed according to the following formulas.

该算法基于这样的观察：两个移动矩形要相交，必须存在一个时间点，使得它们在每个维度上的范围都相交。因此，对于每个维度 $j\left( {j = 1,2,\ldots ,d}\right)$，算法会计算矩形在该维度上的范围相交的时间区间 ${I}_{j} = \left\lbrack  {{t}_{j}^{ \vdash  },{t}_{j}^{ \dashv  }}\right\rbrack   \subset  \left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack$。如果 $I = \mathop{\bigcap }\limits_{{j = 1}}^{d}{I}_{j} =$ $\varnothing$，则移动矩形不相交，并返回空结果；否则，算法会给出矩形相交的时间区间 $I$。每个维度的区间根据以下公式计算。

$$
{I}_{j} = \left\{  \begin{matrix} \varnothing & \text{ if }{a}_{j}^{ \vdash  } > {x}_{j}^{ \dashv  }\left( {t}^{ \vdash  }\right)  \land  {a}_{j}^{ \vdash  }\left( {t}^{ \dashv  }\right)  \vee  \\   & {a}_{j}^{ \dashv  } < {x}_{j}^{ \vdash  }\left( {t}^{ \vdash  }\right)  \land  {a}_{j}^{ \dashv  }\left( {t}^{ \dashv  }\right)  < {x}_{j}^{ \vdash  }\left( {t}^{ \dashv  }\right) \\  \left\lbrack  {{t}_{j}^{ \vdash  },{t}_{j}^{ \dashv  }}\right\rbrack  & \text{ otherwise } \end{matrix}\right. 
$$

The first disjunct in the condition expresses that $Q$ is above $R$ and the second means that $Q$ is below $R$ . Formulas for ${t}_{j}^{ \vdash  }$ and ${t}_{j}^{ - }$ follow.

条件中的第一个析取项表示 $Q$ 在 $R$ 之上，第二个表示 $Q$ 在 $R$ 之下。接下来是 ${t}_{j}^{ \vdash  }$ 和 ${t}_{j}^{ - }$ 的公式。

$$
{t}_{j}^{ \vdash  } = \left\{  \begin{array}{ll} {t}^{ \vdash  } + \frac{{x}_{j}^{ \dashv  }\left( {t}^{ \vdash  }\right)  - {a}_{j}^{ \vdash  }}{{w}_{j}^{ \vdash  } - {v}_{j}^{ \dashv  }} & \text{ if }{a}_{j}^{ \vdash  } > {x}_{j}^{ \dashv  }\left( {t}^{ \vdash  }\right) \\  {t}^{ \vdash  } + \frac{{x}_{j}^{ \vdash  }\left( {t}^{ \vdash  }\right)  - {a}_{j}^{ \dashv  }}{{w}_{j}^{ \dashv  } - {v}_{j}^{ \vdash  }} & \text{ if }{a}_{j}^{ \dashv  } < {x}_{j}^{ \vdash  }\left( {t}^{ \vdash  }\right) \\  {t}^{ \vdash  } & \text{ otherwise } \end{array}\right. 
$$

Here,the first condition states that $Q$ is above $R$ at ${t}^{ \vdash  }$ ,and the second states that $Q$ is below $R$ at ${t}^{ \vdash  }$ .

这里，第一个条件表明 $Q$ 在 ${t}^{ \vdash  }$ 时刻位于 $R$ 之上，第二个条件表明 $Q$ 在 ${t}^{ \vdash  }$ 时刻位于 $R$ 之下。

$$
{t}_{j}^{ \dashv  } = \left\{  \begin{array}{ll} {t}^{ \vdash  } + \frac{{x}_{j}^{ \dashv  }\left( {t}^{ \vdash  }\right)  - {a}_{j}^{ \vdash  }}{{w}_{j}^{ \vdash  } - {v}_{j}^{ \dashv  }} & \text{ if }{a}_{j}^{ \vdash  }\left( {t}^{ \dashv  }\right)  > {x}_{j}^{ \dashv  }\left( {t}^{ \dashv  }\right) \\  {t}^{ \vdash  } + \frac{{x}_{j}^{ \vdash  }\left( {t}^{ \vdash  }\right)  - {a}_{j}^{ \dashv  }}{{w}_{j}^{ \dashv  } - {v}_{j}^{ \vdash  }} & \text{ if }{a}_{j}^{ \dashv  }\left( {t}^{ \dashv  }\right)  < {x}_{j}^{ \vdash  }\left( {t}^{ \dashv  }\right) \\  {t}^{ \dashv  } & \text{ otherwise } \end{array}\right. 
$$

In this formula,the first condition states that $Q$ is above $R$ at ${t}^{ \dashv  }$ ,and the second states that $Q$ is below $R$ at ${t}^{ \dashv  }$ .

在这个公式中，第一个条件表明 $Q$ 在 ${t}^{ \dashv  }$ 时刻位于 $R$ 之上，第二个条件表明 $Q$ 在 ${t}^{ \dashv  }$ 时刻位于 $R$ 之下。

To see how ${t}_{j}^{ \vdash  }$ and ${t}_{j}^{ \dashv  }$ are computed,consider the case where $Q$ is below $R$ at ${t}^{ \dashv  }$ . Then $Q$ must not be below $R$ at ${t}^{ \vdash  }$ ,as otherwise $Q$ is always below $R$ and there is no intersection (the case of no intersection is already accounted for). This means that the line ${a}_{j}^{ \dashv  } + {w}_{j}^{ \dashv  }\left( {t - {t}^{ \vdash  }}\right)$ intersects the line ${x}_{j}^{ \vdash  }\left( {t}^{ \vdash  }\right)  + {v}_{j}^{ \vdash  }\left( {t - {t}^{ \vdash  }}\right)$ within the time interval $\left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack$ . Solving for $t$ gives the desired intersection time $\left( {t}_{j}^{ \dashv  }\right)$ .

为了了解如何计算${t}_{j}^{ \vdash  }$和${t}_{j}^{ \dashv  }$，考虑在${t}^{ \dashv  }$时刻$Q$位于$R$下方的情况。那么在${t}^{ \vdash  }$时刻，$Q$一定不能位于$R$下方，否则$Q$将始终位于$R$下方，也就不存在交点（无交点的情况已经考虑过了）。这意味着直线${a}_{j}^{ \dashv  } + {w}_{j}^{ \dashv  }\left( {t - {t}^{ \vdash  }}\right)$在时间区间$\left\lbrack  {{t}^{ \vdash  },{t}^{ \dashv  }}\right\rbrack$内与直线${x}_{j}^{ \vdash  }\left( {t}^{ \vdash  }\right)  + {v}_{j}^{ \vdash  }\left( {t - {t}^{ \vdash  }}\right)$相交。求解$t$可得到所需的相交时间$\left( {t}_{j}^{ \dashv  }\right)$。

Figure 6 exemplifies a moving query, a bounding rectangle, and their intersection time interval in one dimension.

图6展示了一维情况下的移动查询、边界矩形及其相交时间区间的示例。

### 3.3 Heuristics for Tree Organization

### 3.3 树组织的启发式方法

As a precursor to designing the insertion algorithms for the TPR-tree, we discuss how to group moving objects into nodes so that the tree most efficiently supports timeslice queries when assuming a time horizon $\mathrm{H}$ . The objective is to identify principles, or heuristics, that apply to both dynamic insertions and bulkloading, and to any number of dimensions. The goal is to obtain a versatile index.

作为设计TPR树插入算法的前奏，我们讨论如何将移动对象分组到节点中，以便在假设时间范围为$\mathrm{H}$时，该树能最有效地支持时间片查询。目标是确定适用于动态插入和批量加载，以及任意维度的原则或启发式方法。目的是获得一个通用的索引。

It is clear that when $\mathrm{H}$ is close to zero,the tree may simply use existing R-tree insertion and bulkloading algorithms. The movement of the point objects and the growth of the bounding rectangles become irrelevant-only their initial positions and extents matter. In contrast,when $\mathrm{H}$ is large, grouping the moving points according to their velocity vectors is of essence. It is desirable that the bounding rectangles are as small as possible at all times in $\left\lbrack  {{t}_{l},{t}_{l} + \mathrm{H}}\right\rbrack$ ,the interval during which the result of the operation (insertion or bulkloading) may be visible to queries ( ${t}_{l}$ is thus the time of an insertion or the index creation time). An important aspect in achieving this is to keep the growth rates of the bounding rectangles, and thus the values of their "velocity extents," low. (In one-dimensional space, the velocity extent of a bounding interval is equal to ${v}^{ + } - {v}^{ \vdash  }$ .)

显然，当$\mathrm{H}$接近零时，该树可以简单地使用现有的R树插入和批量加载算法。点对象的移动和边界矩形的增长变得无关紧要——只有它们的初始位置和范围才重要。相反，当$\mathrm{H}$很大时，根据移动点的速度向量对其进行分组至关重要。理想情况下，在$\left\lbrack  {{t}_{l},{t}_{l} + \mathrm{H}}\right\rbrack$这个时间段内，边界矩形应尽可能小，该时间段是操作（插入或批量加载）的结果可能被查询可见的时间段（因此${t}_{l}$是插入时间或索引创建时间）。实现这一点的一个重要方面是保持边界矩形的增长率，从而使其“速度范围”的值较低。（在一维空间中，边界区间的速度范围等于${v}^{ + } - {v}^{ \vdash  }$。）

This leads to the following general approach. The insertion and bulkloading algorithms of the ${\mathrm{R}}^{ * }$ -tree,which we consider extending to moving points, aim to minimize objective functions such as the areas of the bounding rectangles, their margins (perimeters), and the overlap among the bounding rectangles. In our context, these functions are time dependent, and we should consider their evolution in $\left\lbrack  {{t}_{l},{t}_{l} + \mathrm{H}}\right\rbrack$ . Specifically,given an objective function $A\left( t\right)$ ,the following integral should be minimized.

这导致了以下通用方法。我们考虑将${\mathrm{R}}^{ * }$树的插入和批量加载算法扩展到移动点，其目标是最小化诸如边界矩形的面积、边界矩形的边距（周长）以及边界矩形之间的重叠等目标函数。在我们的上下文中，这些函数是时间相关的，我们应该考虑它们在$\left\lbrack  {{t}_{l},{t}_{l} + \mathrm{H}}\right\rbrack$内的变化。具体来说，给定一个目标函数$A\left( t\right)$，应最小化以下积分。

$$
{\int }_{{t}_{l}}^{{t}_{l} + H}A\left( t\right) {dt} \tag{1}
$$

If $A\left( t\right)$ is area,the integral computes the area (volume) of the trapezoid that represents part of the trajectory of a bounding rectangle in $\left( {\bar{x},t}\right)$ -space (see Figure 6).

如果$A\left( t\right)$是面积，该积分计算的是在$\left( {\bar{x},t}\right)$空间中表示边界矩形轨迹一部分的梯形的面积（体积）（见图6）。

We use the integral in Formula 1 in the dynamic update algorithms, described next, and in the bulkloading algorithms, described elsewhere [23].

我们在接下来描述的动态更新算法以及其他地方描述的批量加载算法[23]中使用公式1中的积分。

### 3.4 Insertion and Deletion

### 3.4 插入与删除

The insertion algorithm of the ${\mathrm{R}}^{ * }$ -tree employs functions that compute the area of a bounding rectangle, the intersection of two bounding rectangles, the margin of a bounding rectangle (when splitting a node), and the distance between the centers of two bounding rectangles (used when doing forced reinsertions) [5]. The TPR-tree's insertion algorithm is the same as that of the ${\mathrm{R}}^{ * }$ -tree,with one exception: instead of the functions mentioned here, integrals as in Formula 1 of those functions are used.

${\mathrm{R}}^{ * }$树的插入算法会使用一些函数，这些函数用于计算边界矩形的面积、两个边界矩形的交集、边界矩形的边距（在分裂节点时使用）以及两个边界矩形中心之间的距离（在进行强制重新插入时使用）[5]。TPR树的插入算法与${\mathrm{R}}^{ * }$树的插入算法相同，唯一的例外是：这里不使用上述提到的函数，而是使用这些函数在公式1中的积分形式。

Computing the integrals of the area, margin, and distance are relatively straightforward [23]. The algorithm that computes the integral of the intersection of two time-parameterized rectangles is an extension of the algorithm for checking if such rectangles overlap (see Section 3.2). At each time point when the rectangles intersect, the intersection region is a rectangle and, in each dimensions, the upper (lower) bound of this rectangle is defined by the upper (lower) bound of one of the two intersecting rectangles.

计算面积、边距和距离的积分相对直接[23]。计算两个时间参数化矩形交集积分的算法是检查此类矩形是否重叠的算法的扩展（见3.2节）。在矩形相交的每个时间点，交集区域是一个矩形，并且在每个维度上，该矩形的上（下）界由两个相交矩形之一的上（下）界定义。

The algorithm thus divides the time interval returned by the overlap-checking algorithm into consecutive time intervals so that, during each of these, the intersection is defined by a time-parameterized rectangle. The intersection area integral is then computed as a sum of area integrals. Figure 6 illustrates the subdivision of the intersection time interval into three smaller intervals for the one-dimensional case. The algorithm is given elsewhere [23].

因此，该算法将重叠检查算法返回的时间间隔划分为连续的时间间隔，以便在每个这样的时间间隔内，交集由一个时间参数化矩形定义。然后，将交集面积积分计算为面积积分的总和。图6展示了在一维情况下将交集时间间隔细分为三个较小间隔的情况。该算法在其他地方给出[23]。

In Section 2.3,parameter $\mathrm{H} = \mathrm{U} + \mathrm{W}$ was introduced. This parameter is most intuitive in a static setting, and for static data. In a dynamic setting, $\mathrm{W}$ remains a component of $\mathrm{H}$ ,which is the length of the time period where integrals are computed in the insertion algorithm. How large the other component of $\mathrm{H}$ should be depends on the update frequency. If this is high, the effect of an insertion on the tree will not persist long and, thus, H should not exceed W by much. The experimental studies in Section 4 aim at determining what is a good range of values for $\mathrm{H}$ in terms of the update frequency.

在2.3节中，引入了参数$\mathrm{H} = \mathrm{U} + \mathrm{W}$。这个参数在静态设置和处理静态数据时最为直观。在动态设置中，$\mathrm{W}$仍然是$\mathrm{H}$的一个组成部分，$\mathrm{H}$是插入算法中计算积分的时间段的长度。$\mathrm{H}$的另一个组成部分应该有多大取决于更新频率。如果更新频率较高，插入操作对树的影响不会持续很长时间，因此，H不应比W大太多。第4节的实验研究旨在确定就更新频率而言，$\mathrm{H}$的合适取值范围。

The introduction of the integrals is the most important step in rendering the ${\mathrm{R}}^{ * }$ -tree insertion algorithm suitable for the TPR-tree,but one more aspect of the ${\mathrm{R}}^{ * }$ -tree algorithm must be revisited. The ${\mathrm{R}}^{ * }$ -tree split algorithm selects one distribution of entries between two nodes from a set of candidate distributions, which are generated based on sortings of point positions along each of the coordinate axes. In the TPR-tree split algorithm, moving point (or rectangle) positions at different time points are used when sorting. With load-time bounding rectangles,positions at ${t}_{l}$ are used, and with update-time bounding rectangles, positions at the current time are used.

引入积分是使${\mathrm{R}}^{ * }$树插入算法适用于TPR树的最重要步骤，但${\mathrm{R}}^{ * }$树算法的另一个方面也必须重新审视。${\mathrm{R}}^{ * }$树的分裂算法从一组候选分布中选择一种条目在两个节点之间的分布，这些候选分布是基于沿每个坐标轴的点位置排序生成的。在TPR树的分裂算法中，排序时使用不同时间点的移动点（或矩形）位置。对于加载时边界矩形，使用${t}_{l}$时刻的位置；对于更新时边界矩形，使用当前时间的位置。

Finally, in addition to sortings along the spatial dimensions, the split algorithm is extended to consider also sort-ings along the velocity dimensions, i.e., sortings obtained by sorting on the coordinates of the velocity vectors. The rationale is that distributing the moving points based on the velocity dimensions may result in bounding rectangles with smaller "velocity extents" and which consequently grow more slowly.

最后，除了沿空间维度进行排序外，分裂算法还扩展为考虑沿速度维度的排序，即通过对速度向量的坐标进行排序得到的排序。其基本原理是，基于速度维度分布移动点可能会得到“速度范围”较小的边界矩形，从而这些矩形的增长速度会更慢。

Deletions in the TPR-tree are performed as in the ${\mathrm{R}}^{ * }$ -tree. If a node gets underfull, it is eliminated and its entries are reinserted.

TPR树中的删除操作与${\mathrm{R}}^{ * }$树中的操作相同。如果一个节点变得未满，则将其删除并重新插入其条目。

## 4 Performance Experiments

## 4 性能实验

In this section we report on performance experiments with the TPR-tree. The generation of two- and three-dimensional moving point data and the settings for the experiments are described first, followed by the presentation of the results of the experiments.

在本节中，我们报告了TPR树的性能实验。首先描述二维和三维移动点数据的生成以及实验设置，然后展示实验结果。

### 4.1 Experimental Setup and Workload Generation

### 4.1 实验设置与工作负载生成

The implementation of the TPR-tree used in the experiments is based on the Generalized Search Tree Package, GiST [10]. The page size (and tree node size) is set to $4\mathrm{\;k}$ bytes,which results in 204 and 146 entries per leaf-node for two- and three-dimensional data,respectively. A page buffer of ${200}\mathrm{k}$ bytes, i.e., 50 pages, is used [16], where the root of a tree is pinned and the least-recently-used page replacement policy is employed. The nodes that are modified during an index operation are marked as "dirty" in the buffer and are written to disk at the end of the operation or when they otherwise have to be removed from the buffer.

实验中使用的TPR树的实现基于广义搜索树包GiST[10]。页面大小（即树节点大小）设置为$4\mathrm{\;k}$字节，这分别导致二维和三维数据的每个叶节点有204和146个条目。使用一个大小为${200}\mathrm{k}$字节（即50页）的页面缓冲区[16]，其中树的根节点被固定，并采用最近最少使用的页面替换策略。在索引操作期间被修改的节点在缓冲区中被标记为“脏”，并在操作结束时或因其他原因必须从缓冲区中移除时写入磁盘。

The performance studies are based on workloads that intermix queries and update operations on the index, thus simulating index usage across a period of time. In addition, each workload initially bulkloads the index. An efficient bulkloading algorithm developed for the TPR-tree is used [23]. This algorithm is based on the heuristic of minimizing area integrals and has $\mathrm{H}$ as a parameter. We proceed to describe how the updates, queries, and initial bulkloading data are generated.

性能研究基于对索引混合执行查询和更新操作的工作负载，从而模拟一段时间内的索引使用情况。此外，每个工作负载最初都会对索引进行批量加载。这里使用了为TPR树（Time Parameterized R-tree，时间参数化R树）开发的一种高效批量加载算法[23]。该算法基于最小化面积积分的启发式方法，以$\mathrm{H}$为参数。接下来我们将描述更新、查询和初始批量加载数据是如何生成的。

Because moving objects with positions and velocities that are uniformly distributed seems to be rather unrealistic, we attempt to generate more realistic (and skewed) two-dimensional data by simulating a scenario where the objects, e.g., cars, move in a network of routes, e.g., roads, connecting a number of destinations, e.g., cities. In addition to simulating cars moving between cities, the scenario is also motivated by the fact that usually, even if there is no underlying infrastructure, moving objects tend to have destinations.

由于位置和速度均匀分布的移动对象似乎不太现实，我们尝试通过模拟一个场景来生成更真实（且有偏差）的二维数据，在这个场景中，对象（例如汽车）在连接多个目的地（例如城市）的路线网络（例如道路）中移动。除了模拟汽车在城市之间移动外，这个场景的设计还考虑到一个事实，即通常情况下，即使没有底层基础设施，移动对象也往往有目的地。

With the exception of one experiment, the simulated objects in the scenario move in a region of space with dimensions ${1000} \times  {1000}$ kilometers. A number ND of destinations are distributed uniformly in this space and serve as the vertices in a fully connected graph of routes. In most of the experiments, $\mathrm{{ND}} = {20}$ . This corresponds to 380 one-way routes. The number of points is $N = {100},{000}$ for all but one experiment. No objects disappear, and no new objects appear for the duration of a simulation.

除了一个实验外，场景中的模拟对象在一个尺寸为${1000} \times  {1000}$公里的空间区域内移动。一定数量（ND）的目的地均匀分布在这个空间中，并作为路线的完全连通图中的顶点。在大多数实验中，$\mathrm{{ND}} = {20}$ 。这对应于380条单向路线。除一个实验外，所有实验中的点数均为$N = {100},{000}$ 。在模拟期间，没有对象消失，也没有新对象出现。

For the generation of the initial data set that is bulkloaded, objects are placed at random positions on routes. The objects are assigned with equal probability to one of three groups of points with maximum speeds of0.75,1.5,and 3 $\mathrm{{km}}/\mathrm{{min}}\left( {{45},{90}\text{,and}{180}\mathrm{\;{km}}/\mathrm{h}}\right)$ . During the first sixth of a route, objects accelerate from zero speed to their maximum speeds; during the middle two thirds, they travel at their maximum speeds; and during the last one sixth of a route, they decelerate. When an object reaches its destination, a new destination is assigned to it at random.

为了生成用于批量加载的初始数据集，对象被随机放置在路线上的位置。对象以相等的概率被分配到最大速度分别为0.75、1.5和3 $\mathrm{{km}}/\mathrm{{min}}\left( {{45},{90}\text{,and}{180}\mathrm{\;{km}}/\mathrm{h}}\right)$ 的三组点中。在路线的前六分之一行程中，对象从静止加速到最大速度；在中间的三分之二行程中，它们以最大速度行驶；在路线的最后六分之一行程中，它们减速。当一个对象到达其目的地时，会随机为其分配一个新的目的地。

The workload generation algorithm distributes the updates of an object's movement so that updates are performed during the acceleration and deceleration stretches of a route. The number of updates is chosen so that the total average time interval between two updates is approximately equal to a given parameter UI, which is fixed at 60 in most of the experiments.

工作负载生成算法对对象移动的更新进行分配，使得更新在路线的加速和减速阶段执行。更新次数的选择使得两次更新之间的总平均时间间隔大约等于给定参数UI，在大多数实验中，该参数固定为60。

In addition to using data from the above-described simulation, some experiments also use workloads with two- and three-dimensional uniform data. In these workloads, the initial positions of objects are uniformly distributed in space. The directions of the velocity vectors are assigned randomly, both initially and on each update. The speeds (lengths of velocity vectors) are uniformly distributed between 0 and 3 $\mathrm{{km}}/\mathrm{{min}}$ . The time interval between successive updates is uniformly distributed between 0 and 2UI.

除了使用上述模拟中的数据外，一些实验还使用具有二维和三维均匀数据的工作负载。在这些工作负载中，对象的初始位置在空间中均匀分布。速度向量的方向在初始时和每次更新时都随机分配。速度（速度向量的长度）在0到3 $\mathrm{{km}}/\mathrm{{min}}$ 之间均匀分布。连续两次更新之间的时间间隔在0到2UI之间均匀分布。

To generate workloads, the above-described scenarios are run for 600 time units (minutes). For $\mathrm{{UI}} = {60}$ ,this results in approximately one million update operations.

为了生成工作负载，上述场景运行600个时间单位（分钟）。对于$\mathrm{{UI}} = {60}$ ，这将产生大约一百万次更新操作。

In addition to updates, workloads include queries. Each time unit, four queries are generated (2400 in total). Times-lice, window, and moving queries are generated with probabilities0.6,0.2,and 0.2 . The temporal parts of queries are generated randomly in an interval of length $\mathrm{W}$ and starting at the current time. The spatial part of each query is a square occupying a fraction QS of the space $({QS} = {0.25}\%$ in most of the experiments). The spatial parts of timeslice and window queries have random locations. For moving queries, the center of a query follows the trajectory of one of the points currently in the index.

除了更新操作外，工作负载还包括查询操作。每个时间单位生成四个查询（总共2400个）。时间片查询、窗口查询和移动查询的生成概率分别为0.6、0.2和0.2。查询的时间部分在长度为$\mathrm{W}$ 且从当前时间开始的区间内随机生成。每个查询的空间部分是一个正方形，在大多数实验中，该正方形占据空间$({QS} = {0.25}\%$ 的QS比例。时间片查询和窗口查询的空间部分位置随机。对于移动查询，查询的中心跟随索引中当前某个点的轨迹。

The workload generation parameters that are varied in the experiments are given in Table 1. Standard values, used if a parameter is not varied in an experiment, are given in boldface.

实验中变化的工作负载生成参数见表1。如果某个参数在实验中不变化，则使用的标准值以粗体显示。

### 4.2 Investigating the Insertion Algorithm

### 4.2 研究插入算法

As mentioned in Section 3.4, the TPR-tree insertion algorithm depends on the parameter $\mathrm{H}$ ,which is equal to $\mathrm{W}$ plus some duration that is dependent on the frequency of updates. How the frequency of updates affects the choice of a value for $\mathrm{H}$ was explored in two sets of experiments,for data with ${UI} = {60}$ and for data with ${UI} = {120}$ . Workloads with uniform data were run using the TPR-tree. Different values of $\mathrm{H}$ were tried out in each set of experiments.

如3.4节所述，TPR树插入算法依赖于参数$\mathrm{H}$，该参数等于$\mathrm{W}$加上某个取决于更新频率的时长。在两组实验中探究了更新频率如何影响$\mathrm{H}$值的选择，一组实验针对具有${UI} = {60}$的数据，另一组针对具有${UI} = {120}$的数据。使用TPR树运行具有均匀数据的工作负载。在每组实验中都尝试了不同的$\mathrm{H}$值。

<!-- Media -->

<table><tr><td>Parameter</td><td>Description</td><td>Values Used</td></tr><tr><td>ND</td><td>Number of destinations [cardinal number]</td><td>0,2,10,20,40,160</td></tr><tr><td>$\mathrm{N}$</td><td>Number of points [cardinal number]</td><td>100,000,300,000,500,000,700,000,900,000</td></tr><tr><td>UI</td><td>Update interval length [time units]</td><td>60, 120</td></tr><tr><td>W</td><td>Querying window size [time units]</td><td>0,20,40,80,160,320</td></tr><tr><td>QS</td><td>Query size [% of the data space]</td><td>0.1,0.25,0.5,1,2</td></tr></table>

<table><tbody><tr><td>参数</td><td>描述</td><td>使用的值</td></tr><tr><td>未检测到（Not Detected）</td><td>目的地数量 [基数]</td><td>0,2,10,20,40,160</td></tr><tr><td>$\mathrm{N}$</td><td>点数 [基数]</td><td>100,000,300,000,500,000,700,000,900,000</td></tr><tr><td>用户界面（User Interface）</td><td>更新间隔时长 [时间单位]</td><td>60, 120</td></tr><tr><td>W</td><td>查询窗口大小 [时间单位]</td><td>0,20,40,80,160,320</td></tr><tr><td>查询大小（Query Size）</td><td>查询大小 [数据空间的百分比]</td><td>0.1,0.25,0.5,1,2</td></tr></tbody></table>

Table 1: Workload Parameters

表1：工作负载参数

<!-- Media -->

Figure 7 shows the results for ${UI} = {60}$ . Curves are shown for experiments with different querying windows W. The leftmost point of each curve corresponds to a setting of $\mathrm{H} = 0$ .

图7展示了${UI} = {60}$的结果。图中给出了不同查询窗口W的实验曲线。每条曲线的最左点对应于$\mathrm{H} = 0$的设置。

<!-- Media -->

<!-- figureText: 80 W = 0 W = 20 -------- W = 40 -----* 30 60 90 120 $\mathrm{U} = \mathrm{H} - \mathrm{W}$ 70 Search I/O 60 50 40 30 -40 -20 0 15 -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_8.jpg?x=163&y=764&w=619&h=559&r=0"/>

Figure 7: Search Performance For ${UI} = {60}$ and Varying Settings of $\mathrm{H}$

图7：${UI} = {60}$的搜索性能以及$\mathrm{H}$的不同设置

<!-- Media -->

The experiments demonstrate a pattern, namely that the best values of $\mathrm{H}$ lie between ${UI}/2 + \mathrm{W}$ and ${UI} + \mathrm{W}$ . This is not surprising. In ${UI}/2$ time units,approximately half of the entries of each leaf node in the tree are updated, and after UI time units, almost all entries are updated. The leaf-node bounding rectangles, the characteristics of which we integrate using $\mathrm{H}$ ,survive approximately similar time durations. In the subsequent studies,we use $\mathrm{H} = {UI}/2 + \mathrm{W}$ .

实验呈现出一种模式，即$\mathrm{H}$的最优值介于${UI}/2 + \mathrm{W}$和${UI} + \mathrm{W}$之间。这并不奇怪。在${UI}/2$个时间单位内，树中每个叶节点的大约一半条目会被更新，而在UI个时间单位后，几乎所有条目都会被更新。我们使用$\mathrm{H}$来整合其特征的叶节点边界矩形，其存活时间大致相似。在后续研究中，我们使用$\mathrm{H} = {UI}/2 + \mathrm{W}$。

### 4.3 Comparing the TPR-Tree To Its Alternatives

### 4.3 TPR树与其替代方案的比较

A set of experiments with varying workloads were performed in order to compare the relative performance of the R-tree, the TPR-tree with load-time bounding rectangles, and the TPR-tree with update-time bounding rectangles.

为了比较R树、带有加载时边界矩形的TPR树和带有更新时边界矩形的TPR树的相对性能，进行了一组不同工作负载的实验。

For the former,the regular ${\mathrm{R}}^{ * }$ -tree is used to store fragments of trajectories of points in $\left( {\bar{x},t}\right)$ -space. For this to work correctly, the inserted trajectory fragment for a moving point should start at the insertion time and should span $\mathrm{H}$ time units,where $\mathrm{H}$ is at least equal to the maximum possible period between two successive updates of the point. Not meeting this requirement, the R-tree may return incorrect query results because its bounding rectangles "expire" after $\mathrm{H}$ time units. In our simulation-generated workloads,the slowest moving points on routes spanning from one side of the data space to the other may not be updated for as much as 600 time units. For the R-tree we,thus,set $\mathrm{H} = {600}$ ,which is the duration of the simulation.

对于前者，使用常规的${\mathrm{R}}^{ * }$树来存储$\left( {\bar{x},t}\right)$空间中各点轨迹的片段。为了使其正常工作，插入的移动点的轨迹片段应从插入时间开始，并应跨越$\mathrm{H}$个时间单位，其中$\mathrm{H}$至少等于该点两次连续更新之间的最大可能周期。如果不满足这一要求，R树可能会返回错误的查询结果，因为其边界矩形在$\mathrm{H}$个时间单位后会“过期”。在我们的模拟生成的工作负载中，从数据空间一侧到另一侧的路线上移动最慢的点可能在多达600个时间单位内都不会被更新。因此，对于R树，我们将$\mathrm{H} = {600}$设置为模拟的持续时间。

Figure 8 shows the average number of I/O operations per query for the three indices when the number of destinations in the simulation is varied. Decreasing the number of destinations adds skew to the distribution of the object positions and their velocity vectors. Thus, uniform data is an extreme case.

图8展示了在模拟中目的地数量变化时，三种索引每次查询的平均I/O操作数。减少目的地数量会使对象位置及其速度向量的分布产生偏差。因此，均匀数据是一种极端情况。

<!-- Media -->

<!-- figureText: 900 R-tree TPR-tree with load-time BRs TPR-tree 40 160 Uniform Number of destinations, ND 800 700 600 Search I/O 500 400 300 200 100 2 10 -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_8.jpg?x=891&y=1096&w=607&h=562&r=0"/>

Figure 8: Search Performance For Varying Numbers of Destinations and Uniform Data

图8：不同目的地数量和均匀数据的搜索性能

<!-- Media -->

As shown, increased skew leads to a decrease in the numbers of I/Os for all three approaches, especially for the TPR-tree. This is expected because when there are more objects with similar velocities, it is easier to pack them into bounding rectangles that have small velocity extents and also are relatively narrow in the spatial dimensions.

如图所示，偏差的增加导致所有三种方法的I/O数量减少，尤其是TPR树。这是可以预料的，因为当有更多具有相似速度的对象时，更容易将它们打包到速度范围较小且在空间维度上相对较窄的边界矩形中。

The figure demonstrates that the TPR-tree is an order of magnitude better than the R-tree. The utility of update-time bounding rectangles can also be seen, although it should be noted that tightening of bounding rectangles increases the update cost. For example, for a workload with 10 destinations, the use of update-time bounding rectangles decreases the average number of I/Os for searches from 33 to 17, while update cost changes from 1.3 to 1.6 I/Os. For uniform data, the change is from 211 to 54, for searches, and from 2 to 3.5 , for updates.

该图表明，TPR树比R树好一个数量级。也可以看到更新时边界矩形的效用，不过应该注意的是，收紧边界矩形会增加更新成本。例如，对于有10个目的地的工作负载，使用更新时边界矩形可使搜索的平均I/O数量从33减少到17，而更新成本从1.3个I/O变为1.6个I/O。对于均匀数据，搜索的I/O数量从211变为54，更新的I/O数量从2变为3.5。

Figure 9 explores the effect of the length of the querying window, W, on querying performance. The relatively constant performance of the TPR-tree may be explained by noting that the data in this experiment is skewed $\left( {\mathrm{{ND}} = {20}}\right)$ , with groups of points having similar velocity vectors. Results would be different for uniform data (cf. Figure 7). The relatively constant performance of the R-tree can be explained by viewing the three-dimensional minimum bounding rectangles used in this tree as two-dimensional bounding rectangles that do not change over time. That is why queries issued at different future times have similar performance.

图9探讨了查询窗口长度W对查询性能的影响。TPR树相对稳定的性能可以这样解释：在这个实验中，数据是$\left( {\mathrm{{ND}} = {20}}\right)$偏斜的，存在具有相似速度向量的点组。对于均匀数据，结果会有所不同（参见图7）。R树相对稳定的性能可以这样理解：将该树中使用的三维最小边界矩形视为不随时间变化的二维边界矩形。这就是为什么在不同未来时间发出的查询具有相似性能的原因。

<!-- Media -->

<!-- figureText: 700 R-tree TPR-tree with load-time BRs TPR-tree 80 160 320 Querying window, W 600 500 Search I/O 400 300 200 100 0 0 20 40 -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_9.jpg?x=163&y=923&w=621&h=574&r=0"/>

Figure 9: Search Performance for Varying W

图9：不同W值下的搜索性能

<!-- Media -->

Next, Figure 10 shows the average performance for queries with different-size spatial extents. The experiments were performed with three-dimensional data. The relatively high costs of the queries in this figure are indicative of how the increased dimensionality of the data adversely affects performance. An experiment with an R-tree using the shorter $\mathrm{H}$ of 120 is also included. Using this value for $\mathrm{H}$ is possible because uniform data is generated where no update interval is longer than $2\mathrm{{UI}}$ ,and $\mathrm{{UI}} = {60}$ in our experiments. This significantly improves the performance of the R-tree, but it remains more than a factor of two worse than the TPR-tree.

接下来，图10展示了不同空间范围查询的平均性能。实验使用三维数据进行。该图中查询的相对高成本表明了数据维度的增加如何对性能产生不利影响。还包括了一个使用较短的$\mathrm{H}$（值为120）的R树实验。使用这个$\mathrm{H}$值是可行的，因为生成的是均匀数据，在我们的实验中，没有更新间隔长于$2\mathrm{{UI}}$，且$\mathrm{{UI}} = {60}$。这显著提高了R树的性能，但它仍然比TPR树差两倍多。

<!-- Media -->

<!-- figureText: 1200 R-tree R-tree with reduced $H$ TPR-tree with load time BRs TPR-tree 0.5 1 2 Query size, QS, % of space 1000 Search I/O 800 600 400 200 0.1 0.25 -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_9.jpg?x=891&y=168&w=613&h=563&r=0"/>

Figure 10: Search Performance For Varying Query Sizes and Three-Dimensional Data

图10：不同查询大小和三维数据的搜索性能

<!-- Media -->

To investigate the scalability of the TPR-tree, we performed experiments with varying numbers of indexed objects. When increasing the numbers of objects, we also scaled the spatial dimensions of the data space so that the density of objects remained approximately the same and so that the number of objects returned by a query was largely (although not completely) unaffected. This scenario corresponds to merging databases that are covering different areas into a single database. Uniform two-dimensional data was used in these experiments.

为了研究TPR树的可扩展性，我们对不同数量的索引对象进行了实验。当增加对象数量时，我们还对数据空间的空间维度进行了缩放，以使对象的密度大致保持相同，从而使查询返回的对象数量在很大程度上（尽管不是完全）不受影响。这种情况对应于将覆盖不同区域的数据库合并为一个数据库。这些实验使用了均匀二维数据。

Figure 11 shows that, as expected, the number of I/O operations for the TPR-tree with update-time bounding rectangles remains almost constant (as long as the number of levels in the tree does not change). The results for the R-tree are not provided, because of excessively high numbers of I/O operations.

如图11所示，正如预期的那样，带有更新时间边界矩形的TPR树的I/O操作数量几乎保持不变（只要树的层数不变）。由于R树的I/O操作数量过多，因此未提供其结果。

To explore how the search performances of the indices evolve with the passage of time, we compute, after each 60 time units, the average query performance for the previous 60 time units. Figure 12 shows the results. In this experiment (and in other similar experiments), the performance of the TPR-tree after 360 time units becomes more than two times worse than the performance at the beginning of the experiment, but from 360 to 600 , no degradation occurs. This behavior is similar to the degradation of the performance of most multidimensional tree structures. When, after bulkloading, dynamic updates are performed, node splits occur, the average fan-out of the tree decreases, and the bounding rectangles created by the bulkloading algorithm change. After some time, the tree stabilizes.

为了探究索引的搜索性能如何随时间变化，我们每60个时间单位计算一次前60个时间单位的平均查询性能。图12展示了结果。在这个实验（以及其他类似实验）中，360个时间单位后TPR树的性能比实验开始时差了两倍多，但从360到600个时间单位，性能没有下降。这种行为与大多数多维树结构的性能下降情况类似。在批量加载后进行动态更新时，会发生节点分裂，树的平均扇出减小，批量加载算法创建的边界矩形也会改变。一段时间后，树会趋于稳定。

As expected, the TPR-tree with load-time bounding rectangles shows an increasing degradation of performance. The bounding rectangles computed at bulkloading time become unavoidably larger as the more distant future is queried. The insertion algorithms try to counter this by making the velocity extents of bounding rectangles as small as possible. For example, in this experiment the average velocity extent of a rectangle (in one of the two velocity dimensions) is 1.32 after the bulkloading and becomes 0.35 after 600 time units (recall that the extent of the data space in each velocity dimension is 6 in our simulation).

正如预期的那样，带有加载时间边界矩形的TPR树的性能逐渐下降。随着查询更远的未来，批量加载时计算的边界矩形不可避免地会变大。插入算法试图通过使边界矩形的速度范围尽可能小来应对这一问题。例如，在这个实验中，批量加载后矩形（在两个速度维度之一）的平均速度范围为1.32，600个时间单位后变为0.35（回想一下，在我们的模拟中，每个速度维度的数据空间范围为6）。

<!-- Media -->

<!-- figureText: 400 TPR-tree 500 600 700 800 900 Number of thousands of objects, N 350 300 Search I/O 250 200 150 100 50 0 100 200 300 400 -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_10.jpg?x=162&y=157&w=624&h=574&r=0"/>

Figure 11: Search Performance for Varying Number of Objects

图11：不同对象数量的搜索性能

<!-- Media -->

## 5 Summary and Future Work

## 5 总结与未来工作

Motivated mainly by the rapid advances in positioning systems, wireless communication technologies, and electronics in general, which promise to render it increasingly feasible to track the positions of increasingly large collections of continuously moving objects, this paper proposes a versatile adaptation of the ${\mathrm{R}}^{ * }$ -tree that supports the efficient querying of the current and anticipated future locations of moving points in one-, two-, and three-dimensional space.

主要受定位系统、无线通信技术以及一般电子技术的快速发展的推动，这些技术有望使跟踪越来越多的连续移动对象的位置变得越来越可行，本文提出了一种对${\mathrm{R}}^{ * }$树的通用改进方案，该方案支持在一维、二维和三维空间中高效查询移动点的当前和预期未来位置。

The new TPR-tree supports timeslice, window, and so-called moving queries. Capturing moving points as linear functions of time, the tree bounds these points using so-called conservative bounding rectangles, which are also time-parameterized and which in turn also bound other such rectangles. The tree is equipped with dynamic update algorithms as well as a bulkloading algorithm. Whereas the ${\mathrm{R}}^{ * }$ -tree’s algorithms use functions that compute the areas, margins, and overlaps of bounding rectangles, the TPR-tree employs integrals of these functions, thus taking into consideration the values of these functions across the time when the tree is queried. The bounding rectangles of tree nodes that are read during updates are tightened, the objective being to improve query performance without affecting update performance much. When splitting nodes, not only the positions of the moving points are considered, but also their velocities.

新的TPR树支持时间片查询、窗口查询和所谓的移动查询。该树将移动点表示为时间的线性函数，并使用所谓的保守边界矩形来界定这些点，这些矩形也是时间参数化的，并且还可以界定其他此类矩形。该树配备了动态更新算法和批量加载算法。${\mathrm{R}}^{ * }$树的算法使用计算边界矩形面积、边距和重叠的函数，而TPR树采用这些函数的积分，从而考虑了在查询树的时间段内这些函数的值。在更新期间读取的树节点的边界矩形会被收紧，目的是在不太影响更新性能的情况下提高查询性能。在分裂节点时，不仅会考虑移动点的位置，还会考虑它们的速度。

<!-- Media -->

<!-- figureText: 600 R-tree TPR-tree with load time BRs TPR-tree 300 360 420 480 540 600 Time 500 Search I/O 400 300 200 100 0 60 120 180 240 -->

<img src="https://cdn.noedgeai.com/0195c901-c37e-7b59-bb97-d1cdb2ab0b88_10.jpg?x=893&y=168&w=619&h=560&r=0"/>

Figure 12: Degradation of Search Performance with Time

图12：搜索性能随时间的下降情况

<!-- Media -->

Because no other proposals for indexing two- and three-dimensional moving points exist, the performance study compares the TPR-tree with the TRP-tree without the tightening of bounding rectangles during updates and with a relatively simple adaptation of the ${\mathrm{R}}^{ * }$ -tree. The study indicates quite clearly that the TPR-tree indeed is capable of supporting queries on moving objects quite efficiently and that it outperforms its competitors by far. The study also demonstrates that the tree does not degrade severely as time passes. Finally, the study indicates how the tree can be tuned to take advantage of a specific update rate.

由于目前不存在对二维和三维移动点进行索引的其他方案，性能研究将TPR树（Time Parameterized R-tree，时间参数化R树）与在更新过程中不收紧边界矩形的TRP树以及对${\mathrm{R}}^{ * }$树的相对简单的改进版本进行了比较。该研究非常明确地表明，TPR树确实能够非常高效地支持对移动对象的查询，并且远远优于其竞争对手。研究还表明，随着时间的推移，该树的性能不会严重下降。最后，研究指出了如何调整该树以利用特定的更新率。

This work points to several interesting research directions. Among these, it would be interesting to study the use of more advanced bounding regions as well as different tightening frequencies of these. While the tightening of bounding rectangles increases query performance, it negatively affects the update performance, which is also very important. Next, periodic, partial reloading of the tree appears worthy of further study. It may also be of interest to include support for transaction time, thus enabling the querying of the past positions of the moving objects as well. This may be achieved by making the tree partially persistent, and it will likely increase the data volume to be indexed by several orders of magnitude.

这项工作指出了几个有趣的研究方向。其中，研究使用更高级的边界区域以及这些区域不同的收紧频率会很有意思。虽然收紧边界矩形可以提高查询性能，但它会对更新性能产生负面影响，而更新性能同样非常重要。接下来，对树进行定期的部分重新加载似乎值得进一步研究。纳入对事务时间的支持也可能会很有意义，这样就可以查询移动对象的过去位置。这可以通过使树具有部分持久性来实现，而且这可能会使需要索引的数据量增加几个数量级。

## Acknowledgments

## 致谢

This research was supported in part by a grant from the Nykredit Corporation; by the Danish Technical Research Council, grant 9700780; by the US National Science Foundation, grant IRI-9610240; and by the CHOROCHRONOS project, funded by the European Commission, contract no. FMRX-CT96-0056.

这项研究部分得到了尼科雷德公司（Nykredit Corporation）的资助；丹麦技术研究委员会（Danish Technical Research Council）的资助（资助编号：9700780）；美国国家科学基金会（US National Science Foundation）的资助（资助编号：IRI - 9610240）；以及由欧盟委员会资助的CHOROCHRONOS项目（合同编号：FMRX - CT96 - 0056）的支持。

References

[1] P. K. Agarwal et al. Efficient Searching with Linear Constraints. In Proc. of the PODS Conf., pp. 169-178 (1998).

[2] P. K. Agarwal, L. Arge, and J. Erickson. Indexing Moving Points. In Proc. of the PODS Conf., to appear (2000).

[3] L. Arge, V. Samoladas, and J. S. Vitter. On Two-Dimensional Indexability and Optimal Range Search Indexing. In Proc. of the PODS Conf., pp. 346-357 (1999).

[4] J. Basch, L. Guibas, and J. Hershberger. Data Structures for Mobile Data. In Proc. of the 8th ACM-SIAM Symposium on Discrete Algorithms, pp. 747-756 (1997).

[5] N. Beckmann, H.-P. Kriegel, R. Schneider, and B. Seeger. The ${\mathrm{R}}^{ * }$ -tree: An Efficient and Robust Access Method for Points and Rectangles. In Proc. of the ACM SIGMOD Conf., pp. 322-331 (1990).

[6] B. Becker et al. An Asymptotically Optimal Multiversion B-Tree. The VLDB Journal 5(4): 264-275 (1996).

[7] R. Bliujūtè, C. S. Jensen, S. Šaltenis, and G. Slivinskas. R-tree Based Indexing of Now-Relative Bitemporal Data. In the Proc. of the 24th VLDB Conf., pp. 345- 356 (1998).

[8] J. Goldstein, R. Ramakrishnan, U. Shaft, and J.-B. Yu. Processing Queries By Linear Constraints. In Proc. of the PODS Conf., pp. 257-267 (1997).

[9] O. Günther and E. Wong. A Dual Approach to Detect Polyhedral Intersections in Arbitrary Dimensions. BIT, 31(1): 3-14 (1991).

[10] J. M. Hellerstein, J. F. Naughton, and A. Pfeffer. Generalized Search Trees for Database Systems. In Proc. of the VLDB Conf., pp. 562-573 (1995).

[11] I. Kamel and C. Faloutsos. On Packing R-trees. In Proc. of the CIKM, pp. 490-499 (1993).

[12] J. Karppinen. Wireless Multimedia Communications: A Nokia View. In Proc. of the Wireless Information Multimedia Communications Symposium, Aalborg University, (November 1999).

[13] G. Kollios, D. Gunopulos, and V. J. Tsotras. On Indexing Mobile Objects. In Proc. of the PODS Conf., pp. 261-272 (1999).

[14] W. Konháuser. Wireless Multimedia Communications: A Siemens View. In Proc. of the Wireless Information Multimedia Communications Symposium, Aalborg University, (November 1999).

[15] A. Kumar, V. J. Tsotras, and C. Faloutsos. Designing Access Methods for Bitemporal Databases. IEEE TKDE, 10(1): 1-20 (1998).

[16] S. T. Leutenegger and M. A. Lopez. The Effect of Buffering on the Performance of R-Trees. In Proc. of the ICDE Conf., pp. 164-171 (1998).

[17] J. Moreira, C. Ribeiro, and J. Saglio. Representation and Manipulation of Moving Points: An Extended Data Model for Location Estimation. Cartography and Geographical Information Systems, to appear.

[18] B.-U. Pagel, H.-W. Six, H. Toben, and P. Widmayer. Towards an Analysis of Range Query Performance in Spatial Data Structures. In Proc. of the PODS Conf., pp. 214-221 (1993).

[19] H. Pedersen. Alting bliver on-line. Børsen Informatik, p. 14, September 28, 1999. (In Danish)

[20] D. Pfoser and C. S. Jensen. Capturing the Uncertainty of Moving-Object Representations. In Proc. of the SSDBM Conf., pp. 111-132 (1999).

[21] D. Pfoser, Y. Theodoridis, and C. S. Jensen. Indexing Trajectories of Moving Point Objects. Chorochronos Tech. Rep. CH-99-3, June 1999.

[22] H. Samet. The Design and Analysis of Spatial Data Structures. Addison-Wesley, Reading, MA, 1990.

[23] S. Šaltenis, C. S. Jensen, S. T. Leutenegger, and M. A. Lopez. Indexing the Positions of Continuously Moving Objects. Technical Report R-99-5009, Department of Computer Science, Aalborg University (1999).

[24] S. Saltenis and C. S. Jensen. R-Tree Based Indexing of General Spatio-Temporal Data. TimeCenter Tech. Rep. TR-45 (1999).

[25] A. Schieder. Wireless Multimedia Communications: An Ericsson View. In Proc. of the Wireless Information Multimedia Communications Symposium, Aalborg University, (November 1999).

[26] J. Tayeb, Ö. Ulusoy, and O. Wolfson. A Quadtree Based Dynamic Attribute Indexing Method. The Computer Journal, 41(3): 185-200 (1998).

[27] O. Wolfson, B. Xu, S. Chamberlain, and L. Jiang. Moving Objects Databases: Issues and Solutions. In Proc. of the SSDBM Conf., pp. 111-122 (1998).

[28] O. Wolfson, A. P. Sistla, S. Chamberlain, and Y. Yesha Updating and Querying Databases that Track Mobile Units. Distributed and Parallel Databases 7(3): 257- 387 (1999).