# Semantics of Ranking Queries for Probabilistic Data and Expected Ranks

# 概率数据排序查询的语义与期望排名

Graham Cormode

格雷厄姆·科莫德

AT&T Labs Research

美国电话电报公司实验室研究中心

Florham Park, NJ, USA

美国新泽西州弗洛勒姆帕克

Feifei Li

李菲菲

Computer Science Department

计算机科学系

FSU, Tallahassee, FL, USA

美国佛罗里达州立大学，塔拉哈西，佛罗里达州，美国

$\mathrm{{Ke}}\mathrm{{Yi}}$

Computer Science & Engineering Department

计算机科学与工程系

HKUST, Hong Kong, China

中国香港科技大学，香港，中国

Abstract- When dealing with massive quantities of data, top- $k$ queries are a powerful technique for returning only the $k$ most relevant tuples for inspection, based on a scoring function. The problem of efficiently answering such ranking queries has been studied and analyzed extensively within traditional database settings. The importance of the top- $k$ is perhaps even greater in probabilistic databases, where a relation can encode exponentially many possible worlds. There have been several recent attempts to propose definitions and algorithms for ranking queries over probabilistic data. However, these all lack many of the intuitive properties of a top- $k$ over deterministic data. Specifically,we define a number of fundamental properties, including exact- $k$ , containment, unique-rank, value-invariance, and stability, which are all satisfied by ranking queries on certain data. We argue that all these conditions should also be fulfilled by any reasonable definition for ranking uncertain data. Unfortunately, none of the existing definitions is able to achieve this.

摘要——在处理大量数据时，前 $k$ 查询是一种强大的技术，它基于评分函数，仅返回 $k$ 个最相关的元组以供检查。在传统数据库环境中，已经对高效回答此类排序查询的问题进行了广泛的研究和分析。在前 $k$ 查询在概率数据库中可能更为重要，因为在概率数据库中，一个关系可以编码指数级数量的可能世界。最近有几项尝试为概率数据的排序查询提出定义和算法。然而，这些方法都缺乏确定性数据上的前 $k$ 查询的许多直观属性。具体来说，我们定义了一些基本属性，包括精确 $k$、包含性、唯一排名、值不变性和稳定性，这些属性在确定数据的排序查询中都能满足。我们认为，任何合理的不确定数据排序定义都应该满足所有这些条件。不幸的是，现有的定义都无法实现这一点。

To remedy this shortcoming, this work proposes an intuitive new approach of expected rank. This uses the well-founded notion of the expected rank of each tuple across all possible worlds as the basis of the ranking. We are able to prove that, in contrast to all existing approaches, the expected rank satisfies all the required properties for a ranking query. We provide efficient solutions to compute this ranking across the major models of uncertain data, such as attribute-level and tuple-level uncertainty. For an uncertain relation of $N$ tuples,the processing cost is $O\left( {N\log N}\right)$ —no worse than simply sorting the relation. In settings where there is a high cost for generating each tuple in turn, we provide pruning techniques based on probabilistic tail bounds that can terminate the search early and guarantee that the top- $k$ has been found. Finally,a comprehensive experimental study confirms the effectiveness of our approach.

为了弥补这一缺陷，本文提出了一种直观的期望排名新方法。该方法以每个元组在所有可能世界中的期望排名这一有充分依据的概念作为排序的基础。我们能够证明，与所有现有方法相比，期望排名满足排序查询的所有必要属性。我们为计算不确定数据的主要模型（如属性级和元组级不确定性）中的这种排名提供了高效的解决方案。对于一个包含 $N$ 个元组的不确定关系，处理成本为 $O\left( {N\log N}\right)$ ——不比简单地对关系进行排序更差。在依次生成每个元组成本较高的情况下，我们基于概率尾界提供了剪枝技术，该技术可以提前终止搜索，并保证已经找到了前 $k$ 个元组。最后，全面的实验研究证实了我们方法的有效性。

## I. INTRODUCTION

## 一、引言

Ranking queries are a powerful concept in focusing attention on the most important answers to a query. To deal with massive quantities of data, such as multimedia search, streaming data, web data and distributed systems, tuples from the underlying database are ranked by a score, usually computed based on a user-defined scoring function. Only the top- $k$ tuples with the highest scores are returned for further inspection. Following the seminal work by Fagin et al. [13], such queries have received considerable attention in traditional relational databases, including [23], [19], [36] and many others. See the excellent survey by Ilyas et al. [20] for a more complete overview of the many important studies in this area.

排序查询是一种强大的概念，可将注意力集中在查询的最重要答案上。为了处理大量数据，如多媒体搜索、流数据、网络数据和分布式系统，底层数据库中的元组通常根据用户定义的评分函数计算的分数进行排序。仅返回得分最高的前 $k$ 个元组以供进一步检查。继法金等人的开创性工作 [13] 之后，此类查询在传统关系数据库中受到了相当多的关注，包括 [23]、[19]、[36] 等。有关该领域许多重要研究的更完整概述，请参阅伊利亚斯等人的优秀综述 [20]。

Within these motivating application domains-distributed, streaming, web and multimedia applications-data arrives in massive quantities, underlining the need for ordering by score. But an additional challenge is that the data is also typically inherently fuzzy or uncertain. For instance, multimedia and unstructured web data frequently require data integration or schema mapping [15], [7], [16]. Data items in the output of such operations are usually associated with a confidence, reflecting how well they are matched with other records from different data sources. In applications that handle measurement data, e.g., sensor readings and distances to a query point, the data is inherently noisy, and is better represented by a probability distribution rather than a single deterministic value [9], [11]. In recognition of this aspect of the data, there have been significant research efforts devoted to producing probabilistic database management systems, which can represent and manage data with explicit probabilistic models of uncertainty. The notable examples of such systems include MystiQ [10], Trio [1], and MayBMS [2].

在这些具有启发性的应用领域（分布式、流、网络和多媒体应用）中，数据大量到来，凸显了按分数排序的必要性。但另一个挑战是，数据通常本质上是模糊或不确定的。例如，多媒体和非结构化网络数据经常需要进行数据集成或模式映射 [15]、[7]、[16]。此类操作输出中的数据项通常与一个置信度相关联，反映了它们与来自不同数据源的其他记录的匹配程度。在处理测量数据（如传感器读数和到查询点的距离）的应用中，数据本质上是有噪声的，用概率分布而不是单个确定性值来表示更好 [9]、[11]。认识到数据的这一方面，已经有大量的研究工作致力于开发概率数据库管理系统，这些系统可以用显式的不确定性概率模型来表示和管理数据。此类系统的显著例子包括 MystiQ [10]、Trio [1] 和 MayBMS [2]。

With a probabilistic database, it is possible to represent a huge number of possible (deterministic) realizations of the (probabilistic) data-an exponential blow-up from the size of the relation representing the data. A key problem in such databases is how to extend the familiar semantics of the top- $k$ query to this setting,and how to answer such queries efficiently. To this end, there has been several recent works outlining possible definitions, and associated algorithms. Ré et al. [28] base their ranking on the confidence associated with each query result. Soliman et al. [33] extend the semantics of ranking queries from certain data and study the problem of ranking tuples when there is both a score and probability for each tuple. Subsequently, there have been several other approaches to ranking based on combining score and likelihood [39], [34], [37], [18] (discussed in detail in Section III-B).

使用概率数据库，可以表示（概率）数据的大量可能（确定性）实现——从表示数据的关系的大小来看是指数级的增长。此类数据库中的一个关键问题是如何将熟悉的前 $k$ 查询的语义扩展到这种情况，以及如何高效地回答此类查询。为此，最近有几项工作概述了可能的定义和相关算法。雷等人 [28] 基于与每个查询结果相关联的置信度进行排序。索利曼等人 [33] 扩展了确定数据的排序查询的语义，并研究了每个元组既有分数又有概率时的元组排序问题。随后，还有其他几种基于结合分数和似然性的排序方法 [39]、[34]、[37]、[18]（在第三节 B 中详细讨论）。

For certain data with a single score value, there is a clear total ordering based on score from which the top- $k$ is derived, which leads to a clean and intuitive semantics. This is particularly natural, by analogy with the many occurrences of top- $k$ lists in daily life: movies ranked by box-office receipts, athletes ranked by race times, researchers ranked by number of publications (or other metrics), and so on. With uncertain data, there are two distinct orders to work with: ordering by score, and ordering by probability. There are many possible ways of combining these two, leading to quite different results, as evidenced by the multiple definitions that have been proposed in the literature,such as U-Topk [33], U-kRanks [33], Global-Top $k$ [39] and PT- $k$ [18]. In choosing a definition to work with,

对于具有单一得分值的特定数据，存在一种基于得分的明确全序关系，由此可以得出前 $k$ 项，这带来了清晰直观的语义。通过与日常生活中许多前 $k$ 列表的情况进行类比，这一点尤其自然：按票房收入排名的电影、按比赛用时排名的运动员、按发表论文数量（或其他指标）排名的研究人员等等。对于不确定数据，有两种不同的排序方式可供使用：按得分排序和按概率排序。有许多可能的方法来结合这两种排序方式，从而导致截然不同的结果，文献中提出的多种定义就证明了这一点，例如 U - Topk [33]、U - kRanks [33]、Global - Top $k$ [39] 和 PT - $k$ [18]。在选择使用的定义时，

- computer

- 计算机

Society we must ask, what are the conditions that we want the resulting query answer to satisfy. We address this issue following a principled approach and return to the properties of ranking queries on certain data. We define the following properties that should hold on the output of such a ranking query:

在社会中，我们必须问，我们希望得到的查询答案满足哪些条件。我们采用一种有原则的方法来解决这个问题，并回顾特定数据上排名查询的属性。我们定义了以下排名查询输出应满足的属性：

- Exact- $k$ : The top- $k$ list should contain exactly $k$ items;

- 精确 $k$：前 $k$ 列表应恰好包含 $k$ 个项目；

- Containment: The top- $\left( {k + 1}\right)$ list should contain all items in the top- $k$ ;

- 包含性：前 $\left( {k + 1}\right)$ 列表应包含前 $k$ 中的所有项目；

- Unique-ranking: Within the top- $k$ ,each reported item should be assigned exactly one position: the same item should not be listed multiple times within the top- $k$ .

- 唯一排名：在前 $k$ 中，每个报告的项目应被精确分配一个位置：同一个项目不应在前 $k$ 中多次列出。

- Value-invariance: The scores only determine the relative behavior of the tuples: changing the score values without altering the relative ordering should not change the top- $k$ ;

- 值不变性：得分仅决定元组的相对行为：在不改变相对顺序的情况下改变得分值不应改变前 $k$；

- Stability: Making an item in the top- $k$ list more likely or more important should not remove it from the list.

- 稳定性：使前 $k$ 列表中的一个项目更有可能出现或更重要，不应将其从列表中移除。

We define these properties more formally in Section III-A.

我们将在第三节 A 中更正式地定义这些属性。

These properties are clearly satisfied for certain data, and capture much of our intuition on how a "ranking" query should behave. Moreover, they should seem intuitive and natural (indeed, they should appear almost obvious). A general axiom of work on extending data management from certain data to the uncertain domain has been that basic properties of query semantics should be preserved to the extent possible [10], [4]. But, as we subsequently demonstrate, none of the prior works on ranking queries for probabilistic data satisfies all of these "obvious" properties. Lastly, we note that prior work stated results primarily in the tuple-level uncertainty model [1], [10]; here, we show our results for both the tuple-level and attribute-level uncertainty models [9], [35].

这些属性对于特定数据显然是满足的，并且体现了我们对“排名”查询应如何表现的直觉。此外，它们应该看起来直观自然（实际上，它们几乎应该是显而易见的）。将数据管理从特定数据扩展到不确定领域的工作的一个一般公理是，查询语义的基本属性应尽可能得到保留 [10]，[4]。但是，正如我们随后将证明的，之前关于概率数据排名查询的工作都没有满足所有这些“显而易见”的属性。最后，我们注意到之前的工作主要在元组级不确定性模型中陈述结果 [1]，[10]；在这里，我们展示了我们在元组级和属性级不确定性模型 [9]，[35] 下的结果。

Our contributions. To remedy the shortcomings we identify, this work proposes an intuitive new approach for ranking based on expected rank. It uses the well-founded notion of the expected value of the rank of each tuple across all possible worlds as the basis of the ranking. We are able to prove that, in contrast to all existing approaches, the expected rank satisfies all the required properties for a ranking query across major models of uncertain data. Furthermore, these nice properties do not come at a price of higher computational costs. On the contrary,we design efficient $O\left( {N\log N}\right)$ -time exact algorithms to compute under both the attribute-level model and the tuple-level model, while most of the previous top- $k$ definitions rely on dynamic programming and require $\Omega \left( {N}^{2}\right)$ time to compute the results exactly,and errors have to be tolerated if one wants to process the queries faster by using random sampling or other approximation techniques [17]. In summary, our contributions are the followings:

我们的贡献。为了弥补我们发现的不足，这项工作提出了一种基于期望排名的直观新方法。它使用每个元组在所有可能世界中的排名期望值这一有充分依据的概念作为排名的基础。我们能够证明，与所有现有方法相比，期望排名在不确定数据的主要模型中满足排名查询的所有必要属性。此外，这些良好的属性并不会带来更高的计算成本。相反，我们设计了高效的 $O\left( {N\log N}\right)$ 时间精确算法，用于在属性级模型和元组级模型下进行计算，而之前的大多数前 $k$ 定义依赖于动态规划，需要 $\Omega \left( {N}^{2}\right)$ 时间来精确计算结果，如果想通过随机采样或其他近似技术更快地处理查询，则必须容忍误差 [17]。总之，我们的贡献如下：

- We formalize the necessary semantics of ranking queries in certain data and migrate them to probabilistic data (Section III-A), and show that no existing approaches for this problem achieve all these properties (Section III-B).

- 我们形式化了特定数据中排名查询的必要语义，并将其迁移到概率数据中（第三节 A），并表明现有的解决此问题的方法都无法实现所有这些属性（第三节 B）。

- We propose a new approach based on the expected rank of each tuple across all possible worlds that provably satisfies these requirements. The expected rank definition works seamlessly with both the attribute-level and tuple-level uncertainty models (Section III-C).

- 我们提出了一种基于每个元组在所有可能世界中的期望排名的新方法，该方法被证明满足这些要求。期望排名定义在属性级和元组级不确定性模型中都能无缝工作（第三节 C）。

- We provide efficient algorithms for expected ranks in both models. For an uncertain relation of $N$ tuples, the processing cost of our approach is $O\left( {N\log N}\right)$ . In settings where there is a high cost for accessing tuples, we show pruning techniques based on probabilistic tail bounds that can terminate the search early and guarantee that the top- $k$ has been found (Section IV and V).

- 我们为这两种模型提供了期望排名的高效算法。对于一个包含 $N$ 个元组的不确定关系，我们方法的处理成本是 $O\left( {N\log N}\right)$。在访问元组成本较高的情况下，我们展示了基于概率尾部界限的剪枝技术，该技术可以提前终止搜索并保证找到前 $k$（第四节和第五节）。

- We present a comprehensive experimental study that confirms the effectiveness of our approach (Section VII).

- 我们进行了全面的实验研究，证实了我们方法的有效性（第七节）。

## II. UNCERTAIN DATA MODELS W.R.T RANKING QUERIES

## 二、关于排名查询的不确定数据模型

Many models for describing uncertain data have been presented in the literature. The work by Sarma et al. [29] describes the main features and contrasts their properties and descriptive ability. Each model describes a probability distribution over possible worlds, where each possible world corresponds to a single deterministic data instance. The most expressive approach is to explicitly list each possible world and its associated probability; such a method is referred to as complete, as it can capture all possible correlations. However, complete models are very costly to describe and manipulate since there can be exponentially many combinations of tuples each generating a distinct possible world [29].

文献中已经提出了许多用于描述不确定数据的模型。Sarma等人的研究[29]描述了这些模型的主要特征，并对比了它们的性质和描述能力。每个模型都描述了可能世界上的概率分布，其中每个可能世界对应一个单一的确定性数据实例。最具表达力的方法是明确列出每个可能世界及其相关概率；这种方法被称为完备方法，因为它可以捕捉所有可能的相关性。然而，完备模型在描述和处理时成本非常高，因为元组的组合数量可能呈指数级增长，每个组合都会产生一个不同的可能世界[29]。

Typically, we are able to make certain independence assumptions, that unless correlations are explicitly described, events are assumed to be independent. Consequently, likelihoods can be computed using standard probability calculations (i.e. multiplication of probabilities of independent events). The strongest independence assumptions lead to the basic model, where each tuple has a probability of occurrence, and all tuples are assumed fully independent of each other. This is typically too strong an assumption, and so intermediate models allow the description of simple correlations between tuples. This extends the expressiveness of the models, while keeping computations of probability tractable. We consider two models that have been used frequently within the database community. In our discussion, without loss of generality, a probabilistic database contains simply one relation.

通常，我们能够做出某些独立性假设，即除非明确描述了相关性，否则事件被假定为相互独立。因此，可以使用标准的概率计算方法（即独立事件概率的乘法）来计算可能性。最强的独立性假设产生了基本模型，在该模型中，每个元组都有一个出现概率，并且所有元组都被假定为完全相互独立。这通常是一个过强的假设，因此中间模型允许描述元组之间的简单相关性。这在保持概率计算可处理的同时，扩展了模型的表达能力。我们考虑数据库领域中经常使用的两种模型。在我们的讨论中，不失一般性，假设一个概率数据库仅包含一个关系。

Attribute-level uncertainty model. In this model, the probabilistic database is a table of $N$ tuples. Each tuple has one attribute whose value is uncertain (together with other certain attributes). This uncertain attribute has a discrete pdf describing its value distribution. When instantiating this uncertain relation to a certain instance, each tuple draws a value for its uncertain attribute based on the associated discrete pdf and the choice is independent among tuples. This model has many practical applications such as sensor readings [22], [11], spatial objects with fuzzy locations [35], [9], [5], [26], [25], etc. More important, it is very easy to represent this model using the traditional, relational database, as observed by Antova et al. [3]. For the purpose of ranking queries, the important case is when the uncertain attribute represents the score for the tuple, and we would like to rank the tuples based on this score attribute. Let ${X}_{i}$ be the random variable denoting the score of tuple ${t}_{i}$ . We assume that ${X}_{i}$ has a discrete pdf with bounded size. This is a realistic assumption for many practical applications, including movie ratings [10], and string matching [7]. The general, continuous pdf case is discussed briefly in Section VI. In this model we are essentially ranking the set of independent random variables ${X}_{1},\ldots ,{X}_{N}$ . A relation following this model is illustrated in Figure 1. For tuple ${t}_{i}$ ,the score takes the value ${v}_{i,j}$ with probability ${p}_{i,j}$ for $1 \leq  j \leq  {s}_{i}$ .

属性级不确定性模型。在这个模型中，概率数据库是一个包含$N$个元组的表。每个元组都有一个属性的值是不确定的（以及其他确定的属性）。这个不确定属性有一个离散概率密度函数（pdf）来描述其值的分布。当将这个不确定关系实例化为一个确定的实例时，每个元组根据相关的离散概率密度函数为其不确定属性抽取一个值，并且元组之间的选择是相互独立的。这个模型有许多实际应用，如传感器读数[22]、[11]，具有模糊位置的空间对象[35]、[9]、[5]、[26]、[25]等。更重要的是，正如Antova等人所指出的[3]，使用传统的关系数据库来表示这个模型非常容易。对于排序查询而言，重要的情况是当不确定属性表示元组的得分时，我们希望根据这个得分属性对元组进行排序。设${X}_{i}$为表示元组${t}_{i}$得分的随机变量。我们假设${X}_{i}$有一个规模有界的离散概率密度函数。对于许多实际应用，包括电影评分[10]和字符串匹配[7]，这是一个现实的假设。一般的连续概率密度函数情况将在第六节中简要讨论。在这个模型中，我们本质上是对独立随机变量集合${X}_{1},\ldots ,{X}_{N}$进行排序。遵循这个模型的关系如图1所示。对于元组${t}_{i}$，得分取值为${v}_{i,j}$的概率为${p}_{i,j}$，其中$1 \leq  j \leq  {s}_{i}$。

<!-- Media -->

<table><tr><td/><td>score</td></tr><tr><td/><td>$\left\{  {\left( {{v}_{1,1},{p}_{1,1}}\right) ,\left( {{v}_{1,2},{p}_{1,2}}\right) ,\ldots ,\left( {{v}_{1,{s}_{1}},{p}_{1,{s}_{1}}}\right) }\right\}$</td></tr><tr><td>${t}_{2}$</td><td>$\left\{  {\left( {{v}_{2,1},{p}_{2,1}}\right) ,\ldots ,{v}_{2,{s}_{2}},{p}_{2,{s}_{2}}}\right\}$</td></tr><tr><td>$\vdots$</td><td/></tr><tr><td>${t}_{N}$</td><td>$\left\{  {\left( {{v}_{N,1},{p}_{N,1}}\right) ,\ldots ,\left( {{v}_{N,{s}_{N}},{p}_{N,{s}_{N}}}\right) }\right\}$</td></tr></table>

<table><tbody><tr><td></td><td>分数；得分</td></tr><tr><td></td><td>$\left\{  {\left( {{v}_{1,1},{p}_{1,1}}\right) ,\left( {{v}_{1,2},{p}_{1,2}}\right) ,\ldots ,\left( {{v}_{1,{s}_{1}},{p}_{1,{s}_{1}}}\right) }\right\}$</td></tr><tr><td>${t}_{2}$</td><td>$\left\{  {\left( {{v}_{2,1},{p}_{2,1}}\right) ,\ldots ,{v}_{2,{s}_{2}},{p}_{2,{s}_{2}}}\right\}$</td></tr><tr><td>$\vdots$</td><td></td></tr><tr><td>${t}_{N}$</td><td>$\left\{  {\left( {{v}_{N,1},{p}_{N,1}}\right) ,\ldots ,\left( {{v}_{N,{s}_{N}},{p}_{N,{s}_{N}}}\right) }\right\}$</td></tr></tbody></table>

Fig. 1. Attribute-level uncertainty model.

图1. 属性级不确定性模型。

<table><tr><td>tuples</td><td>score</td></tr><tr><td>${t}_{1}$</td><td>$\{ \left( {{100},{0.4}}\right) ,\left( {{70},{0.6}}\right) \}$</td></tr><tr><td>${t}_{2}$</td><td>$\{ \left( {{92},{0.6}}\right) ,\left( {{80},{0.4}}\right) \}$</td></tr><tr><td>${t}_{3}$</td><td>$\{ \left( {{85},1}\right) \}$</td></tr></table>

<table><tbody><tr><td>元组</td><td>得分</td></tr><tr><td>${t}_{1}$</td><td>$\{ \left( {{100},{0.4}}\right) ,\left( {{70},{0.6}}\right) \}$</td></tr><tr><td>${t}_{2}$</td><td>$\{ \left( {{92},{0.6}}\right) ,\left( {{80},{0.4}}\right) \}$</td></tr><tr><td>${t}_{3}$</td><td>$\{ \left( {{85},1}\right) \}$</td></tr></tbody></table>

<table><tr><td>world $W$</td><td>$\Pr \left\lbrack  W\right\rbrack$</td></tr><tr><td>$\left\{  {{t}_{1} = {100},{t}_{2} = {92},{t}_{3} = {85}}\right\}$</td><td>${0.4} \times  {0.6} \times  1 = {0.24}$</td></tr><tr><td>$\left\{  {{t}_{1} = {100},{t}_{3} = {85},{t}_{2} = {80}}\right\}$</td><td>${0.4} \times  {0.4} \times  1 = {0.16}$</td></tr><tr><td>$\left\{  {{t}_{2} = {92},{t}_{3} = {85},{t}_{1} = {70}}\right\}$</td><td>${0.6} \times  {0.6} \times  1 = {0.36}$</td></tr><tr><td>$\left\{  {{t}_{3} = {85},{t}_{2} = {80},{t}_{1} = {70}}\right\}$</td><td>${0.6} \times  {0.4} \times  1 = {0.24}$</td></tr></table>

<table><tbody><tr><td>世界 $W$</td><td>$\Pr \left\lbrack  W\right\rbrack$</td></tr><tr><td>$\left\{  {{t}_{1} = {100},{t}_{2} = {92},{t}_{3} = {85}}\right\}$</td><td>${0.4} \times  {0.6} \times  1 = {0.24}$</td></tr><tr><td>$\left\{  {{t}_{1} = {100},{t}_{3} = {85},{t}_{2} = {80}}\right\}$</td><td>${0.4} \times  {0.4} \times  1 = {0.16}$</td></tr><tr><td>$\left\{  {{t}_{2} = {92},{t}_{3} = {85},{t}_{1} = {70}}\right\}$</td><td>${0.6} \times  {0.6} \times  1 = {0.36}$</td></tr><tr><td>$\left\{  {{t}_{3} = {85},{t}_{2} = {80},{t}_{1} = {70}}\right\}$</td><td>${0.6} \times  {0.4} \times  1 = {0.24}$</td></tr></tbody></table>

Fig. 2. An example of possible worlds for attribute-level uncertainty model.

图2. 属性级不确定性模型的可能世界示例。

<!-- Media -->

Tuple-level uncertainty model. In the second model, the attributes of each tuple are fixed, but the entire tuple may or may not appear. In the basic model,each tuple $t$ appears with probability $p\left( t\right)$ independently. In more complex models,there are dependencies among the tuples, which can be specified by a set of generation rules. These can be in the form of $x$ -relations [1],[4],complex events [10],or other forms.

元组级不确定性模型。在第二个模型中，每个元组的属性是固定的，但整个元组可能出现也可能不出现。在基本模型中，每个元组$t$以概率$p\left( t\right)$独立出现。在更复杂的模型中，元组之间存在依赖关系，这些依赖关系可以通过一组生成规则来指定。这些规则可以采用$x$ -关系[1]、[4]、复杂事件[10]或其他形式。

All previous work concerned with ranking queries in uncertain data has focused on the tuple-level uncertainty model with exclusion rules [18], [33], [39], [37] where each tuple appears in a single rule $\tau$ . Arbitrary generation rules have been discussed in [33], [34], but they have been shown to require exponential processing complexity [18], [37]. Hence, as with many other works in the literature [33], [18], [37], [38], we primarily consider exclusion rules in this model, where each exclusion rule has a constant number of choices. In addition, each tuple appears in at most one rule. The total probability for all tuples in one rule must be less or equal than one, so that it can be properly interpreted as a probability distribution. To simplify our discussion, we allow rules containing only one tuple and require that all tuples must appear in one of the rules. This is essentially equivalent to the popular x-relations model [1]. This tuple-level uncertainty model is a good fit for applications where it is important to capture the correlations between tuples; this model has been used to fit a large number of real-life examples [4], [10], [33], [18], [38]. An example of a relation in this uncertainty model is shown in Figure 3. This relation has $N$ tuples and $M$ rules. The second rule says that ${t}_{2}$ and ${t}_{4}$ cannot appear together in any certain instance of this relation. It also constrains that $p\left( {t}_{2}\right)  + p\left( {t}_{4}\right)  \leq  1$ .

以往所有关于不确定数据中排名查询的工作都集中在具有排除规则的元组级不确定性模型上[18]、[33]、[39]、[37]，其中每个元组出现在单个规则$\tau$中。任意生成规则已在[33]、[34]中进行了讨论，但已证明它们需要指数级的处理复杂度[18]、[37]。因此，与文献中的许多其他工作[33]、[18]、[37]、[38]一样，我们在这个模型中主要考虑排除规则，其中每个排除规则有固定数量的选择。此外，每个元组最多出现在一个规则中。一个规则中所有元组的总概率必须小于或等于1，以便可以将其正确解释为概率分布。为了简化我们的讨论，我们允许规则只包含一个元组，并要求所有元组必须出现在其中一个规则中。这本质上等同于流行的x -关系模型[1]。这个元组级不确定性模型非常适合需要捕捉元组之间相关性的应用；该模型已被用于拟合大量现实生活中的例子[4]、[10]、[33]、[18]、[38]。这个不确定性模型中的一个关系示例如图3所示。这个关系有$N$个元组和$M$条规则。第二条规则表明，${t}_{2}$和${t}_{4}$不能同时出现在这个关系的任何确定实例中。它还限制了$p\left( {t}_{2}\right)  + p\left( {t}_{4}\right)  \leq  1$。

<!-- Media -->

<table><tr><td>tuples</td><td>score</td><td/></tr><tr><td>${t}_{1}$</td><td>${v}_{1}$</td><td>$p\left( {t}_{1}\right)$</td></tr><tr><td>${t}_{2}$</td><td>${v}_{2}$</td><td>$p\left( {t}_{2}\right)$</td></tr><tr><td>$\vdots$</td><td>- $\vdots$</td><td/></tr><tr><td>${t}_{N}$</td><td>${UN}$</td><td>$p\left( {t}_{N}\right)$</td></tr></table>

<table><tbody><tr><td>元组</td><td>得分</td><td></td></tr><tr><td>${t}_{1}$</td><td>${v}_{1}$</td><td>$p\left( {t}_{1}\right)$</td></tr><tr><td>${t}_{2}$</td><td>${v}_{2}$</td><td>$p\left( {t}_{2}\right)$</td></tr><tr><td>$\vdots$</td><td>- $\vdots$</td><td></td></tr><tr><td>${t}_{N}$</td><td>${UN}$</td><td>$p\left( {t}_{N}\right)$</td></tr></tbody></table>

<table><tr><td colspan="2">rules</td></tr><tr><td>${\tau }_{1}$</td><td>$\left\{  {t}_{1}\right\}$</td></tr><tr><td>${\tau }_{2}$</td><td>$\left\{  {{t}_{2},{t}_{4}}\right\}$</td></tr><tr><td>$\vdots$ ${\tau }_{M}$</td><td>$\vdots$ $\left\{  {{t}_{5},{t}_{8},{t}_{N}}\right\}$</td></tr></table>

<table><tbody><tr><td colspan="2">规则</td></tr><tr><td>${\tau }_{1}$</td><td>$\left\{  {t}_{1}\right\}$</td></tr><tr><td>${\tau }_{2}$</td><td>$\left\{  {{t}_{2},{t}_{4}}\right\}$</td></tr><tr><td>$\vdots$ ${\tau }_{M}$</td><td>$\vdots$ $\left\{  {{t}_{5},{t}_{8},{t}_{N}}\right\}$</td></tr></tbody></table>

Fig. 3. Tuple-level uncertainty model.

图3. 元组级不确定性模型。

<table><tr><td>tuples</td><td>score</td><td>$p\left( t\right)$</td></tr><tr><td>${t}_{1}$</td><td>100</td><td>0.4</td></tr><tr><td>${t}_{2}$</td><td>92</td><td>0.5</td></tr><tr><td>${t}_{3}$</td><td>80</td><td>1</td></tr><tr><td>${t}_{4}$</td><td>70</td><td>0.5</td></tr></table>

<table><tbody><tr><td>元组</td><td>得分</td><td>$p\left( t\right)$</td></tr><tr><td>${t}_{1}$</td><td>100</td><td>0.4</td></tr><tr><td>${t}_{2}$</td><td>92</td><td>0.5</td></tr><tr><td>${t}_{3}$</td><td>80</td><td>1</td></tr><tr><td>${t}_{4}$</td><td>70</td><td>0.5</td></tr></tbody></table>

<table><tr><td colspan="2">rules</td></tr><tr><td>${\tau }_{1}$</td><td>$\left\{  {t}_{1}\right\}$</td></tr><tr><td>${\tau }_{2}$</td><td>$\left\{  {{t}_{2},{t}_{4}}\right\}$</td></tr><tr><td>${\tau }_{3}$</td><td>$\left\{  {t}_{3}\right\}$</td></tr></table>

<table><tbody><tr><td colspan="2">规则</td></tr><tr><td>${\tau }_{1}$</td><td>$\left\{  {t}_{1}\right\}$</td></tr><tr><td>${\tau }_{2}$</td><td>$\left\{  {{t}_{2},{t}_{4}}\right\}$</td></tr><tr><td>${\tau }_{3}$</td><td>$\left\{  {t}_{3}\right\}$</td></tr></tbody></table>

<table><tr><td>world $W$</td><td>$\Pr \left\lbrack  W\right\rbrack$</td></tr><tr><td>$\left\{  {{t}_{1},{t}_{2},{t}_{3}}\right\}$</td><td>$p\left( {t}_{1}\right) p\left( {t}_{2}\right) p\left( {t}_{3}\right)  = {0.2}$</td></tr><tr><td>$\left\{  {{t}_{1},{t}_{3},{t}_{4}}\right\}$</td><td>$p\left( {t}_{1}\right) p\left( {t}_{3}\right) p\left( {t}_{4}\right)  = {0.2}$</td></tr><tr><td>$\left\{  {{t}_{2},{t}_{3}}\right\}$</td><td>$\left( {1 - p\left( {t}_{1}\right) }\right) p\left( {t}_{2}\right) p\left( {t}_{3}\right)  = {0.3}$</td></tr><tr><td>$\left\{  {{t}_{3},{t}_{4}}\right\}$</td><td>$\left( {1 - p\left( {t}_{1}\right) }\right) p\left( {t}_{3}\right) p\left( {t}_{4}\right)  = {0.3}$</td></tr></table>

<table><tbody><tr><td>世界 $W$</td><td>$\Pr \left\lbrack  W\right\rbrack$</td></tr><tr><td>$\left\{  {{t}_{1},{t}_{2},{t}_{3}}\right\}$</td><td>$p\left( {t}_{1}\right) p\left( {t}_{2}\right) p\left( {t}_{3}\right)  = {0.2}$</td></tr><tr><td>$\left\{  {{t}_{1},{t}_{3},{t}_{4}}\right\}$</td><td>$p\left( {t}_{1}\right) p\left( {t}_{3}\right) p\left( {t}_{4}\right)  = {0.2}$</td></tr><tr><td>$\left\{  {{t}_{2},{t}_{3}}\right\}$</td><td>$\left( {1 - p\left( {t}_{1}\right) }\right) p\left( {t}_{2}\right) p\left( {t}_{3}\right)  = {0.3}$</td></tr><tr><td>$\left\{  {{t}_{3},{t}_{4}}\right\}$</td><td>$\left( {1 - p\left( {t}_{1}\right) }\right) p\left( {t}_{3}\right) p\left( {t}_{4}\right)  = {0.3}$</td></tr></tbody></table>

Fig. 4. An example of possible worlds for tuple-level uncertainty model.

图4. 元组级不确定性模型的可能世界示例。

<!-- Media -->

The possible world semantics. We denote the uncertain relation as $\mathcal{D}$ . In the attribute-level uncertainty model,an uncertain relation is instantiated into a possible world by taking one independent value for each tuple's uncertain attribute according to its distribution. Denote a possible world as $W$ and the value for ${t}_{i}$ ’s uncertain attribute in $W$ as ${w}_{{t}_{i}}$ . In the attribute-level uncertainty model,the probability that $W$ occurs is $\Pr \left\lbrack  W\right\rbrack   = \mathop{\prod }\limits_{{j = 1}}^{N}{p}_{j,x}$ ,where $x$ satisfies ${v}_{j,x} = {w}_{{t}_{j}}$ . It is worth mentioning that in the attribute-level case we always have $\forall W \in  \mathcal{W},\left| W\right|  = N$ ,where $\mathcal{W}$ is the space of all the possible worlds. The example in Figure 2 illustrates the possible worlds for an uncertain relation in this model.

可能世界语义。我们将不确定关系表示为 $\mathcal{D}$ 。在属性级不确定性模型中，通过根据每个元组的不确定属性的分布为其选取一个独立值，将不确定关系实例化为一个可能世界。将一个可能世界表示为 $W$ ，并将 ${t}_{i}$ 在 $W$ 中的不确定属性的值表示为 ${w}_{{t}_{i}}$ 。在属性级不确定性模型中， $W$ 出现的概率为 $\Pr \left\lbrack  W\right\rbrack   = \mathop{\prod }\limits_{{j = 1}}^{N}{p}_{j,x}$ ，其中 $x$ 满足 ${v}_{j,x} = {w}_{{t}_{j}}$ 。值得一提的是，在属性级的情况下，我们始终有 $\forall W \in  \mathcal{W},\left| W\right|  = N$ ，其中 $\mathcal{W}$ 是所有可能世界的空间。图2中的示例展示了该模型中不确定关系的可能世界。

For the tuple-level uncertainty model,a possible world $W$ from $\mathcal{W}$ is now a subset of tuples from the uncertain relation $\mathcal{D}$ . The probability of $W$ occurring is $\Pr \left\lbrack  W\right\rbrack   = \mathop{\prod }\limits_{{j = 1}}^{M}{p}_{W}\left( {\tau }_{j}\right)$ , where for any $\tau  \in  \mathcal{D},{p}_{W}\left( \tau \right)$ is defined as

对于元组级不确定性模型，来自 $\mathcal{W}$ 的一个可能世界 $W$ 现在是不确定关系 $\mathcal{D}$ 的元组子集。 $W$ 出现的概率为 $\Pr \left\lbrack  W\right\rbrack   = \mathop{\prod }\limits_{{j = 1}}^{M}{p}_{W}\left( {\tau }_{j}\right)$ ，其中对于任何 $\tau  \in  \mathcal{D},{p}_{W}\left( \tau \right)$ 定义如下

$$
{p}_{W}\left( \tau \right)  = \left\{  \begin{array}{ll} p\left( t\right) , & \text{ if }\tau  \cap  W = \{ t\} \\  1 - \mathop{\sum }\limits_{{{t}_{i} \in  \tau }}p\left( {t}_{i}\right) , & \text{ if }\tau  \cap  W = \varnothing ; \\  0, & \text{ otherwise. } \end{array}\right. 
$$

A notable difference for the tuple-level uncertain model is that given a random possible world $W$ ,not all tuples from $\mathcal{D}$ will appear. Hence, the size of the possible world can range from 0 to $N$ . The example in Figure 4 illustrates the possible worlds for an uncertain relation in this model.

元组级不确定模型的一个显著区别是，给定一个随机的可能世界 $W$ ，并非 $\mathcal{D}$ 中的所有元组都会出现。因此，可能世界的大小范围可以从0到 $N$ 。图4中的示例展示了该模型中不确定关系的可能世界。

We iterate that every uncertain data model can be seen as a succinct description of a distribution over possible worlds $\mathcal{W}$ . Each possible world is a certain table on which we can evaluate any traditional query. The focus of uncertain query processing is (1) how to "combine" the query results from all the possible worlds into a meaningful result for the query; and (2) how to process such a combination efficiently without explicitly materializing the exponentially many possible worlds.

我们重申，每个不确定数据模型都可以看作是对可能世界 $\mathcal{W}$ 上分布的简洁描述。每个可能世界都是一个确定的表，我们可以在其上评估任何传统查询。不确定查询处理的重点是：（1）如何将所有可能世界的查询结果“组合”成一个有意义的查询结果；（2）如何在不显式实例化指数级数量的可能世界的情况下高效地处理这种组合。

Difference of the two models under ranking queries. We would like to emphasize that there is a significant difference for the two models in the context of ranking tuples. More specifically, the semantic of ranking queries in uncertain databases is to derive a meaningful ordering for all tuples in the database $\mathcal{D}$ . Note that this is not equivalent to deriving an ordering for all values that tuples in $\mathcal{D}$ may take. In the attribute-level model,all tuples in $\mathcal{D}$ will participate in the ranking process in every possible world. In contrast, in the tuple-level model only a subset of tuples in $\mathcal{D}$ will participate in the ranking process for a given possible world.

两种模型在排名查询下的差异。我们想强调的是，在对元组进行排名的情况下，这两种模型存在显著差异。更具体地说，不确定数据库中排名查询的语义是为数据库 $\mathcal{D}$ 中的所有元组推导出一个有意义的排序。请注意，这并不等同于为 $\mathcal{D}$ 中的元组可能取的所有值推导出一个排序。在属性级模型中， $\mathcal{D}$ 中的所有元组都会在每个可能世界的排名过程中参与。相比之下，在元组级模型中，对于给定的可能世界，只有 $\mathcal{D}$ 中的一部分元组会参与排名过程。

## III. RANKING QUERY SEMANTICS

## 三、排名查询语义

## A. Properties of Ranking Queries

## A. 排名查询的属性

We now define a set of properties for ranking tuples. These are chosen to describe the key properties of ranking certain data, and hence to give properties which a user would naturally expect of a ranking over uncertain data to have.

我们现在为元组排名定义一组属性。选择这些属性是为了描述对确定数据进行排名的关键属性，从而给出用户自然期望的对不确定数据进行排名应具有的属性。

The first property is very natural, and is also used in [39].

第一个属性非常自然，并且也在文献[39]中使用。

Definition 1 (Exact- $k$ ): Let ${R}_{k}$ be the set of tuples (associated with their ranks) in the top- $k$ query result. If $\left| \mathcal{D}\right|  \geq  k$ , then $\left| {R}_{k}\right|  = k$ .

定义1（精确 - $k$ ）：设 ${R}_{k}$ 是前 $k$ 查询结果中的元组集合（与其排名相关联）。如果 $\left| \mathcal{D}\right|  \geq  k$ ，则 $\left| {R}_{k}\right|  = k$ 。

The second property captures the intuition that if an item is in the top- $k$ ,it should be in the top- ${k}^{\prime }$ for any ${k}^{\prime } > k$ . Equivalently,the choice of $k$ is simply a slider that chooses how many results are to be returned to the user, and changing $k$ should only change the number of results returned,not the underlying set of results.

第二个属性体现了这样一种直觉：如果一个项目在前 $k$ 中，那么对于任何 ${k}^{\prime } > k$ ，它都应该在前 ${k}^{\prime }$ 中。等价地， $k$ 的选择只是一个滑块，用于选择要返回给用户的结果数量，改变 $k$ 应该只改变返回的结果数量，而不改变结果的底层集合。

Definition 2 (Containment): For any $k,{R}_{k} \subset  {R}_{k + 1}$ . Replacing " $\subset$ " with " $\subseteq$ ",gives the weak containment property.

定义2（包含性）：对于任意 $k,{R}_{k} \subset  {R}_{k + 1}$ 。将“ $\subset$ ”替换为“ $\subseteq$ ”，可得到弱包含性属性。

The next property stipulates that the rank assigned to each tuple in the top- $k$ list should be unique.

下一个属性规定，在top- $k$ 列表中分配给每个元组的排名应该是唯一的。

Definition 3 (Unique ranking): Let ${r}_{k}\left( i\right)$ be the identity of the tuple from the input assigned rank $i$ in the output of the ranking procedure. The unique ranking property requires that $\forall i \neq  j.{r}_{k}\left( i\right)  \neq  {r}_{k}\left( j\right)$ .

定义3（唯一排名）：设 ${r}_{k}\left( i\right)$ 为在排名过程的输出中被分配排名 $i$ 的输入元组的标识。唯一排名属性要求 $\forall i \neq  j.{r}_{k}\left( i\right)  \neq  {r}_{k}\left( j\right)$ 。

The next property captures the semantics that the score function is assumed to only give a relative ordering, and is not an absolute measure of the value of a tuple.

下一个属性体现了这样的语义：假设得分函数仅给出相对顺序，而不是元组值的绝对度量。

Definition 4 (Value invariance): Let $\mathcal{D}$ denote the relation which includes score values ${v}_{1} \leq  {v}_{2} \leq  \ldots$ . Let ${s}_{i}^{\prime }$ be any set of score values satisfying ${v}_{1}^{\prime } \leq  {v}_{2}^{\prime } \leq  \ldots$ ,and define ${\mathcal{D}}^{\prime }$ to be $\mathcal{D}$ with all scores ${v}_{i}$ replaced with ${v}_{i}^{\prime }$ . The value invariance property requires that ${R}_{k}\left( \mathcal{D}\right)  = {R}_{k}\left( {\mathcal{D}}^{\prime }\right)$ for any $k$ .

定义4（值不变性）：设 $\mathcal{D}$ 表示包含得分值 ${v}_{1} \leq  {v}_{2} \leq  \ldots$ 的关系。设 ${s}_{i}^{\prime }$ 是满足 ${v}_{1}^{\prime } \leq  {v}_{2}^{\prime } \leq  \ldots$ 的任意得分值集合，并将 ${\mathcal{D}}^{\prime }$ 定义为将 $\mathcal{D}$ 中所有得分 ${v}_{i}$ 替换为 ${v}_{i}^{\prime }$ 后的结果。值不变性属性要求对于任意 $k$ ，有 ${R}_{k}\left( \mathcal{D}\right)  = {R}_{k}\left( {\mathcal{D}}^{\prime }\right)$ 。

For example, consider the relation with tuple-level uncertainty illustrated in Figure 4. Here,the scores are ${70} \leq  {80} \leq$ ${92} \leq  {100}$ . The value invariance property demands that we could replace these scores with,say, $1 \leq  2 \leq  3 \leq  {1000}$ ,and the result of the ranking would still be the same.

例如，考虑图4中所示的具有元组级不确定性的关系。这里，得分是 ${70} \leq  {80} \leq$ ${92} \leq  {100}$ 。值不变性属性要求我们可以将这些得分替换为，比如说 $1 \leq  2 \leq  3 \leq  {1000}$ ，并且排名结果仍然相同。

Finally, Zhang and Chomicki [39] proposed the stability condition in the tuple-level uncertainty model ${}^{1}$ . We adopt this property and generalize it to the attribute-level model:

最后，Zhang和Chomicki [39] 提出了元组级不确定性模型 ${}^{1}$ 中的稳定性条件。我们采用这个属性并将其推广到属性级模型：

Definition 5 (Stability): In the tuple-level model, given a tuple ${t}_{i} = \left( {{v}_{i},p\left( {t}_{i}\right) }\right)$ from $\mathcal{D}$ ,if we replace ${t}_{i}$ with ${t}_{i}^{ \uparrow  } =$ $\left( {{v}_{i}^{ \uparrow  },p\left( {t}_{i}^{ \uparrow  }\right) }\right)$ where ${v}_{i}^{ \uparrow  } \geq  {v}_{i},p\left( {t}_{i}^{ \uparrow  }\right)  \geq  p\left( {t}_{i}\right)$ ,then

定义5（稳定性）：在元组级模型中，给定来自 $\mathcal{D}$ 的元组 ${t}_{i} = \left( {{v}_{i},p\left( {t}_{i}\right) }\right)$ ，如果我们将 ${t}_{i}$ 替换为 ${t}_{i}^{ \uparrow  } =$ $\left( {{v}_{i}^{ \uparrow  },p\left( {t}_{i}^{ \uparrow  }\right) }\right)$ ，其中 ${v}_{i}^{ \uparrow  } \geq  {v}_{i},p\left( {t}_{i}^{ \uparrow  }\right)  \geq  p\left( {t}_{i}\right)$ ，那么

${t}_{i} \in  {R}_{k}\left( \mathcal{D}\right)  \Rightarrow  {t}_{i}^{ \uparrow  } \in  {R}_{k}\left( {\mathcal{D}}^{\prime }\right) ,$

where ${\mathcal{D}}^{\prime }$ is obtained by replacing ${t}_{i}$ with ${t}_{i}^{ \uparrow  }$ in $\mathcal{D}$ .

其中 ${\mathcal{D}}^{\prime }$ 是通过在 $\mathcal{D}$ 中将 ${t}_{i}$ 替换为 ${t}_{i}^{ \uparrow  }$ 得到的。

For the attribute-level model, the statement for stability remains the same but with ${t}_{i}^{ \uparrow  }$ defined as follows. Given a tuple ${t}_{i}$ whose score is a random variable ${X}_{i}$ ,we obtain ${t}_{i}^{ \uparrow  }$ by replacing ${X}_{i}$ with a random variable ${X}_{i}^{ \uparrow  }$ that is stochastically greater or equal than [31] ${X}_{i}$ ,denoted as ${X}_{i}^{ \uparrow  } \succcurlyeq  {X}_{i}$ .

对于属性级模型，稳定性的表述保持不变，但 ${t}_{i}^{ \uparrow  }$ 的定义如下。给定一个得分是随机变量 ${X}_{i}$ 的元组 ${t}_{i}$ ，我们通过将 ${X}_{i}$ 替换为一个随机变量 ${X}_{i}^{ \uparrow  }$ 来得到 ${t}_{i}^{ \uparrow  }$ ，该随机变量在随机意义上大于或等于 [31] ${X}_{i}$ ，表示为 ${X}_{i}^{ \uparrow  } \succcurlyeq  {X}_{i}$ 。

Stability captures the intuition that if a tuple is already in the top- $k$ ,making it "probabilistically larger" should not eject it. Stability also implies that making a non-top- $k$ probabilistically smaller should not bring it into the top- $k$ .

稳定性体现了这样的直觉：如果一个元组已经在top- $k$ 中，使其“概率上更大”不应将其排除在外。稳定性还意味着，使一个非top- $k$ 的元组在概率上更小不应使其进入top- $k$ 。

Note, these conditions make little explicit reference to probability models, and can apply to almost any ranking setting. They trivially hold for the top- $k$ semantics over certain data. Yet perhaps surprisingly, none of the existing definitions for top- $k$ over uncertain data satisfy these natural requirements!

注意，这些条件几乎没有明确提及概率模型，并且几乎可以应用于任何排序场景。对于某些数据上的前 $k$ 语义，它们显然成立。然而，或许令人惊讶的是，现有的针对不确定数据的前 $k$ 定义都不满足这些自然要求！

## B. Top- $k$ Queries on Probabilistic Data

## B. 概率数据上的前 $k$ 查询

We now consider how to extend ranking queries to uncertain data. Details differ slightly for the two uncertainty models: In the attribute-level model, a tuple has a random score but it always exists in any random possible world, i.e., every tuple participates in the ranking process in all possible worlds, and we rank these $N$ tuples based on their score distribution. In contrast, in the tuple-level model, a tuple has a fixed score but it may not always appear, i.e., it may not participate in the ranking process in some possible worlds. We still aim to produce a ranking on all $N$ tuples,taking this into account.

我们现在考虑如何将排序查询扩展到不确定数据。对于两种不确定性模型，细节略有不同：在属性级模型中，元组具有随机得分，但它始终存在于任何随机可能世界中，即每个元组在所有可能世界中都参与排序过程，并且我们根据它们的得分分布对这些 $N$ 个元组进行排序。相比之下，在元组级模型中，元组具有固定得分，但它可能并不总是出现，即它可能在某些可能世界中不参与排序过程。考虑到这一点，我们仍然旨在对所有 $N$ 个元组进行排序。

Considering the tuple-level model, the difficulty of extending ranking queries to probabilistic data is that there are now two distinct orderings present in the data: that given by the score, and that given by the probabilities. These two types of information need to be combined in some way to produce the top- $k$ (this can be orthogonal to the model used to describe the uncertainty in the data). We now detail a variety of approaches that have been taken, and discuss their shortcomings with respect to the conditions we have defined. The key properties are summarized in Figure 5.

考虑元组级模型，将排序查询扩展到概率数据的难点在于，数据中现在存在两种不同的排序：由得分给出的排序和由概率给出的排序。这两种类型的信息需要以某种方式组合起来以产生前 $k$ （这可能与用于描述数据中不确定性的模型无关）。我们现在详细介绍已采用的各种方法，并讨论它们相对于我们所定义条件的缺点。关键属性总结在图 5 中。

Combine two rankings. There has been much work on taking multiple rankings and combining them (e.g. taking the top 50 query web search results from multiple search engines, and combining them to get an overall ranking) based on minimizing disagreements [12]. Likewise, skyline-based approaches extract points which do not dominate each other, and are not themselves dominated, under multiple ordered dimensions [6]. But such approaches fail to account for the inherent semantics of the probability distribution: it is insufficient to treat it simply as an ordinal attribute, as this loses the meaning of the relative likelihoods, and does not guarantee our required properties.

组合两个排序。基于最小化分歧，已经有很多关于获取多个排序并将它们组合起来的工作（例如，从多个搜索引擎获取前 50 个查询网页搜索结果，并将它们组合起来以获得总体排序）[12]。同样，基于天际线的方法在多个有序维度下提取彼此不支配且自身不被支配的点[6]。但此类方法未能考虑概率分布的内在语义：仅仅将其视为序数属性是不够的，因为这会丢失相对可能性的含义，并且不能保证我们所需的属性。

Most likely top- $k$ . Since a probabilistic relation can define exponentially many possible worlds, one approach to the top- $k$ problem finds the top- $k$ set that has the highest support over all possible worlds. In other words, (conceptually) extract the top- $k$ from each possible world,and compute the support (probability) of each distinct top- $k$ set found. The $U$ -Topk approach [33] reports the most likely top- $k$ as the answer to the ranking query. This method has the advantage that it more directly incorporates the likelihood information, and satisfies unique ranking, value invariance, and stability. But it may not always return $k$ tuples when $\mathcal{D}$ is small,as also pointed out in [39]. More importantly, it violates the containment property. In fact,there are simple examples where the top- $k$ can be completely disjoint from the top- $\left( {k + 1}\right)$ . Consider the attribute-level model example in Figure 2. The top-1 result under the U-Top $k$ definition is ${t}_{1}$ ,since its probability of having the highest score in a random possible world is ${0.24} + {0.16} = {0.4}$ , larger than that of ${t}_{2}$ or ${t}_{3}$ . However,the top-2 result is $\left( {{t}_{2},{t}_{3}}\right)$ , whose probability of being the top-2 is 0.36 , larger than that of $\left( {{t}_{1},{t}_{2}}\right)$ or $\left( {{t}_{1},{t}_{3}}\right)$ . Thus,the top-2 list is completely disjoint from the top-1. Similarly one can verify that for the tuple-level model example in Figure 4,the top-1 result is ${t}_{1}$ but the top-2 is $\left( {{t}_{2},{t}_{3}}\right)$ or $\left( {{t}_{3},{t}_{4}}\right)$ . No matter what tie-breaking rule is used, the top-2 is completely disjoint from the top-1 .

最可能的前 $k$ 。由于概率关系可以定义指数级数量的可能世界，解决前 $k$ 问题的一种方法是找到在所有可能世界中支持度最高的前 $k$ 集合。换句话说，（从概念上）从每个可能世界中提取前 $k$ ，并计算找到的每个不同前 $k$ 集合的支持度（概率）。$U$ -Topk 方法[33]将最可能的前 $k$ 作为排序查询的答案进行报告。这种方法的优点是它更直接地纳入了可能性信息，并且满足唯一排序、值不变性和稳定性。但正如 [39] 中也指出的，当 $\mathcal{D}$ 较小时，它可能并不总是返回 $k$ 个元组。更重要的是，它违反了包含属性。事实上，有一些简单的例子表明前 $k$ 可能与前 $\left( {k + 1}\right)$ 完全不相交。考虑图 2 中的属性级模型示例。在 U-Top $k$ 定义下的前 1 个结果是 ${t}_{1}$ ，因为它在随机可能世界中具有最高得分的概率是 ${0.24} + {0.16} = {0.4}$ ，大于 ${t}_{2}$ 或 ${t}_{3}$ 的概率。然而，前 2 个结果是 $\left( {{t}_{2},{t}_{3}}\right)$ ，其成为前 2 个的概率是 0.36 ，大于 $\left( {{t}_{1},{t}_{2}}\right)$ 或 $\left( {{t}_{1},{t}_{3}}\right)$ 的概率。因此，前 2 个列表与前 1 个完全不相交。类似地，可以验证对于图 4 中的元组级模型示例，前 1 个结果是 ${t}_{1}$ ，但前 2 个是 $\left( {{t}_{2},{t}_{3}}\right)$ 或 $\left( {{t}_{3},{t}_{4}}\right)$ 。无论使用何种打破平局规则，前 2 个都与前 1 个完全不相交。

---

<!-- Footnote -->

${}^{1}$ The faithfulness property from [39] is discussed in Section VI.

${}^{1}$ [39] 中的忠实性属性将在第六节讨论。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Ranking method ✘ ✓ ✘ weak ✓ ✓ ✘ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ U-top $k$ [33] ✘ U- $k$ Ranks [33], [24] PT- $k$ [18] ✘ Global-top $k$ [39] ✓ Expected score ✓ Expected rank ✓ -->

<img src="https://cdn.noedgeai.com/0195c907-d507-7338-88cd-e8aa85b079a9_4.jpg?x=185&y=194&w=683&h=348&r=0"/>

Fig. 5. Summary of Ranking Methods for Uncertain Data

图 5. 不确定数据排序方法总结

<!-- Media -->

Most likely tuple at each rank. The previous approach fails because it deals with top- $k$ sets as immutable objects. Instead, we could consider the property of a certain tuple being ranked $k$ th in a possible world. In particular,let ${X}_{i,j}$ be the event that tuple $j$ is ranked $i$ within a possible world. Computing $\Pr \left\lbrack  {X}_{i,j}\right\rbrack$ for all $i,j$ pairs,this approach reports the $i$ th result as $\arg \mathop{\max }\limits_{j}\Pr \left\lbrack  {X}_{i,j}\right\rbrack$ ,i.e.,the tuple that is most likely to be ranked $i$ th over all possible worlds. This is the $U$ - ${kRanks}$ approach [33]; essentially the same definition is proposed as PRank in [24] and analyzed in the context of distributions over spatial data. This definition overcomes the shortcomings of U-Top $k$ and satisfies exact- $k$ and containment. However,it fails on unique ranking, as one tuple may dominate multiple ranks at the same time. A related issue is that some tuples may be quite likely, but never get reported. So in Figure 2, the top-3 under this definition is ${t}_{1},{t}_{3},{t}_{1} : {t}_{1}$ appears twice and ${t}_{2}$ never; for Figure 4,there is a tie for the third position, and there is no fourth placed tuple,even though $N = 4$ . These issues have also been pointed out in [18], [39]. In addition, it fails on stability, as shown in [39], since when the score of a tuple becomes larger, it may leave its original rank but cannot take over any higher ranks as the dominating winner.

每个排名下最有可能的元组。之前的方法之所以失败，是因为它将前 $k$ 集合视为不可变对象。相反，我们可以考虑某个元组在可能世界中排名为 $k$ 的属性。具体来说，设 ${X}_{i,j}$ 为元组 $j$ 在一个可能世界中排名为 $i$ 的事件。针对所有 $i,j$ 对计算 $\Pr \left\lbrack  {X}_{i,j}\right\rbrack$，此方法将第 $i$ 个结果报告为 $\arg \mathop{\max }\limits_{j}\Pr \left\lbrack  {X}_{i,j}\right\rbrack$，即，在所有可能世界中最有可能排名为 $i$ 的元组。这就是 $U$ - ${kRanks}$ 方法 [33]；本质上，[24] 中提出了与 PRank 相同的定义，并在空间数据分布的背景下进行了分析。此定义克服了 U - Top $k$ 的缺点，满足精确 $k$ 和包含性。然而，它在唯一排名方面存在问题，因为一个元组可能同时在多个排名中占主导地位。一个相关的问题是，一些元组可能很有可能出现，但却从未被报告。因此，在图 2 中，在此定义下的前 3 名中 ${t}_{1},{t}_{3},{t}_{1} : {t}_{1}$ 出现了两次，而 ${t}_{2}$ 从未出现；在图 4 中，第三名存在并列情况，并且没有第四名的元组，尽管 $N = 4$。这些问题在 [18]、[39] 中也有指出。此外，如 [39] 所示，它在稳定性方面存在问题，因为当一个元组的分数变大时，它可能会离开原来的排名，但无法作为主导胜者占据任何更高的排名。

Rank by top- $k$ probability. Attempting to patch the previous definition,we can replace the event "tuple $i$ is at rank $k$ " with the event "tuple $i$ is at rank $k$ or better",and reason about the probability of this event. That is,define the top- $k$ probability of a tuple as the probability that it is in the top- $k$ over all possible worlds. The probabilistic threshold top- $k$ query (PT- $k$ for short) returns the set of all tuples whose top- $k$ probability exceeds a user-specified probability $p$ [18]. However,for a user specified $p$ ,the "top- $k$ " list may not contain $k$ tuples, violating exact- $k$ . If we fix $p$ and increase $k$ ,the top- $k$ lists do expand, but they only satisfy the weak containment property. For instance consider the tuple-level example in Figure 2. If we set $p = {0.4}$ ,then the top-1 list is $\left( {t}_{1}\right)$ . But both the top- 2 and top-3 lists contain the same set of tuples: ${t}_{1},{t}_{2},{t}_{3}$ . A further drawback of using PT- $k$ for ranking is that user has to specify the threshold $p$ which greatly affects the result.

按前 $k$ 概率排名。为了修正之前的定义，我们可以将事件“元组 $i$ 排名为 $k$”替换为事件“元组 $i$ 排名为 $k$ 或更靠前”，并分析此事件的概率。也就是说，将一个元组的前 $k$ 概率定义为它在所有可能世界中处于前 $k$ 的概率。概率阈值前 $k$ 查询（简称 PT - $k$）返回所有前 $k$ 概率超过用户指定概率 $p$ 的元组集合 [18]。然而，对于用户指定的 $p$，“前 $k$”列表可能不包含 $k$ 个元组，这违反了精确 $k$。如果我们固定 $p$ 并增加 $k$，前 $k$ 列表确实会扩展，但它们仅满足弱包含性。例如，考虑图 2 中的元组级示例。如果我们设置 $p = {0.4}$，那么前 1 名列表是 $\left( {t}_{1}\right)$。但前 2 名和前 3 名列表包含相同的元组集合：${t}_{1},{t}_{2},{t}_{3}$。使用 PT - $k$ 进行排名的另一个缺点是用户必须指定阈值 $p$，这会极大地影响结果。

Similarly, the Global-Topk method ranks the tuples by their top- $k$ probability,and then takes the top- $k$ of these [39] based on this probability. This makes sure that exactly $k$ tuples are returned, but it again fails on containment. In Figure 2, under the Global-Top $k$ definition,the top-1 is ${t}_{1}$ ,but the top-2 is $\left( {{t}_{2},{t}_{3}}\right)$ . In Figure 4,the top-1 is ${t}_{1}$ ,but the top-2 is $\left( {{t}_{3},{t}_{2}}\right)$ .

同样，Global - Topk 方法根据元组的前 $k$ 概率对其进行排名，然后基于此概率选取这些元组中的前 $k$ 个 [39]。这确保了恰好返回 $k$ 个元组，但它在包含性方面再次失败。在图 2 中，根据 Global - Top $k$ 定义，前 1 名是 ${t}_{1}$，但前 2 名是 $\left( {{t}_{2},{t}_{3}}\right)$。在图 4 中，前 1 名是 ${t}_{1}$，但前 2 名是 $\left( {{t}_{3},{t}_{2}}\right)$。

Further,note that as $k$ increases towards $N$ ,then the importance of the score approaches zero, and these two methods reduce to simply ranking by probability alone.

此外，请注意，当 $k$ 趋近于 $N$ 时，分数的重要性趋近于零，这两种方法简化为仅按概率进行排名。

Expected score. The above approaches all differ from traditional ranking queries, in that they do not define a single ordering of the tuples from which the top- $k$ is taken-in other words,they do not resemble "top- $k$ " in the literal interpretation of the term. A simple approach in this direction is to just compute the expected score of each tuple, and rank by this score,then take the top- $k$ . It is easy to check that such an approach directly implies exact- $k$ ,containment,unique ranking, and stability. However, this is very dependent on the values of the scores: consider a tuple which has very low probability but a score that is orders of magnitude higher than others-then it gets propelled to the top of the ranking, since it has the highest expected score, even though it is unlikely. But if we reduce this score to being just greater than the next highest score, the tuple will drop down the ranking. It therefore violates value invariance. Furthermore, in the tuple-level model, simply using the expected score ignores all the correlation rules completely.

期望得分。上述方法均与传统的排名查询不同，因为它们并未定义元组的单一排序，以便从中选取前 $k$ 个元组——换句话说，从该术语的字面解释来看，它们并不像“前 $k$ 个”。在这方面，一种简单的方法是计算每个元组的期望得分，并根据该得分进行排名，然后选取前 $k$ 个。很容易验证，这种方法直接意味着精确的 $k$ 、包含性、唯一排名和稳定性。然而，这很大程度上取决于得分的值：考虑一个元组，它的概率非常低，但得分比其他元组高出几个数量级——那么它会被推到排名的顶部，因为它具有最高的期望得分，尽管这种情况不太可能发生。但如果我们将这个得分降低到仅比下一个最高得分略高，该元组的排名就会下降。因此，它违反了值不变性。此外，在元组级模型中，仅仅使用期望得分会完全忽略所有的关联规则。

## C. Ranking by Expected Ranks

## C. 按期望排名进行排序

Motivated by the deficiencies of existing definitions, we propose a new ranking method which we call expected rank. The intuition is that top- $k$ over certain data is defined by first providing a total ordering of the tuples, and then selecting the $k$ "best" tuples under the ordering. Any such definition immediately provides the containment and unique-ranking properties. After rejecting expected score due to its sensitivity to the score values, a natural candidate is the expected rank of the tuple over the possible worlds. More formally,

鉴于现有定义的不足，我们提出了一种新的排名方法，我们称之为期望排名。其直觉是，在某些数据上的前 $k$ 个元组的定义是，首先对元组进行全排序，然后在该排序下选择 $k$ 个“最佳”元组。任何这样的定义都能立即提供包含性和唯一排名属性。由于期望得分对得分值敏感而被舍弃后，一个自然的候选方案是元组在所有可能世界中的期望排名。更正式地说，

Definition 6 (Expected Rank): The rank of a tuple ${t}_{i}$ in a possible world $W$ is defined to be the number of tuples whose score is higher than ${t}_{i}$ (so the top tuple has rank 0 ),i.e.,

定义 6（期望排名）：在一个可能世界 $W$ 中，元组 ${t}_{i}$ 的排名定义为得分高于 ${t}_{i}$ 的元组的数量（因此排名第一的元组排名为 0），即

$$
{\operatorname{rank}}_{W}\left( {t}_{i}\right)  = \left| \left\{  {{t}_{j} \in  W \mid  {v}_{j} > {v}_{i}}\right\}  \right| 
$$

In the attribute-level uncertain model, we compute the expected rank $r\left( {t}_{i}\right)$ as above and then return the top- $k$ tuples with the lowest $r\left( {t}_{i}\right)$ . More precisely,

在属性级不确定模型中，我们如上计算期望排名 $r\left( {t}_{i}\right)$ ，然后返回 $r\left( {t}_{i}\right)$ 值最低的前 $k$ 个元组。更准确地说，

$$
r\left( {t}_{i}\right)  = \mathop{\sum }\limits_{{W \in  \mathcal{W},{t}_{i} \in  W}}\Pr \left\lbrack  W\right\rbrack   \cdot  {\operatorname{rank}}_{W}\left( {t}_{i}\right)  \tag{1}
$$

In the tuple-level model, we have to define how to handle possible worlds where ${t}_{i}$ does not appear. For such a world $W$ where ${t}_{i}$ does not appear,we define ${\operatorname{rank}}_{W}\left( {t}_{i}\right)  = \left| W\right|$ ,i.e. we imagine that it follows after all the appearing tuples. So,

在元组级模型中，我们必须定义如何处理元组 ${t}_{i}$ 不出现的可能世界。对于元组 ${t}_{i}$ 不出现的这样一个世界 $W$ ，我们定义 ${\operatorname{rank}}_{W}\left( {t}_{i}\right)  = \left| W\right|$ ，即我们假设它排在所有出现的元组之后。所以，

$$
r\left( {t}_{i}\right)  = \mathop{\sum }\limits_{{{t}_{i} \in  W}}\Pr \left\lbrack  W\right\rbrack  {\operatorname{rank}}_{W}\left( {t}_{i}\right)  + \mathop{\sum }\limits_{{{t}_{i} \notin  W}}\Pr \left\lbrack  W\right\rbrack   \cdot  \left| W\right|  \tag{2}
$$

$$
 = \mathop{\sum }\limits_{{W \in  \mathcal{W}}}\Pr \left\lbrack  W\right\rbrack  {\operatorname{rank}}_{W}\left( {t}_{i}\right) ,
$$

where ${\operatorname{rank}}_{W}\left( {t}_{i}\right)$ is defined to be $\left| W\right|$ if ${t}_{i} \notin  W$ .

其中，如果 ${t}_{i} \notin  W$ ，则 ${\operatorname{rank}}_{W}\left( {t}_{i}\right)$ 定义为 $\left| W\right|$ 。

For the example in Figure 2,the expected rank for ${t}_{2}$ is $r\left( {t}_{2}\right)  = {0.24} \times  1 + {0.16} \times  2 + {0.36} \times  0 + {0.24} \times  1 = {0.8}$ . Similarly $r\left( {t}_{1}\right)  = {1.2},r\left( {t}_{3}\right)  = 1$ . So the final ranking is $\left( {{t}_{2},{t}_{3},{t}_{1}}\right)$ . For the example in Figure $4,r\left( {t}_{2}\right)  = {0.2} \times  1 + {0.2} \times  3 + {0.3} \times$ $0 + {0.3} \times  2 = {1.4}$ . Note that ${t}_{2}$ does not appear in the second and the fourth worlds, so its ranks are taken to be 3 and 2, respectively. Similarly $r\left( {t}_{1}\right)  = {1.2},r\left( {t}_{3}\right)  = {0.9},r\left( {t}_{4}\right)  = {1.9}$ . So the final ranking is $\left( {{t}_{3},{t}_{1},{t}_{2},{t}_{4}}\right)$ .

对于图 2 中的示例，${t}_{2}$ 的期望排名是 $r\left( {t}_{2}\right)  = {0.24} \times  1 + {0.16} \times  2 + {0.36} \times  0 + {0.24} \times  1 = {0.8}$ 。类似地，$r\left( {t}_{1}\right)  = {1.2},r\left( {t}_{3}\right)  = 1$ 。所以最终排名是 $\left( {{t}_{2},{t}_{3},{t}_{1}}\right)$ 。对于图 $4,r\left( {t}_{2}\right)  = {0.2} \times  1 + {0.2} \times  3 + {0.3} \times$ 中的示例，$0 + {0.3} \times  2 = {1.4}$ 。注意，${t}_{2}$ 在第二个和第四个世界中未出现，因此它的排名分别取为 3 和 2。类似地，$r\left( {t}_{1}\right)  = {1.2},r\left( {t}_{3}\right)  = {0.9},r\left( {t}_{4}\right)  = {1.9}$ 。所以最终排名是 $\left( {{t}_{3},{t}_{1},{t}_{2},{t}_{4}}\right)$ 。

We now prove some properties of this definition. For simplicity, we assume that the expected ranks are unique, and so the ranking forms a total ordering. In practice, ties can be broken arbitrarily e.g. based on having the lexicographically smaller id. The same tie-breaking issues affect the ranking of certain data as well.

我们现在证明这个定义的一些性质。为简单起见，我们假设期望排名是唯一的，因此排名形成一个全序关系。在实践中，可以任意打破平局，例如基于字典序较小的ID。同样的平局打破问题也会影响某些数据的排名。

Theorem 1: Expected rank satisfies exact- $k$ ,containment, unique ranking, value invariance, and stability.

定理1：期望排名满足精确$k$、包含性、唯一排名、值不变性和稳定性。

Proof: The first three properties follow immediately from the fact that the expected rank is used to give an ordering. Value invariance follows by observing that changing the score values will not change the rankings in possible worlds, and therefore does not change the expected ranks.

证明：前三个性质可直接从使用期望排名来给出排序这一事实得出。值不变性可通过观察得出，即改变得分值不会改变可能世界中的排名，因此也不会改变期望排名。

For stability we show that when we change a tuple ${t}_{i}$ to ${t}_{i}^{ \uparrow  }$ , its expected rank will not increase, while the expected rank of any other tuple will not decrease. Let ${r}^{\prime }$ be the expected rank in the uncertain relation ${\mathcal{D}}^{\prime }$ after changing ${t}_{i}$ to ${t}_{i}^{ \uparrow  }$ . We need to show that $r\left( {t}_{i}\right)  \geq  {r}^{\prime }\left( {t}_{i}^{ \uparrow  }\right)$ and $r\left( {t}_{{i}^{\prime }}\right)  \leq  {r}^{\prime }\left( {t}_{{i}^{\prime }}\right)$ for any ${i}^{\prime } \neq  i$ .

对于稳定性，我们证明当我们将元组${t}_{i}$更改为${t}_{i}^{ \uparrow  }$时，其期望排名不会增加，而任何其他元组的期望排名不会减少。设${r}^{\prime }$为在不确定关系${\mathcal{D}}^{\prime }$中将${t}_{i}$更改为${t}_{i}^{ \uparrow  }$后的期望排名。我们需要证明对于任何${i}^{\prime } \neq  i$，有$r\left( {t}_{i}\right)  \geq  {r}^{\prime }\left( {t}_{i}^{ \uparrow  }\right)$和$r\left( {t}_{{i}^{\prime }}\right)  \leq  {r}^{\prime }\left( {t}_{{i}^{\prime }}\right)$。

Consider the attribute-level model first. By definition 6 and linearity of expectation, we have

首先考虑属性级模型。根据定义6和期望的线性性质，我们有

$$
r\left( {t}_{i}\right)  = \mathop{\sum }\limits_{{j \neq  i}}\Pr \left\lbrack  {{X}_{i} < {X}_{j}}\right\rbrack   = \mathop{\sum }\limits_{{j \neq  i}}\mathop{\sum }\limits_{\ell }{p}_{j,\ell }\Pr \left\lbrack  {{X}_{i} < {v}_{j,\ell }}\right\rbrack  
$$

$$
 \geq  \mathop{\sum }\limits_{{j \neq  i}}\mathop{\sum }\limits_{\ell }{p}_{j,\ell }\Pr \left\lbrack  {{X}_{i}^{ \uparrow  } < {v}_{j,\ell }}\right\rbrack  \;\left( {\text{ because }{X}_{i} \preccurlyeq  {X}_{i}^{ \uparrow  }}\right) 
$$

$$
 = \mathop{\sum }\limits_{{j \neq  i}}\Pr \left\lbrack  {{X}_{i}^{ \uparrow  } < {X}_{j}}\right\rbrack   = {r}^{\prime }\left( {t}_{i}^{ \uparrow  }\right) .
$$

For any ${i}^{\prime } \neq  i$ ,

对于任何${i}^{\prime } \neq  i$，

$$
r\left( {t}_{{i}^{\prime }}\right)  = \Pr \left\lbrack  {{X}_{{i}^{\prime }} < {X}_{i}}\right\rbrack   + \mathop{\sum }\limits_{{j \neq  {i}^{\prime },j \neq  i}}\Pr \left\lbrack  {{X}_{{i}^{\prime }} < {X}_{j}}\right\rbrack  
$$

$$
 = \mathop{\sum }\limits_{\ell }{p}_{{i}^{\prime },\ell }\Pr \left\lbrack  {{v}_{{i}^{\prime },\ell } < {X}_{i}}\right\rbrack   + \mathop{\sum }\limits_{{j \neq  {i}^{\prime },j \neq  i}}\Pr \left\lbrack  {{X}_{{i}^{\prime }} < {X}_{j}}\right\rbrack  
$$

$$
 \leq  \mathop{\sum }\limits_{\ell }{p}_{{i}^{\prime },\ell }\Pr \left\lbrack  {{v}_{{i}^{\prime },\ell } < {X}_{i}^{ \uparrow  }}\right\rbrack   + \mathop{\sum }\limits_{{j \neq  {i}^{\prime },j \neq  i}}\Pr \left\lbrack  {{X}_{{i}^{\prime }} < {X}_{j}}\right\rbrack  
$$

$$
 = \Pr \left\lbrack  {{X}_{{i}^{\prime }} < {X}_{i}^{ \uparrow  }}\right\rbrack   + \mathop{\sum }\limits_{{j \neq  {i}^{\prime },j \neq  i}}\Pr \left\lbrack  {{X}_{{i}^{\prime }} < {X}_{j}}\right\rbrack   = {r}^{\prime }\left( {t}_{{i}^{\prime }}\right) 
$$

Next consider the tuple-level model. If ${t}_{i}^{ \uparrow  }$ has a larger score than ${t}_{i}$ but the same probability,then $r\left( {t}_{i}\right)  \geq  {r}^{\prime }\left( {t}_{i}^{ \uparrow  }\right)$ follows easily from (2) since ${\operatorname{rank}}_{W}\left( {t}_{i}\right)$ can only get smaller while the second term of (2) remains unchanged. For similar reasons, $r\left( {t}_{{i}^{\prime }}\right)  \leq  {r}^{\prime }\left( {t}_{{i}^{\prime }}\right)$ for any ${i}^{\prime } \neq  i$ .

接下来考虑元组级模型。如果${t}_{i}^{ \uparrow  }$的得分比${t}_{i}$高但概率相同，那么由(2)很容易得出$r\left( {t}_{i}\right)  \geq  {r}^{\prime }\left( {t}_{i}^{ \uparrow  }\right)$，因为${\operatorname{rank}}_{W}\left( {t}_{i}\right)$只会变小，而(2)中的第二项保持不变。出于类似的原因，对于任何${i}^{\prime } \neq  i$，有$r\left( {t}_{{i}^{\prime }}\right)  \leq  {r}^{\prime }\left( {t}_{{i}^{\prime }}\right)$。

If ${t}_{i}^{ \uparrow  }$ has the same score as ${t}_{i}$ but a larger probability, ${\operatorname{rank}}_{W}\left( {t}_{i}\right)$ stays the same for any possible world $W$ ,but $\Pr \left\lbrack  W\right\rbrack$ may change. We divide all the possible worlds into three categories: (a) those containing ${t}_{i}$ ,(b) those containing one of the tuples in the exclusion rule of ${t}_{i}$ (other than ${t}_{i}$ ),and (c) all other possible worlds. Note that $\Pr \left\lbrack  W\right\rbrack$ does not change for any $W$ in category (b),so we only focus on categories (a) and (c). Observe that there is a one-to-one mapping between the possible worlds in category (a) and (c): $W \rightarrow  W \cup  \left\{  {t}_{i}\right\}$ . For each such pair,its contribution to $r\left( {t}_{i}\right)$ is

如果${t}_{i}^{ \uparrow  }$的得分与${t}_{i}$相同但概率更大，对于任何可能世界$W$，${\operatorname{rank}}_{W}\left( {t}_{i}\right)$保持不变，但$\Pr \left\lbrack  W\right\rbrack$可能会改变。我们将所有可能世界分为三类：(a)包含${t}_{i}$的世界，(b)包含${t}_{i}$的排除规则中的某个元组（除${t}_{i}$外）的世界，以及(c)所有其他可能世界。注意，对于类别(b)中的任何$W$，$\Pr \left\lbrack  W\right\rbrack$不会改变，因此我们只关注类别(a)和(c)。观察到类别(a)和(c)中的可能世界之间存在一一对应关系：$W \rightarrow  W \cup  \left\{  {t}_{i}\right\}$。对于每一对这样的世界，其对$r\left( {t}_{i}\right)$的贡献是

$$
\Pr \left\lbrack  W\right\rbrack   \cdot  \left| W\right|  + \Pr \left\lbrack  {W \cup  \left\{  {t}_{i}\right\}  }\right\rbrack   \cdot  {\operatorname{rank}}_{W}\left( {t}_{i}\right) . \tag{3}
$$

Suppose the tuples in the exclusion rule of ${t}_{i}$ are ${t}_{i,1},\ldots ,{t}_{i,s}$ . Note that $W$ and $W \cup  \left\{  {t}_{i}\right\}$ differs only in the inclusion of ${t}_{i}$ ,so we can write $\Pr \left\lbrack  W\right\rbrack   = \pi \left( {1 - \mathop{\sum }\limits_{\ell }p\left( {t}_{i,\ell }\right)  - p\left( {t}_{i}\right) }\right)$ and $\Pr \left\lbrack  {W \cup  \left\{  {t}_{i}\right\}  }\right\rbrack   = {\pi p}\left( {t}_{i}\right)$ for some $\pi$ . When $p\left( {t}_{i}\right)$ increases to $p\left( {t}_{i}^{ \uparrow  }\right)$ ,the increase in (3) is

假设${t}_{i}$的排除规则中的元组为${t}_{i,1},\ldots ,{t}_{i,s}$。注意，$W$和$W \cup  \left\{  {t}_{i}\right\}$仅在是否包含${t}_{i}$上有所不同，因此对于某个$\pi$，我们可以写成$\Pr \left\lbrack  W\right\rbrack   = \pi \left( {1 - \mathop{\sum }\limits_{\ell }p\left( {t}_{i,\ell }\right)  - p\left( {t}_{i}\right) }\right)$和$\Pr \left\lbrack  {W \cup  \left\{  {t}_{i}\right\}  }\right\rbrack   = {\pi p}\left( {t}_{i}\right)$。当$p\left( {t}_{i}\right)$增加到$p\left( {t}_{i}^{ \uparrow  }\right)$时，(3)式的增量为

$$
\pi \left( {p\left( {t}_{i}\right)  - p\left( {t}_{i}^{ \uparrow  }\right) }\right) \left| W\right|  + \pi \left( {p\left( {t}_{i}^{ \uparrow  }\right)  - p\left( {t}_{i}\right) }\right) {\operatorname{rank}}_{W}\left( {t}_{i}\right) 
$$

$$
 = \pi \left( {p\left( {t}_{i}\right)  - p\left( {t}_{i}^{ \uparrow  }\right) }\right) \left( {\left| W\right|  - {\operatorname{rank}}_{W}\left( {t}_{i}\right) }\right)  \leq  0.
$$

The same holds for each pair of possible worlds in categories (a) and (c). Therefore we have $r\left( {t}_{i}\right)  \geq  {r}^{\prime }\left( {t}_{i}^{ \uparrow  }\right)$ .

对于类别(a)和(c)中的每对可能世界，情况都是如此。因此，我们有$r\left( {t}_{i}\right)  \geq  {r}^{\prime }\left( {t}_{i}^{ \uparrow  }\right)$。

For any ${i}^{\prime } \neq  i$ ,the contribution of each pair is

对于任意${i}^{\prime } \neq  i$，每对的贡献为

$$
\Pr \left\lbrack  W\right\rbrack   \cdot  {\operatorname{rank}}_{W}\left( {t}_{{i}^{\prime }}\right)  + \Pr \left\lbrack  {W \cup  \left\{  {t}_{i}\right\}  }\right\rbrack   \cdot  {\operatorname{rank}}_{W \cup  \left\{  {t}_{i}\right\}  }\left( {t}_{{i}^{\prime }}\right) . \tag{4}
$$

When $p\left( {t}_{i}\right)$ increases to $p\left( {t}_{i}^{ \uparrow  }\right)$ ,the increase in (4) is

当$p\left( {t}_{i}\right)$增加到$p\left( {t}_{i}^{ \uparrow  }\right)$时，(4)式的增量为

$$
\pi \left( {p\left( {t}_{i}\right)  - p\left( {t}_{i}^{ \uparrow  }\right) }\right) \left( {{\operatorname{rank}}_{W}\left( {t}_{{i}^{\prime }}\right)  - {\operatorname{rank}}_{W \cup  \left\{  {t}_{i}\right\}  }\left( {t}_{{i}^{\prime }}\right) }\right)  \geq  0.
$$

The same holds for each pair of possible worlds in categories (a) and (c). Therefore we have ${r}^{\prime }\left( {t}_{{i}^{\prime }}\right)  \geq  r\left( {t}_{{i}^{\prime }}\right)$ .

对于类别(a)和(c)中的每对可能世界，情况都是如此。因此，我们有${r}^{\prime }\left( {t}_{{i}^{\prime }}\right)  \geq  r\left( {t}_{{i}^{\prime }}\right)$。

## IV. EXPECTED RANKS IN THE ATTRIBUTE-LEVEL UNCERTAINTY MODEL

## 四、属性级不确定性模型中的期望排名

This section presents efficient algorithms for calculating the expected rank of an uncertain relation $\mathcal{D}$ with $N$ tuples in the attribute-level uncertain model. We first show an exact algorithm that can calculate the expected ranks of all tuples in $\mathcal{D}$ with $O\left( {N\log N}\right)$ processing cost. We then propose an approximate algorithm that can terminate the search as soon as the top- $k$ tuples with the $k$ smallest expected ranks are guaranteed to be found without accessing all tuples.

本节介绍了在属性级不确定模型中计算具有$N$个元组的不确定关系$\mathcal{D}$的期望排名的高效算法。我们首先展示一种精确算法，该算法可以以$O\left( {N\log N}\right)$的处理成本计算$\mathcal{D}$中所有元组的期望排名。然后，我们提出一种近似算法，该算法可以在保证找到具有$k$个最小期望排名的前$k$个元组时立即终止搜索，而无需访问所有元组。

<!-- Media -->

Algorithm 1: A-ERank(D,k)

算法1：A - ERank(D,k)

---

Create $U$ containing values from ${t}_{1}.{X}_{1},\ldots ,{t}_{N}.{X}_{N}$ ,in order;

按顺序创建包含来自${t}_{1}.{X}_{1},\ldots ,{t}_{N}.{X}_{N}$的值的$U$；

Compute $q\left( v\right) \forall v \in  U$ by one pass over $U$ ;

通过对$U$进行一次遍历计算$q\left( v\right) \forall v \in  U$；

Initialize a priority queue $A$ sorted by expected rank;

初始化一个按期望排名排序的优先队列$A$；

for $i = 1,\ldots ,N$ do

对于$i = 1,\ldots ,N$执行

	Compute $r\left( {t}_{i}\right)$ using $q\left( v\right)$ ’s and ${X}_{i}$ using Eqn. (6);

	使用$q\left( v\right)$计算$r\left( {t}_{i}\right)$，并使用公式(6)计算${X}_{i}$；

	Insert $\left( {{t}_{i},r\left( {t}_{i}\right) }\right)$ into $A$ ;

	将$\left( {{t}_{i},r\left( {t}_{i}\right) }\right)$插入到$A$中；

	if $\left| A\right|  > k$ then Drop element with largest expected rank

	如果$\left| A\right|  > k$，则从$\left| A\right|  > k$中删除期望排名最大的元素

	from $A$ ;

	；

return $A$ ;

返回 $A$ ;

---

<!-- Media -->

## A. Exact Computation

## A. 精确计算

By Definition 6 and linearity of expectation, we have

根据定义 6 和期望的线性性质，我们有

$$
r\left( {t}_{i}\right)  = \mathop{\sum }\limits_{{i \neq  j}}\Pr \left\lbrack  {{X}_{j} > {X}_{i}}\right\rbrack   \tag{5}
$$

The brute-force search (BFS) approach requires $O\left( N\right)$ time to compute $r\left( {t}_{i}\right)$ for one tuple and $O\left( {N}^{2}\right)$ time to compute the ranks of all tuples. The quadratic dependence on $N$ is prohibitive when $N$ is large. Below we present an improved algorithm that runs in $O\left( {N\log N}\right)$ time. We observe that (5) can be written as:

暴力搜索（Brute-Force Search，BFS）方法计算一个元组的 $r\left( {t}_{i}\right)$ 需要 $O\left( N\right)$ 时间，计算所有元组的排名需要 $O\left( {N}^{2}\right)$ 时间。当 $N$ 很大时，与 $N$ 的二次相关性是难以接受的。下面我们提出一种改进的算法，其运行时间为 $O\left( {N\log N}\right)$。我们观察到 (5) 可以写成：

$$
r\left( {t}_{i}\right)  = \mathop{\sum }\limits_{{i \neq  j}}\mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{i}}{p}_{i,\ell }\Pr \left\lbrack  {{X}_{j} > {v}_{i,\ell }}\right\rbrack   = \mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{i}}{p}_{i,\ell }\mathop{\sum }\limits_{{j \neq  i}}\Pr \left\lbrack  {{X}_{j} > {v}_{i,\ell }}\right\rbrack  
$$

$$
 = \mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{i}}{p}_{i,\ell }\left( {\mathop{\sum }\limits_{j}\Pr \left\lbrack  {{X}_{j} > {v}_{i,\ell }}\right\rbrack   - \Pr \left\lbrack  {{X}_{i} > {v}_{i,\ell }}\right\rbrack  }\right) 
$$

$$
 = \mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{i}}{p}_{i,\ell }\left( {q\left( {v}_{i,\ell }\right)  - \Pr \left\lbrack  {{X}_{i} > {v}_{i,\ell }}\right\rbrack  }\right) , \tag{6}
$$

where we define $q\left( v\right)  = \mathop{\sum }\limits_{j}\Pr \left\lbrack  {{X}_{j} > v}\right\rbrack$ . Let $U$ be the universe of all possible values of ${X}_{i},i = 1,\ldots ,N$ . Because we assume each pdf has constant size bounded by $s$ ,we have $\left| U\right|  \leq  \left| {sN}\right|$ . When $s$ is a constant,we have $\left| U\right|  = O\left( N\right)$ .

其中我们定义 $q\left( v\right)  = \mathop{\sum }\limits_{j}\Pr \left\lbrack  {{X}_{j} > v}\right\rbrack$。设 $U$ 是 ${X}_{i},i = 1,\ldots ,N$ 所有可能值的全集。因为我们假设每个概率密度函数（Probability Density Function，pdf）的大小都有一个由 $s$ 界定的常数，所以我们有 $\left| U\right|  \leq  \left| {sN}\right|$。当 $s$ 是一个常数时，我们有 $\left| U\right|  = O\left( N\right)$。

Now observe that we can precompute $q\left( v\right)$ for all $v \in  U$ with a linear pass over the input after sorting $U$ which has a cost of $O\left( {N\log N}\right)$ . Following (6),exact computation of the expected rank for a single tuple can now be done in constant time given $q\left( v\right)$ for all $v \in  U$ . While computing these expected ranks,we maintain a priority queue of size $k$ that stores the $k$ tuples with smallest expected ranks dynamically. When all tuples have been processed, the contents of the priority queue are returned as the final answer. Computing $q\left( v\right)$ takes time $O\left( {N\log N}\right)$ ; getting expected ranks of all tuples while maintaining the priority queue takes $O\left( {N\log k}\right)$ time. Hence, the overall cost of this approach is $O\left( {N\log N}\right)$ . We denote this algorithm as A-ERrank and describe it in Algorithm 1.

现在观察到，在对 $U$ 进行排序（成本为 $O\left( {N\log N}\right)$）后，我们可以通过对输入进行一次线性遍历，为所有 $v \in  U$ 预先计算 $q\left( v\right)$。根据 (6)，在已知所有 $v \in  U$ 的 $q\left( v\right)$ 的情况下，现在可以在常数时间内精确计算单个元组的期望排名。在计算这些期望排名时，我们维护一个大小为 $k$ 的优先队列，该队列动态存储期望排名最小的 $k$ 个元组。当所有元组都处理完毕后，优先队列的内容将作为最终答案返回。计算 $q\left( v\right)$ 需要 $O\left( {N\log N}\right)$ 时间；在维护优先队列的同时获取所有元组的期望排名需要 $O\left( {N\log k}\right)$ 时间。因此，这种方法的总体成本是 $O\left( {N\log N}\right)$。我们将此算法记为 A - ERrank，并在算法 1 中进行描述。

## B. Pruning by Expected Scores

## B. 通过期望得分进行剪枝

A-ERank is very efficient even for large $N$ values. However, in certain scenarios accessing a tuple is considerably expensive (if it requires significant IO access). It then becomes desirable to reduce the number of tuples accessed in order to find the answer. It is possible to find a set of (possibly more than $k$ tuples) which is guaranteed to include the true top- $k$ expected ranks,by pruning based on tail bounds of the score distribution. If tuples are sorted in decreasing order of their expected scores,i.e. $\mathrm{E}\left\lbrack  {X}_{i}\right\rbrack$ ’s,we can terminate the search early. In the following discussion,we assume that if $i < j$ , then $\mathrm{E}\left\lbrack  {X}_{i}\right\rbrack   \geq  \mathrm{E}\left\lbrack  {X}_{j}\right\rbrack$ for all $1 \leq  i,j \leq  N$ . Equivalently,we can think of this as an interface which generates each tuple in turn,in decreasing order of $\mathrm{E}\left\lbrack  {X}_{i}\right\rbrack$ .

即使对于较大的 $N$ 值，A - ERank 也非常高效。然而，在某些场景中，访问一个元组的成本相当高（如果需要大量的输入/输出（Input/Output，IO）访问）。为了找到答案，减少访问的元组数量就变得很有必要。通过基于得分分布的尾部边界进行剪枝，有可能找到一组（可能超过 $k$ 个元组），该组保证包含真正的前 $k$ 个期望排名。如果元组按其期望得分（即 $\mathrm{E}\left\lbrack  {X}_{i}\right\rbrack$）降序排序，我们可以提前终止搜索。在下面的讨论中，我们假设如果 $i < j$，那么对于所有 $1 \leq  i,j \leq  N$ 都有 $\mathrm{E}\left\lbrack  {X}_{i}\right\rbrack   \geq  \mathrm{E}\left\lbrack  {X}_{j}\right\rbrack$。等价地，我们可以将其视为一个接口，该接口按 $\mathrm{E}\left\lbrack  {X}_{i}\right\rbrack$ 的降序依次生成每个元组。

The pruning algorithm scans these tuples, and maintains an upper bound on $r\left( {t}_{i}\right)$ ,denoted ${r}^{ + }\left( {t}_{i}\right)$ ,for each ${t}_{i}$ seen so far, and a lower bound on $r\left( {t}_{u}\right)$ for any unseen tuple ${t}_{u}$ ,denoted ${r}^{ - }$ . The algorithm halts when there are at least $k{r}^{ + }\left( {X}_{i}\right)$ ’s that are smaller than ${r}^{ - }$ . Suppose $n$ tuples ${t}_{1},\ldots ,{t}_{n}$ have been scanned. For $\forall i \in  \left\lbrack  {1,n}\right\rbrack$ ,we have:

剪枝算法会扫描这些元组，并为到目前为止所见到的每个 ${t}_{i}$ 维护 $r\left( {t}_{i}\right)$ 的上界（记为 ${r}^{ + }\left( {t}_{i}\right)$），同时为任何未见过的元组 ${t}_{u}$ 维护 $r\left( {t}_{u}\right)$ 的下界（记为 ${r}^{ - }$）。当至少有 $k{r}^{ + }\left( {X}_{i}\right)$ 个值小于 ${r}^{ - }$ 时，算法停止。假设已经扫描了 $n$ 个元组 ${t}_{1},\ldots ,{t}_{n}$。对于 $\forall i \in  \left\lbrack  {1,n}\right\rbrack$，我们有：

$$
r\left( {t}_{i}\right)  = \mathop{\sum }\limits_{{j \leq  n,j \neq  i}}\Pr \left\lbrack  {{X}_{j} > {X}_{i}}\right\rbrack   + \mathop{\sum }\limits_{{n < j \leq  N}}\Pr \left\lbrack  {{X}_{j} > {X}_{i}}\right\rbrack  
$$

$$
 = \mathop{\sum }\limits_{{j \leq  n,j \neq  i}}\Pr \left\lbrack  {{X}_{j} > {X}_{i}}\right\rbrack   + \mathop{\sum }\limits_{{n < j \leq  N}}\mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{i}}{p}_{i,\ell }\Pr \left\lbrack  {{X}_{j} > {v}_{i,\ell }}\right\rbrack  
$$

$$
 \leq  \mathop{\sum }\limits_{{j \leq  n,j \neq  i}}\Pr \left\lbrack  {{X}_{j} > {X}_{i}}\right\rbrack   + \mathop{\sum }\limits_{{n < j \leq  N}}\mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{i}}{p}_{i,\ell }\frac{\mathrm{E}\left\lbrack  {X}_{j}\right\rbrack  }{{v}_{i,\ell }}
$$

$$
\text{(Markov Inequality)}
$$

$$
 \leq  \mathop{\sum }\limits_{{j \leq  n,j \neq  i}}\Pr \left\lbrack  {{X}_{j} > {X}_{i}}\right\rbrack   + \left( {N - n}\right) \mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{i}}{p}_{i,\ell }\frac{\mathrm{E}\left\lbrack  {X}_{n}\right\rbrack  }{{v}_{i,\ell }}. \tag{7}
$$

The first term in (7) can be computed using only the seen tuples ${t}_{1},\ldots ,{t}_{n}$ . The second term could be computed using ${X}_{i}$ and ${X}_{n}$ . Hence,from the scanned tuples,we can maintain an upper bound on $r\left( {t}_{i}\right)$ for each tuple in $\left\{  {{t}_{1},\ldots ,{t}_{n}}\right\}$ ,i.e., we can set ${r}^{ + }\left( {t}_{i}\right)$ to be (7) for $i = 1,\ldots ,n$ . The second term in ${r}^{ + }\left( {t}_{i}\right)$ is updated for every newly scanned tuple ${t}_{n}$ (as well as the first term for ${t}_{n}$ ).

式 (7) 中的第一项仅可使用已见过的元组 ${t}_{1},\ldots ,{t}_{n}$ 来计算。第二项可使用 ${X}_{i}$ 和 ${X}_{n}$ 来计算。因此，从已扫描的元组中，我们可以为 $\left\{  {{t}_{1},\ldots ,{t}_{n}}\right\}$ 中的每个元组维护 $r\left( {t}_{i}\right)$ 的上界，即，我们可以将 ${r}^{ + }\left( {t}_{i}\right)$ 设置为 $i = 1,\ldots ,n$ 时的式 (7)。对于每个新扫描的元组 ${t}_{n}$，${r}^{ + }\left( {t}_{i}\right)$ 中的第二项（以及 ${t}_{n}$ 的第一项）都会更新。

Now we provide the lower bound ${r}^{ - }$ . Consider any unseen tuple ${t}_{u},u > n$ ,we have:

现在我们给出下界 ${r}^{ - }$。考虑任何未见过的元组 ${t}_{u},u > n$，我们有：

$$
r\left( {t}_{u}\right)  \geq  \mathop{\sum }\limits_{{j \leq  n}}\Pr \left\lbrack  {{X}_{j} > {X}_{u}}\right\rbrack   = n - \mathop{\sum }\limits_{{j \leq  n}}\Pr \left\lbrack  {{X}_{u} \geq  {X}_{j}}\right\rbrack  
$$

$$
 = n - \mathop{\sum }\limits_{{j \leq  n}}\mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{j}}{p}_{j,\ell }\Pr \left\lbrack  {{X}_{u} > {v}_{j,\ell }}\right\rbrack  
$$

$$
 \geq  n - \mathop{\sum }\limits_{{j \leq  n}}\mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{j}}{p}_{j,\ell }\frac{\mathrm{E}\left\lbrack  {X}_{n}\right\rbrack  }{{v}_{j,\ell }}.\;\text{ (Markov Ineq.) } \tag{8}
$$

This holds for any unseen tuple. Hence,we set ${r}^{ - }$ to be (8). Note that (8) only depends on the seen tuples. It is updated with every new tuple ${t}_{n}$ .

这对于任何未见过的元组都成立。因此，我们将 ${r}^{ - }$ 设置为式 (8)。注意，式 (8) 仅依赖于已见过的元组。它会随着每个新元组 ${t}_{n}$ 而更新。

These bounds lead immediately to an algorithm that maintains ${r}^{ + }\left( {t}_{i}\right)$ ’s for all tuples ${t}_{1},\ldots ,{t}_{n}$ and ${r}^{ - }$ . For each new tuple ${t}_{n}$ ,the ${r}^{ + }\left( {t}_{i}\right)$ ’s and ${r}^{ - }$ are updated. From these,we find the $k$ th largest ${r}^{ + }\left( {t}_{i}\right)$ value,and compare this to ${r}^{ - }$ . If it is less,then we know for sure that $k$ tuples with smallest expected ranks globally are among the first $n$ tuples,and can stop retrieving tuples. Otherwise, we move on to the next tuple. We refer to this algorithm as A-ERank-Prune.

这些边界条件直接引出了一种算法，该算法为所有元组 ${t}_{1},\ldots ,{t}_{n}$ 和 ${r}^{ - }$ 维护 ${r}^{ + }\left( {t}_{i}\right)$ 的值。对于每个新元组 ${t}_{n}$，${r}^{ + }\left( {t}_{i}\right)$ 的值和 ${r}^{ - }$ 都会更新。由此，我们找到第 $k$ 大的 ${r}^{ + }\left( {t}_{i}\right)$ 值，并将其与 ${r}^{ - }$ 进行比较。如果它更小，那么我们可以确定全局预期排名最小的 $k$ 个元组就在前 $n$ 个元组之中，此时可以停止检索元组。否则，我们继续处理下一个元组。我们将此算法称为 A - ERank - Prune。

A remaining challenge is how to find the $k$ tuples with the smallest expected ranks using the first $n$ tuples alone. This turns out to be difficult as it is not possible to obtain a precise order on their final ranks without inspecting all the $N$ tuples in $\mathcal{D}$ . Instead,we use the curtailed database ${\mathcal{D}}^{\prime } = \left\{  {{t}_{1},\ldots ,{t}_{n}}\right\}$ , and compute the exact expected rank ${r}^{\prime }\left( {t}_{i}\right)$ of every tuple (for $i \in  \left\lbrack  {1,n}\right\rbrack  ){t}_{i}$ in ${\mathcal{D}}^{\prime }$ . The rank ${r}^{\prime }\left( {t}_{i}\right)$ turns out to be an excellent surrogate for $r\left( {t}_{i}\right)$ for $i \in  \left\lbrack  {1,n}\right\rbrack$ in $\mathcal{D}$ (when the pruning algorithm terminates after processing $n$ tuples). Hence,we return the top- $k$ of these as the result of the query. We omit a detailed analysis of the quality of this approach, and instead show an empirical evaluation in our experimental study.

剩下的一个挑战是如何仅使用前 $n$ 个元组来找到预期排名最小的 $k$ 个元组。事实证明这很困难，因为在不检查 $\mathcal{D}$ 中的所有 $N$ 个元组的情况下，不可能获得它们最终排名的精确顺序。相反，我们使用缩减后的数据库 ${\mathcal{D}}^{\prime } = \left\{  {{t}_{1},\ldots ,{t}_{n}}\right\}$，并计算每个元组（对于 ${\mathcal{D}}^{\prime }$ 中的 $i \in  \left\lbrack  {1,n}\right\rbrack  ){t}_{i}$）的精确预期排名 ${r}^{\prime }\left( {t}_{i}\right)$。结果表明，当剪枝算法在处理 $n$ 个元组后终止时，排名 ${r}^{\prime }\left( {t}_{i}\right)$ 是 $\mathcal{D}$ 中 $i \in  \left\lbrack  {1,n}\right\rbrack$ 的 $r\left( {t}_{i}\right)$ 的绝佳替代。因此，我们返回这些元组中排名前 $k$ 的元组作为查询结果。我们省略了对这种方法质量的详细分析，而是在实验研究中展示了实证评估。

A straightforward implementation of A-ERrank-Prune requires $O\left( {n}^{2}\right)$ time. After seeing ${t}_{n}$ ,the bounds in both (7) and (8) can be updated in constant time,by retaining $\mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{j}}\frac{{p}_{i,\ell }}{{v}_{i,\ell }}$ for each seen tuple. The challenge is to update the first term in (7) for all $i \leq  n$ . A basic approach requires linear time,for adding $\Pr \left\lbrack  {{X}_{n} > {X}_{i}}\right\rbrack$ to the already computed $\mathop{\sum }\limits_{{j \leq  n - 1,j \neq  i}}\Pr \left\lbrack  {{X}_{j} > }\right.$ $\left. {X}_{i}\right\rbrack$ for all $i$ ’s as well as computing $\mathop{\sum }\limits_{{i < n - 1}}\Pr \left\lbrack  {{X}_{i} > {X}_{n}}\right\rbrack$ ). This leads to a total running time of $O\left( {n}^{2}\right)$ for algorithm A-ERrank-Prune. Using a similar idea in designing algorithm A-ERank,we could utilize the value universe ${U}^{\prime }$ of all the seen tuples and maintain prefix sums of the $q\left( v\right)$ values,which would drive down the cost of this step to $O\left( {n\log n}\right)$ . We omit full details for space reasons.

A - ERrank - Prune 的直接实现需要 $O\left( {n}^{2}\right)$ 时间。在看到 ${t}_{n}$ 之后，通过为每个已看到的元组保留 $\mathop{\sum }\limits_{{\ell  = 1}}^{{s}_{j}}\frac{{p}_{i,\ell }}{{v}_{i,\ell }}$，可以在常数时间内更新 (7) 和 (8) 中的边界。挑战在于为所有 $i \leq  n$ 更新 (7) 中的第一项。一种基本方法需要线性时间，因为要将 $\Pr \left\lbrack  {{X}_{n} > {X}_{i}}\right\rbrack$ 添加到已经为所有 $i$ 计算出的 $\mathop{\sum }\limits_{{j \leq  n - 1,j \neq  i}}\Pr \left\lbrack  {{X}_{j} > }\right.$ $\left. {X}_{i}\right\rbrack$ 中，以及计算 $\mathop{\sum }\limits_{{i < n - 1}}\Pr \left\lbrack  {{X}_{i} > {X}_{n}}\right\rbrack$。这导致算法 A - ERrank - Prune 的总运行时间为 $O\left( {n}^{2}\right)$。在设计算法 A - ERank 时使用类似的想法，我们可以利用所有已看到元组的值域 ${U}^{\prime }$ 并维护 $q\left( v\right)$ 值的前缀和，这将把这一步的成本降低到 $O\left( {n\log n}\right)$。由于篇幅原因，我们省略了完整细节。

## V. EXPECTED RANKS IN THE TUPLE-LEVEL UNCERTAINTY MODEL

## 五、元组级不确定性模型中的预期排名

We now consider ranking an uncertain database $\mathcal{D}$ in the tuple-level uncertainty model . For $\mathcal{D}$ with $N$ tuples and $M$ rules,the aim is to retrieve the $k$ tuples with the smallest expected ranks. Recall that each rule ${\tau }_{j}$ is a set of tuples,where $\mathop{\sum }\limits_{{{t}_{i} \in  {\tau }_{j}}}p\left( {t}_{i}\right)  \leq  1$ . Without loss of generality we assume the tuples ${t}_{1},\ldots ,{t}_{n}$ are already sorted by the ranking attribute and ${t}_{1}$ is the tuple with the highest score. We use ${t}_{i}\diamond {t}_{j}$ to denote that ${t}_{i}$ and ${t}_{j}$ are in the same exclusion rule and ${t}_{i} \neq  {t}_{j}$ ; we use ${t}_{i}\bar{ \circ  }{t}_{j}$ to denote that ${t}_{i}$ and ${t}_{j}$ are not in the same exclusion rule. We first give an exact algorithm with $O\left( {N\log N}\right)$ complexity that accesses every tuple. Secondly, we show a pruning algorithm with $O\left( {n\log n}\right)$ complexity, that only reads the first $n$ tuples,assuming that the expected number of tuples in $\mathcal{D}$ is known to the algorithm.

我们现在考虑在元组级不确定性模型中对不确定数据库 $\mathcal{D}$ 进行排序。对于包含 $N$ 个元组和 $M$ 条规则的 $\mathcal{D}$，目标是检索出期望排名最小的 $k$ 个元组。回顾一下，每条规则 ${\tau }_{j}$ 是一个元组集合，其中 $\mathop{\sum }\limits_{{{t}_{i} \in  {\tau }_{j}}}p\left( {t}_{i}\right)  \leq  1$ 。不失一般性，我们假设元组 ${t}_{1},\ldots ,{t}_{n}$ 已经按照排名属性排序，并且 ${t}_{1}$ 是得分最高的元组。我们使用 ${t}_{i}\diamond {t}_{j}$ 表示 ${t}_{i}$ 和 ${t}_{j}$ 在同一条互斥规则中且 ${t}_{i} \neq  {t}_{j}$ ；使用 ${t}_{i}\bar{ \circ  }{t}_{j}$ 表示 ${t}_{i}$ 和 ${t}_{j}$ 不在同一条互斥规则中。我们首先给出一个复杂度为 $O\left( {N\log N}\right)$ 的精确算法，该算法会访问每个元组。其次，我们展示一个复杂度为 $O\left( {n\log n}\right)$ 的剪枝算法，该算法仅读取前 $n$ 个元组，假设算法已知 $\mathcal{D}$ 中元组的期望数量。

## A. Exact computation

## A. 精确计算

From Definition 6, in particular (2), given tuples that are sorted by their score attribute, we have:

根据定义 6，特别是 (2)，给定按得分属性排序的元组，我们有：

$$
r\left( {t}_{i}\right)  = p\left( {t}_{i}\right)  \cdot  \mathop{\sum }\limits_{{{t}_{j} \circ  {t}_{i},j < i}}p\left( {t}_{j}\right) 
$$

$$
 + \left( {1 - p\left( {t}_{i}\right) }\right)  \cdot  \left( {\frac{\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right) }{1 - p\left( {t}_{i}\right) } + \mathop{\sum }\limits_{{{t}_{j}\bar{\diamond }{t}_{i}}}p\left( {t}_{j}\right) }\right) .
$$

The first term computes ${t}_{i}$ ’s expected rank for random worlds when it appears, and the second term computes the expected size of a random world $W$ when ${t}_{i}$ does not appear in $W$ . The term $\frac{\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right) }{1 - p\left( {t}_{i}\right) }$ is the expected number of appearing tuples in the same rule as ${t}_{i}$ ,conditioned on ${t}_{i}$ not appearing,while $\mathop{\sum }\limits_{{{t}_{j}\bar{ \circ  }{t}_{i}}}p\left( {t}_{j}\right)$ accounts for the rest of the tuples. Rewriting,

第一项计算 ${t}_{i}$ 出现时在随机世界中的期望排名，第二项计算 ${t}_{i}$ 不在随机世界 $W$ 中出现时 $W$ 的期望大小。项 $\frac{\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right) }{1 - p\left( {t}_{i}\right) }$ 是在 ${t}_{i}$ 不出现的条件下，与 ${t}_{i}$ 在同一条规则中出现的元组的期望数量，而 $\mathop{\sum }\limits_{{{t}_{j}\bar{ \circ  }{t}_{i}}}p\left( {t}_{j}\right)$ 则考虑了其余的元组。重写如下，

$$
r\left( {t}_{i}\right)  = p\left( {t}_{i}\right)  \cdot  \mathop{\sum }\limits_{{{t}_{j} \circ  {t}_{i},j < i}}p\left( {t}_{j}\right) 
$$

$$
 + \mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right)  + \left( {1 - p\left( {t}_{i}\right) }\right)  \cdot  \mathop{\sum }\limits_{{{t}_{j}\widetilde{ \circ  }{t}_{i}}}p\left( {t}_{j}\right) . \tag{9}
$$

<!-- Media -->

Algorithm 2: T-ERank(D,k)

算法 2：T - ERank(D,k)

---

Sort $\mathcal{D}$ by score attribute s.t. if ${t}_{i}.{v}_{i} \geq  {t}_{j}.{v}_{j}$ ,then $i \leq  j$ ;

按得分属性对 $\mathcal{D}$ 进行排序，使得如果 ${t}_{i}.{v}_{i} \geq  {t}_{j}.{v}_{j}$ ，那么 $i \leq  j$ ；

Compute ${q}_{i}\forall i \in  \left\lbrack  {1,N}\right\rbrack$ and $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ by one pass over $\mathcal{D}$ ;

通过对 $\mathcal{D}$ 进行一次遍历计算 ${q}_{i}\forall i \in  \left\lbrack  {1,N}\right\rbrack$ 和 $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ ；

Initialize a priority queue $A$ sorted by expected rank;

初始化一个按期望排名排序的优先队列 $A$ ；

for $i = 1,\ldots ,N$ do

对于 $i = 1,\ldots ,N$ 执行

	Compute $r\left( {t}_{i}\right)$ using (10);

	使用 (10) 计算 $r\left( {t}_{i}\right)$ ；

	if $\left| A\right|  > k$ then drop element with largest expected rank

	如果 $\left| A\right|  > k$ ，则从 $\left| A\right|  > k$ 中移除期望排名最大的元素

	from $A$ ;

	；

return $A$ ;

返回 $A$ ；

---

<!-- Media -->

Let ${q}_{i} = \mathop{\sum }\limits_{{j < i}}p\left( {t}_{j}\right)$ . We first compute ${q}_{i}$ in $O\left( N\right)$ time. At the same time, we find the expected number of tuples, $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   = \mathop{\sum }\limits_{{j = 1}}^{N}p\left( {t}_{j}\right)$ . Now (9) can be rewritten as:

设 ${q}_{i} = \mathop{\sum }\limits_{{j < i}}p\left( {t}_{j}\right)$ 。我们首先在 $O\left( N\right)$ 时间内计算 ${q}_{i}$ 。同时，我们求出元组的期望数量 $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   = \mathop{\sum }\limits_{{j = 1}}^{N}p\left( {t}_{j}\right)$ 。现在，式 (9) 可以重写为：

$$
r\left( {t}_{i}\right)  = p\left( {t}_{i}\right)  \cdot  \left( {{q}_{i} - \mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i},j < i}}p\left( {t}_{j}\right) }\right)  + \mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right) 
$$

$$
 + \left( {1 - p\left( {t}_{i}\right) }\right) \left( {\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   - p\left( {t}_{i}\right)  - \mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right) }\right) . \tag{10}
$$

By keeping the auxiliary information $\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i},j < i}}p\left( {t}_{j}\right)$ (i.e.,the sum of probabilities of tuples that have score values higher than ${t}_{i}$ in the same rule as ${t}_{i}$ ) and $\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right)$ (i.e.,the sum of probabilities of tuples that are in the same rule as ${t}_{i}$ ) for each tuple ${t}_{i}$ in $\mathcal{D},r\left( {t}_{i}\right)$ can be computed in $O\left( 1\right)$ time. By maintaining a priority queue of size $k$ that keeps the $k$ tuples with the smallest $r\left( {t}_{i}\right)$ ’s,we can select the top- $k$ tuples in $O\left( {N\log k}\right)$ time. Note that both $\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i},j < i}}p\left( {t}_{j}\right)$ and $\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right)$ are cheap to calculate initially given all the rules in a single scan of the relation (time $O\left( N\right)$ ). When $\mathcal{D}$ is not presorted by ${t}_{i}$ ’s score attribute,the running time of this algorithm is dominated by the sorting step, $O\left( {N\log N}\right)$ . Algorithm 2 gives pseudo-code for this algorithm, T-ERrank.

通过为 $\mathcal{D},r\left( {t}_{i}\right)$ 中的每个元组 ${t}_{i}$ 保留辅助信息 $\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i},j < i}}p\left( {t}_{j}\right)$（即，与 ${t}_{i}$ 处于同一规则中且得分值高于 ${t}_{i}$ 的元组的概率之和）和 $\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right)$（即，与 ${t}_{i}$ 处于同一规则中的元组的概率之和），可以在 $O\left( 1\right)$ 时间内计算 $\mathcal{D},r\left( {t}_{i}\right)$ 。通过维护一个大小为 $k$ 的优先队列，该队列保存 $k$ 个具有最小 $r\left( {t}_{i}\right)$ 值的元组，我们可以在 $O\left( {N\log k}\right)$ 时间内选择前 $k$ 个元组。请注意，给定关系的单次扫描中的所有规则（时间为 $O\left( N\right)$ ），最初计算 $\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i},j < i}}p\left( {t}_{j}\right)$ 和 $\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{i}}}p\left( {t}_{j}\right)$ 的成本都很低。当 $\mathcal{D}$ 未按 ${t}_{i}$ 的得分属性进行预排序时，该算法的运行时间主要由排序步骤决定，即 $O\left( {N\log N}\right)$ 。算法 2 给出了该算法 T - ERrank 的伪代码。

## B. Pruning

## B. 剪枝

Provided that the expected number of tuples $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ is known,we can answer top- $k$ queries more efficiently using pruning techniques without accessing all tuples. Note that $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ can be efficiently maintained in $O\left( 1\right)$ time when $\mathcal{D}$ is updated with deletion or insertion of tuples. As $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ is simply the sum of all the probabilities (note that it does not depend on the rules), it is reasonable to assume that it is always available. Similar to the attribute-level uncertainty case,we assume that $\mathcal{D}$ provides an interface to retrieve tuples in order of their score attribute from the highest to the lowest.

假设已知元组的期望数量 $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ ，我们可以使用剪枝技术更高效地回答前 $k$ 查询，而无需访问所有元组。请注意，当 $\mathcal{D}$ 因元组的删除或插入而更新时，可以在 $O\left( 1\right)$ 时间内高效地维护 $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ 。由于 $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ 只是所有概率的总和（请注意，它不依赖于规则），因此可以合理地假设它始终可用。与属性级不确定性情况类似，我们假设 $\mathcal{D}$ 提供了一个接口，用于按元组的得分属性从高到低的顺序检索元组。

The pruning algorithm scans the tuples in order. After seeing ${t}_{n}$ ,it can compute $r\left( {t}_{n}\right)$ exactly using $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ and ${q}_{n}$ in $O\left( 1\right)$ time based on (10). It also maintains ${r}^{\left( k\right) }$ ,the $k$ -th smallest $r\left( {t}_{i}\right)$ among all the tuples that have been retrieved. This can be done with a priority queue in $O\left( {\log k}\right)$ time per tuple. A lower bound on $r\left( {t}_{\ell }\right)$ for any $\ell  > n$ is computed as follows:

剪枝算法按顺序扫描元组。在查看 ${t}_{n}$ 之后，它可以根据式 (10) 在 $O\left( 1\right)$ 时间内使用 $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ 和 ${q}_{n}$ 精确计算 $r\left( {t}_{n}\right)$ 。它还维护 ${r}^{\left( k\right) }$ ，即所有已检索元组中第 $k$ 小的 $r\left( {t}_{i}\right)$ 。这可以通过优先队列在每个元组 $O\left( {\log k}\right)$ 时间内完成。对于任何 $\ell  > n$ ，$r\left( {t}_{\ell }\right)$ 的下界计算如下：

$$
r\left( {t}_{\ell }\right)  = p\left( {t}_{\ell }\right)  \cdot  \mathop{\sum }\limits_{{{t}_{j} \circ  {t}_{\ell },j < \ell }}p\left( {t}_{j}\right) 
$$

$$
 + \mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{\ell }}}p\left( {t}_{j}\right)  + \left( {1 - p\left( {t}_{\ell }\right) }\right)  \cdot  \mathop{\sum }\limits_{{{t}_{j}\bar{\diamond }{t}_{\ell }}}p\left( {t}_{j}\right) 
$$

$$
 = p\left( {t}_{\ell }\right)  \cdot  \mathop{\sum }\limits_{{{t}_{j}\bar{ \circ  }{t}_{\ell },j < \ell }}p\left( {t}_{j}\right)  + \mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   - p\left( {t}_{\ell }\right)  - p\left( {t}_{\ell }\right)  \cdot  \mathop{\sum }\limits_{{{t}_{j}\bar{ \circ  }{t}_{\ell }}}p\left( {t}_{j}\right) 
$$

$$
 = \mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   - p\left( {t}_{\ell }\right)  - p\left( {t}_{\ell }\right)  \cdot  \left( {\mathop{\sum }\limits_{{{t}_{j}\bar{ \circ  }{t}_{\ell }}}p\left( {t}_{j}\right)  - \mathop{\sum }\limits_{{{t}_{j}\bar{ \circ  }{t}_{\ell },j < \ell }}p\left( {t}_{j}\right) }\right) 
$$

$$
 = \mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   - p\left( {t}_{\ell }\right)  - p\left( {t}_{\ell }\right)  \cdot  \mathop{\sum }\limits_{{{t}_{j}\bar{ \circ  }{t}_{\ell },j > \ell }}p\left( {t}_{j}\right) . \tag{11}
$$

In the second step, we used the fact that

在第二步中，我们使用了这样一个事实：

$$
\mathop{\sum }\limits_{{{t}_{j}\diamond {t}_{\ell }}}p\left( {t}_{j}\right)  + \mathop{\sum }\limits_{{{t}_{j}\bar{\diamond }{t}_{\ell }}}p\left( {t}_{j}\right)  = \mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   - p\left( {t}_{\ell }\right) .
$$

Now,since ${q}_{\ell } = \mathop{\sum }\limits_{{j < \ell }}p\left( {t}_{j}\right)$ ,we observe that

现在，由于 ${q}_{\ell } = \mathop{\sum }\limits_{{j < \ell }}p\left( {t}_{j}\right)$ ，我们观察到

$$
\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   - {q}_{\ell } = \mathop{\sum }\limits_{{j > \ell }}p\left( {t}_{j}\right)  + p\left( {t}_{\ell }\right)  \geq  \mathop{\sum }\limits_{{{t}_{j}\bar{ \circ  }{t}_{\ell },j > \ell }}p\left( {t}_{j}\right) .
$$

Continuing with (11), we have:

继续看式(11)，我们有：

$$
r\left( {t}_{\ell }\right)  \geq  \mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   - p\left( {t}_{\ell }\right)  - p\left( {t}_{\ell }\right)  \cdot  \left( {\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack   - {q}_{\ell }}\right) 
$$

$$
 \geq  {q}_{\ell } - 1 \geq  {q}_{n} - 1 \tag{12}
$$

The last step uses the monotonicity of ${q}_{i}$ -by definition, ${q}_{n} \leq$ ${q}_{\ell }$ if $n \leq  \ell$ . Since tuples are scanned in order,obviously $\ell  > n$ .

最后一步利用了 ${q}_{i}$ 的单调性——根据定义，如果 $n \leq  \ell$ ，则 ${q}_{n} \leq$ ${q}_{\ell }$ 。由于元组是按顺序扫描的，显然 $\ell  > n$ 。

Thus,when ${r}^{\left( k\right) } \leq  {q}_{n} - 1$ ,we know for sure there are at least $k$ tuples amongst the first $n$ with expected ranks smaller than all unseen tuples. At this point, we can safely terminate the search. In addition, recall that for all the scanned tuples, their expected ranks are calculated exactly by (10). Hence this algorithm-which we dub T-ERank-Prune-can simply return the current top- $k$ tuples. From the above analysis,its time cost is $O\left( {n\log k}\right)$ where $n$ is potentially much smaller than $N$ .

因此，当 ${r}^{\left( k\right) } \leq  {q}_{n} - 1$ 时，我们可以确定在前 $n$ 个元组中至少有 $k$ 个元组的期望排名小于所有未见过的元组。此时，我们可以安全地终止搜索。此外，回想一下，对于所有已扫描的元组，它们的期望排名是通过式(10)精确计算的。因此，我们将此算法命名为T - ERank - Prune，它可以简单地返回当前的前 $k$ 个元组。根据上述分析，其时间成本为 $O\left( {n\log k}\right)$ ，其中 $n$ 可能远小于 $N$ 。

## VI. EXTENSIONS

## 六、扩展

Scoring functions. Our analysis has assumed that the score is a fixed value. In general, the score can be specified at query time by a user defined function. Note that our offline algorithms also work under this setting, as long as the scores can be computed. If the system has some interface that allows us to retrieve tuples in the score order (for the tuple-level order) or in the expected score order (for the attribute-level model), our pruning algorithms are applicable as well.

评分函数。我们的分析假设分数是一个固定值。一般来说，分数可以在查询时由用户定义的函数指定。请注意，只要分数可以计算，我们的离线算法在这种设置下也能工作。如果系统有某种接口允许我们按分数顺序（对于元组级排序）或按期望分数顺序（对于属性级模型）检索元组，我们的剪枝算法同样适用。

A main application of a query-dependent scoring function is $k$ -nearest-neighbor queries,which is the top- $k$ query instantiated in spatial databases. Here, the score is implicitly the distance of a data point to a query point. When the data points are uncertain, the distance to the query is a random variable, which can be modeled as an attribute-level uncertainty relation. Existing works [8],[24] essentially adopt U- $k$ Ranks semantics to define $k$ -nearest-neighbor queries in spatial databases. We believe that the expected rank definition makes a lot of sense in this context, and may have similar benefits over previous definitions of uncertain nearest neighbors.

与查询相关的评分函数的一个主要应用是 $k$ -最近邻查询，它是空间数据库中实例化的前 $k$ 查询。这里，分数隐式地表示数据点到查询点的距离。当数据点不确定时，到查询点的距离是一个随机变量，可以将其建模为属性级不确定性关系。现有工作 [8]、[24] 本质上采用U - $k$ 排名语义来定义空间数据库中的 $k$ -最近邻查询。我们认为，期望排名定义在这种情况下很有意义，并且与之前对不确定最近邻的定义相比可能有类似的优势。

When a relation has multiple (certain and uncertain) attributes on which a ranking query is to be performed, the user typically will give some function that combines this multiple attributes together and then rank on the output of the function. When at least one of the attributes is uncertain, the output of the function is also uncertain. This gives us another instance where our ranking semantics and algorithms could be applied.

当一个关系有多个（确定和不确定的）属性需要进行排名查询时，用户通常会给出一个函数，将这些多个属性组合在一起，然后对该函数的输出进行排名。当至少有一个属性是不确定的时，该函数的输出也是不确定的。这为我们提供了另一个可以应用我们的排名语义和算法的实例。

Continuous distributions. When the input data in the attribute-level uncertainty model is specified by a continuous distribution (e.g. a Gaussian or Poisson), it is often hard compute the probability that one variable exceeds another. However, by discretizing the distributions to an appropriate level of granularity (i.e., represented by a histogram), we can reduce to an instance of the discrete pdf problem. The error in this approach is directly related to the granularity of the discretization. Moreover, observe that our pruning-based methods initially require only information about expected values of the distributions. Since continuous distributions are typically described by their expected value (e.g., a Gaussian distribution is specified by its mean and variance), we can run the pruning algorithm on these parameters directly.

连续分布。当属性级不确定性模型中的输入数据由连续分布（例如高斯分布或泊松分布）指定时，通常很难计算一个变量超过另一个变量的概率。然而，通过将分布离散化到适当的粒度级别（即，用直方图表示），我们可以将其简化为离散概率密度函数问题的一个实例。这种方法的误差与离散化的粒度直接相关。此外，请注意，我们基于剪枝的方法最初只需要分布的期望值信息。由于连续分布通常由其期望值描述（例如，高斯分布由其均值和方差指定），我们可以直接对这些参数运行剪枝算法。

Further properties of a ranking. The ranking properties we define and study in Section III-A are by no means a complete characterization; rather, we argue that they are a minimum requirement for a ranking. Further properties can be defined and analyzed, although care is needed in their formulation. For example, Zhang and Chomicki [39] define the "faithfulness" property, which demands that (in the tuple-level model), given two tuples ${t}_{1} = \left( {{v}_{1},p\left( {t}_{1}\right) }\right)$ and ${t}_{2} = \left( {{v}_{2},p\left( {t}_{2}\right) }\right)$ with ${v}_{1} < {v}_{2}$ and $p\left( {t}_{1}\right)  < p\left( {t}_{2}\right)$ ,then ${t}_{1} \in  {R}_{k} \Rightarrow  {t}_{2} \in  {R}_{k}$ . This intuitive property implies that if ${t}_{2}$ "dominates" ${t}_{1}$ ,then ${t}_{2}$ should always be ranked higher than ${t}_{1}$ . However,there are examples where all existing definitions fail to guarantee faithfulness. Consider the relation:

排名的其他属性。我们在第三节A中定义和研究的排名属性绝不是一个完整的刻画；相反，我们认为它们是排名的最低要求。可以定义和分析更多的属性，不过在表述时需要谨慎。例如，张和乔米奇 [39] 定义了“忠实性”属性，该属性要求（在元组级模型中），给定两个元组 ${t}_{1} = \left( {{v}_{1},p\left( {t}_{1}\right) }\right)$ 和 ${t}_{2} = \left( {{v}_{2},p\left( {t}_{2}\right) }\right)$，且 ${v}_{1} < {v}_{2}$ 和 $p\left( {t}_{1}\right)  < p\left( {t}_{2}\right)$ ，那么 ${t}_{1} \in  {R}_{k} \Rightarrow  {t}_{2} \in  {R}_{k}$ 。这个直观的属性意味着如果 ${t}_{2}$ “支配” ${t}_{1}$ ，那么 ${t}_{2}$ 的排名应该总是高于 ${t}_{1}$ 。然而，存在一些例子表明现有的所有定义都无法保证忠实性。考虑以下关系：

<!-- Media -->

<table><tr><td>${t}_{i}$</td><td>${t}_{1}$</td><td>${t}_{2}$</td><td>${t}_{3}$</td><td>${t}_{4}$</td><td>${t}_{5}$</td></tr><tr><td>${v}_{i}$</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>$p\left( {t}_{i}\right)$</td><td>0.4</td><td>0.45</td><td>0.2</td><td>0.2</td><td>0.2</td></tr></table>

<table><tr><td>${t}_{i}$</td><td>${t}_{1}$</td><td>${t}_{2}$</td><td>${t}_{3}$</td><td>${t}_{4}$</td><td>${t}_{5}$</td></tr><tr><td>${v}_{i}$</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>$p\left( {t}_{i}\right)$</td><td>0.4</td><td>0.45</td><td>0.2</td><td>0.2</td><td>0.2</td></tr></table>

<!-- Media -->

with rules ${\tau }_{1} = \left\{  {{t}_{1},{t}_{3},{t}_{4},{t}_{5}}\right\}  ,{\tau }_{2} = \left\{  {t}_{2}\right\}$ . Here, ${t}_{2}$ "dominates" ${t}_{1}$ ,but all prior definitions (U-top $k,\mathrm{U} - k$ ranks,Global-top $k$ ,and PT- $k$ ) select ${t}_{1}$ is as the top-1. On this example, the expected rank definition will rank ${t}_{2}$ as the top-1,but unfortunately there are other examples where expected rank will also rank a dominating tuple lower than a dominated tuple. Our interpretation is that "faithfulness" defined this way may not be an achievable property, and one has to somehow take rules into consideration in order to make it a viable property.

遵循规则 ${\tau }_{1} = \left\{  {{t}_{1},{t}_{3},{t}_{4},{t}_{5}}\right\}  ,{\tau }_{2} = \left\{  {t}_{2}\right\}$ 。在此，${t}_{2}$ “支配” ${t}_{1}$ ，但所有先前的定义（U-top $k,\mathrm{U} - k$ 排名、Global-top $k$ 以及 PT- $k$ ）都将 ${t}_{1}$ 选为排名第一。在这个例子中，预期排名定义会将 ${t}_{2}$ 排在第一位，但不幸的是，在其他例子中，预期排名也会将一个支配元组排在被支配元组之后。我们的解释是，以这种方式定义的“忠实性”可能不是一个可实现的属性，为了使其成为一个可行的属性，必须以某种方式考虑规则。

Limitation of expected ranks. Our expected rank definition uses the expectation as the basis of ranking, i.e., the absolute ranks of each tuple from all possible worlds are represented by their mean. It is well known that the mean is statistically sensitive to the distribution of the underlying values (in our case, the absolute ranks of the tuple from all possible worlds). Hence, a more general and statistically more stable approach might be to use the median instead of the mean. This can be generalized to any quantile of the collection of absolute ranks for a tuple and derive the final ranking based on such quantiles. It remains an open problem to efficiently compute both the median-rank and the quantile-rank (for any quantile value). Likewise, it will also be important to study the semantics of these definitions, and how they compare to expected rank.

预期排名的局限性。我们的预期排名定义以期望作为排名的基础，即每个元组在所有可能世界中的绝对排名由它们的均值表示。众所周知，均值在统计上对底层值的分布（在我们的例子中，是元组在所有可能世界中的绝对排名）很敏感。因此，一种更通用且在统计上更稳定的方法可能是使用中位数而不是均值。这可以推广到元组绝对排名集合的任何分位数，并基于这些分位数得出最终排名。高效计算中位数排名和分位数排名（对于任何分位数值）仍然是一个未解决的问题。同样，研究这些定义的语义以及它们与预期排名的比较也很重要。

## VII. EXPERIMENTS

## 七、实验

We implemented our algorithms in GNU C++. All experiments were executed on a Linux machine with a $2\mathrm{{GHz}}$ CPU and 2GB main memory. In order to study the impact of data sets with different characteristics on both the score value distribution and the probability distribution, we focused on synthetic data sets. We additionally tested our algorithms on real data sets from the MystiQ project, and the trends there were similar to those reported here on synthetic data. We developed several data generators for both attribute-level and tuple-level uncertain models. Each generator controls the distribution on the score value as well as the probability. For both models, these distributions refer to the universe of score values and probabilities when we take the union of all tuples in $\mathcal{D}{.}^{2}$ The distributions used include uniform, Zipfian and correlated bivariate. They are abbreviated as $u$ , zipf and cor. For each tuple, we draw a score and probability value independently from the score distribution and probability distribution respectively. We refer to the result of drawing from these two distributions by the concatenation of the short names for each distribution for score then probability. For example, ${uu}$ indicates a data set with uniform distributions for both score values and probabilities; zipfu indicates a Zipfian distribution of score values and uniform distribution on the probabilities. The default the skewness parameter for the Zipfian distribution is 1.2,and the default value of $k = {100}$ .

我们用 GNU C++ 实现了我们的算法。所有实验都在一台配备 $2\mathrm{{GHz}}$ CPU 和 2GB 主内存的 Linux 机器上执行。为了研究不同特征的数据集对得分值分布和概率分布的影响，我们专注于合成数据集。我们还在 MystiQ 项目的真实数据集上测试了我们的算法，那里的趋势与这里报告的合成数据的趋势相似。我们为属性级和元组级不确定模型开发了几个数据生成器。每个生成器控制得分值和概率的分布。对于这两种模型，当我们取 $\mathcal{D}{.}^{2}$ 中所有元组的并集时，这些分布指的是得分值和概率的全集。使用的分布包括均匀分布、齐普夫分布和相关二元分布。它们分别缩写为 $u$ 、zipf 和 cor。对于每个元组，我们分别从得分分布和概率分布中独立抽取一个得分和概率值。我们通过将得分和概率的每个分布的简称连接起来来表示从这两个分布中抽取的结果。例如，${uu}$ 表示得分值和概率都具有均匀分布的数据集；zipfu 表示得分值具有齐普夫分布，概率具有均匀分布。齐普夫分布的默认偏度参数为 1.2，$k = {100}$ 的默认值。

## A. Attribute-level Uncertainty Model

## A. 属性级不确定性模型

We first studied the performance of the exact algorithm A-ERank by comparing it to the basic brute-force search (BFS) approach. The distribution on the probability universe does not affect the performance of both algorithms, since both algorithms calculate the expected ranks of all tuples. The score value distribution has no impact on BFS, but does affect A-ERank: the uniform score distribution results in the worst performance given a fixed number of tuples, as it leads to a large set of possible values. So we used ${uu}$ data sets for this experiment, to give the toughest test for this algorithm.

我们首先通过将精确算法 A-ERank 与基本的暴力搜索（BFS）方法进行比较来研究其性能。概率全集上的分布不会影响这两种算法的性能，因为这两种算法都计算所有元组的预期排名。得分值分布对 BFS 没有影响，但会影响 A-ERank：在固定元组数量的情况下，均匀得分分布会导致性能最差，因为它会产生大量可能的值。因此，我们在这个实验中使用 ${uu}$ 数据集，以便对该算法进行最严格的测试。

The score of each tuple is given by a pdf with 5 unique choices (i.e., $s = 5$ ). Figure 6(a) shows the total running time of these two algorithms as the size of $\mathcal{D}$ (i.e. the number of tuples, $N$ ) is varied,up to100,000tuples. A-ERank outperforms BFS by up to six orders of magnitude. This gap grows steadily as $N$ gets larger. A-ERank has very low query cost: it takes only about ${10}\mathrm{\;{ms}}$ to find all tuples expected ranks for $N = {100},{000}$ ,while the brute force approach takes ten minutes. Results are similar for other values of $s$ .

每个元组的得分由一个具有 5 个唯一选择的概率密度函数给出（即 $s = 5$ ）。图 6(a) 显示了随着 $\mathcal{D}$ 的大小（即元组数量 $N$ ）的变化，这两种算法的总运行时间，元组数量最多可达 100,000 个。A-ERank 的性能比 BFS 高出多达六个数量级。随着 $N$ 的增大，这个差距稳步扩大。A-ERank 的查询成本非常低：对于 $N = {100},{000}$ ，找到所有元组的预期排名只需要大约 ${10}\mathrm{\;{ms}}$ ，而暴力方法则需要十分钟。对于 $s$ 的其他值，结果类似。

Figure 6(b) shows the pruning power of A-ERank-Prune. In this experiment $N$ is set to100,000,with $s = 5$ and $k$ is varied from 10 to 100 . It shows that we often only need to materialize a small number of tuples of $\mathcal{D}$ (ordered by expected score) before we can be sure that we have found the top- $k$ ,across a variety of data sets. Intuitively, a more skewed distribution on either dimension should increase the algorithm's pruning power. This intuition is confirmed by the results in Figure 6(b). When both distributions are skewed, A-ERank-Prune could halt the scan after seeing less than ${20}\%$ of the relation. Overall, this shows that expected scores hold enough information to prune, even for more uniform distributions.

图6(b)展示了A - ERank - Prune算法的剪枝能力。在该实验中，$N$被设置为100,000，$s = 5$和$k$的取值范围为10到100。结果表明，在多种数据集上，在确定已找到前$k$个元素之前，我们通常只需实例化$\mathcal{D}$（按期望得分排序）的少量元组。直观地说，任一维度上分布越不均衡，算法的剪枝能力就越强。图6(b)中的结果证实了这一直觉。当两个分布都不均衡时，A - ERank - Prune算法在扫描不到关系的${20}\%$时就可以停止。总体而言，这表明即使对于更均匀的分布，期望得分也包含了足够的剪枝信息。

As discussed in Section IV-B, A-ERank-Prune is an approximate algorithm, in that it may not find the exact top- $k$ . Figure 6 reports its approximation quality on various data sets using the standard precision and recall metrics. Since A-ERank-Prune always returns $k$ tuples,its recall and precision are always the same. Figure 6 shows that it achieves high approximation quality: recall and precision are both in the 90th percentile when the score is distributed uniformly. The worst case occurs when the data is skewed on both dimensions, where the potential for pruning is greatest. The reason for this is that as more tuples are pruned, these unseen tuples have a greater chance to affect the expected ranks of the observed tuples. Even though the pruned tuples all have low expected scores, they could still have values with high probability to be ranked above some seen tuples, because of the heavy tail of their distribution. Even in this worst case, the recall and precision of T-ERank-Prune is about ${80}\%$ .

如第四节B部分所述，A - ERank - Prune是一种近似算法，因为它可能无法找到精确的前$k$个元素。图6使用标准的精确率和召回率指标报告了该算法在各种数据集上的近似质量。由于A - ERank - Prune算法总是返回$k$个元组，因此其召回率和精确率始终相同。图6显示该算法具有较高的近似质量：当得分均匀分布时，召回率和精确率均处于90%分位数。最糟糕的情况发生在数据在两个维度上都不均衡时，此时剪枝的潜力最大。原因在于，随着更多元组被剪枝，这些未被观察到的元组更有可能影响已观察到元组的期望排名。尽管被剪枝的元组期望得分都较低，但由于其分布的重尾特性，它们仍有可能以较高概率排在某些已观察到的元组之上。即使在这种最糟糕的情况下，T - ERank - Prune算法的召回率和精确率约为${80}\%$。

## B. Tuple-level Uncertainty Model

## B. 元组级不确定性模型

For our experiments in the tuple-level uncertainty model, where rules determine exclusions between tuples, we show results on data sets where ${30}\%$ of tuples are involved in rules with other tuples. Experiments with a greater or lesser degree of correlation gave similar results. We first investigate the performance of our algorithms. As before, there is also a brute-force search based approach, and it is also much more expensive than our algorithms, so we do not show these results.

在元组级不确定性模型的实验中，规则决定了元组之间的排除关系。我们展示了在${30}\%$的元组与其他元组存在规则关联的数据集上的实验结果。相关性程度不同的实验也得到了相似的结果。我们首先研究了算法的性能。和之前一样，也存在基于暴力搜索的方法，但它比我们的算法昂贵得多，因此我们未展示这些结果。

A notable difference in this model is that the pruning algorithm is able to output the exact top- $k$ ,provided that $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ , the expected number of tuples of $\mathcal{D}$ ,is known. Figure 7(a) shows the total running time for the T-ERank and T-ERank-Prune algorithms using ${uu}$ data. Both algorithms are extremely efficient. For100,000tuples,the T-ERank algorithm takes 10 milliseconds to compute the expected ranks of all tuples; applying pruning,T-ERank-Prune finds the same $k$ smallest ranks in just 1 millisecond. However, T-ERank is still highly efficient,and is the best solution when $\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$ is unavailable.

该模型的一个显著区别在于，只要已知$\mathcal{D}$的期望元组数$\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$，剪枝算法就能够输出精确的前$k$个元素。图7(a)展示了使用${uu}$数据时T - ERank和T - ERank - Prune算法的总运行时间。这两种算法都非常高效。对于100,000个元组，T - ERank算法计算所有元组的期望排名需要10毫秒；应用剪枝后，T - ERank - Prune算法仅需1毫秒就能找到相同的$k$个最小排名。然而，T - ERank算法仍然非常高效，并且在$\mathrm{E}\left\lbrack  \left| W\right| \right\rbrack$未知时是最佳解决方案。

Figure 7(b) shows the pruning power of T-ERank-Prune for different data sets. We fix $N = {100},{000}$ and vary the $k$ values. Clearly, a skewed distribution on either dimension increases the pruning capability of T-ERank-Prune. More importantly, even in the worst case of processing the ${uu}$ data set,T-ERank-Prune is able to prune more than ${90}\%$ of tuples.

图7(b)展示了T - ERank - Prune算法在不同数据集上的剪枝能力。我们固定$N = {100},{000}$，改变$k$的值。显然，任一维度上分布越不均衡，T - ERank - Prune算法的剪枝能力就越强。更重要的是，即使在处理${uu}$数据集的最坏情况下，T - ERank - Prune算法也能够剪枝超过${90}\%$的元组。

---

<!-- Footnote -->

${}^{2}$ For the attribute-level model,this includes all the value and probability pairs that a tuple's pdf has on its uncertain attribute.

${}^{2}$ 对于属性级模型，这包括元组的概率密度函数在其不确定属性上的所有值 - 概率对。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: DAFS 100 Druu Ouripf Recall and precision in top-k 0.8 ✘zipfu ✘zipfu 日- zipfzip( 日-zipfzipf 0.6 80 100 20 40 100 $\mathrm{k}\left( {\mathrm{N} = {10}^{5},\mathrm{\;s} = 5}\right)$ $\mathrm{k}\left( {\mathrm{N} = {10}^{5},\mathrm{\;s} = 5}\right)$ (c) Precision and Recall of A-ERank-Prune. - A-ERank Running time (secs) ${10}^{2}$ % of tuples pruned 80 60 ${10}^{0}$ ${10}^{ - }$ 20 10 0 40 N: number of tuples $\times  {10}^{4}$ (s=5) (a) Running time of exact algorithms (b) Pruning of A-ERank-Prune. -->

<img src="https://cdn.noedgeai.com/0195c907-d507-7338-88cd-e8aa85b079a9_10.jpg?x=216&y=199&w=1359&h=407&r=0"/>

Fig. 6. Attribute-level uncertain model: performance analysis.

图6. 属性级不确定模型：性能分析。

<!-- figureText: ${10}^{ - }$ 100 Dru Outipf 98 ✘zipfu % of tuples pruned 日- zipfzipf 96 94 92 20 40 60 80 100 $\mathrm{k}\left( {\mathrm{N} = {10}^{5},{30}\% \text{tuples in rules}}\right)$ (b) Pruning of T-ERank-Prune. OT-ERank-Prune Running time (secs) ${10}^{ - }$ ${10}^{ - }$ 2 4 10 N: number of tuples $\times  {10}^{4}$ (a) Running times of T-ERank and T-ERank-Prune -->

<img src="https://cdn.noedgeai.com/0195c907-d507-7338-88cd-e8aa85b079a9_10.jpg?x=398&y=678&w=992&h=433&r=0"/>

Fig. 7. Tuple-level uncertain model: performance analysis.

图7. 元组级不确定模型：性能分析。

<!-- Media -->

Our final set of experiments studies the impact of correlations between tuple's score value and probability. We say that the two are positively correlated when a tuple with higher score value also has a higher probability; a negative correlation means that higher score means lower probability. Such correlations have no impact on the performance of T-ERank as it computes the expected ranks for all tuples. However, correlation does have an interesting effect on the pruning capability of T-ERrank-Prune. Using correlated bivariate data sets of different correlation degrees, Figure 8(a) repeats the pruning experiment for T-ERank-Prune with $N =$ 100,000. The strongly positively correlated data set with a +0.8 correlation degree allows the highest amount of pruning, whereas the strongly negatively correlated data set with a -0.8 correlation degree results in the worst pruning power. But even in that worst case, T-ERank-Prune still pruned more than ${75}\%$ of tuples. Figure 8(b) reflects the running time of the same experiment. T-ERank-Prune consumes between 0.1 and 5 milliseconds to process100,000uncertain tuples.

我们的最后一组实验研究了元组得分值与概率之间的相关性所产生的影响。当得分值较高的元组也具有较高的概率时，我们称这两者为正相关；负相关则意味着得分越高，概率越低。由于T - ERank会计算所有元组的期望排名，因此这种相关性对其性能没有影响。然而，相关性确实会对T - ERrank - Prune的剪枝能力产生有趣的影响。图8(a)使用不同相关程度的二元相关数据集，针对T - ERank - Prune重复进行了剪枝实验，其中$N =$为100,000。相关程度为+0.8的强正相关数据集允许进行最多的剪枝，而相关程度为 - 0.8的强负相关数据集导致剪枝能力最差。但即使在这种最差情况下，T - ERank - Prune仍然能剪去超过${75}\%$的元组。图8(b)反映了同一实验的运行时间。T - ERank - Prune处理100,000个不确定元组的时间在0.1到5毫秒之间。

## VIII. BACKGROUND ON QUERYING UNCERTAIN DATA

## 八、不确定数据查询背景

Much effort has been devoted to modeling and processing uncertain data, so we survey only the most related work. TRIO [1], [4], [29], MayBMS [2] and MystiQ [10] are promising systems that are currently being developed. General query processing techniques have been extensively studied under the possible worlds semantics [9], [10], [14], [21], and important query types with specific query semantics are explored in more depth, skyline queries [27] and heavy hitters [38]. Indexing and nearest neighbor queries under the attribute-level uncertain model have also been explored [25], [32], [35], [5], [9], [26].

人们已经在不确定数据的建模和处理方面投入了大量精力，因此我们仅对最相关的工作进行综述。TRIO [1]、[4]、[29]、MayBMS [2]和MystiQ [10]是目前正在开发的有前景的系统。在可能世界语义下，通用查询处理技术已得到广泛研究[9]、[10]、[14]、[21]，并且对具有特定查询语义的重要查询类型进行了更深入的探索，如天际线查询[27]和频繁项查询[38]。在属性级不确定模型下，索引和最近邻查询也得到了研究[25]、[32]、[35]、[5]、[9]、[26]。

Section III-B discusses the most closely related works on answering top- $k$ queries on uncertain databases [18],[33], [39], [37]. Techniques used have included the Monte Carlo approach of sampling possible worlds [28], AI-style branch-and-bound search of the probability state space [33], dynamic programming approaches [37], [39], [17], and applying tail (Chernoff) bounds to determine when to prune [18]. There is ongoing work to understand semantics of top- $k$ queries in a variety of contexts. For example, the work of Lian and Chen [24] deals with ranking objects based on spatial uncertainty, and ranking based on linear functions. Recently, Soliman et al. [34] have extended their study on top- $k$ queries [33] to Group-By aggregate queries.

第三节B部分讨论了关于在不确定数据库上回答前$k$查询的最相关工作[18]、[33]、[39]、[37]。所使用的技术包括对可能世界进行采样的蒙特卡罗方法[28]、对概率状态空间进行人工智能式的分支限界搜索[33]、动态规划方法[37]、[39]、[17]，以及应用尾部（切尔诺夫）界来确定何时进行剪枝[18]。目前正在开展工作，以理解在各种上下文中前$k$查询的语义。例如，Lian和Chen [24]的工作涉及基于空间不确定性对对象进行排名，以及基于线性函数进行排名。最近，Soliman等人[34]将他们对前$k$查询[33]的研究扩展到了分组聚合查询。

Our study on the tuple-level uncertainty model limits us to considering correlations in the form of mutual exclusions. More advanced rules and processing techniques may be needed for complex correlations. Recent works based on graphical probabilistic models and Bayesian networks have shown promising results in both offline [30] and streaming data [22]. In these situations, initial approaches are based on Monte-Carlo simulations [21], [28].

我们对元组级不确定性模型的研究使我们仅限于考虑互斥形式的相关性。对于复杂的相关性，可能需要更高级的规则和处理技术。最近基于图形概率模型和贝叶斯网络的工作在离线数据[30]和流数据[22]方面都显示出了有前景的结果。在这些情况下，初始方法基于蒙特卡罗模拟[21]、[28]。

<!-- Media -->

<!-- figureText: 100 ${10}^{-2}$ Decor=+0.8 Ocor=0.0 Running time (secs) ✘ ${10}^{-5}$ 10 20 40 60 80 100 $\mathrm{k}$ ( $\mathrm{N} = {10}^{5},{30}\%$ tuples in rules) (b) Running time. 95 % of tuples pruned 90 85 80 $\rightarrow$ cor=+0.8 75 $\ominus  \operatorname{cor} = {0.0}$ ✘ $\sim$ cor=-0.8 20 40 60 80 100 $\mathrm{k}\left( {\mathrm{N} = {10}^{5},{30}\% \text{tuples in rules}}\right)$ (a) Pruning power. -->

<img src="https://cdn.noedgeai.com/0195c907-d507-7338-88cd-e8aa85b079a9_11.jpg?x=389&y=199&w=988&h=435&r=0"/>

Fig. 8. Impact of tuple's core and probability correlations on T-ERank-Prune.

图8. 元组得分与概率相关性对T - ERank - Prune的影响。

<!-- Media -->

## IX. Conclusion

## 九、结论

We have studied the semantics of ranking queries in probabilistic data. We adapt important properties that guide the definition of ranking queries in traditional, relational databases and analyze the limitations of existing top- $k$ ranking queries for probabilistic data. These properties naturally lead to the expected rank approach in uncertain domain. Efficient algorithms for two major models of uncertain data ensure the practicality of the expected rank. Our experiments convincingly demonstrate that ranking by expected ranks is very efficient in both attribute-level and tuple-level uncertainty models.

我们研究了概率数据中排名查询的语义。我们采用了指导传统关系数据库中排名查询定义的重要属性，并分析了现有前$k$排名查询在概率数据方面的局限性。这些属性自然地引出了不确定领域中的期望排名方法。针对不确定数据的两种主要模型的高效算法确保了期望排名的实用性。我们的实验令人信服地表明，在属性级和元组级不确定性模型中，基于期望排名进行排序都非常高效。

## REFERENCES

## 参考文献

[1] P. Agrawal, O. Benjelloun, A. Das Sarma, C. Hayworth, S. Nabar, T. Sugihara, and J. Widom, "Trio: A system for data, uncertainty, and lineage," in VLDB, 2006.

[2] L. Antova,C. Koch,and D. Olteanu," ${10}^{{10}^{6}}$ worlds and beyond: Efficient representation and processing of incomplete information," in ICDE, 2007.

[3] L. Antova, T. Jansen, C. Koch, and D. Olteanu, "Fast and simple relational processing of uncertain data," in ICDE, 2008.

[4] O. Benjelloun, A. D. Sarma, A. Halevy, and J. Widom, "ULDBs: databases with uncertainty and lineage," in VLDB, 2006.

[5] G. Beskales, M. A. Soliman, and I. F. Ilyas, "Efficient search for the top- $\mathrm{k}$ probable nearest neighbors in uncertain databases," in VLDB,2008.

[6] S. Borzsonyi, D. Kossmann, and K. Stocker, "The skyline operator," in ${ICDE},{2001}$ .

[7] S. Chaudhuri, K. Ganjam, V. Ganti, and R. Motwani, "Robust and efficient fuzzy match for online data cleaning," in SIGMOD, 2003.

[8] R. Cheng, J. Chen, M. Mokbel, and C.-Y. Chow, "Probabilistic verifiers: Evaluating constrained nearest-neighbor queries over uncertain data," in ICDE, 2008.

[9] R. Cheng, D. Kalashnikov, and S. Prabhakar, "Evaluating probabilistic queries over imprecise data," in SIGMOD, 2003.

[10] N. Dalvi and D. Suciu, "Efficient query evaluation on probabilistic databases," VLDB Journal, vol. 16, no. 4, pp. 523-544, 2007.

[11] A. Deshpande, C. Guestrin, S. Madden, J. Hellerstein, and W. Hong, "Model-driven data acquisition in sensor networks," in VLDB, 2004.

[12] C. Dwork, R. Kumar, M. Naor, and D. Sivakumar, "Rank aggregation methods for the web," in WWW Conference, 2001.

[13] R. Fagin, A. Lotem, and M. Naor, "Optimal aggregation algorithms for middleware," in PODS, 2001.

[14] A. Fuxman, E. Fazli, and R. J. Miller, "ConQuer: efficient management of inconsistent databases," in SIGMOD, 2005.

[15] A. Halevy, A. Rajaraman, and J. Ordille, "Data integration: the teenage year," in VLDB, 2006.

[16] M. A. Hernandez and S. J. Stolfo, "Real-world data is dirty: Data cleansing and the merge/purge problem," Data Mining and Knowledge Discovery, vol. 2, no. 1, pp. 9-37, 1998.

[17] M. Hua, J. Pei, W. Zhang, and X. Lin, "Efficiently answering proba-

[17] M. Hua、J. Pei、W. Zhang和X. Lin，“高效回答概率性……”

bilistic threshold top-k queries on uncertain data," in ICDE, 2008.

[18] M. Hua, J. Pei, W. Zhang, and X. Lin, "Ranking queries on uncertain data: A probabilistic threshold approach," in SIGMOD, 2008.

[19] I. F. Ilyas, W. G. Aref, A. K. Elmagarmid, H. Elmongui, R. Shah, and J. S. Vitter, "Adaptive rank-aware query optimization in relational databases," TODS, vol. 31, 2006.

[20] I. F. Ilyas, G. Beskales, and M. A. Soliman, "Survey of top-k query processing techniques in relational database systems," ACM Computing Surveys, 2008.

[21] R. Jampani, F. Xu, M. Wu, L. L. Perez, C. M. Jermaine, and P. J. Haas, "MCDB: a monte carlo approach to managing uncertain data," in SIGMOD, 2008.

[22] B. Kanagal and A. Deshpande, "Online filtering, smoothing and probabilistic modeling of streaming data," in ICDE, 2008.

[23] C. Li, K. C.-C. Chang, I. Ilyas, and S. Song, "RankSQL: Query algebra and optimization for relational top-k queries," in SIGMOD, 2005.

[24] X. Lian and L. Chen, "Probabilistic ranked queries in uncertain databases," in ${EDBT},{2008}$ .

[25] V. Ljosa and A. Singh, "APLA: Indexing arbitrary probability distributions," in ICDE, 2007.

[26] V. Ljosa and A. K. Singh, "Top-k spatial joins of probabilistic objects," in ${ICDE},{2008}$ .

[27] J. Pei, B. Jiang, X. Lin, and Y. Yuan, "Probabilistic skylines on uncertain data," in VLDB, 2007.

[28] C. Re, N. Dalvi, and D. Suciu, "Efficient top-k query evaluation on probabilistic databases," in ICDE, 2007.

[29] A. D. Sarma, O. Benjelloun, A. Halevy, and J. Widom, "Working models for uncertain data," in ICDE, 2006.

[30] P. Sen and A. Deshpande, "Representing and querying correlated tuples in probabilistic databases," in ICDE, 2007.

[31] J. G. Shanthikumar and M. Shaked, Stochastic Orders and Their Applications. Academic Press, 1994.

[32] S. Singh, C. Mayfield, S. Prabhakar, R. Shah, and S. Hambrusch, "Indexing uncertain categorical data," in ICDE, 2007.

[33] M. A. Soliman, I. F. Ilyas, and K. C.-C. Chang, "Top-k query processing in uncertain databases," in ICDE, 2007.

[34] M. A. Soliman, I. F. Ilyas, and K. C.-C. Chang, "Probabilistic top-k and ranking-aggregate queries," TODS, vol. 33, no. 3, 2008.

[35] Y. Tao, R. Cheng, X. Xiao, W. K. Ngai, B. Kao, and S. Prabhakar, "Indexing multi-dimensional uncertain data with arbitrary probability density functions," in VLDB, 2005.

[36] D. Xin, J. Han, and K. C.-C. Chang, "Progressive and selective merge: Computing top-k with ad-hoc ranking functions," in SIGMOD, 2007.

[37] K. Yi, F. Li, D. Srivastava, and G. Kollios, "Efficient processing of top-k queries in uncertain databases with x-relations," IEEE TKDE, vol. 20, no. 12, pp. 1669-1682, 2008.

[38] Q. Zhang, F. Li, and K. Yi, "Finding frequent items in probabilistic data," in SIGMOD, 2008.

[39] X. Zhang and J. Chomicki, "On the semantics and evaluation of top-k queries in probabilistic databases," in DBRank, 2008.