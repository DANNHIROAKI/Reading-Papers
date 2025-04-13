# Range Search on Multidimensional Uncertain Data

# 多维不确定数据的范围搜索

YUFEI TAO and XIAOKUI XIAO

陶宇飞和肖小奎

Chinese University of Hong Kong

香港中文大学

and

和

REYNOLD CHENG

郑宇

Hong Kong Polytechnic University

香港理工大学

In an uncertain database,every object $o$ is associated with a probability density function,which describes the likelihood that $o$ appears at each position in a multidimensional workspace. This article studies two types of range retrieval fundamental to many analytical tasks. Specifically, a nonfuzzy query returns all the objects that appear in a search region ${r}_{q}$ with at least a certain probability ${t}_{q}$ . On the other hand,given an uncertain object $q$ ,fuzzy search retrieves the set of objects that are within distance ${\varepsilon }_{q}$ from $q$ with no less than probability ${t}_{q}$ . The core of our methodology is a novel concept of "probabilistically constrained rectangle", which permits effective pruning/validation of nonqualifying/qualifying data. We develop a new index structure called the U-tree for minimizing the query overhead. Our algorithmic findings are accompanied with a thorough theoretical analysis, which reveals valuable insight into the problem characteristics, and mathematically confirms the efficiency of our solutions. We verify the effectiveness of the proposed techniques with extensive experiments.

在不确定数据库中，每个对象 $o$ 都与一个概率密度函数相关联，该函数描述了 $o$ 在多维工作空间中每个位置出现的可能性。本文研究了对许多分析任务至关重要的两种范围检索类型。具体而言，非模糊查询返回所有以至少某个概率 ${t}_{q}$ 出现在搜索区域 ${r}_{q}$ 中的对象。另一方面，给定一个不确定对象 $q$，模糊搜索会检索出与 $q$ 的距离在 ${\varepsilon }_{q}$ 以内且概率不小于 ${t}_{q}$ 的对象集合。我们方法的核心是“概率约束矩形”这一新颖概念，它允许对不符合/符合条件的数据进行有效剪枝/验证。我们开发了一种名为 U 树的新索引结构，以最小化查询开销。我们的算法研究成果伴随着全面的理论分析，该分析揭示了问题特征的宝贵见解，并从数学上证实了我们解决方案的效率。我们通过大量实验验证了所提出技术的有效性。

Categories and Subject Descriptors: H.2.2 [Database Management]: Physical Design-Access Methods; H.3.3 [Information Storage and Retrieval]: Information Search and Retrieval

分类和主题描述符：H.2.2 [数据库管理]：物理设计 - 访问方法；H.3.3 [信息存储与检索]：信息搜索与检索

General Terms: Algorithms, Experimentation

通用术语：算法、实验

Additional Key Words and Phrases: Uncertain databases, range search

其他关键词和短语：不确定数据库、范围搜索

## ACM Reference Format:

## ACM 引用格式：

Tao, Y., Xiao, X., and Cheng, R. 2007. Range search on multidimensional uncertain data. ACM Trans. Datab. Syst. 32, 3, Article 15 (August 2007), 54 pages. DOI $= {10.1145}/{1272743.1272745}$ http://doi.acm.org/10.1145/1272743.1272745

陶宇飞、肖小奎和郑宇，2007 年。多维不确定数据的范围搜索。ACM 数据库系统汇刊 32 卷 3 期，文章编号 15（2007 年 8 月），54 页。DOI $= {10.1145}/{1272743.1272745}$ http://doi.acm.org/10.1145/1272743.1272745

---

<!-- Footnote -->

This work was sponsored by two CERG grants from the Research Grant Council of the HKSAR government. Specifically, Y. Tao and X. Xiao were supported by Grant CUHK 1202/06, and R. Cheng by Grant PolyU 5138/06E.

这项工作由香港特别行政区政府研究资助局的两项研究资助局（CERG）资助。具体而言，陶宇飞和肖小奎得到了编号为 CUHK 1202/06 的资助，郑宇得到了编号为 PolyU 5138/06E 的资助。

Authors' addresses: Y. Tao and X. Xiao, Department of Computer Science and Engineering, Chinese University of Hong Kong, New Territories, Hong Kong; email: \{taoyf; xkxiao\}@cse.cuhk.edu.hk; R. Cheng, Department of Computing, Hong Kong Polytechnic University, Hung Hom, Kowloon, Hong Kong; email: csckcheng@comp.polyu.edu.hk.

作者地址：陶宇飞和肖小奎，香港中文大学计算机科学与工程系，香港新界；电子邮件：\{taoyf; xkxiao\}@cse.cuhk.edu.hk；郑宇，香港理工大学计算学系，香港九龙红磡；电子邮件：csckcheng@comp.polyu.edu.hk。

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or direct commercial advantage and that copies show this notice on the first page or initial screen of a display along with the full citation. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, to republish, to post on servers, to redistribute to lists, or to use any component of this work in other works requires prior specific permission and/or a fee. Permissions may be requested from Publications Dept., ACM, Inc., 2 Penn Plaza, Suite 701, New York, NY 10121-0701 USA, fax +1 (212) 869-0481, or permissions@acm.org. © 2007 ACM 0362-5915/2007/08-ART15 \$5.00 DOI 10.1145/1272743.1272745 http://doi.acm.org/ 10.1145/1272743.1272745

允许个人或课堂使用本作品的部分或全部内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或直接商业利益，并且在显示的第一页或初始屏幕上要显示此通知以及完整的引用信息。必须尊重本作品中除 ACM 之外其他所有者的版权。允许进行带引用的摘要。否则，如需复制、重新发布、发布到服务器、分发给列表或在其他作品中使用本作品的任何部分，则需要事先获得特定许可和/或支付费用。许可申请可向美国纽约州纽约市第 2 宾夕法尼亚广场 701 室的 ACM 出版部提出，传真：+1 (212) 869 - 0481，或发送电子邮件至 permissions@acm.org。© 2007 ACM 0362 - 5915/2007/08 - ART15 5 美元 DOI 10.1145/1272743.1272745 http://doi.acm.org/ 10.1145/1272743.1272745

<!-- Footnote -->

---

## 1. INTRODUCTION

## 1. 引言

Traditionally, a database describes objects with precise attribute values. Real-world entities, however, are often accompanied with uncertainty. Common sources of uncertainty include measurement errors, incomplete information (typically, missing data), variance in estimation from a random sample set, and so on. Furthermore, uncertainty may even be manually introduced in order to preserve privacy,which is the methodology behind " $k$ -anonymity" [Sweeney 2002] and "location-privacy" [Cheng et al. 2006b]. In recent years, the database community has witnessed an increasing amount of research on modeling and manipulating uncertain data, due to its importance in many emerging and traditional applications.

传统上，数据库使用精确的属性值来描述对象。然而，现实世界中的实体往往伴随着不确定性。不确定性的常见来源包括测量误差、信息不完整（通常是数据缺失）、从随机样本集进行估计时的差异等等。此外，为了保护隐私，甚至可能会人为引入不确定性，这正是“$k$ -匿名性” [Sweeney 2002] 和“位置隐私” [Cheng 等人 2006b] 背后的方法。近年来，由于不确定数据在许多新兴和传统应用中的重要性，数据库领域见证了越来越多关于对不确定数据进行建模和处理的研究。

Consider a meteorology system that monitors the temperature, relative humidity, and pollution level at a large number of sites. The corresponding readings are taken by sensors in local areas, and transmitted to a central database periodically (e.g., every 30 minutes). The database has a 3-value tuple for each site, which, however, may not exactly reflect the current conditions. For instance, the temperature at a site may have changed since it was measured. Therefore, various probabilistic models should be deployed to capture different attributes more accurately. For example, the actual temperature may be assumed to follow a Gaussian distribution with a mean calculated based on the last reported value (e.g., in the daytime, when temperature is rising, the mean should be set higher than the latest sensor reading).

考虑一个气象系统，它会监测大量地点的温度、相对湿度和污染水平。相应的读数由当地的传感器获取，并定期（例如，每 30 分钟）传输到中央数据库。数据库为每个地点存储一个三元组，但这些数据可能无法准确反映当前的实际情况。例如，某个地点的温度在测量之后可能已经发生了变化。因此，应该采用各种概率模型来更准确地捕捉不同的属性。例如，可以假设实际温度遵循高斯分布，其均值根据最后一次报告的值计算得出（例如，在白天温度上升时，均值应设置为高于最新的传感器读数）。

In some scenarios, an object cannot be represented with a "regular" model (like uniform, Gaussian, and Zipf distributions, etc.), but demands a complex probability density function (pdf) in the form of a histogram. This is true in location-based services, where a server maintains the locations of a set of moving objects such as vehicles. Each vehicle $o$ sends (through a wireless network) its current location, whenever it has moved away from its previously updated position $x$ by a certain distance $\varepsilon$ [Wolfson et al. 1999]. Therefore,at any time, the server does not have the precise whereabout of $o$ ,except for the fact that $o$ must be inside a circle centering at $x$ with radius $\varepsilon$ ,as shown in Figure 1(a). Evidently, $o$ cannot appear anywhere in the circle,since it is constrained by the underlying road network, illustrated with the segments in Figure 1(a). The distribution of $o$ can be approximated using a grid,where $o$ can fall only in the grey cells that intersect the circle and the network simultaneously. The probability that $o$ is covered by a particular cell is decided according to the application requirements. A simple choice is to impose an equal chance for all the grey cells, while more realistic modeling should take into account the distance between the cell and $x$ ,the speed limits of roads,and so on.

在某些场景中，一个对象无法用“常规”模型（如均匀分布、高斯分布和齐普夫分布等）来表示，而是需要一个以直方图形式呈现的复杂概率密度函数（pdf）。基于位置的服务就是这样的情况，服务器会维护一组移动对象（如车辆）的位置信息。每辆车 $o$ 每当它从之前更新的位置 $x$ 移动了一定距离 $\varepsilon$ 时，就会（通过无线网络）发送其当前位置 [Wolfson 等人 1999]。因此，在任何时候，服务器都无法确切知道 $o$ 的位置，只知道 $o$ 一定位于以 $x$ 为圆心、半径为 $\varepsilon$ 的圆内，如图 1(a) 所示。显然，$o$ 不可能出现在圆内的任何位置，因为它受到底层道路网络的限制，如图 1(a) 中的线段所示。$o$ 的分布可以用网格来近似，其中 $o$ 只能落在同时与圆和道路网络相交的灰色单元格中。$o$ 被特定单元格覆盖的概率是根据应用需求来确定的。一个简单的选择是为所有灰色单元格赋予相等的概率，而更符合实际情况的建模则应考虑单元格与 $x$ 之间的距离、道路的限速等等因素。

Unlike in the above environments, where uncertainty is caused by the delay in database updates, the raw data in some applications is inherently imprecise. Imagine a recommender company that assists clients to make promising investment plans, based on their preferences on the principle amount, the number of years before advantageous gains (i.e., the "cold period duration"), etc. In reality, it is difficult, or simply impossible, for a customer to specify unique values for these attributes. For instance, the amount of principle s/he is willing

与上述因数据库更新延迟导致不确定性的环境不同，一些应用中的原始数据本身就是不精确的。想象一家推荐公司，它根据客户对本金金额、获得有利收益前的年数（即“冷期时长”）等方面的偏好，协助客户制定有前景的投资计划。在现实中，客户很难，甚至根本不可能为这些属性指定唯一的值。例如，客户愿意投入的本金金额

Article 15 / 3 to lay down may fall in a wide range of $\left\lbrack  {\$ {10}\mathrm{k},\$ {40}\mathrm{k}}\right\rbrack$ . Obviously,depending on the principle, her/his expectation for other attributes may also vary (e.g., with a large principle, it would be reasonable to anticipate a shorter cold period). Therefore, a preference profile is also an uncertain object whose pdf can be described by a histogram, as demonstrated in Figure 1(b). The percentage in each cell indicates the overall interest of the client in a plan whose (principle, cold-period) 2D representation falls in the cell. For instance, s/he would not favor small investments with short cold durations (which would involve high risk), or large investments with lengthy cold periods (it would not be worthwhile to have a huge amount of money nonspendable for a long time).

第 15 条 / 3 可能会落在一个很宽的范围 $\left\lbrack  {\$ {10}\mathrm{k},\$ {40}\mathrm{k}}\right\rbrack$ 内。显然，根据本金的不同，客户对其他属性的期望也可能会有所变化（例如，本金较大时，期望冷期更短是合理的）。因此，偏好概况也是一个不确定对象，其概率密度函数可以用直方图来描述，如图 1(b) 所示。每个单元格中的百分比表示客户对某个投资计划的总体兴趣，该计划的（本金，冷期）二维表示落在该单元格内。例如，客户不会青睐冷期短的小额投资（这可能涉及高风险），也不会青睐冷期长的大额投资（长时间让大量资金无法使用是不值得的）。

<!-- Media -->

<!-- figureText: (a) The location of a vehicle cold period (years) 5% 0% 0% 0% 6 0% 10% 10% 0% 3 0% 5% 10% 15% 1 0% 5% 20% 20% 10k 20k 30k 40k principle (dollars) (b) A customer's investment profile -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_2.jpg?x=545&y=343&w=735&h=388&r=0"/>

Fig. 1. Probabilistic modeling of uncertain objects.

图 1. 不确定对象的概率建模。

<!-- Media -->

### 1.1 Motivation

### 1.1 动机

As in traditional spatial databases, range search is also an important operation in the applications mentioned earlier, and the building block for many other analytical tasks. However, since data is uncertain, it is no longer meaningful to simply declare that an object $o$ appears or does not appear in a query region ${r}_{q}$ . Instead,the notion of "appearance" should be accompanied with a value $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ ,capturing the probability that $o$ falls in ${r}_{q}$ ,according to the uncertainty modeling of $o$ . In practice,users are typically only concerned about events that may happen with a sufficiently large chance,that is,objects $o$ whose $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ is at least a certain threshold ${t}_{q}$ .

与传统空间数据库一样，范围搜索在前面提到的应用中也是一项重要操作，并且是许多其他分析任务的基础。然而，由于数据具有不确定性，简单地声明某个对象 $o$ 是否出现在查询区域 ${r}_{q}$ 中已不再有意义。相反，“出现”这一概念应伴随着一个值 $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ ，该值根据 $o$ 的不确定性建模，捕捉 $o$ 落入 ${r}_{q}$ 的概率。在实践中，用户通常只关注有足够大概率发生的事件，即 $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ 至少达到某个阈值 ${t}_{q}$ 的对象 $o$ 。

The above requirements lead to a novel type of queries: probability threshold-ing range retrieval. For instance, in the meteorology system discussed before, such a query would "find every high-fire-hazard site satisfying the following condition with at least ${50}\%$ probability: its temperature is at least ${86}\mathrm{\;F}$ degrees, humidity at most 5%, and pollution index no less than 7 (on a scale from 1 to 10)". In this query, ${t}_{q}$ equals ${50}\%$ ,and ${r}_{q}$ is a $3\mathrm{D}$ rectangle whose projections on the temperature, humidity, and pollution dimensions correspond to ranges $\left\lbrack  {{86}\mathrm{\;F},\infty ),\left\lbrack  {0,5\% }\right\rbrack  ,\left\lbrack  {7,{10}}\right\rbrack  \text{respectively. As another example,in the vehicle track-}}\right\rbrack$ ing application, a user may request to "identify all the cabs that are located in the downtown area $\left( { = {r}_{q}}\right)$ with at least ${80}\% \left( { = {t}_{q}}\right)$ likelihood".

上述需求催生了一种新型查询：概率阈值范围检索。例如，在之前讨论的气象系统中，这样的查询可以是“找出每个满足以下条件且概率至少为 ${50}\%$ 的高火灾危险地点：其温度至少为 ${86}\mathrm{\;F}$ 度，湿度至多为 5%，污染指数不低于 7（范围为 1 到 10）”。在这个查询中，${t}_{q}$ 等于 ${50}\%$ ，${r}_{q}$ 是一个 $3\mathrm{D}$ 矩形，其在温度、湿度和污染维度上的投影对应于范围 $\left\lbrack  {{86}\mathrm{\;F},\infty ),\left\lbrack  {0,5\% }\right\rbrack  ,\left\lbrack  {7,{10}}\right\rbrack  \text{respectively. As another example,in the vehicle track-}}\right\rbrack$ 。在打车应用中，用户可能会要求“识别所有以至少 ${80}\% \left( { = {t}_{q}}\right)$ 的可能性位于市区 $\left( { = {r}_{q}}\right)$ 的出租车”。

Range search can become "fuzzier", when the query region itself is uncertain. Assume that we would like to retrieve every police car $o$ qualifying the next predicate with a chance no less than ${30}\%$ : it is currently within 1 mile from the cab $q$ having licence number NY3852. Here,the precise locations of $o$ and $q$ are unknown, but obey pdfs created as explained in Figure 1(a). Intuitively, for every possible location $x$ of $q$ ,there is a query region which is a circle centering at $x$ ,and has a radius of 1 mile. A police car $o$ qualifies the spatial predicate,as long as it falls in the circle. The complication, however, lies in the large number of search regions that must be considered: we have to exhaust the circles of all the $x$ ,in order to calculate the overall probability that the distance between $o$ and $q$ is bounded by 1 mile.

当查询区域本身不确定时，范围搜索会变得更加“模糊”。假设我们希望检索每辆以不低于 ${30}\%$ 的概率满足下一个谓词的警车 $o$ ：它目前距离车牌号为 NY3852 的出租车 $q$ 在 1 英里以内。这里，$o$ 和 $q$ 的精确位置未知，但遵循如图 1(a) 所述创建的概率密度函数（pdf）。直观地说，对于 $q$ 的每个可能位置 $x$ ，都有一个以 $x$ 为圆心、半径为 1 英里的查询区域。只要警车 $o$ 落在这个圆内，就满足该空间谓词。然而，复杂之处在于必须考虑大量的搜索区域：为了计算 $o$ 和 $q$ 之间的距离在 1 英里以内的总概率，我们必须遍历所有 $x$ 对应的圆。

In general,given an uncertain object $q$ ,a probability threshold ${t}_{q}$ ,and a distance threshold ${\varepsilon }_{q}$ ,a probability thresholding fuzzy range query returns all the objects $o$ fulfilling $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)  \geq  {t}_{q}$ ,which represents the probability that $o$ is within distance ${\varepsilon }_{q}$ from $q$ (in the previous example, $q$ is the cab, ${t}_{q}$ is ${30}\%$ , and ${\varepsilon }_{q}$ equals 1 mile). Fuzzy search is also common in recommender systems. Consider that, in the scenario of Figure 1(b), the company has designed an investment plan that requires a principle of ${30}\mathrm{k}$ dollars,and its cold period may have a length of 2 (or 3) years with 75% (or 25%) probability (in practice,it may be difficult to conclude a unique cold period duration). Hence, the package is an uncertain object $q$ that can be described by a histogram analogous to a customer profile in Figure 1(b). Then, the manager can identify the potential interested clients, by issuing, on the customer-profile database, a fuzzy query with $q$ and suitable values for ${t}_{q}$ and ${\varepsilon }_{q}$ .

一般来说，给定一个不确定对象 $q$、一个概率阈值 ${t}_{q}$ 和一个距离阈值 ${\varepsilon }_{q}$，概率阈值模糊范围查询会返回所有满足 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)  \geq  {t}_{q}$ 的对象 $o$，$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)  \geq  {t}_{q}$ 表示 $o$ 与 $q$ 的距离在 ${\varepsilon }_{q}$ 以内的概率（在前面的例子中，$q$ 是出租车，${t}_{q}$ 是 ${30}\%$，${\varepsilon }_{q}$ 等于 1 英里）。模糊搜索在推荐系统中也很常见。考虑到，在图 1(b) 的场景中，公司设计了一个投资计划，需要 ${30}\mathrm{k}$ 美元的本金，其冷静期可能为 2 年（概率为 75%）或 3 年（概率为 25%）（实际上，可能很难确定一个唯一的冷静期时长）。因此，该投资计划是一个不确定对象 $q$，可以用类似于图 1(b) 中客户画像的直方图来描述。然后，经理可以通过在客户画像数据库上发出一个带有 $q$ 以及合适的 ${t}_{q}$ 和 ${\varepsilon }_{q}$ 值的模糊查询，来识别潜在的感兴趣客户。

### 1.2 Contributions and Article Organization

### 1.2 贡献与文章结构

Although range search on traditional "precise" data has been very well studied [Gaede and Gunther 1998], the existing methods are not applicable to uncertain objects, since they do not consider the probabilistic requirements [Cheng et al. 2004b]. As a result, despite its significant importance, multidimensional uncertain data currently cannot be efficiently manipulated. This article remedies the problem with a comprehensive analysis, which provides careful solutions to a wide range of issues in designing a fast mechanism for supporting (both nonfuzzy and fuzzy) range queries.

尽管对传统“精确”数据的范围搜索已经得到了深入研究 [Gaede 和 Gunther 1998]，但现有的方法并不适用于不确定对象，因为它们没有考虑概率要求 [Cheng 等人 2004b]。因此，尽管多维不确定数据非常重要，但目前无法对其进行高效处理。本文通过全面分析解决了这个问题，为设计支持（非模糊和模糊）范围查询的快速机制时遇到的一系列广泛问题提供了细致的解决方案。

The core of our techniques is a novel concept of "probabilistically constrained rectangles" (PCRs), which are concise summaries of objects' probabilistic modeling. In terms of functionalities, PCRs are similar to minimum bounding rectangles (MBR) in spatial databases. Specifically, they permit the development of an economical filter step, which prunes/validates a majority of the nonqualifying/qualifying data. Therefore, the subsequent refinement phase only needs to inspect a small number of objects, by invoking more expensive procedures (which, in our context, include loading an object's pdf, and/or calculating its qualification probability). As expected, the pruning/validating heuristics with PCRs are considerably more complicated (than those with MBRs), due to the higher complexity of uncertain objects (than spatial data).

我们技术的核心是“概率约束矩形”（Probabilistically Constrained Rectangles，PCRs）这一新颖概念，它是对象概率建模的简洁概括。在功能方面，PCRs 类似于空间数据库中的最小边界矩形（Minimum Bounding Rectangles，MBR）。具体来说，它们允许开发一个经济的过滤步骤，该步骤可以修剪/验证大部分不符合/符合条件的数据。因此，后续的细化阶段只需要通过调用更昂贵的程序（在我们的上下文中，包括加载对象的概率密度函数（pdf）和/或计算其符合条件的概率）来检查少量对象。正如预期的那样，由于不确定对象的复杂性高于空间数据，使用 PCRs 的修剪/验证启发式方法比使用 MBRs 的方法要复杂得多。

As a second step, we propose the U-tree, an index structure on multidimensional uncertain objects that is optimized to reduce the I/O cost of range queries. The U-tree leverages the properties of PCRs to effectively prune the subtrees

第二步，我们提出了 U 树，这是一种针对多维不确定对象的索引结构，经过优化以降低范围查询的 I/O 成本。U 树利用 PCRs 的特性来有效修剪

Range Search on Multidimensional Uncertain Data - Article 15 that cannot have any query result, and thus, limits the scope of search to a fraction of the database. The new access method is fully dynamic, and allows an arbitrary sequence of object insertions and deletions.

不可能有任何查询结果的子树，从而将搜索范围限制在数据库的一小部分。这种新的访问方法是完全动态的，允许任意顺序的对象插入和删除。

Finally, we accompany our algorithmic findings with a thorough performance analysis. Our theoretical results reveal valuable insight into the problem characteristics, and mathematically confirm the effectiveness of the proposed algorithms. In particular, we derive cost models that accurately quantify the overhead of range retrieval, and can be utilized by a query optimizer for tuning the parameters of a U-tree, and seeking a suitable execution plan.

最后，我们对算法的研究结果进行了全面的性能分析。我们的理论结果揭示了问题特征的宝贵见解，并从数学上证实了所提出算法的有效性。特别是，我们推导出了能够准确量化范围检索开销的成本模型，查询优化器可以利用这些模型来调整 U 树的参数，并寻找合适的执行计划。

The rest of the article is organized as follows. Section 2 formally defines probabilistic thresholding range search. Section 3 introduces PCRs and elaborates the pruning/validating strategies for nonfuzzy queries, while Section 4 extends the heuristics to fuzzy retrieval. Section 5 clarifies the details of the U-tree, as well as the query algorithms. Section 6 presents the theoretical analysis about the effectiveness of our solutions, and applies the findings to U-tree optimization. Section 7 contains an extensive experimental evaluation that demonstrates the efficiency of the proposed techniques. Section 8 surveys the previous work related to ours, and Section 9 concludes the article with directions for future work.

本文的其余部分组织如下。第 2 节正式定义了概率阈值范围搜索。第 3 节介绍了 PCRs，并详细阐述了非模糊查询的修剪/验证策略，而第 4 节将这些启发式方法扩展到模糊检索。第 5 节阐明了 U 树的细节以及查询算法。第 6 节对我们的解决方案的有效性进行了理论分析，并将研究结果应用于 U 树的优化。第 7 节进行了广泛的实验评估，证明了所提出技术的效率。第 8 节回顾了与我们相关的先前工作，第 9 节总结了本文并指出了未来的工作方向。

## 2. PROBLEM DEfiNITIONS

## 2. 问题定义

We consider a $d$ -dimensional workspace,where each axis has a unit range $\left\lbrack  {0,1}\right\rbrack$ . Each uncertain object $o$ is associated with (i) a probability density function o.pdf(x),where $x$ is an arbitrary $d$ -dimensional point,and (ii) an uncertainty region o.ur,which confines the area where the actual location of $o$ could possibly reside. Specifically,the value of $o.{pdf}\left( x\right)$ equals 0 for any $x$ outside $o.{ur}$ , whereas ${\int }_{\text{our }}$ o.pdf $\left( x\right) {dx}$ equals 1 . The pdfs of different objects are mutually independent.

我们考虑一个$d$维的工作空间，其中每个轴的单位范围为$\left\lbrack  {0,1}\right\rbrack$。每个不确定对象$o$关联着：（i）一个概率密度函数o.pdf(x)，其中$x$是任意的$d$维点；（ii）一个不确定区域o.ur，它限定了$o$的实际位置可能所在的区域。具体而言，对于$o.{ur}$之外的任何$x$，$o.{pdf}\left( x\right)$的值等于0，而${\int }_{\text{our }}$ o.pdf $\left( x\right) {dx}$等于1。不同对象的概率密度函数相互独立。

We do not place any other constraint on objects' pdfs. In particular, various objects can have totally different pdfs, that is, an object may have a pdf of the uniform distribution, another could follow the Gaussian distribution, and yet another one could possess an irregular distribution that can only be described using a histogram (as in Figure 1). Furthermore, the uncertainty region of an object does not have to be convex, or can even be broken into multiple pieces (e.g., the object may appear inside two separate buildings, but not on the roads between the buildings).

我们不对对象的概率密度函数施加任何其他约束。特别地，不同对象可以具有完全不同的概率密度函数，即一个对象可能具有均匀分布（uniform distribution）的概率密度函数，另一个对象可能遵循高斯分布（Gaussian distribution），还有一个对象可能具有只能用直方图描述的不规则分布（如图1所示）。此外，对象的不确定区域不必是凸的，甚至可以分成多个部分（例如，对象可能出现在两栋独立的建筑物内，但不会出现在建筑物之间的道路上）。

Definition 1. Let $S$ be a set of uncertain objects. Given a query region ${r}_{q}$ , and a value ${t}_{q} \in  (0,1\rbrack$ ,a nonfuzzy probability thresholding range query returns all the objects $o \in  S$ such that $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)  \geq  {t}_{q}$ ,where $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ is the appearance probability of $o$ in ${r}_{q}$ ,and is computed as

定义1。设$S$为一组不确定对象。给定一个查询区域${r}_{q}$和一个值${t}_{q} \in  (0,1\rbrack$，一个非模糊概率阈值范围查询返回所有满足$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)  \geq  {t}_{q}$的对象$o \in  S$，其中$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$是$o$在${r}_{q}$中的出现概率，其计算方式为

$$
\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)  = {\int }_{{r}_{q} \cap  o.{ur}}o \cdot  {pdf}\left( x\right) {dx}. \tag{1}
$$

The polygon in Figure 2(a) illustrates the uncertainty region o.ur of an object $o$ ,and the rectangle corresponds to a query region ${r}_{q}$ . If the possible location of $o$ uniformly distributes inside o.ur, $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ equals the area of the intersection between o.ur and ${r}_{q}$ (i.e.,the hatched region). In general,when ${r}_{q}$ is an axis-parallel rectangle,we denote its projection on the $i$ th axis $\left( {1 \leq  i \leq  d}\right)$ as $\left\lbrack  {{r}_{q\left\lbrack  i\right\rbrack  , - },{r}_{q\left\lbrack  i\right\rbrack   + }}\right\rbrack$ . Such ${r}_{q}$ is the focus of our analysis,because queries with these search regions are predominant in reality [Gaede and Gunther 1998]. Nevertheless, as elaborated later, our techniques can be extended to query regions of other shapes.

图2（a）中的多边形表示对象$o$的不确定区域o.ur，矩形对应一个查询区域${r}_{q}$。如果$o$的可能位置在o.ur内均匀分布，$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$等于o.ur和${r}_{q}$的交集面积（即阴影区域）。一般来说，当${r}_{q}$是一个轴平行矩形时，我们将其在第$i$个轴$\left( {1 \leq  i \leq  d}\right)$上的投影记为$\left\lbrack  {{r}_{q\left\lbrack  i\right\rbrack  , - },{r}_{q\left\lbrack  i\right\rbrack   + }}\right\rbrack$。此类${r}_{q}$是我们分析的重点，因为在实际应用中，使用这些搜索区域的查询占主导地位[Gaede和Gunther 1998]。不过，正如后面所阐述的，我们的技术可以扩展到其他形状的查询区域。

<!-- Media -->

<!-- figureText: 0.ur o.ur $q.{ur}$ (b) A fuzzy query (under the ${L}_{2}$ norm) (a) A nonfuzzy query -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_5.jpg?x=545&y=338&w=712&h=320&r=0"/>

Fig. 2. Range search on uncertain data.

图2. 不确定数据上的范围搜索。

<!-- Media -->

Definition 2. Let $S$ be a set of uncertain objects,and $q$ be another uncertain object that does not belong to $S$ . Given a distance threshold ${\varepsilon }_{q}$ ,and a value ${t}_{q} \in  (0,1\rbrack$ ,a fuzzy probability thresholding range query returns all the objects $o \in  S$ such that $\mathop{\operatorname{Prfuzzy}}\limits_{\text{Pfazzy }}\left( {o,q,{\varepsilon }_{q}}\right)  \geq  {t}_{q}$ ,where $\mathop{\operatorname{Prfuzzy}}\limits_{\text{Pfazzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ is the probability that $o$ and $q$ have distance at most ${\varepsilon }_{q}$ . Formally,if we regard $o\left( q\right)$ as a random variable obeying a pdf o.pdf $\left( x\right) \left( {q \cdot  {pdf}\left( x\right) }\right)$ ,then

定义2. 设$S$为一组不确定对象，$q$为另一个不属于$S$的不确定对象。给定一个距离阈值${\varepsilon }_{q}$和一个值${t}_{q} \in  (0,1\rbrack$，模糊概率阈值范围查询会返回所有满足$\mathop{\operatorname{Prfuzzy}}\limits_{\text{Pfazzy }}\left( {o,q,{\varepsilon }_{q}}\right)  \geq  {t}_{q}$的对象$o \in  S$，其中$\mathop{\operatorname{Prfuzzy}}\limits_{\text{Pfazzy }}\left( {o,q,{\varepsilon }_{q}}\right)$是$o$与$q$的距离至多为${\varepsilon }_{q}$的概率。形式上，如果我们将$o\left( q\right)$视为一个服从概率密度函数o.pdf $\left( x\right) \left( {q \cdot  {pdf}\left( x\right) }\right)$的随机变量，那么

$$
\operatorname{Prfuzzy}\left( {o,q,{\varepsilon }_{q}}\right)  = \Pr \left\{  {\operatorname{dist}\left( {o,q}\right)  \leq  {\varepsilon }_{q}}\right\}   \tag{2}
$$

Since $o$ and $q$ are independent,it is not hard to see that Equation 2 can be re-written as

由于$o$和$q$相互独立，不难看出方程2可以改写为

$$
\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)  = {\int }_{x \in  q.{ur}}q \cdot  {pdf}\left( x\right)  \cdot  \mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right) {dx}. \tag{3}
$$

where $\mathop{\Pr }\limits_{\text{range }}$ is represented in Equation 1,and $\odot  \left( {x,{\varepsilon }_{q}}\right)$ is a circle that centers at point $x$ and has radius ${\varepsilon }_{q}$ . As an example,the left and right polygons in Figure 2(b) demonstrate the uncertainty regions of a data object $o$ and a query object $q$ . The figure also shows the $\odot  \left( {x,{\varepsilon }_{q}}\right)$ ,when $x$ lies at point $A$ and $B$ ,respectively. Again,for simplicity,assume that o.pdf follows a uniform distribution inside o.ur. The area of the upper (lower) hatched region equals the probability $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$ for $o$ and $q$ to have a distance at most ${\varepsilon }_{q}$ ,when $q$ is located at $x = A\left( B\right)$ . In order to calculate $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ ,(conceptually) we must examine the $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$ of all $x \in  q$ .ur.

其中$\mathop{\Pr }\limits_{\text{range }}$在方程1中表示，$\odot  \left( {x,{\varepsilon }_{q}}\right)$是以点$x$为圆心、半径为${\varepsilon }_{q}$的圆。例如，图2(b)中的左右多边形分别展示了数据对象$o$和查询对象$q$的不确定区域。该图还展示了当$x$分别位于点$A$和$B$时的$\odot  \left( {x,{\varepsilon }_{q}}\right)$。同样，为简单起见，假设o.pdf在o.ur内服从均匀分布。当$q$位于$x = A\left( B\right)$时，上（下）阴影区域的面积等于$o$和$q$的距离至多为${\varepsilon }_{q}$的概率$\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$。为了计算$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$，（从概念上讲）我们必须检查所有$x \in  q$.ur的$\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$。

Definition 2 is independent of the distance metric employed. Although our methodology is applicable to any metric, we derive concrete solutions for the ${L}_{\infty }$ and ${L}_{2}$ norms, ${}^{1}$ which are of particular importance in practice. Note that, under ${L}_{\infty }, \odot  \left( {x,{\varepsilon }_{q}}\right)$ is a square whose centroid falls at $x$ and has a side length $2{\varepsilon }_{q}$ .

定义2与所采用的距离度量无关。尽管我们的方法适用于任何度量，但我们针对${L}_{\infty }$和${L}_{2}$范数${}^{1}$推导出了具体的解决方案，这在实践中尤为重要。请注意，在${L}_{\infty }, \odot  \left( {x,{\varepsilon }_{q}}\right)$下是一个质心位于$x$且边长为$2{\varepsilon }_{q}$的正方形。

---

<!-- Footnote -->

${}^{1}$ Let ${x}_{1}$ and ${x}_{2}$ be two $d$ -dimensional points. Their distance under the ${L}_{\infty }$ norm is $\mathop{\max }\limits_{{i = 1}}^{d}\left| {{x}_{1}\left\lbrack  i\right\rbrack   - }\right|$ ${x}_{2}\left\lbrack  i\right\rbrack   \mid$ ,where ${x}_{1}\left\lbrack  i\right\rbrack$ and ${x}_{2}\left\lbrack  i\right\rbrack$ are the $i$ -th coordinates of ${x}_{1}$ and ${x}_{2}$ ,respectively. The distance under the ${L}_{2}$ norm is simply the length of the line segment connecting ${x}_{1}$ and ${x}_{2}$ .

${}^{1}$ 设 ${x}_{1}$ 和 ${x}_{2}$ 为两个 $d$ 维点。它们在 ${L}_{\infty }$ 范数下的距离为 $\mathop{\max }\limits_{{i = 1}}^{d}\left| {{x}_{1}\left\lbrack  i\right\rbrack   - }\right|$ ${x}_{2}\left\lbrack  i\right\rbrack   \mid$ ，其中 ${x}_{1}\left\lbrack  i\right\rbrack$ 和 ${x}_{2}\left\lbrack  i\right\rbrack$ 分别是 ${x}_{1}$ 和 ${x}_{2}$ 的第 $i$ 个坐标。在 ${L}_{2}$ 范数下的距离就是连接 ${x}_{1}$ 和 ${x}_{2}$ 的线段长度。

<!-- Footnote -->

---

Since all the queries are "probability thresholding", we will omit this phrase in referring to their names. Furthermore, when the query type is clear, we also use the term qualification probability for $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ or $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ . As mentioned in Section 1.2, we adopt a filter-refinement framework in query processing on uncertain data. Specifically, the filter step first retrieves a small candidate set, after which the refinement phase computes the exact qualification probabilities of all the objects in the set, and then produces the query result. Hence, the candidate set should be a superset of the final result, that is, any object excluded from the set must violate the spatial predicates with an adequately high chance.

由于所有查询都是“概率阈值查询”，我们在提及它们的名称时将省略该短语。此外，当查询类型明确时，我们也使用合格概率这一术语来表示 $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ 或 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ 。如第1.2节所述，我们在不确定数据的查询处理中采用过滤 - 细化框架。具体而言，过滤步骤首先检索一个小的候选集，然后细化阶段计算该集合中所有对象的精确合格概率，最后产生查询结果。因此，候选集应该是最终结果的超集，即从该集合中排除的任何对象必须有足够高的概率违反空间谓词。

As analyzed in Section 5.3, it is usually expensive to compute the qualification probabilities (typically, Eqs. (1) and (3) can be evaluated only numerically). Therefore, the first objective of the filter step is to prune as many nonqualifying objects as possible, using computationally economical operations, which involve only checking the topological relationships between two rectangles (i.e., whether they intersect, contain each other, or are disjoint). Meanwhile, the filter step also achieves another equally important objective: validating as many qualifying objects as possible, again by analyzing only rectangles' topological relationships. We need to invoke the expensive process of qualification probability evaluation, only if an object can be neither pruned nor validated.

正如第5.3节所分析的，计算合格概率通常代价高昂（通常，只能通过数值方法计算方程(1)和(3)）。因此，过滤步骤的首要目标是使用计算成本较低的操作尽可能多地修剪不合格对象，这些操作仅涉及检查两个矩形之间的拓扑关系（即它们是否相交、包含或不相交）。同时，过滤步骤还实现了另一个同样重要的目标：同样通过仅分析矩形的拓扑关系来验证尽可能多的合格对象。只有当一个对象既不能被修剪也不能被验证时，我们才需要调用代价高昂的合格概率评估过程。

The above discussion implies that, towards fast query processing, a crucial task is to efficiently derive tight lower and upper bounds for the qualification probability of an object. Specifically, the lower bound is for validating (i.e., an object is guaranteed to be a result, if the lower bound is at least the query probability threshold ${t}_{q}$ ),whereas the upper bound is for pruning. Next,we clarify how to obtain these bounds for each type of queries. Table I lists the symbols frequently used in our presentation (some symbols have not appeared so far, but will be introduced later).

上述讨论表明，为了实现快速查询处理，一项关键任务是有效地得出对象合格概率的紧密下界和上界。具体而言，下界用于验证（即，如果下界至少为查询概率阈值 ${t}_{q}$ ，则保证该对象是一个结果），而上界用于修剪。接下来，我们将阐明如何为每种类型的查询获得这些界。表I列出了我们表述中常用的符号（有些符号目前尚未出现，但将在后面介绍）。

## 3. NONFUZZY RANGE SEARCH

## 3. 非模糊范围搜索

In this section, we will discuss the fundamental properties of probabilistically constrained rectangles (PCR), particularly, how they can be applied to assist pruning and validating for nonfuzzy range queries. Unless specifically stated, all the queries have axis-parallel rectangular search regions (queries with general shapes of search areas are the topic of Section 3.5).

在本节中，我们将讨论概率约束矩形（PCR）的基本性质，特别是它们如何应用于非模糊范围查询的修剪和验证。除非特别说明，所有查询的搜索区域均为轴平行矩形（具有一般形状搜索区域的查询是第3.5节的主题）。

### 3.1 Intuition behind Probabilistically Constrained Rectangles

### 3.1 概率约束矩形的直观理解

A PCR of an object $o$ depends on a parameter $c \in  \left\lbrack  {0,{0.5}}\right\rbrack$ ,and hence,is represented as $o.{pcr}\left( c\right)$ . It is a $d$ -dimensional rectangle,obtained by pushing,respectively,each face of $o.{mbr}$ inward,until the appearance probability of $o$ in the area swept by the face equals $c$ . Figure 3(a) illustrates the construction of a 2D o.pcr(c), where the polygon represents the uncertainty region o.ur of o, and the dashed rectangle is the MBR of $o$ ,denoted as $o.{mbr}$ . The $o.{pcr}\left( c\right)$ ,which is the grey area,is decided by 4 lines ${l}_{\left\lbrack  1\right\rbrack   + },{l}_{\left\lbrack  1\right\rbrack   - },{l}_{\left\lbrack  2\right\rbrack   + }$ ,and ${l}_{\left\lbrack  2\right\rbrack   - }$ . Line ${l}_{\left\lbrack  1\right\rbrack   + }$ has the property that,the appearance probability of $o$ on the right of ${l}_{\left\lbrack  1\right\rbrack   + }$ (i.e.,the hatched area) is $c$ . Similarly, ${l}_{\left\lbrack  1\right\rbrack   - }$ is obtained in such a way that the appearance likelihood of $o$ on the left of ${l}_{\left\lbrack  1\right\rbrack   - }$ equals $c$ (it follows that the probability that $o$ lies between ${l}_{\left\lbrack  1\right\rbrack   - }$ and ${l}_{\left\lbrack  1\right\rbrack   + }$ is $\left. {1 - {2c}}\right)$ . Lines ${l}_{\left\lbrack  2\right\rbrack   + }$ and ${l}_{\left\lbrack  2\right\rbrack   - }$ are obtained in the same way, except that they horizontally partition o.ur.

对象 $o$ 的概率约束矩形（PCR）取决于参数 $c \in  \left\lbrack  {0,{0.5}}\right\rbrack$，因此表示为 $o.{pcr}\left( c\right)$。它是一个 $d$ 维矩形，通过分别将 $o.{mbr}$ 的每个面向内推，直到该面扫过的区域中 $o$ 的出现概率等于 $c$ 而得到。图 3(a) 展示了二维对象概率约束矩形（o.pcr(c)）的构建，其中多边形表示对象 o 的不确定区域 o.ur，虚线矩形是 $o$ 的最小边界矩形（MBR），表示为 $o.{mbr}$。灰色区域 $o.{pcr}\left( c\right)$ 由 4 条线 ${l}_{\left\lbrack  1\right\rbrack   + },{l}_{\left\lbrack  1\right\rbrack   - },{l}_{\left\lbrack  2\right\rbrack   + }$ 和 ${l}_{\left\lbrack  2\right\rbrack   - }$ 确定。线 ${l}_{\left\lbrack  1\right\rbrack   + }$ 具有这样的性质：$o$ 在 ${l}_{\left\lbrack  1\right\rbrack   + }$ 右侧（即阴影区域）的出现概率为 $c$。类似地，${l}_{\left\lbrack  1\right\rbrack   - }$ 的确定方式是 $o$ 在 ${l}_{\left\lbrack  1\right\rbrack   - }$ 左侧的出现概率等于 $c$（由此可知 $o$ 位于 ${l}_{\left\lbrack  1\right\rbrack   - }$ 和 ${l}_{\left\lbrack  1\right\rbrack   + }$ 之间的概率为 $\left. {1 - {2c}}\right)$）。线 ${l}_{\left\lbrack  2\right\rbrack   + }$ 和 ${l}_{\left\lbrack  2\right\rbrack   - }$ 以相同的方式获得，只是它们水平划分 o.ur。

<!-- Media -->

Table I. Frequently Used Symbols

表 I. 常用符号

<table><tr><td>Symbol</td><td>Description</td><td>Section of Definition</td></tr><tr><td>$d$</td><td>the workspace dimensionality</td><td>2</td></tr><tr><td>o.ur, o.pdf</td><td>the uncertainty region and pdf of $o$ ,</td><td>2</td></tr><tr><td>o.mbr</td><td>the MBR of o.ur</td><td>3.1</td></tr><tr><td>o.pcr(c),</td><td>a probabilistically constrained rectangle (PCR) of $o$ ,</td><td>3.2</td></tr><tr><td>$\left\lbrack  {o.{pc}{r}_{\left\lbrack  i\right\rbrack   - }\left( c\right) ,o.{pc}{r}_{\left\lbrack  i\right\rbrack   + }\left( c\right) }\right\rbrack$</td><td>the projection of the PCR on the $i$ -th dimension</td><td>3.2</td></tr><tr><td>${r}_{q},{t}_{q}$</td><td>the search region and probability threshold of a nonfuzzy query</td><td>2</td></tr><tr><td>${r}_{q},{\varepsilon }_{q}{t}_{q}$</td><td>the search region, distance threshold and probability threshold of a fuzzy query</td><td>2</td></tr><tr><td>Prange(o,r)</td><td>the probability of $o$ appearing in a region $r$</td><td>2</td></tr><tr><td>$\mathop{\Pr }\limits_{{fuzzy}}\left( {o,q,{\varepsilon }_{q}}\right)$</td><td>the probability for $o$ and $q$ to be within distance ${\varepsilon }_{q}$</td><td>2</td></tr><tr><td>${C}_{1},\ldots ,{C}_{m}$</td><td>the values of a U-catalog (in ascending order)</td><td>3.3</td></tr><tr><td>$U{B}_{range}\left( {o,r}\right) ,L{B}_{range}\left( {o,r}\right)$</td><td>an upper and a lower bound of ${\Pr }_{\text{range }}\left( {o,r}\right)$</td><td>3.4</td></tr><tr><td>$U{B}_{fuzzy}\left( {r,o,\varepsilon }\right) ,L{B}_{fuzzy}\left( {r,o,\varepsilon }\right)$</td><td>an upper and a lower bound of $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,\varepsilon }\right)$</td><td>4.2</td></tr><tr><td>e.mbr(c)</td><td>the MBR of the o.pcr( $c$ ) of all the objects $o$ in the subtree of an intermediate U-tree entry $e$</td><td>5.1</td></tr><tr><td>e. ${sl}\left( c\right)$</td><td>the minimum projection length on any dimension of o. $\operatorname{pcr}\left( c\right)$ of any object $o$ in the subtree of $e$</td><td>5.1</td></tr></table>

<table><tbody><tr><td>符号</td><td>描述</td><td>定义章节</td></tr><tr><td>$d$</td><td>工作空间维度</td><td>2</td></tr><tr><td>o.ur, o.pdf</td><td>$o$的不确定区域和概率密度函数</td><td>2</td></tr><tr><td>o.mbr</td><td>o.ur的最小边界矩形（MBR）</td><td>3.1</td></tr><tr><td>o.pcr(c)</td><td>$o$的概率约束矩形（PCR）</td><td>3.2</td></tr><tr><td>$\left\lbrack  {o.{pc}{r}_{\left\lbrack  i\right\rbrack   - }\left( c\right) ,o.{pc}{r}_{\left\lbrack  i\right\rbrack   + }\left( c\right) }\right\rbrack$</td><td>PCR在第$i$维上的投影</td><td>3.2</td></tr><tr><td>${r}_{q},{t}_{q}$</td><td>非模糊查询的搜索区域和概率阈值</td><td>2</td></tr><tr><td>${r}_{q},{\varepsilon }_{q}{t}_{q}$</td><td>模糊查询的搜索区域、距离阈值和概率阈值</td><td>2</td></tr><tr><td>Prange(o,r)</td><td>$o$出现在区域$r$中的概率</td><td>2</td></tr><tr><td>$\mathop{\Pr }\limits_{{fuzzy}}\left( {o,q,{\varepsilon }_{q}}\right)$</td><td>$o$和$q$的距离在${\varepsilon }_{q}$以内的概率</td><td>2</td></tr><tr><td>${C}_{1},\ldots ,{C}_{m}$</td><td>U目录的值（按升序排列）</td><td>3.3</td></tr><tr><td>$U{B}_{range}\left( {o,r}\right) ,L{B}_{range}\left( {o,r}\right)$</td><td>${\Pr }_{\text{range }}\left( {o,r}\right)$的上界和下界</td><td>3.4</td></tr><tr><td>$U{B}_{fuzzy}\left( {r,o,\varepsilon }\right) ,L{B}_{fuzzy}\left( {r,o,\varepsilon }\right)$</td><td>$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,\varepsilon }\right)$的上界和下界</td><td>4.2</td></tr><tr><td>e.mbr(c)</td><td>中间U树条目$e$子树中所有对象$o$的o.pcr($c$)的最小边界矩形（MBR）</td><td>5.1</td></tr><tr><td>e. ${sl}\left( c\right)$</td><td>$e$子树中任意对象$o$的o.$\operatorname{pcr}\left( c\right)$在任意维度上的最小投影长度</td><td>5.1</td></tr></tbody></table>

<!-- Media -->

PCRs can be used to prune or validate an object, without computing its accurate qualification probability. Let us assume that the grey box in Figure 3(a) is the o.pcr(0.1) of o. Figure 3(b) shows the same PCR and o.mbr again, together with the search region ${r}_{{q}_{1}}$ of a nonfuzzy range query ${q}_{1}$ whose probability threshold ${t}_{{q}_{1}}$ equals 0.9 . As ${r}_{{q}_{1}}$ does not fully contain o.pcr(0.1),we can immediately assert that $o$ cannot qualify ${q}_{1}$ . Indeed,since $o$ falls in the hatched region with probability 0.1,the appearance probability of $o$ in ${r}_{{q}_{1}}$ must be smaller than $1 - {0.1} = {0.9}$ . Figure 3(c) illustrates pruning the same object with respect to another query ${q}_{2}$ having ${t}_{{q}_{2}} = {0.1}$ . This time, $o$ is disqualified because ${r}_{{q}_{2}}$ does not intersect $o.{pcr}\left( {0.1}\right)$ (the pruning conditions are different for ${q}_{1}$ and ${q}_{2}$ ). In fact,since ${r}_{{q}_{2}}$ lies entirely on the right of ${l}_{\left\lbrack  1\right\rbrack   + }$ ,the appearance probability of $o$ in ${r}_{{q}_{2}}$ is definitely smaller than 0.1 .

概率约束矩形（PCRs）可用于修剪或验证对象，而无需计算其准确的合格概率。假设图3(a)中的灰色框是对象o的o.pcr(0.1)。图3(b)再次展示了相同的概率约束矩形（PCR）和对象o的最小边界矩形（o.mbr），以及一个非模糊范围查询${q}_{1}$的搜索区域${r}_{{q}_{1}}$，其概率阈值${t}_{{q}_{1}}$等于0.9。由于${r}_{{q}_{1}}$并未完全包含o.pcr(0.1)，我们可以立即断言$o$不符合${q}_{1}$的条件。实际上，由于$o$落在阴影区域的概率为0.1，$o$出现在${r}_{{q}_{1}}$中的出现概率必定小于$1 - {0.1} = {0.9}$。图3(c)展示了针对另一个具有${t}_{{q}_{2}} = {0.1}$的查询${q}_{2}$对同一对象进行修剪的情况。这次，$o$不符合条件，因为${r}_{{q}_{2}}$与$o.{pcr}\left( {0.1}\right)$不相交（${q}_{1}$和${q}_{2}$的修剪条件不同）。事实上，由于${r}_{{q}_{2}}$完全位于${l}_{\left\lbrack  1\right\rbrack   + }$的右侧，$o$在${r}_{{q}_{2}}$中的出现概率肯定小于0.1。

The second row of Figure 3 presents three situations where $o$ can be validated using o.pcr(0.1),with respect to queries ${q}_{3},{q}_{4},{q}_{5}$ having probability thresholds ${t}_{{q}_{3}} = {0.9},{t}_{{q}_{4}} = {0.8}$ ,and ${t}_{{q}_{5}} = {0.1}$ ,respectively. In Figure 3(d) (or Figure 3(f)), $o$ must satisfy ${q}_{3}$ (or ${q}_{5}$ ) due to the fact that ${r}_{{q}_{3}}$ (or ${r}_{{q}_{5}}$ ) fully covers the part of $o.{mbr}$ on the right (or left) of ${l}_{\left\lbrack  1\right\rbrack   - }$ ,which implies that the appearance probability of $o$ in the query region must be at least $1 - {0.1} = {0.9}$ (or 0.1),where 0.1 is the likelihood for $o$ to fall in the hatched area. Similarly,in Figure 3(e), $o$ definitely qualifies ${q}_{4}$ ,since ${r}_{{q}_{4}}$ contains the portion of $o.{mbr}$ between ${l}_{\left\lbrack  1\right\rbrack   - }$ and ${l}_{\left\lbrack  1\right\rbrack   + }$ ,where the appearance probability of $o$ equals $1 - {0.1} - {0.1} = {0.8}$ .

图3的第二行展示了三种情况，在这些情况中，针对概率阈值分别为${t}_{{q}_{3}} = {0.9},{t}_{{q}_{4}} = {0.8}$和${t}_{{q}_{5}} = {0.1}$的查询${q}_{3},{q}_{4},{q}_{5}$，可以使用o.pcr(0.1)来验证$o$。在图3(d)（或图3(f)）中，$o$必定满足${q}_{3}$（或${q}_{5}$），因为${r}_{{q}_{3}}$（或${r}_{{q}_{5}}$）完全覆盖了$o.{mbr}$在${l}_{\left\lbrack  1\right\rbrack   - }$右侧（或左侧）的部分，这意味着$o$在查询区域中的出现概率至少为$1 - {0.1} = {0.9}$（或0.1），其中0.1是$o$落在阴影区域的可能性。类似地，在图3(e)中，$o$肯定符合${q}_{4}$的条件，因为${r}_{{q}_{4}}$包含了$o.{mbr}$在${l}_{\left\lbrack  1\right\rbrack   - }$和${l}_{\left\lbrack  1\right\rbrack   + }$之间的部分，在该部分$o$的出现概率等于$1 - {0.1} - {0.1} = {0.8}$。

Article 15 / 9

第15条 / 9

<!-- Media -->

<!-- figureText: o.mbr o.mbr ${l}_{\left\lbrack  1\right\rbrack   + }$ ${l}_{\left\lbrack  1\right\rbrack   + }$ (b) Pruning $\left( {{t}_{{q}_{1}} = {0.9}}\right)$ (c) Pruning $\left( {{t}_{{q}_{2}} = {0.1}}\right)$ ${r}_{{q}_{4}}$ o.mbr ${l}_{\left\lbrack  1\right\rbrack   + }$ ${l}_{\left\lbrack  1\right\rbrack   - }$ (e) Validating $\left( {{t}_{{q}_{4}} = {0.8}}\right.$ , (f) Validating $\left( {{t}_{{q}_{5}} = {0.1}}\right.$ , 1-covering) 1-covering) ${l}_{\left\lbrack  2\right\rbrack  }$ ${l}_{\left\lbrack  2\right\rbrack  }$ ${l}_{\left\lbrack  1\right\rbrack   + }$ ${l}_{\left\lbrack  1\right\rbrack   + }$ (h) Validating $\left( {{t}_{{q}_{7}} = {0.7}}\right.$ , (i) Validating $\left( {{t}_{{q}_{8}} = {0.6}}\right.$ , 0-covering) 0-covering) o.mbr ${l}_{\left\lbrack  2\right\rbrack   + }$ ${l}_{\text{[2]-}}$ ${r}_{{q}_{1}}$ ${l}_{\left\lbrack  1\right\rbrack   - }$ ${l}_{\left\lbrack  1\right\rbrack   + }^{\prime }$ (a) Constructing a PCR of $o$ o.mbr o.mbr ${l}_{\left\lbrack  1\right\rbrack   - }$ (d) Validating $\left( {{t}_{{q}_{3}} = {0.9}}\right.$ , 1-covering) ${l}_{\left\lbrack  2\right\rbrack  }$ $D$ ${l}_{\left\lbrack  1\right\rbrack   - }$ o.mbr ${l}_{\left\lbrack  1\right\rbrack   - }$ (g) Validating $\left( {{t}_{{q}_{6}} = {0.8}}\right.$ , 0-covering) -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_8.jpg?x=484&y=337&w=858&h=952&r=0"/>

Fig. 3. Pruning/validating with a 2D probabilistically constrained rectangle.

图3. 使用二维概率约束矩形进行修剪/验证。

<!-- Media -->

The queries in Figures 3(d)-3(f) share a common property: the projection of the search region contains that of o.mbr along one (specifically, the vertical) dimension. Accordingly, we say that those queries 1-cover o.mbr. In fact, validation is also possible, even if a query 0 -covers o.mbr, namely, the projection of the query area does not contain that of o.mbr on any dimension. Next, we illustrate this using the third row of Figure 3,where the queries ${q}_{6},{q}_{7},{q}_{8}$ have probability thresholds ${t}_{{q}_{6}} = {0.8},{t}_{{q}_{7}} = {0.7}$ ,and ${t}_{{q}_{8}} = {0.6}$ ,respectively.

图3(d) - 3(f)中的查询具有一个共同特性：搜索区域的投影在一个（具体为垂直）维度上包含了对象o的最小边界矩形（o.mbr）的投影。因此，我们称这些查询1 - 覆盖了o.mbr。实际上，即使一个查询0 - 覆盖了o.mbr，即查询区域的投影在任何维度上都不包含o.mbr的投影，验证也是可行的。接下来，我们使用图3的第三行来说明这一点，其中查询${q}_{6},{q}_{7},{q}_{8}$的概率阈值分别为${t}_{{q}_{6}} = {0.8},{t}_{{q}_{7}} = {0.7}$和${t}_{{q}_{8}} = {0.6}$。

In Figure 3(g), $o$ is guaranteed to qualify ${q}_{6}$ ,since ${r}_{{q}_{6}}$ covers entirely the part of o.mbr outside the hatched area. Observe that the appearance probability of $o$ in the hatched area is at most 0.2 . To explain this,we decompose the area into three rectangles ${ABCD},{DCEF},{BCGH}$ ,and denote the probabilities for $o$ to lie in them as ${\rho }_{ABCD},{\rho }_{DCEF}$ ,and ${\rho }_{BCGH}$ ,respectively. By the definition of ${l}_{\left\lbrack  1\right\rbrack   - }$ ,we know that ${\rho }_{ABCD} + {\rho }_{DCEF} = {0.1}$ ,whereas,by ${l}_{\left\lbrack  2\right\rbrack   + }$ ,we have ${\rho }_{ABCD} + {\rho }_{BCGH} = {0.1}$ . Since ${\rho }_{ABCD},{\rho }_{DCEF}$ ,and ${\rho }_{BCGH}$ are nonnegative,it holds that ${\rho }_{ABCD} + {\rho }_{DCEF} + {\rho }_{BCGH} \leq$ 0.2. This,in turn,indicates that $o$ falls in ${r}_{{q}_{6}}$ with probability at least ${0.8}\left( { = {t}_{{q}_{6}}}\right)$ . With similar reasoning, it is not hard to verify that, in Figure 3(h) (Figure 3(i)), the appearance probability of $o$ in the hatched area is at most 0.3 (0.4),meaning that $o$ definitely satisfies ${q}_{7}\left( {q}_{8}\right)$ .

在图3(g)中，$o$ 必定满足 ${q}_{6}$，因为 ${r}_{{q}_{6}}$ 完全覆盖了对象最小边界矩形（o.mbr）中阴影区域以外的部分。观察可知，$o$ 在阴影区域内的出现概率至多为0.2。为解释这一点，我们将该区域分解为三个矩形 ${ABCD},{DCEF},{BCGH}$，并分别用 ${\rho }_{ABCD},{\rho }_{DCEF}$ 和 ${\rho }_{BCGH}$ 表示 $o$ 落在这些矩形内的概率。根据 ${l}_{\left\lbrack  1\right\rbrack   - }$ 的定义，我们知道 ${\rho }_{ABCD} + {\rho }_{DCEF} = {0.1}$，而根据 ${l}_{\left\lbrack  2\right\rbrack   + }$，我们有 ${\rho }_{ABCD} + {\rho }_{BCGH} = {0.1}$。由于 ${\rho }_{ABCD},{\rho }_{DCEF}$ 和 ${\rho }_{BCGH}$ 是非负的，所以有 ${\rho }_{ABCD} + {\rho }_{DCEF} + {\rho }_{BCGH} \leq$ ≤ 0.2。这反过来表明，$o$ 落在 ${r}_{{q}_{6}}$ 内的概率至少为 ${0.8}\left( { = {t}_{{q}_{6}}}\right)$。通过类似的推理，不难验证，在图3(h)（图3(i)）中，$o$ 在阴影区域内的出现概率至多为0.3（0.4），这意味着 $o$ 肯定满足 ${q}_{7}\left( {q}_{8}\right)$。

### 3.2 Formalization of PCRs

### 3.2 概率约束矩形（PCRs）的形式化定义

We are ready to formalize PCRs and the related pruning/validating rules.

我们准备对概率约束矩形（PCRs）以及相关的剪枝/验证规则进行形式化定义。

Definition 3. Given a value $c \in  \left\lbrack  {0,{0.5}}\right\rbrack$ ,the probabilistically constrained rectangle $o \cdot  {pcr}\left( c\right)$ of an uncertain object $o$ is a $d$ -dimensional rectangle,representable by a ${2d}$ -dimensional vector $\left\{  {o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( c\right) ,o.{pc}{r}_{\left\lbrack  1\right\rbrack   + }\left( c\right) ,\ldots ,o.{pc}{r}_{\left\lbrack  d\right\rbrack   - }\left( c\right) }\right.$ , $\left. {o.{\operatorname{pcr}}_{\left\lbrack  d\right\rbrack   + }\left( c\right) }\right\}$ ,where $\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right) ,o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right) }\right\rbrack$ is the projection of $o.{pcr}\left( c\right)$ on the $i$ th dimension. In particular, ${\operatorname{o.pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right)$ and ${\operatorname{o.pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right)$ satisfy

定义3. 给定一个值 $c \in  \left\lbrack  {0,{0.5}}\right\rbrack$，不确定对象 $o$ 的概率约束矩形 $o \cdot  {pcr}\left( c\right)$ 是一个 $d$ 维矩形，可用一个 ${2d}$ 维向量 $\left\{  {o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( c\right) ,o.{pc}{r}_{\left\lbrack  1\right\rbrack   + }\left( c\right) ,\ldots ,o.{pc}{r}_{\left\lbrack  d\right\rbrack   - }\left( c\right) }\right.$，$\left. {o.{\operatorname{pcr}}_{\left\lbrack  d\right\rbrack   + }\left( c\right) }\right\}$ 表示，其中 $\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right) ,o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right) }\right\rbrack$ 是 $o.{pcr}\left( c\right)$ 在第 $i$ 维上的投影。特别地，${\operatorname{o.pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right)$ 和 ${\operatorname{o.pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right)$ 满足

$$
{\int }_{x\left\lbrack  i\right\rbrack   \leq  o.{pc}{r}_{\left\lbrack  i\right\rbrack   - }\left( c\right) }o.{pdf}\left( x\right) {dx} = {\int }_{x\left\lbrack  i\right\rbrack   \geq  o.{pc}{r}_{\left\lbrack  i\right\rbrack   + }\left( c\right) }o.{pdf}\left( x\right) {dx} = c, \tag{4}
$$

where $x\left\lbrack  i\right\rbrack$ denotes the $i$ th coordinate of a $d$ -dimensional point $x$ .

其中 $x\left\lbrack  i\right\rbrack$ 表示一个 $d$ 维点 $x$ 的第 $i$ 个坐标。

o.pcr(c) can be computed by considering each individual dimension in turn. We illustrate this using Figure 3(a) but the idea extends to arbitrary dimensionality in a straightforward manner. To decide,for example,line ${l}_{\left\lbrack  1\right\rbrack   - }\left( {l}_{\left\lbrack  1\right\rbrack   + }\right)$ ,we resort to the cumulative density function o.cdf(y)of o.pdf(x)on the horizontal dimension. Specifically, $o.{cdf}\left( {x}_{1}\right)$ is the probability that $o$ appears on the left of a vertical line intersecting the axis at $y$ . Thus, ${l}_{\left\lbrack  1\right\rbrack   - }$ can be decided by solving $y$ from the equation $o.{cdf}\left( y\right)  = c$ ,and similarly, ${l}_{\left\lbrack  1\right\rbrack   + }$ from $o.{cdf}\left( y\right)  = 1 - c$ . When $o.{pdf}\left( x\right)$ is regular (e.g.,uniform or Gaussian),given a constant $c$ ,the $y$ satisfying $o.{cdf}\left( y\right)  = c$ can be obtained precisely. In any case,there is a standard sampling approach to evaluate the equation numerically. Specifically, let us randomly generate $s$ points ${x}_{1},{x}_{2},\ldots ,{x}_{s}$ in o.mbr,sorted in ascending order of their x-coordinates. Then,we slowly move a vertical line $l$ from the left edge of o.mbr to its right edge. Every time $l$ crosses a sample,we compute a value $v = \frac{vol}{{s}^{\prime }} \cdot  \mathop{\sum }\limits_{{i = 1}}^{{s}^{\prime }}o \cdot  {pdf}\left( {x}_{i}\right)$ ,where ${s}^{\prime }$ is the number of samples crossed so far,and ${vol}$ the area of the part of $o.{mbr}$ on the left of $l$ . As soon as $v$ exceeds $c$ ,and the solution of $o.{cdf}\left( y\right)  = c$ is taken as the x -coordinate of the last sample crossed by $l$ .

o.pcr(c) 可以通过依次考虑每个单独的维度来计算。我们使用图 3(a) 来说明这一点，但这个思路可以直接推广到任意维度。例如，要确定直线 ${l}_{\left\lbrack  1\right\rbrack   - }\left( {l}_{\left\lbrack  1\right\rbrack   + }\right)$，我们借助 o.pdf(x) 在水平维度上的累积分布函数 o.cdf(y)。具体来说，$o.{cdf}\left( {x}_{1}\right)$ 是 $o$ 出现在与轴相交于 $y$ 的垂直线左侧的概率。因此，可以通过从方程 $o.{cdf}\left( y\right)  = c$ 中求解 $y$ 来确定 ${l}_{\left\lbrack  1\right\rbrack   - }$，类似地，从 $o.{cdf}\left( y\right)  = 1 - c$ 中确定 ${l}_{\left\lbrack  1\right\rbrack   + }$。当 $o.{pdf}\left( x\right)$ 是规则的（例如，均匀分布或高斯分布）时，给定一个常数 $c$，可以精确地得到满足 $o.{cdf}\left( y\right)  = c$ 的 $y$。在任何情况下，都有一种标准的采样方法来对该方程进行数值评估。具体来说，让我们在 o.mbr 中随机生成 $s$ 个点 ${x}_{1},{x}_{2},\ldots ,{x}_{s}$，并按其 x 坐标升序排序。然后，我们将一条垂直线 $l$ 从 o.mbr 的左边缘缓慢移动到其右边缘。每次 $l$ 穿过一个样本时，我们计算一个值 $v = \frac{vol}{{s}^{\prime }} \cdot  \mathop{\sum }\limits_{{i = 1}}^{{s}^{\prime }}o \cdot  {pdf}\left( {x}_{i}\right)$，其中 ${s}^{\prime }$ 是到目前为止穿过的样本数量，${vol}$ 是 $o.{mbr}$ 在 $l$ 左侧部分的面积。一旦 $v$ 超过 $c$，就将 $o.{cdf}\left( y\right)  = c$ 的解作为 $l$ 穿过的最后一个样本的 x 坐标。

In general,for any $c$ and ${c}^{\prime }$ satisfying $0 \leq  c < {c}^{\prime } \leq  {0.5}$ ,o.pcr(c) always contains $o . {pcr}\left( {c}^{\prime }\right)$ . Specially, $o . {pcr}\left( 0\right)$ is the MBR of the uncertainty region of $o$ ,and $o \cdot  {pcr}\left( {0.5}\right)$ degenerates into a point. The next theorem summarizes the pruning rules.

一般来说，对于任何满足 $0 \leq  c < {c}^{\prime } \leq  {0.5}$ 的 $c$ 和 ${c}^{\prime }$，o.pcr(c) 总是包含 $o . {pcr}\left( {c}^{\prime }\right)$。特别地，$o . {pcr}\left( 0\right)$ 是 $o$ 不确定区域的最小边界矩形（MBR），并且 $o \cdot  {pcr}\left( {0.5}\right)$ 退化为一个点。下一个定理总结了剪枝规则。

THEOREM 1. Given a nonfuzzy range query with search region ${r}_{q}$ and probability threshold ${t}_{q}$ ,the following holds:

定理 1. 给定一个具有搜索区域 ${r}_{q}$ 和概率阈值 ${t}_{q}$ 的非模糊范围查询，以下结论成立：

(1) for ${t}_{q} > {0.5}$ ,o can be pruned,if ${r}_{q}$ does not contain o.pcr $\left( {1 - {t}_{q}}\right)$ ;

(1) 对于 ${t}_{q} > {0.5}$，如果 ${r}_{q}$ 不包含 o.pcr $\left( {1 - {t}_{q}}\right)$，则可以对 o 进行剪枝；

(2) for ${t}_{q} \leq  {0.5}$ ,o can be pruned,if ${r}_{q}$ does not intersect o.pcr $\left( {t}_{q}\right)$ .

(2) 对于 ${t}_{q} \leq  {0.5}$，如果 ${r}_{q}$ 与 o.pcr $\left( {t}_{q}\right)$ 不相交，则可以对 o 进行剪枝。

Proof. The proofs of all lemmas and theorems can be found in the appendix.

证明. 所有引理和定理的证明可以在附录中找到。

In general,given two $d$ -dimensional rectangles $r$ and ${r}^{\prime }$ ,we say that ${rl}$ - covers ${r}^{\prime }\left( {0 \leq  l \leq  d}\right)$ ,if there exist $l$ dimensions along which the projection of $r$ encloses that of ${r}^{\prime }$ . As a special case,if ${rd}$ -covers ${r}^{\prime }$ ,then $r$ contains the entire ${r}^{\prime }$ . Based on the notion of $l$ -covering,we present the validating rules as follows.

一般来说，给定两个$d$维矩形$r$和${r}^{\prime }$，如果存在$l$个维度，使得$r$在这些维度上的投影包含${r}^{\prime }$的投影，我们就称${rl}$覆盖${r}^{\prime }\left( {0 \leq  l \leq  d}\right)$。作为一种特殊情况，如果${rd}$覆盖${r}^{\prime }$，那么$r$包含整个${r}^{\prime }$。基于$l$覆盖的概念，我们给出以下验证规则。

THEOREM 2. Given a nonfuzzy range query $q$ with search region ${r}_{q}$ and probability threshold ${t}_{q}$ ,consider an object o whose o.pcr(0) is $l$ -covered by ${r}_{q}$ $\left( {1 \leq  l \leq  d}\right)$ . If $l = d$ ,of alls in ${r}_{q}$ with 100% likelihood. Otherwise,without loss of generality,assume that the projection of o.pcr(0)is not covered by that of ${r}_{q}$ on dimensions $1,2,\ldots ,d - l$ . Then:

定理2。给定一个非模糊范围查询$q$，其搜索区域为${r}_{q}$，概率阈值为${t}_{q}$，考虑一个对象o，其o.pcr(0)被${r}_{q}$ $\left( {1 \leq  l \leq  d}\right)$ $l = d$覆盖。如果$l = d$，则在${r}_{q}$中的所有元素都有100%的可能性。否则，不失一般性，假设o.pcr(0)在维度$1,2,\ldots ,d - l$上的投影不被${r}_{q}$的投影所覆盖。那么：

(1) for any ${t}_{q}$ ,o definitely satisfies $q$ ,if there exist $2\left( {d - l}\right)$ values ${c}_{i},{c}_{i}^{\prime }(1 \leq  i \leq$ $d - l$ ) such that ${t}_{q} \leq  1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$ ,and on the ith dimension $\left( {1 \leq  i \leq  d - l}\right)$ , $\left\lbrack  {{r}_{q\left\lbrack  i\right\rbrack   - },{r}_{q\left\lbrack  i\right\rbrack   + }}\right\rbrack$ contains $\left\lbrack  {o.{pc}{r}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{i}\right) ,o.{pc}{r}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{i}^{\prime }\right) }\right\rbrack$ ;

(1) 对于任意${t}_{q}$，如果存在$2\left( {d - l}\right)$个值${c}_{i},{c}_{i}^{\prime }(1 \leq  i \leq$ $d - l$)使得${t}_{q} \leq  1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$，并且在第i个维度$\left( {1 \leq  i \leq  d - l}\right)$上，$\left\lbrack  {{r}_{q\left\lbrack  i\right\rbrack   - },{r}_{q\left\lbrack  i\right\rbrack   + }}\right\rbrack$包含$\left\lbrack  {o.{pc}{r}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{i}\right) ,o.{pc}{r}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{i}^{\prime }\right) }\right\rbrack$，则对象o肯定满足$q$；

(2) for $l = d - 1$ and ${t}_{q} \leq  {0.5}$ ,o definitely satisfies $q$ ,if there exist values ${c}_{1}$ , ${c}_{1}^{\prime }$ with ${c}_{1} \leq  {c}_{1}^{\prime }$ such that ${t}_{q} \leq  {c}_{1}^{\prime } - {c}_{1}$ ,and $\left\lbrack  {{r}_{q\left\lbrack  1\right\rbrack   - },{r}_{q\left\lbrack  1\right\rbrack   + }}\right\rbrack$ contains either $\left\lbrack  {o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$ or $\left\lbrack  {o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}^{\prime }\right) ,o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}\right) }\right\rbrack$ .

(2) 对于$l = d - 1$和${t}_{q} \leq  {0.5}$，如果存在值${c}_{1}$、${c}_{1}^{\prime }$，且${c}_{1} \leq  {c}_{1}^{\prime }$使得${t}_{q} \leq  {c}_{1}^{\prime } - {c}_{1}$，并且$\left\lbrack  {{r}_{q\left\lbrack  1\right\rbrack   - },{r}_{q\left\lbrack  1\right\rbrack   + }}\right\rbrack$包含$\left\lbrack  {o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$或$\left\lbrack  {o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}^{\prime }\right) ,o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}\right) }\right\rbrack$，则对象o肯定满足$q$。

Theorem 2 generalizes the reasoning behind the validation performed in Figure 3. To illustrate Rule (1),let us review Figure 3(d),where ${r}_{{q}_{3}}1$ -covers o.pcr $\left( 0\right) \left( { = \text{o.mbr}}\right)$ . It is the horizontal dimension,denoted as dimension 1, on which $\operatorname{o.pcr}\left( 0\right)$ is not covered. There exist ${c}_{1} = 0$ and ${c}_{1}^{\prime } = {0.1}$ ,such that $\left\lbrack  {{r}_{q\left\lbrack  1\right\rbrack   - },{r}_{q\left\lbrack  1\right\rbrack   + }}\right\rbrack$ contains $\left\lbrack  {o \cdot  {\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o \cdot  {\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}^{\prime }\right) }\right\rbrack$ . Since ${t}_{{q}_{3}} = {0.9} \leq  1 - \left( {{c}_{1} + {c}_{1}^{\prime }}\right)  =$ 0.9,by Theorem 2, $o$ is guaranteed to satisfy ${q}_{3}$ .

定理2对图3中执行的验证背后的推理进行了推广。为了说明规则(1)，让我们回顾图3(d)，其中${r}_{{q}_{3}}1$ -覆盖o.pcr $\left( 0\right) \left( { = \text{o.mbr}}\right)$ 。在水平维度（记为维度1）上，$\operatorname{o.pcr}\left( 0\right)$ 未被覆盖。存在${c}_{1} = 0$ 和${c}_{1}^{\prime } = {0.1}$ ，使得$\left\lbrack  {{r}_{q\left\lbrack  1\right\rbrack   - },{r}_{q\left\lbrack  1\right\rbrack   + }}\right\rbrack$ 包含$\left\lbrack  {o \cdot  {\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o \cdot  {\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}^{\prime }\right) }\right\rbrack$ 。由于${t}_{{q}_{3}} = {0.9} \leq  1 - \left( {{c}_{1} + {c}_{1}^{\prime }}\right)  =$ 0.9，根据定理2，保证$o$ 满足${q}_{3}$ 。

As another example,consider Figure 3(h). Here, ${r}_{{q}_{7}}$ 0-covers 0.pcr(0). We can find ${c}_{1} = {c}_{1}^{\prime } = {0.1}$ for the horizontal axis,and ${c}_{2} = 0,{c}_{2}^{\prime } = {0.1}$ for the vertical axis (dimension 2),which fulfill the following condition: $\left\lbrack  {{r}_{q\left\lbrack  i\right\rbrack   - },{r}_{q\left\lbrack  i\right\rbrack   + }}\right\rbrack$ encloses $\left\lbrack  {{\text{ o.pcr }}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{i}\right) ,{\text{ o.pcr }}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{i}^{\prime }\right) }\right\rbrack$ for both $i = 1$ and2. As ${t}_{{q}_{7}} = {0.7} \leq  1 - \mathop{\sum }\limits_{{i = 1}}^{2}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)  = {0.7}$ , we can assert that $o$ satisfies ${q}_{7}$ according to Rule (1).

再举一个例子，考虑图3(h)。在这里，${r}_{{q}_{7}}$ 0 -覆盖0.pcr(0)。我们可以为水平轴找到${c}_{1} = {c}_{1}^{\prime } = {0.1}$ ，为垂直轴（维度2）找到${c}_{2} = 0,{c}_{2}^{\prime } = {0.1}$ ，它们满足以下条件：对于$i = 1$ 和2，$\left\lbrack  {{r}_{q\left\lbrack  i\right\rbrack   - },{r}_{q\left\lbrack  i\right\rbrack   + }}\right\rbrack$ 都包含$\left\lbrack  {{\text{ o.pcr }}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{i}\right) ,{\text{ o.pcr }}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{i}^{\prime }\right) }\right\rbrack$ 。由于${t}_{{q}_{7}} = {0.7} \leq  1 - \mathop{\sum }\limits_{{i = 1}}^{2}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)  = {0.7}$ ，根据规则(1)，我们可以断言$o$ 满足${q}_{7}$ 。

Rule (2) can be applied only if the query region(d - 1)-covers o.mbr. For instance,in Figure 3(f), ${r}_{{q}_{5}}$ does not cover o.mbr only on the horizontal dimension. We may set ${c}_{1},{c}_{1}^{\prime }$ to 0 and 0.1,respectively,such that $\left\lbrack  {{r}_{q\left\lbrack  1\right\rbrack   - },{r}_{q\left\lbrack  1\right\rbrack   + }}\right\rbrack$ includes $\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$ . Since ${t}_{{q}_{5}} \leq  {c}_{1}^{\prime } - {c}_{1} = {0.1}$ ,by Rule $2,o$ definitely qualifies ${r}_{{q}_{5}}$ .

规则 (2) 仅在查询区域对对象的最小边界矩形（o.mbr）进行 (d - 1) 覆盖时才可应用。例如，在图 3(f) 中，${r}_{{q}_{5}}$ 仅在水平维度上未覆盖 o.mbr。我们可以分别将 ${c}_{1},{c}_{1}^{\prime }$ 设置为 0 和 0.1，使得 $\left\lbrack  {{r}_{q\left\lbrack  1\right\rbrack   - },{r}_{q\left\lbrack  1\right\rbrack   + }}\right\rbrack$ 包含 $\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$。由于 ${t}_{{q}_{5}} \leq  {c}_{1}^{\prime } - {c}_{1} = {0.1}$，根据规则 $2,o$，${r}_{{q}_{5}}$ 肯定符合条件。

### 3.3 Heuristics with a Finite Number of PCRs

### 3.3 具有有限数量概率覆盖区域（PCR）的启发式方法

The effectiveness of Theorems 1 and 2 is maximized if we could precompute the PCRs of an object for all $c \in  \left\lbrack  {0,{0.5}}\right\rbrack$ . Since this is clearly impossible,for each object $o$ ,we obtain its $o \cdot  {pcr}\left( c\right)$ only at some predetermined values of $c$ ,which are common to all objects and constitute the $U$ -catalog. ${}^{2}$ We denote the values of the U-catalog as (in ascending order) ${C}_{1},{C}_{2},\ldots ,{C}_{m}$ ,where $m$ is the size of the catalog. In particular, ${C}_{1}$ is fixed to 0,that is, $o.{mbr} = o.{pcr}\left( {C}_{1}\right)$ is always captured.

如果我们能预先计算出对象在所有 $c \in  \left\lbrack  {0,{0.5}}\right\rbrack$ 情况下的概率覆盖区域（PCR），那么定理 1 和定理 2 的有效性将达到最大。显然这是不可能的，因此对于每个对象 $o$，我们仅在 $c$ 的一些预定值处获取其 $o \cdot  {pcr}\left( c\right)$，这些值对所有对象都是通用的，并构成了 $U$ 目录（U - catalog）。${}^{2}$ 我们将 U - 目录的值按升序表示为 ${C}_{1},{C}_{2},\ldots ,{C}_{m}$，其中 $m$ 是目录的大小。特别地，${C}_{1}$ 固定为 0，即 $o.{mbr} = o.{pcr}\left( {C}_{1}\right)$ 总是被捕获。

A problem, however, arises. Given an arbitrary query probability threshold ${t}_{q}$ ,the corresponding PCR needed for pruning/validating may not exist. For instance,in Figure 3(b),as mentioned earlier,disqualifying object $o$ for query ${q}_{1}$ requires $o.{pcr}\left( {0.1}\right)$ . Thus,the pruning cannot be performed if 0.1 is not in the U-catalog.

然而，会出现一个问题。给定任意查询概率阈值 ${t}_{q}$，用于剪枝/验证所需的相应概率覆盖区域（PCR）可能不存在。例如，在图 3(b) 中，如前所述，要排除查询 ${q}_{1}$ 中的对象 $o$ 需要 $o.{pcr}\left( {0.1}\right)$。因此，如果 0.1 不在 U - 目录中，就无法进行剪枝操作。

We solve this problem by applying the heuristics of Section 3.2 in a conservative way. Assuming a U-catalog with $m = 2$ values $\left\{  {{C}_{1} = 0,{C}_{2} = {0.25}}\right\}$ ,Figure 4 shows an example where the dashed rectangle is $o.{mbr}$ ,and the grey box is o. $\operatorname{pcr}\left( {C}_{2}\right)$ . Rectangle ${r}_{{q}_{1}}$ is the search region of ${q}_{1}$ whose probability threshold ${t}_{{q}_{1}}$ equals 0.8. Since o.pcr(0.25) is not contained in ${r}_{{q}_{1}}$ ,by Rule (1) of Theorem 1,o does not qualify ${q}_{1}$ even if the query probability threshold were $1 - {0.25} = {0.75}$ , let alone a larger value 0.8 .

我们通过保守地应用 3.2 节中的启发式方法来解决这个问题。假设 U - 目录有 $m = 2$ 个值 $\left\{  {{C}_{1} = 0,{C}_{2} = {0.25}}\right\}$，图 4 展示了一个示例，其中虚线矩形是 $o.{mbr}$，灰色框是 o. $\operatorname{pcr}\left( {C}_{2}\right)$。矩形 ${r}_{{q}_{1}}$ 是查询 ${q}_{1}$ 的搜索区域，其概率阈值 ${t}_{{q}_{1}}$ 等于 0.8。由于 o.pcr(0.25) 不包含在 ${r}_{{q}_{1}}$ 中，根据定理 1 的规则 (1)，即使查询概率阈值为 $1 - {0.25} = {0.75}$，对象 o 也不符合 ${q}_{1}$ 的条件，更不用说更大的值 0.8 了。

---

<!-- Footnote -->

${}^{2}$ U’ here reminds of the fact that the catalog is created for uncertain data.

${}^{2}$ 这里的 U’ 提醒我们该目录是为不确定数据创建的。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: o.per(0.25) o.mbr ${l}_{\left\lbrack  1\right\rbrack   + }$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_11.jpg?x=696&y=338&w=412&h=270&r=0"/>

Fig. 4. Pruning and validating using only o.pcr(0.25) and o.pcr(0).

图 4. 仅使用 o.pcr(0.25) 和 o.pcr(0) 进行剪枝和验证。

<!-- Media -->

Let us consider another query ${q}_{2}$ with ${t}_{{q}_{2}} = {0.7}$ and a search region ${r}_{{q}_{2}}$ shown in Figure 4. We can validate $o$ for ${q}_{2}$ by examining only $o.{mbr}$ and $o.{pcr}\left( {0.25}\right)$ . In fact,since ${r}_{{q}_{2}}$ completely covers the part of $o.{mbr}$ on the left of line ${l}_{\left\lbrack  1\right\rbrack   + }$ ,we can assert (by Rule (1) of Theorem 2) that $o$ appears in ${r}_{{q}_{2}}$ with a probability at least 0.75,which is larger than ${t}_{{q}_{2}}$ .

让我们考虑另一个查询${q}_{2}$，其中${t}_{{q}_{2}} = {0.7}$，以及图4所示的搜索区域${r}_{{q}_{2}}$。我们可以仅通过检查$o.{mbr}$和$o.{pcr}\left( {0.25}\right)$来验证${q}_{2}$的$o$。事实上，由于${r}_{{q}_{2}}$完全覆盖了$o.{mbr}$在线${l}_{\left\lbrack  1\right\rbrack   + }$左侧的部分，我们可以（根据定理2的规则(1)）断言$o$出现在${r}_{{q}_{2}}$中的概率至少为0.75，这一概率大于${t}_{{q}_{2}}$。

In general,given a finite number of PCRs,we can still prune an object $o$ ,if those PCRs allow us to verify that $o$ cannot appear in the query region ${r}_{q}$ even with a probability lower than or equal to the probability threshold ${t}_{q}$ . Based on this reasoning, the next theorem presents the adapted version of the heuristics in Theorem 1:

一般来说，给定有限数量的概率约束区域（PCRs），如果这些PCRs使我们能够验证对象$o$即使以低于或等于概率阈值${t}_{q}$的概率也不会出现在查询区域${r}_{q}$中，我们仍然可以剪枝该对象$o$。基于这一推理，下一个定理给出了定理1中启发式方法的改进版本：

THEOREM 3. Given a nonfuzzy range query with search region ${r}_{q}$ and probability threshold ${t}_{q}$ ,the following holds:

定理3. 给定一个具有搜索区域${r}_{q}$和概率阈值${t}_{q}$的非模糊范围查询，以下结论成立：

(1) for ${t}_{q} > 1 - {C}_{m}$ (recall that ${C}_{m}$ is the largest value in the $U$ -catalog),o can be pruned,if ${r}_{q}$ does not contain o.pcr $\left( {c}_{ \vdash  }\right)$ ,where ${c}_{ \vdash  }$ is the smallest value in the $U$ -catalog that is at least $1 - {t}_{q}$ ;

(1) 对于${t}_{q} > 1 - {C}_{m}$（回想一下，${C}_{m}$是$U$ - 目录中的最大值），如果${r}_{q}$不包含对象o的概率约束区域o.pcr $\left( {c}_{ \vdash  }\right)$，则可以剪枝对象o，其中${c}_{ \vdash  }$是$U$ - 目录中至少为$1 - {t}_{q}$的最小值；

(2) for any ${t}_{q}$ ,o can be pruned,if ${r}_{q}$ does not intersect o.pcr $\left( {c}_{ \dashv  }\right)$ ,where ${c}_{ \dashv  }$ is the largest value in the catalog that is at most ${t}_{q}$ .

(2) 对于任何${t}_{q}$，如果${r}_{q}$与对象o的概率约束区域o.pcr $\left( {c}_{ \dashv  }\right)$不相交，则可以剪枝对象o，其中${c}_{ \dashv  }$是目录中至多为${t}_{q}$的最大值。

Similarly,using only $m$ PCRs,we may still validate an object $o$ ,as long as we can infer that $o$ falls in ${r}_{q}$ with a chance higher than or equal to ${t}_{q}$ . Actually,in this case, validating can still be performed using Theorem 2, except that all the ${c}_{i},{c}_{i}^{\prime }\left( {1 \leq  i \leq  d - l}\right)$ should be taken from the U-catalog,and Rule 2 is applied only if ${t}_{q} \leq  {C}_{m}$ (as opposed to ${t}_{q} \leq  {0.5}$ in Theorem 2).

类似地，仅使用$m$个概率约束区域（PCRs），只要我们能够推断出对象$o$落在${r}_{q}$中的概率大于或等于${t}_{q}$，我们仍然可以验证该对象$o$。实际上，在这种情况下，仍然可以使用定理2进行验证，只是所有的${c}_{i},{c}_{i}^{\prime }\left( {1 \leq  i \leq  d - l}\right)$都应从U - 目录中选取，并且仅当${t}_{q} \leq  {C}_{m}$时应用规则2（与定理2中的${t}_{q} \leq  {0.5}$相反）。

### 3.4 Computing the Lower and Upper Bounds of Qualification Probability

### 3.4 计算符合概率的上下界

Application of Theorem 3 is trivial,because both ${c}_{ \vdash  }$ and ${c}_{ \dashv  }$ are well-defined,and simple to identify. The utility of Theorem 2, however, is less straightforward, because the appropriate values of ${c}_{i},{c}_{i}^{\prime }\left( {1 \leq  i \leq  d - l}\right)$ for successful validation are not immediately clear. A naive method (of trying all the possibilities) may entail expensive overhead,because each ${c}_{i}$ or ${c}_{i}^{\prime }$ can be any value in the U-catalog,resulting in totally ${m}^{2\left( {d - l}\right) }$ possibilities.

定理3的应用很简单，因为${c}_{ \vdash  }$和${c}_{ \dashv  }$都有明确定义，并且易于识别。然而，定理2的实用性不那么直接，因为用于成功验证的${c}_{i},{c}_{i}^{\prime }\left( {1 \leq  i \leq  d - l}\right)$的合适值并不明确。一种简单的方法（尝试所有可能性）可能会带来高昂的开销，因为每个${c}_{i}$或${c}_{i}^{\prime }$都可以是U - 目录中的任何值，总共会产生${m}^{2\left( {d - l}\right) }$种可能性。

Range Search on Multidimensional Uncertain Data - Article 15 / 13

多维不确定数据的范围搜索 - 文章15 / 13

<!-- Media -->

---

Algorithm 1: Nonfuzzy-Range-Quali-Prob-Bounds $\left( {o,{r}_{q}}\right)$

算法1：非模糊范围符合概率边界$\left( {o,{r}_{q}}\right)$

/* $o$ is an uncertain object and ${r}_{q}$ a rectangle. The algorithm returns a lower bound $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$

/* $o$是一个不确定对象，${r}_{q}$是一个矩形。该算法返回一个下界$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$

and an upper bound $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ of $P{r}_{\text{range }}\left( {o,{r}_{q}}\right)$ (Equation 1). Both bounds tight as far as

以及$P{r}_{\text{range }}\left( {o,{r}_{q}}\right)$的一个上界$U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$（公式1）。就定理2和定理3而言，这两个边界都是紧的（见引理2）。

Theorems 2 and 3 are concerned (see Lemma 2). */

就定理2和定理3而言（见引理2）。 */

1. if ${r}_{q}$ fully covers $o.{mbr}$ then $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = 1$

1. 如果${r}_{q}$完全覆盖$o.{mbr}$，那么$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = 1$

2. else if ${r}_{q}$ is disjoint with $o.{mbr}$ then $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = 0$

2. 否则，如果${r}_{q}$与$o.{mbr}$不相交，那么$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = 0$

	else

	否则

	Lines 4-9 decide $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ */

	第4 - 9行确定$U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ */

			${c}_{ \leq  } =$ the smallest value $c$ in the U-catalog such that ${r}_{q}$ is disjoint with $o.{pcr}\left( c\right)$

			${c}_{ \leq  } =$是U - 目录中使得${r}_{q}$与$o.{pcr}\left( c\right)$不相交的最小的$c$值

	if ${c}_{ \leq  }$ exists then $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = {c}_{ \leq  } - \delta$ ,where $\delta$ is an infinitely small positive

	如果${c}_{ \leq  }$存在，那么$U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = {c}_{ \leq  } - \delta$ ，其中$\delta$是一个无穷小的正数

			else

			否则

					${c}_{ > } =$ the largest value $c$ in the U-catalog such that ${r}_{q}$ does not fully cover $o \cdot  {pcr}\left( c\right)$

					${c}_{ > } =$是U - 目录中使得${r}_{q}$不完全覆盖$o \cdot  {pcr}\left( c\right)$的最大的$c$值

					/* $c >$ always exists */

					/* $c >$总是存在 */

				${UB}_{\text{range }}\left( {o,{r}_{q}}\right)  = 1 - {c}_{ \geq  } - \delta$ ,where $\delta$ is as defined in Line 5

				${UB}_{\text{range }}\left( {o,{r}_{q}}\right)  = 1 - {c}_{ \geq  } - \delta$ ，其中$\delta$如第5行所定义

	The following lines decide $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ */

	以下几行确定$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ */

			$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = 0$

			$l =$ the number of dimensions on which the projection of ${r}_{q}$ contains that of $o.{mbr}$ ;

			$l =$是${r}_{q}$的投影包含$o.{mbr}$的投影的维度数量；

			without loss of generality,let these dimensions be $1,2,\ldots ,l$

			不失一般性，设这些维度为$1,2,\ldots ,l$

			if ${r}_{q}$ contains $o.{pcr}\left( {C}_{m}\right)$

			如果${r}_{q}$包含$o.{pcr}\left( {C}_{m}\right)$

					for $i = 1$ to $d - l$

					从 $i = 1$ 到 $d - l$

					/* consider, in turn, each "uncovered" dimension of o.mbr */

					/* 依次考虑对象o的最小边界矩形（MBR）的每个“未覆盖”维度 */

							${c}_{i} =$ the smallest value $c$ in the U-catalog such that ${r}_{q\left\lbrack  i\right\rbrack   - } \leq  o.{pc}{r}_{\left\lbrack  i\right\rbrack   - }\left( c\right)$

							${c}_{i} =$ 是U目录中满足 ${r}_{q\left\lbrack  i\right\rbrack   - } \leq  o.{pc}{r}_{\left\lbrack  i\right\rbrack   - }\left( c\right)$ 的最小值 $c$

							${c}_{i}^{\prime } =$ the smallest value $c$ in the U-catalog such that ${r}_{q\left\lbrack  i\right\rbrack   + } \geq  o.{pc}{r}_{\left\lbrack  i\right\rbrack   + }\left( c\right)$

							${c}_{i}^{\prime } =$ 是U目录中满足 ${r}_{q\left\lbrack  i\right\rbrack   + } \geq  o.{pc}{r}_{\left\lbrack  i\right\rbrack   + }\left( c\right)$ 的最小值 $c$

							/* ${c}_{i}$ and ${c}_{i}^{\prime }$ always exist */

							/* ${c}_{i}$ 和 ${c}_{i}^{\prime }$ 始终存在 */

					$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = \max \left\{  {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) ,1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right) }\right\}$

			else if $l = d - 1$

			否则，如果 $l = d - 1$

					${c}_{1} =$ the smallest value $c$ in the U-catalog such that ${r}_{q\left\lbrack  1\right\rbrack   - } \leq  o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( c\right)$ or

					${c}_{1} =$ 是U目录中满足 ${r}_{q\left\lbrack  1\right\rbrack   - } \leq  o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( c\right)$ 或者……的最小值 $c$

					${r}_{q\left\lbrack  1\right\rbrack   + } \geq  o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   + }\left( c\right)$ ; /* ${c}_{1}$ always exists */

					${r}_{q\left\lbrack  1\right\rbrack   + } \geq  o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   + }\left( c\right)$ ; /* ${c}_{1}$ 始终存在 */

					${c}_{1}^{\prime } =$ the largest value $c$ in the U-catalog such that ${r}_{q\left\lbrack  1\right\rbrack   + } \geq  o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( c\right)$ or

					${c}_{1}^{\prime } =$ 是U目录中满足 ${r}_{q\left\lbrack  1\right\rbrack   + } \geq  o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( c\right)$ 或者……的最大值 $c$

					${r}_{q\left\lbrack  1\right\rbrack   - } \leq  o \cdot  {pc}{r}_{\left\lbrack  1\right\rbrack   + }\left( c\right)$

					if ${c}_{1}^{\prime }$ exists then $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = \max \left\{  {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) ,{c}_{1}^{\prime } - {c}_{1}}\right\}$

					如果 ${c}_{1}^{\prime }$ 存在，则 $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = \max \left\{  {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) ,{c}_{1}^{\prime } - {c}_{1}}\right\}$

			Fig. 5. Finding a lower and an upper bound of an object's qualification probability.

			图5. 寻找对象合格概率的上下界。

---

<!-- Media -->

Figure 5 presents an algorithm that allows us to determine whether an object can be pruned/validated in $O\left( {m + d\log m}\right)$ time. Specifically,the algorithm returns an upper bound $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ and a lower bound $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ for the actual qualification probability $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ of an object $o$ . Given these bounds,we can prune $o$ if ${t}_{q} > U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ ,or validate $o$ if ${t}_{q} \leq$ $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ . Numerical calculation of $P{r}_{\text{range }}\left( {o,{r}_{q}}\right)$ is necessary if and only if ${t}_{q} \in  \left( {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) ,U{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right\rbrack$ .

图5展示了一种算法，该算法使我们能够在 $O\left( {m + d\log m}\right)$ 时间内确定一个对象是否可以被剪枝/验证。具体来说，该算法为对象 $o$ 的实际合格概率 $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ 返回一个上界 $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ 和一个下界 $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$。给定这些边界，如果 ${t}_{q} > U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$，我们可以对 $o$ 进行剪枝；如果 ${t}_{q} \leq$ $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$，则可以验证 $o$。当且仅当 ${t}_{q} \in  \left( {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) ,U{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right\rbrack$ 时，才需要对 $P{r}_{\text{range }}\left( {o,{r}_{q}}\right)$ 进行数值计算。

LEMMA 1. Let $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ and $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ be the values produced by ${Al}$ - gorithm 1 (Figure 5). For any query probability threshold ${t}_{q} > U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ , Theorem 3 always disqualifies of from appearing in ${r}_{q}$ with a probability at least ${t}_{q}$ . For any ${t}_{q} \leq  L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ ,Theorem 2 always validates o as an object appearing in ${r}_{q}$ with a probability at least ${t}_{q}$ .

引理1。设$U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$和$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$为由${Al}$ - 算法1（图5）产生的值。对于任意查询概率阈值${t}_{q} > U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$，定理3总能以至少${t}_{q}$的概率排除o出现在${r}_{q}$中。对于任意${t}_{q} \leq  L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$，定理2总能以至少${t}_{q}$的概率确认o是出现在${r}_{q}$中的对象。

<!-- Media -->

<!-- figureText: o.mbr (b) Reduction to rectangles ${l}_{\left\lbrack  1\right\rbrack   - }$ (a) Pruning possible for ${q}_{2}$ ,but not for ${q}_{1}\left( {{t}_{{q}_{1}} = {t}_{{q}_{2}} = {0.9}}\right.$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_13.jpg?x=462&y=339&w=883&h=302&r=0"/>

Fig. 6. Rationale of supporting circular search regions.

图6. 支持圆形搜索区域的原理。

<!-- Media -->

Having shown that Algorithm 1 is correct, we proceed with another lemma that confirms the tightness of the resulting lower and upper bounds, as far as our pruning and validating heuristics are concerned.

在证明了算法1的正确性之后，我们接着给出另一个引理，该引理确认了就我们的剪枝和验证启发式方法而言，所得上下界的紧性。

LEMMA 2. Let $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ and $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ be the values produced by ${Al}$ - gorithm 1. Theorem 3 cannot prune 0,if the query probability threshold ${t}_{q}$ does not exceed $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ . Similarly,Theorem 2 cannot validate o,if ${t}_{q}$ is higher than $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ .

引理2。设$U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$和$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$为由${Al}$ - 算法1产生的值。如果查询概率阈值${t}_{q}$不超过$U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$，定理3不能剪枝o。类似地，如果${t}_{q}$高于$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$，定理2不能验证o。

It is not hard to verify that the computation time of Algorithm 1 is $O(m +$ $d\log m$ ). In particular,except Lines 12-14,every other line incurs either $O\left( 1\right)$ or $O\left( m\right)$ cost. Each of Lines 13 and 14 can be implemented in $O\left( {\log m}\right)$ time, by performing a binary search, so that the for-loop initiated at Line 12 entails totally $O\left( {\left( {d - l}\right)  \cdot  \log m}\right)  = O\left( {d\log m}\right)$ overhead.

不难验证算法1的计算时间为$O(m +$ $d\log m$）。特别地，除了第12 - 14行，其他每一行的代价要么是$O\left( 1\right)$要么是$O\left( m\right)$。通过执行二分查找，第13行和第14行中的每一行都可以在$O\left( {\log m}\right)$时间内实现，因此从第12行开始的for循环总共会产生$O\left( {\left( {d - l}\right)  \cdot  \log m}\right)  = O\left( {d\log m}\right)$的开销。

### 3.5 Supporting Search Regions of Arbitrary Shapes

### 3.5 支持任意形状的搜索区域

The above sections assume axis-parallel rectangular regions. The pruning/validating heuristics presented so far, unfortunately, do not apply to queries with arbitrary shapes of search areas. For example, consider Figure 6(a), where the dashed and grey rectangles represent o.mbr and o.pcr(0.1), respectively. Circle ${r}_{{q}_{1}}$ is the search region of a query ${q}_{1}$ with probability threshold ${t}_{{q}_{1}} = {0.9}$ . Notice that ${r}_{{q}_{1}}$ does not fully cover o.pcr(0.1),and therefore,Theorem 1 would determine $o$ as nonqualifying. This decision,however,may be wrong,because $o$ could have a probability 0.1 falling in the hatched area, and probability 0.8 in the unhatched portion of ${r}_{{q}_{1}}$ .

上述各节假设搜索区域为轴平行矩形区域。遗憾的是，到目前为止所提出的剪枝/验证启发式方法并不适用于具有任意形状搜索区域的查询。例如，考虑图6(a)，其中虚线矩形和灰色矩形分别表示o.mbr和o.pcr(0.1)。圆${r}_{{q}_{1}}$是概率阈值为${t}_{{q}_{1}} = {0.9}$的查询${q}_{1}$的搜索区域。注意，${r}_{{q}_{1}}$并未完全覆盖o.pcr(0.1)，因此，定理1会判定$o$不符合条件。然而，这个判定可能是错误的，因为$o$可能有0.1的概率落在阴影区域，有0.8的概率落在${r}_{{q}_{1}}$的非阴影部分。

Interestingly, we can correctly prune/validate an object for a query with any search region ${r}_{q}$ ,utilizing directly our solutions for axis-parallel rectangles. For this purpose,we resort to a rectangle $r$ that contains ${r}_{q}$ ,and another rectangle ${r}^{\prime }$ that is fully enclosed in ${r}_{q}$ (Figure 6(b) demonstrates $r$ and ${r}^{\prime }$ for a circular ${r}_{q}$ ). Then,if an object $o$ appears in $r$ with a probability less than ${t}_{q}$ (the query probability threshold),then $o$ can be safely pruned. Likewise,if $o$ falls in ${r}^{\prime }$ with a chance at least ${t}_{q},o$ can be immediately validated.

有趣的是，我们可以利用我们针对轴平行矩形的解决方案，直接为具有任意搜索区域${r}_{q}$的查询正确地修剪/验证一个对象。为此，我们借助一个包含${r}_{q}$的矩形$r$，以及另一个完全包含在${r}_{q}$内的矩形${r}^{\prime }$（图6(b)展示了圆形${r}_{q}$对应的$r$和${r}^{\prime }$）。然后，如果一个对象$o$出现在$r$中的概率小于${t}_{q}$（查询概率阈值），那么可以安全地修剪掉$o$。同样地，如果$o$落在${r}^{\prime }$中的概率至少为${t}_{q},o$，则可以立即验证该对象。

In general,for any search area ${r}_{q}$ and uncertain object $o$ ,we can use $r$ and ${r}^{\prime }$ (obtained as described earlier) to calculate a range $\left\lbrack  {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right.$ , $\left. {U{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right\rbrack$ ,which is guaranteed to contain the actual probability $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ of $o$ appearing in ${r}_{q}$ . Specifically, $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ equals the upper bound returned by Algorithm 1 (Figure 5) with respect to $r$ ,whereas $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ is the lower bound produced by the algorithm for ${r}^{\prime }$ . The precise $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ needs to be derived,if and only if ${t}_{q}$ falls in $\left( {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right.$ , $\left. {U{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right\rbrack$ .

一般来说，对于任何搜索区域${r}_{q}$和不确定对象$o$，我们可以使用$r$和${r}^{\prime }$（如前文所述获得）来计算一个范围$\left\lbrack  {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right.$，$\left. {U{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right\rbrack$，该范围保证包含$o$出现在${r}_{q}$中的实际概率$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$。具体而言，$U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$等于算法1（图5）针对$r$返回的上界，而$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$是该算法针对${r}^{\prime }$产生的下界。当且仅当${t}_{q}$落在$\left( {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right.$，$\left. {U{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right\rbrack$范围内时，才需要推导出精确的$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$。

It remains to clarify the computation of $r$ and ${r}^{\prime }$ . Although any $r\left( {r}^{\prime }\right)$ containing (contained in) the search region ${r}_{q}$ guarantees the correctness of the query result, $r\left( {r}^{\prime }\right)$ should be as small (large) as possible,to maximize the effectiveness of the above approach. Obviously,the smallest $r$ is the minimum bounding rectangle of ${r}_{q}$ . On the other hand,when ${r}_{q}$ is a polygon,the ${r}^{\prime }$ with the largest area can be found using the algorithm in Daniels et al. [1997].

还需要阐明$r$和${r}^{\prime }$的计算方法。尽管任何包含（被包含在）搜索区域${r}_{q}$的$r\left( {r}^{\prime }\right)$都能保证查询结果的正确性，但$r\left( {r}^{\prime }\right)$应尽可能小（大），以最大化上述方法的有效性。显然，最小的$r$是${r}_{q}$的最小边界矩形。另一方面，当${r}_{q}$是一个多边形时，可以使用Daniels等人[1997]提出的算法找到面积最大的${r}^{\prime }$。

The above approach is especially useful when ${r}_{q}$ is a rather irregular region, such that it is difficult to test the topological relationships between ${r}_{q}$ and other geometric objects (we will use the approach to tackle fuzzy search in Section 4). However,approximating ${r}_{q}$ with $r$ and ${r}^{\prime }$ may be overly conservative, resulting in an $\left\lbrack  {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) ,U{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right\rbrack$ that may be unnecessarily wide. In fact,when ${r}_{q}$ has only limited complexity,we can achieve better pruning and validating effects by extending the analysis of the previous sections.

当${r}_{q}$是一个相当不规则的区域，以至于难以测试${r}_{q}$与其他几何对象之间的拓扑关系时（我们将在第4节中使用该方法处理模糊搜索），上述方法特别有用。然而，用$r$和${r}^{\prime }$来近似${r}_{q}$可能过于保守，导致$\left\lbrack  {L{B}_{\text{range }}\left( {o,{r}_{q}}\right) ,U{B}_{\text{range }}\left( {o,{r}_{q}}\right) }\right\rbrack$可能不必要地宽。实际上，当${r}_{q}$的复杂度有限时，我们可以通过扩展前面章节的分析来实现更好的修剪和验证效果。

Let us examine Figure 6(a) again,this time focusing on ${r}_{{q}_{2}}$ ,the search area of query ${q}_{2}$ with probability threshold ${t}_{{q}_{2}} = {0.9}$ (the semantics of the dashed and grey rectangles are the same as mentioned before). Observe that ${r}_{{q}_{2}}$ lies completely on the right of (and does not touch) line ${l}_{\left\lbrack  1\right\rbrack   - }$ . Since $o$ appears on the left of ${l}_{\left\lbrack  1\right\rbrack   - }$ with probability 0.1,it falls out of ${r}_{{q}_{2}}$ with at least ${90}\%$ likelihood; hence, $o$ can be eliminated. Note that pruning $o$ in this case essentially follows the rationale illustrated in Figure 3(b). Indeed, even for general search areas, the reasoning discussed in Section 3.1 of using PCRs for pruning/invalidating is still applicable. Based on such reasoning, we present the generic versions of Theorems 3 and 2.

让我们再次查看图6(a)，这次聚焦于${r}_{{q}_{2}}$，即查询${q}_{2}$在概率阈值${t}_{{q}_{2}} = {0.9}$下的搜索区域（虚线和灰色矩形的语义与之前所述相同）。观察可知，${r}_{{q}_{2}}$完全位于直线${l}_{\left\lbrack  1\right\rbrack   - }$的右侧（且不与之接触）。由于$o$以0.1的概率出现在${l}_{\left\lbrack  1\right\rbrack   - }$的左侧，它至少以${90}\%$的可能性落在${r}_{{q}_{2}}$之外；因此，可以排除$o$。注意，在这种情况下修剪$o$本质上遵循了图3(b)所示的原理。实际上，即使对于一般的搜索区域，第3.1节中讨论的使用概率约束区域（PCRs）进行修剪/无效化的推理仍然适用。基于这样的推理，我们给出定理3和定理2的通用版本。

THEOREM 4. Given a nonfuzzy range query with search region ${r}_{q}$ and probability threshold ${t}_{q}$ ,the following holds:

定理4. 给定一个具有搜索区域${r}_{q}$和概率阈值${t}_{q}$的非模糊范围查询，以下结论成立：

(1) for ${t}_{q} > 1 - {C}_{m}$ ,o can be pruned,if ${r}_{q}$ falls completely on the right (or left) of the plane containing the left (or right) face of o.pcr $\left( {c}_{ \vdash  }\right)$ on any dimension, where ${c}_{ \vdash  }$ is the smallest value in the $U$ -catalog that is at least $1 - {t}_{q}$ ;

(1) 对于${t}_{q} > 1 - {C}_{m}$，如果${r}_{q}$完全落在包含对象o的概率约束区域（PCR）$\left( {c}_{ \vdash  }\right)$左（或右）面的平面的右（或左）侧，其中${c}_{ \vdash  }$是$U$ - 目录中至少为$1 - {t}_{q}$的最小值，则可以修剪对象o；

(2) for any ${t}_{q}$ ,o can be pruned,if ${r}_{q}$ falls completely on the left (or right) of the plane containing the left (or right) face of $o.{pcr}\left( {c}_{ \dashv  }\right)$ ,where ${c}_{ \dashv  }$ is the largest value in the $U$ -catalog that is at most ${t}_{q}$ .

(2) 对于任何${t}_{q}$，如果${r}_{q}$完全落在包含$o.{pcr}\left( {c}_{ \dashv  }\right)$左（或右）面的平面的左（或右）侧，其中${c}_{ \dashv  }$是$U$ - 目录中至多为${t}_{q}$的最大值，则可以修剪对象o。

THEOREM 5. Given a nonfuzzy range query $q$ with search region ${r}_{q}$ and probability threshold ${t}_{q}$ ,o is guaranteed to satisfy ${r}_{q}$ in either of the following situations

定理5. 给定一个具有搜索区域${r}_{q}$和概率阈值${t}_{q}$的非模糊范围查询$q$，在以下任何一种情况下，对象o保证满足${r}_{q}$：

(1) we can find ${2d}$ values ${c}_{i},{c}_{i}^{\prime }\left( {1 \leq  i \leq  d}\right)$ in the $U$ -catalog,such that ${t}_{q} \leq$ $1 - \mathop{\sum }\limits_{{i = 1}}^{d}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$ ,and ${r}_{q}$ completely covers a $d$ -dimensional rectangler,whose projection on the $i$ -th dimension $\left( {1 \leq  i \leq  d}\right)$ is $\left\lbrack  {{\operatorname{o.pcr}}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{i}\right) ,{\operatorname{o.pcr}}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{i}^{\prime }\right) }\right\rbrack$ ;

(1) 我们可以在$U$ - 目录中找到${2d}$个值${c}_{i},{c}_{i}^{\prime }\left( {1 \leq  i \leq  d}\right)$，使得${t}_{q} \leq$ $1 - \mathop{\sum }\limits_{{i = 1}}^{d}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$，并且${r}_{q}$完全覆盖一个$d$维矩形，其在第$i$维上的投影$\left( {1 \leq  i \leq  d}\right)$为$\left\lbrack  {{\operatorname{o.pcr}}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{i}\right) ,{\operatorname{o.pcr}}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{i}^{\prime }\right) }\right\rbrack$；

(2) we can find a dimension $i \in  \left\lbrack  {1,d}\right\rbrack$ ,and 2 values $c,{c}^{\prime }$ in the U-catalog, such that ${t}_{q} \leq  {c}^{\prime } - c$ ,and ${r}_{q}$ completely covers a $d$ -dimensional rectangle $r$ ,whose projection on the $i$ -th dimension is $\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right) ,o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( {c}^{\prime }\right) }\right\rbrack$ or

(2) 我们可以在U目录中找到一个维度 $i \in  \left\lbrack  {1,d}\right\rbrack$ 以及两个值 $c,{c}^{\prime }$，使得 ${t}_{q} \leq  {c}^{\prime } - c$ 且 ${r}_{q}$ 完全覆盖一个 $d$ 维矩形 $r$，该矩形在第 $i$ 维上的投影为 $\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right) ,o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( {c}^{\prime }\right) }\right\rbrack$ 或者

$\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( {c}^{\prime }\right) ,o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right) }\right\rbrack$ ,and $r$ shares the same projection as $o.{mbr}$ on the other dimensions.

$\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( {c}^{\prime }\right) ,o.{\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right) }\right\rbrack$ 以及 $r$ 在其他维度上与 $o.{mbr}$ 具有相同的投影。

Theorems 4 and 5 can be employed, as long as it is feasible to check (i) whether ${r}_{q}$ falls completely on one side of an axis-parallel plane,and (ii) whether ${r}_{q}$ contains a $d$ -dimensional rectangle. The only problem,however,is that there does not exist an efficient algorithm for finding the set of ${2d}$ values in Rule (1) of Theorem 5,in order to perform successful validation (recall that,if ${r}_{q}$ is axis-parallel,we can use Algorithm 1 to validate an object $o$ in $O\left( {m + d\log m}\right)$ time,whenever $o$ can be validated by Theorem 2). A brute-force method (of attempting all possible ${c}_{i},{c}_{i}^{\prime }$ for $i \in  \left\lbrack  {1,d}\right\rbrack$ ) may not work,because each ${c}_{i}$ or ${c}_{i}^{\prime }$ can be any of the $m$ values in the U-catalog,rendering ${m}^{2d}$ possibilities. In practice, a useful trick for alleviating the problem is to restrict the number of these ${2d}$ values that are not zero to a small value $\alpha$ . In this case,there are only $\left( \begin{matrix} {2d} \\  \alpha  \end{matrix}\right)  \cdot  {m}^{\alpha }$ possibilities. For $\alpha  = 1$ or 2,it is computationally tractable to examine all of them. In any case, the correctness of the query result is never compromised,because if Theorem 5 cannot validate $o$ ,the actual qualification probability of $o$ will be calculated.

只要能够检查 (i) ${r}_{q}$ 是否完全落在一个轴平行平面的一侧，以及 (ii) ${r}_{q}$ 是否包含一个 $d$ 维矩形，就可以应用定理4和定理5。然而，唯一的问题在于，为了进行有效的验证，不存在一种高效的算法来找到定理5规则(1)中的 ${2d}$ 值集合（回想一下，如果 ${r}_{q}$ 是轴平行的，每当 $o$ 可以由定理2验证时，我们可以使用算法1在 $O\left( {m + d\log m}\right)$ 时间内验证对象 $o$）。一种暴力方法（尝试 $i \in  \left\lbrack  {1,d}\right\rbrack$ 的所有可能 ${c}_{i},{c}_{i}^{\prime }$）可能行不通，因为每个 ${c}_{i}$ 或 ${c}_{i}^{\prime }$ 可以是U目录中的任意 $m$ 个值，这会产生 ${m}^{2d}$ 种可能性。实际上，缓解这个问题的一个有用技巧是将这些非零的 ${2d}$ 值的数量限制为一个较小的值 $\alpha$。在这种情况下，只有 $\left( \begin{matrix} {2d} \\  \alpha  \end{matrix}\right)  \cdot  {m}^{\alpha }$ 种可能性。对于 $\alpha  = 1$ 或2，检查所有这些可能性在计算上是可行的。无论如何，查询结果的正确性永远不会受到影响，因为如果定理5无法验证 $o$，将计算 $o$ 的实际合格概率。

It is worth noting that the above technique may not be general enough to support any nonrectangular search regions efficiently, especially if the region has a complex shape. Processing such queries requires dedicated solutions beyond the scope of this article.

值得注意的是，上述技术可能不够通用，无法有效地支持任何非矩形搜索区域，特别是当该区域形状复杂时。处理此类查询需要本文范围之外的专门解决方案。

## 4. FUZZY RANGE SEARCH

## 4. 模糊范围搜索

We have shown in Section 3 that PCRs enable efficient pruning/validating for nonfuzzy range search. In the sequel, we will demonstrate that PCRs are also useful for fuzzy queries. As formulated in Definition 2, given an uncertain object $q$ ,a distance value ${\varepsilon }_{q}$ ,and a probability threshold ${t}_{q}$ ,such a query finds all the objects $o$ in a dataset satisfying $\mathop{\Pr }\limits_{{fuzzy}}\left( {o,q,{\varepsilon }_{q}}\right)  \geq  {t}_{q}$ ,where $\mathop{\Pr }\limits_{{fuzzy}}\left( {o,q,{\varepsilon }_{q}}\right)$ is given in Eq. (3). Section 4.1 provides the rationale behind our heuristics, which are formally presented in Section 4.2.

我们在第3节中已经表明，概率约束矩形（PCRs）能够为非模糊范围搜索实现高效的剪枝/验证。接下来，我们将证明PCRs对于模糊查询也很有用。如定义2所述，给定一个不确定对象 $q$、一个距离值 ${\varepsilon }_{q}$ 和一个概率阈值 ${t}_{q}$，这样的查询会找到数据集中满足 $\mathop{\Pr }\limits_{{fuzzy}}\left( {o,q,{\varepsilon }_{q}}\right)  \geq  {t}_{q}$ 的所有对象 $o$，其中 $\mathop{\Pr }\limits_{{fuzzy}}\left( {o,q,{\varepsilon }_{q}}\right)$ 由公式(3)给出。第4.1节提供了我们启发式方法背后的基本原理，这些方法将在第4.2节中正式介绍。

### 4.1 Intuition of Pruning and Validating

### 4.1 剪枝和验证的直觉

Evaluation of Eq. (3) is usually costly,especially if $q,o$ ,or both have irregular uncertainty regions and pdfs. Our objective is to prune or validate $o$ without going through the expensive evaluation. Next, we explain the underlying rationale,assuming that the distance metric employed is the ${L}_{\infty }$ norm; nevertheless, our discussion can be extended to the ${L}_{2}$ norm in a straightforward manner.

对公式(3)的计算通常成本较高，尤其是当$q,o$或两者都具有不规则的不确定区域和概率密度函数（pdf）时。我们的目标是在不进行昂贵计算的情况下对$o$进行剪枝或验证。接下来，我们将解释其基本原理，假设所采用的距离度量是${L}_{\infty }$范数；不过，我们的讨论可以直接扩展到${L}_{2}$范数。

Let us consider a query ${q}_{1}$ with probability threshold ${t}_{{q}_{1}} = {0.5}$ . Assume that we have already calculated ${q}_{1}$ . ${pcr}\left( {0.3}\right)$ and $o.{pcr}\left( {0.3}\right)$ ,which are the left and right grey boxes in Figure 7(a), respectively. The dashed rectangle ${ABCD}$ is the MBR (denoted as ${q}_{1}.{mbr}$ ) of the uncertainty region of ${q}_{1}$ . The parameter ${\varepsilon }_{{q}_{1}}$ of ${q}_{1}$ equals half of the side length of square ${r}_{1}$ or ${r}_{2}$ . By examining only ${q}_{1}.{mbr}$ and the two PCRs,we can assert that $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)$ is at most 0.42,and hence, $o$ can be safely eliminated. To explain this, we need to cut ${ABCD}$ into two disjoint

让我们考虑一个概率阈值为${t}_{{q}_{1}} = {0.5}$的查询${q}_{1}$。假设我们已经计算出了${q}_{1}$、${pcr}\left( {0.3}\right)$和$o.{pcr}\left( {0.3}\right)$，它们分别是图7(a)中的左右灰色框。虚线矩形${ABCD}$是${q}_{1}$不确定区域的最小边界矩形（MBR，记为${q}_{1}.{mbr}$）。${q}_{1}$的参数${\varepsilon }_{{q}_{1}}$等于正方形${r}_{1}$或${r}_{2}$边长的一半。仅通过检查${q}_{1}.{mbr}$和两个概率约束区域（PCR），我们可以断言$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)$至多为0.42，因此可以安全地排除$o$。为了解释这一点，我们需要将${ABCD}$切割成两个不相交的

Article 15 / 17 rectangles ${EBCF}$ and ${AEFD}$ ,and then rewrite Eq. 3 as:

第15/17条 矩形${EBCF}$和${AEFD}$，然后将公式3重写为：

$$
\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)  = {\int }_{x \in  {EBCF}}{q}_{1} \cdot  {pdf}\left( x\right)  \cdot  \mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right) {dx}
$$

$$
 + {\int }_{x \in  {AEFD}}{q}_{1}.{pdf}\left( x\right)  \cdot  \mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right) {dx}, \tag{5}
$$

<!-- Media -->

<!-- figureText: squares with side length ${2\varepsilon }{q}_{2}$ ${q}_{2}$ . mbr ${q}_{2}$ .pcr(0.3) o.mbr $A \circ$ $G\diamond$ ${D}^{\prime }$ o.pcr(0.3) ${r}_{2}$ (b) Validating $o\left( {{t}_{q} = {0.3}}\right)$ side length $2{\varepsilon }_{{q}_{1}}$ ${A}_{c}$ ${q}_{1}$ . mbr o.pcr(0.3) $D$ $C$ ${q}_{1}{pcr}\left( {0.3}\right)$ (a) Pruning $o\left( {{t}_{q} = {0.5}}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_16.jpg?x=521&y=338&w=785&h=404&r=0"/>

Fig. 7. Pruning/validating with PCRs for fuzzy queries (under the ${L}_{\infty }$ norm).

图7. 使用概率约束区域（PCR）对模糊查询进行剪枝/验证（在${L}_{\infty }$范数下）。

<!-- Media -->

where $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ represents a square that centers at point $x$ ,and has a side length of $2{\varepsilon }_{{q}_{1}}$ . Observe that,for any $x \in  {EBCF},\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right)$ must be bounded by 0.7,due to the fact that $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ does not fully cover o.pcr(0.3). For instance,rectangle ${r}_{1}$ illustrates the $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ when $x$ lies at point $B$ ; by Rule 1 of Theorem 1, $o$ appears in ${r}_{1}$ with a probability at most 0.7 . On the other hand,for any $x \in  {AEFD},P{r}_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right)$ never exceeds 0.3,because $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ does not intersect o.pcr(0.3). As an example,rectangle ${r}_{2}$ shows the $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ for $x = G$ ; according to Rule (2) of Theorem 1, $o$ falls in ${r}_{2}$ with no more than 0.3 probability. As a result:

其中 $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ 表示一个以点 $x$ 为中心、边长为 $2{\varepsilon }_{{q}_{1}}$ 的正方形。观察可知，对于任意 $x \in  {EBCF},\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right)$ 必定以 0.7 为界，这是因为 $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ 并未完全覆盖 o.pcr(0.3)。例如，矩形 ${r}_{1}$ 展示了当 $x$ 位于点 $B$ 时的 $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$；根据定理 1 的规则 1，$o$ 出现在 ${r}_{1}$ 中的概率至多为 0.7。另一方面，对于任意 $x \in  {AEFD},P{r}_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right)$ 绝不超过 0.3，因为 $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ 与 o.pcr(0.3) 不相交。例如，矩形 ${r}_{2}$ 展示了 $x = G$ 对应的 $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$；根据定理 1 的规则 (2)，$o$ 落入 ${r}_{2}$ 的概率不超过 0.3。因此：

$$
\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)  \leq  {0.7}{\int }_{x \in  {EBCF}}{q}_{1} \cdot  {pdf}\left( x\right) {dx} + {0.3}{\int }_{x \in  {AEFD}}{q}_{1} \cdot  {pdf}\left( x\right) {dx}
$$

$$
 = {0.7} \times  {0.3} + {0.3} \times  {0.7} = {0.42}\text{.} \tag{6}
$$

Let ${q}_{2}$ be another query with probability threshold ${t}_{{q}_{2}} = {0.3}$ . The left and right grey boxes in Figure 7(b) demonstrate ${q}_{2}{pcr}\left( {0.3}\right)$ and o.pcr(0.3),respectively,whereas the larger and smaller dashed rectangles capture ${q}_{2}.{mbr}$ and o.mbr,respectively. The parameter ${\varepsilon }_{{q}_{2}}$ of ${q}_{2}$ equals half of the side length of square ${r}_{1}$ or ${r}_{2}$ . Based on only the above information,we can claim that $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)  \geq  {0.3}$ ,and hence, $o$ can be validated. To clarify this,we again divide ${q}_{2}$ .mbr into rectangles ${EBCF}$ and ${AEFD}$ ,and scrutinize Eq. (5). Here,for any $x \in  {EBCF},P{r}_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right) }\right)$ is a constant 1,because $\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$ necessarily contains $o.{mbr}\left( {r}_{1}\right.$ illustrates an example of $\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$ for $\left. {x = E}\right)$ . However,when $x$ distributes in AEFD, $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right) }\right)$ may drop to 0,as is the case for ${r}_{2}$ , which is the $\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$ for $x = G$ . It follows that

设 ${q}_{2}$ 是另一个概率阈值为 ${t}_{{q}_{2}} = {0.3}$ 的查询。图 7(b) 中左侧和右侧的灰色框分别展示了 ${q}_{2}{pcr}\left( {0.3}\right)$ 和 o.pcr(0.3)，而较大和较小的虚线矩形分别表示 ${q}_{2}.{mbr}$ 和 o.mbr。${q}_{2}$ 的参数 ${\varepsilon }_{{q}_{2}}$ 等于正方形 ${r}_{1}$ 或 ${r}_{2}$ 边长的一半。仅基于上述信息，我们可以断言 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)  \geq  {0.3}$，因此，$o$ 可以得到验证。为了阐明这一点，我们再次将 ${q}_{2}$.mbr 划分为矩形 ${EBCF}$ 和 ${AEFD}$，并仔细研究方程 (5)。在此，对于任意 $x \in  {EBCF},P{r}_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right) }\right)$ 是常数 1，因为 $\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$ 必定包含 $o.{mbr}\left( {r}_{1}\right.$ 展示了 $\left. {x = E}\right)$ 对应的 $\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$ 的一个示例。然而，当 $x$ 分布在 AEFD 中时，$\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right) }\right)$ 可能降至 0，就像 ${r}_{2}$ 的情况一样，它是 $x = G$ 对应的 $\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$。由此可得

$$
\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)  \geq  1 \cdot  {\int }_{x \in  {EBCF}}{q}_{1} \cdot  {pdf}\left( x\right) {dx} + 0{\int }_{x \in  {AEFD}}{q}_{1} \cdot  {pdf}\left( x\right) {dx}
$$

$$
 = 1 \times  {0.3} + 0 \times  {0.7} = {0.3}\text{.} \tag{7}
$$

<!-- Media -->

<!-- figureText: ${q}_{1}$ . mbr squares with side length $2{\varepsilon }_{q}$ , ${q}_{2}$ . ${mbr}$ ${r}_{1}$ ${q}_{2}$ .pcr $\left( {0.3}\right) \mathrm{V}$ ${A}_{\mathrm{O}}$ o.mbr o.pcr(0.3) $H$ (b) Validating $o\left( {{t}_{q} = {0.4}}\right)$ o.mbr a square with side length ${2\varepsilon }{q}_{1}$ (a) Pruning $o\left( {{t}_{q} = {0.4}}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_17.jpg?x=518&y=339&w=767&h=387&r=0"/>

Fig. 8. Enhanced pruning/validating for fuzzy queries with more "slices" (under the ${L}_{\infty }$ norm).

图 8. 针对具有更多“切片”的模糊查询的增强剪枝/验证（在 ${L}_{\infty }$ 范数下）。

<!-- Media -->

In the above examples, we "sliced" $q.{mbr}$ into two rectangles for pruning and validating. In fact, stronger pruning/validation effects are possible by performing the slicing more aggressively. Assume that, instead of 0.5 , the query ${q}_{1}$ in Figure 7(a) has a lower ${t}_{{q}_{1}} = {0.4}$ ; hence, $o$ can no longer be disqualified as described with Inequality (6) (as ${0.42} > {t}_{{q}_{1}}$ ). However,we can actually derive a tighter upper bound 0.33 of $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)$ ,and thus,still eliminate $o$ . For this purpose,we should divide $q$ .mbr into three rectangles ${EBCF},{IEFJ}$ ,and ${AIJD}$ as in Figure 8(a),which repeats the content of Figure 7(a),except for including o.mbr (i.e.,the right-dashed box). Accordingly: $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)  =$

在上述示例中，我们将 $q.{mbr}$ “切片”成两个矩形以进行剪枝和验证。实际上，通过更积极地进行切片，可以获得更强的剪枝/验证效果。假设在图 7(a) 中，查询 ${q}_{1}$ 的下限不是 0.5，而是 ${t}_{{q}_{1}} = {0.4}$；因此，如不等式 (6) 所述，$o$ 不再被排除（因为 ${0.42} > {t}_{{q}_{1}}$）。然而，我们实际上可以推导出 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)$ 的更严格上界 0.33，从而仍然排除 $o$。为此，我们应将 $q$.mbr 划分为三个矩形 ${EBCF},{IEFJ}$ 和 ${AIJD}$，如图 8(a) 所示，该图重复了图 7(a) 的内容，只是包含了 o.mbr（即右侧虚线框）。相应地：$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)  =$

$$
{\int }_{x \in  {EBCF}}{q}_{1} \cdot  {pdf}\left( x\right)  \cdot  \mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right) {dx} + {\int }_{x \in  {IEFJ}}{q}_{1} \cdot  {pdf}\left( x\right)  \cdot  \mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right) {dx}
$$

$$
 + {\int }_{x \in  {AIJD}}{q}_{1} \cdot  {pdf}\left( x\right)  \cdot  \mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right) {dx}. \tag{8}
$$

As analyzed earlier with Figure 7(a),for any point $x \in  {EBCF}$ , $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right)  \leq  {0.7}$ ,whereas,for any point $x \in$ IEFJ $\subset  {ABCD}$ , $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right)  \leq  {0.3}$ . Furthermore,notice that,given any point $x \in  {AIJD}$ , $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)$ is always 0,because $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ is disjoint with $o.{mbr}$ . For instance,rectangle ${r}_{3}$ is the $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ when $x$ lies at $H$ ; evidently,it is impossible for $o$ to appear in ${r}_{3}$ . Therefore,Eq. $8 \geq$

如前文结合图 7(a) 所分析的，对于任意点 $x \in  {EBCF}$，有 $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right)  \leq  {0.7}$，而对于任意点 $x \in$ IEFJ $\subset  {ABCD}$，有 $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right) }\right)  \leq  {0.3}$。此外，请注意，对于任意点 $x \in  {AIJD}$，$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{1},{\varepsilon }_{{q}_{1}}}\right)$ 始终为 0，因为 $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$ 与 $o.{mbr}$ 不相交。例如，当 $x$ 位于 $H$ 时，矩形 ${r}_{3}$ 就是 $\odot  \left( {x,{\varepsilon }_{{q}_{1}}}\right)$；显然，$o$ 不可能出现在 ${r}_{3}$ 中。因此，公式 $8 \geq$

$$
{0.7}{\int }_{x \in  {EBCF}}{q}_{1}.{pdf}\left( x\right) {dx} + {0.3}{\int }_{x \in  {IEFJ}}{q}_{1}.{pdf}\left( x\right) {dx} + 0{\int }_{x \in  {AIJD}}{q}_{1}.{pdf}\left( x\right) {dx}
$$

$$
 = {0.7} \times  {0.3} + {0.3} \times  {0.4} + 0 \times  {0.3} = {0.33}\text{.} \tag{9}
$$

Similarly,suppose that the query ${q}_{2}$ in Figure 7(b) has a probability threshold ${t}_{{q}_{2}} = {0.4}$ ,in which case $o$ cannot be confirmed as a qualifying object with Inequality (7). Next, we will use Figure 8(b), where the grey and dashed rectangles have the same meaning as in Figure 7(b), to derive a new lower bound 0.42 of $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)$ ,which thus validates $o$ .

类似地，假设图 7(b) 中的查询 ${q}_{2}$ 具有概率阈值 ${t}_{{q}_{2}} = {0.4}$，在这种情况下，无法使用不等式 (7) 确认 $o$ 为符合条件的对象。接下来，我们将使用图 8(b)（其中灰色和虚线矩形的含义与图 7(b) 相同）来推导 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)$ 的新下界 0.42，从而验证 $o$。

Let us break ${q}_{2}.{mbr}$ into rectangles ${EBCF},{IEFJ}$ ,and ${AIJD}$ . Then, $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)$ can be represented as Eq. (8). Following the analysis that led to Inequality (7),we know that,for $x \in  {EBCF},P{r}_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)  = 1$ ,and,for $x \in  {AIJD}$ ,(obviously) $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)  \geq  0$ . The new observation here is that, for $x \in  {IEFJ},P{r}_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)  \geq  {0.3}$ ,since $\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$ always fully covers the hatched area in Figure 8(b), which is the part of o.mbr on the left of o.pcr(0.3). Rectangle ${r}_{3}$ shows an example of $\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$ when $x = H$ ; according to Rule (2) of Theorem 2, $o$ has a probability of at least 0.3 to lie in ${r}_{3}$ . Therefore,Eq. (8) $\geq$

让我们将${q}_{2}.{mbr}$分解为矩形${EBCF},{IEFJ}$和${AIJD}$。然后，$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)$可以表示为等式(8)。按照推导出不等式(7)的分析过程，我们知道，对于$x \in  {EBCF},P{r}_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)  = 1$，并且对于$x \in  {AIJD}$（显然）有$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)  \geq  0$。这里的新发现是，对于$x \in  {IEFJ},P{r}_{\text{fuzzy }}\left( {o,{q}_{2},{\varepsilon }_{{q}_{2}}}\right)  \geq  {0.3}$，由于$\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$总是完全覆盖图8(b)中的阴影区域，该区域是o.mbr在o.pcr(0.3)左侧的部分。矩形${r}_{3}$展示了当$x = H$时$\odot  \left( {x,{\varepsilon }_{{q}_{2}}}\right)$的一个示例；根据定理2的规则(2)，$o$位于${r}_{3}$中的概率至少为0.3。因此，等式(8) $\geq$

$$
1{\int }_{x \in  {EBCF}}{q}_{1}{pdf}\left( x\right) {dx} + {0.3}{\int }_{x \in  {IEFJ}}{q}_{1}{pdf}\left( x\right) {dx} + 0{\int }_{x \in  {AIJD}}{q}_{1}{pdf}\left( x\right) {dx}
$$

$$
 = 1 \times  {0.3} + {0.3} \times  {0.4} + 0 \times  {0.3} = {0.42}\text{.} \tag{10}
$$

### 4.2 Formal Results

### 4.2 正式结果

Before processing a fuzzy query $q$ ,we compute its $q \cdot  {pcr}\left( c\right)$ at ${m}_{q}$ values of $c$ in the range $\left\lbrack  {0,{0.5}}\right\rbrack$ . We denote these values as $Q{C}_{1},Q{C}_{2},\ldots ,Q{C}_{{m}_{a}}$ ,respectively, sorted in ascending order. Similar to the smallest value ${C}_{1}$ in the U-catalog, $Q{C}_{1}$ is fixed to 0,so that $q \cdot  {pcr}\left( {c}_{1}\right)$ always equals $q \cdot  {mbr}$ . Note that ${m}_{q}$ does not need to be the size $m$ of the U-catalog (we will examine the influence of ${m}_{q}$ in the experiments).

在处理模糊查询$q$之前，我们在$c$取值范围为$\left\lbrack  {0,{0.5}}\right\rbrack$的${m}_{q}$个值处计算其$q \cdot  {pcr}\left( c\right)$。我们将这些值分别记为$Q{C}_{1},Q{C}_{2},\ldots ,Q{C}_{{m}_{a}}$，并按升序排列。与U - 目录中的最小值${C}_{1}$类似，$Q{C}_{1}$固定为0，这样$q \cdot  {pcr}\left( {c}_{1}\right)$始终等于$q \cdot  {mbr}$。注意，${m}_{q}$不必是U - 目录的大小$m$（我们将在实验中研究${m}_{q}$的影响）。

Given an object $o$ ,we aim at obtaining an upper bound $U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ and a lower bound $L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ of $P{r}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ ,by studying only the PCRs of $q$ and $o\left( {\varepsilon }_{q}\right.$ is the distance threshold of $q$ ). Then, $o$ can be pruned if the query probability threshold ${t}_{q}$ is larger than $U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ ,or it can be validated if ${t}_{q}$ does not exceed $L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ . Accurate evaluation of $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ is performed only when both pruning and validating have failed.

给定一个对象$o$，我们的目标是仅通过研究$q$的PCR（$o\left( {\varepsilon }_{q}\right.$是$q$的距离阈值）来获得$P{r}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$的上界$U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$和下界$L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$。然后，如果查询概率阈值${t}_{q}$大于$U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$，则可以修剪$o$；如果${t}_{q}$不超过$L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$，则可以验证它。仅当修剪和验证都失败时，才对$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$进行精确评估。

We can reduce the computation of $U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ and $L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ to the derivation of upper and lower bounds of nonfuzzy range search qualification probabilities, which is solved by Algorithm 1 (Figure 5). In Section 4.2.1, we will first settle a related problem underlying the reduction, which is then clarified in Section 4.2.2.

我们可以将$U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$和$L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$的计算简化为非模糊范围搜索合格概率上下界的推导，这可通过算法1（图5）解决。在4.2.1节中，我们将首先解决简化过程中涉及的一个相关问题，然后在4.2.2节中进行详细说明。

4.2.1 Bounds of $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$ . Given any axis-parallel rectangle $r$ ,in the sequel,we will develop two values $U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ and $L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ satisfying

4.2.1 $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$的边界。给定任意轴平行矩形$r$，接下来，我们将推导出满足以下条件的两个值$U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$和$L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$

$$
L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)  \leq  \mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)  \leq  U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)  \tag{11}
$$

for any $x \in  r$ . To interpret $U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ and $L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ in a more intuitive manner,imagine that we examine all the circles $\odot  \left( {x,{\varepsilon }_{q}}\right)$ whose centers $x$ fall in $r$ ,and for every $\odot  \left( {x,{\varepsilon }_{q}}\right)$ ,record the probability $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$ that $o$ appears in $\odot  \left( {x,{\varepsilon }_{q}}\right)$ . Then, $U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ is a value never smaller than any $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$ ,and similarly, $L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ is never larger than any $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$ . A tighter range $\left\lbrack  {L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right) ,U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right) }\right\rbrack$ leads to stronger pruning/validating power, as will be clear shortly.

对于任意$x \in  r$。为了更直观地解释$U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$和$L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$，假设我们检查所有圆心$x$落在$r$内的圆$\odot  \left( {x,{\varepsilon }_{q}}\right)$，并且对于每个$\odot  \left( {x,{\varepsilon }_{q}}\right)$，记录$o$出现在$\odot  \left( {x,{\varepsilon }_{q}}\right)$中的概率$\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$。那么，$U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$是一个不小于任何$\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$的值，同样，$L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$是一个不大于任何$\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$的值。范围$\left\lbrack  {L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right) ,U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right) }\right\rbrack$越窄，剪枝/验证能力越强，这一点很快就会清楚。

We use $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ to denote the union of all the $\odot  \left( {x,{\varepsilon }_{q}}\right)$ with $x \in  r$ ,and $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ for the intersection of those circles. Depending on the distance metric deployed, $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ and $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ have different shapes. Next,we will discuss this about the ${L}_{\infty }$ and ${L}_{2}$ norms in 2D space,assuming that $r$ has side lengths $s{l}_{\left\lbrack  1\right\rbrack  }$ and $s{l}_{\left\lbrack  2\right\rbrack  }$ on the horizontal and vertical dimensions, respectively. The discussion can be directly extended to higher dimensionalities.

我们用$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$表示所有满足$x \in  r$的$\odot  \left( {x,{\varepsilon }_{q}}\right)$的并集，用$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$表示这些圆的交集。根据所采用的距离度量，$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$和$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$具有不同的形状。接下来，我们将讨论二维空间中${L}_{\infty }$和${L}_{2}$范数的情况，假设$r$在水平和垂直维度上的边长分别为$s{l}_{\left\lbrack  1\right\rbrack  }$和$s{l}_{\left\lbrack  2\right\rbrack  }$。该讨论可直接扩展到更高维度。

Consider the grey rectangle $r$ in Figure 9(a). Its $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ under the ${L}_{\infty }$ norm is rectangle ${ABCD}$ ,which shares the same centroid as $r$ ,and has side length $s{l}_{\left\lbrack  i\right\rbrack  } + 2{\varepsilon }_{q}$ on the $i$ th dimension $\left( {1 \leq  i \leq  2}\right)$ . Corner $A$ of $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ ,for instance,is decided by the $\odot  \left( {x,{\varepsilon }_{q}}\right)$ when $x$ lies at the upper-right corner of $r$ . Also focusing on ${L}_{\infty }$ ,Figure 9(b) demonstrates $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ ,which is also a rectangle ${ABCD}$ sharing the same centroid as $r$ ,but its side length is $2{\varepsilon }_{q} - s{l}_{\left\lbrack  i\right\rbrack  }$ along the $i$ th dimension $\left( {1 \leq  i \leq  2}\right)$ . In this case,corner $A$ is determined by the $\odot  \left( {x,{\varepsilon }_{q}}\right)$ when $x$ is located at the bottom-left corner of $r$ . In general, $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ does not exist if ${\varepsilon }_{q} <$ $\mathop{\max }\limits_{{i = 1}}^{d}\left( {s{l}_{\left\lbrack  i\right\rbrack  }/2}\right)$ .

考虑图9(a)中的灰色矩形$r$。它在${L}_{\infty }$范数下的$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$是矩形${ABCD}$，该矩形与$r$具有相同的质心，并且在第$i$维$\left( {1 \leq  i \leq  2}\right)$上的边长为$s{l}_{\left\lbrack  i\right\rbrack  } + 2{\varepsilon }_{q}$。例如，$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$的角$A$由当$x$位于$r$的右上角时的$\odot  \left( {x,{\varepsilon }_{q}}\right)$决定。同样关注${L}_{\infty }$，图9(b)展示了$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$，它也是一个与$r$具有相同质心的矩形${ABCD}$，但其在第$i$维$\left( {1 \leq  i \leq  2}\right)$上的边长为$2{\varepsilon }_{q} - s{l}_{\left\lbrack  i\right\rbrack  }$。在这种情况下，当$x$位于$r$的左下角时，角$A$由$\odot  \left( {x,{\varepsilon }_{q}}\right)$确定。一般来说，如果${\varepsilon }_{q} <$ $\mathop{\max }\limits_{{i = 1}}^{d}\left( {s{l}_{\left\lbrack  i\right\rbrack  }/2}\right)$，则$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$不存在。

<!-- Media -->

<!-- figureText: $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ $\Pi \left( {r,{\varepsilon }_{q}}\right)$ $s{l}_{\left\lbrack  2\right\rbrack  }$ $2{\varepsilon }_{q} - s{l}_{\left\lbrack  2\right\rbrack  }$ $\rightarrow  {12}{\varepsilon }_{q} - s{l}_{\left\lbrack  1\right\rbrack  } \leftarrow$ (b) $\Pi \left( {r,{\varepsilon }_{q}}\right)$ of ${L}_{\infty }$ $\Pi \left( {r,{\varepsilon }_{q}}\right)$ length ${\mathcal{E}}_{i}$ A ${DD}$ _____ $G$ (d) $\Pi \left( {r,{\varepsilon }_{q}}\right)$ of ${L}_{2}$ $2{\varepsilon }_{q} + s{l}_{\lbrack 1\rbrack }$ ${\varepsilon }_{q}$ ${\varepsilon }_{q}$ $2{\varepsilon }_{q} + s{l}_{\left\lbrack  2\right\rbrack  }$ $s{l}_{\left\lbrack  1\right\rbrack  }$ (a) $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ of ${L}_{\infty }$ $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ A $r$ $D$ $\breve{K}$ (c) $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ of ${L}_{2}$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_19.jpg?x=605&y=336&w=597&h=658&r=0"/>

Fig. 9. $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ and $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ under the ${L}_{\infty }$ and ${L}_{2}$ norms.

图9. $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$和$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$在${L}_{\infty }$和${L}_{2}$范数下的情况。

<!-- Media -->

As in Figure 9(c),under the ${L}_{2}$ norm, $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ has a more complex contour, which consists of line segments ${AB},{CD},{EF},{GH}$ ,as well as arcs ${BC},{DE},{FG}$ , and ${HA}$ . For example,arc ${HA}$ is formed by the $\odot  \left( {x,{\varepsilon }_{q}}\right)$ when $x$ is the upper-right corner of $r$ ; segment ${AB}$ is created jointly by all the $\odot  \left( {x,{\varepsilon }_{q}}\right)$ as $x$ moves on the upper edge of $r$ . We give the $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ of ${L}_{2}$ in Figure 9(d),whose boundary has 4 arcs ${AB},{BC},{CD}$ ,and ${DA}$ . Here, ${AB}$ is determined by the $\odot  \left( {x,{\varepsilon }_{q}}\right)$ when $x$ is positioned at the bottom-right corner of $r$ . As with its ${L}_{\infty }$ counterpart, $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ of ${L}_{2}$ is not always present; it exists only if ${\varepsilon }_{q} \geq  \frac{1}{2}{\left( \mathop{\sum }\limits_{{i = 1}}^{d}s{l}_{\left\lbrack  i\right\rbrack  }^{2}\right) }^{1/2}$ . In general,under any distance metric $L, \sqcup  \left( {r,{\varepsilon }_{q}}\right)$ is essentially the Minkowski sum [Berg et al. 2000] of $r$ and an $L$ -sphere with radius ${\varepsilon }_{q}$ .

如图9(c)所示，在${L}_{2}$范数下，$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$具有更复杂的轮廓，它由线段${AB},{CD},{EF},{GH}$以及圆弧${BC},{DE},{FG}$和${HA}$组成。例如，圆弧${HA}$是由当$x$位于$r$的右上角时的$\odot  \left( {x,{\varepsilon }_{q}}\right)$形成的；线段${AB}$是当$x$在$r$的上边缘移动时，所有$\odot  \left( {x,{\varepsilon }_{q}}\right)$共同形成的。我们在图9(d)中给出了${L}_{2}$的$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$，其边界有4条圆弧${AB},{BC},{CD}$和${DA}$。这里，当$x$位于$r$的右下角时，线段${AB}$由$\odot  \left( {x,{\varepsilon }_{q}}\right)$确定。与它的${L}_{\infty }$对应物一样，${L}_{2}$的$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$并不总是存在；只有当${\varepsilon }_{q} \geq  \frac{1}{2}{\left( \mathop{\sum }\limits_{{i = 1}}^{d}s{l}_{\left\lbrack  i\right\rbrack  }^{2}\right) }^{1/2}$时它才存在。一般来说，在任何距离度量下，$L, \sqcup  \left( {r,{\varepsilon }_{q}}\right)$本质上是$r$与半径为${\varepsilon }_{q}$的$L$ - 球的闵可夫斯基和（Minkowski sum，[Berg等人，2000]）。

LEMMA 3. Let $r$ be an axis-parallel rectangle. Given any point $x \in  r$ ,we have (for any distance metric):

引理3. 设$r$是一个轴平行矩形。给定任意点$x \in  r$，我们有（对于任何距离度量）：

$$
\mathop{\Pr }\limits_{\text{range }}\left( {o, \sqcap  \left( {r,{\varepsilon }_{q}}\right) }\right)  \leq  \mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)  \leq  \mathop{\Pr }\limits_{\text{range }}\left( {o, \sqcup  \left( {r,{\varepsilon }_{q}}\right) }\right) , \tag{12}
$$

where $\mathop{\Pr }\limits_{\text{range }}$ is defined in Eq. (1). Specially,if $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ does not exist,then $\mathop{\Pr }\limits_{\text{range }}\left( {o, \sqcap  \left( {r,{\varepsilon }_{q}}\right) }\right)  = 0.$

其中$\mathop{\Pr }\limits_{\text{range }}$在等式(1)中定义。特别地，如果$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$不存在，那么$\mathop{\Pr }\limits_{\text{range }}\left( {o, \sqcap  \left( {r,{\varepsilon }_{q}}\right) }\right)  = 0.$

We employ the above lemma to calculate $U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ and $L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ , which are defined in Inequality (11). Specifically, we invoke the Algorithm 1 (Figure 5) with the parameters $o$ and $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ ,and then set $U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ to the upper bound returned by the algorithm. Similarly, $L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ equals the lower bound produced by the same algorithm,when its parameters are $o$ and $\Pi \left( {r,{\varepsilon }_{q}}\right)$ .

我们使用上述引理来计算不等式(11)中定义的$U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$和$L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$。具体来说，我们使用参数$o$和$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$调用算法1（图5），然后将$U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$设为该算法返回的上界。类似地，当参数为$o$和$\Pi \left( {r,{\varepsilon }_{q}}\right)$时，$L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$等于同一算法产生的下界。

There remains, however, a subtle issue: the second parameter of Algorithm 1 must be a rectangle,whereas $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ and $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ may have irregular shapes under the ${L}_{2}$ norm (see Figures 9(c) and 9(d)). We remedy the problem using the approach explained with Figure 6(b). To derive $U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ of ${L}_{2}$ ,we pass, instead of $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ ,its MBR (e.g., ${IJKL}$ in Figure 9(c)) into the second parameter (notice that the MBR is essentially the $\sqcup  \left( {r,{\varepsilon }_{q}}\right)$ of ${L}_{\infty }$ ). Likewise,for computing $L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$ ,the parameter is set to an "inner rectangle" of $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ created as follows. For every corner $x$ of $r$ ,we connect it with its opposing corner using a line $l$ . Then,the intersection between $l$ and circle $\odot  \left( {x,{\varepsilon }_{q}}\right)$ becomes a corner of the inner rectangle. After all the corners of $r$ have been considered,the inner rectangle is fixed. As an example, in Figure 9(d), the inner rectangle of the illustrated $\sqcap  \left( {r,{\varepsilon }_{q}}\right)$ is ${EFGH}$ (e.g., $E$ is decided by considering the bottom-right corner of $r$ ).

然而，仍存在一个微妙的问题：算法1的第二个参数必须是一个矩形，而在${L}_{2}$范数下，$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$和$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$可能具有不规则的形状（见图9(c)和9(d)）。我们使用图6(b)中解释的方法来解决这个问题。为了推导${L}_{2}$的$U{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$，我们将$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$的最小边界矩形（MBR，例如图9(c)中的${IJKL}$）作为第二个参数传入，而不是$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$本身（注意，MBR本质上是${L}_{\infty }$的$\sqcup  \left( {r,{\varepsilon }_{q}}\right)$）。同样，为了计算$L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)$，将参数设置为按以下方式创建的$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$的“内矩形”。对于$r$的每个角点$x$，我们用一条直线$l$将其与对角相连。然后，$l$与圆$\odot  \left( {x,{\varepsilon }_{q}}\right)$的交点就成为内矩形的一个角点。在考虑了$r$的所有角点之后，内矩形就确定了。例如，在图9(d)中，所示$\sqcap  \left( {r,{\varepsilon }_{q}}\right)$的内矩形是${EFGH}$（例如，$E$是通过考虑$r$的右下角点确定的）。

4.2.2 Bounds of $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ . We are ready to explain how to determine the upper bound $U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ and lower bound $L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ of $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ . Recall that we have prepared ${m}_{q}$ PCRs: $q$ . ${pcr}\left( c\right)$ at $c =$ $Q{C}_{1},\ldots ,Q{C}_{{m}_{a}}$ . Let us consider the projections of these PCRs along the $i$ th dimension $\left( {1 \leq  i \leq  d}\right)  : \left\lbrack  {q \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right) ,q \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right) }\right\rbrack$ . We slice $q \cdot  {mbr}$ with $2\left( {{m}_{q} - 1}\right)$ planes,which are perpendicular to the $i$ th axis,and intersect this axis at $q \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right)$ or $q \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right)$ ,for every $c \in  \left\lbrack  {2,{m}_{q}}\right\rbrack$ . The slicing divides $q \cdot  {mbr}$ into $2{m}_{q} - 1$ disjoint rectangles ${r}_{1},{r}_{2},\ldots ,{r}_{2{m}_{q} - 1}$ ,sorted in ascending order of their projections on the $i$ th dimension.

4.2.2 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$的边界。我们准备解释如何确定$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$的上界$U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$和下界$L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$。回顾一下，我们已经准备了${m}_{q}$个平行坐标区域（PCR，Parallel Coordinate Region）：$q$。在$c =$ $Q{C}_{1},\ldots ,Q{C}_{{m}_{a}}$处的${pcr}\left( c\right)$。让我们考虑这些平行坐标区域沿着第$i$维$\left( {1 \leq  i \leq  d}\right)  : \left\lbrack  {q \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right) ,q \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right) }\right\rbrack$的投影。我们用$2\left( {{m}_{q} - 1}\right)$个平面来切割$q \cdot  {mbr}$，这些平面垂直于第$i$轴，并在每个$c \in  \left\lbrack  {2,{m}_{q}}\right\rbrack$对应的$q \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( c\right)$或$q \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( c\right)$处与该轴相交。切割将$q \cdot  {mbr}$划分为$2{m}_{q} - 1$个不相交的矩形${r}_{1},{r}_{2},\ldots ,{r}_{2{m}_{q} - 1}$，它们按照在第$i$维上的投影升序排列。

For instance,assume that ${m}_{q} = 2$ ,and $Q{C}_{1} = 0,Q{C}_{2} = {0.3}$ ; rectangle ${ABCD}$ and the grey box in Figure 8(a) illustrate $q \cdot  \operatorname{pcr}\left( {Q{C}_{1}}\right)$ and $q \cdot  \operatorname{pcr}\left( {Q{C}_{2}}\right)$ , respectively. Lines ${IJ}$ and ${EF}$ are the two slicing planes at $q.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {Q{C}_{2}}\right)$ and $q.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   + }\left( {\mathrm{{QC}}}_{2}\right)$ ,where the subscript 1 denotes the horizontal dimension. The result of the slicing is $2{m}_{q} - 1 = 3$ rectangles ${BCFE},{IEFJ}$ ,and ${AIJD}$ .

例如，假设${m}_{q} = 2$，且$Q{C}_{1} = 0,Q{C}_{2} = {0.3}$；图8(a)中的矩形${ABCD}$和灰色框分别表示$q \cdot  \operatorname{pcr}\left( {Q{C}_{1}}\right)$和$q \cdot  \operatorname{pcr}\left( {Q{C}_{2}}\right)$。直线${IJ}$和${EF}$是在$q.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {Q{C}_{2}}\right)$和$q.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   + }\left( {\mathrm{{QC}}}_{2}\right)$处的两个切割平面，其中下标1表示水平维度。切割的结果是$2{m}_{q} - 1 = 3$个矩形${BCFE},{IEFJ}$和${AIJD}$。

LEMMA 4. Let $q$ be a fuzzy query with distance threshold ${\varepsilon }_{q}$ ,whose $q$ .mbr has been partitioned into rectangles ${r}_{1},\ldots ,{r}_{2{m}_{a} - 1}$ on the ith dimension (for some $i \in  \left\lbrack  {1,d}\right\rbrack$ ) as described earlier. Then,for any object $o$ :

引理4。设$q$是一个距离阈值为${\varepsilon }_{q}$的模糊查询，如前所述，其$q$.mbr已在第i维（对于某个$i \in  \left\lbrack  {1,d}\right\rbrack$）上被划分为矩形${r}_{1},\ldots ,{r}_{2{m}_{a} - 1}$。那么，对于任何对象$o$：

$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)  \leq  \left( {1 - {2Q}{C}_{{m}_{q}}}\right)  \cdot  U{B}_{Pr}\left( {{r}_{{m}_{q}},o,{\varepsilon }_{q}}\right)  +$

$$
\mathop{\sum }\limits_{{i = 1}}^{{{m}_{q} - 1}}\left( {Q{C}_{i + 1} - Q{C}_{i}}\right)  \cdot  \left\lbrack  {U{B}_{Pr}\left( {{r}_{i},o,{\varepsilon }_{q}}\right)  + U{B}_{Pr}\left( {{r}_{2{m}_{q} - i},o,{\varepsilon }_{q}}\right) }\right\rbrack  . \tag{13}
$$

Similarly, $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)  \geq  \left( {1 - {2Q}{C}_{{m}_{q}}}\right)  \cdot  L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)  +$

同样地，$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)  \geq  \left( {1 - {2Q}{C}_{{m}_{q}}}\right)  \cdot  L{B}_{Pr}\left( {r,o,{\varepsilon }_{q}}\right)  +$

$$
\mathop{\sum }\limits_{{i = 1}}^{{{m}_{q} - 1}}\left( {Q{C}_{i + 1} - Q{C}_{i}}\right)  \cdot  \left\lbrack  {L{B}_{Pr}\left( {{r}_{i},o,{\varepsilon }_{q}}\right)  + L{B}_{Pr}\left( {{r}_{2{m}_{q} - i},o,{\varepsilon }_{q}}\right) }\right\rbrack  . \tag{14}
$$

The computation of functions $U{B}_{\text{range }}\left( \text{.}\right) {andL}{B}_{\text{range }}\left( \text{.}\right) {hasbeen}\;{discussedin}$ Section 4.2.1.

函数 $U{B}_{\text{range }}\left( \text{.}\right) {andL}{B}_{\text{range }}\left( \text{.}\right) {hasbeen}\;{discussedin}$ 的计算见4.2.1节。

The right-hand sides of Inequalities (13) and (14) can directly be taken as $U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ and $L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ ,respectively. Remember that,there is a set of these two inequalities for every dimension $i \in  \left\lbrack  {1,d}\right\rbrack$ . To tighten the range $\left\lbrack  {L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right) ,U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right) }\right\rbrack$ (for maximizing the pruning and validating power,as discussed at the beginning of Section 4.2),we set $U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ (or $L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ ) to the smallest (or largest) of the right-hand sides of the $d$ versions of Inequality (13) (or Inequality (14)) on the $d$ dimensions,respectively.

不等式(13)和(14)的右侧可以分别直接取为 $U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ 和 $L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$。请记住，对于每个维度 $i \in  \left\lbrack  {1,d}\right\rbrack$ 都有一组这样的两个不等式。为了收紧范围 $\left\lbrack  {L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right) ,U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right) }\right\rbrack$（如4.2节开头所讨论的，以最大化剪枝和验证能力），我们分别将 $U{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$（或 $L{B}_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$）设置为 $d$ 个维度上不等式(13)（或不等式(14)）的 $d$ 个版本右侧的最小值（或最大值）。

## 5. QUERY ALGORITHMS

## 5. 查询算法

The analysis in Section 3 (or 4) leads to an algorithm that processes a nonfuzzy (or fuzzy) query by scanning the entire dataset. Specifically, the filter step of the algorithm decides whether each object can be pruned, validated, or must be added to a candidate set,by analyzing the PCRs of $o$ (for fuzzy search,also the PCRs of the query object $q$ ). Then,the refinement phase calculates the accurate qualification probability of every object in the candidate set, to determine if it satisfies the query.

第3节（或第4节）的分析得出了一种通过扫描整个数据集来处理非模糊（或模糊）查询的算法。具体来说，该算法的过滤步骤通过分析 $o$ 的概率约束区域（PCR，Probability Constraint Region）（对于模糊搜索，还包括查询对象 $q$ 的PCR）来决定每个对象是否可以被剪枝、验证，或者必须添加到候选集中。然后，细化阶段计算候选集中每个对象的准确合格概率，以确定它是否满足查询条件。

This section achieves two objectives. First, we design an access method, called the U-tree, for multidimensional uncertain data (Section 5.1), and utilize it to reduce the cost of the filter step, by avoiding examination of all the objects (Section 5.2). Second, we clarify the details of refinement, and develop solutions with different trade-offs between precision and computation overhead (Section 5.3).

本节实现两个目标。首先，我们为多维不确定数据设计了一种称为U树（U-tree）的访问方法（5.1节），并利用它来避免检查所有对象，从而降低过滤步骤的成本（5.2节）。其次，我们阐明了细化的细节，并开发了在精度和计算开销之间进行不同权衡的解决方案（5.3节）。

### 5.1 The U-tree

### 5.1 U树

The U-tree is a balanced external-memory structure, where each node occupies a disk page. We number the levels of the tree in a bottom-up manner; namely, if the tree has a height of $h$ ,then all the leaf nodes are at level 0,whereas the root is at level $h - 1$ . Each entry in a leaf node corresponds an object $o$ . This entry keeps (i) o. ${pcr}\left( c\right)$ for the values of $c$ in the U-catalog: ${C}_{1}\left( { = 0}\right) ,\ldots ,{C}_{m}$ (recall that o. ${pcr}\left( 0\right)$ equals $o.{mbr}$ ),and (ii) a descriptor about $o.{pdf}$ ,whose information depends on the complexity of o.pdf. Specifically, if o.pdf is simple (i.e., a common distribution with a regular uncertainty region o.ur), then the descriptor contains all the details of o.pdf. Otherwise, the descriptor is a pointer that references a disk address where the representation of o.pdf (e.g., a histogram) is stored; in this case,additional I/Os are required to retrieve $o.{pdf}$ after the leaf entry has been found.

U树是一种平衡的外部内存结构，其中每个节点占用一个磁盘页面。我们以自底向上的方式对树的层级进行编号；即，如果树的高度为 $h$，那么所有叶节点位于第0层，而根节点位于第 $h - 1$ 层。叶节点中的每个条目对应一个对象 $o$。该条目保存 (i) 对象o在U目录中 $c$ 的值 ${pcr}\left( c\right)$：${C}_{1}\left( { = 0}\right) ,\ldots ,{C}_{m}$（回想一下，o. ${pcr}\left( 0\right)$ 等于 $o.{mbr}$），以及 (ii) 关于 $o.{pdf}$ 的描述符，其信息取决于对象o的概率密度函数（o.pdf）的复杂度。具体来说，如果o.pdf很简单（即，具有规则不确定性区域o.ur的常见分布），那么描述符包含o.pdf的所有细节。否则，描述符是一个指针，指向存储o.pdf表示（例如，直方图）的磁盘地址；在这种情况下，在找到叶条目后需要额外的输入/输出（I/O）操作来检索 $o.{pdf}$。

Let $e$ be a level-1 entry,that is, $e$ is the parent entry of a leaf node. Without loss of generality,assume that the node has $f$ objects ${o}_{1},\ldots ,{o}_{f}$ . In addition to a pointer to its child node, $e$ also retains (i) $m$ rectangles $e.{mbr}\left( {C}_{1}\right) \ldots ,e.{mbr}\left( {C}_{m}\right)$ , where,for any $c$ in the U-catalog, $e.{mbr}\left( c\right)$ is the MBR of ${o}_{1}.{pcr}\left( c\right) ,\ldots ,{o}_{f}.{pcr}\left( c\right)$ , and (ii) $m$ values $\operatorname{e.}{sl}\left( {C}_{1}\right) ,\ldots ,e.{sl}\left( {C}_{m}\right)$ such that $e.{sl}\left( c\right)$ (here,"sl" means side length) equals the length of the shortest projection of ${o}_{1}$ . $\operatorname{pcr}\left( c\right) ,\ldots ,{o}_{f}$ . $\operatorname{pcr}\left( c\right)$ along all dimensions.

设$e$为一级条目，即$e$是一个叶节点的父条目。不失一般性，假设该节点有$f$个对象${o}_{1},\ldots ,{o}_{f}$。除了指向其子节点的指针外，$e$还保留了：(i) $m$个矩形$e.{mbr}\left( {C}_{1}\right) \ldots ,e.{mbr}\left( {C}_{m}\right)$，其中，对于U目录中的任意$c$，$e.{mbr}\left( c\right)$是${o}_{1}.{pcr}\left( c\right) ,\ldots ,{o}_{f}.{pcr}\left( c\right)$的最小边界矩形（MBR）；(ii) $m$个值$\operatorname{e.}{sl}\left( {C}_{1}\right) ,\ldots ,e.{sl}\left( {C}_{m}\right)$，使得$e.{sl}\left( c\right)$（这里“sl”表示边长）等于${o}_{1}$、$\operatorname{pcr}\left( c\right) ,\ldots ,{o}_{f}$、$\operatorname{pcr}\left( c\right)$在所有维度上的最短投影长度。

Figure 10 provides an example that illustrates the information recorded in leaf and level-1 entries,assuming that the U-catalog has $m = 2$ values ${C}_{1} = 0$ and ${C}_{2} = {0.3}$ . The left- and right-dashed rectangles correspond to the MBRs of objects ${o}_{1}$ and ${o}_{2}$ ,(they are ${o}_{1}.{pcr}\left( 0\right)$ and ${o}_{2}.{pcr}\left( 0\right)$ ,respectively). The grey

图10给出了一个示例，说明了叶条目和一级条目中记录的信息，假设U目录有$m = 2$个值${C}_{1} = 0$和${C}_{2} = {0.3}$。左右虚线矩形分别对应对象${o}_{1}$和${o}_{2}$的最小边界矩形（它们分别是${o}_{1}.{pcr}\left( 0\right)$和${o}_{2}.{pcr}\left( 0\right)$）。灰色

Article 15 / 23 box inside ${o}_{1}.{mbr}$ (or ${o}_{2}.{mbr}$ ) is ${o}_{1}.{pcr}\left( {0.3}\right)$ (or ${o}_{2}.{pcr}\left( {0.3}\right)$ ),which is associated with the leaf entry of ${o}_{1}$ (or ${o}_{2}$ ). Consider a leaf node that contains only ${o}_{1}$ , ${o}_{2}$ ,and has $e$ as its parent entry at level 1 . Entry $e$ carries two rectangles e.mbr(0) and e.mbr(0.3). As shown in Figure 10, the former rectangle tightly bounds the MBRs of ${o}_{1}$ and ${o}_{2}$ ,while $e$ .mbr(0.3) is the MBR of ${o}_{1}$ .pcr(0.3) and ${o}_{2}$ . pcr(0.3). Furthermore, $e$ also stores two values $\operatorname{e.sl}\left( {C}_{1}\right)$ and $\operatorname{e.sl}\left( {C}_{2}\right)$ ,which equal the lengths of segments ${AB}$ and ${CD}$ ,respectively. Specifically,e. ${sl}\left( {C}_{1}\right)  =$ ${AB}$ because the vertical edge of ${o}_{2}.{mbr}$ is the shortest among all the edges of ${o}_{1}$ . ${mbr}$ and ${o}_{2}$ . ${mbr}$ . Similarly, $e.{sl}\left( {C}_{2}\right)  = {CD}$ since the vertical edge of ${o}_{1}$ . ${pcr}\left( {0.3}\right)$ has the smallest length among all the edges of ${o}_{1}$ .pcr(0.3) and ${o}_{2}$ .pcr(0.3).

第15/23条 ${o}_{1}.{mbr}$（或${o}_{2}.{mbr}$）内的方框是${o}_{1}.{pcr}\left( {0.3}\right)$（或${o}_{2}.{pcr}\left( {0.3}\right)$），它与${o}_{1}$（或${o}_{2}$）的叶条目相关联。考虑一个仅包含${o}_{1}$、${o}_{2}$的叶节点，其一级父条目为$e$。条目$e$包含两个矩形e.mbr(0)和e.mbr(0.3)。如图10所示，前一个矩形紧密包围${o}_{1}$和${o}_{2}$的最小边界矩形，而$e$.mbr(0.3)是${o}_{1}$.pcr(0.3)和${o}_{2}$.pcr(0.3)的最小边界矩形。此外，$e$还存储了两个值$\operatorname{e.sl}\left( {C}_{1}\right)$和$\operatorname{e.sl}\left( {C}_{2}\right)$，它们分别等于线段${AB}$和${CD}$的长度。具体来说，e.${sl}\left( {C}_{1}\right)  =$${AB}$，因为${o}_{2}.{mbr}$的垂直边是${o}_{1}$.${mbr}$和${o}_{2}$.${mbr}$所有边中最短的。类似地，$e.{sl}\left( {C}_{2}\right)  = {CD}$，因为${o}_{1}$.${pcr}\left( {0.3}\right)$的垂直边是${o}_{1}$.pcr(0.3)和${o}_{2}$.pcr(0.3)所有边中最短的。

<!-- Media -->

<!-- figureText: ${o}_{1}{pcr}\left( {0.3}\right)$ ${r}_{{q}_{2}}$ VIII ${o}_{2}$ . mbr ${o}_{2}$ . ${pcr}\left( {0.3}\right)$ $B$ e. ${MBR}\left( 0\right)$ ${o}_{1}$ . ${mbr}$ ${r}_{{q}_{1}}$ e. $\operatorname{MBR}\left( {0.3}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_22.jpg?x=636&y=337&w=558&h=360&r=0"/>

Fig. 10. A leaf node containing objects ${o}_{1}$ and ${o}_{2}$ .

图10. 包含对象${o}_{1}$和${o}_{2}$的叶节点。

<!-- Media -->

An entry $e$ of a higher level $i > 1$ has similar formats. To elaborate this,suppose that the child node of $e$ has (intermediate) entries ${e}_{1},\ldots ,{e}_{f}$ . Then, $e$ is associated (i) a pointer to the node,(ii) $m$ rectangles $e.{mbr}\left( {C}_{1}\right) ,\ldots ,e.{mbr}\left( {C}_{m}\right)$ ,where $e.{mbr}\left( c\right)$ is the MBR of ${e}_{1}.{mbr}\left( c\right) ,\ldots ,{e}_{f}.{mbr}\left( c\right)$ ,for any $c$ in the U-catalog,and (iii) $m$ values $\operatorname{esl}\left( {C}_{1}\right) ,\ldots ,\operatorname{esl}\left( {C}_{m}\right)$ such that $\operatorname{esl}\left( c\right)$ is the smallest of ${e}_{1}.{sl}\left( c\right) ,\ldots$ , ${e}_{f} \cdot  {sl}\left( c\right)$ .

较高级别 $i > 1$ 的条目 $e$ 具有相似的格式。为详细说明这一点，假设 $e$ 的子节点有（中间）条目 ${e}_{1},\ldots ,{e}_{f}$。那么，$e$ 关联着 (i) 指向该节点的指针；(ii) $m$ 个矩形 $e.{mbr}\left( {C}_{1}\right) ,\ldots ,e.{mbr}\left( {C}_{m}\right)$，其中 $e.{mbr}\left( c\right)$ 是 ${e}_{1}.{mbr}\left( c\right) ,\ldots ,{e}_{f}.{mbr}\left( c\right)$ 的最小边界矩形（MBR），对于 U 目录中的任意 $c$；以及 (iii) $m$ 个值 $\operatorname{esl}\left( {C}_{1}\right) ,\ldots ,\operatorname{esl}\left( {C}_{m}\right)$，使得 $\operatorname{esl}\left( c\right)$ 是 ${e}_{1}.{sl}\left( c\right) ,\ldots$、${e}_{f} \cdot  {sl}\left( c\right)$ 中的最小值。

Note that an intermediate entry $e$ in the U-tree consumes more space than a leaf entry. In particular, $e$ keeps $e.{sl}\left( c\right)$ ,which is not present at the leaf level (but is needed for improving the I/O performance, as explained in the next subsection). In general, it is reasonable to retain more information at the intermediate levels, if such information can reduce the number of leaf nodes accessed. After all, in processing a query, the cost at the leaf level usually significantly dominates the overall overhead.

请注意，U 树中的中间条目 $e$ 比叶条目占用更多空间。具体而言，$e$ 保存了 $e.{sl}\left( c\right)$，而叶级别不存在该信息（但如接下来的小节所述，这对于提高 I/O 性能是必需的）。一般来说，如果中间级别保留的更多信息能够减少访问的叶节点数量，那么这样做是合理的。毕竟，在处理查询时，叶级别的成本通常在总体开销中占主导地位。

The U-tree is dynamic, because objects can be inserted/deleted in an arbitrary order,by resorting to the update algorithms of the R*-tree [Beckmann et al. 1990]. Specifically,a U-tree is analogous to an ${\mathrm{R}}^{ * }$ -tree built on the o.pcr $\left( {C}_{\left\lbrack  m/2\right\rbrack  }\right)$ of the objects $\left( {C}_{\left\lceil  m/2\right\rceil  }\right.$ is the median value in the U-catalog). The difference is that, (conceptually) after objects have been grouped into leaf nodes, the contents of the intermediate entries need to be "filled in" as mentioned earlier. Clearly, with this analogy, a U-tree can also be constructed with the bulkloading algorithm of R*-trees [Leutenegger et al. 1997].

U 树是动态的，因为可以通过采用 R* 树的更新算法 [Beckmann 等人，1990 年] 以任意顺序插入/删除对象。具体来说，U 树类似于基于对象的 o.pcr $\left( {C}_{\left\lbrack  m/2\right\rbrack  }\right)$ 构建的 ${\mathrm{R}}^{ * }$ 树（$\left( {C}_{\left\lceil  m/2\right\rceil  }\right.$ 是 U 目录中的中值）。不同之处在于，（从概念上讲）在将对象分组到叶节点之后，需要如前所述“填充”中间条目的内容。显然，通过这种类比，也可以使用 R* 树的批量加载算法 [Leutenegger 等人，1997 年] 来构建 U 树。

### 5.2 The Filter Step

### 5.2 过滤步骤

Given a nonfuzzy range query $q$ (with search region ${r}_{q}$ and probability threshold ${t}_{q}$ ),the filter step traverses the U-tree in a depth-first manner. Specifically,

给定一个非模糊范围查询 $q$（搜索区域为 ${r}_{q}$，概率阈值为 ${t}_{q}$），过滤步骤以深度优先的方式遍历 U 树。具体来说，

Article 15 / 24 . Y. Tao et al. the search starts by accessing the root. For each root entry $e$ ,the algorithm computes an upper bound $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$ of the qualification probability of any object that lies in the subtree of $e$ . If $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$ is smaller than ${t}_{q}$ ,the subtree of $e$ is pruned; otherwise,we fetch the child node of $e$ ,and carry out the above operations recursively for the entries encountered there. When a leaf node is reached, we attempt to prune or validate the objects discovered. Objects that can neither be pruned nor validated are added to a candidate set ${S}_{\text{can }}$ . Then, the search backtracks to the previous level, and continues this way until no more subtree needs to be visited.

文章 15 / 24。Y. Tao 等人 搜索从访问根节点开始。对于每个根条目 $e$，算法计算位于 $e$ 子树中的任何对象的符合概率的上界 $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$。如果 $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$ 小于 ${t}_{q}$，则修剪 $e$ 的子树；否则，我们获取 $e$ 的子节点，并对在那里遇到的条目递归地执行上述操作。当到达叶节点时，我们尝试修剪或验证发现的对象。既不能被修剪也不能被验证的对象被添加到候选集 ${S}_{\text{can }}$ 中。然后，搜索回溯到上一级，并继续进行，直到无需再访问更多子树。

<!-- Media -->

---

Algorithm 2: Nonfuzzy-Range-Quali-Prob-Upper-Bound $\left( {e,{r}_{q}}\right)$

算法 2：非模糊范围符合概率上界 $\left( {e,{r}_{q}}\right)$

/* $e$ is an intermediate entry of a U-tree and ${r}_{q}$ a rectangle. The algorithm returns a value $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$ that

/* $e$ 是 U 树的一个中间条目，${r}_{q}$ 是一个矩形。该算法返回一个值 $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$，该值

upper bounds the $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ (Equation 1) of any object $o$ in the subtree of $e$ . */

对 $e$ 子树中任意对象 $o$ 的 $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$（公式 1）进行上界约束。 */

	$U{B}_{\text{range }}\left( {e,{r}_{q}}\right)  = 1$

	for $i = 1$ to $m$ /* consider the U-catalog values in ascending order */

	从 $i = 1$ 到 $m$ /* 按升序考虑 U 目录的值 */

		if ${r}_{q}$ is disjoint with $e.{mbr}\left( {C}_{i}\right)$

		如果 ${r}_{q}$ 与 $e.{mbr}\left( {C}_{i}\right)$ 不相交

			$U{B}_{\text{range }}\left( {e,{r}_{q}}\right)  = {C}_{i}$ ; return

			$U{B}_{\text{range }}\left( {e,{r}_{q}}\right)  = {C}_{i}$ ; 返回

	for $i = m$ downto 1 /* consider the U-catalog values in descending order */

	从 $i = m$ 递减到 1 /* 按降序考虑 U 目录的值 */

		$r =$ the intersection between $e.{mbr}\left( {C}_{i}\right)$ and ${r}_{q}$ /* $r$ is a rectangle $*$ /

		$r =$ 是 $e.{mbr}\left( {C}_{i}\right)$ 和 ${r}_{q}$ 的交集 /* $r$ 是一个矩形 $*$ /

		if the projection length of $r$ on any dimension is smaller than $\operatorname{e.sl}\left( {C}_{i}\right)$

		如果 $r$ 在任意维度上的投影长度小于 $\operatorname{e.sl}\left( {C}_{i}\right)$

			$U{B}_{\text{range }}\left( {e,{r}_{q}}\right)  = 1 - {C}_{i}$ ; return

			$U{B}_{\text{range }}\left( {e,{r}_{q}}\right)  = 1 - {C}_{i}$ ; 返回

---

Fig. 11. Finding an upper bound of an object's qualification probability (nonfuzzy range search).

图 11. 寻找对象合格概率的上界（非模糊范围搜索）。

<!-- Media -->

The filter phase of a fuzzy query $q$ (with distance and probability thresholds ${\varepsilon }_{q}$ and ${t}_{q}$ ,respectively) is exactly the same,except that $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$ is replaced with $U{B}_{fuzzy}\left( {e,q,{\varepsilon }_{q}}\right)$ ,which upper bounds the $U{B}_{fuzzy}\left( {o,q,{\varepsilon }_{q}}\right)$ (as in Eq. (3)) of any object $o$ underneath $e$ .

模糊查询 $q$（距离阈值为 ${\varepsilon }_{q}$，概率阈值为 ${t}_{q}$）的过滤阶段完全相同，只是用 $U{B}_{fuzzy}\left( {e,q,{\varepsilon }_{q}}\right)$ 替换了 $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$，$U{B}_{fuzzy}\left( {e,q,{\varepsilon }_{q}}\right)$ 对 $e$ 下方任意对象 $o$ 的 $U{B}_{fuzzy}\left( {o,q,{\varepsilon }_{q}}\right)$（如公式 (3) 所示）进行上界约束。

It remains to clarify the computation of $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$ and $U{B}_{\text{fuzzy }}\left( {e,q,{\varepsilon }_{q}}\right)$ . We settle the former with Algorithm 2 (Figure 11),assuming a rectangular ${r}_{q}$ . To illustrate the algorithm, let us consider Figure 10 again, where, as mentioned earlier,the U-catalog has $m = 2$ values ${C}_{1} = 0,{C}_{2} = {0.3}$ ,and $e$ is the parent entry of the leaf node containing only ${o}_{1}$ and ${o}_{2}$ . Rectangles ${r}_{{q}_{1}}$ and ${r}_{{q}_{2}}$ are the search regions of two nonfuzzy range queries ${q}_{1}$ and ${q}_{2}$ ,respectively. Given parameters $e$ and ${r}_{{q}_{1}}$ ,Algorithm 2 returns (at Line 4) $U{B}_{\text{range }}\left( {e,{r}_{{q}_{1}}}\right)  = {0.3}$ . This is because ${r}_{{q}_{1}}$ does not intersect $e.{mbr}\left( {0.3}\right)$ ,which indicates that the $o.{pcr}\left( {0.3}\right)$ of any object $o$ must be disjoint with ${r}_{{q}_{1}}$ . Hence,according to Rule 2 of Theorem 1, $o$ cannot appear in ${r}_{{q}_{1}}$ with a probability at least 0.3 .

仍需明确 $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$ 和 $U{B}_{\text{fuzzy }}\left( {e,q,{\varepsilon }_{q}}\right)$ 的计算方法。假设 ${r}_{q}$ 为矩形，我们使用算法 2（图 11）来解决前者的计算问题。为说明该算法，让我们再次考虑图 10，如前所述，U 目录有 $m = 2$ 个值 ${C}_{1} = 0,{C}_{2} = {0.3}$，$e$ 是仅包含 ${o}_{1}$ 和 ${o}_{2}$ 的叶节点的父条目。矩形 ${r}_{{q}_{1}}$ 和 ${r}_{{q}_{2}}$ 分别是两个非模糊范围查询 ${q}_{1}$ 和 ${q}_{2}$ 的搜索区域。给定参数 $e$ 和 ${r}_{{q}_{1}}$，算法 2（第 4 行）返回 $U{B}_{\text{range }}\left( {e,{r}_{{q}_{1}}}\right)  = {0.3}$。这是因为 ${r}_{{q}_{1}}$ 与 $e.{mbr}\left( {0.3}\right)$ 不相交，这表明任意对象 $o$ 的 $o.{pcr}\left( {0.3}\right)$ 必定与 ${r}_{{q}_{1}}$ 不相交。因此，根据定理 1 的规则 2，$o$ 出现在 ${r}_{{q}_{1}}$ 中的概率至少为 0.3 的情况不可能发生。

On the other hand,given $e$ and ${r}_{{q}_{2}}$ ,Algorithm 2 produces (at Line 8) $U{B}_{\text{range }}\left( {e,{r}_{{q}_{2}}}\right)  = {0.7}$ . To explain why,we need to focus on the hatched area of Figure 10,which is the intersection between ${r}_{{g}_{2}}$ and $e.{mbr}\left( {0.3}\right)$ ,and is the rectangle $r$ computed at Line 6 of Algorithm 2. Observe that the length of the vertical projection of $r$ is shorter than e.sl (0.3),which,as discussed in Section 5.1, equals the length of segment ${CD}$ . This implies that none of the o.pcr(0.3) of any object $o$ in the subtree of $e$ can possibly be entirely covered by ${r}_{{q}_{2}}$ . As a result, by Rule 1 of Theorem 1, $o$ falls in ${r}_{{q}_{2}}$ with a probability less than 0.7 .

另一方面，给定$e$和${r}_{{q}_{2}}$，算法2（在第8行）生成$U{B}_{\text{range }}\left( {e,{r}_{{q}_{2}}}\right)  = {0.7}$。为了解释原因，我们需要关注图10中的阴影区域，该区域是${r}_{{g}_{2}}$和$e.{mbr}\left( {0.3}\right)$的交集，也是算法2第6行计算得到的矩形$r$。观察可知，$r$的垂直投影长度小于e.sl (0.3)，正如5.1节所讨论的，它等于线段${CD}$的长度。这意味着$e$子树中任何对象$o$的o.pcr(0.3)都不可能完全被${r}_{{q}_{2}}$覆盖。因此，根据定理1的规则1，$o$落入${r}_{{q}_{2}}$的概率小于0.7。

When ${r}_{q}$ is not rectangular, $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$ can be calculated using Algorithm 2, by passing the MBR of ${r}_{q}$ as the second parameter. Finally,the $U{B}_{\text{fuzzy }}\left( {e,q,{\varepsilon }_{q}}\right)$ for a fuzzy query $q$ can also be calculated using Algorithm 2,leveraging the reduction proposed in Section 4.2. Instead of repeating the theoretical reasoning

当${r}_{q}$不是矩形时，可以使用算法2来计算$U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$，方法是将${r}_{q}$的最小边界矩形（MBR）作为第二个参数传入。最后，对于模糊查询$q$的$U{B}_{\text{fuzzy }}\left( {e,q,{\varepsilon }_{q}}\right)$也可以使用算法2来计算，利用4.2节提出的简化方法。这里不再重复理论推导

<!-- Media -->

---

Algorithm 3: Fuzzy-Quali-Prob-Upper-Bound $\left( {e,q,{\varepsilon }_{q}}\right)$

算法3：模糊质量概率上界$\left( {e,q,{\varepsilon }_{q}}\right)$

/* $e$ is an intermediate entry of a U-tree, $q$ a query uncertain object,and ${\varepsilon }_{q}$ the distance threshold of $q$ .

/* $e$是U树的一个中间条目，$q$是一个查询不确定对象，${\varepsilon }_{q}$是$q$的距离阈值。

Applicable to both the ${L}_{\infty }$ and ${L}_{2}$ norms,the algorithm returns a value $U{B}_{fuzzy}\left( {e,q,{\varepsilon }_{q}}\right)$ that upper bounds the

该算法适用于${L}_{\infty }$和${L}_{2}$范数，返回一个值$U{B}_{fuzzy}\left( {e,q,{\varepsilon }_{q}}\right)$，该值是

$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ (Equation 3) of any object $o$ in the subtree of $e$ . */

$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$（公式3）对于$e$子树中任何对象$o$的上界。 */

	$U{B}_{fuzzy}\left( {e,q,{\varepsilon }_{q}}\right)  = 1$

	for $i = 1$ to $d$ /* consider each dimension in turn */

	从$i = 1$到$d$ /* 依次考虑每个维度 */

		partition $q.{mbr}$ into rectangles ${r}_{1},\ldots ,{r}_{2{m}_{q} - 1}$ as in Lemma 4 /* ${m}_{q}$ is defined in Section 4.2 */

		按照引理4将$q.{mbr}$划分为矩形${r}_{1},\ldots ,{r}_{2{m}_{q} - 1}$ /* ${m}_{q}$在4.2节中定义 */

		for $j = 1$ to $2{m}_{q} - 1$

		从$j = 1$到$2{m}_{q} - 1$

			$U{B}_{Pr}\left( {{r}_{j},e,{\varepsilon }_{q}}\right)  =$ Range-Quali-Prob-Upper-Bound $\left( {e, \sqcup  \left( {{r}_{j},{\varepsilon }_{q}}\right) \text{of}{L}_{\infty }}\right)$

			$U{B}_{Pr}\left( {{r}_{j},e,{\varepsilon }_{q}}\right)  =$ 范围质量概率上界$\left( {e, \sqcup  \left( {{r}_{j},{\varepsilon }_{q}}\right) \text{of}{L}_{\infty }}\right)$

			/* Algorithm 2 is given in Figure 11 */

			/* 算法2如图11所示 */

			/* $U{B}_{Pr}\left( {{r}_{j},e,{\varepsilon }_{q}}\right)$ is an upper bound of $\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$ for any object $o$ in the subtree of

			/* $U{B}_{Pr}\left( {{r}_{j},e,{\varepsilon }_{q}}\right)$是$\mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)$对于

			$e$ ,and any point $x$ in ${r}_{j}$ . The computation of $\sqcup  \left( {{r}_{j},{\varepsilon }_{q}}\right)$ of ${L}_{\infty }$ is illustrated in Figure 9a */

			$e$子树中任何对象${r}_{j}$以及${r}_{j}$中任何点$x$的上界。${L}_{\infty }$的$\sqcup  \left( {{r}_{j},{\varepsilon }_{q}}\right)$的计算如图9a所示 */

		${UB} =$ the right hand side of Inequality 13,replacing all occurrences of $o$ with $e$

		${UB} =$ 不等式13的右侧，将所有$o$的出现替换为$e$

		$U{B}_{fuzzy}\left( {e,q,{\varepsilon }_{q}}\right)  = \min \left\{  {U{B}_{fuzzy}\left( {e,q,{\varepsilon }_{q}}\right) ,{UB}}\right\}$

---

Fig. 12. Finding an upper bound of an object's qualification probability (fuzzy search).

图12. 寻找对象合格概率的上界（模糊搜索）。

<!-- Media -->

of the reduction, we simply present the details in Figure 12, which applies to both the ${L}_{\infty }$ and ${L}_{2}$ norms.

关于约简，我们仅在图12中展示细节，该图适用于${L}_{\infty }$范数和${L}_{2}$范数。

### 5.3 The Refinement Step

### 5.3 细化步骤

For each object $o$ in the ${S}_{\text{can }}$ output by the filter step,the refinement phase calculates the qualification probability of $o$ ,for comparison with the probability threshold ${t}_{q}$ . Next,we elaborate the details of the calculation,starting with $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ for a nonfuzzy range query $q$ ,before discussing $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ for a fuzzy query $q$ .

对于过滤步骤输出的${S}_{\text{can }}$中的每个对象$o$，细化阶段会计算$o$的合格概率，以便与概率阈值${t}_{q}$进行比较。接下来，我们详细阐述计算细节，先从非模糊范围查询$q$的$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$开始，再讨论模糊查询$q$的$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$。

If $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ can be solved into a closed equation,its computation entails negligible cost. For instance,when $o.{ur}$ and ${r}_{q}$ are rectangles and $o.{pdf}$ describes a uniform distribution, $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ is simply the ratio between the areas of o.ur $\cap  {r}_{q}$ and ${r}_{q}$ ,both of which can be easily computed. In general,however, integrating a complex multidimensional function (i.e., o.pdf) over a potentially irregular region $o.{ur} \cap  {r}_{q}$ is a well-known difficult problem,for which the Monte-Carlo (MC) method [Press et al. 2002] is a standard remedy.

如果$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$可以求解为一个封闭方程，其计算成本可以忽略不计。例如，当$o.{ur}$和${r}_{q}$是矩形且$o.{pdf}$描述的是均匀分布时，$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$就是o.ur $\cap  {r}_{q}$和${r}_{q}$的面积之比，这两个面积都可以轻松计算。然而，一般来说，在一个可能不规则的区域$o.{ur} \cap  {r}_{q}$上对一个复杂的多维函数（即o.pdf）进行积分是一个众所周知的难题，对此，蒙特卡罗（MC）方法[Press等人，2002]是一种标准的解决办法。

Specifically,to apply MC,we need to formulate a function $f\left( x\right)$ ,which equals o.pdf(x)for a $d$ -dimensional point $x$ in $o.{ur} \cap  {r}_{q}$ ,or 0 for any other $x$ . Then,we take the MBR ${r}_{mbr}$ of ${r}_{q}$ (note that ${r}_{q}$ may not be a rectangle),and uniformly generate a number $s$ of points inside ${r}_{mbr} \cap  o.{MBR}$ (which is a rectangle). Let us denote these points as ${x}_{1},\ldots ,{x}_{s}$ ,respectively. Equation (1) can be estimated as (using $E$ to denote the estimate):

具体来说，要应用MC方法，我们需要构造一个函数$f\left( x\right)$，对于$o.{ur} \cap  {r}_{q}$中的一个$d$维点$x$，该函数等于o.pdf(x)，对于其他任何$x$，该函数等于0。然后，我们取${r}_{q}$的最小边界矩形（MBR）${r}_{mbr}$（注意，${r}_{q}$可能不是矩形），并在${r}_{mbr} \cap  o.{MBR}$（这是一个矩形）内均匀生成数量为$s$的点。我们分别将这些点记为${x}_{1},\ldots ,{x}_{s}$。方程（1）可以估计为（用$E$表示估计值）：

$$
E = \operatorname{vol} \cdot  \frac{1}{s}\mathop{\sum }\limits_{{i = 1}}^{s}f\left( {x}_{i}\right)  \tag{15}
$$

where vol returns the volume of ${r}_{mbr} \cap  o.{MBR}$ . The value of $s$ may need to be really large in order to produce an accurate estimate. This is why refinement of an object can be rather costly, and should be prevented as much as possible.

其中vol返回${r}_{mbr} \cap  o.{MBR}$的体积。为了得到准确的估计值，$s$的值可能需要非常大。这就是为什么对一个对象进行细化的成本可能相当高，并且应尽可能避免。

$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ is more expensive to evaluate,because Eq. (3) essentially has two layers of integrals. In particular,the inner layer derives $\mathop{\Pr }\limits_{\text{range }}\left( {x, \odot  \left( {o,{\varepsilon }_{q}}\right) }\right)$ , which can be settled as described earlier. The outer layer can also be calculated by Eq. (15) with the following changes. First,function $f\left( x\right)$ should be replaced with $g\left( x\right)$ ,which equals 0 if $x$ lies outside $q.{ur}$ ; otherwise, $g\left( x\right)$ is the integrand of Eq. (3). Second, ${r}_{mbr}$ now becomes $q$ . ${mbr}$ .

$\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$的计算成本更高，因为方程（3）本质上有两层积分。特别是，内层计算$\mathop{\Pr }\limits_{\text{range }}\left( {x, \odot  \left( {o,{\varepsilon }_{q}}\right) }\right)$，可以如前文所述进行求解。外层也可以通过方程（15）进行计算，但需要做以下更改。首先，函数$f\left( x\right)$应替换为$g\left( x\right)$，如果$x$位于$q.{ur}$之外，$g\left( x\right)$等于0；否则，$g\left( x\right)$是方程（3）的被积函数。其次，${r}_{mbr}$现在变为$q$。${mbr}$。

## 6. COST-BASED INDEX OPTIMIZATION

## 6. 基于成本的索引优化

The U-tree takes a parameter $m$ ,which is the size of the U-catalog,and has a significant impact on the query performance. A large $m$ leads to more precomputed PCRs, which reduce the chance of calculating an object's actual qualification probability, and hence, the overhead of the refinement step. On the other hand,as $m$ increases,the node fanout decreases,which adversely affects the I/O efficiency of the filter phase.

U树采用一个参数$m$，它是U目录的大小，并且对查询性能有显著影响。较大的$m$会导致更多预先计算的概率约束区域（PCR，Probabilistic Constraint Region），这会降低计算对象实际合格概率的机会，从而减少细化步骤的开销。另一方面，随着$m$的增大，节点扇出会减小，这会对过滤阶段的I/O效率产生不利影响。

The best $m$ depends on the dataset characteristics. As an extreme example, imagine that all uncertainty regions are so small that their extents can be ignored. In this case, the dataset degenerates into a set of points, for which (intuitively) the best index is simply an ${\mathrm{R}}^{ * }$ -tree,or a special U-tree with $m = 1$ (i.e.,for each object $o$ ,only $\operatorname{o.pcr}\left( 0\right)  = \operatorname{o.mbr}$ is stored). On the other hand, consider an object $o$ that has a sizable uncertainty region o.ur,and a Gaussian pdf with a large variance (i.e.,o.pdf(x)peaks at the center of o.ur,but quickly diminishes as $x$ drifts away). It is beneficial to materialize o.pcr(c) at some values of $c \in  (0,{0.5}\rbrack$ ,all of which have significantly smaller extents than 0.mbr, and effectively prevent the refinement of $o$ .

最佳的$m$取决于数据集的特征。举一个极端的例子，假设所有的不确定区域都非常小，以至于它们的范围可以忽略不计。在这种情况下，数据集退化为一组点，直观地说，对于这些点，最佳索引就是一个${\mathrm{R}}^{ * }$树，或者是一个$m = 1$的特殊U树（即，对于每个对象$o$，只存储$\operatorname{o.pcr}\left( 0\right)  = \operatorname{o.mbr}$）。另一方面，考虑一个对象$o$，它有一个相当大的不确定区域o.ur，并且具有一个方差很大的高斯概率密度函数（即，o.pdf(x)在o.ur的中心达到峰值，但随着$x$的偏离而迅速减小）。在$c \in  (0,{0.5}\rbrack$的某些值下具体化o.pcr(c)是有益的，所有这些值的范围都比0.mbr小得多，并且能有效地避免对$o$进行细化。

In the sequel,we provide a method for deciding a good value of $m$ prior to the construction of a U-tree. Our objective is to minimize the cost of nonfuzzy range search (Definition 1). Towards this purpose, Section 6.1 first analyzes how often PCRs can prevent the numerical process discussed in Section 5.3. Then, Section 6.2 derives a formula that accurately captures both the filter and refinement overhead. Finally,Section 6.3 applies the cost model to optimize $m$ for an arbitrary dataset.

接下来，我们提供一种在构建U树之前确定$m$的合适值的方法。我们的目标是最小化非模糊范围搜索的成本（定义1）。为此，第6.1节首先分析概率约束区域（PCR）能够避免第5.3节中讨论的数值计算过程的频率。然后，第6.2节推导出一个公式，该公式能准确地计算过滤和细化的开销。最后，第6.3节应用成本模型为任意数据集优化$m$。

### 6.1 Probability of Numerical Evaluation

### 6.1 数值评估的概率

Consider an object $o$ with pre-computed o.pcr $\left( {C}_{1}\right) ,\ldots ,o \cdot  \operatorname{pcr}\left( {C}_{m}\right)$ ,where ${C}_{1},\ldots$ , ${C}_{m}$ are the values in the U-catalog. Let $q$ be a nonfuzzy range query with probability threshold ${t}_{q}$ and a rectangular search region ${r}_{q}$ whose projection on the $i$ th dimension has length $s{l}_{q\left\lbrack  i\right\rbrack  }$ (much smaller than 1),and its centroid is uniformly distributed in the workspace. We aim at deriving the probability $o.P{r}_{comp}$ that $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ must be numerically computed in processing $q$ . Alternatively, $o.P{r}_{comp}$ is the likelihood that $o$ can be neither pruned by Theorem 3 nor validated by Theorem 2.

考虑一个具有预先计算的o.pcr $\left( {C}_{1}\right) ,\ldots ,o \cdot  \operatorname{pcr}\left( {C}_{m}\right)$的对象$o$，其中${C}_{1},\ldots$、${C}_{m}$是U目录中的值。设$q$是一个非模糊范围查询，其概率阈值为${t}_{q}$，矩形搜索区域为${r}_{q}$，该区域在第$i$维上的投影长度为$s{l}_{q\left\lbrack  i\right\rbrack  }$（远小于1），并且其质心在工作空间中均匀分布。我们的目标是推导出在处理$q$时必须对$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$进行数值计算的概率$o.P{r}_{comp}$。换句话说，$o.P{r}_{comp}$是$o$既不能由定理3进行剪枝也不能由定理2进行验证的可能性。

We are mainly interested in query regions that are "larger" than o.mbr. Formally,if the projection length of o.mbr on the $i$ th $\left( {1 \leq  i \leq  d}\right)$ dimension is $o.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }$ ,then $s{l}_{q\left\lbrack  i\right\rbrack  } \geq  o.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }$ for all $i \in  \left\lbrack  {1,d}\right\rbrack$ . The reasons for concentrating on such voluminous queries are two fold. First, they are (much) more expensive than queries with small search regions, and hence, the target of optimization in finding an appropriate U-catalog size $m$ . Second,they simplify Theorem 2, which allows us to avoid excessively complex equations. Specifically, for a voluminous query,at least one value in each pair of ${c}_{i},{c}_{i}^{\prime }\left( {1 \leq  i \leq  d}\right)$ must be 0 in Rule 1 of Theorem 2; likewise,in Rule (2),either ${c}_{1}$ or ${c}_{1}^{\prime }$ equals 0 .

我们主要关注那些比对象o的最小边界矩形（o.mbr）“更大”的查询区域。形式上，如果o.mbr在第$i$个$\left( {1 \leq  i \leq  d}\right)$维度上的投影长度为$o.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }$，那么对于所有的$i \in  \left\lbrack  {1,d}\right\rbrack$，都有$s{l}_{q\left\lbrack  i\right\rbrack  } \geq  o.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }$。专注于此类大规模查询的原因有两个。首先，与搜索区域较小的查询相比，它们的开销（要）大得多，因此是寻找合适的U - 目录大小$m$时的优化目标。其次，它们简化了定理2，使我们能够避免过于复杂的方程。具体而言，对于一个大规模查询，在定理2的规则1中，每对${c}_{i},{c}_{i}^{\prime }\left( {1 \leq  i \leq  d}\right)$中至少有一个值必须为0；同样，在规则（2）中，要么${c}_{1}$等于0，要么${c}_{1}^{\prime }$等于0。

Range Search on Multidimensional Uncertain Data Article 15 / 27

多维不确定数据上的范围搜索 文章15 / 27

<!-- Media -->

<!-- figureText: o. $\operatorname{pcr}\left( {c}_{ \vdash  }\right)$ ${a}_{3}$ ${r}_{{q}_{\mathrm{╲}}}$ $\rightarrow  {a}_{2} \rightarrow$ (b) Approximating the evaluation region o. ${pcr}\left( {c}_{4}\right)$ -o.pcr(c_) o.mbr (d) Illustration of the lower bound o.mbr o. ${pcr}\left( {c}_{ \bot  }\right)$ (a) "Concentric" o.mbr,o.pcr $\left( {c}_{ \dashv  }\right)$ ,and o.pcr $\left( {c}_{ \vdash  }\right)$ ${r}_{q}$ 3 o.mbr ${B}_{\mathrm{O}}$ $o \cdot  {pcr}\left( {c}_{\llcorner }\right)$ o. ${pcr}\left( {c}_{ \rightarrow  }\right)$ (c) Illustration of the upper bound -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_26.jpg?x=368&y=337&w=1092&h=836&r=0"/>

Fig. 13. Analysis of $o.P{r}_{\text{comp }}$ for ${t}_{q} > 1 - {C}_{m}$ .

图13. 针对${t}_{q} > 1 - {C}_{m}$对$o.P{r}_{\text{comp }}$的分析。

<!-- Media -->

Our analysis on voluminous nonfuzzy range search proceeds in two parts, focusing on large ${t}_{q}$ (Section 6.1.1),small ${t}_{q}$ (Section 6.1.2),and median ${t}_{q}$ (Section 6.1.3), respectively. Finally, Section 6.1.4 mathematically explains why PCRs are a useful tool for reducing the refinement cost, and elaborates how to support nonvoluminous queries.

我们对大规模非模糊范围搜索的分析分为两部分，分别关注较大的${t}_{q}$（6.1.1节）、较小的${t}_{q}$（6.1.2节）和中等的${t}_{q}$（6.1.3节）。最后，6.1.4节从数学上解释了为什么部分覆盖区域（PCRs）是降低细化成本的有用工具，并详细说明了如何支持非大规模查询。

6.1.1 Case 1: ${t}_{q} > 1 - {C}_{m}$ . As opposed to the original settings (the location of o.mbr is fixed while that of ${r}_{q}$ is uniformly distributed),for deriving o.Pr ${}_{\text{comp }}$ , it is more convenient to consider the equivalent opposite. Specifically,we fix ${r}_{q}$ , but move the centroid of o.mbr around in the workspace, following a uniform distribution. In doing so,we keep track of whether $o$ can be pruned/validated when its MBR equals the current o.mbr. After o.mbr has been placed at all possible locations, we have collected an evaluation region (ER), which consists of all the centroids of o.mbr that do not allow our heuristics to prune and validate o. Thus, $o.P{r}_{comp}$ is exactly the area of this region (remember that the workspace has an area 1).

6.1.1 情况1：${t}_{q} > 1 - {C}_{m}$。与原始设置（o.mbr的位置固定，而${r}_{q}$的位置均匀分布）不同，为了推导o.Pr ${}_{\text{comp }}$，考虑等效的相反情况会更方便。具体来说，我们固定${r}_{q}$，但让o.mbr的质心在工作空间中按照均匀分布移动。这样做时，我们会跟踪当$o$的最小边界矩形（MBR）等于当前的o.mbr时，它是否可以被剪枝/验证。在o.mbr被放置在所有可能的位置之后，我们收集到了一个评估区域（ER），它由所有不允许我们的启发式方法对o进行剪枝和验证的o.mbr的质心组成。因此，$o.P{r}_{comp}$恰好就是这个区域的面积（记住工作空间的面积为1）。

Assume that $o.{mbr}$ is the dashed rectangle in Figure 13(a),which also demonstrates the $\operatorname{o.pcr}\left( {c}_{ \dashv  }\right)$ and $\operatorname{o.pcr}\left( {c}_{ \vdash  }\right)$ of $o$ ,where ${c}_{ \vdash  }\left( {c}_{ \dashv  }\right)$ is the smallest (largest) U-catalog value at least (most) $1 - {t}_{q}$ . Here, $\operatorname{o.pcr}\left( {c}_{ \dashv  }\right)$ ,o. $\operatorname{pcr}\left( {c}_{ \vdash  }\right)$ ,and $\operatorname{o.mbr}$ are "concentric", that is, they have the same centroid. Although the concentric behavior is not always true, it facilitates explaining the crux of our analysis. Later, we will generalize the results to the general situation where PCRs are not concentric.

假设$o.{mbr}$是图13（a）中的虚线矩形，该图还展示了$o$的$\operatorname{o.pcr}\left( {c}_{ \dashv  }\right)$和$\operatorname{o.pcr}\left( {c}_{ \vdash  }\right)$，其中${c}_{ \vdash  }\left( {c}_{ \dashv  }\right)$是至少（至多）为$1 - {t}_{q}$的最小（最大）U - 目录值。这里，$\operatorname{o.pcr}\left( {c}_{ \dashv  }\right)$、o. $\operatorname{pcr}\left( {c}_{ \vdash  }\right)$和$\operatorname{o.mbr}$是“同心的”，即它们具有相同的质心。尽管同心情况并非总是成立，但它有助于解释我们分析的关键要点。稍后，我们将把结果推广到部分覆盖区域（PCRs）不同心的一般情况。

The outmost rectangle in Figure 13(b) corresponds to the query region ${r}_{q}$ . Next, we will construct an area that closely approximates the ER. The hatched region in Figure 13(b) is a ring bounded by two rectangles, both of which have the same centroid as ${r}_{q}$ . In particular,the outer (inner) rectangle is shorter than ${r}_{q}$ by $2{a}_{1}\left( {2{a}_{2}}\right)$ on the horizontal dimension,and by $2{b}_{1}\left( {2{b}_{2}}\right)$ on the vertical dimension. As illustrated in Figure 13(a), $2{a}_{1},2{b}_{1}$ (or $2{a}_{2},2{b}_{2}$ ) are the lengths of the horizontal and vertical edges of $o.{pcr}\left( {c}_{ \vdash  }\right)$ (or $o.{pcr}\left( {c}_{ \dashv  }\right)$ ),respectively. In Figure 13(b), there are 4 grey boxes near the corners of the ring. The upper-left grey box is decided by (i) the (inner) corner of the ring that the box is adjacent to,and (ii) the point inside ${r}_{q}$ ,having horizontal (vertical) distance ${a}_{3}\left( {b}_{3}\right)$ from the upper-left corner of ${r}_{q}$ ,where ${a}_{3}$ and ${b}_{3}$ are the projection lengths of $o.{mbr}$ (see Figure 13(a)). The other grey boxes are obtained in the same way, but with respect to other corners of the ring and ${r}_{q}$ .

图13(b)中最外层的矩形对应查询区域${r}_{q}$。接下来，我们将构建一个与ER紧密近似的区域。图13(b)中的阴影区域是一个由两个矩形界定的环形区域，这两个矩形的质心与${r}_{q}$的质心相同。具体而言，外部（内部）矩形在水平维度上比${r}_{q}$短$2{a}_{1}\left( {2{a}_{2}}\right)$，在垂直维度上比${r}_{q}$短$2{b}_{1}\left( {2{b}_{2}}\right)$。如图13(a)所示，$2{a}_{1},2{b}_{1}$（或$2{a}_{2},2{b}_{2}$）分别是$o.{pcr}\left( {c}_{ \vdash  }\right)$（或$o.{pcr}\left( {c}_{ \dashv  }\right)$）的水平和垂直边的长度。在图13(b)中，环形区域的角落附近有4个灰色方块。左上角的灰色方块由以下两个因素决定：(i) 该方块相邻的环形区域的（内部）角落；(ii) ${r}_{q}$内部的一个点，该点与${r}_{q}$左上角的水平（垂直）距离为${a}_{3}\left( {b}_{3}\right)$，其中${a}_{3}$和${b}_{3}$是$o.{mbr}$的投影长度（见图13(a)）。其他灰色方块以相同的方式获得，但相对于环形区域和${r}_{q}$的其他角落。

When the centroid of o.mbr falls outside the hatched and grey area in Figure 13(b), o can always be pruned or validated. For example, if the centroid lies at point $A$ in Figure 13(c), ${r}_{q}$ does not fully cover $o.{pcr}\left( {c}_{ \vdash  }\right)$ ; hence, $o$ is eliminated by Rule 1 of Theorem 3. On the other hand, if the centroid falls at $B,o$ can be validated by Rule 1 of Theorem 2,setting ${c}_{1} = 0$ and ${c}_{1}^{\prime } = {c}_{ \vdash  }$ (the subscript 1 represents the horizontal dimension). On the other hand, as long as the centroid of o.mbr lies in the hatched region in Figure 13(b), o can never be pruned/validated,but always requires numeric evaluation of $o.P{r}_{\text{comp }}$ . For instance, let us examine the case where the centroid lies at point $C$ in Figure 13(d). Given parameters $o$ and ${r}_{q}$ ,Algorithm 1 returns $\left\lbrack  {1 - {c}_{ \vdash  },1 - {c}_{ \dashv  }}\right\rbrack$ ,which contains ${t}_{q}$ . Hence,as proved in Section 3.4, $o$ can be neither pruned by Theorem 1 nor validated by Theorem 2.

当对象o的最小边界矩形（o.mbr）的质心落在图13(b)中的阴影和灰色区域之外时，对象o总是可以被剪枝或验证。例如，如果质心位于图13(c)中的点$A$，${r}_{q}$不能完全覆盖$o.{pcr}\left( {c}_{ \vdash  }\right)$；因此，根据定理3的规则1，$o$被排除。另一方面，如果质心落在$B,o$，根据定理2的规则1可以验证，设置${c}_{1} = 0$和${c}_{1}^{\prime } = {c}_{ \vdash  }$（下标1表示水平维度）。另一方面，只要对象o的最小边界矩形（o.mbr）的质心位于图13(b)中的阴影区域内，对象o就永远不能被剪枝/验证，但总是需要对$o.P{r}_{\text{comp }}$进行数值评估。例如，让我们考察质心位于图13(d)中的点$C$的情况。给定参数$o$和${r}_{q}$，算法1返回$\left\lbrack  {1 - {c}_{ \vdash  },1 - {c}_{ \dashv  }}\right\rbrack$，其中包含${t}_{q}$。因此，如第3.4节所证明的，$o$既不能由定理1剪枝，也不能由定理2验证。

The implication of the above discussion is that o.Pr ${}_{\text{comp }}$ is at most the total area of the hatched and grey regions, but at least the area of the hatched region itself. Formally, $o.P{r}_{comp} \in  \left\lbrack  {o.P{r}_{comp}^{LB},o.P{r}_{comp}^{UB}}\right\rbrack$ ,with

上述讨论的含义是，对象o的概率o.Pr ${}_{\text{comp }}$至多是阴影和灰色区域的总面积，但至少是阴影区域本身的面积。形式上，$o.P{r}_{comp} \in  \left\lbrack  {o.P{r}_{comp}^{LB},o.P{r}_{comp}^{UB}}\right\rbrack$，其中

$$
{o.{Pr}}_{comp}^{LB} = \mathop{\prod }\limits_{{i = 1}}^{d}\left( {s{l}_{q\left\lbrack  i\right\rbrack  } - {o.{sl}}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \vdash  }\right) }\right)  - \mathop{\prod }\limits_{{i = 1}}^{d}\left( {s{l}_{q\left\lbrack  i\right\rbrack  } - {o.{sl}}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \dashv  }\right) }\right) \text{,and} \tag{16}
$$

$$
o.P{r}_{comp}^{UB} = o \cdot  P{r}_{comp}^{LB} + \mathop{\prod }\limits_{{i = 1}}^{d}\left( {o.s{l}_{{mbr}\left\lbrack  i\right\rbrack  } - o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \dashv  }\right) }\right) , \tag{17}
$$

where function $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ gives the projection length of $o.{pcr}\left( c\right)$ on the $i$ th dimension. For example,in Figure 13(a), ${\operatorname{o.sl}}_{{pcr}\left\lbrack  1\right\rbrack  }\left( {c}_{ \vdash  }\right)  = 2{a}_{1}$ and ${\operatorname{o.sl}}_{{pcr}\left\lbrack  2\right\rbrack  }\left( {c}_{ \vdash  }\right)  = 2{b}_{1}$ . Notice that we presented the above equations in their general forms applicable to all dimensionalities. In fact, the construction of the approximate ER in Figure 13(b) can be extended to any dimensionality in a straightforward manner; the resulting ring and grey boxes possess the same properties.

其中函数 $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ 给出了 $o.{pcr}\left( c\right)$ 在第 $i$ 维上的投影长度。例如，在图13(a)中， ${\operatorname{o.sl}}_{{pcr}\left\lbrack  1\right\rbrack  }\left( {c}_{ \vdash  }\right)  = 2{a}_{1}$ 且 ${\operatorname{o.sl}}_{{pcr}\left\lbrack  2\right\rbrack  }\left( {c}_{ \vdash  }\right)  = 2{b}_{1}$ 。请注意，我们以上述方程的通用形式呈现，这些形式适用于所有维度。实际上，图13(b)中近似ER（近似误差区域，Approximate Error Region）的构建可以直接扩展到任何维度；所得的环和灰色框具有相同的属性。

So far we have implicitly assumed the presence of ${c}_{ \vdash  }$ ,which,however,is not always true. If ${c}_{ \vdash  }$ does not exist,Rule 1 of Theorem 3 is no longer applicable; thus,pruning relies on Rule (2),where,for ${t}_{q}$ over $1 - {C}_{m}$ ,the value $c$ equals ${C}_{m}$ . Accordingly, the outer boundary of the ring in Figure 13(b) becomes a rectangle

到目前为止，我们隐式地假设了 ${c}_{ \vdash  }$ 的存在，但实际情况并非总是如此。如果 ${c}_{ \vdash  }$ 不存在，定理3的规则1将不再适用；因此，剪枝依赖于规则(2)，其中，对于在 $1 - {C}_{m}$ 上的 ${t}_{q}$ ，值 $c$ 等于 ${C}_{m}$ 。相应地，图13(b)中环的外边界变为一个矩形

Article 15 / 29 (again,sharing the centroid of ${r}_{q}$ ) with a projection length longer than that of ${r}_{q}$ by $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {C}_{m}\right)$ on the $i$ th dimension $\left( {1 \leq  i \leq  2}\right)$ . As a result, $s{l}_{q\left\lbrack  i\right\rbrack  } - o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \vdash  }\right)$ should be replaced with $s{l}_{q\left\lbrack  i\right\rbrack  } + o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {C}_{m}\right)$ in Eq. (16).

第15/29条（同样，与 ${r}_{q}$ 共享质心），其在第 $i$ 维 $\left( {1 \leq  i \leq  2}\right)$ 上的投影长度比 ${r}_{q}$ 的投影长度长 $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {C}_{m}\right)$ 。因此，在公式(16)中， $s{l}_{q\left\lbrack  i\right\rbrack  } - o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \vdash  }\right)$ 应替换为 $s{l}_{q\left\lbrack  i\right\rbrack  } + o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {C}_{m}\right)$ 。

<!-- Media -->

<!-- figureText: o. $\operatorname{pcr}\left( {c}_{ \leftarrow  }\right)$ $\leftarrow  {a}_{2} \rightarrow$ (b) Approximating the evaluation region $\rightarrow  {a}_{1} \leftarrow$ Theorem o.mbr o. ${pcr}\left( {c}_{4}\right)$ (a) Concentric $o.{mbr},o.{pcr}\left( {c}_{ \dashv  }\right)$ ,and $o.{pcr}\left( {c}_{ \vdash  }\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_28.jpg?x=372&y=337&w=1082&h=453&r=0"/>

Fig. 14. Analysis of o. $\mathop{\Pr }\limits_{\text{comp }}$ for ${t}_{q} \leq  {C}_{m}$ .

图14. o的分析。 $\mathop{\Pr }\limits_{\text{comp }}$ 对应于 ${t}_{q} \leq  {C}_{m}$ 。

<!-- Media -->

Equations (16) and (17) are valid even if the PCRs of an object are not concentric. In that case, the only modification to all our analysis lies in the ring construction. To explain this,let $p$ be the centroid of o.mbr with coordinates $p\left\lbrack  1\right\rbrack  ,\ldots ,p\left\lbrack  d\right\rbrack$ . On each dimension $i \in  \left\lbrack  {1,d}\right\rbrack$ ,the left (or right) edge of the outer rectangle of the ring is obtained by moving the left (or right) edge of ${r}_{q}$ inward at the distance of $p\left\lbrack  i\right\rbrack   - o.{pc}{r}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{ \vdash  }\right)$ (or $o.{pc}{r}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{ \vdash  }\right)  - p\left\lbrack  i\right\rbrack$ ). The inner rectangle is decided in the same way,except that ${c}_{ \vdash  }$ is replaced with ${c}_{ \dashv  }$ .

即使对象的可能区域（PCR，Possible Configuration Region）不同心，方程(16)和(17)仍然有效。在这种情况下，我们所有分析的唯一修改在于环的构造。为了解释这一点，设$p$为对象最小边界矩形（o.mbr，object minimum bounding rectangle）的质心，其坐标为$p\left\lbrack  1\right\rbrack  ,\ldots ,p\left\lbrack  d\right\rbrack$。在每个维度$i \in  \left\lbrack  {1,d}\right\rbrack$上，环的外矩形的左（或右）边缘是通过将${r}_{q}$的左（或右）边缘向内移动距离$p\left\lbrack  i\right\rbrack   - o.{pc}{r}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{ \vdash  }\right)$（或$o.{pc}{r}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{ \vdash  }\right)  - p\left\lbrack  i\right\rbrack$）得到的。内矩形的确定方式相同，只是用${c}_{ \dashv  }$代替${c}_{ \vdash  }$。

6.1.2 Case 2: ${t}_{q} \leq  {C}_{m}$ . We will derive $o.P{r}_{\text{comp }}$ using the methodology in Section 6.1.1. Figure 14(a) repeats the content of Figure 13(a), except that here ${c}_{ \vdash  }\left( {c}_{ \dashv  }\right)$ should be understood as the smallest (largest) U-catalog value at least (most) ${t}_{q}$ . The approximate ER (evaluation region) also consists of a ring (the hatched region) and four grey boxes. Specifically, the outer (inner) rectangle of the ring shares a common centroid with ${r}_{q}$ ,but is longer than ${r}_{q}$ by $2{a}_{2}\left( {2{a}_{1}}\right)$ and $2{b}_{2}\left( {2{b}_{1}}\right)$ on the horizontal and vertical dimensions,respectively. The grey boxes are obtained in the same way as in Figure 13(b).

6.1.2 情况2：${t}_{q} \leq  {C}_{m}$。我们将使用6.1.1节中的方法推导$o.P{r}_{\text{comp }}$。图14(a)重复了图13(a)的内容，只是这里的${c}_{ \vdash  }\left( {c}_{ \dashv  }\right)$应理解为至少（至多）为${t}_{q}$的最小（最大）U - 目录值。近似评估区域（ER，Evaluation Region）同样由一个环（阴影区域）和四个灰色框组成。具体来说，环的外（内）矩形与${r}_{q}$有共同的质心，但在水平和垂直维度上分别比${r}_{q}$长$2{a}_{2}\left( {2{a}_{1}}\right)$和$2{b}_{2}\left( {2{b}_{1}}\right)$。灰色框的获取方式与图13(b)相同。

The approximate ER bears two properties identical to those in the previous subsection. Namely, $o$ can definitely be pruned or validated if the centroid of $o.{mbr}$ is outside the hatched and grey area,whereas $P{r}_{\text{range }}\left( {o,{r}_{q}}\right)$ must be numerically calculated if the centroid falls in the hatched region. Therefore, $o.P{r}_{comp}$ is guaranteed to fall in a range $\left\lbrack  {o.P{r}_{comp}^{LB},o.P{r}_{comp}^{UB}}\right\rbrack$ ,where the lower and upper bounds are given by two equations analogous to Eqs. (16) and (17):

近似评估区域具有与上一小节相同的两个性质。即，如果$o.{mbr}$的质心在阴影和灰色区域之外，则$o$肯定可以被剪枝或验证；而如果质心落在阴影区域内，则必须对$P{r}_{\text{range }}\left( {o,{r}_{q}}\right)$进行数值计算。因此，$o.P{r}_{comp}$保证落在一个范围$\left\lbrack  {o.P{r}_{comp}^{LB},o.P{r}_{comp}^{UB}}\right\rbrack$内，其中下限和上限由两个类似于方程(16)和(17)的方程给出：

$$
{o.{Pr}}_{comp}^{LB} = \mathop{\prod }\limits_{{i = 1}}^{d}\left( {s{l}_{q\left\lbrack  i\right\rbrack  } + {o.{sl}}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \dashv  }\right) }\right)  - \mathop{\prod }\limits_{{i = 1}}^{d}\left( {s{l}_{q\left\lbrack  i\right\rbrack  } + {o.{sl}}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \vdash  }\right) }\right) \text{,and} \tag{18}
$$

$$
o.P{r}_{comp}^{UB} = o \cdot  P{r}_{comp}^{LB} + \mathop{\prod }\limits_{{i = 1}}^{d}\left( {o.s{l}_{{mbr}\left\lbrack  i\right\rbrack  } + o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \vdash  }\right) }\right) , \tag{19}
$$

Finally, the above formulae are correct in any dimensionality, even when the PCRs of an object are not concentric.

最后，上述公式在任何维度上都是正确的，即使对象的可能区域（PCR）不同心。

<!-- Media -->

<!-- figureText: o. ${pcr}\left( {c}_{m}\right)$ ${b}_{i}$ A ${r}_{q}$ $\rightarrow  {a}_{1} \leftarrow$ (b) Approximating the evaluation region 土 ${b}_{3}$ ${a}_{3}$ o.mbr (a) Concentric $o.{mbr}$ ,and $o.{pcr}\left( {c}_{m}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_29.jpg?x=484&y=341&w=833&h=379&r=0"/>

Fig. 15. Analysis of o.Promp for ${C}_{m} < {t}_{q} \leq  1 - {C}_{m}$ .

图15. 对${C}_{m} < {t}_{q} \leq  1 - {C}_{m}$的o.Promp分析。

<!-- figureText: o.mbr $\rightarrow  S \rightarrow  {SA}$ (b) The evaluation region without PCRs (for any ${t}_{q}$ ) Proof. (a) Extents of o.mbr -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_29.jpg?x=506&y=796&w=788&h=414&r=0"/>

Fig. 16. Explanation about why refinement is more frequent without PCRs.

图16. 解释为什么没有可能区域（PCR）时细化操作更频繁。

<!-- Media -->

6.1.3 Case 3: ${C}_{m} < {t}_{q} \leq  1 - {C}_{m}$ . This remaining case is the simplest: both ${c}_{ \vdash  }$ and ${c}_{ \dashv  }$ correspond to ${c}_{m}$ ,that is,the largest U-catalog value. The counterpart of Figures 13 and 14 here is Figure 15. Since the derivation is similar to that of the previous two cases,we directly present the final equations of $o.P{r}_{comp}^{LB}$ and o. $\mathop{\Pr }\limits_{\text{comp }}^{{UB}}$ (which,again,are applicable in any dimensionality,regardless of whether $o.{mbr}$ and $o.{pcr}\left( {c}_{m}\right)$ are concentric):

6.1.3 情况3：${C}_{m} < {t}_{q} \leq  1 - {C}_{m}$。剩下的这种情况最简单：${c}_{ \vdash  }$和${c}_{ \dashv  }$都对应于${c}_{m}$，即最大的U - 目录值。这里与图13和图14对应的是图15。由于推导过程与前两种情况类似，我们直接给出$o.P{r}_{comp}^{LB}$和o. $\mathop{\Pr }\limits_{\text{comp }}^{{UB}}$的最终方程（同样，这些方程适用于任何维度，无论$o.{mbr}$和$o.{pcr}\left( {c}_{m}\right)$是否同心）：

$$
{o.{Pr}}_{comp}^{LB} = \mathop{\prod }\limits_{{i = 1}}^{d}\left( {s{l}_{q\left\lbrack  i\right\rbrack  } + {o.{sl}}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{m}\right) }\right)  - \mathop{\prod }\limits_{{i = 1}}^{d}\left( {s{l}_{q\left\lbrack  i\right\rbrack  } - {o.{sl}}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{m}\right) }\right) \text{,and} \tag{20}
$$

$$
o.P{r}_{comp}^{UB} = o \cdot  P{r}_{comp}^{LB} + \mathop{\prod }\limits_{{i = 1}}^{d}\left( {o.s{l}_{{mbr}\left\lbrack  i\right\rbrack  } - o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( {c}_{m}\right) }\right) , \tag{21}
$$

6.1.4 Discussion. The analysis of Sections 6.1.1 through 6.1.3 also explain why PCRs can effectively reduce the refinement cost. Imagine that we do not have any PCR. In this case,regardless of ${t}_{q}$ ,pruning (or validation) of $o$ is possible if and only if ${r}_{q}$ does not intersect (or completely covers) o.mbr. With respect to the extents of o.mbr in Figure 16(a), the grey portion of Figure 16(b) illustrates the exact evaluation region of $o$ ,which unions the centroids of all o.mbr causing $o.P{r}_{comp}$ to be numerically computed. The region is a ring,whose outer (inner) boundary is a rectangle that shares the centroid of ${r}_{q}$ ,and is longer (shorter) than ${r}_{q}$ by $2{a}_{3}$ on the horizontal dimension,and by $2{b}_{3}$ on the vertical dimension. The area of the evaluation region is much larger than those of the approximate evaluation regions in Figures 13(b), 14(b), and 15(b), which, as explained earlier,give pessimistic upper bounds of o.Pr comp (for large,small, and median ${t}_{q}$ respectively) when PCRs are used.

6.1.4 讨论。6.1.1节至6.1.3节的分析也解释了为什么部分剪枝规则（PCR，Partial Pruning Rules）能够有效降低细化成本。假设我们没有任何部分剪枝规则。在这种情况下，当且仅当${r}_{q}$与对象最小边界矩形（o.mbr，object minimum bounding rectangle）不相交（或完全覆盖）时，才可能对$o$进行剪枝（或验证）。对于图16(a)中o.mbr的范围，图16(b)中的灰色部分展示了$o$的确切评估区域，该区域是所有导致$o.P{r}_{comp}$需要进行数值计算的o.mbr质心的并集。该区域是一个环形，其外（内）边界是一个与${r}_{q}$共享质心的矩形，在水平维度上比${r}_{q}$长（短）$2{a}_{3}$，在垂直维度上比${r}_{q}$长（短）$2{b}_{3}$。该评估区域的面积远大于图13(b)、图14(b)和图15(b)中的近似评估区域，如前文所述，当使用部分剪枝规则时，这些近似评估区域分别给出了对象剪枝计算复杂度（o.Pr comp，object Pruning computation complexity）对于大、小和中等${t}_{q}$的悲观上界。

The above discussion applies only to voluminous queries (i.e., the projection of ${r}_{q}$ is longer than that of o.mbr on every dimension). When a query is not voluminous,its evaluation region is significantly more complex,because $o$ can be validated in many additional ways (the voluminous requirement simplifies Theorem 2,as mentioned at the beginning of Section 6.1). However, $o.P{r}_{comp}$ can still be estimated as follows. First,we generate a large number ${s}_{1}$ of $o.{mbr}$ , by randomly distributing their centroids in the workspace. Then, we count the number ${s}_{2}$ of $o.{mbr}$ that leads to numerical evaluation of $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ ,after which $\mathop{\Pr }\limits_{\text{comp }}\left( {o,{r}_{q}}\right)$ can be approximated as ${s}_{1}/{s}_{2}$ . We note that nonvoluminous queries are much less important than the voluminous counterpart for index optimization, as explained at the beginning of Section 6.1.

上述讨论仅适用于大规模查询（即${r}_{q}$在每个维度上的投影都比o.mbr的投影长）。当查询不是大规模时，其评估区域会显著复杂得多，因为$o$可以通过许多额外的方式进行验证（如6.1节开头所述，大规模要求简化了定理2）。然而，$o.P{r}_{comp}$仍可按如下方式估计。首先，我们通过在工作空间中随机分布其质心，生成大量（${s}_{1}$个）$o.{mbr}$。然后，我们统计导致对$\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$进行数值评估的$o.{mbr}$的数量（${s}_{2}$个），之后$\mathop{\Pr }\limits_{\text{comp }}\left( {o,{r}_{q}}\right)$可以近似为${s}_{1}/{s}_{2}$。如6.1节开头所述，我们注意到对于索引优化而言，非大规模查询远不如大规模查询重要。

### 6.2 A Cost Model

### 6.2 成本模型

In this section, we will derive analytical formulae that quantify the overhead of nonfuzzy range search on multidimensional uncertain data. Specifically, let $q$ be a query with a probability threshold ${t}_{q}$ ,and a rectangular search region ${r}_{q}$ that has projection length $s{l}_{q\left\lbrack  i\right\rbrack  }$ on the $i$ th dimension $\left( {1 \leq  i \leq  d}\right)$ ,and its centroid follows a uniform distribution in the workspace. The objective is to compute the expected query time $\operatorname{cost}\left( q\right)$ ,which sums the cost ${\operatorname{cost}}_{flt}\left( q\right)$ of the filter step,and the refinement overhead ${\operatorname{cost}}_{rfn}\left( q\right)$ .

在本节中，我们将推导量化多维不确定数据上非模糊范围搜索开销的解析公式。具体而言，设$q$是一个具有概率阈值${t}_{q}$的查询，其矩形搜索区域为${r}_{q}$，该区域在第$i$维（$\left( {1 \leq  i \leq  d}\right)$）上的投影长度为$s{l}_{q\left\lbrack  i\right\rbrack  }$，并且其质心在工作空间中遵循均匀分布。目标是计算预期查询时间$\operatorname{cost}\left( q\right)$，它是过滤步骤的成本${\operatorname{cost}}_{flt}\left( q\right)$和细化开销${\operatorname{cost}}_{rfn}\left( q\right)$之和。

In Section 6.2.1, we settle the problem for a "regular" dataset generated as follows. First,we create $n$ objects $o$ with the same o.ur and o.pdf. Then, these objects are positioned in the workspace such that the centroids of their o.mbr distribute uniformly. Section 6.2.2 generalizes our analytical results to arbitrary datasets.

在6.2.1节中，我们解决针对按如下方式生成的“规则”数据集的问题。首先，我们创建$n$个具有相同对象不确定区域（o.ur，object uncertain region）和对象概率密度函数（o.pdf，object probability density function）的对象$o$。然后，将这些对象放置在工作空间中，使得它们的o.mbr质心均匀分布。6.2.2节将我们的分析结果推广到任意数据集。

6.2.1 Regular Data. In a regular dataset, the PCRs (at the same U-catalog value $c$ ) of all objects are equally large,that is, ${o}_{1}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)  = {o}_{2}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ for any dimension $i \in  \left\lbrack  {1,d}\right\rbrack$ and arbitrary objects ${o}_{1},{o}_{2}$ . Therefore,for every object $o$ , the likelihood $o.P{r}_{comp}$ that $o$ can be neither pruned nor validated is equivalent. This leads to:

6.2.1 规则数据。在规则数据集中，所有对象的PCR（在相同的U目录值 $c$ 下）大小相等，即对于任意维度 $i \in  \left\lbrack  {1,d}\right\rbrack$ 和任意对象 ${o}_{1},{o}_{2}$ ，有 ${o}_{1}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)  = {o}_{2}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ 。因此，对于每个对象 $o$ ，$o$ 既不能被剪枝也不能被验证的可能性 $o.P{r}_{comp}$ 是相等的。这导致：

$$
{n}_{NE} = n \cdot  o.P{r}_{comp}, \tag{22}
$$

where ${n}_{NE}$ is the number of numerical evaluations needed to answer a query. As for $o.P{r}_{comp}$ ,we employ the following estimation:

其中 ${n}_{NE}$ 是回答一个查询所需的数值评估次数。至于 $o.P{r}_{comp}$ ，我们采用以下估计：

$$
o.P{r}_{comp} = \left( {o \cdot  P{r}_{comp}^{LB} + o \cdot  P{r}_{comp}^{UB}}\right) /2, \tag{23}
$$

where $o.P{r}_{comp}^{LB}$ and $o.P{r}_{comp}^{UB}$ are represented in Eqs. (16) and (17) for ${t}_{q} > 1 - {C}_{m}$ , Eqs. (18) and (19) for ${t}_{q} \leq  {C}_{m}$ ,or Eqs. (20) and (21) for ${C}_{m} < {t}_{q} \leq  1 - {C}_{m}$ . Hence,

其中 $o.P{r}_{comp}^{LB}$ 和 $o.P{r}_{comp}^{UB}$ 对于 ${t}_{q} > 1 - {C}_{m}$ 在式(16)和式(17)中表示，对于 ${t}_{q} \leq  {C}_{m}$ 在式(18)和式(19)中表示，或者对于 ${C}_{m} < {t}_{q} \leq  1 - {C}_{m}$ 在式(20)和式(21)中表示。因此，

$$
{\operatorname{cost}}_{rfn} = {n}_{NE} \cdot  o.{\operatorname{cost}}_{rfn} = n \cdot  o.P{r}_{\text{comp }} \cdot  o.{\operatorname{cost}}_{rfn}, \tag{24}
$$

where ${\text{o.cost}}_{rfn}$ is the overhead of MC (Monte-Carlo) for calculating the qualification probability of a single object (see Section 5.3). o.cos ${t}_{rfn}$ is identical for all objects,since it depends only on the cost of loading o.pdf and the number of samples used in MC.

其中 ${\text{o.cost}}_{rfn}$ 是蒙特卡罗（Monte - Carlo，MC）计算单个对象合格概率的开销（见5.3节）。对于所有对象，o.cos ${t}_{rfn}$ 是相同的，因为它仅取决于加载o.pdf的成本和MC中使用的样本数量。

On the other hand, ${\operatorname{cost}}_{flt}\left( q\right)$ corresponds to the time of accessing the leaf nodes of the U-tree in the filter step (we do not include the overhead of visiting the intermediate nodes, since it is by far dominated by the cost at the leaf level, especially if the intermediate levels are buffered). The probability ${nd}.P{r}_{acs}$ that a leaf node ${nd}$ is accessed depends on the characteristics of the data inside ${nd}$ . For a regular dataset,the characteristics are the same across the entire workspace; hence, nd.Pr ${}_{acs}$ is equivalent for all leaf nodes. It follows that:

另一方面，${\operatorname{cost}}_{flt}\left( q\right)$ 对应于过滤步骤中访问U树叶子节点的时间（我们不包括访问中间节点的开销，因为到目前为止，它远小于叶子节点级别的成本，特别是如果中间级别有缓存的话）。叶子节点 ${nd}$ 被访问的概率 ${nd}.P{r}_{acs}$ 取决于 ${nd}$ 内数据的特征。对于规则数据集，整个工作空间的数据特征是相同的；因此，对于所有叶子节点，nd.Pr ${}_{acs}$ 是相等的。由此可得：

$$
{n}_{NA} = n \cdot  o.P{r}_{acs}, \tag{25}
$$

where ${n}_{NA}$ the number of nodes accessed in processing a query. Therefore,

其中 ${n}_{NA}$ 是处理一个查询时访问的节点数量。因此，

$$
{\operatorname{cost}}_{flt}\left( q\right)  = {n}_{NA} \cdot  {\operatorname{cost}}_{ranIO} = \left( {n/f}\right)  \cdot  {nd} \cdot  \mathop{\Pr }\limits_{{acs}} \cdot  {\operatorname{cost}}_{ranIO}, \tag{26}
$$

where $f$ is the average fanout of a node, $n/f$ is the total number of leaf nodes, and ${\operatorname{cost}}_{\text{ranIO }}$ the time of a random I/O. Note that $f$ is determined by the page size,and very importantly,the U-catalog size $m$ (recall that each leaf entry keeps $m$ PCRs of an object).

其中 $f$ 是节点的平均扇出，$n/f$ 是叶子节点的总数，${\operatorname{cost}}_{\text{ranIO }}$ 是一次随机I/O的时间。注意，$f$ 由页面大小决定，并且非常重要的是，由U目录大小 $m$ 决定（回想一下，每个叶子条目保存一个对象的 $m$ 个PCR）。

Let $e$ be the parent entry of a leaf node ${nd}$ . As mentioned in Section 5.1,for each U-catalog value $c,e$ retains (i) $e.{mbr}\left( c\right)$ ,which is the MBR of the $o.{pcr}\left( c\right)$ of all the objects $o$ in ${nd}$ ,and (ii) a value $e.{sl}\left( c\right)$ ,equal to the smallest projection length of any o. ${pcr}\left( c\right)$ on any dimension. In the sequel,we use $e.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }\left( c\right)$ to denote the projection length of $e.{mbr}\left( c\right)$ on the $i$ th dimension.

设 $e$ 是叶子节点 ${nd}$ 的父条目。如5.1节所述，对于每个U目录值 $c,e$ 保留：(i) $e.{mbr}\left( c\right)$ ，它是 ${nd}$ 中所有对象 $o$ 的 $o.{pcr}\left( c\right)$ 的最小边界矩形（MBR）；(ii) 值 $e.{sl}\left( c\right)$ ，等于任何o. ${pcr}\left( c\right)$ 在任何维度上的最小投影长度。在后续内容中，我们用 $e.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }\left( c\right)$ 表示 $e.{mbr}\left( c\right)$ 在第 $i$ 维上的投影长度。

Calculating ${nd}.P{r}_{acs}$ requires the values of $e.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }\left( c\right)$ and $e.{sl}\left( c\right)$ . The analysis of the former can be reduced to estimating the MBR size of a leaf node in an R-tree,by regarding ${nd}$ as a leaf R-tree node,and $e.{mbr}\left( c\right)$ its MBR. Leveraging the findings ${}^{3}$ of Theodoridis and Sellis [1996],we have:

计算 ${nd}.P{r}_{acs}$ 需要 $e.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }\left( c\right)$ 和 $e.{sl}\left( c\right)$ 的值。对前者的分析可以简化为估计 R 树（R-tree）中一个叶节点的最小边界矩形（MBR，Minimum Bounding Rectangle）大小，将 ${nd}$ 视为一个叶 R 树节点，$e.{mbr}\left( c\right)$ 为其最小边界矩形。利用西奥多里迪斯（Theodoridis）和塞利斯（Sellis）[1996]的研究结果 ${}^{3}$，我们有：

$$
{\text{ e.sh }}_{{mbr}\left\lbrack  i\right\rbrack  }\left( c\right)  = {\text{ o.sl }}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)  + {\left( f/n\right) }^{1/d} \tag{27}
$$

where $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ captures the projection length of the $o.{pcr}\left( c\right)$ of an object $o$ on the $i$ th dimension. On the other hand,the analysis of $e.{sl}\left( c\right)$ is straightforward:

其中 $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ 表示对象 $o$ 的 $o.{pcr}\left( c\right)$ 在第 $i$ 维上的投影长度。另一方面，对 $e.{sl}\left( c\right)$ 的分析很直接：

$$
\text{ e.sl }\left( c\right)  = \mathop{\min }\limits_{{i = 1}}^{d}{\text{ o.sl }}_{\text{ ok }\left\lbrack  i\right\rbrack  }\left( c\right) . \tag{28}
$$

We are ready to elaborate the derivation of ${nd}.P{r}_{acs}$ . Recall that ${nd}$ must be visited,if and only if the following "access condition" holds: ${t}_{q}$ is at most the $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$ returned by Algorithm 2. Let us first consider the case ${t}_{q} > 1 - {C}_{m}$ , where we use ${c}_{ \vdash  }$ to denote the smallest U-catalog value at least $1 - {t}_{q}$ . Thus, the access condition is satisfied only when the rectangle ${r}_{q} \cap  e \cdot  {mbr}\left( {c}_{ \vdash  }\right)$ has a projection length at least $e.{sl}\left( {c}_{ \vdash  }\right)$ on all the dimensions. Note that ${c}_{ \vdash  }$ may not exist,in which scenario the access condition is equivalent to ${r}_{q}$ intersecting e.mbr(0). Similarly,for ${t}_{q} \leq  1 - {C}_{m}$ ,we deploy ${c}_{ \dashv  }$ to represent the largest $\mathrm{U}$ - catalog value at most ${t}_{q}$ (there is always such a value). Accordingly,the access condition is valid if and only if $e.{mbr}\left( {c}_{ \dashv  }\right)$ intersects ${r}_{q}$ .

我们准备详细阐述 ${nd}.P{r}_{acs}$ 的推导过程。回顾一下，当且仅当满足以下“访问条件”时，才必须访问 ${nd}$：${t}_{q}$ 至多为算法 2 返回的 $U{B}_{\text{range }}\left( {e,{r}_{q}}\right)$。让我们首先考虑 ${t}_{q} > 1 - {C}_{m}$ 的情况，其中我们用 ${c}_{ \vdash  }$ 表示至少为 $1 - {t}_{q}$ 的最小 U 目录值。因此，只有当矩形 ${r}_{q} \cap  e \cdot  {mbr}\left( {c}_{ \vdash  }\right)$ 在所有维度上的投影长度至少为 $e.{sl}\left( {c}_{ \vdash  }\right)$ 时，访问条件才满足。注意，${c}_{ \vdash  }$ 可能不存在，在这种情况下，访问条件等价于 ${r}_{q}$ 与 e.mbr(0) 相交。类似地，对于 ${t}_{q} \leq  1 - {C}_{m}$，我们用 ${c}_{ \dashv  }$ 表示至多为 ${t}_{q}$ 的最大 $\mathrm{U}$ 目录值（总是存在这样的值）。因此，当且仅当 $e.{mbr}\left( {c}_{ \dashv  }\right)$ 与 ${r}_{q}$ 相交时，访问条件才有效。

---

<!-- Footnote -->

${}^{3}$ Consider an R-tree that indexes $n$ equivalent rectangles,which are uniformly distributed in a $d$ -dimensional workspace,and their side lengths on the $i$ th $\left( {1 \leq  i \leq  d}\right)$ dimension equal $x\left\lbrack  i\right\rbrack$ . Let $y\left\lbrack  i\right\rbrack$ denote the side length,on the $i$ th dimension,of the MBR of a level-1 entry in an R-tree. If each dimension of the workspace has a unit length, $y\left\lbrack  i\right\rbrack$ can be accurately estimated as $x\left\lbrack  i\right\rbrack   + {\left( f/n\right) }^{1/d}$ , where $f$ is the average tree fanout. This result is due to the grid-modeling of the MBRs of the leaf nodes in Theodoridis and Sellis [1996].

${}^{3}$ 考虑一个 R 树，它对 $n$ 个等价矩形进行索引，这些矩形均匀分布在一个 $d$ 维的工作空间中，并且它们在第 $i$ 个 $\left( {1 \leq  i \leq  d}\right)$ 维度上的边长等于 $x\left\lbrack  i\right\rbrack$。用 $y\left\lbrack  i\right\rbrack$ 表示 R 树中一级条目的最小边界矩形在第 $i$ 维上的边长。如果工作空间的每个维度的长度为单位长度，则 $y\left\lbrack  i\right\rbrack$ 可以准确估计为 $x\left\lbrack  i\right\rbrack   + {\left( f/n\right) }^{1/d}$，其中 $f$ 是平均树扇出。这一结果源于西奥多里迪斯和塞利斯 [1996] 对叶节点最小边界矩形的网格建模。

<!-- Footnote -->

---

It is clear from the above discussion that ${nd}.P{r}_{acs}$ is essentially the probability that (a uniformly distributed) ${r}_{q}$ intersects a rectangle $e \cdot  {mbr}\left( c\right)$ in a certain way,where $c$ is an appropriate U-catalog value selected as mentioned earlier. This is a problem that has been studied by Pagel et al. [1993]. Based on their results ${}^{4}$ we have ${nd}.P{r}_{acs} =$

从上述讨论可以清楚地看出，${nd}.P{r}_{acs}$ 本质上是（均匀分布的）${r}_{q}$ 以某种方式与矩形 $e \cdot  {mbr}\left( c\right)$ 相交的概率，其中 $c$ 是如前所述选择的合适 U 目录值。这是帕格尔（Pagel）等人 [1993] 已经研究过的问题。基于他们的研究结果 ${}^{4}$，我们有 ${nd}.P{r}_{acs} =$

$$
\left\{  \begin{array}{ll} \mathop{\prod }\limits_{{i = 1}}^{d}\left( {s{l}_{q\left\lbrack  i\right\rbrack  } + e.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \vdash  }\right)  - e.{sl}\left( {c}_{ \vdash  }\right) }\right) & {if}{t}_{q} > 1 - {C}_{m}\text{ and }{c}_{ \vdash  }\text{ exists } \\  \mathop{\prod }\limits_{{i = 1}}^{d}\left( {s{l}_{q\left\lbrack  i\right\rbrack  } + e.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }\left( 0\right) }\right) & {if}{t}_{q} > 1 - {C}_{m}\text{ and }{c}_{ \vdash  }\text{ does not exist } \\  \mathop{\prod }\limits_{{i = 1}}^{d}\left( {s{l}_{q\left\lbrack  i\right\rbrack  } + e.s{l}_{{mbr}\left\lbrack  i\right\rbrack  }\left( {c}_{ \dashv  }\right) }\right) & {if}{t}_{q} \leq  1 - {C}_{m} \end{array}\right.  \tag{29}
$$

where functions ${\operatorname{e.sl}}_{{mbr}\left\lbrack  i\right\rbrack  }\left( \text{.}\right) {andelsl}\left( \text{.}\right) {arepresentedinEqs}.\left( {27}\right) {and}\left( {28}\right) ,{re}$ - spectively.

其中函数 ${\operatorname{e.sl}}_{{mbr}\left\lbrack  i\right\rbrack  }\left( \text{.}\right) {andelsl}\left( \text{.}\right) {arepresentedinEqs}.\left( {27}\right) {and}\left( {28}\right) ,{re}$ 分别为。

All the components in Eqs. (24) and (26) have been represented as functions of the dimensionality $d$ ,the query parameters,the PCR sizes of an object,the dataset cardinality $n$ ,and the node fanout $f$ ,all of which are readily obtainable. Therefore, we have derived the expected cost of nonfuzzy range search on a regular dataset.

式(24)和(26)中的所有分量都已表示为维度 $d$、查询参数、对象的PCR大小、数据集基数 $n$ 和节点扇出 $f$ 的函数，所有这些都很容易获得。因此，我们推导出了在常规数据集上进行非模糊范围搜索的预期成本。

6.2.2 Arbitrary Data. The dataset-dependent parameters to the above "regular" cost model involve only the cardinality $n$ ,and projection length $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ of the $o.{pcr}\left( c\right)$ of an object $o$ on the $i$ th dimension $\left( {1 \leq  i \leq  d}\right)$ ,where $c$ is a U-catalog value. Given an arbitrary dataset,a naive approach of applying the model to estimate $\operatorname{cost}\left( q\right)$ is to feed those parameters with the average statistics. In particular, $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ can be set to the average projection length on the $i$ th dimension of the ${o}^{\prime } \cdot  \operatorname{pcr}\left( c\right)$ of all objects ${o}^{\prime }$ in the dataset. This approach, however, may not produce accurate estimates, due to the potentially large variance in objects' projection lengths.

6.2.2 任意数据。上述“常规”成本模型中与数据集相关的参数仅涉及基数 $n$ 以及对象 $o$ 的 $o.{pcr}\left( c\right)$ 在第 $i$ 维 $\left( {1 \leq  i \leq  d}\right)$ 上的投影长度 $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$，其中 $c$ 是一个U目录值。对于任意数据集，应用该模型来估计 $\operatorname{cost}\left( q\right)$ 的一种简单方法是将这些参数用平均统计值代入。具体而言，可以将 $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ 设置为数据集中所有对象 ${o}^{\prime }$ 的 ${o}^{\prime } \cdot  \operatorname{pcr}\left( c\right)$ 在第 $i$ 维上的平均投影长度。然而，由于对象投影长度可能存在较大差异，这种方法可能无法产生准确的估计。

This problem can be alleviated using the local smoothing technique [Theodor-idis and Sellis 1996] originally proposed for capturing R-tree performance. The basic observation underlying the technique is that, the variance in the characteristics of objects in a small region is (possibly much) lower than that in the whole workspace. Hence, the application of the regular model to a subset of the dataset tends to be more effective.

这个问题可以使用最初为评估R树性能而提出的局部平滑技术[Theodor - idis和Sellis 1996]来缓解。该技术的基本观察结果是，小区域内对象特征的差异（可能）远低于整个工作空间内的差异。因此，将常规模型应用于数据集的一个子集往往更有效。

Specifically,we divide the data space into a grid of ${\lambda }^{d}$ identical cells,where $\lambda$ is the resolution of the grid,and equals 5 in our experiments. For each cell ${cl}$ ,we will develop a value ${cl}.\operatorname{cost}\left( q\right)$ that estimates the expected cost of $q$ ,when the centroid of $q$ .mbr lies in ${cl}$ . Obviously, ${\lambda }^{d}$ values are obtained after examining all the cells. Our final estimate for the expected cost $\operatorname{cost}\left( q\right)$ (of all queries) equals the average of these values.

具体来说，我们将数据空间划分为 ${\lambda }^{d}$ 个相同单元格的网格，其中 $\lambda$ 是网格的分辨率，在我们的实验中等于5。对于每个单元格 ${cl}$，当 $q$ 的最小边界矩形（MBR）的质心位于 ${cl}$ 中时，我们将计算一个值 ${cl}.\operatorname{cost}\left( q\right)$ 来估计 $q$ 的预期成本。显然，在检查所有单元格后会得到 ${\lambda }^{d}$ 个值。我们对（所有查询的）预期成本 $\operatorname{cost}\left( q\right)$ 的最终估计值等于这些值的平均值。

---

<!-- Footnote -->

${}^{4}$ Consider two $d$ -dimensional axis-parallel rectangles whose side lengths on the $i$ th $\left( {1 \leq  i \leq  d}\right)$ dimension equal $x\left\lbrack  i\right\rbrack$ and $y\left\lbrack  i\right\rbrack$ ,respectively. Let the two rectangles (independently) uniformly distribute in a workspace where each axis has a unit length. Examine the probability that they intersect into a box whose side length on the $i$ th $\left( {1 \leq  i \leq  d}\right)$ dimension is at least $z\left\lbrack  i\right\rbrack$ . The analysis of Pagel et al. [1993] shows that the probability equals ${\Pi }_{i = 1}^{d}\left( {x\left\lbrack  i\right\rbrack   + y\left\lbrack  i\right\rbrack   - z\left\lbrack  i\right\rbrack  }\right)$ . As a corollary,the probability that the two rectangles intersect is ${\Pi }_{i = 1}^{d}\left( {x\left\lbrack  i\right\rbrack   + y\left\lbrack  i\right\rbrack  }\right)$ .

${}^{4}$ 考虑两个 $d$ 维的轴平行矩形，它们在第 $i$ 个 $\left( {1 \leq  i \leq  d}\right)$ 维度上的边长分别为 $x\left\lbrack  i\right\rbrack$ 和 $y\left\lbrack  i\right\rbrack$。假设这两个矩形（相互独立地）均匀分布在一个各轴长度均为单位长度的工作空间中。研究它们相交形成一个在第 $i$ 个 $\left( {1 \leq  i \leq  d}\right)$ 维度上边长至少为 $z\left\lbrack  i\right\rbrack$ 的盒子的概率。Pagel 等人（1993 年）的分析表明，该概率等于 ${\Pi }_{i = 1}^{d}\left( {x\left\lbrack  i\right\rbrack   + y\left\lbrack  i\right\rbrack   - z\left\lbrack  i\right\rbrack  }\right)$。作为推论，这两个矩形相交的概率为 ${\Pi }_{i = 1}^{d}\left( {x\left\lbrack  i\right\rbrack   + y\left\lbrack  i\right\rbrack  }\right)$。

<!-- Footnote -->

---

It remains to explain the computation of ${cl}.\operatorname{cost}\left( q\right)$ . For this purpose,we need the number ${cl}.n$ of objects ${o}^{\prime }$ such that the centroids of their ${o}^{\prime }.{mbr}$ are contained by ${cl}$ . Furthermore,for these objects ${o}^{\prime }$ and every value $c$ in the U-catalog,we compute the average projection length ${cl}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ on the $i$ th dimension $(1 \leq  i \leq$ $d$ ) of ${o}^{\prime }$ . ${pcr}\left( c\right)$ . Then, ${cl}$ .cost(q)is calculated by the regular model in Section 6.2.1, after setting $n = {cl}.n \cdot  {\lambda }^{d}$ ,and $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)  = {cl}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ . Note that the statistics (i.e., ${cl}.n$ and ${cl}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ ) required for calculating $\operatorname{cost}\left( q\right)$ can be obtained by a single scan of the dataset.

接下来需要解释 ${cl}.\operatorname{cost}\left( q\right)$ 的计算方法。为此，我们需要知道对象 ${o}^{\prime }$ 的数量 ${cl}.n$，使得它们的 ${o}^{\prime }.{mbr}$ 的质心包含在 ${cl}$ 中。此外，对于这些对象 ${o}^{\prime }$ 以及 U - 目录中的每个值 $c$，我们计算 ${o}^{\prime }$ 在第 $i$ 个维度 $(1 \leq  i \leq$ $d$ 上的平均投影长度 ${cl}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$。${pcr}\left( c\right)$。然后，在设置 $n = {cl}.n \cdot  {\lambda }^{d}$ 和 $o.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)  = {cl}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$ 之后，根据 6.2.1 节中的常规模型计算 ${cl}$.cost(q)。请注意，计算 $\operatorname{cost}\left( q\right)$ 所需的统计信息（即 ${cl}.n$ 和 ${cl}.s{l}_{{pcr}\left\lbrack  i\right\rbrack  }\left( c\right)$）可以通过对数据集进行一次扫描获得。

### 6.3 Optimizing the U-catalog Size

### 6.3 优化 U - 目录大小

We close this section by elaborating the procedures of tuning the U-catalog size $m$ . Remember that the cost model of Section 6.2.1 is,in fact,a function $m$ , which influences the query overhead in two important ways. First, $m$ directly decides the node fanout $f$ (which drops if more PCRs are retained for each object). Second,various $m$ leads to different values in the U-catalog,which in turn affect the ${c}_{ + }$ and ${c}_{ + }$ in Eqs. (16),(17),(18),(19),and (29). These formulae determine both the filter and refinement cost.

在本节结尾，我们详细阐述调整 U - 目录大小 $m$ 的过程。请记住，6.2.1 节中的成本模型实际上是一个关于 $m$ 的函数，它以两种重要方式影响查询开销。首先，$m$ 直接决定节点扇出 $f$（如果为每个对象保留更多的 PCR，则节点扇出会降低）。其次，不同的 $m$ 会导致 U - 目录中的值不同，进而影响方程 (16)、(17)、(18)、(19) 和 (29) 中的 ${c}_{ + }$ 和 ${c}_{ + }$。这些公式决定了过滤和细化成本。

The number of possibilities of $m$ is limited,because this parameter should be a small integer, for example, no more than 10 . Therefore, we use our analytical formulae to predict the expected query cost for every possible value of $m$ ,and choose the value that yields the lowest prediction. This strategy, however, raises two questions. First,what are the $m$ values in the U-catalog? Second,how to decide the query parameters (i.e., $s{l}_{q\left\lbrack  1\right\rbrack  },\ldots ,s{l}_{q\left\lbrack  d\right\rbrack  }$ and ${t}_{q}$ ) to be plugged into the model?

$m$的可能取值数量是有限的，因为该参数应为一个小整数，例如不超过10。因此，我们使用解析公式来预测$m$每个可能取值的预期查询成本，并选择预测值最低的那个取值。然而，这种策略引发了两个问题。首先，U目录中的$m$值是多少？其次，如何确定要代入模型的查询参数（即$s{l}_{q\left\lbrack  1\right\rbrack  },\ldots ,s{l}_{q\left\lbrack  d\right\rbrack  }$和${t}_{q}$）？

We settle the first question by fixing the first value to 0 , and placing the other $m - 1$ values evenly in the range of $(0,{0.5}\rbrack$ . For example,for $m = 2$ , the U-catalog consists of $\{ 0,1/4\}$ ,whereas the catalog becomes $\{ 0,1/6,2/6\}$ for $m = 3$ . The answer to the second question largely depends on the preferences of the database administrator. For instance, s/he could manually select some parameter values that are proved to be popular among users based on the past statistics. Another option is to generate many sets of parameters (e.g., one set concerns large ${r}_{q}$ and ${t}_{q}$ ,while another explores their small counterparts),and deploy our model to produce an estimate for every set. The overall quality of the $m$ under consideration can be gauged by the average of the estimates of all sets (possibly assigning different weights to the estimates of various sets).

我们通过将第一个值固定为0，并将其他$m - 1$值均匀分布在$(0,{0.5}\rbrack$范围内来解决第一个问题。例如，对于$m = 2$，U目录由$\{ 0,1/4\}$组成，而对于$m = 3$，目录变为$\{ 0,1/6,2/6\}$。第二个问题的答案在很大程度上取决于数据库管理员的偏好。例如，他/她可以根据过去的统计数据手动选择一些被证明在用户中很受欢迎的参数值。另一种选择是生成多组参数（例如，一组关注较大的${r}_{q}$和${t}_{q}$，而另一组探索它们的较小对应值），并使用我们的模型为每组参数生成一个估计值。所考虑的$m$的整体质量可以通过所有组估计值的平均值（可能为不同组的估计值分配不同的权重）来衡量。

## 7. PERFORMANCE EVALUATION

## 7. 性能评估

In this section, we empirically evaluate the effectiveness and efficiency of the proposed techniques. All the experiments are performed on a machine running a Pentium IV 3.6GHz CPU. The disk page size is fixed to 4096 bytes. The workspace is normalized to have a domain of $\left\lbrack  {0,{10000}}\right\rbrack$ on every dimension.

在本节中，我们通过实验评估所提出技术的有效性和效率。所有实验均在运行奔腾IV 3.6GHz CPU的机器上进行。磁盘页面大小固定为4096字节。工作空间在每个维度上被归一化为具有$\left\lbrack  {0,{10000}}\right\rbrack$的域。

Given a set $X$ of points,we generate uncertain data as follows to simulate a database storing the positions of mobile clients in a location-based service [Wolfson et al. 1999]. For each point $p \in  X$ ,we create an uncertain object $o$ ,whose uncertainty region $o.{ur}$ is an ${L}_{2}$ circle that centers at $p$ ,and has a radius ${\operatorname{rad}}_{o}$ . We examine the type of $o.{pdf}\left( x\right)$ that has been experimented most in the literature: Gaussian. A traditional Gaussian distribution, however, has an infinite domain,that is, $o.{pdf}\left( x\right)$ is a positive value for any $x$ in the entire workspace,which contradicts the requirement that o.pdf(x)equals 0 at a point $x$ outside o.ur. Hence,following the practice of Cheng et al. [2004b], we consider the "constrained Gaussian" distribution. Formally,let $g\left( x\right)$ be a conventional Gaussian function whose mean falls at the centroid of o.ur, and its variance equals ${\left( ra{d}_{o}/2\right) }^{2}$ (i.e.,the standard deviation ${ra}{d}_{o}/2$ is half the radius of an object's uncertainty region). Then, the corresponding constrained Gaussian o.pdf(x)is defined as:

给定一组点$X$，我们按如下方式生成不确定数据，以模拟一个存储基于位置服务中移动客户端位置的数据库[Wolfson等人，1999]。对于每个点$p \in  X$，我们创建一个不确定对象$o$，其不确定区域$o.{ur}$是以$p$为中心、半径为${\operatorname{rad}}_{o}$的${L}_{2}$圆。我们研究了文献中实验最多的$o.{pdf}\left( x\right)$类型：高斯分布。然而，传统的高斯分布具有无限的域，即对于整个工作空间中的任何$x$，$o.{pdf}\left( x\right)$都是正值，这与o.pdf(x)在o.ur之外的点$x$处等于0的要求相矛盾。因此，遵循Cheng等人[2004b]的做法，我们考虑“受限高斯”分布。形式上，设$g\left( x\right)$为一个传统的高斯函数，其均值位于o.ur的质心，方差等于${\left( ra{d}_{o}/2\right) }^{2}$（即标准差${ra}{d}_{o}/2$是对象不确定区域半径的一半）。那么，相应的受限高斯o.pdf(x)定义为：

$$
\text{o.pdf}\left( x\right)  = \left\{  \begin{array}{ll} g\left( x\right) /{\int }_{x \in  \text{ o.ur }}g\left( x\right) {dx} & \text{ if }x \in  \text{ o.ur } \\  0 & \text{ otherwise } \end{array}\right.  \tag{30}
$$

By setting $X$ respectively to two-dimensional point sets ${LB},{CA}$ ,and ${RAN}$ ,we obtain uncertain databases where the centroids of objects' uncertainty regions follow three different distributions. Both ${LB}$ and ${CA}$ are real spatial datasets downloadable at the R-tree portal (http://www.rtreeportal.org), and are produced from the Tiger project of the US Census Bureau (http://tiger.census.gov).Specifically,the former and latter contain ${53}\mathrm{k}$ and ${62}\mathrm{k}$ points representing addresses in the Long Beach county and Los Angeles,respectively. ${RAN}$ consists of ${100}\mathrm{k}$ points randomly distributed in the workspace. In the sequel,we will use ${LB}$ -rad ${}_{o},{CA}$ -rad ${}_{o}$ ,and ${RAN}$ -rad ${}_{o}$ to refer to the uncertain datasets where the radii of objects’ uncertainty regions equal ${\mathrm{{rad}}}_{i}$ (e.g., ${LB} - {100}$ indicates the dataset created with $X = {LB}$ and ${\operatorname{rad}}_{o} = {100}$ ).

通过分别将$X$设为二维点集${LB},{CA}$和${RAN}$，我们得到了不确定数据库，其中对象不确定区域的质心遵循三种不同的分布。${LB}$和${CA}$均为可从R树门户（http://www.rtreeportal.org）下载的真实空间数据集，它们来自美国人口普查局的Tiger项目（http://tiger.census.gov）。具体而言，前者和后者分别包含${53}\mathrm{k}$和${62}\mathrm{k}$个点，分别代表长滩县和洛杉矶的地址。${RAN}$由在工作空间中随机分布的${100}\mathrm{k}$个点组成。接下来，我们将使用${LB}$ -rad ${}_{o},{CA}$ -rad ${}_{o}$和${RAN}$ -rad ${}_{o}$来表示对象不确定区域半径等于${\mathrm{{rad}}}_{i}$的不确定数据集（例如，${LB} - {100}$表示使用$X = {LB}$和${\operatorname{rad}}_{o} = {100}$创建的数据集）。

The search region ${r}_{q}$ of a nonfuzzy range query is a circle,under the ${L}_{\infty }$ or ${L}_{2}$ norm,that has a radius ${\operatorname{rad}}_{q}$ ,and its center follows the distribution of the points in $X$ (i.e.,the original dataset used to synthesize uncertain objects). In particular,when the ${L}_{\infty }$ norm is used,the search region is a square with side length ${2ra}{d}_{q}$ . On the other hand,for fuzzy range search,the query object $q$ is randomly sampled from the underlying uncertain dataset. As with nonfuzzy retrieval,a fuzzy query is also associated with a defining norm $\left( {L}_{\infty }\right.$ or $\left. {L}_{2}\right)$ , which governs the distance metric in Definition 2.

非模糊范围查询的搜索区域${r}_{q}$是一个在${L}_{\infty }$或${L}_{2}$范数下半径为${\operatorname{rad}}_{q}$的圆，其圆心遵循$X$中各点的分布（即用于合成不确定对象的原始数据集）。特别地，当使用${L}_{\infty }$范数时，搜索区域是边长为${2ra}{d}_{q}$的正方形。另一方面，对于模糊范围搜索，查询对象$q$是从底层不确定数据集中随机采样得到的。与非模糊检索一样，模糊查询也与一个定义范数$\left( {L}_{\infty }\right.$或$\left. {L}_{2}\right)$相关联，该范数决定了定义2中的距离度量。

In our experiments, we will often execute a workload of 10000 similar queries in order to measure their average performance. Specifically, a workload has four properties: (i) fuzzy or nonfuzzy,(ii) ${\operatorname{rad}}_{q}$ (or ${\varepsilon }_{q}$ ) for a nonfuzzy (or fuzzy) workload,(iii) the defining norm,and (iv) ${t}_{q}$ . For instance,"a nonfuzzy ${L}_{\infty }$ workload with ${\operatorname{rad}}_{q} = {500}$ and ${t}_{q} = {0.3}$ " contains purely nonfuzzy queries whose search regions are squares with side length 1000 , and their probability thresholds equal 0.3 . We sometimes set ${t}_{q}$ to a special value "mixed",to indicate a workload where the probability threshold of a query is randomly generated in the range of $\left\lbrack  {{0.1},{0.9}}\right\rbrack$ .

在我们的实验中，我们通常会执行包含10000个相似查询的工作负载，以衡量它们的平均性能。具体来说，一个工作负载有四个属性：（i）模糊或非模糊；（ii）对于非模糊（或模糊）工作负载为${\operatorname{rad}}_{q}$（或${\varepsilon }_{q}$）；（iii）定义范数；（iv）${t}_{q}$。例如，“一个具有${\operatorname{rad}}_{q} = {500}$和${t}_{q} = {0.3}$的非模糊${L}_{\infty }$工作负载”包含纯非模糊查询，其搜索区域是边长为1000的正方形，且它们的概率阈值等于0.3。我们有时会将${t}_{q}$设置为特殊值“混合”，以表示一个查询的概率阈值在$\left\lbrack  {{0.1},{0.9}}\right\rbrack$范围内随机生成的工作负载。

Table II summarizes the data/query parameters mentioned earlier, together with their values to be tested, and the default values in bold fonts.

表二总结了前面提到的数据/查询参数、待测试的值以及用粗体显示的默认值。

<!-- Media -->

Table II. Data and Query Parameters Varied in our Experiments

表二. 我们实验中变化的数据和查询参数

<table><tr><td>Parameter</td><td>Meaning</td><td>Values</td></tr><tr><td>${ra}{d}_{o}$</td><td>The radius of an object's uncertainty region</td><td>5, 100, 250</td></tr><tr><td>${\mathit{{rad}}}_{a}$</td><td>The radius of the search region of a nonfuzzy range query</td><td>250, 500, 750</td></tr><tr><td>${\varepsilon }_{q}$</td><td>The distance threshold of a fuzzy range query</td><td>250, 500, 750</td></tr><tr><td>${t}_{q}$</td><td>The probability threshold of a query</td><td>Uniform in [0.1, 0.9]</td></tr></table>

<table><tbody><tr><td>参数</td><td>含义</td><td>取值</td></tr><tr><td>${ra}{d}_{o}$</td><td>对象不确定区域的半径</td><td>5, 100, 250</td></tr><tr><td>${\mathit{{rad}}}_{a}$</td><td>非模糊范围查询的搜索区域半径</td><td>250, 500, 750</td></tr><tr><td>${\varepsilon }_{q}$</td><td>模糊范围查询的距离阈值</td><td>250, 500, 750</td></tr><tr><td>${t}_{q}$</td><td>查询的概率阈值</td><td>在[0.1, 0.9]内均匀分布</td></tr></tbody></table>

<!-- Media -->

### 7.1 Cost of Evaluating the Exact Qualification Probability

### 7.1 评估精确合格概率的成本

Given a nonfuzzy query,calculating the qualification probability $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ of an object $o$ (Eq. (1)) requires integrating $o.{pdf}\left( x\right)$ ,as given in Eq. (30),inside the intersection between the uncertainty region $o.{ur}$ of $o$ and the search area ${r}_{q}$ . Remember that $o.{ur}$ is an ${L}_{2}$ circle,and ${r}_{q}$ can be a square or an ${L}_{2}$ circle. In any case, $o.{ur} \cap  {r}_{q}$ may have an irregular shape,thus preventing the result of the integral from being solved into a closed form. The same problem also exists in computing the qualification probability $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ of $o\left( {\mathrm{{Eq}}.\left( 3\right) }\right)$ with respect to a fuzzy query. In fact, the computation here is even more difficult, because the evaluation of $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ requires solving $\mathop{\Pr }\limits_{\text{range }}\left( \text{.}\right) {inthefirstplace}$

给定一个非模糊查询，计算对象 $o$ 的合格概率 $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$（公式 (1)）需要对 $o.{pdf}\left( x\right)$ 进行积分，如公式 (30) 所示，积分区域为 $o$ 的不确定区域 $o.{ur}$ 与搜索区域 ${r}_{q}$ 的交集。请记住，$o.{ur}$ 是一个 ${L}_{2}$ 圆，而 ${r}_{q}$ 可以是正方形或 ${L}_{2}$ 圆。无论如何，$o.{ur} \cap  {r}_{q}$ 可能具有不规则形状，因此无法将积分结果求解为封闭形式。在计算 $o\left( {\mathrm{{Eq}}.\left( 3\right) }\right)$ 相对于模糊查询的合格概率 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ 时，也存在同样的问题。实际上，这里的计算更加困难，因为评估 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ 需要求解 $\mathop{\Pr }\limits_{\text{range }}\left( \text{.}\right) {inthefirstplace}$

Since $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ and $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ can only be calculated numerically,the result of the calculation cannot be always guaranteed to match the theoretical value. This raises an important question: what should be accepted as a "correct result"? As the numerical process is carried out with the Monte-Carlo method (discussed in Section 5.3), a natural answer to this question is: the result obtained from a sample set with a sufficiently large size $s$ . Therefore,the first set of experiments aims at identifying the magic value $s$ .

由于 $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ 和 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ 只能通过数值方法计算，因此不能保证计算结果始终与理论值相符。这就引出了一个重要问题：什么应该被视为“正确结果”？由于数值计算过程采用蒙特卡罗方法（在第 5.3 节讨论），这个问题的一个自然答案是：从样本量足够大 $s$ 的样本集中获得的结果。因此，第一组实验旨在确定这个神奇的值 $s$。

Our methodology is as follows. First, we obtain an extremely accurate estimate of the real $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$ (or $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$ ),by using a huge sample size ${10}^{10}$ . Then,we increase $s$ gradually from a small value,and measure the error obtained from each $s$ against the accurate estimate obtained earlier. Note that, for a nonfuzzy (or fuzzy) query, the error depends on the relative positions of o.ur and ${r}_{q}$ (or $q$ .ur). Therefore,after fixing an object $o$ , we create a special workload with 1000 random queries satisfying the condition $0 < \mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right) \left( {\operatorname{or}\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right) }\right)  < 1$ (the qualification probability of $o$ should not be 0 and 1,since numerical evaluation is not needed otherwise). Then,given a particular $s$ ,the workload error is measured as the average of the absolute errors of all queries contained.

我们的方法如下。首先，我们使用巨大的样本量 ${10}^{10}$ 获得真实 $\mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right)$（或 $\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)$）的极其准确的估计值。然后，我们从一个较小的值逐渐增加 $s$，并将每个 $s$ 得到的误差与之前获得的准确估计值进行比较。请注意，对于非模糊（或模糊）查询，误差取决于 o.ur 和 ${r}_{q}$（或 $q$.ur）的相对位置。因此，在固定一个对象 $o$ 之后，我们创建一个包含 1000 个随机查询的特殊工作负载，这些查询满足条件 $0 < \mathop{\Pr }\limits_{\text{range }}\left( {o,{r}_{q}}\right) \left( {\operatorname{or}\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right) }\right)  < 1$（$o$ 的合格概率不应为 0 和 1，因为否则不需要进行数值评估）。然后，对于给定的特定 $s$，工作负载误差被测量为所有包含的查询的绝对误差的平均值。

In Figure 17(a) (or 17(b)), we demonstrate the workload error as a function of $s$ using ${L}_{\infty }$ (or ${L}_{2}$ ) workloads of nonfuzzy queries. Here,we examine two extreme combinations of parameters ${ra}{d}_{o}$ and ${ra}{d}_{q}$ : small queries on a small object $\left( {{ra}{d}_{q} = {250},{ra}{d}_{o} = 5}\right.$ ),and large queries on a large object $\left( {{ra}{d}_{q} = {750}}\right.$ , ${ra}{d}_{o} = {250}$ ). Figures 17(c) and 17(d) illustrate the results of similar experiments utilizing workloads of fuzzy queries. It turns out that the precision of Monte-Carlo is dependent solely on the sample size $s$ ,and is not affected by the data and query parameters. Recall that the rationale of Monte-Carlo stems from the sampling theory, that is, its accuracy is decided only by two factors: the fraction of samples falling into the integration region, and the function being integrated [Press et al. 2002]. These factors are identical in all the experiments

在图17(a)（或图17(b)）中，我们展示了使用非模糊查询的${L}_{\infty }$（或${L}_{2}$）工作负载时，工作负载误差随$s$的变化情况。在这里，我们研究了参数${ra}{d}_{o}$和${ra}{d}_{q}$的两种极端组合：小对象上的小查询（$\left( {{ra}{d}_{q} = {250},{ra}{d}_{o} = 5}\right.$），以及大对象上的大查询（$\left( {{ra}{d}_{q} = {750}}\right.$，${ra}{d}_{o} = {250}$）。图17(c)和图17(d)展示了使用模糊查询工作负载进行类似实验的结果。结果表明，蒙特卡罗方法的精度仅取决于样本大小$s$，而不受数据和查询参数的影响。回顾一下，蒙特卡罗方法的原理源于抽样理论，即其准确性仅由两个因素决定：落入积分区域的样本比例，以及被积函数[Press等人，2002]。在所有实验中，这些因素都是相同的

Range Search on Multidimensional Uncertain Data in Figures 17(a) and 17(b) with the same $s$ ,which explains the analogous behavior in those figures. Finally, the similarity between nonfuzzy and fuzzy queries is because the calculation of $\mathop{\Pr }\limits_{\text{fuzzy }}\left( \text{.}\right) {isreducedtto}\mathop{\Pr }\limits_{\text{range }}\left( \text{.}\right) .$

在图17(a)和图17(b)中，对多维不确定数据进行范围搜索时，$s$相同，这解释了这些图中类似的行为。最后，非模糊查询和模糊查询之间的相似性是因为$\mathop{\Pr }\limits_{\text{fuzzy }}\left( \text{.}\right) {isreducedtto}\mathop{\Pr }\limits_{\text{range }}\left( \text{.}\right) .$的计算

<!-- Media -->

<!-- figureText: 100% average absolute error 100% ${rad}_{a} = {250},{rad}_{o} = 5$ 10% ${rad}_{a} = {750},{rad}_{o} = {250}$ 1% 0.1% 0.01% ${10}^{3}$ ${10}^{4}$ ${10}^{5}$ ${10}^{6}$ sample size s (b) ${L}_{2}$ nonfuzzy workloads 100% average absolute error 10% 1% 0.1% 0.01% ${10}^{2}$ ${10}^{ \circ  }$ ${10}^{4}$ ${10}^{5}$ ${10}^{6}$ sample size s (d) ${L}_{2}$ fuzzy workloads ${rad}_{a} = {250},{rad}_{o} = 5$ 10% ${rad}_{a}^{a} = {750},{rad}_{o}^{o} = {250}$ 1% 0.1% 0.01% ${10}^{2}$ ${10}^{3}$ ${10}^{4}$ ${10}^{5}$ ${10}^{6}$ sample size $s$ (a) ${L}_{\infty }$ nonfuzzy workloads 100% average absolute error 10% ${rad}_{a} = {750},{rad}_{o} = {250}$ 1% 0.1% 0.01% ${10}^{2}$ ${10}^{4}$ ${10}^{4}$ ${10}^{5}$ ${10}^{6}$ sample size s (c) ${L}_{\infty }$ fuzzy workloads -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_36.jpg?x=529&y=336&w=772&h=689&r=0"/>

Fig. 17. Error of Monte-Carlo vs. sample size.

图17. 蒙特卡罗方法的误差与样本大小的关系。

<!-- Media -->

We will set $s$ to ${10}^{4}$ in the rest experiments,since it is the smallest sample size that leads to a reasonable workload error 1%. In other words, from now on, an approximate result derived from this value of $s$ will be claimed as correct. Accordingly,a single evaluation of $\mathop{\Pr }\limits_{\text{range }}\left( \text{.}\right) {and}\mathop{\Pr }\limits_{\text{fuzzy }}\left( \text{.}\right) {demands}\;{approximately}$ 1.7 milliseconds and 0.35 seconds, respectively.

在其余实验中，我们将把$s$设置为${10}^{4}$，因为这是导致合理工作负载误差1%的最小样本大小。换句话说，从现在起，从这个$s$值得到的近似结果将被认为是正确的。相应地，对$\mathop{\Pr }\limits_{\text{range }}\left( \text{.}\right) {and}\mathop{\Pr }\limits_{\text{fuzzy }}\left( \text{.}\right) {demands}\;{approximately}$的单次评估分别为1.7毫秒和0.35秒。

### 7.2 Tuning the U-catalog Size

### 7.2 调整U-目录大小

The size $m$ of a U-catalog has important influence on query performance. In Section 6 , we presented a method for automatic tuning of this parameter, according to the characteristics of the input dataset. In this section, we demonstrate the effectiveness of the method.

U-目录的大小$m$对查询性能有重要影响。在第6节中，我们根据输入数据集的特征，提出了一种自动调整该参数的方法。在本节中，我们将展示该方法的有效性。

7.2.1 Cost Model Accuracy. A fundamental component of our tuning approach is a cost model that predicts the overhead of nonfuzzy range search. To verify the accuracy of the model,we use an $m = 3$ catalog (the values in the catalog are decided as elaborated in Section 6.3),employ the dataset ${RAN}$ - 100 (i.e.,generated from the point set ${RAN}$ with ${\operatorname{rad}}_{o} = {100}$ ),and compare the estimated query cost (from our model) against the actual cost. In particular, the comparison includes two aspects: (i) the I/O overhead, which is the cost of the filter step, and proportional to the number of leaf node accesses in the U-tree, and (ii) the CPU time, which is dominated by the overhead of the refinement phase, and proportional to the number of numerical evaluations of $P{r}_{\text{range }}\left( \text{.}\right) .$

7.2.1 成本模型的准确性。我们调整方法的一个基本组成部分是一个成本模型，用于预测非模糊范围搜索的开销。为了验证该模型的准确性，我们使用一个$m = 3$目录（目录中的值如第6.3节所述确定），采用数据集${RAN}$ - 100（即从点集${RAN}$中生成，其中${\operatorname{rad}}_{o} = {100}$），并将（从我们的模型得到的）估计查询成本与实际成本进行比较。具体来说，比较包括两个方面：(i) I/O开销，即过滤步骤的成本，与U-树中叶子节点的访问次数成正比；(ii) CPU时间，主要由细化阶段的开销决定，与$P{r}_{\text{range }}\left( \text{.}\right) .$的数值评估次数成正比

<!-- Media -->

<!-- figureText: 50 160 140 estimated actual ${zzzzzz}$ 120 80 60 40 20 250 500 750 search region radius ${\text{rad}}_{q}$ (b) CPU cost vs. ${\operatorname{rad}}_{o}\left( {t}_{q}\right.$ uniform in $\left\lbrack  {{0.1},{0.9}}\right\rbrack$ ) 350 number of numerical evaluation: 300 estimated ... actual 250 200 150 ${t}_{a} = {0.167}$ ${t}_{a} = {0.833}$ 100 50 0 probability threshold ${t}_{q}$ (d) CPU cost vs. ${t}_{q}\left( {{ra}{d}_{q} = {500}}\right)$ estimated 40 actual E I VIII III 30 20 10 250 500 750 search region radius ${\mathrm{{rad}}}_{q}$ (a) I/O cost vs. ${\operatorname{rad}}_{o}\left( {t}_{q}\right.$ uniform in $\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$ 35 number of node accesses estimated actual 25 20 15 0.1 probability threshold ${t}_{q}$ (c) I/O cost vs. ${t}_{q}\left( {{ra}{d}_{q} = {500}}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_37.jpg?x=403&y=329&w=998&h=695&r=0"/>

Fig. 18. Accuracy of the proposed cost model (dataset: ${RAN} - {100};{L}_{\infty }$ nonfuzzy workloads).

图18. 所提出的成本模型的准确性（数据集：${RAN} - {100};{L}_{\infty }$非模糊工作负载）。

<!-- Media -->

Figures 18(a) and 18(b) plot the I/O and CPU comparison as a function of the radius ${\operatorname{rad}}_{q}$ of search regions,respectively. Here,at each value $x$ of ${\operatorname{rad}}_{q}$ ,the actual I/O (or CPU) cost corresponds to the average number of node accesses (or numerical evaluations) in executing a query in a nonfuzzy ${L}_{\infty }$ workload with parameters ${\operatorname{rad}}_{q} = x$ and ${t}_{q} =$ mixed. Given a query,we estimate its I/O and CPU overhead using Eqs. (25) and (22), respectively. Figures 18(c) and 18(d) illustrate the comparison as ${t}_{q}$ varies from 0.1 to 0.9 . The actual and estimated results are obtained in the same manner as explained earlier, except that the workloads have a ${t}_{q}$ equal to the value being tested,and an ${ra}{d}_{q}$ fixed to 500 .

图18(a)和图18(b)分别绘制了输入/输出（I/O）和中央处理器（CPU）开销随搜索区域半径${\operatorname{rad}}_{q}$变化的对比图。在此，对于${\operatorname{rad}}_{q}$的每个值$x$，实际的I/O（或CPU）开销对应于在参数为${\operatorname{rad}}_{q} = x$和${t}_{q} =$混合的非模糊${L}_{\infty }$工作负载中执行查询时节点访问（或数值计算）的平均次数。给定一个查询，我们分别使用公式(25)和(22)来估算其I/O和CPU开销。图18(c)和图18(d)展示了${t}_{q}$从0.1变化到0.9时的对比情况。实际结果和估算结果的获取方式与前面所述相同，不同之处在于工作负载的${t}_{q}$等于测试值，且${ra}{d}_{q}$固定为500。

It is clear that the proposed model is highly accurate, incurring an error less than $5\%$ in all cases. In particular,our analytical formulae capture exactly the changes of query cost. First of all, there should be no surprise in witnessing the cost increase continuously with ${\mathrm{{rad}}}_{q}$ (see Figures 18(a) and 18(b)): a larger search region leads to more qualifying objects, thus entailing more expensive I/O and CPU overhead. The change behavior with respect to ${t}_{q}$ is more complex. As this parameter grows, the I/O cost decreases monotonically, while the number of numerical evaluations initially decreases until ${t}_{q}$ reaches 0.167,stays low for a wide range of ${t}_{q}$ ,and then bounces up when ${t}_{q}$ becomes $1 - {0.167} = {0.833}$ .

显然，所提出的模型具有很高的准确性，在所有情况下产生的误差均小于$5\%$。特别是，我们的解析公式准确地捕捉到了查询开销的变化。首先，看到开销随${\mathrm{{rad}}}_{q}$持续增加并不奇怪（见图18(a)和图18(b)）：更大的搜索区域会产生更多符合条件的对象，从而导致更高的I/O和CPU开销。关于${t}_{q}$的变化行为更为复杂。随着该参数的增大，I/O开销单调递减，而数值计算的次数最初会减少，直到${t}_{q}$达到0.167，在${t}_{q}$的较大取值范围内保持较低水平，然后当${t}_{q}$变为$1 - {0.167} = {0.833}$时又会上升。

To understand the above " ${t}_{q}$ -phenomenon", recall that this experiment is based on a U-catalog with three values: 0,0.167,and 0.334 . Let $o$ be an object whose uncertainty region partially intersects the search region (the other objects can always be pruned/validated, and hence, are irrelevant to explaining the phenomenon). The power of Theorem 3 in pruning $o$ is increasingly stronger as ${t}_{q}$ grows (see the footnote ${}^{5}$ ),which is the reasoning behind Figure 18(c). As for Figure 17d,we point out that,for ${t}_{q} \in  \lbrack 0,{0.167})$ ,it is not possible to prune $o$ with Theorem 3,although $o$ may be validated using Theorem 2. The opposite is true for ${t}_{q} \in  ({0.833},1\rbrack$ ,that is, $o$ may be pruned,but it can never be validated. Our heuristics are most effective when ${t}_{q}$ distributes in [0.167,0.833]. In this case,both pruning and validation of $o$ are likely; therefore,the least number of numerical evaluations is needed.

为了理解上述“${t}_{q}$现象”，请回想一下，该实验基于一个包含三个值（0、0.167和0.334）的不确定性目录（U - catalog）。设$o$为一个其不确定区域与搜索区域部分相交的对象（其他对象总是可以被剪枝/验证，因此与解释该现象无关）。随着${t}_{q}$的增大，定理3对$o$进行剪枝的能力越来越强（见脚注${}^{5}$），这就是图18(c)背后的原因。至于图18(d)，我们指出，对于${t}_{q} \in  \lbrack 0,{0.167})$，不可能用定理3对$o$进行剪枝，尽管可以用定理2对$o$进行验证。对于${t}_{q} \in  ({0.833},1\rbrack$则相反，即$o$可以被剪枝，但永远无法被验证。当${t}_{q}$分布在[0.167, 0.833]时，我们的启发式方法最为有效。在这种情况下，$o$的剪枝和验证都有可能；因此，所需的数值计算次数最少。

---

<!-- Footnote -->

${}^{5}$ Specifically,for ${t}_{q} \in  \lbrack 0,{0.167}),o$ can be never be pruned. For ${t}_{q} \in  \lbrack {0.167},{0.334}),o$ can be pruned

${}^{5}$具体而言，对于${t}_{q} \in  \lbrack 0,{0.167}),o$永远无法进行剪枝。对于${t}_{q} \in  \lbrack {0.167},{0.334}),o$可以进行剪枝

<!-- Footnote -->

---

7.2.2 Effects of the U-catalog Size. We are ready to inspect the effectiveness of the proposed method for tuning the U-catalog size $m$ . For this purpose, we employ only the uncertain databases generated from the real datasets ${CA}$ and ${LB}$ . As discussed in Section 6.2.2,when the data distribution is irregular,our tuning solution applies the local smoothing technique based on a $\lambda  \times  \lambda$ histogram; in the sequel, $\lambda$ is fixed to 5 .

7.2.2 不确定性目录（U - catalog）大小的影响。我们准备考察所提出的调整不确定性目录大小$m$的方法的有效性。为此，我们仅使用从真实数据集${CA}$和${LB}$生成的不确定数据库。如6.2.2节所述，当数据分布不规则时，我们的调整解决方案采用基于$\lambda  \times  \lambda$直方图的局部平滑技术；随后，$\lambda$固定为5。

We aim at minimizing the expected overall overhead (i.e., including both I/O and CPU time) of a nonfuzzy query whose ${ra}{d}_{q}$ equals the median value 500,and its ${t}_{q}$ follows a uniform distribution in [0.1,0.9]. To achieve this goal, given a particular $m$ ,we utilize our cost model to estimate the expected overhead of queries with eighty-one $\left\{  {{ra}{d}_{q},{t}_{q}}\right\}$ combinations,respectively: $\{ {500},{0.1}\}$ , $\{ {500},{0.11}\} ,\ldots ,\{ {500},{0.89}\} ,\{ {500},{0.9}\}$ . Each estimate is the sum of ${\operatorname{cost}}_{\text{frn }}$ and ${\text{cost}}_{flt}$ computed from Eqs. (26) and (24),respectively,setting ${\operatorname{cost}}_{\text{ranIO }}$ to 20 milliseconds,and ${\operatorname{cost}}_{rfn}$ to 1.7 milliseconds (according to the experiments in Section 7.1). The penalty of $m$ is the average of the estimates of all the combinations. The best $m$ is the one with the lowest penalty.

我们的目标是最小化一个非模糊查询的预期总开销（即包括I/O时间和CPU时间），该查询的${ra}{d}_{q}$等于中值500，并且其${t}_{q}$在[0.1, 0.9]内服从均匀分布。为实现这一目标，给定一个特定的$m$，我们利用我们的成本模型分别估计具有八十一种$\left\{  {{ra}{d}_{q},{t}_{q}}\right\}$组合的查询的预期开销：$\{ {500},{0.1}\}$，$\{ {500},{0.11}\} ,\ldots ,\{ {500},{0.89}\} ,\{ {500},{0.9}\}$。每个估计值分别是根据公式(26)和(24)计算出的${\operatorname{cost}}_{\text{frn }}$和${\text{cost}}_{flt}$之和，将${\operatorname{cost}}_{\text{ranIO }}$设为20毫秒，将${\operatorname{cost}}_{rfn}$设为1.7毫秒（根据7.1节中的实验）。$m$的惩罚值是所有组合估计值的平均值。最佳的$m$是惩罚值最低的那个。

In the experiment of Figure 19(a), we select the uncertain dataset CA-5 (where the uncertainty region of each object is a circle with radius ${\operatorname{rad}}_{o} = 5$ ). The curve labeled as "actual" presents the average query cost in a nonfuzzy ${L}_{\infty }$ workload with ${\operatorname{rad}}_{q} = {500}$ and ${t}_{q} =$ mixed,when the catalog size $m$ varies from 1 to 10 . The curve "estimated" shows the penalties of $m$ . Figures 19(b) through 19(f) demonstrate similar results for datasets ${LB} - 5,{CA} - {100},{LB} - {100},{CA} - {250}$ , and ${LB}$ -250, respectively.

在图19(a)的实验中，我们选择不确定数据集CA - 5（其中每个对象的不确定区域是半径为${\operatorname{rad}}_{o} = 5$的圆）。标记为“实际”的曲线展示了当目录大小$m$从1变化到10时，在包含${\operatorname{rad}}_{q} = {500}$和${t}_{q} =$混合的非模糊${L}_{\infty }$工作负载中的平均查询成本。“估计”曲线显示了$m$的惩罚值。图19(b)至图19(f)分别展示了数据集${LB} - 5,{CA} - {100},{LB} - {100},{CA} - {250}$和${LB}$ - 250的类似结果。

In every figure, the two curves are very close to each other, which proves that our performance analysis is effective also for irregular data distributions. Furthermore,the optimal U-catalog size $m$ is clearly related to the data characteristics. In particular, when objects have very small uncertainty regions (as in ${CA} - 5$ and ${LB} - 5$ ),the best $m$ equals 1,that is,only a single PCR (i.e.,the MBR o.pcr(0)) of each object $o$ should be indexed. This is reasonable because,if the query region ${r}_{q}$ is much larger than o.ur,the chance of ${r}_{q}$ partially intersecting o. $\operatorname{pcr}\left( 0\right)$ is very low,meaning that $o$ can already be pruned or validated with a very high probability even if no other PCR is available.

在每个图中，两条曲线非常接近，这证明我们的性能分析对于不规则数据分布也是有效的。此外，最优的U - 目录大小$m$显然与数据特征相关。特别是，当对象的不确定区域非常小（如在${CA} - 5$和${LB} - 5$中）时，最佳的$m$等于1，即每个对象$o$仅应索引单个PCR（即MBR o.pcr(0)）。这是合理的，因为如果查询区域${r}_{q}$远大于o.ur，则${r}_{q}$与o. $\operatorname{pcr}\left( 0\right)$部分相交的可能性非常低，这意味着即使没有其他PCR可用，$o$也已经有非常高的概率被剪枝或验证。

As shown in Figures 19(c) through 19(d), when objects have sizable uncertainty regions,the optimal $m$ tends to increase with ${ra}{d}_{o}$ . For each dataset, before $m$ reaches its optimal value,enlarging the U-catalog brings more PCRs to each object, strengthens the pruning and validating power of our heuristics, and reduces the query cost. After $m$ passes the optimum,however,further increasing it no longer enhances the effectiveness of the heuristics significantly, but necessitates more I/Os (due to the decrease of node fanout), thus compromising query performance. Our tuning method captures such behavior precisely, and always identifies the optimal catalog size.

如图19(c)至图19(d)所示，当对象具有相当大的不确定区域时，最优的$m$倾向于随${ra}{d}_{o}$增加。对于每个数据集，在$m$达到其最优值之前，增大U - 目录会为每个对象带来更多的PCR，增强我们启发式算法的剪枝和验证能力，并降低查询成本。然而，在$m$超过最优值之后，进一步增大它不再显著提高启发式算法的有效性，但会需要更多的I/O操作（由于节点扇出的减少），从而影响查询性能。我们的调优方法能够精确捕捉这种行为，并始终能确定最优的目录大小。

---

<!-- Footnote -->

if the query region ${r}_{q}$ is disjoint with o.pcr(0.167),while for ${t}_{q} \in  \lbrack {0.334},{0.666})$ ,pruning can be performed if ${r}_{q}$ does not intersect $o.{pcr}\left( {0.334}\right)$ ,which is smaller than $o.{pcr}\left( {0.167}\right)$ . For ${t}_{q} \in  \lbrack {0.666},{0.833})$ (or [0.666,0.833]),we can prune $o$ if ${r}_{q}$ does not contain $o.{pcr}\left( {0.334}\right)$ (or $o.{pcr}\left( {0.167}\right)$ ).

如果查询区域${r}_{q}$与o.pcr(0.167)不相交，而对于${t}_{q} \in  \lbrack {0.334},{0.666})$，若${r}_{q}$与$o.{pcr}\left( {0.334}\right)$不相交，则可以进行剪枝，其中$o.{pcr}\left( {0.334}\right)$小于$o.{pcr}\left( {0.167}\right)$。对于${t}_{q} \in  \lbrack {0.666},{0.833})$（或[0.666,0.833]），若${r}_{q}$不包含$o.{pcr}\left( {0.334}\right)$（或$o.{pcr}\left( {0.167}\right)$），则可以对$o$进行剪枝。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 0.9 0.8 0.7 estimated 0.6 0.4 0.3 0.2 0.1 0 1 2 3 6 8 10 catalog size $m$ (b) Query cost vs. $m\left( {{LB} - 5}\right)$ query time (sec) estimated 0.8 0.6 0.4 0.2 catalog size $m$ d) Query cost vs. $m\left( {{LB} - {100}}\right)$ query time (sec) estimated actual 1.2 0.4 0.2 2 6 9 10 catalog size $m$ (f) Query cost vs. $m\left( {{LB} - {250}}\right)$ 0.8 estimated 0.7 actual 0.6 0.5 0.4 0.3 0.2 0.1 0 1 2 3 4 6 8 10 catalog size $m$ (a) Query cost vs. $m\left( {{CA} - 5}\right)$ query time (sec) estimated 0.8 0.6 0.4 0.2 0 7 10 catalog size $m$ c) Query cost vs. $m\left( {{CA} - {100}}\right)$ query time (sec) 2 actual 1.5 0.5 2 9 10 catalog size $m$ (e) Query cost vs. $m\left( {{CA} - {250}}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_39.jpg?x=512&y=337&w=782&h=1028&r=0"/>

Fig. 19. Effectiveness of our method for U-catalog size tuning (ra ${d}_{q} = {500},{t}_{q}$ uniform in [0.1, 0.9]).

图19. 我们的方法在U目录大小调整方面的有效性（ra ${d}_{q} = {500},{t}_{q}$在[0.1, 0.9]内均匀分布）。

<!-- Media -->

### 7.3 Cost of Nonfuzzy Range Search

### 7.3 非模糊范围搜索的成本

Range search on multidimensional uncertain data is a novel topic that has not been previously studied. Since there does not exist a nontrivial competitor, next we compare the U-tree against the R-tree in nonfuzzy retrieval. Specifically, an R-tree refers to a special U-tree with a U-catalog size $m = 1$ . We will focus on two uncertain datasets: ${CA} - {100}$ and ${LB} - {100}$ ,for both of which the best U-catalog size equals 3 , as shown in Figure 19. A memory cache is introduced to buffer all the intermediate levels of an index.

多维不确定数据的范围搜索是一个此前未被研究过的新课题。由于不存在非平凡的竞争对手，接下来我们将在非模糊检索方面将U树与R树进行比较。具体而言，R树是指U目录大小为$m = 1$的特殊U树。我们将重点关注两个不确定数据集：${CA} - {100}$和${LB} - {100}$，如图19所示，这两个数据集的最佳U目录大小均为3。引入了一个内存缓存来缓冲索引的所有中间层。

Range Search on Multidimensional Uncertain Data Article 15 / 41

多维不确定数据的范围搜索 文章15 / 41

<!-- Media -->

<!-- figureText: $U$ -tree filter $U$ -tree refinement query time (sec) 1.2 0.8 0.6 0.4 0.2 250 500 750 search region radius rad ${}_{q}$ (b) Cost vs. ${\operatorname{rad}}_{q}\left( {{LB} - {100},{t}_{q}\text{uniform in}\left\lbrack  {{0.1},{0.9}}\right\rbrack  }\right)$ 300 number of false hits 250 200 150 100 50 500 750 search region radius rad ${}_{q}$ (d) Number of false hits in (b) query time (sec) $R$ -tree U-tree 0.8 0.6 0.4 0.2 0 0.1 0.7 0.9 probability threshold ${t}_{q}$ (f) Cost vs. ${t}_{q}\left( {{LB} - {100},{ra}{d}_{q} = {500}}\right)$ 1.4 query time (sec) 1.2 0.8 0.6 0.4 0.2 0 250 500 750 search region radius ${\mathrm{{rad}}}_{q}$ (a) Cost vs. ${\operatorname{rad}}_{q}\left( {{CA} - {100},{t}_{q}\text{uniform in}\left\lbrack  {{0.1},{0.9}}\right\rbrack  }\right)$ 300 number of false hits 250 $U$ -tree 200 150 100 50 250 500 750 search region radius rad ${}_{q}$ (c) Number of false hits in (a) 1.4 query time (sec) 1.2 $R$ -tree $U$ -tree 1 0.8 0.6 0.4 0.2 0 0.9 probability threshold ${t}_{q}$ (e) Cost vs. ${t}_{q}\left( {{CA} - {100},{ra}{d}_{q} = {500}}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_40.jpg?x=367&y=320&w=1058&h=1122&r=0"/>

Fig. 20. Nonfuzzy query cost comparison between R- and U-trees $\left( {L}_{\infty }\right.$ -workloads).

图20. R树和U树在$\left( {L}_{\infty }\right.$工作负载下的非模糊查询成本比较。

<!-- Media -->

Focusing on ${CA}$ -100,the experiment of Figure 20(a) uses three nonfuzzy ${L}_{\infty }$ workloads with ${t}_{q} =$ mixed,and ${ra}{d}_{q} = {250},{500}$ ,and 750,respectively. For each workload, we measure the average overhead of processing a query with $\mathrm{R}$ - and $\mathrm{U}$ -trees,respectively. At each value of ${\operatorname{rad}}_{q}$ ,the result of a method is further broken into two parts, representing the cost of its filter and refinement steps,respectively. Figure 20(b) shows the comparison for ${LB}$ -100 under the same settings.

针对${CA}$ - 100，图20(a)的实验使用了三种非模糊${L}_{\infty }$工作负载，分别混合了${t}_{q} =$、${ra}{d}_{q} = {250},{500}$和750。对于每个工作负载，我们分别测量了使用$\mathrm{R}$树和$\mathrm{U}$树处理查询的平均开销。在${\operatorname{rad}}_{q}$的每个取值下，一种方法的结果进一步分为两部分，分别代表其过滤步骤和细化步骤的成本。图20(b)展示了在相同设置下${LB}$ - 100的比较情况。

The U-tree outperforms its rival in all cases, achieving a maximum speedup of 3 . In particular, the U-tree entails higher filter cost, but is significantly faster in the refinement phase. This is expected because (i) by retaining several PCRs per object, the U-tree has a lower fanout, and hence, a larger number of nodes, rendering more node accesses in filtering; (ii) the U-tree needs to refine much fewer objects.

在所有情况下，U树的性能都优于其竞争对手，实现了最高3倍的加速。具体而言，U树的过滤成本较高，但在细化阶段明显更快。这是可以预料的，因为（i）通过为每个对象保留多个概率约束矩形（PCR，Probabilistic Constraint Rectangle），U树的扇出较低，因此节点数量较多，在过滤时需要访问更多节点；（ii）U树需要细化的对象要少得多。

An object is a false hit, if its precise qualification probability is calculated, but it does not qualify the query. False hits should be avoided as much as possible to ensure fast response time. Figures 20(c) and 20(d) demonstrate the average number of false hits per query in the experiments of Figures 20(a) and 20(b), respectively. Obviously, the U-tree incurs significantly fewer false hits.

如果计算了一个对象的精确合格概率，但该对象不符合查询条件，则该对象为误命中。应尽可能避免误命中，以确保快速响应时间。图20(c)和图20(d)分别展示了图20(a)和图20(b)实验中每个查询的平均误命中数量。显然，U树产生的误命中明显更少。

Figure 20(e) and 20(f)) plots the query time of R- and U-trees as a function of ${t}_{q}$ ,using nonfuzzy ${L}_{\infty }$ workloads with ${\operatorname{rad}}_{q} = {500}$ ,and ${t}_{q} = {0.1},\ldots ,{0.9}$ , respectively. Again, the U-tree is the clear winner, and its behavior is similar to that illustrated in Figure 18(d). The performance of the R-tree is not affected by ${t}_{q}$ ,since keeping only the objects’ MBRs offers equivalent pruning/validating power for all ${t}_{q}$ .

图20(e)和图20(f)绘制了R树和U树的查询时间随${t}_{q}$变化的函数关系，分别使用了包含${\operatorname{rad}}_{q} = {500}$和${t}_{q} = {0.1},\ldots ,{0.9}$的非模糊${L}_{\infty }$工作负载。同样，U树明显胜出，其表现与图18(d)所示类似。R树的性能不受${t}_{q}$的影响，因为仅保留对象的最小边界矩形（MBR，Minimum Bounding Rectangle）对所有${t}_{q}$都提供了等效的剪枝/验证能力。

Figure 21 presents the results of the same experiments in Figure 20 but with respect to ${L}_{2}$ workloads. These results confirm the phenomena observed from ${L}_{\infty }$ workloads,except that U-trees achieve a lower performance speedup over R-trees. To explain this,recall that we process an ${L}_{2}$ query,by conservatively bounding its search region ${r}_{q}$ using an outside rectangle $r$ and an inside rectangle ${r}^{\prime }$ respectively,as illustrated in Figure 6(b). The conservative approach reduces the pruning/validating power of our heuristics. Although both U- and R-trees are affected, the effects on U-trees are more significant, since only limited pruning/validating is possible for R-trees in any case.

图21展示了与图20相同实验的结果，但针对的是${L}_{2}$工作负载。这些结果证实了从${L}_{\infty }$工作负载中观察到的现象，只是U树相对于R树实现的性能加速较低。为了解释这一点，请回想一下，我们处理${L}_{2}$查询时，分别使用外部矩形$r$和内部矩形${r}^{\prime }$保守地界定其搜索区域${r}_{q}$，如图6(b)所示。这种保守方法降低了我们启发式算法的剪枝/验证能力。虽然U树和R树都会受到影响，但对U树的影响更为显著，因为无论如何，R树的剪枝/验证能力都有限。

### 7.4 Cost of Fuzzy Search

### 7.4 模糊搜索的成本

Now we continue to evaluate the algorithm in Section 4 for fuzzy range queries, also using the datasets ${CA} - {100}$ and ${LB} - {100}$ . As mentioned in Section 4.2,the algorithm requires a parameter ${m}_{q}$ ,which is the number of PCRs computed for the query object $q$ . Hence,we first select an appropriate value for this parameter. For this purpose,given a particular ${m}_{q}$ ,we measure the average time of processing a query with a U-tree in a fuzzy workload with ${\varepsilon }_{q} = {500}$ and ${t}_{q} =$ mixed. Figure 22(a) and (22(b)) plots the average cost as a function of ${m}_{q}$ for both ${CA} - {100}$ and ${LB} - {100}$ ,using ${L}_{\infty }\left( {L}_{2}\right)$ workloads. Clearly,an excessively small ${m}_{q}$ results in expensive query overhead,because in this case only a limited amount of information about the query is available for pruning and validating. On the other hand,once ${m}_{q}$ reaches 10,further increasing this parameter does not lead to significant improvement,indicating that 10 PCRs of $q$ is already sufficient for efficient processing. In the rest experiments,we fix ${m}_{q}$ to 10 .

现在我们继续评估第4节中针对模糊范围查询的算法，同样使用数据集${CA} - {100}$和${LB} - {100}$。如第4.2节所述，该算法需要一个参数${m}_{q}$，它是为查询对象$q$计算的PCR（部分匹配区域，Partial Completion Region）数量。因此，我们首先为该参数选择一个合适的值。为此，给定一个特定的${m}_{q}$，我们测量在${\varepsilon }_{q} = {500}$和${t}_{q} =$混合的模糊工作负载中使用U树处理查询的平均时间。图22(a)和图22(b)绘制了在${L}_{\infty }\left( {L}_{2}\right)$工作负载下，${CA} - {100}$和${LB} - {100}$的平均成本随${m}_{q}$变化的函数关系。显然，${m}_{q}$过小会导致查询开销过大，因为在这种情况下，只有有限的查询信息可用于剪枝和验证。另一方面，一旦${m}_{q}$达到10，进一步增加该参数并不会带来显著改善，这表明$q$的10个PCR已经足以进行高效处理。在其余实验中，我们将${m}_{q}$固定为10。

Next, we compare the query performance of R- and U-trees, by repeating the experiments of Figures 20 and 21 with respect to fuzzy workloads. The results are presented in Figures 23 and 24. Each column can be interpreted as either the overall query time or refinement overhead (averaged over all the queries in a workload). In fuzzy search, the filter step cost is negligible compared to the overall time, and is demonstrated on top of each column.

接下来，我们通过重复图20和图21针对模糊工作负载的实验，比较R树和U树的查询性能。结果如图23和图24所示。每一列可以解释为总体查询时间或细化开销（对工作负载中的所有查询求平均值）。在模糊搜索中，过滤步骤的成本与总体时间相比可以忽略不计，并在每列的顶部展示。

The U-tree is again the better solution in all experiments, having a maximum speedup of 5 over the R-tree (in Figure 23(c)). As in Figures 23 and 24, the cost changes of U-trees with respect to ${t}_{q}$ are much smoother than those in nonfuzzy search (see Figures 20 and 21). Namely,the " ${t}_{q}$ -phenomenon",defined

在所有实验中，U树再次是更好的解决方案，相对于R树的最大加速比为5（如图23(c)所示）。如图23和图24所示，U树相对于${t}_{q}$的成本变化比非模糊搜索中的变化（见图20和图21）要平缓得多。即，在第7.2.1节中定义的“${t}_{q}$现象”

Range Search on Multidimensional Uncertain Data Article 15 / 43 in Section 7.2.1, disappears. To understand this, recall that, as elaborated in Section 7.2.1, the condition of the phenomenon is that, for a nonfuzzy query $q$ ,only one PCR (selected according to ${t}_{q}$ ) is used for pruning/validating an object. The condition no longer holds: given a fuzzy query, multiple PCRs may be utilized by our pruning/validating approach in Section 4.

多维不确定数据上的范围搜索 文章15 / 43消失了。要理解这一点，请回想一下，如第7.2.1节所述，该现象的条件是，对于非模糊查询$q$，仅使用一个PCR（根据${t}_{q}$选择）来对对象进行剪枝/验证。该条件不再成立：对于模糊查询，我们在第4节中的剪枝/验证方法可能会使用多个PCR。

<!-- Media -->

<!-- figureText: $U$ -tree filter $U$ -tree refinement query time (sec) 1.8 1.6 1.4 1.2 0.8 0.6 0.4 0.2 0 250 500 750 search region radius rad ${}_{q}$ (b) Cost vs. ${\operatorname{rad}}_{q}\left( {{LB} - {100},{t}_{q}}\right.$ uniform in $\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$ 500 number of false hits $R$ -tree $- \Delta  -$ 300 200 100 250 500 750 search region radius rad ${}_{q}$ (d) Number of false hits in (b) query time (sec) $R$ -tree $U$ -tree 1.2 0.8 0.6 0.4 0.2 0 0.2 0.3 0.5 0.6 0.7 0.8 0.9 probability threshold ${t}_{d}$ (f) Cost vs. ${t}_{q}\left( {{LB} - {100},{ra}{d}_{q} = {500}}\right)$ query time (sec) 2 1.5 1 0.5 0 250 500 750 search region radius ${\text{rad}}_{q}$ (a) Cost vs. ${\operatorname{rad}}_{q}\left( {{CA} - {100},{t}_{q}}\right.$ uniform in $\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$ 600 number of false hits $R$ -tree $- \Delta  -$ 500 400 300 200 100 250 500 750 search region radius rad, (c) Number of false hits in (a) query time (sec) 1.6 $R$ -tree 1.4 $U$ -tree 1.2 0.8 0.6 0.4 0.2 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 probability threshold ${t}_{q}$ (e) Cost vs. ${t}_{q}\left( {{CA} - {100},{ra}{d}_{q} = {500}}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_42.jpg?x=396&y=332&w=1047&h=1041&r=0"/>

Fig. 21. Nonfuzzy query cost comparison between R- and U-trees $\left( {L}_{2}\right.$ -workloads).

图21. R树和U树在$\left( {L}_{2}\right.$工作负载下的非模糊查询成本比较。

<!-- Media -->

Finally, as in the nonfuzzy scenario, the performance superiority of U-trees over $\mathrm{R}$ -trees is more obvious in ${L}_{\infty }$ queries (than ${L}_{2}$ ). Similar to the reasons given in Section 7.3, this is due to the conservative approximation deployed in processing an ${L}_{2}$ query,except that here the approximation is illustrated in Figures 9(c) and 9(d).

最后，与非模糊场景一样，U树相对于$\mathrm{R}$树在${L}_{\infty }$查询（而非${L}_{2}$查询）中的性能优势更为明显。与第7.3节给出的原因类似，这是由于处理${L}_{2}$查询时采用了保守近似，只是这里的近似情况如图9(c)和图9(d)所示。

### 7.5 Index Construction Overhead

### 7.5 索引构建开销

We proceed to evaluate the efficiency of the U-tree's construction algorithm. In Figure 25(a) and (25(b)), we demonstrate the cost of building a U- and an R-tree on dataset ${CA} - {100}\left( {{LB} - {100}}\right)$ ,by incrementally inserting all the objects.

我们接下来评估U树构建算法的效率。在图25(a)和图25(b)中，我们展示了通过逐个插入所有对象，在数据集${CA} - {100}\left( {{LB} - {100}}\right)$上构建U树和R树的成本。

<!-- Media -->

<!-- figureText: 250 query time (sec) 500 query time (sec) ${LB}$ 400 ${CA}$ 300 \\ △ △ △ △ △ △ △ △ △ △ △ 200 100 90100 ${m}_{q}$ (b) U-tree query cost vs. ${m}_{q}\left( {L}_{2}\right.$ workloads) ${LB}$ 200 ${CA}$ 150 100 50 2022 40 50 ( 70 90100 ${m}_{q}$ (a) U-tree query cost vs. ${m}_{q}\left( {L}_{\infty }\right.$ workloads) -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_43.jpg?x=431&y=388&w=948&h=345&r=0"/>

Fig. 22. Tuning ${m}_{q}$ for fuzzy retrieval $\left( {{\varepsilon }_{q} = {500},{t}_{q}}\right.$ uniform in $\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$ .

图22. 为模糊检索调整${m}_{q}$，$\left( {{\varepsilon }_{q} = {500},{t}_{q}}\right.$在$\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$上均匀分布。

<!-- figureText: $R$ -tree $U$ -tree query time (sec) 300 filter step cost 0.36 250 200 150 100 0.67 50 0.11 0.32 0 250 500 750 distance threshold ${\varepsilon }_{q}$ (b) Cost vs. ${\varepsilon }_{q}\left( {{LB} - {100},{t}_{q}}\right.$ uniform in $\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$ 800 number of false hits 700 $R$ -tree -- $\Delta$ -- 600 500 400 300 200 100 250 500 750 distance threshold ${\varepsilon }_{a}$ (d) Number of false hits in (b) 300 query time (sec, 250 $U$ -tree 200 150 100 50 0.8 probability threshold ${t}_{q}$ (f) Cost vs. ${t}_{q}\left( {{LB} - {100},{\varepsilon }_{q} = {500}}\right)$ query time (sec) 350 filter step cost 0.40 300 250 0.21 200 150 100 0.09 0.77 50 0.37 0 250 500 750 distance threshold ${\varepsilon }_{q}$ (a) Cost vs. ${\varepsilon }_{q}\left( {{CA} - {100},{t}_{q}}\right.$ uniform in $\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$ 1000 $R$ -tree $\sim  \Delta  \sim  \Delta$ 800 $U$ -tree 600 400 200 250 500 750 distance threshold ${\varepsilon }_{a}$ (c) Number of false hits in (a) 350 query time (sec) 300 $U$ -tree 200 150 100 50 0.4 0.7 0.8 probability threshold ${t}_{a}$ (3) Cost vs. ${t}_{q}\left( {{CA} - {100},{\varepsilon }_{q} = {500}}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_43.jpg?x=407&y=880&w=990&h=1020&r=0"/>

Fig. 23. Fuzzy query cost comparison between R- and U-trees $\left( {L}_{\infty }\right.$ -workloads).

图23. R树和U树的模糊查询成本比较（$\left( {L}_{\infty }\right.$工作负载）。

<!-- Media -->

Range Search on Multidimensional Uncertain Data Article 15 / 45 In particular, the U-tree result consists of three components, capturing respectively the overhead of (i) optimizing the U-catalog size, (ii) preparing the PCRs of each object, and (iii) incremental insertion. The U-tree catalog contains 3 values.

多维不确定数据上的范围搜索 文章15 / 45 特别地，U树的结果由三个部分组成，分别反映了以下方面的开销：(i) 优化U目录大小；(ii) 准备每个对象的可能区域（PCR）；(iii) 增量插入。U树目录包含3个值。

<!-- Media -->

<!-- figureText: $R$ -tree $U$ -tree query time (sec) 700 600 500 400 300 200 100 0 250 500 750 distance threshold ${\varepsilon }_{q}$ (b) Cost vs. ${\varepsilon }_{q}\left( {{LB} - {100},{t}_{q}}\right.$ uniform in $\left\lbrack  {{0.1},{0.9}}\right\rbrack$ ) number of false hits 1200 $R$ -tree $- \Delta$ 1000 800 600 400 200 250 500 750 distance threshold ${\varepsilon }_{a}$ (d) Number of false hits in (b) query time (sec) 500 $R$ -tree 400 $U$ -tree 300 200 100 0 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 probability threshold ${t}_{q}$ (f) Cost vs. ${t}_{q}\left( {{LB} - {100},{\varepsilon }_{q} = {500}}\right)$ query time (sec) 900 800 filter step cost 0.36 0.77 600 500 400 0.40 300 0.11 200 100 0.21 250 500 750 distance threshold ${\varepsilon }_{q}$ (a) Cost vs. ${\varepsilon }_{q}\left( {{CA} - {100},{t}_{q}}\right.$ uniform in $\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$ number of false hits 1600 1400 $R$ -tree $- \Delta  -$ 1200 1000 800 600 400 250 500 750 distance threshold ${\varepsilon }_{a}$ (c) Number of false hits in (a) query time (sec) 600 R-tree 500 $U$ -tree 400 300 200 100 0.2 0.4 0.5 0.6 0.7 0.8 0.9 probability threshold ${t}_{q}$ (e) Cost vs. ${t}_{q}\left( {{CA} - {100},{\varepsilon }_{q} = {500}}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_44.jpg?x=365&y=326&w=1098&h=1144&r=0"/>

Fig. 24. Fuzzy query cost comparison between R- and U-trees $\left( {L}_{2}\right.$ -workloads).

图24. R树和U树的模糊查询成本比较（$\left( {L}_{2}\right.$工作负载）。

<!-- Media -->

Evidently, the cost of U-catalog optimization and PCR computation accounts for a very small fraction of the overall overhead (particularly, finding all the PCRs of an object takes around 1.8 milliseconds). After those two tasks are completed, a U-tree can be built in time similar to an R-tree: on average 30 milliseconds to insert one object.

显然，U目录优化和PCR计算的成本在总开销中所占比例非常小（特别是，找到一个对象的所有PCR大约需要1.8毫秒）。完成这两项任务后，构建U树的时间与构建R树相近：平均插入一个对象需要30毫秒。

### 7.6 Three-Dimensional Results

### 7.6 三维结果

So far, we have focused on 2D data. The last set of experiments examines the performance of U-trees on three-dimensional objects. For this purpose, we generate a dataset ${RAN} - {3D}$ in the same way as ${RAN} - {250}$ ,except that each object’s uncertainty region is a 3D sphere (with radius 250 ),and the function $g\left( x\right)$ in Eq. (30) is the pdf of a three-variate normal distribution with standard deviation 125. The U-tree on ${RAN} - {3D}$ has a U-catalog size 3 . A (nonfuzzy/fuzzy) query workload is created in the same manner as a 2D counterpart.

到目前为止，我们主要关注二维数据。最后一组实验考察了U树在三维对象上的性能。为此，我们以与${RAN} - {250}$相同的方式生成数据集${RAN} - {3D}$，不同之处在于每个对象的不确定区域是一个三维球体（半径为250），并且式(30)中的函数$g\left( x\right)$是标准差为125的三元正态分布的概率密度函数。${RAN} - {3D}$上的U树的U目录大小为3。（非模糊/模糊）查询工作负载的创建方式与二维情况相同。

<!-- Media -->

<!-- figureText: tree building cost PCR computation $U$ -catalog tuning 2000 construction time (sec) DXXXXXXXXXXXXXXXXXXXX 1500 1000 500 0 $R$ -tree $U$ -tree (b) ${LB} - {100}$ 2500 construction time (sec) 2000 1500 1000 500 0 $R$ -tree $U$ -tree (a) ${CA} - {100}$ -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_45.jpg?x=469&y=337&w=867&h=402&r=0"/>

Fig. 25. Index construction cost.

图25. 索引构建成本。

<!-- figureText: $R$ -tree filter $R$ -tree refinement $U$ -tree refinement query time (sec) filter step cost 500 400 300 200 0.52 1.3 100 0.52 250 500 750 distance threshold ${\varepsilon }_{q}$ (b) Fuzzy search $U$ -tree filter query time (sec) 1 0.8 0.6 0.4 0.2 0 250 500 750 search region radius rad ${}_{q}$ (a) Nonfuzzy search -->

<img src="https://cdn.noedgeai.com/0195c90e-a59d-7468-9063-488689eb0411_45.jpg?x=468&y=803&w=866&h=424&r=0"/>

Fig. 26. Query cost on three dimensional data $\left( {{RAN} - {3D},{L}_{\infty }}\right.$ -workloads, ${t}_{q}$ uniform in $\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$ .

图26. 三维数据上的查询成本（$\left( {{RAN} - {3D},{L}_{\infty }}\right.$工作负载，${t}_{q}$在$\left. \left\lbrack  {{0.1},{0.9}}\right\rbrack  \right)$上均匀分布）。

<!-- Media -->

Figure 26(a) and (26(b)) compares the cost of answering a nonfuzzy (fuzzy) workload with the U- and R-trees on ${RAN} - {3D}$ ,when ${\operatorname{rad}}_{q}\left( {\epsilon }_{q}\right)$ changes from 250 to 750 . Each cost is broken down into the overhead of the filter and refinement steps, respectively. Since the filter costs are unnoticeable in Figure 26(b), we illustrate them on top of the columns. The U-tree consistently outperforms significantly the R-tree in all cases.

图26(a)和图26(b)比较了在${RAN} - {3D}$上使用U树和R树回答非模糊（模糊）工作负载的成本，此时${\operatorname{rad}}_{q}\left( {\epsilon }_{q}\right)$从250变化到750。每种成本分别分解为过滤步骤和细化步骤的开销。由于图26(b)中的过滤成本不明显，我们将其显示在柱状图的顶部。在所有情况下，U树的性能始终显著优于R树。

## 8. RELATED WORK

## 8. 相关工作

Section 8.1 first surveys various approaches for modeling uncertainty. Then, Section 8.2 discusses query algorithms for producing probabilistic results.

第8.1节首先综述了各种建模不确定性的方法。然后，第8.2节讨论了生成概率结果的查询算法。

### 8.1 Uncertainty Models

### 8.1 不确定性模型

Several models have been proposed to incorporate uncertain objects in databases. These models differ mainly in the semantics and complexities of the data in the underlying applications [Sarma et al. 2006]. In general, uncertainty can be represented in a "qualitative" or "quantitative" manner. A qualitative model captures the presence/absence of data, typically using a "NULL" keyword to describe a missing value. Accordingly, SQL needs to be augmented with additional keywords for querying such incomplete tuples, for example, "definite", "indefinite", "maybe" and "must" [Liu and Sunderraman 1987, 1991; Sistla et al. 1997].

已经提出了几种将不确定对象纳入数据库的模型。这些模型的主要区别在于底层应用中数据的语义和复杂度[Sarma等人，2006]。一般来说，不确定性可以用“定性”或“定量”的方式表示。定性模型捕捉数据的存在/缺失情况，通常使用“NULL”关键字来描述缺失值。因此，SQL需要增加额外的关键字来查询此类不完整的元组，例如“definite”（确定）、“indefinite”（不确定）、“maybe”（可能）和“must”（必须）[Liu和Sunderraman，1987，1991；Sistla等人，1997]。

To provide a more rigorous treatment of uncertain data, a quantitative approach describes uncertainty through mathematical modeling. These approaches include the fuzzy model [Galindo et al. 2006], the Dempster-Shafer (evidence-oriented) model [Lee 1992; Lim et al. 1996] and the probabilistic model. In particular, the probabilistic model can be further classified into three categories: "table-based", "tuple-based" and "attribute-based" solutions, which handle different granularities of uncertainty. Specifically, a table-based approach concerns the "coverage" of a table, that is, how much percentage of tuples are present in a table [Widom 2005]. A tuple-based solution, on the other hand, associates each individual tuple with a probability, which indicates the likelihood that the tuple exists in the table [Dalvi and Suciu 2004; Dalvi and Suciu 2005; Fuhr 1995]. This methodology has been applied to various forms of semi-structured data, such as XML documents [Nierman and Jagadish 2002] and other acyclic graphs [Hung et al. 2003]. Finally, when an attribute of a tuple is not known precisely, an attribute-based method introduces a probability distribution for describing a set of possible values, together with their occurring probabilities [Cheng et al. 2003; Deshpande et al. 2004; Pfoser and Jensen 1999; Wolfson et al. 1999].

为了更严谨地处理不确定数据，定量方法通过数学建模来描述不确定性。这些方法包括模糊模型[Galindo等人，2006年]、Dempster - Shafer（面向证据）模型[Lee，1992年；Lim等人，1996年]和概率模型。特别是，概率模型可以进一步分为三类：“基于表”“基于元组”和“基于属性”的解决方案，它们处理不同粒度的不确定性。具体而言，基于表的方法关注表的“覆盖率”，即表中存在的元组的百分比[Widom，2005年]。另一方面，基于元组的解决方案为每个单独的元组关联一个概率，该概率表示该元组存在于表中的可能性[Dalvi和Suciu，2004年；Dalvi和Suciu，2005年；Fuhr，1995年]。这种方法已应用于各种形式的半结构化数据，如XML文档[Nierman和Jagadish，2002年]和其他无环图[Hung等人，2003年]。最后，当元组的某个属性不能精确得知时，基于属性的方法引入一个概率分布来描述一组可能的值及其出现的概率[Cheng等人，2003年；Deshpande等人，2004年；Pfoser和Jensen，1999年；Wolfson等人，1999年]。

The attribute-based category, which is the focus of this article, has received a large amount of attention in the literature of spatiotemporal databases and sensor networks. For example, the modeling of vehicle locations illustrated in Figure 1(a) is due to Wolfson et al. [1999]. This model is extended by Pfoser and Jensen [1999] to enable estimation of the modeling error, by Trajcevski et al. [2004] to support trajectories, and by Teixeira de Almeida and Güting [2005] to road networks. A one-dimensional version of the model of Wolfson et al. [1999] is also employed to handle continuous sensor data in Cheng et al. [2003, 2006a]. In a similar context [Deshpande et al. 2004], a joint pdf of multiple attributes is deployed to capture the correlation of physical entities (e.g., temperature and pressure). The above work concentrates on continuous attributes, whereas uncertainty of discrete attributes is discussed in Barbará et al. [1992] and Lakshmanan et al. [1997]. The solutions developed in our article can be applied to all the models mentioned earlier.

基于属性的类别是本文的重点，在时空数据库和传感器网络的文献中受到了大量关注。例如，图1(a)所示的车辆位置建模来自Wolfson等人[1999年]。Pfoser和Jensen[1999年]扩展了该模型以实现对建模误差的估计，Trajcevski等人[2004年]扩展该模型以支持轨迹，Teixeira de Almeida和Güting[2005年]将其扩展到道路网络。Wolfson等人[1999年]模型的一维版本也被用于处理Cheng等人[2003年，2006a]中的连续传感器数据。在类似的背景下[Deshpande等人，2004年]，部署多个属性的联合概率密度函数（pdf）来捕捉物理实体（如温度和压力）的相关性。上述工作集中在连续属性上，而离散属性的不确定性在Barbará等人[1992年]和Lakshmanan等人[1997年]中进行了讨论。本文开发的解决方案可以应用于前面提到的所有模型。

Finally, there is a bulk of research [Cheng et al. 2003, 2006a; Khanna and Tan 2001; Olston et al. 2001; Olston and Widom 2000, 2002] that investigates how to reduce the cost of monitoring objects' uncertain representations (e.g., as temperature is rising, the sensor must decide whether to issue an update to the server, taking into account the tradeoff between communication overhead and the precision of modeling). The approaches there are complementary to our work, because they can be applied to generate objects' pdf updates to the U-tree. Article 15 / 48 * Y. Tao et al.

最后，有大量的研究[Cheng等人，2003年，2006a；Khanna和Tan，2001年；Olston等人，2001年；Olston和Widom，2000年，2002年]探讨了如何降低监控对象不确定表示的成本（例如，随着温度升高，传感器必须考虑通信开销和建模精度之间的权衡，决定是否向服务器发送更新）。那里的方法与我们的工作是互补的，因为它们可以用于生成对象的概率密度函数更新到U树。文章15 / 48 * Y. Tao等人

### 8.2 Query Evaluation

### 8.2 查询评估

In a broad sense, a "probabilistic query" is a user inquiry that retrieves the objects qualifying a set of predicates with certain probabilistic guarantees. Such queries are usually raised against a tuple-based or attribute-based uncertainty model. In particular, queries with respect to tuple-based modeling are formulated through the notion of either "intensional semantics" [Fuhr 1995] or "extensional semantics" [Dalvi and Suciu 2004, 2005]. For the attribute-based category, [Cheng et al. 2003, 2006a] present a detailed taxonomy that classifies a variety of probabilistic search, based on factors such as whether the result values are continuous or discrete, whether there is any relationship among the retrieved objects, and so on. Cheng et al. [2003, 2006a] also develop algorithms for evaluating queries of each class in the taxonomy. These algorithms are later adapted to solve problems in spatiotemporal databases [Cheng et al. 2004a], and sensor networks [Cheng et al. 2006a; Deshpande et al. 2004; Han et al. 2007]. Recently, join operations between two uncertain datasets are investigated in Kriegel et al. [2006].

从广义上讲，“概率查询”是一种用户查询，它以一定的概率保证检索符合一组谓词的对象。此类查询通常是针对基于元组或基于属性的不确定性模型提出的。特别是，关于基于元组建模的查询是通过“内涵语义”[Fuhr，1995年]或“外延语义”[Dalvi和Suciu，2004年，2005年]的概念来表述的。对于基于属性的类别，[Cheng等人，2003年，2006a]提出了一个详细的分类法，根据结果值是连续的还是离散的、检索对象之间是否存在任何关系等因素，对各种概率搜索进行分类。Cheng等人[2003年，2006a]还开发了用于评估分类法中每个类别的查询的算法。这些算法后来被用于解决时空数据库[Cheng等人，2004a]和传感器网络[Cheng等人，2006a；Deshpande等人，2004年；Han等人，2007年]中的问题。最近，Kriegel等人[2006年]研究了两个不确定数据集之间的连接操作。

In practice, the above methods may incur expensive cost, since they must compute the actual qualification probability of every object. Motivated by this, (targeting attribute-based modeling), Cheng et al. [2004b] introduce the concept of "probability thresholding", as formally defined in Section 2. In Cheng et al. [2004b], the authors also explore access methods that minimize the I/O cost of one dimensional probability threshold range search. They argue that uncertain databases are inherently more difficult to handle (than the precise counterpart), and support their claim by proving an asymptotical lower bound for the optimal I/O performance. They also develop several index structures that (almost) achieve the lower bound, but, unfortunately, are limited to one-dimensional spaces. In the preliminary version [Tao et al. 2005] of the current article, we tackle multidimensional data with the basic version of the heuristics in Section 3, and describe a compression-based implementation of the U-tree.

实际上，上述方法可能会产生高昂的成本，因为它们必须计算每个对象的实际符合概率。受此启发（针对基于属性的建模），Cheng等人 [2004b] 引入了“概率阈值化”的概念，如第2节中正式定义的那样。在Cheng等人 [2004b] 的研究中，作者还探索了使一维概率阈值范围搜索的I/O成本最小化的访问方法。他们认为，不确定数据库本质上比精确数据库更难处理，并通过证明最优I/O性能的渐近下界来支持他们的观点。他们还开发了几种（几乎）能达到该下界的索引结构，但不幸的是，这些结构仅限于一维空间。在本文的初步版本 [Tao等人2005] 中，我们使用第3节中启发式方法的基本版本处理多维数据，并描述了基于压缩的U树实现方法。

The techniques proposed in this article extend beyond the methods in Cheng et al. [2004b] and Tao et al. [2005]. Specifically, we (i) present a thorough set of heuristics for pruning and validation of nonqualifying and qualifying objects, respectively, (ii) perform a careful theoretical analysis to prove the effectiveness of those heuristics, and (iii) devise fast algorithms for fuzzy range search, which is not addressed in Cheng et al. [2004b] and Tao et al. [2005].

本文提出的技术超越了Cheng等人 [2004b] 和Tao等人 [2005] 的方法。具体来说，我们（i）分别提出了一套全面的启发式方法，用于修剪和验证不符合和符合条件的对象；（ii）进行了细致的理论分析，以证明这些启发式方法的有效性；（iii）设计了用于模糊范围搜索的快速算法，这在Cheng等人 [2004b] 和Tao等人 [2005] 的研究中并未涉及。

## 9. CONCLUSIONS AND FUTURE WORK

## 9. 结论与未来工作

As has been proved in spatial databases, range search is a problem fundamental to a large number of analytical tasks [Gaede and Gunther 1998]. Unfortunately, there has been no formal research about optimizing this operation on multidimensional uncertain objects, thus currently preventing such data from being manipulated and analyzed efficiently. This article alleviates the situation by presenting a comprehensive study on two forms of range retrieval common in practice: nonfuzzy and fuzzy search. Based on a novel concept of "probabilistically constrained rectangle" (PCR), we carefully developed a set of heuristics for effectively pruning (or validating) nonqualifying (or qualifying) objects. PCR also motivates a new index structure called the U-tree for minimizing the I/O overhead of range queries. Finally, we accompany our algorithmic findings with a thorough performance analysis, which explains the reasoning behind the efficiency of the proposed techniques, and leads to the development of a cost model that can be applied to query optimization.

正如空间数据库中所证明的那样，范围搜索是大量分析任务的基础问题 [Gaede和Gunther 1998]。不幸的是，目前还没有关于优化多维不确定对象上此操作的正式研究，因此目前阻碍了对这类数据的高效操作和分析。本文通过对实践中常见的两种范围检索形式（非模糊搜索和模糊搜索）进行全面研究，缓解了这一情况。基于“概率约束矩形”（PCR，probabilistically constrained rectangle）这一新颖概念，我们精心开发了一套启发式方法，用于有效修剪（或验证）不符合（或符合）条件的对象。PCR还催生了一种名为U树的新索引结构，用于最小化范围查询的I/O开销。最后，我们对算法结果进行了全面的性能分析，解释了所提出技术高效性背后的原因，并开发了一个可应用于查询优化的成本模型。

Query processing on multidimensional uncertain databases, in general, remains an attractive research topic that has not been extensively explored. The work of this article lays down the foundation for designing fast algorithms towards accomplishing various data mining goals, such as clustering, outlier detection, etc (investigation of these algorithms would very likely motivate alternative access methods, which, in turn, may actually inspire improvement of the U-tree). A challenge, however, lies in the semantics/representations of the mined results. For example, what does a "cluster of uncertain objects" mean exactly? How to store a cluster concisely without losing its semantics? As another example, what is a (global/local) "outlier" in a dataset when each object is described with a pdf? Answers to these questions would naturally spawn new, practical, definitions of the existing data mining concepts.

一般来说，多维不确定数据库上的查询处理仍然是一个有吸引力但尚未得到广泛探索的研究课题。本文的工作为设计快速算法以实现各种数据挖掘目标（如聚类、异常值检测等）奠定了基础（对这些算法的研究很可能会催生其他访问方法，而这些方法反过来又可能会启发U树的改进）。然而，挑战在于挖掘结果的语义/表示。例如，“不确定对象的聚类”究竟是什么意思？如何在不丢失语义的情况下简洁地存储一个聚类？再比如，当每个对象都用概率密度函数（pdf，probability density function）描述时，数据集中的（全局/局部）“异常值”是什么？对这些问题的回答自然会催生对现有数据挖掘概念的新的、实用的定义。

## APPENDIX: PROOFS OF LEMMAS AND THEOREMS

## 附录：引理和定理的证明

Proof of Theorem 1. We will only prove Rule (1), because Rule (2) can be established in a similar manner. Since ${r}_{q}$ does not fully contain o.pcr $\left( {1 - {t}_{q}}\right)$ , there must be a face of $o \cdot  {pcr}\left( {1 - {t}_{q}}\right)$ such that,both $o \cdot  {pcr}\left( {1 - {t}_{q}}\right)$ and ${r}_{q}$ lie on the same side of the $d$ -dimensional plane containing the face. Let us denote the place as $l$ . Consider the portion of o.ur that lies on the opposite side of $l$ with respect to ${r}_{q}$ . By the definition of $o \cdot  {pcr}\left( {1 - {t}_{q}}\right) ,o$ has probability $1 - {t}_{q}$ to appear in that portion. As a result,the probability that $o$ falls in ${r}_{q}$ is smaller than $1 - \left( {1 - {t}_{q}}\right)  = {t}_{q}$ . Thus, $o$ can be safely pruned.

定理1的证明。我们仅证明规则（1），因为规则（2）可以用类似的方式证明。由于${r}_{q}$并未完全包含o.pcr $\left( {1 - {t}_{q}}\right)$，因此$o \cdot  {pcr}\left( {1 - {t}_{q}}\right)$必定存在一个面，使得$o \cdot  {pcr}\left( {1 - {t}_{q}}\right)$和${r}_{q}$都位于包含该面的$d$维平面的同一侧。我们将该平面记为$l$。考虑o.ur中相对于${r}_{q}$位于$l$另一侧的部分。根据$o \cdot  {pcr}\left( {1 - {t}_{q}}\right) ,o$的定义，$o \cdot  {pcr}\left( {1 - {t}_{q}}\right) ,o$出现在该部分的概率为$1 - {t}_{q}$。因此，$o$落入${r}_{q}$的概率小于$1 - \left( {1 - {t}_{q}}\right)  = {t}_{q}$。所以，可以安全地修剪$o$。

Proof of Theorem 2. To prove Rule (1),we aim at obtaining a rectangle $r$ which is completely covered by ${r}_{q}$ . The rectangle $r$ has the property that $o$ has at least $1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$ probability to appear in $r$ ,which,combined with the fact $1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)  \geq  {t}_{q}$ ,confirms that $o$ indeed qualifies $q$ . More specifically, at the beginning, $r$ is initialized to $\operatorname{o.pcr}\left( 0\right)$ . Then,we will shrink $r$ along each of the $d - l$ dimensions on which o.pcr(0) is not covered by ${r}_{q}$ . During the whole process,we will use $\rho$ to denote a lower bound for the probability that $o$ appears in the current $r$ . The starting value of $\rho$ is 1,and its final value will be exactly $1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$ .

定理2的证明。为了证明规则(1)，我们的目标是得到一个矩形$r$，它能被${r}_{q}$完全覆盖。矩形$r$具有这样的性质：$o$至少有$1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$的概率出现在$r$中，结合事实$1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)  \geq  {t}_{q}$，这证实了$o$确实满足$q$。更具体地说，一开始，$r$被初始化为$\operatorname{o.pcr}\left( 0\right)$。然后，我们将沿着$d - l$个维度收缩$r$，在这些维度上o.pcr(0)未被${r}_{q}$覆盖。在整个过程中，我们将用$\rho$表示$o$出现在当前$r$中的概率的下界。$\rho$的初始值为1，其最终值将恰好为$1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$。

On dimension 1,we shrink $r$ by moving its left edge to the left boundary of o. $\operatorname{pcr}\left( {c}_{1}\right)$ . Compared to the $r$ before the shrinking,the appearance probability of $o$ inside $r$ has been reduced by at most ${c}_{1}$ ,due to the formulation of $o.{pcr}\left( {c}_{1}\right)$ . Hence,with respect to the new $r$ ,the value of $\rho$ can be updated to $1 - {c}_{1}$ . Similarly,we shrink $r$ again by moving its right edge (still,on dimension 1) to the right boundary of $\operatorname{o.pcr}\left( {c}_{1}^{\prime }\right)$ ,and update $\rho$ to $1 - {c}_{1} - {c}_{1}^{\prime }$ .

在第1维上，我们通过将$r$的左边缘移动到o. $\operatorname{pcr}\left( {c}_{1}\right)$的左边界来收缩$r$。与收缩前的$r$相比，由于$o.{pcr}\left( {c}_{1}\right)$的公式，$o$在$r$内的出现概率最多降低了${c}_{1}$。因此，对于新的$r$，$\rho$的值可以更新为$1 - {c}_{1}$。类似地，我们再次收缩$r$，将其右边缘（仍然在第1维上）移动到$\operatorname{o.pcr}\left( {c}_{1}^{\prime }\right)$的右边界，并将$\rho$更新为$1 - {c}_{1} - {c}_{1}^{\prime }$。

Performing the above shrinking on all the dimensions $1,2,\ldots ,d - l$ ,we end up with (i) a rectangle $r$ whose left (or right) edge along the $i$ th dimension $\left( {1 \leq  i \leq  d - l}\right)$ coincides with that of $o.{pcr}\left( {c}_{i}\right)$ (or $o.{pcr}\left( {c}_{i}^{\prime }\right)$ ),and (ii) a $\rho$ with value $1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$ . Hence,the projections of $r$ on all these $d - l$ dimensions are contained by those of ${r}_{q}$ . Furthermore, $r$ is covered by ${r}_{q}$ along the remaining $l$ dimensions; therefore,we have discovered an $r$ and a $\rho$ needed for proving the first rule, as stated at the beginning of the proof.

在所有维度$1,2,\ldots ,d - l$上执行上述收缩操作后，我们得到：(i) 一个矩形$r$，其沿第$i$维$\left( {1 \leq  i \leq  d - l}\right)$的左（或右）边缘与$o.{pcr}\left( {c}_{i}\right)$（或$o.{pcr}\left( {c}_{i}^{\prime }\right)$）的左（或右）边缘重合；(ii) 一个值为$1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)$的$\rho$。因此，$r$在所有这些$d - l$维度上的投影都包含在${r}_{q}$的投影内。此外，$r$在其余$l$个维度上被${r}_{q}$覆盖；因此，正如证明开始时所述，我们已经找到了证明第一条规则所需的$r$和$\rho$。

Finally, Rule (2) can be established in a similar, but simpler, way. Due to symmetry,let us consider the case where $\left\lbrack  {{r}_{q\left\lbrack  1\right\rbrack   - },{r}_{q\left\lbrack  1\right\rbrack   + }}\right\rbrack$ encloses $\left\lbrack  {o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$ . Initially, $r$ equals a rectangle that shares the same extents as o.mbr on all the dimensions except the first one,along which $r$ has a projection $\left\lbrack  {o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( 0\right) ,o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$ . We set $\rho$ to ${c}_{1}^{\prime }$ ,which is the exact probability that $o$ appears in $r$ . Then,we shrink $r$ on the first dimension,by moving its left edge to the left boundary of $\operatorname{o.pcr}\left( {c}_{1}\right)$ . Accordingly, $\rho$ can be updated to ${c}_{1}^{\prime } - {c}_{1}$ . The current $r$ is contained in ${r}_{q}$ ; hence,Rule (2) holds.

最后，可以用类似但更简单的方法来证明规则 (2)。由于对称性，让我们考虑 $\left\lbrack  {{r}_{q\left\lbrack  1\right\rbrack   - },{r}_{q\left\lbrack  1\right\rbrack   + }}\right\rbrack$ 包含 $\left\lbrack  {o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$ 的情况。最初，$r$ 等于一个矩形，该矩形在除第一个维度之外的所有维度上与对象最小边界矩形（o.mbr）具有相同的范围，在第一个维度上，$r$ 的投影为 $\left\lbrack  {o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( 0\right) ,o.{pc}{r}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$。我们将 $\rho$ 设置为 ${c}_{1}^{\prime }$，这是 $o$ 出现在 $r$ 中的精确概率。然后，我们在第一个维度上缩小 $r$，将其左边缘移动到 $\operatorname{o.pcr}\left( {c}_{1}\right)$ 的左边界。相应地，$\rho$ 可以更新为 ${c}_{1}^{\prime } - {c}_{1}$。当前的 $r$ 包含在 ${r}_{q}$ 中；因此，规则 (2) 成立。

Proof of Theorem 3. Let us first establish Rule (1). Since ${r}_{q}$ does not fully cover $o \cdot  \operatorname{pcr}\left( {c}_{ \vdash  }\right)$ ,by Rule (1) of Theorem 1, $o$ does not qualify $q$ if its probability threshold ${t}_{q}$ were $1 - {c}_{ \vdash  }$ . In fact, ${c}_{ \vdash  } \geq  1 - {t}_{q}$ ,that is,the actual ${t}_{q}$ is at least $1 - {c}_{ \vdash  }$ ; therefore, $o$ can be safely eliminated. Rule (2) can be verified in a similar manner. Specifically,since ${r}_{q}$ is disjoint with $\operatorname{o.pcr}\left( {c}_{ \dashv  }\right)$ ,according to Rule (2) of Theorem 1, $o$ does not satisfy $q$ even if its ${t}_{q}$ were ${c}_{\neg }$ ,which is at most the actual ${t}_{q}$ . Hence, $o$ can again be pruned.

定理 3 的证明。我们首先证明规则 (1)。由于 ${r}_{q}$ 没有完全覆盖 $o \cdot  \operatorname{pcr}\left( {c}_{ \vdash  }\right)$，根据定理 1 的规则 (1)，如果 $o$ 的概率阈值 ${t}_{q}$ 为 $1 - {c}_{ \vdash  }$，则 $o$ 不符合 $q$ 的条件。实际上，${c}_{ \vdash  } \geq  1 - {t}_{q}$，即实际的 ${t}_{q}$ 至少为 $1 - {c}_{ \vdash  }$；因此，可以安全地排除 $o$。规则 (2) 可以用类似的方式验证。具体来说，由于 ${r}_{q}$ 与 $\operatorname{o.pcr}\left( {c}_{ \dashv  }\right)$ 不相交，根据定理 1 的规则 (2)，即使 $o$ 的 ${t}_{q}$ 为 ${c}_{\neg }$（这最多是实际的 ${t}_{q}$），$o$ 也不满足 $q$ 的条件。因此，$o$ 再次可以被剪枝。

Proof of LEMMA 1. The lemma trivially holds if ${r}_{q}$ contains or is disjoint with o.mbr. In the sequel,we discuss the case where ${r}_{q}$ partially overlaps o.mbr, starting with the pruning part of the lemma.

引理 1 的证明。如果 ${r}_{q}$ 包含对象最小边界矩形（o.mbr）或与对象最小边界矩形（o.mbr）不相交，则该引理显然成立。接下来，我们讨论 ${r}_{q}$ 与对象最小边界矩形（o.mbr）部分重叠的情况，从引理的剪枝部分开始。

Pruning Case $1 : {t}_{q} \leq  1 - {C}_{m}$ . In this scenario,the value ${c}_{ \leq  }$ at Line 4 of Algorithm 1 must exist; otherwise, $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ would be decided at Line 8, and larger than $1 - {C}_{m}$ ,violating the condition ${t}_{q} > U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ . Hence, $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = {c}_{ \leq  } - \delta$ (where $\delta$ is an infinitely small positive),leading to ${c}_{ \leq  } \leq  {t}_{q}$ . Let ${c}_{ \dashv  }$ be the largest value in the U-catalog that is at most ${t}_{q}$ . It follows that ${c}_{ \leq  } \leq  {c}_{ \dashv  }$ ,that is, $o \cdot  \operatorname{pcr}\left( {c}_{ \leq  }\right)$ contains $o \cdot  \operatorname{pcr}\left( {c}_{ \dashv  }\right)$ . By the way ${c}_{ \leq  }$ is decided, $o \cdot  \operatorname{pcr}\left( {c}_{ \leq  }\right)$ is disjoint with ${r}_{q}$ . Therefore, $o.{pcr}\left( {c}_{\neg }\right)$ is also disjoint with ${r}_{q}$ ,so that $o$ is pruned by Rule (2) of Theorem 3.

剪枝情况 $1 : {t}_{q} \leq  1 - {C}_{m}$ 。在这种情况下，算法1第4行的数值 ${c}_{ \leq  }$ 必定存在；否则， $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ 将在第8行被确定，且大于 $1 - {C}_{m}$ ，这违反了条件 ${t}_{q} > U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ 。因此， $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = {c}_{ \leq  } - \delta$ （其中 $\delta$ 是一个无穷小的正数），从而得到 ${c}_{ \leq  } \leq  {t}_{q}$ 。设 ${c}_{ \dashv  }$ 是U目录中至多为 ${t}_{q}$ 的最大值。由此可得 ${c}_{ \leq  } \leq  {c}_{ \dashv  }$ ，即 $o \cdot  \operatorname{pcr}\left( {c}_{ \leq  }\right)$ 包含 $o \cdot  \operatorname{pcr}\left( {c}_{ \dashv  }\right)$ 。根据 ${c}_{ \leq  }$ 的确定方式， $o \cdot  \operatorname{pcr}\left( {c}_{ \leq  }\right)$ 与 ${r}_{q}$ 不相交。因此， $o.{pcr}\left( {c}_{\neg }\right)$ 也与 ${r}_{q}$ 不相交，所以 $o$ 被定理3的规则(2)剪枝。

Pruning Case 2: ${t}_{q} > 1 - {C}_{m}$ . Assume,on the contrary,that $o$ cannot be pruned by Theorem 3. Thus, ${r}_{q}$ fully covers $\operatorname{o.pcr}\left( {c}_{ \vdash  }\right)$ ,where ${c}_{ \vdash  }$ is the smallest value in the U-catalog at least $1 - {t}_{q}$ ; otherwise, $o$ would have been eliminated by Rule (1) of Theorem 3. Therefore, ${r}_{q}$ definitely contains o.pcr $\left( {C}_{m}\right)$ ,where ${C}_{m}$ is the largest value in the U-catalog. It follows that ${c}_{ \leq  }$ does not exist at Line 4 . Let us examine the ${c}_{ \geq  }$ produced at Line 7. As ${t}_{q} > {\overline{UB}}_{\text{range }}\left( {o,{r}_{q}}\right)  = 1 - {c}_{ \geq  } - \delta$ , we have ${c}_{ \geq  } \geq  1 - {t}_{q}$ . Hence, ${c}_{ \geq  } \geq  {c}_{ \vdash  }$ (recall the criterion of choosing ${c}_{ \geq  }$ ),and o. $\operatorname{pcr}\left( {c}_{ \vdash  }\right)$ contains $\operatorname{o.pcr}\left( {c}_{ \geq  }\right)$ . Since,due to the way ${c}_{ \geq  }$ is selected, ${r}_{q}$ does not fully cover $\operatorname{o.pcr}\left( {c}_{ \geq  }\right) ,{r}_{q}$ cannot enclose $\operatorname{o.pcr}\left( {c}_{ \vdash  }\right)$ ,either. Here,we arrive at a contradiction.

剪枝情况2： ${t}_{q} > 1 - {C}_{m}$ 。相反地，假设 $o$ 不能被定理3剪枝。因此， ${r}_{q}$ 完全覆盖 $\operatorname{o.pcr}\left( {c}_{ \vdash  }\right)$ ，其中 ${c}_{ \vdash  }$ 是U目录中至少为 $1 - {t}_{q}$ 的最小值；否则， $o$ 会被定理3的规则(1)排除。因此， ${r}_{q}$ 肯定包含o.pcr $\left( {C}_{m}\right)$ ，其中 ${C}_{m}$ 是U目录中的最大值。由此可知，第4行的 ${c}_{ \leq  }$ 不存在。让我们考察第7行产生的 ${c}_{ \geq  }$ 。由于 ${t}_{q} > {\overline{UB}}_{\text{range }}\left( {o,{r}_{q}}\right)  = 1 - {c}_{ \geq  } - \delta$ ，我们有 ${c}_{ \geq  } \geq  1 - {t}_{q}$ 。因此， ${c}_{ \geq  } \geq  {c}_{ \vdash  }$ （回顾选择 ${c}_{ \geq  }$ 的标准），并且o. $\operatorname{pcr}\left( {c}_{ \vdash  }\right)$ 包含 $\operatorname{o.pcr}\left( {c}_{ \geq  }\right)$ 。由于 ${c}_{ \geq  }$ 的选择方式， ${r}_{q}$ 不能完全覆盖 $\operatorname{o.pcr}\left( {c}_{ \geq  }\right) ,{r}_{q}$ ，也不能包含 $\operatorname{o.pcr}\left( {c}_{ \vdash  }\right)$ 。在这里，我们得到了一个矛盾。

We proceed with the validating part of the lemma, also considering two cases.

我们继续进行引理的验证部分，同样考虑两种情况。

Validating Case 1: $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ produced at Line 15. Let us apply the ${c}_{1}$ , ${c}_{1}^{\prime },\ldots ,{c}_{d - l},{c}_{d - l}^{\prime }$ calculated at Lines 13 and 14 in Rule (1) of Theorem 2 (the projection of ${r}_{q}$ does not contain that of $o.{mbr}$ on dimensions $1,\ldots ,d - l$ ). By the way these $2\left( {d - l}\right)$ values are decided,the projection of ${r}_{q}$ on each dimension $i \in  \left\lbrack  {1,d - l}\right\rbrack$ encloses $\left\lbrack  {o \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{i}\right) ,o \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{i}^{\prime }\right) }\right\rbrack$ . Since ${t}_{q} \leq  L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = 1 -$ $\mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right) ,o$ is validated by Rule (1).

验证情况1：第15行生成的$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$。让我们应用定理2的规则(1)中第13行和第14行计算得到的${c}_{1}$、${c}_{1}^{\prime },\ldots ,{c}_{d - l},{c}_{d - l}^{\prime }$（${r}_{q}$在维度$1,\ldots ,d - l$上的投影不包含$o.{mbr}$的投影）。根据这些$2\left( {d - l}\right)$值的确定方式，${r}_{q}$在每个维度$i \in  \left\lbrack  {1,d - l}\right\rbrack$上的投影包含$\left\lbrack  {o \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   - }\left( {c}_{i}\right) ,o \cdot  {\operatorname{pcr}}_{\left\lbrack  i\right\rbrack   + }\left( {c}_{i}^{\prime }\right) }\right\rbrack$。由于${t}_{q} \leq  L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = 1 -$ $\mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right) ,o$通过规则(1)得到验证。

Validating Case 2: $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ produced at Line 19. In this scenario, $l =$ $d - 1$ . We apply the values of ${c}_{1}$ and ${c}_{1}^{\prime }$ computed at Lines 17 and 18 in Rule (2) of Theorem 2. According to the manner these two values are selected, the projection of ${r}_{q}$ on dimension 1 encloses either $\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$ or $\left\lbrack  {{\operatorname{o.pcr}}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}^{\prime }\right) ,{\operatorname{o.pcr}}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}\right) }\right\rbrack$ . In both situations,as ${t}_{q} \leq  {c}_{1}^{\prime } - {c}_{1},o$ is validated by Rule (2).

验证情况2：第19行生成的$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$。在这种情况下，$l =$ $d - 1$。我们应用定理2的规则(2)中第17行和第18行计算得到的${c}_{1}$和${c}_{1}^{\prime }$的值。根据这两个值的选择方式，${r}_{q}$在维度1上的投影包含$\left\lbrack  {o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}\right) ,o.{\operatorname{pcr}}_{\left\lbrack  1\right\rbrack   - }\left( {c}_{1}^{\prime }\right) }\right\rbrack$或$\left\lbrack  {{\operatorname{o.pcr}}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}^{\prime }\right) ,{\operatorname{o.pcr}}_{\left\lbrack  1\right\rbrack   + }\left( {c}_{1}\right) }\right\rbrack$。在这两种情况下，由于${t}_{q} \leq  {c}_{1}^{\prime } - {c}_{1},o$通过规则(2)得到验证。

Proof of LEMMA 2. The lemma is obviously true if ${r}_{q}$ completely covers or is disjoint with $o.{mbr}$ . In the sequel,we focus on the case where ${r}_{q}$ partially overlaps o.mbr. To prove the part of the lemma about pruning, assume, on the contrary,that pruning with Theorem 3 is possible for a ${t}_{q} \leq  U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ . We discuss two cases separately.

引理2的证明。如果${r}_{q}$完全覆盖$o.{mbr}$或与$o.{mbr}$不相交，那么该引理显然成立。接下来，我们关注${r}_{q}$与o.mbr部分重叠的情况。为了证明引理中关于剪枝的部分，假设相反的情况，即对于一个${t}_{q} \leq  U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$，使用定理3进行剪枝是可能的。我们分别讨论两种情况。

Pruning Case 1: $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ produced at Line 5 of Algorithm 1. Accordingly, ${t}_{q} \leq  U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  < {C}_{m}$ ; hence,only Rule (2) of Theorem 3 could have eliminated $o$ ,meaning that ${r}_{q}$ is disjoint with $o \cdot  {pcr}\left( {c}_{ \dashv  }\right)$ ,where ${c}_{ \dashv  }$ is the largest U-catalog value at most ${t}_{q}$ . Let ${c}_{ \leq  }$ be the value computed at Line 4,that is, ${c}_{ \leq  } = U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  + \delta$ ,where $\delta$ is an infinitely small positive. Since ${c}_{ \dashv  } \leq  {t}_{q} < {c}_{ \leq  }$ , ${c}_{ \leq  }$ is no longer the smallest value $c$ in the U-catalog such that o.pcr(c) is disjoint with ${r}_{q}$ ,which violates the definition of ${c}_{ \leq  }$ .

剪枝情况1：$U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ 在算法1的第5行产生。因此，${t}_{q} \leq  U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  < {C}_{m}$ ；所以，只有定理3的规则(2)可能排除了$o$ ，这意味着${r}_{q}$ 与$o \cdot  {pcr}\left( {c}_{ \dashv  }\right)$ 不相交，其中${c}_{ \dashv  }$ 是至多为${t}_{q}$ 的最大U目录值。设${c}_{ \leq  }$ 是在第4行计算得到的值，即${c}_{ \leq  } = U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  + \delta$ ，其中$\delta$ 是一个无穷小的正数。由于${c}_{ \dashv  } \leq  {t}_{q} < {c}_{ \leq  }$ ，${c}_{ \leq  }$ 不再是U目录中使得o.pcr(c)与${r}_{q}$ 不相交的最小的$c$ 值，这违反了${c}_{ \leq  }$ 的定义。

Pruning Case 2: $U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ produced at Line 8. Let ${c}_{ \geq  }$ be the value computed at Line 7,that is, ${c}_{ \geq  } = 1 - U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  - \delta  < 1 - {t}_{q}$ . As Line 7 has been executed,all o.pcr(c) (for any $c$ in the U-catalog) must intersect ${r}_{q}$ . Thus, $o$ can be pruned only by Rule (1) of Theorem 3,meaning that ${r}_{q}$ does not fully cover o. $\operatorname{pcr}\left( {c}_{ \vdash  }\right)$ ,where ${c}_{ \vdash  }$ is a U-catalog value at least $1 - {t}_{q}$ . Since ${c}_{ \vdash  } \geq  1 - {t}_{q} > {c}_{ \geq  }$ , ${c}_{ \geq  }$ is no longer the largest value $c$ in the U-catalog such that ${r}_{q}$ does not fully cover $o.{pcr}\left( c\right)$ . This violates the definition of $c$ .

剪枝情况2：$U{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ 在第8行产生。设${c}_{ \geq  }$ 是在第7行计算得到的值，即${c}_{ \geq  } = 1 - U{B}_{\text{range }}\left( {o,{r}_{q}}\right)  - \delta  < 1 - {t}_{q}$ 。由于第7行已执行，所有o.pcr(c)（对于U目录中的任何$c$ ）必须与${r}_{q}$ 相交。因此，$o$ 只能由定理3的规则(1)进行剪枝，这意味着${r}_{q}$ 不能完全覆盖o. $\operatorname{pcr}\left( {c}_{ \vdash  }\right)$ ，其中${c}_{ \vdash  }$ 是一个至少为$1 - {t}_{q}$ 的U目录值。由于${c}_{ \vdash  } \geq  1 - {t}_{q} > {c}_{ \geq  }$ ，${c}_{ \geq  }$ 不再是U目录中使得${r}_{q}$ 不能完全覆盖$o.{pcr}\left( c\right)$ 的最大的$c$ 值。这违反了$c$ 的定义。

We continue to prove the part of the lemma about validating. Assume, on the contrary,that validating with Theorem 2 is possible for a ${t}_{q} > L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ . We again distinguishing two cases.

我们继续证明引理中关于验证部分的内容。相反，假设对于一个${t}_{q} > L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ ，使用定理2进行验证是可能的。我们再次区分两种情况。

Validating Case 1: $o$ is validated by Rule (1) of Theorem 2. In this scenario, ${r}_{q}$ encloses $\operatorname{o.pcr}\left( {C}_{m}\right)$ ,and thus, $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ is determined at Line 15 . Let ${c}_{1}$ , ${c}_{1}^{\prime },\ldots ,{c}_{d - l},{c}_{d - l}^{\prime }$ be the $2\left( {d - l}\right)$ values computed at Lines 13 and 14. Similarly, we use ${c}_{1}^{ * },{c}_{1}^{{ * }^{\prime }},\ldots ,{c}_{d - l}^{ * },{c}_{d - l}^{{ * }^{\prime }}$ to denote the values used in Rule (1) for validating o. For every $i \in  \left\lbrack  {1,d - l}\right\rbrack$ ,where $l$ is as defined at Line 10,we have ${c}_{i} \leq  {c}_{i}^{ * }$ and ${c}_{i}^{\prime } \leq  {c}_{i}^{{ * }^{\prime }}$ ,due to the way that ${c}_{i}$ and ${c}_{i}^{\prime }$ are selected. Therefore, $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  =$ $1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)  \geq  1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i}^{ * } + {c}_{i}^{{ * }^{\prime }}}\right)  \geq  {t}_{q}$ ,leading to a contradiction.

验证情况1：$o$由定理2的规则(1)验证。在这种情况下，${r}_{q}$包含$\operatorname{o.pcr}\left( {C}_{m}\right)$，因此，$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$在第15行确定。设${c}_{1}$、${c}_{1}^{\prime },\ldots ,{c}_{d - l},{c}_{d - l}^{\prime }$为在第13行和第14行计算得到的$2\left( {d - l}\right)$值。类似地，我们用${c}_{1}^{ * },{c}_{1}^{{ * }^{\prime }},\ldots ,{c}_{d - l}^{ * },{c}_{d - l}^{{ * }^{\prime }}$表示规则(1)中用于验证o的值。对于每个$i \in  \left\lbrack  {1,d - l}\right\rbrack$，其中$l$如第10行所定义，由于${c}_{i}$和${c}_{i}^{\prime }$的选择方式，我们有${c}_{i} \leq  {c}_{i}^{ * }$和${c}_{i}^{\prime } \leq  {c}_{i}^{{ * }^{\prime }}$。因此，$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  =$ $1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i} + {c}_{i}^{\prime }}\right)  \geq  1 - \mathop{\sum }\limits_{{i = 1}}^{{d - l}}\left( {{c}_{i}^{ * } + {c}_{i}^{{ * }^{\prime }}}\right)  \geq  {t}_{q}$，导致矛盾。

Validating Case 2: $o$ is validated by Rule (2) of Theorem 2. When this happens, ${r}_{q}$ does not contain $\operatorname{o.pcr}\left( {C}_{m}\right)$ (otherwise,it is easy to observe that Rule (1) can also validate $o$ ); hence, $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$ is obtained at Line 19. Let ${c}_{1},{c}_{1}^{\prime }$ be the values computed at Lines 17,18,and ${c}_{1}^{ * },{c}_{1}^{{ * }^{\prime }}$ the values used in Rule (2) for validating $o$ . By the way that ${c}_{1}$ and ${c}_{1}^{\prime }$ are chosen,we have ${c}_{1}^{\prime } \geq  {c}_{1}^{{ * }^{\prime }}$ and ${c}_{1} \leq  {c}_{1}^{\prime }$ . Therefore, $L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = {c}_{1}^{\prime } - {c}_{1} \geq  {c}_{1}^{{ * }^{\prime }} - {c}_{1}^{ * } \geq  {t}_{q}$ ,resulting in a contradiction.

验证情况2：$o$由定理2的规则(2)验证。当这种情况发生时，${r}_{q}$不包含$\operatorname{o.pcr}\left( {C}_{m}\right)$（否则，很容易观察到规则(1)也可以验证$o$）；因此，$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)$在第19行得到。设${c}_{1},{c}_{1}^{\prime }$为在第17行、第18行计算得到的值，${c}_{1}^{ * },{c}_{1}^{{ * }^{\prime }}$为规则(2)中用于验证$o$的值。根据${c}_{1}$和${c}_{1}^{\prime }$的选择方式，我们有${c}_{1}^{\prime } \geq  {c}_{1}^{{ * }^{\prime }}$和${c}_{1} \leq  {c}_{1}^{\prime }$。因此，$L{B}_{\text{range }}\left( {o,{r}_{q}}\right)  = {c}_{1}^{\prime } - {c}_{1} \geq  {c}_{1}^{{ * }^{\prime }} - {c}_{1}^{ * } \geq  {t}_{q}$，导致矛盾。

Proof of Theorem 4. The theorem is a direct corollary of the definitions of PCRs.

定理4的证明。该定理是PCR（概率一致性规则，Probabilistic Consistency Rules）定义的直接推论。

Proof of Theorem 5. Same as the proof of Theorem 2, except that, in the part establishing Rule (1), $l$ ’ should be replaced with $d$ ’,while,in the part about Rule (2), "dimension 1" is now "dimension i".

定理5的证明。与定理2的证明相同，只是在建立规则(1)的部分，应将$l$’替换为$d$’，而在关于规则(2)的部分，“维度1”现在是“维度i”。

Proof of LEMMA 3. Inequality (12) follows immediately the definition of $P{r}_{\text{range }}$ and the fact that $\sqcap  \left( {r,{\varepsilon }_{q}}\right)  \subseteq   \odot  \left( {x,{\varepsilon }_{q}}\right)  \subseteq   \sqcup  \left( {r,{\varepsilon }_{q}}\right)$ .

引理3的证明。不等式(12)可直接由$P{r}_{\text{range }}$的定义以及$\sqcap  \left( {r,{\varepsilon }_{q}}\right)  \subseteq   \odot  \left( {x,{\varepsilon }_{q}}\right)  \subseteq   \sqcup  \left( {r,{\varepsilon }_{q}}\right)$这一事实得出。

Proof of LEMMA 4. Due to symmetry, it suffices to prove only Inequality (13). Given ${r}_{1},\ldots ,{r}_{2{m}_{q} - 1}$ ,Eq. (3) can be rewritten as

引理4的证明。由于对称性，只需证明不等式(13)即可。给定${r}_{1},\ldots ,{r}_{2{m}_{q} - 1}$，方程(3)可重写为

$$
\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)  = \mathop{\sum }\limits_{{i = 1}}^{{2{m}_{q} - 1}}{\int }_{x \in  {r}_{i}}q \cdot  {pdf}\left( x\right)  \cdot  \mathop{\Pr }\limits_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right) {dx}. \tag{31}
$$

When $x \in  {r}_{i},P{r}_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)  \leq  U{B}_{Pr}\left( {{r}_{i},o,{\varepsilon }_{q}}\right)$ (see Inequality (11)). Hence:

当$x \in  {r}_{i},P{r}_{\text{range }}\left( {o, \odot  \left( {x,{\varepsilon }_{q}}\right) }\right)  \leq  U{B}_{Pr}\left( {{r}_{i},o,{\varepsilon }_{q}}\right)$时（见不等式(11)）。因此：

$$
\mathop{\Pr }\limits_{\text{fuzzy }}\left( {o,q,{\varepsilon }_{q}}\right)  \leq  \mathop{\sum }\limits_{{i = 1}}^{{2{m}_{q} - 1}}\left( {U{B}_{\Pr }\left( {{r}_{i},o,{\varepsilon }_{q}}\right)  \cdot  {\int }_{x \in  {r}_{i}}q \cdot  {pdf}\left( x\right) {dx}}\right) . \tag{32}
$$

Furthermore:

此外：

$$
{\int }_{x \in  {r}_{i}}q \cdot  {pdf}\left( x\right) {dx} = \left\{  \begin{array}{ll} Q{C}_{i + 1} - Q{C}_{i} & \text{ if }i \in  \left\lbrack  {1,{m}_{q} - 1}\right\rbrack  \\  1 - {2Q}{C}_{{m}_{q}} & \text{ if }i = {m}_{q} \\  Q{C}_{2{m}_{q} - i + 1} - Q{C}_{2{m}_{q} - i} & \text{ if }i \in  \left\lbrack  {{m}_{q} + 1,2{m}_{q} - 1}\right\rbrack   \end{array}\right.  \tag{33}
$$

Substituting the above equation into Inequality (32), we arrive at Inequality (13).

将上述方程代入不等式(32)，我们得到不等式(13)。

## ACKNOWLEDGMENTS

## 致谢

The authors would like to thank the anonymous reviewers for their insightful comments.

作者们感谢匿名审稿人提出的富有洞察力的意见。

## REFERENCES

## 参考文献

BarвАгй, D., Garcia-Molina, H., and PorтER, D. 1992. The management of probabilistic data. IEEE Trans. Knowl. Data Eng. 4, 5, 487-502.

Beckhann, N., Kriegel, H.-P., Schneider, R., And Seeger, B. 1990. The R*-tree: An efficient and robust access method for points and rectangles. In Proceedings of ACM SIGMOD. ACM, New York. 322-331.

BERG, M., KREVELD, M., OVERMARS, M., AND SCHWARZKOPF, O. 2000. Computational Geometry: Algorithms and Applications. Springer-Verlag, New York.

Cheng, R., Kalashnikov, D., and PraвнАкаг, S. 2006a. The evaluation of probabilistic queries over imprecise data in constantly-evolving environments. Inf. Syst. 32, 1, 104-130.

Cheng, R., Kalashnikov, D. V., and Praвнакак, S. 2003. Evaluating probabilistic queries over imprecise data. In Proceedings of ACM SIGMOD. ACM, New York. 551-562.

Cheng, R., Kalashnikov, D. V., and PraвнАкак, S. 2004a. Querying imprecise data in moving object environments. IEEE Trans. Knowl. Data Eng. 16, 9, 1112-1127.

Cheng, R., Xia, Y., Praвнакаг, S., Sнан, R., and Virт𝙴к, J. S. 2004b. Efficient indexing methods for probabilistic threshold queries over uncertain data. In Proceedings of the Symposium on Very Large Databases. 876-887.

Cheng, R., Zhang, Y., Bertino, E., and Praabhakar, S. 2006b. Preserving user location privacy in mobile data management infrastructures. In Proceedings of the Privacy Enhancing Technology

Workshop (PET 2006) (Cambridge, UK, June). Lecture Notes in Computer Science. Springer-Verlag, New York, 393-412.

DALVI, N. AND SUCIU, D. 2005. Answering queries from statistics and probabilistic views. In Proceedings of the Symposium on Very Large Databases. 805-816.

DALVI, N. N. AND SUCIU, D. 2004. Efficient query evaluation on probabilistic databases. In Proceedings of the Symposium on Very Large Databases. 864-875.

DANIELS, K. L., MILENKOVIC, V. J., AND ROTH, D. 1997. Finding the largest area axis-parallel rectangle in a polygon. Comput. Geom. 7, 125-148.

DE ALMEIDA, V. T. AND GÜTING, R. H. 2005. Supporting uncertainty in moving objects in network databases. In Proceedings of the ACM International Symposium on Advances in Geographie Information Systems. ACM, New York, 31-40.

Deshpande, A., Guestrin, C., Madden, S., Hellerstein, J., and Hong, W. 2004. Model-driven data acquisition in sensor networks. In Proceedings of the Symposium on Very Large Databases. 588- 599.

FüHR, N. 1995. Probabilistic datalog - a logic for powerful retrieval methods. In SIGIR. 282-290.

GAEDE, V. AND GUNTHER, O. 1998. Multidimensional access methods. ACM Comput. Surv. 30, 2, 170-231.

GALINDO, J., URRUTIA, A., AND PIATTINI, M. 2006. Fuzzy Databases: Modeling, Design, and Implementation. Idea Group Publishing, ISBN: 1-59140-324-3.

Han, S., CнАN, E., CнENG, R., AND LAM, K. Y. 2007. A statistics-based sensor selection scheme for continuous probabilistic queries in sensor networks. Real-Time Syst. J. 35, 1, 33-58.

Hung, E., GEToor, L., AND Subrakhmanian, V. S. 2003. PXML: A probabilistic semistructured data model and algebra. In Proceedings of the IEEE International Conference on Data Engineering, IEEE Computer Society Press, Los Alamistos, CA, 467.

KhannA, S. AND TAN, W. 2001. On computing functions with uncertainty. In Proceedings of the 20th ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems, ACM, New York, 171-182.

KRIEGEL, H.-P., KUNATH, P., PfeIFLE, M., AND RENZ, M. 2006. Probabilistic similarity join on uncertain data. In Proceedings of the International Conference on Database Systems for Advanced Applications. 295-309.

LAKSHMANAN, L., LEONE, N., ROSS, R., AND SUBRAHMANIAN, V. 1997. Probview: A flexible probabilistic database system. Trans. Datab. Syst. 22, 3, 419-469.

LEE, S. K. 1992. An extended relational database model for uncertain and imprecise information. In Proceedings of the Conference on Very Large Databases. 211-220.

LEUTENEGER, S. T., EDGINGTON, J. M., AND LOPEZ, M. A. 1997. STR: A simple and efficient algorithm for r-tree packing. In Proceedings of the IEEE International Conference on Data Engineering, IEEE Computer Society Press, Los Alamitos, CA, 497-506.

LIM, E.-P., SRIVASTAVA, J., AND SHEKHAR, S. 1996. An evidential reasoning approach to attribute value conflict resolution in database integration. IEEE Trans. Knowl. Data Eng. 8, 5, 707-723.

Liu, K. AND SUNDERRAMAN, R. 1987. An extension to the relational model for indefinite databases. In Proceedings of the ACM-IEEE Computer Society Fall Joint Computer Conference. ACM, New York, 428-435.

LIU, K. AND SUNDERRAMAN, R. 1991. A generalized relational model for indefinite and maybe information. IEEE Trans. Knowl. Data Eng. 3, 1, 65-77.

Nierman, A. and JagabißH, H. V. 2002. ProTDB: Probabilistic data in XML. In Proceedings of the Conference on Very Large Databases. ACM, New York, 646-657.

OLSTON, C., Loo, B. T., AND WIDOM, J. 2001. Adaptive precision setting for cached approximate values. In Proceedings of the ACM SIGMOD Symposium. ACM, New York, 355-366.

OLSTON, C. AND WIDOM, J. 2000. Offering a precision-performance tradeoff for aggregation queries over replicated data. In Proceedings of the Conference on Very Large Databases. ACM, New York, ${144} - {155}$ .

OLSTON, C. AND WIDOM, J. 2002. Best-effort cache synchronization with source cooperation. In Proceedings of the ACM SIGMOD Symposium. ACM, New York, 73-84.

PageL, B.-U., Six, H.-W., Toßen, H., and WinмАуек, P. 1993. Towards an analysis of range query performance in spatial data structures. In Proceedings of the ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems. ACM, New York, 214-221.

Proser, D. AND Jensen, C. S. 1999. Capturing the uncertainty of moving-object representations. In Proceedings of the Symposium on Advances in Spatial Databases. Springer-Verlag, New York, 111-132.

Press, W. H., TEUKOLSKY, S. A., VETTERLING, W. T., AND FLANNERY, B. P. 2002. Numerical Recipes in $C +  +$ . Cambridge University Press,Cambridge,MA.

Sarma, A. D., Benjelloun, O., Winom, J., AND Halevy, A. 2006. Working models for uncertain data. In Proceedings of the IEEE International Conference on Data Engineering. IEEE Computer Society Press, Los Alamitos, CA.

SISTLA, A. P., WoLFSON, O., CHAMBERLAIN, S., AND DAO, S. 1997. Querying the uncertain position of moving objects. In Temporal Databases, Dagstuhl. 310-337.

Sweeney, L. 2002. k-anonymity: A model for protecting privacy. Int. J. Uncer. Fuzziness Knowl.- based Syst. 10, 5, 557-570.

Tao, Y., Cheng, R., Xiao, X., NgaI, W. K., Kao, B., and Prabakhaka, S. 2005. Indexing multidimensional uncertain data with arbitrary probability density functions. In Proceedings of the Symposium on Very Large Databases. 922-933.

Teixeira De. Almeida, V., and Güting, R. H. 2005. Supporting uncertainty in moving objects in network databases. In Proceedings of the GIS, 31-40.

THEODORIDIS, Y. AND SELLIS, T. K. 1996. A model for the prediction of R-tree performance. In Proceedings of the ACM SIGACT-SIGMOD SIGART Symposium on Principles of Database Systems. ACM, New York, 161-171.

Trajcevski, G., Wolfson, O., Hinrichs, K., and Chamberlain, S. 2004. Managing uncertainty in moving objects databases. Trans. Datab. Syst. 29, 3, 463-507.

WIDOM, J. 2005. Trio: A system for integrated management of data, accuracy, and lineage. In Proceedings of CIDR. 262-276.

WOLFSON, O., SISTLA, A. P., CHAMBERLAIN, S., AND YESHA, Y. 1999. Updating and querying databases that track mobile units. Distrib. Paral. Datab. 7, 3, 257-387.