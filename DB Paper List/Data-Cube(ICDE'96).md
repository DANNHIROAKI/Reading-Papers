# Data Cube: A Relational Aggregation Operator Generalizing Group-By, Cross-Tab, and Sub-Totals

# 数据立方体：一种关系聚合运算符，推广了分组、交叉表和小计功能

Jim Gray

吉姆·格雷

Adam Bosworth

亚当·博斯沃思

Andrew Layman

安德鲁·莱曼

Hamid Pirahesh

哈米德·皮拉赫什

Microsoft

微软

Microsoft

微软

Microsoft

微软

IBM

Gray@Microsoft.com

AdamB@Microsoft.com

AndrewL@Microsoft.com

Pirahesh@Almaden.IBM.com Abstract: Data analysis applications typically aggregate data across many dimensions looking for unusual patterns. The SQL aggregate functions and the GROUP BY operator produce zero-dimensional or one-dimensional answers. Applications need the N-dimensional generalization of these operators. This paper defines that operator, called the data cube or simply cube. The cube operator generalizes the histogram, cross-tabulation, roll-up, drill-down, and sub-total constructs found in most report writers. The cube treats each of the $N$ aggregation attributes as a dimension of $N$ -space. The aggregate of a particular set of attribute values is a point in this space. The set of points forms an $N$ -dimensional cube. Super-aggregates are computed by aggregating the $N$ -cube to lower dimensional spaces. Aggregation points are represented by an "infinite value", ALL, so the point (ALL, ALL, ..., ALL, sum(*)) represents the global sum of all items. Each ALL value actually represents the set of values contributing to that aggregation.

皮拉赫什@阿尔马登.ibm.com 摘要：数据分析应用程序通常会跨多个维度聚合数据，以寻找异常模式。SQL聚合函数和GROUP BY运算符只能产生零维或一维的结果。应用程序需要这些运算符的N维泛化。本文定义了这个运算符，称为数据立方体（data cube），简称立方体（cube）。立方体运算符推广了大多数报表生成器中常见的直方图、交叉表、汇总、钻取和小计结构。立方体将每个$N$聚合属性视为$N$空间的一个维度。特定属性值集合的聚合是这个空间中的一个点。这些点的集合形成一个$N$维立方体。超级聚合是通过将$N$立方体聚合到低维空间来计算的。聚合点用“无穷值”ALL表示，因此点(ALL, ALL, ..., ALL, sum(*))表示所有项的全局总和。每个ALL值实际上代表对该聚合有贡献的值的集合。

## 1. Introduction

## 1. 引言

Data analysis applications look for unusual patterns in data. They summarize data values, extract statistical information, and then contrast one category with another. There are two steps to such data analysis:

数据分析应用程序会在数据中寻找异常模式。它们总结数据值，提取统计信息，然后将一个类别与另一个类别进行对比。这种数据分析有两个步骤：

extracting the aggregated data from the database into a file or table, and

从数据库中提取聚合数据到文件或表中，以及

visualizing the results in a graphical way.

以图形方式可视化结果。

Visualization tools display trends, clusters, and differences. The most exciting work in data analysis focuses on presenting new graphical metaphors that allow people to discover data trends and anomalies. Many tools represent the dataset as an $N$ -dimensional space. Two and three-dimensional sub-slabs of this space are rendered as $2\mathrm{D}$ or 3D objects. Color and time (motion) add two more dimensions to the display giving the potential for a 5D display.

可视化工具可以显示趋势、聚类和差异。数据分析中最令人兴奋的工作集中在呈现新的图形隐喻，使人们能够发现数据趋势和异常。许多工具将数据集表示为一个$N$维空间。这个空间的二维和三维子块被渲染为$2\mathrm{D}$或三维对象。颜色和时间（运动）为显示增加了另外两个维度，从而有可能实现5D显示。

How do traditional relational databases fit into this picture? How can flat files (SQL tables) possibly model an $N$ - dimensional problem? Relational systems model $N$ - dimensional data as a relation with $N$ -attribute domains. For example, 4-dimensional earth-temperature data is typically represented by a Weather table shown below. The first four columns represent the four dimensions: $x,y,z,t$ . Additional columns represent measurements at the $4\mathrm{D}$

传统关系数据库如何适应这种情况呢？平面文件（SQL表）如何能够对$N$维问题进行建模呢？关系系统将$N$维数据建模为具有$N$个属性域的关系。例如，四维地球温度数据通常由下面所示的天气表表示。前四列代表四个维度：$x,y,z,t$。其他列代表在$4\mathrm{D}$处的测量值

<!-- Media -->

points such as temperature, pressure, humidity, and wind velocity. Often these measured values are aggregates over time (the hour) or space (a measurement area).

如温度、压力、湿度和风速等点的值。通常，这些测量值是随时间（小时）或空间（测量区域）的聚合值。

<table><tr><td colspan="6">Table 1: Weather</td></tr><tr><td>Time (UCT)</td><td>Latitude</td><td>Longitude</td><td>Altitude (m)</td><td>Temp (c)</td><td>Pres (mb)</td></tr><tr><td/><td>37:58:33N</td><td>122:45:28W</td><td>102</td><td>21</td><td>1009</td></tr><tr><td>27/11/94:150034:16:18N</td><td/><td>27:05:55w</td><td>10</td><td>23</td><td>1024</td></tr></table>

<table><tbody><tr><td colspan="6">表1：天气</td></tr><tr><td>时间（协调世界时）</td><td>纬度</td><td>经度</td><td>海拔（米）</td><td>温度（摄氏度）</td><td>气压（毫巴）</td></tr><tr><td></td><td>37:58:33N</td><td>122:45:28W</td><td>102</td><td>21</td><td>1009</td></tr><tr><td>27/11/94:150034:16:18N</td><td></td><td>27:05:55w</td><td>10</td><td>23</td><td>1024</td></tr></tbody></table>

<!-- Media -->

The SQL standard provides five functions to aggregate the values in a table: COUNT (   ), SUM (   ), MIN (   ), MAX (   ), and AVG (   ). For example, the average of all measured temperatures is expressed as:

SQL标准提供了五个函数来聚合表中的值：COUNT ( )、SUM ( )、MIN ( )、MAX ( ) 和 AVG ( )。例如，所有测量温度的平均值表示为：

SELECT AVG (Temp)

SELECT AVG (Temp)

FROM Weather;

FROM Weather;

In addition, SQL allows aggregation over distinct values. The following query counts the distinct number of reporting times in the Weather table:

此外，SQL允许对不同的值进行聚合。以下查询统计了Weather表中不同的报告时间数量：

SELECT COUNT (DISTINCT Time)

SELECT COUNT (DISTINCT Time)

FROM Weather;

FROM Weather;

Many SQL systems add statistical functions (median, standard deviation, variance, etc.), physical functions (center of mass, angular momentum, etc.), financial analysis (volatility, Alpha, Beta, etc.), and other domain-specific functions.

许多SQL系统添加了统计函数（中位数、标准差、方差等）、物理函数（质心、角动量等）、财务分析函数（波动率、阿尔法、贝塔等）以及其他特定领域的函数。

Some systems allow users to add new aggregation functions. The Illustra system, for example, allows users to add aggregate functions by adding a program with the following three callbacks to the database system [Illustra]:

一些系统允许用户添加新的聚合函数。例如，Illustra系统允许用户通过向数据库系统添加一个包含以下三个回调的程序来添加聚合函数 [Illustra]：

Init (&handle): Allocates the handle and initializes the aggregate computation.

Init (&handle)：分配句柄并初始化聚合计算。

Iter ( &handle, value): Aggregates the next value into the current aggregate.

Iter ( &handle, value)：将下一个值聚合到当前聚合结果中。

value = Final (&handle) : Computes and returns the resulting aggregate by using data saved in the handle. This invocation deallocates the handle.

value = Final (&handle) ：使用句柄中保存的数据计算并返回最终的聚合结果。此调用会释放句柄。

Consider implementing the Average (   ) function. The handle stores the count and the sum initialized to zero. When passed a new non-null value, Iter (   ) increments the count and adds the sum to the value. The Final (   ) call deallocates the handle and returns sum divided by count.

考虑实现Average ( ) 函数。句柄存储计数和初始化为零的总和。当传入一个新的非空值时，Iter ( ) 会增加计数并将该值累加到总和中。Final ( ) 调用会释放句柄并返回总和除以计数的结果。

Aggregate functions return a single value. Using the GROUP BY construct, SQL can also create a table of many aggregate values indexed by a set of attributes. For example, The following query reports the average temperature for each reporting time and altitude:

聚合函数返回单个值。使用GROUP BY结构，SQL还可以创建一个由一组属性索引的多个聚合值的表。例如，以下查询报告了每个报告时间和海拔高度的平均温度：

SELECT Time, Altitude, AVG (Temp)

SELECT Time, Altitude, AVG (Temp)

FROM Weather

FROM Weather

GROUP BY Time, Altitude;

GROUP BY Time, Altitude;

GROUP BY is an unusual relational operator: It partitions the relation into disjoint tuple sets and then aggregates over each set as illustrated in Figure 1.

分组（GROUP BY）是一种特殊的关系运算符：它将关系划分为不相交的元组集合，然后对每个集合进行聚合操作，如图1所示。

<!-- Media -->

<!-- figureText: Grouping Values Aggregate Values Sum(   ) Partitioned Table -->

<img src="https://cdn.noedgeai.com/0195c917-ccec-7034-a8ba-1657644e94c3_1.jpg?x=206&y=629&w=649&h=274&r=0"/>

Figure 1: The GROUP BY relational operator partitions a table into groups. Each group is then aggregated by a function. The aggregation function summarizes some column of groups returning a value for each group.

图1：分组（GROUP BY）关系运算符将表划分为多个组。然后通过一个函数对每个组进行聚合操作。聚合函数对组的某一列进行汇总，为每个组返回一个值。

<!-- Media -->

Red Brick systems added some interesting aggregate functions that enhance the GROUP BY mechanism [Red Brick]:

红杉系统（Red Brick）添加了一些有趣的聚合函数，这些函数增强了分组（GROUP BY）机制 [红杉（Red Brick）]：

Rank (expression): returns the expression's rank in the set of all values of this domain of the table. If there are $N$ values in the column,and this is the highest value, the rank is $N$ ,if it is the lowest value the rank is 1 .

排名（Rank）(表达式)：返回该表达式在表的此域的所有值集合中的排名。如果列中有$N$个值，并且这是最高值，则排名为$N$；如果这是最低值，则排名为1。

N_tile (expression, n): The range of the expression (over all the input values of the table) is computed and divided into $n$ value ranges of approximately equal population. The function returns the number of the range holding the value of the expression. If your bank account was among the largest ${10}\%$ then your rank (account.balance,10) would return 10 . Red Brick provides just N_tile (expression,3).

分位（N_tile）(表达式, n)：计算表达式的范围（在表的所有输入值上），并将其划分为$n$个值范围，每个范围的数量大致相等。该函数返回包含表达式值的范围编号。如果你的银行账户余额处于最大的${10}\%$个之中，那么你的排名（账户.余额,10）将返回10。红杉系统（Red Brick）仅提供分位（N_tile）(表达式,3)。

Ratio_To_Total (expression): Sums all the expressions and then divides the expression by the total sum.

占比（Ratio_To_Total）(表达式)：对所有表达式求和，然后将该表达式除以总和。

## To give an example:

## 举个例子：

SELECT Percentile, MIN (Temp), MAX (Temp)

SELECT 百分位, MIN (温度)

FROM Weather

FROM 天气

GROUP BY N tile (Temp, 10) as Percentile

GROUP BY 分位（温度, 10） AS 百分位

HAVING Percentile $= 5$ ;

HAVING 百分位 $= 5$ ;

returns one row giving the minimum and maximum temperatures of the middle 10% of all temperatures. As mentioned later, allowing function values in the GROUP BY is not yet allowed by the SQL standard.

返回一行，给出所有温度中中间10%的最低和最高温度。如后文所述，SQL标准目前尚不允许在分组（GROUP BY）中使用函数值。

Red Brick also offers three cumulative aggregates that operate on ordered tables.

红杉系统（Red Brick）还提供了三种对有序表进行操作的累积聚合函数。

Cumulative (expression): Sums all values so far in an ordered list.

累积（Cumulative）(表达式)：对有序列表中到目前为止的所有值求和。

Running_Sum (expression, $\mathrm{n}$ ) : Sums the most recent $\mathrm{n}$ values in an ordered list. The initial $n - 1$ values are NULL.

滚动求和（Running_Sum）(表达式, $\mathrm{n}$ )：对有序列表中最近的$\mathrm{n}$个值求和。最初的$n - 1$个值为NULL。

Running_Average (expression,n) : Averages the most recent $n$ values in an ordered list. The initial $n - 1$ values are null.

滚动平均（Running_Average）(表达式,n)：对有序列表中最近的$n$个值求平均值。最初的$n - 1$个值为NULL。

These aggregate functions are optionally reset each time a grouping value changes in an ordered selection.

在有序选择中，每当分组值发生变化时，这些聚合函数可选择重置。

## 2. Problems With GROUP BY:

## 2. GROUP BY的问题：

SQL's aggregation functions are widely used. In the spirit of aggregating data, Table 2 shows how frequently the database and transaction processing benchmarks use aggregation and GROUP BY. Surprisingly, aggregates also appear in the online-transaction processing TPC-C query set. Paradoxically, the TPC-A and TPC-B benchmark transactions spend most of their energies maintaining aggregates dynamically: they maintain the summary bank account balance, teller cash-drawer balance, and branch balance. All these can be computed as aggregates from the history table [TPC].

SQL的聚合函数被广泛使用。本着聚合数据的精神，表2展示了数据库和事务处理基准测试使用聚合和GROUP BY的频率。令人惊讶的是，聚合也出现在在线事务处理TPC - C查询集中。矛盾的是，TPC - A和TPC - B基准测试事务大部分精力都花在动态维护聚合上：它们维护银行账户汇总余额、柜员现金抽屉余额和分行余额。所有这些都可以从历史表中作为聚合计算得出[TPC]。

<!-- Media -->

<table><tr><td colspan="4">Table 2: SQL Aggregates in Standard Benchmarks</td></tr><tr><td>Benchmark</td><td>Queries</td><td>Aggregates</td><td>GROUP BYs</td></tr><tr><td>TPC-A, B</td><td>1</td><td>0</td><td>0</td></tr><tr><td>TPC-C</td><td>18</td><td>4</td><td>0</td></tr><tr><td>TPC-D</td><td>16</td><td>27</td><td>15</td></tr><tr><td>Wisconsin</td><td>18</td><td>3</td><td>2</td></tr><tr><td>${\mathrm{{AS}}}^{3}\mathrm{{AP}}$</td><td>23</td><td>20</td><td>2</td></tr><tr><td>SetQuery</td><td>7</td><td>5</td><td>1</td></tr></table>

<table><tbody><tr><td colspan="4">表2：标准基准测试中的SQL聚合函数</td></tr><tr><td>基准测试</td><td>查询</td><td>聚合函数</td><td>分组（GROUP BY）</td></tr><tr><td>TPC - A、B</td><td>1</td><td>0</td><td>0</td></tr><tr><td>TPC - C</td><td>18</td><td>4</td><td>0</td></tr><tr><td>TPC - D</td><td>16</td><td>27</td><td>15</td></tr><tr><td>威斯康星（Wisconsin）</td><td>18</td><td>3</td><td>2</td></tr><tr><td>${\mathrm{{AS}}}^{3}\mathrm{{AP}}$</td><td>23</td><td>20</td><td>2</td></tr><tr><td>集合查询（SetQuery）</td><td>7</td><td>5</td><td>1</td></tr></tbody></table>

<!-- Media -->

The TPC-D query set has one 6D GROUP BY and three 3D GROUP BYS. 1D and 2D GROUP BYS are most common.

TPC - D查询集有一个六维分组（6D GROUP BY）和三个三维分组（3D GROUP BY）。一维分组（1D GROUP BY）和二维分组（2D GROUP BY）最为常见。

Certain forms of data analysis are difficult if not impossible with the SQL constructs. As explained next, three common problems are: (1) Histograms, (2) Roll-up Totals and Sub-Totals for drill-downs, (3) Cross Tabulations.

使用SQL结构进行某些形式的数据分析即便不是不可能，也是很困难的。接下来将解释，三个常见的问题是：（1）直方图；（2）用于向下钻取的汇总和小计；（3）交叉表。

The standard SQL GROUP BY operator does not allow a direct construction of histograms (aggregation over computed categories.) For example, for queries based on the Weather table, it would be nice to be able to group times into days, weeks, or months, and to group locations into areas (e.g., US, Canada, Europe,...). This would be easy if function values were allowed in the GROUP BY list. If that were allowed, the following query would give the daily

标准SQL的分组（GROUP BY）运算符不允许直接构建直方图（对计算得出的类别进行聚合）。例如，对于基于天气表的查询，如果能够将时间按天、周或月分组，将地点按区域（例如，美国、加拿大、欧洲等）分组，那就再好不过了。如果分组列表中允许使用函数值，这将很容易实现。如果允许这样做，以下查询将给出每日

maximum reported temperature.

报告的最高温度。

SELECT day, nation, MAX (Temp)

选择日期、国家和最高温度（Temp）

FROM Weather

从天气表中

GROUP BY Day(Time) AS day,

按时间的日期（Day(Time)）分组为日期

Country (Latitude, Longitude)

按纬度和经度确定的国家（Country (Latitude, Longitude)）

AS nation;

作为国家；

Some SQL systems support histograms but the standard does not. Rather, one must construct a table-valued expression and then aggregate over the resulting table. The following statement demonstrates this SQL92 construct.

一些SQL系统支持直方图，但标准SQL并不支持。相反，必须构造一个表值表达式，然后对结果表进行聚合。以下语句展示了这种SQL92结构。

SELECT day, nation, MAX(Temp)

选择日期、国家和最高温度（Temp）

FROM ( SELECT Day (Time) AS day,

从（选择时间的日期（Day (Time)）作为日期

Country ( Latitude, Longitude )

按纬度和经度确定的国家（Country ( Latitude, Longitude )）

AS nation,

作为国家

Temp

温度（Temp）

FROM Weather ) AS foo

从天气表）作为临时表foo

GROUP BY day, nation;

按日期和国家分组;

A second problem relates to roll-ups using totals and subtotals for drill-down reports. Reports commonly aggregate data at a coarse level, and then at successively finer levels. The car sales report in Table 3 shows the idea. Data is aggregated by Model, then by Year, then by Color. The report shows data aggregated at three levels. Going up the levels is called rolling-up the data. Going down is called drilling-down into the data.

第二个问题与使用总计和小计进行向下钻取报告的汇总操作有关。报告通常先在较粗的粒度级别汇总数据，然后再在更细的级别进行汇总。表3中的汽车销售报告展示了这一概念。数据先按车型（Model）汇总，然后按年份（Year）汇总，最后按颜色（Color）汇总。该报告展示了三个级别的数据汇总。向上汇总数据称为数据上卷（rolling-up the data），向下查看数据称为数据下钻（drilling-down into the data）。

<!-- Media -->

Table 3: Sales Roll Up by Model by Year by Color

表3：按车型、年份和颜色的销售数据汇总

<table><tr><td>Model</td><td>Year</td><td>Color</td><td>Sales by Model by Year by Color</td><td>Sales by Model by Year</td><td>Sales by Model</td></tr><tr><td>Chevv</td><td>1994</td><td>black</td><td>50</td><td/><td/></tr><tr><td/><td/><td>white</td><td>40</td><td/><td/></tr><tr><td/><td/><td/><td/><td>90</td><td/></tr><tr><td/><td>1995</td><td>black</td><td>85</td><td/><td/></tr><tr><td/><td/><td>white</td><td>115</td><td/><td/></tr><tr><td/><td/><td/><td/><td>200</td><td/></tr><tr><td/><td/><td/><td/><td/><td>290</td></tr></table>

<table><tbody><tr><td>型号</td><td>年份</td><td>颜色</td><td>按型号、年份和颜色划分的销量</td><td>按型号和年份划分的销量</td><td>按型号划分的销量</td></tr><tr><td>雪佛兰（Chevv）</td><td>1994</td><td>黑色</td><td>50</td><td></td><td></td></tr><tr><td></td><td></td><td>白色</td><td>40</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>90</td><td></td></tr><tr><td></td><td>1995</td><td>黑色</td><td>85</td><td></td><td></td></tr><tr><td></td><td></td><td>白色</td><td>115</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>200</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>290</td></tr></tbody></table>

<!-- Media -->

Table 3 is not relational -null values in the primary key are not allowed. It is also not convenient - the number of columns grows as the power set of the number of aggregated attributes. Table 4 is a relational and more convenient representation. The dummy value "ALL" has been added to fill in the super-aggregation items. The symmetric aggregation result is a table called a cross-tabulation, or cross tab for short (spreadsheets and some desktop databases call them pivot tables.) Cross tab data is routinely displayed in the more compact format of Table 5. This cross tab is a two-dimensional aggregation. If other automobile models are added,it becomes a $3\mathrm{D}$ aggregation. For example, data for Ford products adds an additional cross tab plane as in Table 5.a.

表3不具有关系性——主键中不允许出现空值。它也不方便——列的数量会随着聚合属性数量的幂集增长。表4是一种更具关系性且更方便的表示方式。已添加虚拟值“ALL”来填充超级聚合项。对称聚合结果是一个称为交叉表（cross-tabulation，简称交叉表；电子表格和一些桌面数据库称它们为数据透视表）的表。交叉表数据通常以表5更紧凑的格式显示。这个交叉表是二维聚合。如果添加其他汽车型号，它就变成了$3\mathrm{D}$聚合。例如，福特产品的数据会像表5.a那样增加一个额外的交叉表平面。

<!-- Media -->

<table><tr><td colspan="4">Table 5: Chevy Sales Cross Tab</td></tr><tr><td>Chevy</td><td>1994</td><td>1995</td><td>total (ALL)</td></tr><tr><td>black</td><td>50</td><td>85</td><td>135</td></tr><tr><td>white</td><td>40</td><td>115</td><td>155</td></tr><tr><td>total (ALL)</td><td>90</td><td>200</td><td>290</td></tr></table>

<table><tbody><tr><td colspan="4">表5：雪佛兰汽车销售交叉表</td></tr><tr><td>雪佛兰（Chevy）</td><td>1994</td><td>1995</td><td>总计（ALL）</td></tr><tr><td>黑色</td><td>50</td><td>85</td><td>135</td></tr><tr><td>白色</td><td>40</td><td>115</td><td>155</td></tr><tr><td>总计（ALL）</td><td>90</td><td>200</td><td>290</td></tr></tbody></table>

<table><tr><td colspan="4">Table 4: Sales Summary</td></tr><tr><td>Model</td><td>Year</td><td>Color</td><td>Units</td></tr><tr><td>Chevy</td><td>1994</td><td>black</td><td>50</td></tr><tr><td>Chevy</td><td>1994</td><td>white</td><td>40</td></tr><tr><td>Chevy</td><td>1994</td><td>ALL</td><td>90</td></tr><tr><td>Chevy</td><td>1995</td><td>black</td><td>85</td></tr><tr><td>Chevy</td><td>1995</td><td>white</td><td>115</td></tr><tr><td>Chevy</td><td>1995</td><td>ALL</td><td>200</td></tr><tr><td>Chevy</td><td>ALL</td><td>ALL</td><td>290</td></tr></table>

<table><tbody><tr><td colspan="4">表4：销售总结</td></tr><tr><td>型号</td><td>年份</td><td>颜色</td><td>数量</td></tr><tr><td>雪佛兰（Chevy）</td><td>1994</td><td>黑色</td><td>50</td></tr><tr><td>雪佛兰（Chevy）</td><td>1994</td><td>白色</td><td>40</td></tr><tr><td>雪佛兰（Chevy）</td><td>1994</td><td>全部</td><td>90</td></tr><tr><td>雪佛兰（Chevy）</td><td>1995</td><td>黑色</td><td>85</td></tr><tr><td>雪佛兰（Chevy）</td><td>1995</td><td>白色</td><td>115</td></tr><tr><td>雪佛兰（Chevy）</td><td>1995</td><td>全部</td><td>200</td></tr><tr><td>雪佛兰（Chevy）</td><td>全部</td><td>全部</td><td>290</td></tr></tbody></table>

The SQL statement to build this SalesSummary table from the raw Sales data is:

从原始销售数据构建此销售汇总表的SQL语句如下：

SELECT Model, ALL, ALL, SUM(Sales)

SELECT 车型, ALL, ALL, SUM(销售额)

FROM Sales

FROM 销售表

WHERE Model = 'Chevy'

WHERE 车型 = '雪佛兰'

GROUP BY Model

GROUP BY 车型

UNION

SELECT Model, Year, ALL, SUM(Sales)

SELECT 车型, 年份, ALL, SUM(销售额)

FROM Sales

FROM 销售表

WHERE Model = 'Chevy'

WHERE 车型 = '雪佛兰'

GROUP BY Model, Year

GROUP BY 车型, 年份

UNION

SELECT Model, Year, Color, SUM(Sales)

SELECT 车型, 年份, 颜色, SUM(销售额)

FROM Sales

FROM 销售表

WHERE Model = 'Chevy'

WHERE 车型 = '雪佛兰'

GROUP BY Model, Year, Color;

GROUP BY 车型, 年份, 颜色;

This is a simple 3-dimensional roll-up. Aggregating over $N$ dimensions requires $N$ such unions.

这是一个简单的三维上卷操作。对 $N$ 个维度进行聚合需要 $N$ 个这样的并集操作。

Roll-up is asymmetric - notice that the table above does not aggregate the sales by year. It lacks the rows aggregating sales by color rather than by year. These rows are:

上卷操作是非对称的 —— 注意，上面的表没有按年份对销售额进行聚合。它缺少按颜色而非按年份聚合销售额的行。这些行如下：

<table><tr><td>Model</td><td>Year</td><td>Color</td><td>Units</td></tr><tr><td>Chevy</td><td>ALL</td><td>black</td><td>135</td></tr><tr><td>Chevy</td><td>ALL</td><td>white</td><td>155</td></tr></table>

<table><tbody><tr><td>型号</td><td>年份</td><td>颜色</td><td>数量</td></tr><tr><td>雪佛兰（Chevy）</td><td>全部</td><td>黑色</td><td>135</td></tr><tr><td>雪佛兰（Chevy）</td><td>全部</td><td>白色</td><td>155</td></tr></tbody></table>

These additional rows could be captured by adding the following clause to the SQL statement above: UNION SELECT Model, ALL, Color, SUM(Sales) FROM Sales WHERE Model = 'Chevy' GROUP BY Model, Color;

可以通过在上述SQL语句中添加以下子句来捕获这些额外的行：UNION SELECT Model, ALL, Color, SUM(Sales) FROM Sales WHERE Model = 'Chevy' GROUP BY Model, Color;

<table><tr><td colspan="4">Table 5a: Ford Sales Cross Tab</td></tr><tr><td>Ford</td><td>1994</td><td>1995</td><td>total (ALL)</td></tr><tr><td>black</td><td>50</td><td>85</td><td>135</td></tr><tr><td>white</td><td>10</td><td>75</td><td>85</td></tr><tr><td>total (ALL)</td><td>60</td><td>160</td><td>220</td></tr></table>

<table><tbody><tr><td colspan="4">表5a：福特汽车销售交叉表</td></tr><tr><td>福特</td><td>1994</td><td>1995</td><td>总计（全部）</td></tr><tr><td>黑色</td><td>50</td><td>85</td><td>135</td></tr><tr><td>白色</td><td>10</td><td>75</td><td>85</td></tr><tr><td>总计（全部）</td><td>60</td><td>160</td><td>220</td></tr></tbody></table>

<!-- Media -->

The cross tab array representation is equivalent to the relational representation using the ALL value. Both generalize to an $N$ -dimensional cross tab.

交叉表数组表示法等同于使用ALL值的关系表示法。两者都可推广到$N$维交叉表。

The representation of Table 4 and unioned GROUP BYS "solve" the problem of representing aggregate data in a relational data model. The problem remains that expressing histogram, roll-up, drill-down, and cross-tab queries with conventional SQL is daunting. A 6D cross-tab requires a 64-way union of 64 different GROUP BY operators to build the underlying representation. Incidentally, on most SQL systems this will result in 64 scans of the data, 64 sorts or hashes, and a long wait.

表4的表示法和联合GROUP BY操作“解决”了在关系数据模型中表示聚合数据的问题。但问题仍然存在，即使用传统SQL表达直方图、汇总、钻取和交叉表查询非常困难。一个6维交叉表需要64个不同的GROUP BY运算符进行64路联合来构建底层表示。顺便说一下，在大多数SQL系统中，这将导致对数据进行64次扫描、64次排序或哈希操作，并且需要长时间等待。

Building a cross-tabulation with SQL is even more daunting since the result is not a really a relational object - the bottom row and the right column are "unusual". Most report writers build in a cross-tabs feature, building the report up from the underlying tabular data such as Table 4 and its extension. See for example the TRANSFORM-PIVOT operator of Microsoft Access [Access].

使用SQL构建交叉表更加困难，因为结果实际上并不是一个真正的关系对象——底行和右列是“特殊的”。大多数报表生成器都内置了交叉表功能，从底层的表格数据（如表4及其扩展）构建报表。例如，可参考Microsoft Access的TRANSFORM - PIVOT运算符[Access]。

### 3.The Data CUBE Operator

### 3. 数据立方体（Data CUBE）运算符

The generalization of these ideas seems obvious: Figure 2 shows the concept for aggregation up to 3-dimensions. The traditional GROUP BY can generate the core of the $N$ - dimensional data cube. The $N - 1$ lower-dimensional aggregates appear as points, lines, planes, cubes, or hyper-cubes hanging off the core data cube.

这些概念的推广似乎很明显：图2展示了最多3维聚合的概念。传统的GROUP BY可以生成$N$维数据立方体的核心。$N - 1$低维聚合以点、线、面、立方体或超立方体的形式依附于核心数据立方体。

The data cube operator builds a table containing all these aggregate values. The total aggregate is represented as the tuple: ALL, ALL, ALL, ..., ALL, f(*)

数据立方体运算符构建一个包含所有这些聚合值的表。总聚合表示为元组：ALL, ALL, ALL, ..., ALL, f(*)

Points in higher dimensional planes or cubes have fewer ALL values. Figure 3 illustrates this idea with an example.

高维平面或立方体中的点具有较少的ALL值。图3通过一个示例说明了这一概念。

We extend SQL's SELECT-GROUP-BY-HAVING syntax to support histograms, decorations, and the CUBE operator. Currently the SQL GROUP BY syntax is: GROUP BY \{<column name> [<collate clause>],...\} To support histograms, extend the syntax to: GROUP BY

我们扩展了SQL的SELECT - GROUP BY - HAVING语法，以支持直方图、修饰列和CUBE运算符。目前SQL的GROUP BY语法是：GROUP BY {<列名> [<排序子句>],...} 为了支持直方图，将语法扩展为：GROUP BY

\{ ( <column name> | <expression>)

{ ( <列名> | <表达式>)

[ AS <correlation name>

[ AS <关联名>

[ <collate clause>

[ <排序子句>

, ... \}

<!-- Media -->

<!-- figureText: Aggregate By Color The Data Cube and The Sub-Space Aggregates By Yea By Make By Color & Year By Make & Color Sum By Color Sum Group By (with total) By Color RED WHITE BLUE Sum Cross Tab Chevy Ford RED WHITE BLUE By Make Sum By Make & Year -->

<img src="https://cdn.noedgeai.com/0195c917-ccec-7034-a8ba-1657644e94c3_3.jpg?x=917&y=212&w=636&h=596&r=0"/>

Figure 2: The CUBE operator is the $N$ -dimensional generalization of simple aggregate functions. The 0D data cube is a point. The ID data cube is a line with a point. The 2D data cube is a cross tab, a plane, two lines, and a point. The 3D data cube is a cube with three intersecting 2D cross tabs.

图2：CUBE运算符是简单聚合函数的$N$维推广。0维数据立方体是一个点。1维数据立方体是带有一个点的线。2维数据立方体是一个交叉表、一个平面、两条线和一个点。3维数据立方体是一个带有三个相交2维交叉表的立方体。

<!-- Media -->

The next step is to allow decorations, columns that do not appear in the GROUP BY but that are functionally dependent on the grouping columns. Consider the example: SELECT department.name, sum (sales) FROM sales JOIN department USING (department_number) GROUP BY sales.department_number;

下一步是允许使用修饰列，即那些不出现在GROUP BY中但在功能上依赖于分组列的列。考虑以下示例：SELECT department.name, sum (sales) FROM sales JOIN department USING (department_number) GROUP BY sales.department_number;

The department . name column in the answer set is not allowed in current SQL, it is neither an aggregation column (appearing in the GROUP BY list) nor is it an aggregate. It is just there to decorate the answer set with the name of the department. We recommend the rule that if $a$ decoration column (or column value) is functionally dependent on the aggregation columns, then it may be included in the SELECT answer list.

在当前SQL中，结果集中的department.name列是不允许的，它既不是聚合列（出现在GROUP BY列表中），也不是聚合函数。它只是用于用部门名称修饰结果集。我们建议，如果$a$修饰列（或列值）在功能上依赖于聚合列，那么它可以包含在SELECT结果列表中。

These extensions are independent of the CUBE operator. They remedy some pre-existing problems with GROUP BY. Some systems already allow these extensions, for example Microsoft Access allows function-valued GROUP BYs.

这些扩展与CUBE运算符无关。它们解决了GROUP BY之前存在的一些问题。一些系统已经允许这些扩展，例如Microsoft Access允许使用函数值的GROUP BY。

Creating the CUBE requires generating the power set (set of all subsets) of the aggregation columns. We propose the following syntax to extend SQL's GROUP BY operator:

创建CUBE需要生成聚合列的幂集（所有子集的集合）。我们提出以下语法来扩展SQL的GROUP BY运算符：

GROUP BY

---

\{ ( <column name> | <expression>)

{ ( <列名> | <表达式>)

	[ AS <correlation name> ]

	[ 作为 <关联名> ]

	[ <collate clause> ]

	[ <排序子句> ]

...\}

	[ WITH ( CUBE | ROLLUP ) ]

	[ 使用 ( CUBE | ROLLUP ) ]

---

<!-- Media -->

<!-- figureText: FROM Saies MANKE blue Chevy 1990 Chevy 1990 ALL 154 Chevy 1991 Chevy 1991 1 ed 54 Chevi 1,001 Chevi 1992 blue 71 100 Chevy 1990 white 54 Chevr, 1992 Chevy ALL 1 ed 90 Chavy ALL Chevi ALL ALL 508 blue Fouri 1990 Foi-1 1990 white -60 Fould 1 ""1 blue 55 Found 1991 52 Ford 1.59% ALL 116 Ford 1990 For a 1000 tod 27 Foid 1992 128 Foid ALL blue 157 ted Foid ALL white 133 For a 4.3.3 [ALL 1990 Led 69 IALL 1990 14.9 [AL]. 1990 ALL 343 fall 1 ** 104 ALL 1991 white 110 ALL 1990 blue 110 1990 white ALL (100) ALL [8] ALL ALL 3,3 ALL Ali i et 233 WHERE Model in \{'Ford', 'Chevy'\} GROUP BY Model, Yeal, Color WITH CUBE; SALES Model Year Color Sales Chovy 1990 which is CUBE Chevy 1990 blue Chevy 1991 Chevy 1990 blue Chevy 1992 white 54 Chev, 1992 blue Foid 1990 t 6 d Found 1990 which is 62 Foid 1991 1 test Foid 1991 which 9 Fot 4 1992 1 ed Fot d 1992 white -->

<img src="https://cdn.noedgeai.com/0195c917-ccec-7034-a8ba-1657644e94c3_4.jpg?x=238&y=229&w=639&h=699&r=0"/>

Figure 3: A 3D data cube (right) built from the table at the left by the CUBE statement at the top of the figure.

图3：通过图顶部的CUBE语句从左侧的表构建的三维数据立方体（右侧）。

Figure 3 has an example of this syntax. To give another, here follows a statement to aggregate the set of temperature observations: SELECT day, nation, MAX (Temp) FROM Weather GROUP BY Day(Time) AS day, Country (Latitude, Longitude) AS nation WITH CUBE;

图3给出了此语法的一个示例。再举一个例子，以下是一条用于聚合温度观测值集合的语句：SELECT day, nation, MAX (Temp) FROM Weather GROUP BY Day(Time) AS day, Country (Latitude, Longitude) AS nation WITH CUBE;

<!-- Media -->

The semantics of the CUBE operator are that it first aggregates over all the <select list> attributes as in a standard GROUP BY. Then, it UNIONs in each super-aggregate of the global cube - substituting ALL for the aggregation columns. If there are $N$ attributes in the select list,there will be ${2}^{N} - 1$ super-aggregate values. If the cardinality of the $N$ attributes are ${C}_{1},{C}_{2},\ldots ,{C}_{N}$ then the cardinality of the resulting cube relation is $\Pi \left( {{C}_{i} + 1}\right)$ . The extra value in each domain is ALL. For example, the SALES table has $2\mathrm{x}3\mathrm{x}3 = {18}$ rows,while the derived data cube has $3\mathrm{x}4\mathrm{x}4 =$ 48 rows.

CUBE运算符的语义是，它首先像标准的GROUP BY一样对所有<选择列表>属性进行聚合。然后，它将全局立方体的每个超级聚合结果进行UNION操作——用ALL替换聚合列。如果选择列表中有$N$个属性，将会有${2}^{N} - 1$个超级聚合值。如果$N$个属性的基数为${C}_{1},{C}_{2},\ldots ,{C}_{N}$，那么生成的立方体关系的基数为$\Pi \left( {{C}_{i} + 1}\right)$。每个域中的额外值为ALL。例如，SALES表有$2\mathrm{x}3\mathrm{x}3 = {18}$行，而派生的数据立方体有$3\mathrm{x}4\mathrm{x}4 =$ 48行。

Each ALL value really represents a set - the set over which the aggregate was computed. In the SalesSummary table the respective sets are:

每个ALL值实际上代表一个集合——进行聚合计算所基于的集合。在SalesSummary表中，相应的集合如下：

---

Model.ALL = ALL (Model) = \{Chevy, Ford \}

Model.ALL = ALL (Model) = {雪佛兰, 福特}

	Year. ALL $=$ ALL (Year) $= \{ {1990},{1991},{1992}\}$

	Year. ALL $=$ ALL (Year) $= \{ {1990},{1991},{1992}\}$

	Color.ALL = ALL(Color) = \{red,white,blue\}

	Color.ALL = ALL(Color) = {红色, 白色, 蓝色}

---

Thinking of the ALL value as a token representing these sets defines the semantics of the relational operators (e.g., equals and IN). The ALL string is for display. A new ALL (   ) function generates the set associated with this value as in the examples above. ALL (   ) applied to any other value returns NULL. This design is eased by SQL3's support for set-valued variables and domains.

将ALL值视为代表这些集合的标记，就定义了关系运算符（例如，等于和IN）的语义。ALL字符串用于显示。一个新的ALL ( )函数会生成与该值关联的集合，如上述示例所示。将ALL ( )应用于任何其他值将返回NULL。SQL3对集合值变量和域的支持简化了这种设计。

The ALL value appears to be essential, but creates substantial complexity. It is a non-value, like NULL. We do not add it lightly - adding it touches many aspects of the SQL language. To name a few:

ALL值似乎是必不可少的，但会带来很大的复杂性。它是一个非值，类似于NULL。我们不会轻易添加它——添加它会涉及SQL语言的许多方面。仅举几例：

- Treating each ALL value as the set of aggregates guides the meaning of the ALL value.

- 将每个ALL值视为聚合集合有助于理解ALL值的含义。

- ALL becomes a new keyword denoting the set value.

- ALL成为一个表示集合值的新关键字。

- ALL [NOT] ALLOWED is added to the column definition syntax and to the column attributes in the system catalogs.

- ALL [NOT] ALLOWED被添加到列定义语法和系统目录中的列属性中。

- ALL, like NULL, does not participate in any aggregate except COUNT (   ) .

- 与 NULL 一样，ALL 不参与除 COUNT (   ) 之外的任何聚合运算。

- The set interpretation guides the meaning of the relational operators $\{  = , < , <  = , = , >  = , > ,\mathrm{{IN}}\}$ .

- 集合解释指导着关系运算符 $\{  = , < , <  = , = , >  = , > ,\mathrm{{IN}}\}$ 的含义。

There are more such rules, but this gives a hint of the added complexity. As an aside, to be consistent, if the ALL value is a set then the other values of that domain must be treated as singleton sets in order to have uniform operators on the domain.

还有更多这样的规则，但这暗示了额外的复杂性。顺便说一下，为了保持一致性，如果 ALL 值是一个集合，那么该域的其他值必须被视为单元素集合，以便在该域上使用统一的运算符。

Decoration's interact with aggregate values. If the aggregate tuple functionally defines the decoration value, then the value appears in the resulting tuple. Otherwise the decoration field is NULL. For example, in the following query the continent is not specified unless nation is.

修饰符与聚合值相互作用。如果聚合元组在功能上定义了修饰值，那么该值会出现在结果元组中。否则，修饰字段为 NULL。例如，在以下查询中，除非指定了国家，否则不会指定大洲。

SELECT day, nation, MAX (Temp),

SELECT 日期, 国家, MAX (温度),

continent (nation)

大洲 (国家)

FROM Weather

FROM 天气表

GROUP BY Day (Time) AS day,

GROUP BY 按时间计算的日期 AS 日期,

Country (Latitude, Longitude)

由纬度和经度确定的国家

AS nation

AS 国家

WITH CUBE;

WITH CUBE;

The query would produce the sample tuples:

该查询将生成示例元组：

<!-- Media -->

<table><tr><td colspan="4">Table 6: Demonstrating decorations and ALL</td></tr><tr><td>day</td><td>nation</td><td>max (Temp)</td><td>continent</td></tr><tr><td>25/1/1995</td><td>USA</td><td>28</td><td>North America</td></tr><tr><td>ALL</td><td>USA</td><td>37</td><td>North America</td></tr><tr><td>25/1/1995</td><td>ALL</td><td>41</td><td>NULL</td></tr><tr><td>ALL</td><td>ALL</td><td>48</td><td>NULL</td></tr></table>

<table><tbody><tr><td colspan="4">表6：展示装饰和全部情况</td></tr><tr><td>天</td><td>国家</td><td>最高（温度）</td><td>大洲</td></tr><tr><td>25/1/1995</td><td>美国</td><td>28</td><td>北美洲</td></tr><tr><td>全部</td><td>美国</td><td>37</td><td>北美洲</td></tr><tr><td>25/1/1995</td><td>全部</td><td>41</td><td>空值</td></tr><tr><td>全部</td><td>全部</td><td>48</td><td>空值</td></tr></tbody></table>

<!-- Media -->

If the application wants only a roll-up or drill-down report, the full cube is overkill. It is reasonable to offer the additional function ROLLUP in addition to CUBE. ROLLUP produces just the super-aggregates:

如果应用程序只需要汇总（roll - up）或钻取（drill - down）报告，那么完整的多维数据集（cube）就有些大材小用了。除了CUBE函数外，提供额外的ROLLUP函数是合理的。ROLLUP仅生成超级聚合结果：

(fl , f2 , . . ., ALL) ,

(fl , f2 , ..., 全部) ,

(fl , ALL, ..., ALL),

(fl , 全部, ..., 全部),

(ALL, ALL, ..., ALL) .

(全部, 全部, ..., 全部) .

Cumulative aggregates , like running sum or running average, work especially well with ROLLUP since the answer set is naturally sequential (linear) while the CUBE is naturally non-linear (multi-dimensional). But, ROLLUP and CUBE must be ordered for cumulative operators to apply.

累积聚合，如累计总和或累计平均值，与ROLLUP函数配合使用效果特别好，因为结果集本质上是顺序的（线性的），而CUBE本质上是非线性的（多维的）。但是，要应用累积运算符，必须对ROLLUP和CUBE进行排序。

We investigated letting the programmer specify the exact list of super-aggregates but encountered complexities with collation, correlation, and expressions. We believe ROLLUP and CUBE will serve the needs of most applications.

我们曾研究让程序员指定确切的超级聚合列表，但在排序规则、相关性和表达式方面遇到了复杂性问题。我们认为ROLLUP和CUBE能满足大多数应用程序的需求。

It is convenient to know when a column value is an aggregate. One way to test this is to apply the ALL (   ) function to the value and test for a non-NULL value. This is so useful that we propose a Boolean function GROUPING (   ) that, given a select list element, returns TRUE if the element is an ALL value, and FALSE otherwise.

了解某列值是否为聚合值很方便。一种测试方法是对该值应用ALL()函数，并检查其是否为非空值。这非常有用，因此我们提议使用布尔函数GROUPING()，给定一个选择列表元素，如果该元素是ALL值，则返回TRUE，否则返回FALSE。

Veteran SQL implementers will be terrified of the ALL value - like NULL, it will create many special cases. If the goal is to help report writer and GUI visualization software, then it may be simpler to adopt the following approach ${}^{1}$ :

有经验的SQL实现者会对ALL值感到担忧——就像NULL一样，它会产生许多特殊情况。如果目标是帮助报表编写者和图形用户界面（GUI）可视化软件，那么采用以下方法${}^{1}$可能更简单：

- Use the NULL value in place of the ALL value.

- 用NULL值代替ALL值。

- Do not implement the ALL (   ) function.

- 不实现ALL()函数。

- Implement the GROUPING (   ) function to discriminate between NULL and ALL .

- 实现GROUPING()函数以区分NULL和ALL。

In this minimalist design, tools and users can simulate the ALL value as by for example:

在这种简约设计中，工具和用户可以通过以下方式模拟ALL值，例如：

SELECT Model, Year, Color, SUM (sales),

SELECT 型号, 年份, 颜色, SUM (销售额),

GROUPING (Model) ,

GROUPING (型号) ,

GROUPING (Year) ,

GROUPING (年份) ,

GROUPING (Color)

GROUPING (颜色)

FROM Sales

来自销售数据

GROUP BY Model, Year, Color WITH CUBE; Wherever the ALL value appeared before, now the corresponding value will be NULL in the data field and TRUE in the corresponding grouping field. For example, the global sum of Table 2 will be the tuple:

按型号、年份、颜色进行分组，并使用 CUBE 操作符；之前出现 ALL 值的地方，现在数据字段中对应的数值将为 NULL，而相应的分组字段中为 TRUE。例如，表 2 的全局总和将是元组：

(NULL, NULL, NULL, 941, TRUE, TRUE, TRUE)

(NULL, NULL, NULL, 941, TRUE, TRUE, TRUE)

while the "real" cube operator would give:

而“真正的” CUBE 操作符会给出：

$\left( {\;\text{ALL},\;\text{ALL},\;\text{ALL},\;\text{941}\;}\right) .$

## 4. Addressing The Data Cube

## 4. 处理数据立方体

Section 5 discusses how to compute the cube and how users can add new aggregate operators. This section considers extensions to SQL syntax to easily access the elements of the data cube - making it recursive and allowing aggregates to reference sub-aggregates.

第 5 节讨论了如何计算数据立方体以及用户如何添加新的聚合操作符。本节考虑对 SQL 语法进行扩展，以便轻松访问数据立方体的元素——使其具有递归性，并允许聚合引用子聚合。

It is not clear where to draw the line between the reporting/visualization tool and the query tool. Ideally, application designers should be able to decide how to split the function between the query system and the visualization tool. Given that perspective, the SQL system must be a Turing-complete programming environment.

目前尚不清楚报表/可视化工具和查询工具之间的界限在哪里。理想情况下，应用程序设计人员应该能够决定如何在查询系统和可视化工具之间分配功能。从这个角度来看，SQL 系统必须是一个图灵完备的编程环境。

SQL3 defines a Turing-complete programming language. So, anything is possible. But, many things are not easy. Our task is to make simple and common things easy.

SQL3 定义了一种图灵完备的编程语言。所以，一切皆有可能。但很多事情并不容易。我们的任务是让简单和常见的事情变得容易。

The most common request is for percent-of-total as an aggregate function. In SQL this is computed as two SQL statements. SELECT Model, Year, Color, SUM (Sales), SUM(Sales) / (SELECT SUM(Sales) FROM Sales WHERE Model IN \{ 'Ford' , 'Chevy' \} AND Year Between 1990 AND 1992 ) FROM Sales WHERE Model IN \{ 'Ford' , 'Chevy'\} AND Year Between 1990 AND 1992 GROUP BY CUBE (Model, Year, Color);

最常见的需求是将占比作为一种聚合函数。在 SQL 中，这需要通过两条 SQL 语句来计算。SELECT 型号, 年份, 颜色, SUM (销售数据), SUM(销售数据) / (SELECT SUM(销售数据) FROM 销售数据 WHERE 型号 IN { '福特' , '雪佛兰' } AND 年份 BETWEEN 1990 AND 1992 ) FROM 销售数据 WHERE 型号 IN { '福特' , '雪佛兰' } AND 年份 BETWEEN 1990 AND 1992 GROUP BY CUBE (型号, 年份, 颜色);

It seems natural to allow the shorthand syntax to name the global aggregate:

允许使用简写语法来命名全局聚合似乎是很自然的：

SELECT Model, Year, Color

SELECT 型号, 年份, 颜色

SUM(Sales) AS total,

SUM(销售数据) AS 总计,

SUM(Sales) / total (ALL, ALL, ALL)

SUM(销售数据) / 总计 (ALL, ALL, ALL)

FROM Sales

FROM 销售数据

WHERE MOdel IN \{ 'Ford' , 'Chevy' \}

WHERE 型号 IN { '福特' , '雪佛兰' }

AND Year Between 1990 AND 1992

AND 年份 BETWEEN 1990 AND 1992

GROUP BY CUBE (Model, Year, Color);

按 CUBE (型号, 年份, 颜色) 分组;

This leads into deeper water. The next step is a desire to compute the index of a value - an indication of how far the value is from the expected value. In a set of $N$ values, one expects each item to contribute one $N$ th to the sum. So the 1D index of a set of values is:

这会引入更深入的内容。下一步是希望计算某个值的索引——该值与预期值的偏离程度的一种指示。在一组 $N$ 个值中，人们期望每个元素对总和的贡献为 $N$ 分之一。因此，一组值的一维索引为：

$\operatorname{index}\left( {v}_{i}\right)  = {v}_{i}/\left( {{\sum }_{j}{v}_{j}}\right)$

If the value set is two dimensional, this commonly used financial function is a nightmare of indices. It is best described in a programming language. The current approach to selecting an field value from a 2D cube with fields row and column would read as:

如果值集是二维的，这个常用的金融函数在处理索引时会变得异常复杂。最好用编程语言来描述它。当前从具有行和列字段的二维数据立方体中选择字段值的方法如下：

---

	SELECT V

	FROM cube

	从 数据立方体 中选取

	WHERE row = : i

	条件是 行 = : i

	AND column $=  : j$

	并且 列 $=  : j$

We recommend the simpler syntax:

我们推荐更简单的语法：

	cube.v(:i, :j)

	数据立方体.v(:i, :j)

---

---

<!-- Footnote -->

${}^{1}$ This is the syntax and approach is used by Microsoft’s SQLserver (version 6.5) as designed and implemented by Don Reichart.

${}^{1}$ 这是微软 SQL Server（6.5 版本）所采用的语法和方法，由唐·赖卡特（Don Reichart）设计并实现。

<!-- Footnote -->

---

as a shorthand for the above selection expression. With this notation added to the SQL programming language, it should be fairly easy to compute super-super-aggregates from the base cube.

作为上述选择表达式的简写形式。将这种表示法添加到 SQL 编程语言中后，从基础数据立方体计算超超级聚合应该相当容易。

## 5. Computing the Data Cube

## 5. 计算数据立方体

CUBE generalizes aggregates and GROUP BY, so all the technology for computing those results also applies to computing the core of the cube. The main techniques are:

数据立方体（CUBE）对聚合和分组依据（GROUP BY）进行了泛化，因此所有用于计算这些结果的技术也适用于计算数据立方体的核心。主要技术如下：

- Minimize data movement and consequent processing cost by computing aggregates at the lowest possible level.

- 通过在尽可能低的级别计算聚合来最小化数据移动和随之而来的处理成本。

- If possible, use arrays or hashing to organize aggregation columns in memory, storing one aggregate in each array or hash entry.

- 若有可能，使用数组或哈希在内存中组织聚合列，在每个数组或哈希条目中存储一个聚合值。

- If the aggregation values are large strings, keep a hashed symbol table that maps each string to an integer so that the aggregate values are small. When a new value appears, it is assigned a new integer. This organization makes values dense and the aggregates can be stored as an $N$ -dimensional array.

- 如果聚合值是长字符串，维护一个哈希符号表，将每个字符串映射到一个整数，以使聚合值变小。当出现新值时，为其分配一个新整数。这种组织方式使值变得密集，并且聚合可以存储为 $N$ 维数组。

- If the number of aggregates is too large to fit in memory, use sorting or hybrid hashing to organize the data by value and then aggregate with a sequential scan of the sorted data.

- 如果聚合数量太多无法全部放入内存，使用排序或混合哈希按值对数据进行组织，然后通过对排序后的数据进行顺序扫描来进行聚合。

- If the source data spans many disks or nodes, use parallelism to aggregate each partition and then coalesce these aggregates.

- 如果源数据跨越多个磁盘或节点，可使用并行处理来聚合每个分区，然后合并这些聚合结果。

Some innovation is needed to compute the "ALL" tuples of the cube from the GROUP BY core. The ALL value adds one extra value to each dimension in the CUBE. So,an $N$ - dimensional cube of $N$ attributes each with cardinality ${C}_{i}$ , will have $\Pi \left( {{C}_{i} + 1}\right)$ . If each ${C}_{i} = 4$ then a ${4D}$ CUBE is 2.4 times larger than the base GROUP BY. We expect the ${C}_{i}$ to be large (tens or hundreds) so that the CUBE will be only a little larger than the GROUP BY.

需要一些创新方法来从GROUP BY核心计算立方体的“ALL”元组。ALL值会为立方体的每个维度添加一个额外的值。因此，一个具有$N$个属性（每个属性的基数为${C}_{i}$）的$N$维立方体将有$\Pi \left( {{C}_{i} + 1}\right)$。如果每个${C}_{i} = 4$，那么一个${4D}$立方体的大小是基础GROUP BY的2.4倍。我们预计${C}_{i}$会很大（几十或几百），这样立方体只会比GROUP BY略大一点。

The cube operator allows many aggregate functions in the aggregation list of the GROUP BY clause. Assume in this discussion that there is a single aggregate function $\mathrm{F}\left( \right)$ being computed on an $N$ -dimensional cube. The extension to a computing a list of functions is a simple generalization.

立方体运算符允许在GROUP BY子句的聚合列表中使用多个聚合函数。在本次讨论中，假设正在一个$N$维立方体上计算单个聚合函数$\mathrm{F}\left( \right)$。将其扩展到计算函数列表是一种简单的泛化。

The simplest algorithm to compute the cube is to allocate a handle for each cube cell. When a new tuple: $\left( {{x}_{1},{x}_{2},\ldots ,}\right.$ ${\mathrm{x}}_{\mathrm{N}},\mathrm{v})$ arrives,the Iter (handle, $\mathrm{v}$ ) function is called ${2}^{N}$ times - once for each handle of each cell of the cube matching this value. The ${2}^{N}$ comes from the fact that each coordinate can either be ${x}_{i}$ or ALL. When all the input tuples have been computed, the system invokes the final (&handle) function for each of the $\Pi \left( {{C}_{i} + 1}\right)$ nodes in the cube. Call this the ${2}^{N}$ -algorithm.

计算立方体的最简单算法是为每个立方体单元格分配一个句柄。当一个新的元组：$\left( {{x}_{1},{x}_{2},\ldots ,}\right.$ ${\mathrm{x}}_{\mathrm{N}},\mathrm{v})$到达时，Iter(句柄, $\mathrm{v}$)函数会被调用${2}^{N}$次——对于与该值匹配的立方体每个单元格的每个句柄调用一次。${2}^{N}$源于每个坐标可以是${x}_{i}$或ALL这一事实。当所有输入元组都计算完毕后，系统会为立方体中的$\Pi \left( {{C}_{i} + 1}\right)$个节点分别调用final(&句柄)函数。将此称为${2}^{N}$算法。

If the base table has cardinality $T$ ,the ${2}^{N}$ -algorithm invokes the Iter (   ) function ${Tx}{2}^{N}$ times. It is often faster to compute the super-aggregates from the core GROUP BY, reducing the number of calls by approximately a factor of $T$ . It is often possible to compute the cube from the core or from intermediate results only $M$ times larger than the core. The following trichotomy characterizes the options in computing super-aggregates.

如果基表的基数为$T$，${2}^{N}$算法会调用Iter( )函数${Tx}{2}^{N}$次。通常，从核心GROUP BY计算超级聚合会更快，调用次数大约会减少$T$倍。通常可以从核心或仅比核心大$M$倍的中间结果来计算立方体。以下三种情况描述了计算超级聚合的选项。

Consider aggregating a two dimensional set of values $\left\{  {X}_{ij}\right.$ $\left| {i = 1,\ldots ,I;j = 1,\ldots ,J\} \text{. Aggregate functions can be classi-}}\right|$ fied into three categories:

考虑对二维值集$\left\{  {X}_{ij}\right.$ $\left| {i = 1,\ldots ,I;j = 1,\ldots ,J\} \text{. Aggregate functions can be classi-}}\right|$进行聚合，分为三类：

Distributive: Aggregate function $F\left( \right)$ is distributive if there is a function $G\left( \right)$ such that $F\left( \left\{  {X}_{i,j}\right\}  \right)  = G\left( \left\{  {F\left( \left\{  {X}_{i,j}\right. \right. }\right. \right.$ $\left. {\mid i = 1,\ldots ,l\} }\right)  \mid  j = 1,\ldots J\} )$ . COUNT (   ), MIN (   ), MAX (   ), SUM (   ) are all distributive. In fact, $F = G$ for all but COUNT (   ). G= SUM(   ) for the COUNT (   ) function. Once order is imposed, the cumulative aggregate functions also fit in the distributive class.

可分配的：如果存在一个函数$G\left( \right)$使得$F\left( \left\{  {X}_{i,j}\right\}  \right)  = G\left( \left\{  {F\left( \left\{  {X}_{i,j}\right. \right. }\right. \right.$ $\left. {\mid i = 1,\ldots ,l\} }\right)  \mid  j = 1,\ldots J\} )$，则聚合函数$F\left( \right)$是可分配的。COUNT( )、MIN( )、MAX( )、SUM( )都是可分配的。实际上，除了COUNT( )之外，所有情况都有$F = G$。对于COUNT( )函数，G = SUM( )。一旦施加了顺序，累积聚合函数也属于可分配类别。

Algebraic: Aggregate function $F\left( \right)$ is algebraic if there is an $M$ -tuple valued function $G\left( \right)$ and a function $H\left( \right)$ such that

代数的：如果存在一个$M$元组值函数$G\left( \right)$和一个函数$H\left( \right)$使得

$F\left( \left\{  {X}_{i,j}\right\}  \right)  = H\left( \left\{  {G\left( {\left\{  {{X}_{i,j} \mid  i = 1,\ldots ,1}\right\}   \mid  j = 1,\ldots ,J}\right) }\right\}  \right) .\;$ Average(   ), standard deviation, MaxN(   ), MinN(   ), center_of_mass(   ) are all algebraic. For Average, the function $G\left( \right)$ records the sum and count of the subset. The $H\left( \right)$ function adds these two components and then divides to produce the global average. Similar techniques apply to finding the $N$ largest values,the center of mass of group of objects, and other algebraic functions. The key to algebraic functions is that a fixed size result (an M-tuple) can summarize the sub-aggregation.

$F\left( \left\{  {X}_{i,j}\right\}  \right)  = H\left( \left\{  {G\left( {\left\{  {{X}_{i,j} \mid  i = 1,\ldots ,1}\right\}   \mid  j = 1,\ldots ,J}\right) }\right\}  \right) .\;$ 平均值（Average( )）、标准差（standard deviation）、最大N值（MaxN( )）、最小N值（MinN( )）、质心（center_of_mass( )）均为代数函数。对于平均值函数，$G\left( \right)$函数会记录子集的总和与数量。$H\left( \right)$函数将这两个分量相加，然后相除得出全局平均值。类似的技术也适用于找出$N$个最大值、一组对象的质心以及其他代数函数。代数函数的关键在于，固定大小的结果（一个M元组）可以概括子聚合。

Holistic: Aggregate function $F\left( \right)$ is holistic if there is no constant bound on the size of the storage needed to describe a sub-aggregate. That is,there is no constant $M$ , such that an $M$ -tuple characterizes the computation $F\left( \left\{  {{X}_{i,j} \mid  i = 1,\ldots ,I}\right\}  \right)$ . Median(   ),MostFrequent(   ) (also called the Mode(   )), and Rank(   ) are common examples of holistic functions.

整体函数：如果描述子聚合所需的存储大小没有固定的界限，则聚合函数$F\left( \right)$为整体函数。也就是说，不存在常数$M$，使得一个$M$元组能够表征计算$F\left( \left\{  {{X}_{i,j} \mid  i = 1,\ldots ,I}\right\}  \right)$。中位数（Median( )）、最频繁值（MostFrequent( )，也称为众数（Mode( )））和排名（Rank( )）是常见的整体函数示例。

We know of no more efficient way of computing super-aggregates of holistic functions than the ${2}^{N}$ -algorithm using the standard GROUP BY techniques. We will not say more about cubes of holistic functions.

除了使用标准GROUP BY技术的${2}^{N}$算法外，我们不知道有更高效的方法来计算整体函数的超级聚合。关于整体函数的立方体，我们不再赘述。

Cubes of distributive functions are relatively easy to compute. Given that the core is represented as an $N$ - dimensional array in memory, each dimension having size ${C}_{i} + 1$ ,the $N - 1$ dimensional slabs can be computed by projecting (aggregating) one dimension of the core. For example the following computation aggregates the first dimension.

分布函数的立方体相对容易计算。假设核心在内存中表示为一个$N$维数组，每个维度的大小为${C}_{i} + 1$，则可以通过对核心的一个维度进行投影（聚合）来计算$N - 1$维切片。例如，以下计算对第一个维度进行聚合。

$\operatorname{cuBE}\left( {\mathrm{{ALL}},{x}_{2},\ldots ,{x}_{N}}\right)  = F\left( \left\{  {\operatorname{cuBE}\left( {i,{x}_{2},\ldots ,{x}_{N}}\right)  \mid  i = 1,\ldots {C}_{1}}\right\}  \right) .$ $N$ such computations compute the $N - 1$ dimensional super-aggregates. The distributive nature of the function $F\left( \right)$ allows aggregates to be aggregated. The next step is to compute the next lower dimension - an (...ALL,..., ALL...) case. Thinking in terms of the cross tab, one has a choice of computing the result by aggregating the lower row, or aggregating the right column (aggregate (ALL, *) or (*, ALL)). Either approach will give the same answer. The algorithm will be most efficient if it aggregates the smaller of the two (pick the * with the smallest ${C}_{i}$ .) In this way,the super-aggregates can be computed dropping one dimension at a time.

$\operatorname{cuBE}\left( {\mathrm{{ALL}},{x}_{2},\ldots ,{x}_{N}}\right)  = F\left( \left\{  {\operatorname{cuBE}\left( {i,{x}_{2},\ldots ,{x}_{N}}\right)  \mid  i = 1,\ldots {C}_{1}}\right\}  \right) .$ $N$ 这样的计算可以计算出$N - 1$维超级聚合。函数$F\left( \right)$的分布性质允许对聚合结果进行进一步聚合。下一步是计算下一个较低维度——即(...ALL,..., ALL...)的情况。从交叉表的角度考虑，可以选择通过聚合下一行或聚合右列（聚合(ALL, *)或(*, ALL)）来计算结果。两种方法得出的答案相同。如果算法对两者中较小的那个进行聚合（选择${C}_{i}$最小的*），则效率最高。通过这种方式，可以一次降低一个维度来计算超级聚合。

Algebraic aggregates are more difficult to compute than distributive aggregates. Recall that an algebraic aggregate saves its computation in a handle and produces a result in the end - at the Final (   ) call. Average (   ) for example maintains the count and sum values in its handle. The super-aggregate needs these intermediate results rather than just the raw sub-aggregate. An algebraic aggregate must maintain a handle ( $M$ -tuple) for each element of the cube (this is a standard part of the group-by operation). When the core GROUP BY operation completes, the CUBE algorithm passes the set of handles to each $N - 1$ dimensional super-aggregate. When this is done the handles of these super-aggregates are passed to the super-super aggregates, and so on until the (ALL, ALL, ..., ALL) aggregate has been computed. This approach requires a new call for distributive aggregates:

代数聚合比分布聚合更难计算。回想一下，代数聚合将其计算结果保存在一个句柄中，并在最后（在Final( )调用时）产生结果。例如，平均值（Average( )）在其句柄中维护计数和总和值。超级聚合需要这些中间结果，而不仅仅是原始的子聚合。代数聚合必须为立方体的每个元素维护一个句柄（$M$元组）（这是分组操作的标准部分）。当核心分组操作完成后，CUBE算法将句柄集传递给每个$N - 1$维超级聚合。完成此操作后，这些超级聚合的句柄将传递给超级 - 超级聚合，依此类推，直到计算出(ALL, ALL, ..., ALL)聚合。这种方法需要为分布聚合进行新的调用：

Iter_super(&handle, &handle)

Iter_super(&handle, &handle)

which folds the sub-aggregate on the right into the super aggregate on the left. The same ordering ideas (aggregate on the smallest list) applies.

该函数将右侧的子聚合合并到左侧的超级聚合中。同样的排序思路（对最小列表进行聚合）也适用。

If the data cube does not fit into memory, array techniques do not work. Rather one must either partition the cube with a hash function or sort it. These are standard techniques for computing the GROUP BY. The super-aggregates are likely to be orders of magnitude smaller than the core, so they are very likely to fit in memory.

如果数据立方体无法装入内存，数组技术就无法使用。相反，必须使用哈希函数对立方体进行分区或对其进行排序。这些都是计算GROUP BY的标准技术。超级聚合结果可能比核心数据小几个数量级，因此它们很可能能够装入内存。

It is possible that the core of the cube is sparse. In that case, only the non-null elements of the core and of the super-aggregates should be represented. This suggests a hashing or a B-tree be used as the indexing scheme for aggregation values [Essbase].

立方体的核心数据可能是稀疏的。在这种情况下，应该只表示核心数据和超级聚合结果中的非空元素。这表明可以使用哈希或B树作为聚合值的索引方案[Essbase]。

## 6. Summary:

## 6. 总结：

The cube operator generalizes and unifies several common and popular concepts:

立方体运算符对几个常见且流行的概念进行了推广和统一：

aggregates,

聚合（aggregates）

group by,

分组（group by）

histograms,

直方图（histograms）

roll-ups and drill-downs and,

上卷和下钻（roll-ups and drill-downs）以及

cross tabs.

交叉表（cross tabs）

The cube is based on a relational representation of aggregate data using the ALL value to denote the set over which each aggregation is computed. In certain cases it makes sense to restrict the cube to just a roll-up aggregation for drill-down reports.

立方体基于聚合数据的关系表示，使用ALL值来表示每个聚合计算所基于的集合。在某些情况下，将立方体限制为仅用于下钻报告的上卷聚合是有意义的。

The cube is easy to compute for a wide class of functions (distributive and algebraic functions). SQL's basic set of five aggregate functions needs careful extension to include functions such as rank, $N$ _tile,cumulative,and percent of total to ease typical data mining operations.

对于广泛的函数类（分配函数和代数函数），立方体很容易计算。SQL的五个基本聚合函数集需要仔细扩展，以包括诸如排名、$N$分位数、累积和占总数百分比等函数，以便于进行典型的数据挖掘操作。

## 7. Acknowledgments

## 7. 致谢

Joe Hellerstein suggested interpreting the ALL value as a set. Tanj Bennett, David Maier and Pat O'Neil made many helpful suggestions that improved the presentation. Don Reichart's implementation of the CUBE in Micro-soft's SQLserver caused us to adopt his syntax.

乔·赫勒斯泰因（Joe Hellerstein）建议将ALL值解释为一个集合。坦吉·贝内特（Tanj Bennett）、大卫·迈尔（David Maier）和帕特·奥尼尔（Pat O'Neil）提出了许多有用的建议，改进了本文的表述。唐·赖卡特（Don Reichart）在微软SQL Server中对CUBE的实现促使我们采用了他的语法。

## 8. References

## 8. 参考文献

[Access] Microsoft Access Relational Database Management System for Windows, Language Reference - Functions, Statements, Methods, Properties, and Actions, DB26142, Microsoft, Redmond, WA, 1994.

[Essbase] Method and apparatus for storing and retrieving multi-dimensional data in computer memory, Inventor: Earle; Robert J., Assignee: Arbor Software Corporation, US Patent 05359724, October 1994,

[Illustra] Illustra DataBlade Developer's Kit I.I., Illustra Information Technologies, Oakland, CA, 1994.

[Melton & Simon] Jim Melton and Alan Simon, Understanding the New SQL: A Complete Guide, Morgan Kaufmann, San Francisco, CA, 1993.

[Red Brick] RISQL Reference Guide, Red Brick Warehouse VPT Version 3, Part no: 401530, Red Brick Systems, Los Gatos, CA, 1994

[TPC] The Benchmark Handbook for Database and Transaction Processing Systems - 2nd edition, J. Gray (ed.), Morgan Kaufmann, San Francisco, CA, 1993. Or http://www.tpc.org/