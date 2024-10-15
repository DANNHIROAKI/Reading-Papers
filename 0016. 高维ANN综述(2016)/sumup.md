# $\textbf{0. Abstract}$

> :one:本文干了啥：对多种最先进的$\text{ANNS}$算法进行实验评估
>
> :two:本文实验的特点
>
> 1. 跨学科：所选取的算法来自不同领域
> 2. 多元评估：选取了$\text{20}$多种数据集 / 一大堆指标 / 不同查询负载(处理查询的数量)
>
> :three:关于结果分析
>
> 1. 以理解算法的性能表现
> 2. 提出了一种方法，使得能在大多数据集上提高查询效率

# $\textbf{1. Introduction}$

> ## $\textbf{1.0.}$ 关于$\textbf{ANNS}$
>
> > :one:最邻近查询：
> >
> > 1. 是啥：从数据库中，找到与查询对象最近的对象
> > 2. 困难：在高维欧几里得空间找到精确最邻近很难(难以突破暴力解法)，$\text{aka}$维度诅咒(灾难)
> >
> > :two:近似最邻近查询：相比精确算法能更高效执行，对许多实际问题很有用
>
> ## $\textbf{1.1.}$ 研究动机: 先进$\textbf{ANN}$算法缺乏系统性
>
> > :one:关于不同领域的$\text{ANN}$算法
> >
> > 1. 当前存在的问题：
> >    - $\text{ANN}$在不同领域有需求，对应的$\text{ANN}$方法被分别提出
> >    - 每个领域都有各自的数据集用于测试$\text{ANN}$，不同领域共用的数据集很少
> > 2. 本文要干的：选取不同领域的**多个最先进算法**，在不同领域的**多个数据集**上测试
> >
> > :two:关于现有$\text{ANN}$算法的评价指标
> >
> > 1. 现有的几类评价指标：
> >    - 性能类：搜索的时间复杂度，搜索质量(精确度/正确率)
> >    - 资源类：索引大小
> >    - 耐草类：可扩展性，鲁棒性
> >    - 维护类：可更新性，更新参数的成本
> > 2. 当前存在的问题：
> >    - 指标上：任何单一指标都无法全面评估某$\text{ANN}$算法
> >    - 测试上：使用的待查询点分布，与数据库中点分布实际是相似的 (难以反映实际情况)
> > 3. 本文要干的：评估了算法在多种设置和指标下的性能
> >
> > :three:关于目前$\text{ANN}$算法测试的结果：$\text{Discrepancies}$ 
> >
> > 1. 实例：一些文献中性能$\text{SpectralHashing>AGH}$，但其它文献相反
> > 2. 原因：测试算法时，不同的设置/数据集/调优方式$\text{etc.}$ 
> > 3. 本文要干啥：尽力对几种算法公平比较，使得不同场景性能对比一致
>
> ## $\textbf{1.2.Contributions}$ 
>
> > :one:比较多个不同领域的最先进算法：
> >
> > 1. 关于比较实验：
> >    - 所有算法在实现时，都没进行额外的优化/改进
> >    - 采用多种指标
> > 2. 关于算法选择：本文还给出了**不同场景/不同设置**下，选择$\text{ANN}$算法的经验法则($\text{Rule-of-thumb}$)
> >
> >  :two:算法分类与分析：
> >
> > 1. 算法分类：分为基于哈希/划分/图三类，评估三类算法内部间/跨类别的性能对比
> > 2. 数据驱动($\text{data-based}$)分析 
> >    - 是个啥：分析与测试实际数据，得出基于数据(而非推理)的结论
> >    - 干了啥：总结了通用优化/选择方法$+$提供一些对数据集的解释$\to$设计了新算法$\text{DPG}$ 

# $\textbf{2. Background}$ 

> ## $\textbf{2.1. Problem Definition}$
>
> > :one:度量空间/距离：$\mathbb{R}^d$(数据为$d$维向量)/欧几里得空间
> >
> > :two:最邻近查询
> >
> > 1. 最邻近($\text{NNS}$)：与查询点最近的唯一点
> > 2. $\text{k-}$最邻近($\text{k-NNS}$)：与查询点距离最近的$k$个点
> >
> > :three:近似最邻近查询：最邻近查询的$\text{Recall<100\%}$版本，即$\text{ANNS/k-ANNS}$ 
>
> ## $\textbf{2.2. Scope}$(研究中的人为限制) 
>
> > :one:算法选择：只选择当前最先进的算法，排除被明显超过的其它算法
> >
> > :two:算法实现：注重算法技术本身，削弱实现时的优化 (如取消多线程/多$\text{CPU}$等)
> >
> > :three:密集向量：默认向量都密集，不考虑对稀疏数据的特殊处理
> >
> > :four:标签：将每个点的真实$k$个最邻近点作为标签，以便得到召回率

# $\textbf{3. Related Work}$ 

> ### $\textbf{3.1. Hashing-based Methods}$
>
> > 💡基本特征：将数据点转换为低维表示$\to$每个数据点可以用哈希码表示
> >
> > #### $\textbf{3.1.1. Locality Sensitive Hashing}$ (有理论保证)
> >
> > > :zero:$\text{(r,c)-ANN}$问题​​：给定距离阈值$r$，查询点$q$，考虑数据库点$e_i$在$q$周围$r$以及$cr$范围的分布
> > >
> > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240803185122335.png" alt="image-20240803185122335" style="zoom: 30%;" /> 
> > >
> > > | $\textbf{Case}$ |     $\exist{}e_i使\text{D}\in[0,r]$      |    $\exist{}e_i使\text{D}\in{}[r,cr]$    | $\exist{}e_i使\text{D}\in[cr,\infin{}]$  |        返回对象        |
> > > | :-------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :--------------------: |
> > > | $\text{Case 1}$ | <span style="color:#00FF00;">一定</span> | <span style="color:#FF9900;">可能</span> | <span style="color:#FF9900;">可能</span> | 满足$D\leq{cr}$的$e_i$ |
> > > | $\text{Case 2}$ |  <span style="color:red;">不可能</span>  |  <span style="color:red;">不可能</span>  |  <span style="color:red;">不可能</span>  |          寂寞          |
> > > | $\text{Case 3}$ |  <span style="color:red;">不可能</span>  | <span style="color:#00FF00;">一定</span> | <span style="color:#FF9900;">可能</span> | 满足$D\leq{cr}$的$e_i$ |
> > >
> > > - 表中$\text{D}=\text{dist}(q,e_i)$ 
> > >
> > > :one:$\text{LSH}$原理
> > >
> > > |            输入点$e_i,e_j$            | $\xrightarrow{局部敏感哈希函数}$ |            映射结果            |
> > > | :-----------------------------------: | :------------------------------: | :----------------------------: |
> > > | 相似度高，即$\text{dist}(e_i,e_j)<r$  | $\xrightarrow{局部敏感哈希函数}$ | 高概率被映射到**相同哈希码**上 |
> > > | 相似度低，即$\text{dist}(e_i,e_j)>cr$ | $\xrightarrow{局部敏感哈希函数}$ | 高概率被映射到**不同哈希码**上 |
> > >
> > > - 需满足假设：对于$e_i,e_j$，哈希函数的选择是**随机**和**独立**的，见[$\text{CIKM'13}$](https://doi.org/10.1145/2505515.2505765)
> > >
> > > :two:$\text{LSH}$函数：影响性能的关键
> > >
> > > 1. 一些常用的$\text{LSH}$函数：
> > >    - 针对欧几里得空间的：[$\text{SCG'04}$](https://doi.org/10.1145/997817.997857) / [$\text{FOCS'06}$](https://doi.org/10.1109/FOCS.2006.49) / [$\text{SODA'14}$](https://dl.acm.org/doi/abs/10.5555/2634074.2634150) / [$\text{WADS'07}$](https://dl.acm.org/doi/10.5555/2394893.2394899) / [$\text{STOC'15}$](https://doi.org/10.1145/2746539.2746553)
> > >    - 基于随机线性投影的：[$\text{SCG'04}$](https://doi.org/10.1145/997817.997857) / [$\text{VLDB'99}$](https://dl.acm.org/doi/10.5555/645925.671516) / [$\text{SODA'06}$](https://dl.acm.org/doi/10.5555/1109557.1109688) / [$\text{SIGMOD'09}$](https://doi.org/10.1145/1559845.1559905)
> > >
> > > 2. $\text{LSH}$函数的连接：
> > >
> > >    - 是啥：将多个哈希函数首尾相连，降低不相似点的碰撞概率
> > >    - 弊端：增加了哈希表数量(时空开销)$+$也可能减少相似点碰撞概率
> > >
> > > :three:$\text{LSH}$函数及$\text{LSH}$方法的改进研究
> > >
> > > 1. 动态$\text{LSH}$函数
> > >    - 静态$\text{LSH}$原理：处理所有点构建哈希表$\to$哈希表一构建就不变$\to$执行查询
> > >    - 静态$\text{LSH}$弊端：哈希表随机构建，会导致**与查询点很近的点**与**查询点**不碰撞(被忽略)
> > >    - 动态$\text{LSH}$：查询时**动态地**计数和调整碰撞情况，[$\text{VLDB'07}$](https://doi.org/10.14778/3137765.3137836) / [$\text{SIGMOD'07}$](https://doi.org/10.1145/2213836.2213898) / [$\text{SIGMOD'16}$](https://doi.org/10.1145/2882903.2882930)  
> > >
> > > 2. 启发式寻桶
> > >    - 咋办：通过启发式方法(靠直觉)检查查询点附近的其它桶，[$\text{VLDB'07}$](https://dl.acm.org/doi/10.5555/1325851.1325958) / [$\text{MM'08}$](https://doi.org/10.1145/1459359.1459388) / [$\text{VLDB'07}$](https://doi.org/10.14778/3137765.3137836) 
> > >    - 好处：提高搜索质量同时，不增加哈希表数量(相比连接$\text{LSH}$函数) 
> >
> > #### $\textbf{3.1.2. Learning to Hash}$ (无理论保证)
> >
> > > :one:原理：
> > >
> > > 1. 学习原有数据的分布$\xrightarrow{生成}$特定哈希
> > >    - 大多哈希方法要求：哈希码平衡(每个桶均匀分布)$+$无相关性(不同哈希码间无依赖)
> > > 2. 经过哈希后，原始空间中点的近似关系，在哈希编码空间得到最大程度保留
> > >
> > > :two:类型：
> > >
> > > | $\textbf{Type}$  | $\textbf{Pub.}$                                              |
> > > | :--------------: | ------------------------------------------------------------ |
> > > | 成对相似性保持类 | [$\text{ICML'11}$]( https://dl.acm.org/doi/10.5555/3104482.3104483) / [$\text{NIPS'08}$]( https://dl.acm.org/doi/10.5555/2981780.2981999) / [$\text{NIPS'14}$](https://dl.acm.org/doi/10.5555/2969033.2969208) / [$\text{KDD'10}$](https://doi.org/10.1145/1835804.1835946) / [$\text{CVPR'13}$](https://doi.org/10.1109/CVPR.2013.64) |
> > > | 多重相似性保持类 | [$\text{ICCV'13}$]( https://doi.org/10.1109/ICCV.2013.377) / [$\text{MM'13}$](https://doi.org/10.1145/2502081.2502100) |
> > > | 隐式相似性保持类 | [$\text{CVPR'11}$]( https://doi.org/10.1109/CVPR.2011.5995709) / [$\text{ICCV'13}$]( https://doi.org/10.1109/ICCV.2013.377) |
> > > |      量化类      | [$\text{TPAMI'11}$](https://doi.org/10.1109/TPAMI.2010.57) / [$\text{TPAMI'13}$](https://doi.org/10.1109/TPAMI.2012.193) / [$\text{NIPS'12}$](https://dl.acm.org/doi/10.5555/2999134.2999318) |
> > >
> > > :three:关于量化类方法：最有效的$\text{L2H}$方法
> > >
> > > 1. 核心：最小化量化失真 (及 $\min\displaystyle\sum$ 每个数据点$\xleftrightarrow{ }$其最邻近指的差)
> > >
> > > 2. $\text{PQ(Product Quantization)}$算法，[$\text{TPAMI'11}$](https://doi.org/10.1109/TPAMI.2010.57) / [$\text{TIT'06}$](https://doi.org/10.1109/18.720541)($\text{Quantization}$) 
> > >
> > >    - 原理：$\text{M}$维原始向量$\xrightarrow{分割}$$\text{N}$个$\cfrac{\text{M}}{\text{N}}$维子向量$\xrightarrow[(寻求每个子向量最近的质心\text{Index})]{向量量化}$$\text{N}$维短代码(向量)
> > >
> > >    - 改善途径：
> > >
> > >      |      $\textbf{Type}$      | $\textbf{Pub.}$                                              |
> > >      | :-----------------------: | ------------------------------------------------------------ |
> > >      | 改善$\text{PQ}$的索引步骤 | [$\text{TPAMI'13}$](https://doi.org/10.1109/TPAMI.2013.240) / [$\text{CVPR'13}$](https://doi.org/10.1109/CVPR.2013.388) / [$\text{CVPR'15}$](https://doi.org/10.1109/CVPR.2015.7299052) / [$\text{ICCV'13}$](https://doi.org/10.1109/ICCV.2013.424) / [$\text{CVPR'14}$](https://doi.org/10.1109/CVPR.2014.298) |
> > >      | 改善$\text{PQ}$的搜索步骤 | [$\text{CVPR'14}$](https://doi.org/10.1109/CVPR.2014.298) / [$\text{CVPR'12}$](https://doi.org/10.1109/CVPR.2012.6248038) / [$\text{ICASSP'11}$](https://doi.org/10.48550/arXiv.1102.3828) / [$\text{CVPR'16}$](https://doi.org/10.1109/CVPR.2016.221) |
> > >
> > >    - 扩展$\text{PQ}$算法：优化$\text{PQ}$([$\text{CVPR'14}$](https://doi.org/10.1109/CVPR.2014.298))，加性量化([$\text{CVPR'14}$](https://doi.org/10.1109/CVPR.2014.124))，复合量化([$\text{ICML'14}$](https://dl.acm.org/doi/abs/10.5555/3044805.3044986))
> > >
> > > :four:基于神经网络的(无监督)哈希方法
> > >
> > > 1. Semantic哈希：
> > >
> > >    - 原理：构建多层$\text{RBM(Restricted Boltzmann Machines)}$
> > >    - 目标：为文本(文档)学习紧凑的二进制代码
> > >
> > > 2. 如何学习二进制代码
> > >
> > >    - 通过生成二进制代码：
> > >
> > >      |   $\textbf{Type}$    | $\textbf{Pub.}$                                              |
> > >      | :------------------: | ------------------------------------------------------------ |
> > >      |   设计符号激活层以   | [$\text{CVPR'16}$](https://doi.org/10.1109/CVPR.2016.133) / [$\text{CVPR'15}$](https://doi.org/10.1109/CVPR.2015.7298862) / [$\text{IJCAI'17}$](https://doi.org/10.24963/ijcai.2017/429) / [$\text{TIP'17}$](https://doi.org/10.1109/TIP.2017.2678163) |
> > >      | 提出了约束倒数第二层 | [$\text{ECCV'16}$](https://doi.org/10.48550/arXiv.1607.05140) |
> > >
> > >    - 通过重构数据：使用自编码器作为隐藏层，[$\text{CVPR'15}$](https://doi.org/10.48550/arXiv.1501.00756) / [$\text{IPTA'17}$](https://doi.org/10.1109/IPTA.2016.7821007) 
> > >
> > > 3. 二进制约束的优化问题
> > >
> > >    - 成因：必须从哈希函数的输出获得二进制代码，是一个$\text{NP-Hard}$问题
> > >    - 优化：由$\text{relaxation+rounding}$法使二进制代码次优，如离散优化[$\text{NIPS'14}$](https://dl.acm.org/doi/10.5555/2969033.2969208) / [$\text{TPAMI'18}$](https://doi.org/10.1109/TPAMI.2018.2789887) 
>
> ### 1.2. 基于划分的方法
>
> > :one:原理：
> >
> > 1. 构建：将整个高维空间(递归式)划分为多个不相交的区域
> > 2. 核心：默认如果$q$在$r_q$内，则$q$的最邻近也在$r_q$(或其附近)
> >
> > :two:空间的划分方式
> >
> > 1. 枢轴法($\text{pivoting}$)：
> >    - 根据点$\text{-}$轴距来划分点：$\text{VP-Tree}$([$\text{SODA'93}$](https://dl.acm.org/doi/10.5555/313559.313789)) / $\text{Ball Tree}$([$\text{ICML'08}$](https://doi.org/10.1145/1390156.1390171)) 
> > 2. 超平面法($\text{hyperplane}$)：
> >    - 随机方向的超平面：随机投影树([$\text{STOC'08}$](https://doi.org/10.1145/1374376.1374452)) 
> >    - 轴对齐的分离超平面：随机$\text{KD}$树([$\text{CVPR'08}$](https://doi.org/10.1109/CVPR.2008.4587638)/[$\text{TPAMI'14}$](https://doi.org/10.1109/TPAMI.2014.2321376))
> > 3. 紧凑法($\text{compact}$)：
> >    - 将数据划分为簇：[$\text{T-C'75}$](https://doi.org/10.1109/T-C.1975.224297)
> >    - 创建$\text{Voronoi}$划分：[$\text{SPIRE'99}$](https://doi.org/10.1109/SPIRE.1999.796589) / [$\text{ICML'06}$](https://doi.org/10.1145/1143844.1143857) 
>
> ### 1.3. 基于图的方法
>
> > :one:原理：
> >
> > 1. 构建：数据$\xleftrightarrow{对应}$图结点$+$数据邻近关系$\xleftrightarrow{对应}$图边$\xrightarrow{组成}$邻近图
> > 2. 核心：默认邻居的邻居也是邻居
> > 3. 方法：通过迭代扩展邻居的邻居$+$遵循边的最佳优先搜索策略
> >
> > :two:第一大类：构建近似$\text{KNN-Graph}$，图中每个节点**指向**最近的$k$个邻居
> >
> > 1. 在高维空间的应用：[$\text{IJCAI'11}$](https://dl.acm.org/doi/10.5555/2283516.2283615) / [$\text{CVPR'12}$](https://doi.org/10.1109/CVPR.2012.6247790) / [$\text{CoRR'17}$](http://arxiv.org/abs/1701.08475) / [$\text{WWW'11}$](https://doi.org/10.1145/1963405.1963487) 
> > 2. 关于算法初始点：
> >    - 随机初始点：容易陷入局部最优，[$\text{ComACM'80}$](https://doi.org/10.1145/358841.358850) 
> >    - 改进工作：让$\text{LSH}$([$\text{TCYB'14}$](https://doi.org/10.1109/TCYB.2014.2302018)) / 随机$\text{KD}$树([$\text{CoRR'16}$](http://arxiv.org/abs/1609.07228))生成初始点
> >
> > :three:第二大类：$\text{SW(Small-World)-Graph}$，图中任两节点可较少步到达 ([$\text{Nature'20}$](https://doi.org/10.1038/35022643))  
> >
> > 1. $\text{NSW}$方法：通过迭代插入点来构建$\text{SW-Graph}$ ([$\text{IS'14}$](https://doi.org/10.1016/j.is.2013.10.006))
> > 2. $\text{HNSW}$方法：$\text{NSW}$的扩展，最有效的$\text{ANNS}$算法之一 ([$\text{CoRR'16}$](http://arxiv.org/abs/1603.09320)) 

# $\textbf{4. Diversified Proximity Graph}$   

> ## $\textbf{4.1. }$传统$\textbf{KNN}$图：连通性较差
>
> > :one:原因之一：最邻近聚集在一个方向
> >
> > 1. 实例：如下**$\textbf{2-NN}$图**中，搜索路径只能$p\text{→}\{a_3,a_4\}$而不能$p\text{→}b$即使$p\xleftrightarrow{}b$很近
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240928155307724.png" alt="image-20240928155307724" style="zoom: 15%;" />  
> >
> > 2. 咋整：选取邻居时不仅考虑距离，还需考虑角度
> >
> > :two:原因之二：中心性问题
> >
> > 1. 是啥：$\text{KNN}$图中很多的点没有入度，即不作为其他点的最邻近([$\text{JMLR'11}$](https://dl.acm.org/doi/10.5555/1756006.1953015))，如上图点$p$
> >
> > 2. 咋整：将单向边变成双向边
>
> ## $\textbf{4.2. DPG}$  
>
> > :one:相似度：给定$p$及其$K$最邻近列表$\mathcal{L}$，对于$x,y\in{}\mathcal{L}$用角度$\theta(x, y)\text{=}\angle x p y$衡量$xy$的相似度 
> >
> > :two:$\text{DPG}$的构建算法：
> >
> > 1. 算法流程
> >
> >    - 对于$p$，先找出其$K$个最邻近点(组成$\mathcal{L}$列表)
> >
> >    - 从$\mathcal{L}$列表中选择子集$\mathcal{S}$($\text{|}\mathcal{S}\text{|=}\kappa$)使$\mathcal{S}$中两点**平均角度**最大；选择方法遵循以下贪婪启发式算法
> >
> >      <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241012235347641.png" alt="image-20241012235132614" style="zoom: 40%;" /> 
> >
> >    - 将所有边双向化，即$\forall{}u\in\mathcal{S}$将$(p, u)$与$(u, p)$都包括在邻近图中
> >
> > 2. 关于构建算法
> >
> >    - 时间复杂度为$O\left(\kappa^2 K n\right)$，本文也实现了一个简化的**性能较差**版本复杂度为$O\left(K^2 n\right)$ 
> >    - 为$\text{|}\mathcal{S}\text{|=}\kappa$选取$K$至关重要，实证表明$K\text{=}2 \kappa$最佳
> >
> > :three:搜索过程：与$\text{KGraph}$的搜索完全相同

# $\textbf{5. Experiment}$ 

> ## $\textbf{5.1. Experimental Setting}$ 
>
> > ### $\textbf{5.1.1. Compared Algorithms }$
> >
> > > :one:测试的算法：三类(包括本文提出的$\text{DPG}$)，共$\text{19}$种算法
> > >
> > > 1. 基于$\text{LSH}$​：$\small\text{QALSH}$([$\small\text{VLDB'15}$](https://doi.org/10.14778/2850469.2850470))，$\small\text{SRS}$([$\small\text{VLDB'14}$](https://doi.org/10.14778/2735461.2735462))，$\small\text{FALCONN}$([$\small\text{NIPS'15}$](https://arxiv.org/abs/1509.02897))
> > >
> > > 2. 基于$\text{L2H}$：
> > >
> > >    |  算法类型  | 算法                                                         |
> > >    | :--------: | :----------------------------------------------------------- |
> > >    | 基于二进制 | $\small\text{SGH}$([$\small\text{IJCAI'15}$](https://dl.acm.org/doi/10.5555/2832415.2832561))，$\small\text{AGH}$([$\small\text{ICML'11}$](https://dl.acm.org/doi/10.5555/3104482.3104483))，$\small\text{NSH}$([$\small\text{VLDB'15}$](https://doi.org/10.14778/2850583.2850589)) |
> > >    |  基于量化  | $\small\text{OPQ}$([$\small\text{TPAMI'14}$](https://doi.org/10.1109/TPAMI.2013.240))，$\small\text{CQ}$([$\small\text{ICML'14}$](https://dl.acm.org/doi/abs/10.5555/3044805.3044986)) |
> > >    |    其它    | $\small\text{SH}$([$\small\text{SIGKDD'15}$](https://doi.org/10.1145/2783258.2783284))，$\small\text{NAPP}$([$\small\text{VLDB'15}$](https://doi.org/10.48550/arXiv.1506.03163)) |
> > >
> > > 3. 基于划分的：
> > >
> > >    - $\text{FLANN}$类：$\small\text{FLANN/FLANN-HKM/FLANN-KD}$([$\small\text{TPAMI'14}$](https://doi.org/10.1109/TPAMI.2014.2321376))，$\small\text{Annoy}$ 
> > >    - $\text{VP}$树([$\small\text{TPAMI'14}$](https://doi.org/10.1109/TPAMI.2014.2321376)) 
> > >
> > > 4. 基于图的：
> > >
> > >    |      算法类型      | 算法                                                         |
> > >    | :----------------: | :----------------------------------------------------------- |
> > >    |    基于小世界的    | $\small\text{SW}$([$\small\text{IJCAI'15}$](https://dl.acm.org/doi/10.5555/2832415.2832561))，$\small\text{HNSW}$([$\small\text{CoRR’16}$](http://arxiv.org/abs/1603.09320)) |
> > >    | 基于$\text{KNN}$图 | $\small\text{KGraph}$([$\small\text{WWW'11}$](https://doi.org/10.1145/1963405.1963487))，$\small\text{DPG}$(本文) |
> > >    |       基于树       | $\small\text{RCT}$([$\small\text{TPAMI’15}$](https://doi.org/10.1109/TPAMI.2014.2343223 )) |
> > >
> > > :two:测试配置
> > >
> > > 1. 选择并使用来自$\text{NMSLIB}$库中已经实现的几种算法($\text{NAPP/VP-Tree/SW/HNSW}$)
> > >    - $\text{NMSLIB}$库：专用于非度量空间的开源库，实现并提供了诸多高维相似性搜索算法
> > >    - 度量空间：距离计算具备传统的几何性质，比如欧几里得空间
> > > 2. 仔细调整了每个算法的超参数
> > > 3. 关闭了特定的硬件优化，比如禁用$\text{KGraph}$的多线程等
> > >
> > > :three:环境：
> > >
> > > 1. 系统：$\text{Linux}$服务器
> > > 2. 计算：$\text{Intel Xeon e5-2690}+\text{32G RAM}$ 
> > > 3. 编译：$\text{C}$++由$\text{g}$++$\small\text{4.7}$编译，$\text{MATLAB}$由$\text{MATLAB 8.5}$编译
> >
> > ### $\textbf{5.1.2. Datasets and Query Workload}$ 
> >
> > > :one:数据集概述：$\text{18}$个真实数据集(图像/音频/视频/文本)$+$$\text{2}$个合成($\text{Synthetic}$)数据集
> > >
> > > $\small\begin{array}{cccccc}
> > > \hline \text { Name } & n\left(\times 10^3\right) & d & \text { RC } & \text { LID } & \text { Type } \\
> > > \hline \text { Nus }^* & 269 & 500 & 1.67 & 24.5 & \text { Image } \\
> > > \text { Gist }^* & 983 & 960 & 1.94 & 18.9 & \text { Image } \\
> > > \text { Rand }^* & 1,000 & 100 & 3.05 & 58.7 & \text { Synthetic } \\
> > > \text { Glove }^* & 1,192 & 100 & 1.82 & 20.0 & \text { Text } \\
> > > \text { .... } & .... & .... & .... & .... & \text { .... } \\
> > > \hline
> > > \end{array}
> > > $ 
> > >
> > > :two:度量数据集难度的指标
> > >
> > > 1. 相对对比度($\text{RC}$)：
> > >    - 计算：$\text{RC=}\cfrac{\small\text{每两点距离的平均}}{\small\text{每点与其最邻近距离的平均}}$ 
> > >    - 含义：较小的$\text{RC}$会导致最邻近不易区分，导致搜索难度变大
> > > 2. 局部内在维度($\text{LID}$)：数据集在某个局部区域的内在维度，越高意味着结构越复杂难以查询
> > >
> > > :three:查询负载
> > >
> > > 1. 对每个数据集：从每个数据集中移出$\text{200}$个点作为查询点
> > > 2. 对于$\text{k-NN}$图算法：进行性能测试时，默认$\text{k=20}$
>
> ## $\textbf{5.2. Evaluation Measures}$ 
>
> > :one:查询精度指标：运行算法找到$\text{N}$个候选最邻近，将$\text{N}$点按离查询点的距离**排序**，引出以下指标
> >
> > 1. 基础指标：
> >
> >    |        指标        |                             含义                             |
> >    | :----------------: | :----------------------------------------------------------: |
> >    |  $\text{Recall}$   |   $\cfrac{\text{N}个候选项目中真实最邻近的数目}{\text{k}}$   |
> >    | $\text{Precision}$ |   $\cfrac{\text{N}个候选项目中真实最邻近的数目}{\text{k}}$   |
> >    | $\text{F1 Score}$  | $2\text{×}\cfrac{\text{Precision+Recall}}{\text{Precision+Recall}}$ |
> >
> > 2. $\text{AP(Average Precision)}=\cfrac{\displaystyle{}\sum_{i=1}^{\small\text{N}}[\text{P(i)×Rel(i)}]}{\text{N}中真实最邻近数量}$  
> >
> >    - 位置参数$i$：介于$\text{1→N}$间用于标记候选点，$i\text{=}1$表示距离查询点最近，$i\text{=}\text{N}$表示最远
> >
> >    - 相关性标记：将候选$\text{N}$个最邻近点中，真实的最邻近点标记为相关记作$\text{Rel(i)=1}$
> >
> >    - 精确率：定义为$\text{P(i)=}\cfrac{截至位置i时,最邻近的数目}{ i}$ 
> >
> >    - $\text{mAP}$：就是所有查询点的$\text{AP}$的平均，本文采用此指标
> >
> > 3. $\text{Accuracy}=\displaystyle{}\sum_{i=0}^k \cfrac{\text{dist(q, kANN(q)[i])}}{\text{dist(q, kNN(q)[i])}}$参数含义如下，**越接近$1$表示最邻近查找越精准**
> >
> >    - $\text{dist(q, kANN(q)[i])}$：查询点$\xleftrightarrow{距离}$使用某个$\text{ANN}$算法排序后第$i$个最邻近点
> >    - $\text{dist(q, kNN(q)[i])}$：查询点$\xleftrightarrow{距离}$真实的第$i$个最邻近点
> >
> > :two:查询效率(时间)指标：
> >
> > 1. 加速比$\cfrac{\bar{t}}{t^{\prime}}$：即查询时间比上线性暴力扫描的时间
> > 2. 文中还提到了，除了基于图的算法，都可以用调整$\text{N}$的方法调整查询指标
> >
> > :three:其它指标
> >
> > 1. 索引指标：索引构建时间，索引大小，索引内存
> > 2. 可扩展性
>
> ## $\textbf{5.3. Comparison with Each Category}$ 
>
> > :one:第一轮评估的工作
> >
> > 1. 将所有算法置于$\text{Sift/Notre}$数据集上测试
> > 2. 权衡查询速度/召回的$\text{Trade Off}$，以从每个类别中选出算法进行下一轮评估
> >
> > :two:评估：
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240928160353049.png" alt="image-20240928160353049" width=300 /><img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240928160425131.png" alt="image-20240928160425131" width=300 />
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240928160500641.png" alt="image-20240928160500641" width=300 /> <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240928160814641.png" alt="image-20240928160814641" width=300 /> 
> >
> > 1. 评估标准
> >    - 认为相同召回率下速度提升更大的为更优
> >    - 对于算法数据存在外部的($\text{IO}$次数决定速度)，故将**总页数**/**搜索时访问页数**作为速率提升
> > 2. 评估结果：
> >    - $\text{LSH}$类：$\text{SRS/QALSH}$间选取$\text{SRS}$，$\text{FALCONN}$在$\text{L2}$距离下性能缺乏保证故放第二轮
> >    - $\text{L2H}$类：选取$\text{OPQ}$ 
> >    - 分割类：排除$\text{VP-Tree}$
> >    - 基于图的：选取$\text{KGraph}$和$\text{HNSW}$，$\text{DPG}$延后到下一轮
>
> ## $\textbf{5.4. Second Round Evaluation}$ 
>
> > ### $\textbf{5.4.1. Comparison of Search Quality and Time}$
> >
> > > :one:加速比$@\text{Recall=0.8}+\text{Recall}@$加速比$\text{=50}$:
> > >
> > > 1. $\text{DPG}$和$\text{HNSW}$性能最佳
> > > 2. 其中$\text{DPG}$对$\text{KGraph}$的改良显著，尤其在难数据集上
> > > 3. $\text{SRS}$及其拉跨，源于其没有利用数据集的分布
> > >
> > > :two:加速比$@\text{Recall=0→1}$ 
> > >
> > > 1. $\text{HNSW/KGraph/Annoy}$整体性能优越
> > > 2. $\text{DPG/KGraph}$在高$\text{Recall}$下性能优越，但整体不如$\text{HNSW}$等
> > >
> > > :three:$\text{Recall}@$访问数据比$\text{=}0\%\text{→}100\%$ 
> > >
> > > 1. 除$\text{HNSW}$外基于图的算法在百分比低时拉跨，源于其算法入口点随机
> > > 2. $\text{HNSW}$的分层结构中每层入口不随机，所以性能保持优越
> > >
> > > :four:$\text{Accuracy}@$$\text{Recall}\text{=}0\text{→}1$：专为$\text{c-ANN}$设计的$\text{SRS}$和$\text{FLACONN}$性能优越
> >
> > ### $\textbf{5.4.2. Comparison of Indexing Cost  }$
> >
> > > :one:$\cfrac{\text{index size}}{\text{data size}}$的评估
> > >
> > > 1. 索引大小规模
> > >    - 最大：$\text{Annoy}$(大于数据大小)，源于其需要维护数量庞大的$\text{Tree}$结构
> > >    - 最小：$\text{OPQ/SRS}$ 
> > > 2. 索引大小与维度无关：$\text{DPG/KGraph/HNSW/SRS/FLACONN}$ 
> > > 3. 索引大小剧烈变化：$\text{FLANN}$，源于其有三种不同索引结构供选择
> > >
> > > :two:索引构建时间
> > >
> > > 1. 索引时间最小：$\text{FALCOMNN}$，其次是$\text{SRS}$
> > > 2. 索引时间与维度无关：$\text{OPQ}$，源于其涉及子码字的计算
> > > 3. 相比于$\text{DPG/KGraph}$，$\text{DPG}$在图的多样化构建上**没花太多额外时间**
> > >
> > > :three:索引内存成本：$\text{OPO}$在索引构建时内存开销低，由此在大规模数集上高效
>
> ## $\textbf{5.5. Summary  }$ 
>
> > :one:性能排名
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240928171652969.png" alt="image-20240928171652969" style="zoom:38%;" /> 
> >
> > :two:推荐与策略
> >
> > 1. 计算/主存足够时：选择$\text{DPG/HNSW}$，其次选$\text{Annoy}$以在硬件和搜索性能上折中
> > 2. 看重索引构建时间时：选择$\text{FALCONN}$ 
> > 3. 处理大规模数据：$\text{OPQ/SRS}$，源于二者内存成本/构建时间较小

# $\textbf{6. Further Analysis}$ 

> ## $\textbf{6.1. Space Partition-based Approach}$  
>
> > :one:关于$k\text{-Mean}$算法：
> >
> > 1. 什么是$k\text{-Mean}$：将空间分为$\text{k}$个尽可能内部紧凑/互相远离的部分，分为以下两个阶段
> >    - 数据分配：将每个数据点分给最近的聚类中心，复杂度为$O(nk)$ 
> >    - 重置中心点：重新计算每个聚类的中心点，复杂度为$O(n)$ 
> > 2. 关于$k\text{-Mean}$的大聚类数目
> >    - 是什么：在进行聚类时，选取${k}$等于一个很大的数，以至于达到$\Theta{}{(n)}$规模
> >    - 为何不能$k\text{=}\Theta(n)$：数据分配复杂度为$O(nk)\text{=}O(n^2)$，查询(计算与$k$个中心距离)为$O(n)$ 
> >
> > :two:$k\text{-Mean}$类的$\text{ANN}$算法：如何规避$k\text{=}\Theta(n)$
> >
> > 1. $\text{FLANN}/\text{Annoy}$(递归树思想)：每个结点将数据划为$k$块(子节点)直到叶节点
> >
> >    - 二者在基于划分的算法中性能最佳
> >    - $\text{FLANN}$在大多情况下选择$\text{FLANN-HKM}$(层次$k\text{-Mean}$) 
> >
> > 2. $\text{OPQ}$(子空间划分思想)：将整体分为$\text{M}$块$\to$每块中进行$k'\text{-Mean}$($k'$较小)$\to$组合每块聚类结果
> >
> > 3. 实验证明：在$\text{Audio}$类型数据上，除了使用$k\text{-Means}$的暴力方法，$\text{FLANN-HKM}$($\text{L=2}$)最好
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240928171854841.png" alt="image-20240928171748672" width=300 /> 
> >
> > :three:进一步实验证明：多层次$k\text{-Means}$树的$\text{FLANN-HKM}$(类似$\text{Annoy}$)不能提高性能
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240928171949136.png" alt="image-20240928171949136" width=300 /> 
>
> ## $\textbf{6.2. Graph-based Approach  }$  
>
> > :one:为何基于图的算法($\text{KGraph/DPG/HNSW}$)表现好
> >
> > 1. 图结构上：高连通性$+$全局可达性
> > 2. 搜索算法上：得益于高连通性，算法可沿边逼近最邻近$+$存在多条路径(避免局部最优)
> >
> > :two:$\text{KGraph}$在部分数据集上不佳：算法入口点随机$+$缺乏跨聚类的
