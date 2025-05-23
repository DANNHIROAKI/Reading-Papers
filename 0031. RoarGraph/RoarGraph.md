# $\textbf{1. }$导论

> ## $\textbf{1.1. }$研究背景
>
> > :one:跨模态检索：
> >
> > 1. 含义：使用某个模态的数据作为$\text{query}$，返回另一个模态中语义相似的内容
> > 2. 示例：输入`"Apple"`后，返回苹果的照片
> >
> > :two:模态差距$\text{(gap)}$：不同模态数据即使映射到同一语义空间(比如用$\text{CLIP}$)，其分布特征仍差距显著
> >
> > <img src="https://i-blog.csdnimg.cn/direct/bb909726a7274ce795269abed46d8b05.png" alt="image-20250103234040081" width=400 /> 
> >
> > :three:两种$\text{ANN}$
> >
> > 1. 单模态$\text{ANN}$：查询向量分布$\xleftrightarrow{\text{ID}}$基础数据分布，即查询来源于与数据库数据**相同**的分布
> > 2. 跨模态$\text{ANN}$​：查询向量分布$\xleftrightarrow{\text{OOD}}$基础数据分布，即查询来源于与数据库数据**不同**的分布
>
> ## $\textbf{1.2. }$本文的研究
>
> > :one:研究动机：当前$\text{SOTA}$的$\text{ANN}$​都是单模态的，在$\text{OOD}$负载上表现差
> >
> > :two:研究内容
> >
> > 1. $\text{OOD}$工作负载分析：跨模态后性能下降，源于查询过远$+$标签分散$\text{→}$收敛变慢$/$跳数增加
> >
> >    |        类型        | 查询$\boldsymbol{\xleftrightarrow{距离}}$基础数据 | 查询最邻近$\boldsymbol{i\xleftrightarrow{距离}}$查询最邻近 | 查询$\boldsymbol{\xleftrightarrow{分布}}$基础数据 |
> >    | :----------------: | :-----------------------------------------------: | :--------------------------------------------------------: | :-----------------------------------------------: |
> >    | 单模态$\text{ANN}$ |                   近(基本假设)                    |                        近(基本假设)                        |                    $\text{ID}$                    |
> >    | 跨模态$\text{ANN}$ |                   远(实验得到)                    |                        远(实验得到)                        |                   $\text{OOD}$                    |
> >
> > 2. $\text{RoarGraph}$的提出：  
> >
> >    - 原理：让查询参与图构建$\text{→}$将[查询点$\xleftrightarrow{}$基础点]邻接关系投影到基础点$\text{→}$形成仅有基础点的图
> >
> >    - 意义：让空间上很远但是查询上很近的点相连，从而能高效处理$\text{OOD-ANNS}$
> >
> >       <img src="https://i-blog.csdnimg.cn/direct/4f4512ff6bfc455d9e096f7393b2d773.png" alt="image-20250104005140458" width=390 /> 
> >
> >    - 效果：在跨模态数据集上实现了$\text{QPS}$和$\text{Recall}$指标的提升
>
> ## $\textbf{1.3. }$有关工作
>
> > |         方法         | 核心思想                                | 优缺点                         |
> > | :------------------: | --------------------------------------- | ------------------------------ |
> > |      束搜索终止      | 利用查询训练分类模型判断何时终止搜索    | 提升效率，但训练成本较高       |
> > | 图卷积$\text{(GCN)}$ | 引入$\text{GCN}$学习最优搜索路径        | 路径优化明显，但训练成本较高   |
> > |   $\text{GCN+RL}$    | 强化学习与$\text{GCN}$结合引导搜索路由  | 提升效果显著，但训练成本较高   |
> > |    $\text{GraSP}$    | 概率模型与子图采样学习边重要性          | 性能优化明显，但索引构建成本高 |
> > |    $\text{ScaNN}$    | 结合向量量化和$\text{PQ}$进行分区与压缩 | 压缩与搜索性能高效，但依赖调参 |

# $\textbf{2. }$对$\textbf{OOD}$负载的分析与验证

> ## $\textbf{2.1. }$初步的背景及其验证
>
> > ### $\textbf{2.1.1. }$对模态差距的验证
> >
> > > :one:$\text{OOD}$的量化
> > >
> > > |         距离类型         | 衡量什么                 | 如何理解                           |
> > > | :----------------------: | :----------------------- | :--------------------------------- |
> > > | $\text{Wasserstein}$距离 | 两个分布间的差异         | 把一个分布搬到另一个的**最小代价** |
> > > | $\text{Mahalanobis}$距离 | 一个向量到一个分布的距离 | 一个点相对于一个分布的**异常程度** |
> > >
> > > :one:实验$1$：用$\text{Wasserstein}$距离衡量$\text{OOD}$特性
> > >
> > > 1. 数据集：基础数据集中抽取的无交叉集$B_1/B_2$，$\text{OOD}$的查询集$Q$
> > > 2. 结果：$\text{Wasserstein}(B_1,Q)$和$\text{Wasserstein}(B_2,Q)$，大致是$\text{Wasserstein}(B_1,B_2)$两倍
> > >
> > > :two:实验$2$：用$\text{Mahalanobis}$距离衡量$\text{OOD}$特性
> > >
> > > 1. 数据集：满足分布$P$的基础数据，来自$\text{ID}$查询集的$q_{id}$，来自$\text{OOD}$查询集的$q_{ood}$
> > > 2. 结果：$\text{Mahalanobis}(q_{\text{id}},P)\text{<}\text{Mahalanobis}(q_{\text{ood}},P)$ 
> >
> > ### $\textbf{2.1.2. }\textbf{SOTA-ANN}$在$\textbf{OOD}$任务上的表现
> >
> > > :one:对传统的$\text{SOTA-ANN}$
> > >
> > > |    索引方法     | 在$\textbf{OOD}$上的表现(相比在$\textbf{ID}$上)              |
> > > | :-------------: | ------------------------------------------------------------ |
> > > |  $\text{HNSW}$  | 性能显著下降，在$\text{BeamSearch}$过程显著访问更多的结点(要经历更多跳) |
> > > | $\text{IVF-PQ}$ | 性能显著下降，需要更多的聚类数才能达到相同的$\text{Recall}$  |
> > >
> > > :two:对改进的$\text{ANN}$：针对$\text{OOD-ANNS}$的首个图索引$\text{RobustVamana(OOD-DiskANN)}$  
> > >
> > > 1. 原理：先用$\text{Vamana}$建图，然后再用$\text{RobustStitch}$根据查询向量，连接新的边
> > > 2. 性能：比$\text{DiskANN}$在$\text{OOD}$任务上提升了$\text{40\%}$性能，但是查询速度慢了${\text{×}4\text{-10}}$
>
> ## $\textbf{2.2. }$对$\textbf{OOD}$上$\textbf{ANN}$工作负载的分析
>
> > ### $\textbf{2.2.1. OOD-ANNS}$和$\textbf{ID-ANNS}$的两个差异
> >
> > > :one:两种差异及实验结果
> > >
> > > 1. $\text{OOD}$查询离其最邻近很远：即$\delta\left(q_{\text{ood}}, i^{t h} \text{-NN}_{\text{ood}}\right) \text{≫} \delta\left(q_{\text{id}}, i^{t h} \text{-NN}_{\text{id}}\right)$，左为$i\text{=}1$时的分布结果
> > >
> > > 2. $\text{OOD}$查询的最邻近彼此原理：$100^{t h} \text{-NN}$互相之间的平均距离，实验结果如右
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/5061971fc661498489a62e2b56e2afa8.png" alt="image-20250104144814904" width=230 /> <img src="https://i-blog.csdnimg.cn/direct/419820e665d94db786f5f0b1a2065c12.png" alt="image-20250104150124562" width=230 />  
> > >
> > > :two:对差异的直观理解
> > >
> > > 1. 简单(概念)示例：
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/5298aad2e05c4401ac86e4d98ac10db2.png" alt="image-20250104150619812" width=400 /> 
> > >
> > >    - $\text{ID}$查询：查询与其最邻近在球面上，相互靠近
> > >    - $\text{ODD}$查询：查询在球心，其最邻近在球面上(由此距离较远且查询不多$\text{+}$分散分布)
> > >
> > > 2. 真实示例：真实数据$\text{PCA}$降到二维的视图，$\text{ID}$查询更为集中
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/ff1e0512f7ed4245a7f2c5e4ac3d9725.png" alt="image-20250104151013397" width=540 /> 
> >
> > ### $\textbf{2.2.2. }$为何传统$\textbf{SOTA-ANN}$在$\textbf{ODD}$表现不佳
> >
> > > :zero:传统$\text{ANN}$的设计
> > >
> > > 1. 基于两假设：查询$/$数据同分布$+k$个最近邻彼此相互靠近(邻居的邻居是邻居)，刚好全反的
> > > 2. 设计的思路：
> > >    - 建图：用$\text{BeamSearch}$来构建$\text{KNN}$图$\text{→}$空间中相近的点转化为图中紧密连接的结点
> > >    - 搜索：从中心点开始$\text{GreedySearch}$
> > >
> > > :one:在基于图$\text{ANN}$上：$\text{OOD}$会使得搜索空间增大
> > >
> > > 1. 可识别搜索空间：包围当前访问结点$x$的$B^{s}(x)\text{+}B^{k}\left(1^{\text{st}}\text{-NN}, R\right)$
> > >
> > >    - 球$B^{k}\left(1^{\text{st}}\text{-NN}, R\right)$：以$1^{\text{st}}\text{-NN}$为球心，$k$邻近间互相距离$\delta\left(i^{\text{th}}\text{-NN}, j^{\text{th}}\text{-NN}\right)$最大值为半径
> > >    - 球$B^{s}(x)$：以当前结点$x$为圆心，以$\delta\left(x, i^{\text{th}}\text{-NN}\right)$的最大值(到最远最邻近的距离)为半径
> > >
> > > 2. $\text{OOD}$的影响：搜索空间大幅增大
> > >
> > >    - 对$B^{k}$：由于$\text{OOD}$的性质$R_{\text {ood }}\text{≫}R_{\text{id}}$，这一差异在体积层面放大到$\left(\cfrac{R_{\text {ood }}}{R_{\text{id}}}\right)^D$级别
> > >    - 对$B^{s}$：由于$\text{OOD}$的性质$\delta\left(x, i^{\text{th}}\text{-NN}_{\text{ood}}\right)\text{≫}\delta\left(x, i^{\text{th}}\text{-NN}_{\text{id}}\right)$，使得体积也大幅膨胀
> > >
> > > 3. 对搜索过程的影响：
> > >
> > >    - 对于$\text{ID}$查询：由于最近邻彼此靠近，$\text{GreedySearch}$可以使$B^{s}(x)$轻松收敛
> > >
> > >      ```txt
> > >      起点 -> 近邻1 -> 近邻2 -> 近邻3 (一个小范围内)
> > >      ```
> > >
> > >    - 对于$\text{OOD}$查询：最近邻方向分散难以收敛，需要更大的$\text{Beam}$宽度$/$搜索路径等
> > >
> > >      ```txt
> > >             近邻2
> > >            ↗️     
> > >      起点 -> 近邻1 -> 近邻3 (分散在大范围内)
> > >            ↘️     
> > >             近邻4
> > >      ```
> > >
> > > :two:在基于划分$\text{IVF}$上
> > > 
> > > 1. 原理上：$\text{IVF}$先将原数据分簇
> > >    - $\text{ID}$查询：最邻近集中在少数几个相邻簇中
> > >    - $\text{OOD}$查询：最邻近分散在多个不相邻簇中
> > > 2. 实验上：$\text{OOD}$查询需要扫描更多的簇，性能下降$2.5$倍

# $\textbf{3. RoarGraph}$

> ## $\textbf{3.1. RoarGraph}$的设计思路
>
> > :one:面向解决三种挑战
> >
> > 1. 边的建立：如何连接查询$/$基础两类结点，同时避免基础结点度数太高
> > 2. 搜索效率：查询结点要保持极高出度以覆盖基础节点，但同时也会大幅增加跳数$/$内存开销
> > 3. 连通性：避免出现孤立结点，独立子图
> >
> > :one:大致的设计流程
> >
> > 1. 构建：建立查询$\boldsymbol{\xleftrightarrow{}}$基础二分图$\text{→}$将邻接信息投影到基础点中$\text{→}$增强连接
> > 2. 查询：同样是用$\text{BeamSearch}$ 
>
> ## $\textbf{3.2. RoarGraph}$的构建: 三个阶段
>
> > ### $\textbf{3.2.1. }$阶段$\textbf{1}$: 查询$\boldsymbol{\xleftrightarrow{}}$基础二分图构建
> >
> > > :one:二分图概述：
> > >
> > > 1. 基本概念：将所有的点分为两个集合，所有边必须连接不同子集的点，不能内部连接
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/539e1546388344628ab812ea91ad9b1e.png" alt="image-20250104160314817" width=230 /> 
> > >
> > > 2. 在此处：两子集查询结点$+$基础节点，两种边[查询结点$\text{→}$基础结点]$\text{+}$[查询结点$\text{←}$基础结点]
> > >
> > > :two:构建过程概述
> > >
> > > $\quad$<img src="https://i-blog.csdnimg.cn/direct/5864da038784414e91d0d01ad6314b7f.png" alt="image-20250104164020863" width=350 /> 
> > >
> > > 1. 预处理：计算每个查询向量的真实$N_q\text{-NN}$标签
> > >
> > > 2. 边构建：
> > >
> > >    |          方向          | 操作                                                         |
> > >    | :--------------------: | ------------------------------------------------------------ |
> > >    | 查询点$\text{→}$基础点 | 查询点$\xrightarrow{连接}$查询点的$N_q\text{-NN}$基础点      |
> > >    | 基础点$\text{→}$查询点 | 查询点$\xleftarrow{连接}$查询点的$1\text{-NN}$基础点，查询点$\xrightarrow{断连}$查询点的$1\text{-NN}$基础点 |
> > >
> > > 3. 示例： 
> > >
> > >    ```txt
> > >    预处理: T1 -> X1, X2, X3 (Nq=3)
> > >    边构建: T1 -> X2, X3 
> > >           T1 <- X1 
> > >    ```
> > >
> > >
> > > :two:构建过程分析
> > >
> > > 1. 结点度数的考量：
> > >
> > >    - 高查询结点出度：提高$N_q$值，增加[基础点$\xrightarrow[覆盖性]{重叠性}$查询点]，使多基础点可由同一查询点联系
> > >    - 低基础节点出度：为了解决上述挑战$1$，目的在于提高二分图上的搜索效率
> > >
> > > 2. 边方向的考虑：不进行双向连接，避免二分图搜索时要去检查邻居的邻居($N_q^2$)
> > >
> > >    ```txt
> > >    预处理: T1 -> X1, X2, X3 (Nq=3)
> > >    边构建: T1 -> X1, X2, X3 
> > >           T1 <- X1
> > >           T1 <- X2
> > >           T1 <- X3
> > >    ```
> >
> > ### $\textbf{3.2.2. }$阶段$\textbf{2}$: 领域感知投影
> >
> > > :one:一些分析
> > >
> > > 1. 优化动机：二分图内存消耗高(额外存储了查询节点)，搜索路径长(需要额外经过查询结点)
> > > 2. 关于投影：
> > >    - 目的：移除二分图中的查询结点，并保留从查询分布获得的邻近关系
> > >    - 方式：最简单的可将查询点所连的全部基础点全连接(度数太高)，优化方法如领域感知投影
> > >
> > > :two:投影过程：
> > >
> > > 1. 预处理：
> > >
> > >    - 遍历查询点：获得与查询点相连的最邻近基础点
> > >
> > >      ```txt
> > >      查询Q -> {B1, B2, B3, B4, B5}  (Q连接了5个基础节点)
> > >      ```
> > >
> > >    - 选择中心点：即查询点的$\text{1-NN}$点，作为$\text{Pivot}$
> > >
> > >       ```txt
> > >       查询Q -> {B1, B2, B3, B4, B5}  (Q连接了5个基础节点)
> > >                👆
> > >               pivot
> > >       ```
> > >
> > >    - 排序基础结点：将余下$N_q\text{-NN}$点，按与$\text{Pivot}$的距离排序
> > >    
> > > 2. 感知投影：
> > > 
> > >    - 连接：让中心点与余下点建立连接
> > > 
> > >      ```txt
> > >      B1 -> B2 (最近)
> > >      B1 -> B3 (次近)
> > >     B1 -> B4 (较远)
> > >      B1 -> B5 (最远)
> > >     ```
> > > 
> > >   - 过滤：保证与$\text{Pivot}$连接方向的多样性
> > > 
> > >      <img src="https://i-blog.csdnimg.cn/direct/d84d7b2160b440da9c6feb5791bc246f.png" alt=" " width=200 /> 
> > > 
> > >      |                         条件                          |       含义       |                 操作                  |
> > >      | :---------------------------------------------------: | :--------------: | :-----------------------------------: |
> > >      | $\text{Dist}(X,Y)\text{<}\text{Dist}(\text{Pivot},Y)$ |  该方向已有连接  | 则筛掉$Y$(不与$\text{Pivot}$建立连接) |
> > >     | $\text{Dist}(X,Y)\text{>}\text{Dist}(\text{Pivot},Y)$ | 代表新的搜索方向 | 则保留$Y$(可与$\text{Pivot}$建立连接) |
> > > 
> > >   - 填充：当$\text{Pivot}$的出度小于度数限制，则又重新连接之前过滤掉的结点
> >
> > ### $\textbf{3.2.3. }$连通性增强
> >
> > > <img src="https://i-blog.csdnimg.cn/direct/ae25bdc424d743d698dd3e8e85a9f0f8.png" alt="image-20250104172351909" width=500 /> 
> > >
> > > :one:为何要增强：仅依赖于二分图的覆盖范围，投影图的连通性还太低，对$\text{GreedySearch}$不友好
> > >
> > > :two:增强的方法：
> > >
> > > 1. 检索：从基础集的$\text{Medoid}$开始，对每个基础点执行$\text{BeamSearch}$得到最邻近(作为候选点)
> > > 2. 连边：在不超过度数限制的前提下，让该基础点连接一定数量的候选点作
>
> ## $\textbf{3.3. RoarGraph}$性能的验证
>
> > ### $\textbf{3.3.1. }$实验设置
> >
> > > :one:数据集
> > >
> > > |         数据集         | 描述                                 |      查询集       |     索引集     |
> > > | :--------------------: | ------------------------------------ | :---------------: | :------------: |
> > > | $\text{Text-to-Image}$ | 流行基准数据集，含图像和文本查询向量 | 官方$1\text{w}$条 | 余下不重叠数据 |
> > > |     $\text{LAION}$     | 数百万对图像$-$替代文本对            | 采样$1\text{w}$条 | 余下不重叠数据 |
> > > |    $\text{WebVid}$     | 素材网站获取的字幕和视频对           | 采样$1\text{w}$条 | 余下不重叠数据 |
> > >
> > > :two:超参数设置
> > >
> > > |         模型          | 超参数列表                                                   |
> > > | :-------------------: | ------------------------------------------------------------ |
> > > |     $\text{HNSW}$     | $M\text{=}32$, $\text{efConstruction}\text{=}500$            |
> > > |     $\text{NSG}$      | $R\text{=}64$, $C\text{=}L\text{=}500$                       |
> > > |   $\tau\text{-MNG}$   | $R\text{=}64$, $C\text{=}L\text{=}500$, $\tau\text{=}0.01$   |
> > > | $\text{RobustVamana}$ | $R\text{=}64$, $L\text{=}500$, $\alpha\text{=}1.0$           |
> > > |  $\text{RoarGraph}$   | $N_q\text{=}100$(最近邻候选数量), $M\text{=}35$(出度约束), $L\text{=}500$(候选集大小) |
> > >
> > > :three:性能指标：$\text{Recall@k}$和$\text{QPS}$(检索速度)
> >
> > ### $\textbf{3.3.2. }$实验结果
> >
> > > :one:$\text{QPS}$与召回：$\text{RoarGraph}$最优(超过$\text{RobustVamana}$)，$\text{HNSW/NSG}$差不多, $\tau\text{-MNG}$最差
> > >
> > > <img src="https://i-blog.csdnimg.cn/direct/2969ad3ab0614715b9035aa0077d7f00.png" alt="image-20250104174815800" width=660 /> 
> > >
> > > :two:跳数与召回：$\text{RoarGraph}$跳数显著减少，且随$\text{Recall@}$的$k$增大，减少趋势下降
> > >
> > > <img src="https://i-blog.csdnimg.cn/direct/d0cb9f9104314f20a32fbdb908411a8c.png" alt="image-20250104175148632" width=660 /> 
> > >
> > > :three:消融实验：对比了二分图$/$投影图$/$完整图，可见通过邻域感知投影显著提升性能
> > >
> > > <img src="https://i-blog.csdnimg.cn/direct/9dbad4e41850462d8c9a1d8a080e2d28.png" alt="image-20250104175148632" width=660 /> 
> > >
> > > :four:查询集规模：即查询集大小占基础集大小比重对索引性能的影响；可见起始模型对规模**并不敏感**
> > >
> > > <img src="https://i-blog.csdnimg.cn/direct/50cb7718c5ff4b1ca4ae220034f47c89.png" alt="image-20250104175148632" width=660 /> 
> > >
> > > :five:在$\text{ID}$负载上的性能：$\text{RoarGraph}$依旧能打，和$\text{HNSW}$相当
> > >
> > > <img src="https://i-blog.csdnimg.cn/direct/74ea1abefab84ac9b4922c793721a60c.png" alt="image-20250104175148632" width=660 /> 
> > >
> > > :six:索引开销成本：使用$10\%$数据可大幅降低构建成本，同时保持搜索性能
> > >
> > > $\quad$<img src=https://i-blog.csdnimg.cn/direct/99786c1f116245709141af6a0c05ec01.png alt="image-20250104180704959" width=580 />  
>
> ## $\textbf{3.4. RoarGraph}$的一些讨论
>
> > :one:运用场景：结合大量历史查询数据，用多模态深度学习模型生成嵌入，部署在大型检索$/$推荐系统
> >
> > :two:更新机制：
> >
> > 1. 初始搜索：
> >    - 结点查询：将新插入下新基础节点$v$作为查询，在基础数据集中搜索其最邻近
> >    - 结点筛选：要求最邻近满足，曾在图构建过程中与**至少一个查询点**连接过的基础点
> >    - 反向回溯：对该最邻近点，回溯到与其曾建立过连接的距离最近的查询点$q$ 
> > 2. 子图构建：
> >    - 二分子图：将$q\xleftrightarrow{}N_{\text {out}}\text{∪}v$整合为二分子图
> >    - 邻域投影：将$v$作为$\text{Pivot}$按同样的方式，生成投影图
> >
> > :three:删除操作：采用墓碑标记法$\text{Tombstones}$，即被删结点任参与路由，但排除在搜索结果中
