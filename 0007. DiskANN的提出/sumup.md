# $\textbf{1. }$写在前面

> ## $\textbf{1.1. }$预备知识与文中标记
>
> > :one:最邻近查询问题：
> >
> > 1. 精确最邻近($k\text{-NN}$)：
> >
> >    - 含义：在数据库$P$中，找到离查询点$q$最近的$k$个点
> >    - 瓶颈：由于维度诅咒的存在，$k\text{-ANN}$无法突破线性暴力扫描，故引入近似最邻近查询
> >
> > 2. 近似最邻近($k\text{-ANN}$)：
> >
> >    - 含义：$k\text{-NN}$的$\text{Recall<100\%}$版本，需要在检索速度和$\text{Recall}$中权衡
> >
> >    - 一些针对$k\text{-ANN}$的索引
> >
> >      |             算法              | 优点                       | 缺点                              |
> >      | :---------------------------: | -------------------------- | --------------------------------- |
> >      |       $\text{k-d Tree}$       | 索引空间小，低维下表现良好 | 维度大于$\text{20}$时性能急剧下降 |
> >      |         $\text{LSH}$          | 最坏情况下有理论保障       | 无法利用数据分布                  |
> >      | 图类($\small\text{HNSW/NSG}$) | 高维度下表现良好           | 索引(图)构建极其耗时              |
> >
> > :two:其它预备知识
> >
> > 1. 倒排索引($\text{Inverted Index}$)：用于快速全文检索的数据结构，示例如下
> >
> >    - 文档
> >
> >      ```txt
> >      Doc1: fat cat rat rat 
> >      Doc2: fat cat 
> >      Doc3: fat
> >      ```
> >
> >    - 构建的倒排索引
> >
> >      ```txt
> >      fat: Doc1 Doc2 Doc3
> >      cat: Doc1 Doc2
> >      rat: Doc1
> >      ```
> >
> > 2. 查询性能指标：
> >
> >    - $m\text{-recall@}n=\cfrac{|A\cap{}B|}{|A|}=\cfrac{|A\cap{}B|}{m}$
> >    - 其中$A\text{=}$查询点真实的前$m$个最邻近集合，$B\text{=}$算法返回的前$n$个结果集合
> >
> > 3. 中位数结点($\text{Medoid}$)：数据集中，到所有其它点平均距离最小的点
> >
> > 4. $\text{PQ}$算法原理：$\text{M}$维原始向量$\xrightarrow{分割}$$\text{N}$个$\cfrac{\text{M}}{\text{N}}$维子向量$\xrightarrow[(寻求每个子向量最近的质心\text{Index})]{向量量化}$$\text{N}$维短代码(向量)
> >
> > :three:本文符号表示
> >
> > |         符号          | 含义与备注                                 |
> > | :-------------------: | ------------------------------------------ |
> > |          $P$          | 数据集(点集)，点集规模用$|P|\text{=}n$表示 |
> > |   $G\text{=}(P, E)$   | 有向图，顶点集为$P$边集为$E$               |
> > | $N_{\text {out }}(p)$ | 查询点$p$的出边集合                        |
> > |    $\mathbf{x}_p$     | 查询点$p$的向量表示                        |
> > |       $d(p, q)$       | 查询点$p$和$q$之间的欧氏距离               |
>
> ## $\textbf{1.2. }$关于大规模索引：本文的背景与研究
>
> >
> >:one:大规模数据的索引：在十亿级个点中找最邻近，主要有以下两种方法
> >
> >1. 倒排索引$\text{+}$数据压缩：$\small\text{FAISS}$([$\small\text{TBDATA'19}$](https://doi.org/10.1109/TBDATA.2019.2921572))，$\small\text{IVFOADC+G+P}$[$\small\text{(ECCV'18)}$](https://doi.org/10.48550/arXiv.1802.02422) 
> >
> >   - 方法概述：
> >
> >     |   方法   | 含义                                                         |
> >     | :------: | ------------------------------------------------------------ |
> >     | 数据分区 | 将数据库点分为多个分区，查询最邻近时只考虑查询点附近几个分区 |
> >     | 数据压缩 | 即产品量化，原始高维向量$\xrightarrow[每个子向量量化成低维]{分为多个低维子向量}$低维向量 |
> >
> >   - 硬件性能：占用内存较小(十亿点索引后小于$\text{64GB}$)$\text{+}$可利用$\text{GPU}$加速压缩后的数据
> >
> >   - 查询性能：对于$\text{1-recall@}k$，$k\text{=1}$时较低$k\text{=100}$时较高
> >
> >2. 分片($\text{Shard}$)法：$\small\text{MRNG}$[$\small\text{(VLDB'19)}$](https://doi.org/10.14778/3303753.3303754) 
> >   - 索引构建：分割数据为多片$\to$为每片构建索引(并加载到内存中)，其中数据维度并未压缩
> >   - 查询流程：查询请求发给每片/特定几片$\to$相应片执行查询$\to$排序合并各片查询结果
> >   - 性能：由于未压缩故占用内存较大，但也正因此查询精准度更高，扩展成本极大
> >
> >:two:大规模索引的难题：物理化$\text{RAM or SSD}$  
> >
> >1. 目前$\text{ANN}$算法的索引都存储在$\text{RAM}$中，可实现高速存取但存储空间贵
> >2. 将$\text{ANN}$索引放在$\text{SSD}$的尝试：
> >   - $\text{SSD}$的读取延时比$\text{RAM}$高出几个数量级，对查询性能是灾难性的
> >   - 改进途径在于，==**减少每次查询所需读取$\textbf{SSD}$磁盘的次数**== 
> >
> >:three:本文对大规模索引的贡献：提出依托图算法$\text{Vamana}$的$\text{DiskANN}$，使$\text{SSD}$也能支持大规模$\text{ANN}$ 
> >
> >1. 关于$\text{Vamana}$图算法：
> >   - 生成图索引直径远小于$\small\text{NSG/HNSW}$的$\to\begin{cases}若放内存中:性能优于\text{HNSW/NSG}\\\\若放磁盘中:减少\text{DiskANN}访问磁盘次数\end{cases}$
> >   - 为分割后的每个(重叠)数据分区构建$\small\text{Vamana}$索引再合并成大索引，与不分割暴力构建性能相当
> >   - $\text{Vamana}$图构建可与$\text{PQ}$结合$\to\begin{cases}放内存中的:压缩向量\\\\放磁盘中的:全精度向量\text{+}构建的图索引\end{cases}$ 
> >2. 关于$\text{DiskANN}$的实际表现：
> >   - 条件：$\text{64GB}$内存，十亿个百维数据点
> >   - 性能：$\text{1-recall@1=95}\%$并且$\text{Latency<5ms}$ 

# $\textbf{2. }$$\small\textbf{Vamana}$图构建算法 

> ## $\textbf{2.1. }$ $\textbf{ANN}$图算法: 构建/剪枝/查询
>
> > :one:图查询算法：贪心搜索$\text{GreedySearch} \left(s, \mathrm{x}_q, k, L\right)$
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/%E5%9B%BE%E7%89%872.png" alt="image-20241019000253117" width=480 /> 
> >
> > :two:图构建与剪枝算法
> >
> > 1. 图构建算法：稀疏邻居域图$\text{SNG}$[$\text{(SODA'93)}$](http://dl.acm.org/citation.cfm?id=313559.313768)，对$\forall{}p\text{∈}P$按照以下方法确定出边   
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241018003422968.png" alt="image-20241018003422968" width=550 /> 
> >
> >    - 目的：使$\text{GreedySearch}$算法能从任意点快速收敛到最邻近的**充分条件** 
> >    - 缺陷：构建时间复杂度为${O(n^2)}+$生成图在直径/密度上无灵活性
> >
> > 2. 图剪枝算法：健壮性剪枝$\text{RobustPrune}(p, R, \alpha, L)$
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/图片4.png" alt="图片4"  width=600 />  
> >
> >    - 目的：构建稀疏图(避免冗余连接)$+$保持连通性
> >    - 关于剪枝条件：对于$p$，如果可以通过其最邻近$p^*$快速到达$p{'}$，那么就没必要直连$p$与$p{'}$ 
>
> ## $\textbf{2.2. Vamana}$
>
> >  :one:$\text{Vamana}$的提出的背景
> >
> >  1. 贪心搜索：算法从初始点$\xrightarrow{接近}$最邻近的**距离递减**模式
> >
> >     |         图结构         |            距离递减模式             | 磁盘读取 |           示例           |
> >     | :--------------------: | :---------------------------------: | :------: | :----------------------: |
> >     |    传统$\text{SNG}$    |          逐步线性缓慢减少           |   高频   | `D → D-d → D-2d →...→ 0` |
> >     | $\text{RobustPrune}$后 | 按以$\alpha\text{>1}$的指数快速收敛 |   低频   | `D → D/α → D/α² →...→ 0` |
> >
> >  2. 全局候选集的剪枝：$\text{RobustPrune}(p, P \backslash\{p\}, \alpha, |P|-1)$，其中$|P|=n$
> >     - 含义：考虑所有的结点可作为$p$的潜在出邻居
> >     - 性能：可使$\text{GreedySearch}(s, p, 1,1)$在$O(\log n)$事件内收敛到$p$，但构建复杂度高达$O(n^2)$ 
> >
> >  3. $\text{Vamana}$改进思路：将候选集大小由$n-1$缩小至$O(\log{}n)$或$O(1)$，使构建复杂度降到$O(n\log{}n)$
> >
> >  :two:$\text{Vamana}$索引算法
> >
> >  1. 算法输入：
> >
> >     |  类型  | 备注                                                         |
> >     | :----: | ------------------------------------------------------------ |
> >     | 数据集 | $P$为数据集(用于构建图$G$)，含$n$个数据点(第$i$点的坐标为$\mathrm{x}_i$) |
> >     |  参数  | $\alpha$为距离阈值(控制剪枝)，$L$为列表大小(控制搜索广度)，$R$为结点出度限制 |
> >
> >  2. 算法流程：
> >
> >     <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/图片7.png" alt="图片6" width=520 /> 
> >
> >     - 添加反向边的意义：确保访问集$V$中所有点都可与$p$连接，使后续搜索可快速收敛到$p$
> >     - 算法需两次遍历：令$\alpha{}\text{=}1$初步构建一次(确保连通)$+$令$\alpha{}\text{>}1$(用户定义)再构建一次(优化收敛)
> >
> >  :three:$\text{Vamana/HNSW/NSG}$的对比
> >
> >  1. 共同：都是用$\text{GreedySearch}\left(s, \mathcal{p}, 1, L\right)$和$\text{RobustPrune}(p, \mathcal{V}, \alpha, R)$来确定$p$邻居
> >
> >  2. 不同：
> >
> >     |      不同点       |    $\text{Vamana}$     |     $\text{HNSW}$      |        $\text{NSG}$         | 备注                                                |
> >     | :---------------: | :--------------------: | :--------------------: | :-------------------------: | --------------------------------------------------- |
> >     |   $\alpha$可调    |           ✅            |  ❌($\alpha\text{=}1$)  |    ❌($\alpha\text{=}1$)     | 使$\small\text{Vamana}$可很好权衡度数和直径         |
> >     | 剪枝$\mathcal{V}$ | $\small{}GS$==访问集== | $\small{}GS$==结果集== |   $\small{}GS$==访问集==    | 使$\small\text{Vamana/NSG}$有长距边(无需层次)       |
> >     |      初始图       |         随机图         |          空图          | 近似$\small{}k\text{-NN}$图 | 随即图质量高于空且成本远低于$\small{}k\text{-NN}$图 |
> >     |       遍历        |          两次          |          一次          |            一次             | 基于观察，二次遍历可以提高图的质量                  |
> >

# $\textbf{3. DiskANN}$总体设计: 索引$\&$搜索

> ## $\textbf{3.0. }$概览
>
> > :one:$\text{DiskANN}$总体工作流程
> >
> > 1. 索引构建：将数据集$P$加载入内存$\to$在$P$上运行$\text{Vamana}$$\to$将生成图存储在$\text{SSD}$上
> > 2. 查询方式：从$\text{SSD}$加载图信息到内存$\to$获取邻居信息$+$计算/比较距离$\to$迭代搜索
> >
> > :two:$\text{DiskANN}$索引布局
> >
> > |     介质     | 每点存储 | 每边(图结构)存储                                  |
> > | :----------: | :------: | :------------------------------------------------ |
> > |     内存     | 压缩向量 | $\text{NULL}$                                     |
> > | $\text{SSD}$ | 原始向量 | 存储每点定长$R$的邻居标识(邻居$\text{<}R$时补$0$) |
> >
> > 1. 关于定长邻居标识：
> >    - 使得$\text{SSD}$对每点的存储(全精度向量$+$邻居标识)都是定长的
> >    - 遍历了偏移的计算($\text{OffSet}_{i}=i\text{×FixedSize}$)，无需再在内存中存储偏移信息
> > 2. $\text{SSD}$存储的扇结构
> >    - 将一点定长的全精度向量$+$邻居标识$\xleftrightarrow{统一放在}$一个扇区中对齐(如$\text{4KB}$) 
> >    - 对$\text{SSD}$的读取以扇为单位，比如读取某点邻居信息时$\to$必定能同时获取该点全精度向量
>
> ## $\textbf{3.1. }$索引构建设计: 面向内存空间的优化
>
> > :one:存在的问题：
> >
> > 1. 构建过程需先将数据点向量载入内存
> > 2. 一股脑将==全部数据==暴力载入内存将导致内存过载
> >
> > :two:构建优化：将数据集$P$进行重叠分簇
> >
> > 1. 步骤：
> >    - 划分：用$k\text{-means}$将$P$分为多干簇(每簇有一中心)，再将$P$所有点分给$\ell\text{>}1$个中心以构成重叠簇
> >    - 索引：在每个重叠簇中执行$\text{Vamana}$算法，构建相应有向边
> >    - 合并：将所有构建的有向边合并在一个图中，完成构建
> > 2. 重叠分簇：为了保证图的连通性，以及后续搜索的$\text{Navigable}$
>
> ## $\textbf{3.2. }$查询方法设计
>
> > ### $\textbf{3.2.1. }$查询方法设计: 面向减少$\textbf{SSD}$访问的优化
> >
> > > :one:$\text{BeamSearch}$算法
> > >
> > > 1. 算法流程：$\text{BeamSearch}$与$\text{GreedySearch}$
> > >
> > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/wrfebargsh.png" alt="wrfebargsh" width=550 /> 
> > >
> > > 2. 优化原理：
> > >
> > >    - 一次性获取多点的邻居信息，有助于减少访问$\text{SSD}$的频次
> > >    - 从$\text{SSD}$获取多个(但少量)随机扇区，与获得一个扇区所需时间几乎相同
> > >
> > > 3. 带宽参数$\text{W}$：
> > >
> > >    - 含义：$\text{BeamSearch}$一次性获取$W$个结点($p^*$及其$W\text{-}1$个邻居)的邻居信息
> > >
> > >    - 选取：当$W$增加时吞吐量增加$+$延时会因$\text{IO}$饱和而恶化，$\text{Trade-Off}$实验结果如下
> > >
> > >      |  $W$大小  | 对性能的影响                                   |
> > >      | :-------: | ---------------------------------------------- |
> > >      |   $W=1$   | $\text{BeamSearch}$退化为$\text{GreedySearch}$ |
> > >      | $W=2,4,8$ | 可在延迟和吞吐量之间取得良好平衡               |
> > >      |  $W>16$   | 容易导致$\text{IO}$队列饱和从而增加延时        |
> > >
> > > :two:$\text{DiskANN}$缓存
> > >
> > > 1. 原理：将$\text{SSD}$中一部分结点缓存到$\text{DRAM(Cache)}$中，以超高速访问并避免访问$\text{SSD}$
> > > 2. 缓存策略：
> > >    - 基于已知的查询分布
> > >    - 从$s$开始在$C\text{=}3,4$跳的结点 (节点数随$C$指数增长$\text{→}C$不宜太大)
> >
> > ### $\textbf{3.2.2. }$查询方法设计: 面向内存空间的优化
> >
> > > :one:存在的问题：
> > >
> > > 1. 查询过程需先将$\text{SSD}$中存储的图结点(向量)载入内存
> > > 2. 将==全精度==向量暴力载入内存会导致内存过载
> > >
> > > :two:查询优化：使==所有结点==能放进内存
> > >
> > > 1. 用$\text{PQ}$将所有$p\text{∈}P$(以及查询点)压成低维$\widetilde{x_p}$并载入内存
> > > 2. 查询时对比近似距离$d\left(\widetilde{x_p}, \mathrm{x}_q\right)$  
> > >
> > > :three:隐式重排($\text{Implicit Re-Ranking}$)：对==一部分点==进行全精度距离计算的隐式方法
> > >
> > > 1. 原理：$\text{BeamSearch}$缓存结点邻居时也会缓存其全精度向量$\to$可精确返回离$q$最近的$L$个候选点
> > > 2. 思考：**这正是有必要在$\textbf{SSD}$中存放全精度向量的原因**

# $\textbf{4. }$评估与对比

> :one:$\text{HNSW/NSG/Vamana}$的$\text{In-Memory}$搜索性能​评估
>
> 1. 实验设置
>
>    | $\textbf{item}$ | 设置                          | 备注                                            |
>    | :-------------: | ----------------------------- | ----------------------------------------------- |
>    |     数据集      | $\text{SIFT1M/GIST1M/DEEP1M}$ | 高维/百万数量级数据集                           |
>    |    物理实现     | 数据都完全加载到内存中        | $\text{Vamana}$也在内存(而非$\text{SSD}$)上实现 |
>
> 2. 实验结果(相同延迟下$\text{Recall}$更高者视为更好)：
>
>    - 所有情况下$\text{NSG}$和$\text{Vamana}$好于$\text{HNSW}$，在最高维数据上$\text{Vamana}$性能最佳
>    - $\text{Vamana}$索引构建时间最快
>
> :two:$\text{HNSW/NSG/Vamana}$的跳数($\text{Hops}$)评估
>
> 1. 跳数：搜索关键路径上**磁盘读取的轮次数**，直接影响搜索延时
> 2. 实验结果：
>    - $\text{Vamana}$可大幅减少跳数(从而快速收敛)，尤其在高维数据上
>    - 随着$\alpha$和最大出度的增加$\text{Vamana}$跳数会减少，而$\text{NSG/HNSW}$的基本不变
>
> :three:$\text{HNSW/NSG/Vamana}$在十亿级数据上的评估
>
> 1. 两种$\text{Vamana}$算法：
>    - 单一$\text{Vamana}$：将十亿级数据整块载入内存暴力构建
>    - 合并$\text{Vamana}$：将十亿级数据进行(重叠)分簇$\to$每簇分别构建$\to$合并
> 2. 实验结果
>    - 单一索引的性能优于合并索引，这源于合并索引需要遍历更多结点才能找到相同邻域
>    - 合并索引也超过了其它算法，故可认为合并索引更能权衡内存$\xleftrightarrow{}$性能
>
> :four:$\text{IVF-based}$方法$\text{/Vamana}$在十亿级数据上的评估
>
> 1. 关于$\text{FAISS}$：由于其性能劣于$\text{IVFOADC+G+P}$以及需要$\text{GPU}$构建索引，故忽略
> 2. 对于$\text{IVFOADC+G+P}$：分别使用$\text{16/32}$字节的$\text{OPQ}$码本构建，$\text{IVFOADC+G+P-32}$性能更优
> 3. 对于$\text{DiskANN}$：在与$\text{IVFOADC+G+P-32}$相同内存占用下比较，性能远优