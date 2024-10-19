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
> >
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
> > 1. 贪心搜索：算法从初始点$\xrightarrow{接近}$最邻近的**距离递减**模式
> >
> >    |         图结构         |            距离递减模式             | 磁盘读取 |           示例           |
> >    | :--------------------: | :---------------------------------: | :------: | :----------------------: |
> >    |    传统$\text{SNG}$    |          逐步线性缓慢减少           |   高频   | `D → D-d → D-2d →...→ 0` |
> >    | $\text{RobustPrune}$后 | 按以$\alpha\text{>1}$的指数快速收敛 |   低频   | `D → D/α → D/α² →...→ 0` |
> >
> > 2. 全局候选集的剪枝：$\text{RobustPrune}(p, P \backslash\{p\}, \alpha, |P|-1)$，其中$|P|=n$
> >
> >    - 含义：考虑所有的结点可作为$p$的潜在出邻居
> >    - 性能：可使$\text{GreedySearch}(s, p, 1,1)$在$O(\log n)$事件内收敛到$p$，但构建复杂度高达$O(n^2)$ 
> >
> > 3. $\text{Vamana}$改进思路：将候选集大小由$n-1$缩小至$O(\log{}n)$或$O(1)$，使构建复杂度降到$O(n\log{}n)$
> >
> > :two:$\text{Vamana}$索引算法
> >
> > 1. 算法输入：
> >
> >    |  类型  | 备注                                                         |
> >    | :----: | ------------------------------------------------------------ |
> >    | 数据集 | $P$为数据集(用于构建图$G$)，含$n$个数据点(第$i$点的坐标为$\mathrm{x}_i$) |
> >    |  参数  | $\alpha$为距离阈值(控制剪枝)，$L$为列表大小(控制搜索广度)，$R$为结点出度限制 |
> >
> > 2. 算法流程：
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/图片7.png" alt="图片6" width=500 />   
> >
> >    - 添加反向边的意义：确保访问集$V$中所有点都可与$p$连接，使后续搜索可快速收敛到$p$
> >    - 算法需两次遍历：令$\alpha{}\text{=}1$初步构建一次(确保连通)$+$令$\alpha{}\text{>}1$(用户定义)再构建一次(优化收敛)
> >
> > :three:$\text{Vamana/HNSW/NSG}$的对比
> >
> > 1. 共同：都是用$\text{GreedySearch}\left(s, \mathcal{p}, 1, L\right)$和$\text{RobustPrune}(p, \mathcal{V}, \alpha, R)$来确定$p$邻居
> >
> > 2. 不同：
> >
> >    |      不同点       |    $\text{Vamana}$     |     $\text{HNSW}$      |        $\text{NSG}$         | 备注                                                |
> >    | :---------------: | :--------------------: | :--------------------: | :-------------------------: | --------------------------------------------------- |
> >    |   $\alpha$可调    |           ✅            |  ❌($\alpha\text{=}1$)  |    ❌($\alpha\text{=}1$)     | 使$\small\text{Vamana}$可很好权衡度数和直径         |
> >    | 剪枝$\mathcal{V}$ | $\small{}GS$==访问集== | $\small{}GS$==结果集== |   $\small{}GS$==访问集==    | 使$\small\text{Vamana/NSG}$有长距边(无需层次)       |
> >    |      初始图       |         随机图         |          空图          | 近似$\small{}k\text{-NN}$图 | 随即图质量高于空且成本远低于$\small{}k\text{-NN}$图 |
> >    |       遍历        |          两次          |          一次          |            一次             | 基于观察，二次遍历可以提高图的质量                  |
> >
> >    
> >
> > 
> >
> > 

# 