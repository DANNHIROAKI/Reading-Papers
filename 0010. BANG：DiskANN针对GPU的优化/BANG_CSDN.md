# <span style="color:red;">$\textbf{PART }$Ⅰ: 导论与预备知识</span>

# $\textbf{1. }$导论

> ## $\textbf{1.1. }$关于$\textbf{ANN}$
>
> > :one:高维$k$最邻近查询
> >
> > 1. 精确查询$\text{(NN)}$：
> >    - 含义：找到与给定查询点最近的$k$个数据点
> >    - 困难：由于维度诅咒$\to$难以摆脱暴力扫描$O(n*\text{dimension})$的复杂度
> > 2. 近似查询$\text{(ANN)}$：
> >    - 核心：通过牺牲准确性来换取速度，以减轻维度诅咒
> >    - $\text{On GPU}$：大规模并行处理可以提高$\text{ANN}$吞吐量(固定时间内的查询数量)
> > 3. 基于图的$\text{ANN}$：
> >    - 处理大规模数据最为高效的$\text{ANN}$方法
> >    - $\text{Vamana/DiskANN}$是目前最先进的基于图的$\text{ANN}$(详细的设计[$\text{Click Here}$](https://blog.csdn.net/qq_64091900/article/details/143091781))   
>
> ## $\textbf{1.2. }$$\textbf{ANN}$的$\textbf{GPU}$实现难点
>
> > :one:$\text{GPU}$的内存有限
> >
> > 1. 含义：目前主流$\text{GPU}$内存有限，无法将构建好的图结构完整载入
> >
> > 2. 现有方案：
> >
> >    |      方案      | 描述                                                         | 缺陷                     | 文献                                                         |
> >    | :------------: | ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ |
> >    |      分片      | 将图分片$\to$不断在$\text{CPU}\leftrightarrows{}\text{GPU}$交换片以处理整个图 | $\text{PCIe}$带宽不够    | [$\text{GGNN}$](https://doi.org/10.1109/TBDATA.2022.3161156) |
> >    | 多$\text{GPU}$ | 将图有效分割到所有$\text{GPU}$上以容纳并处理整个图           | 硬件成本高               | [$\text{SONG}$](https://doi.org/10.1109/ICDE48307.2020.00094)/[$\text{FAISS}$](https://doi.org/10.1109/TBDATA.2019.2921572) |
> >    |      压缩      | 压缩图数据维度使图结构能北方进$\text{GPU}$内存               | 召回率下降(只适合小数据) | [$\text{GGNN}$](https://doi.org/10.1109/TBDATA.2022.3161156) |
> >
> > :two:最有硬件使用
> >
> > 1. $\text{GPU}\leftrightarrows{}\text{CPU}$负载平衡：确保二者持续并行工作不空闲，并且数据传输量不超过$\text{PCIe}$极限
> > 2. 主存占用：基于$\text{GPU}$的$\text{ANN}$搜索占用的内存显著增加
>
> ## $\textbf{1.3. BANG}$的总体优化思路
>
> > :one:硬件优化
> >
> > 1. 总线优化：减少$\text{CPU-GPU}$间$\text{PCIe}$的通信量$\to$提高吞吐
> >
> >    | 优化思路                 | 具体措施                                                |
> >    | :----------------------- | ------------------------------------------------------- |
> >    | 减少(总共的)总线传输次数 | 负载平衡，预取/流水线(让$\text{CPU/GPU}$尽量没空闲时间) |
> >    | 降低(一次的)总线传输量   | 传输$\text{PQ}$压缩后的向量(而非原始向量)               |
> >
> >
> > 2.  $\text{GPU}$内存优化：避免存放图结构$+$只存放$\text{PQ}$压缩后的向量
> >
> > :two:计算优化
> >
> > 1. 加速遍历/搜索：使用$\text{Bloom}$过滤器，快速判断$a\text{∈}A$式命题的真伪
> > 2. 加速距离计算：使用$\text{PQ}$压缩后的向量计算距离
> >
> > :three:软件优化：设立微内核，将距离计算/排序/更新表操作拆分成更原子化的操作，以提高并行化

# $\textbf{2. GPU}$架构与$\textbf{CUDA}$编程模型

> ## $\textbf{2.1. }\textbf{GPU}$体系结构
>
> > :one:计算单元组织架构
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241030201057518.png" alt="image-20241030201057518" width=680 /> 
> >
> > |       结构        | 功能                                                         |
> > | :---------------: | ------------------------------------------------------------ |
> > | $\text{CUDA}$核心 | 类似$\text{ALU}$(但远没$\text{CPU}$的灵活)，可执行浮点运算/张量运算/光线追踪(高级核心) |
> > |   $\text{Warp}$   | 多核心共用一个取指/译码器，按$\text{SIMT}$工作(所有线程指令相同/数据可不同) |
> > |    $\text{SM}$    | 包含多组$\text{Warps}$，所有$\text{CUDA}$核心共用一套执行上下文(缓存)$\&$共享内存 |
> >
> > :two:存储层次架构：
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241031001314018.png" alt="image-20241031001314018" width=600 />  
> >
> > 1. 不同$\text{SM}$能够$\text{Access}$相同的$\text{L2 Cache}$ 
> >
> > 2. 显存与缓存之间的带宽极高，但是相比$\text{GPU}$的运算能力仍然有瓶颈
>
> ## $\textbf{2.2. }$$\textbf{CUDA}$编程模型
>
> > :one:$\text{CUDA}$程序简述
> >
> > 1. $\text{CUDA}$程序的两部分
> >
> >    |     程序     |   运行位置   | 主要职责                               |
> >    | :----------: | :----------: | -------------------------------------- |
> >    |  `Host`程序  | $\text{CPU}$ | 任务管理/数据传输/启动$\text{GPU}$内核 |
> >    | `Device`程序 | $\text{GPU}$ | 执行内核/处理数据                      |
> >
> > 2. $\text{Kernel}$即在$\text{GPU}$上运行的函数，如下简单内核定义示例
> >
> >    ```c++
> >    //通过__global__关键字声名内核函数
> >    __global__ void VecAdd(float* A, float* B, float* C)
> >    {
> >       int i = threadIdx.x;
> >       C[i] = A[i] + B[i];
> >    }
> >    int main()
> >    {
> >       //通过<<<...>>>中参数指定执行kernel的CUDA thread数量
> >       VecAdd<<<1, N>>>(A, B, C); 
> >    }
> >    ```
> >
> > :two:线程并行执行架构
> >
> > 1. 线程层次：
> >
> >    |           结构           | 地位                         | 功能                                            |
> >    | :----------------------: | ---------------------------- | ----------------------------------------------- |
> >    |     $\text{Thread}$      | 并行执行最小单元             | 执行$\text{Kernel}$的一段代码                   |
> >    | $\text{Warp(32Threads)}$ | 线程调度的基本单位           | 所有线程以$\text{SIMD}$方式执行相同指令         |
> >    |      $\text{Block}$      | $\text{GPU}$执行线程基本单位 | 使块内线程==内存共享/指令同步==                 |
> >    |      $\text{Grid}$       | 并行执行的最大单元           | 执行整个内核(启动内核时必启动整个$\text{Grid}$) |
> >
> > 2. 线程在计算单元的映射：线程层次$\xleftrightarrow{层次对应}\text{GPU}$物理架构
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241030233145491.png" alt="image-20241030230857521" width=680 /> 
> >
> >    - 注意$\text{SM}$和$\text{Block}$不必$\text{1v1}$对应也可$\text{Nv1}$对应
> >
> > 3. 线程在存储单元的映射
> >
> >    |    线程结构     | 可$\textbf{Access}$的内存结构                          |                 访问速度                  |
> >    | :-------------: | ------------------------------------------------------ | :---------------------------------------: |
> >    | $\text{Thread}$ | 每线程唯一的$\text{Local Memory}$                      | <span style="color: #90EE90;">极快</span> |
> >    | $\text{Block}$  | 每块唯一的$\text{Shared Memory}$(块中每个线程都可访问) |  <span style="color:orange;">较快</span>  |
> >    |    所有线程     | 唯一且共享的$\text{Global Memory}$                     |   <span style="color:red;">较慢</span>    |
>
> ## $\textbf{2.3. CPU}$与$\textbf{GPU}$ 
>
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241102011403143.png" alt="image-20241030175627888" width=600 /> 
> >
> > :one:$\text{CPU/}\text{GPU}$结构对比    
> >
> > |                | $\text{GPU}$                                       | $\text{CPU}$               |
> > | :------------: | -------------------------------------------------- | -------------------------- |
> > |  $\text{ALU}$  | 功能强但数量少(只占$\text{GPU}$小部)，时钟频率极高 | 功能弱但数量大，时钟频率低 |
> > | $\text{Cache}$ | 容量大并分级，缓存后续访问数据                     | 容量很小，用于提高线程服务 |
> > |      控制      | 复杂串行逻辑，如流水/分支预测/乱序执行             | 简单(但大规模)并行逻辑     |
> >
> > :three:$\text{CPU} \xleftrightarrow[数据/指令传输]{\text{PCIe}} \text{GPU}$交互
> >
> > |     设备     | 逻辑地位 |    $\textbf{IO}$模块    | 任务分配             |
> > | :----------: | :------: | :---------------------: | -------------------- |
> > | $\text{GPU}$ |   外设   | $\text{IO Block}$(南桥) | 控制逻辑和任务调度   |
> > | $\text{CPU}$ |   主机   |  $\text{Copy Engine}$   | 执行大量并行计算任务 |

# <span style="color:red;">$\textbf{PART }$Ⅱ:  $\textbf{BANG}$的设计</span> 

# $\textbf{1. BANG}$的总体设计

> ## $\textbf{1.1. BANG}$的索引架构
>
> > ### $\textbf{1.1.1. }$$\textbf{BANG}$索引(硬件)布局
> >
> > > |         结构         | 功能                                                         |
> > > | :------------------: | ------------------------------------------------------------ |
> > > |     $\text{RAM}$     | 存放$\text{Vamana}$算法构建的图结构$+$数据点                 |
> > > |   $\text{GPU}$内存   | 存放$\text{Vamana}$算法构建的图中点经过$\text{PQ}$压缩后的向量 |
> > > | $\text{CPU-GPU}$总线 | 传输压缩向量$\&$协调并行                                     |
> >
> > ### $\textbf{1.1.2. BANG}$索引构建算法: $\textbf{Vamana}$图
> >
> > > :one:$\text{Vamana}$图构建基本操作
> > >
> > > 1. 图查询算法：贪心搜索$\text{GreedySearch} \left(s, \mathrm{x}_q, k, L\right)$
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/23b6a1f08a65497fb5a8dfc97d7f0964.png" alt="image-20241019000253117" width=460 /> 
> > >
> > > 2. 图剪枝算法：健壮性剪枝$\text{RobustPrune}(p, R, \alpha, L)$
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/bd587cf172774e6fa390e31940d1a72e.png" alt="图片4"  width=580 /> 
> > >
> > > :two:$\text{Vamana}$图构建总体流程
> > >
> > > <img src="https://i-blog.csdnimg.cn/direct/f6d574fd5df646438b305b804f314a68.png" alt="图片6" width=500 />  
> >
> > ### $\textbf{1.1.3. BANG}$索引构建方法: 类似$\textbf{DiskANN}$架构
> >
> > > :one:构建步骤：面向面向内存空间的优化
> > >
> > > 1. 划分：用$k\text{-means}$将$P$分为多簇(每簇有一中心)，再将$P$所有点分给$\ell\text{>}1$个中心以构成重叠簇
> > > 2. 索引：在每个重叠簇中执行$\text{Vamana}$算法，构建相应有向边
> > > 3. 合并：将所有构建的有向边合并在一个图中，完成构建
> > >
> > > :two:关于重叠分簇：为了保证图的连通性，以及后续搜索的$\text{Navigable}$
>
> ## $\textbf{1.2. BANG}$的查询架构  
>
> > ### $\textbf{1.2.1. }$第一阶段: 初始化$\textbf{\&PQ}$表的构建
> >
> > > :one:执行的操作
> > >
> > > 1. 并行化：为查询集$Q_\rho$中的每个查询$\{q_1,q_2,...,q_{\rho}\}$分配一个独立的$\text{CUDA}$线程$\text{Block}$ 
> > >
> > > 2. 距离表：在每个线程块上为每个$q_i$计算并构建$\text{PQ}$距离子表，最终合并$\rho$个子表为距离表
> > >
> > > 3. 搜索起点：每个$q_i$从图质心开始，即$\text{CPU}\xleftarrow{传输给}\text{WorkList}\xleftarrow{放入}\textbf{u}_i^*\textbf{(当前/候选点)}\xleftarrow{初始化}\text{Centroid}$ 
> > >
> > > :two:$\text{PQ}$表构建的时序逻辑
> > >
> > > |    时期    | 操作                                                         |
> > > | :--------: | ------------------------------------------------------------ |
> > > | 查询开始前 | 将查询点送入$\text{GPU}$的$\text{Copy}$引擎，在$\text{CUDA}$核心上计算/构建/存储距离表 |
> > > | 查询开始后 | 保留距离表在$\text{GPU}$上直到查询结束                       |
> >
> > ### $\textbf{1.2.2. }$第二阶段: 并行$\textbf{GreedySearch}$主循环
> >
> > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/图片的4.png" alt="图片xcxcxcxcxxx3" width=600 /> 
> > >
> > > :one:前$\text{CPU}$阶段：$\text{CPU}$从内存中获取当前在处理节点$u_i^*$的邻居集$N_i$
> > >
> > > ⇄ <span style="color:orange;">数据传输：$\text{CPU}\xrightarrow{邻居集N_i}\text{GPU}$</span>  
> > >
> > > :two:中$\text{GPU}$阶段：接收$u_i^*$的邻居集$N_i$后，并行地执行内核$\text{\&}$全精度向量的异步传输
> > >
> > > 1. 执行内核：按顺序执行以下内核及操作
> > >
> > >    |   步骤   | 操作                                                         | 内核与否 |
> > >    | :------: | ------------------------------------------------------------ | :------: |
> > >    | 过滤邻居 | 用$\text{Bloom}$==并行检查==$\forall{}n\text{∈}N_i$中未被访问点$\to$并放入${}N_i'$(未访问集)$\text{+}$更新$\text{Bloom}$ |    ✔️     |
> > >    | 距离计算 | 用$\text{PQ}$距离表==并行计算==所有未处理邻居${}n_k\text{∈}N_i'$与查询点$q_i$距离，并存在$\mathcal{D}_i[k]$ |    ✔️     |
> > >    | 邻居排序 | 将${}N_i'$和$\mathcal{D}_i[k]$按与$q_i$的距离执行归并排序，得到排序后的距离$\mathcal{D}_i'$和节点$\mathcal{N}_i'$ |    ✔️     |
> > >    | 合并列表 | 合并当前$\text{WorkLisk}(\mathcal{L}_i)$与新排序的节点列表$\mathcal{N}_i'$形成新的$\mathcal{L}_i$ |    ✔️     |
> > >    | 更新节点 | 又将$\mathcal{L}_i$排序后选取最近的未访问点${}u_i^*$作为下一个当前节点 |    ❌     |
> > >
> > > 2. 异步传输：执行内核的同时，$\text{CPU}$将$u_i^*$的全精度向量传输给$\text{GPU}$$\to$以便后续重排
> > >
> > >  ⇄ <span style="color:orange;">数据传输：$\text{CPU}\xleftarrow{当前节点u_i^*}\text{GPU}$</span>   
> > >
> > > :three:后$\text{CPU}$阶段：若$\mathcal{L}_i$中所有点都被访问过且$|\mathcal{L}_i|\text{=}t$，则认为已经收敛$\to$结束循环
> >
> > ### $\textbf{1.2.3. }$第三阶段: (搜索收敛后的)重排与输出
> >
> > > :one:重排与输出
> > >
> > > 1. 重排的时序逻辑
> > >
> > >    |    时间    | 操作                                                         |         位置         |
> > >    | :--------: | ------------------------------------------------------------ | :------------------: |
> > >    | 搜索过程中 | 用一个数据结构，存储每个$\text{Iter}$中向$\text{CPU}$发送的全精度候选点 | **$\text{CPU→GPU}$** |
> > >    | 搜索完成后 | 计算所有候选点到查询点距离，按全精度距离排序后选取前若干     |     $\text{GPU}$     |
> > >
> > > 2. 输出：选取重排后的$\mathcal{L}_i$中，离$q_i$最近的$k$个节点$\to$作为$k\text{-}$最邻近返回 
> > >
> > > :two:重排的意义：用小成本(仅极小部分即候选点以全精度送往$\text{GPU}$)，补偿由压缩距离产生的误差
>

# ==$\textbf{2. BANG}$的微内核设计与并行优化==

> ## $\textbf{2.0. }$微内核总体设计概览
>
> > :one:设立独立微内核的操作：
> >
> > |       阶段       | 有独立微内核的操作                           |
> > | :--------------: | -------------------------------------------- |
> > |  第一阶段(建表)  | $\text{PQ}$表构建操作                        |
> > | 第二阶段(主查询) | 过滤邻居，距离计算，邻居(归并)排序，归并列表 |
> > |  第三阶段(重排)  | 重排操作                                     |
> >
> > :two:动态线程块的优化：
> >
> > 1. 每个查询分配到一线程块执行，查询过程会依次执行多个内核
> >
> > 2. 执行不同内核时按经验调整线程块大小(如计算密集型内核的块更大)，以保证$\text{GPU}$的高占有
>
> ## $\textbf{2.1. }$第一阶段: $\textbf{PQ}$表的构建微内核
>
> > ###  $\textbf{2.1.1. }$ 关于向量压缩的$\textbf{PQ}$办法
> >
> > > :one:$k\text{-Means}$分簇方法
> > >
> > > 1. 含义：一种无监督学习，用于将数据集分为$k$个簇(每簇一个质心)，使同簇点靠近/异簇点远离
> > >
> > > 2. 流程：
> > >
> > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241031100601961.png" alt="image-20241031100601961" width=320 />  
> > >
> > > :two:$\text{PQ}$算法流程
> > >
> > > 0. 给定$k$个$D$维向量
> > >
> > >    $\begin{cases}
> > >    \textbf{v}_1=[x_{11},x_{12},x_{13},x_{14},...,x_{1D}]\\\\
> > >    \textbf{v}_2=[x_{21},x_{22},x_{23},x_{24},...,x_{2D}]\\\\
> > >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >    \textbf{v}_k=[x_{k1},x_{k2},x_{k3},x_{k4},...,x_{aD}]
> > >    \end{cases}\xleftrightarrow{}
> > >    \begin{cases}
> > >    \textbf{v}_{1}=\{[x_{11},x_{12},x_{13}],[x_{14},x_{15},x_{16}],...,[x_{1(D-1)},x_{1(D-1)},x_{1D}]\}\\\\
> > >    \textbf{v}_{2}=\{[x_{21},x_{22},x_{23}],[x_{24},x_{25},x_{26}],...,[x_{2(D-1)},x_{2(D-1)},x_{2D}]\}\\\\
> > >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >    \textbf{v}_{k}=\{[x_{k1},x_{k2},x_{k3}],[x_{k4},x_{k5},x_{k6}],...,[x_{k(D-1)},x_{k(D-1)},x_{kD}]\}
> > >    \end{cases}$ 
> > >
> > > 1. 分割子空间：将$D$维向量分为$M$个$\cfrac{D}{M}$维向量
> > >
> > >    $子空间1\begin{cases}
> > >    \textbf{v}_{11}=[x_{11},x_{12},x_{13}]\\\\
> > >    \textbf{v}_{21}=[x_{21},x_{22},x_{23}]\\\\
> > >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >    \textbf{v}_{k1}=[x_{k1},x_{k2},x_{k3}]
> > >     \end{cases}\&子空间2
> > >    \begin{cases}
> > >     \textbf{v}_{12}=[x_{14},x_{15},x_{16}]\\\\
> > >    \textbf{v}_{22}=[x_{24},x_{25},x_{26}]\\\\
> > >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >    \textbf{v}_{k2}=[x_{k4},x_{k5},x_{k6}]
> > >    \end{cases}\&...\&子空间M
> > >    \begin{cases}
> > >    \textbf{v}_{1M}=[x_{1(D-1)},x_{1(D-1)},x_{1D}]\\\\
> > >    \textbf{v}_{2M}=[x_{2(D-1)},x_{2(D-1)},x_{2D}]\\\\
> > >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >    \textbf{v}_{kM}=[x_{k(D-1)},x_{k(D-1)},x_{kD}]
> > >    \end{cases}$ 
> > >
> > > 2. 生成$\text{PQ}$编码:
> > >
> > >    $子空间1\begin{cases}
> > >    \textbf{v}_{11}\xleftarrow{替代}\text{Centriod}_{11}\\\\
> > >    \textbf{v}_{21}\xleftarrow{替代}\text{Centriod}_{21}\\\\
> > >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >    \textbf{v}_{k1}\xleftarrow{替代}\text{Centriod}_{k1}
> > >     \end{cases}\&子空间2
> > >    \begin{cases}
> > >     \textbf{v}_{12}\xleftarrow{替代}\text{Centriod}_{12}\\\\
> > >    \textbf{v}_{22}\xleftarrow{替代}\text{Centriod}_{22}\\\\
> > >     \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >    \textbf{v}_{k2}\xleftarrow{替代}\text{Centriod}_{k2}
> > >     \end{cases}\&...\&子空间M
> > >    \begin{cases}
> > >    \textbf{v}_{1M}\xleftarrow{替代}\text{Centriod}_{1M}\\\\
> > >    \textbf{v}_{2M}\xleftarrow{替代}\text{Centriod}_{2M}\\\\
> > >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >    \textbf{v}_{kM}\xleftarrow{替代}\text{Centriod}_{kM}
> > >    \end{cases}$ 
> > >
> > >    - 聚类：在每个子空间上运行$k\text{-Means}$算法(一般$k\text{=}256$)$\to$每个$\textbf{v}_{ij}$都会分到一个$\cfrac{D}{M}$维的质心
> > >
> > >    - 编码：将每个子向量$\textbf{v}_{ij}$所属质心的==索引==作为其$\text{PQ}$编码，并替代原有子向量
> > >
> > > 3. 生成最终的压缩向量$\to\begin{cases}
> > >    \widetilde{\textbf{v}_{1}}=\{\text{Centriod}_{11},\text{Centriod}_{12},...,\text{Centriod}_{1M}\}\\\\
> > >    \widetilde{\textbf{v}_{2}}=\{\text{Centriod}_{21},\text{Centriod}_{22},...,\text{Centriod}_{2M}\}\\\\
> > >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >    \widetilde{\textbf{v}_{k}}=\{\text{Centriod}_{k1},\text{Centriod}_{k2},...,\text{Centriod}_{kM}\}
> > >    \end{cases}$ 
> >
> > ### $\textbf{2.1.2. }$$\textbf{PQ}$表的构建内核设计
> >
> > > :one:$\text{PQ}$压缩：将原有$D$维向量分为$M$个子空间，每个子空间$k\text{-Means}$聚类出$k$个簇/质心
> > >
> > > $\begin{cases}数据点\text{: }\begin{cases}
> > > \textbf{v}_1=[x_{11},x_{12},x_{13},x_{14},...,x_{1D}]\\\\
> > > \textbf{v}_2=[x_{21},x_{22},x_{23},x_{24},...,x_{2D}]\\\\
> > > \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > > \textbf{v}_α=[x_{α1},x_{α2},x_{α3},x_{α4},...,x_{αD}]\\\\
> > > \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > > \textbf{v}_k=[x_{k1},x_{k2},x_{k3},x_{k4},...,x_{kD}]
> > > \end{cases}
> > > \xrightarrow[分割]{\text{PQ}}\begin{cases}
> > > \widetilde{\textbf{v}_{1}}=\{\textbf{Centriod}_{11},\textbf{Centriod}_{12},...,\textbf{Centriod}_{1M}\}\\\\
> > > \widetilde{\textbf{v}_{2}}=\{\textbf{Centriod}_{21},\textbf{Centriod}_{22},...,\textbf{Centriod}_{2M}\}\\\\
> > > \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > > \widetilde{\textbf{v}_{α}}=\{\textbf{Centriod}_{α1},\textbf{Centriod}_{α2},...,\textbf{Centriod}_{αM}\}\\\\
> > > \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > > \widetilde{\textbf{v}_{k}}=\{\textbf{Centriod}_{k1},\textbf{Centriod}_{k2},...,\textbf{Centriod}_{kM}\}
> > > \end{cases}\\\\
> > > 查询点\text{: }\textbf{q}=[q_{1},q_{2},q_{3},q_{4},...,q_{D}]\xrightarrow[分割]{与\text{PQ}子空间的维度划分对齐}\textbf{q}=\{\textbf{q}_1,\textbf{q}_2,...,\textbf{q}_M\}
> > > \end{cases}$  
> > >
> > > :two:线程映射
> > >
> > > 1. 到线程块：每个$q_i$构建$\text{PQ}$距离子表的操作分给一个线程块
> > > 2. 到单线程：每个子空间对应一个独立线程，依次计算$q_s \text{∈} \mathbb{R}^{^{\frac{D}{M}}}$与每个$k$个质心的距离
> > >
> > > :three:构建操作
> > >
> > > 1. 构建子表：在每个子空间中计算查询点$q_s \text{∈} \mathbb{R}^{^{\frac{D}{M}}}$$\xleftrightarrow{\text{Euclidean距离平方}}$所有簇的质心$\xrightarrow{得到}$$M\text{×}k$维子表，一线程负责一子空间
> > >
> > >    | $\text{Dist. Tab.}$ |                        子空间/线程$1$                        |                        子空间/线程$2$                        |   ....   |                        子空间/线程$M$                        |
> > >    | :-----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :------: | :----------------------------------------------------------: |
> > >    |   $\textbf{v}_1$    | <span style="color:red;">$\text{dist}^2(\textbf{q}_1,\textbf{Centriod}_{11})$</span> | <span style="color:red;">$\text{dist}^2(\textbf{q}_2,\textbf{Centriod}_{12})$</span> | **....** | <span style="color:red;">$\text{dist}^2(\textbf{q}_M,\textbf{Centriod}_{1M})$</span> |
> > >    |   $\textbf{v}_2$    | <span style="color:red;">$\text{dist}^2(\textbf{q}_1,\textbf{Centriod}_{21})$</span> | <span style="color:red;">$\text{dist}^2(\textbf{q}_2,\textbf{Centriod}_{22})$</span> | **....** | <span style="color:red;">$\text{dist}^2(\textbf{q}_M,\textbf{Centriod}_{2M})$</span> |
> > >    |    **.........**    |                        **.........**                         |                        **.........**                         | **....** |                        **.........**                         |
> > >    |   $\textbf{v}_k$    | <span style="color:red;">$\text{dist}^2(\textbf{q}_1,\textbf{Centriod}_{k1})$</span> | <span style="color:red;">$\text{dist}^2(\textbf{q}_2,\textbf{Centriod}_{k2})$</span> | **....** | <span style="color:red;">$\text{dist}^2(\textbf{q}_M,\textbf{Centriod}_{kM})$</span> |
> > >
> > > 2. 合并子表：将$\rho$个$M\text{×}k$维的子表合并为最终$\rho\text{×}M\text{×}k$维表，并存储在$\text{GPU}$内存上直到查询结束
> > >
> > > :four:一些说明$\&$算法分析
> > >
> > > 1. 距离计算原理：$\text{dist}(q,\textbf{v}_α)=\displaystyle{}\sum\limits_{i=1}^{M}\text{dist}^2(q_i,\textbf{Centriod}_{αi})$，就是表中$\textbf{v}_α$行所有内容相加
> > >
> > > 2. 参数设定：更具经验设定$k\text{=}256$，由消融实验确定$M\text{=}74$最优
> > >
> > > 3. 算法分析：
> > >
> > >    | $\textbf{Item}$ | 含义                                       | 复杂度                                                    |
> > >    | :-------------: | ------------------------------------------ | :-------------------------------------------------------- |
> > >    |  $\text{Work}$  | 算法串行执行总耗时                         | $O((m \cdot \text{subspace\_size}) \cdot 256 \cdot \rho)$ |
> > >    |  $\text{Span}$  | 算法并行执行耗时，即**最耗时串行步骤**耗时 | $O(m \cdot \text{subspace\_size}) = O(d)$                 |
>
> ## ${}{}\textbf{2.2. }$第二阶段: 主循环的主要内核$\textbf{+}$并行优化
>
> > ### $\textbf{2.2.1. }$中$\textbf{GPU}$阶段的$\textbf{GPU}$微内核
> >
> > > #### $\textbf{2.2.1.1. }$$\textbf{Bloom Filter}$微内核: 为$N_i$过滤已访问点
> > >
> > > > :one:一些背景
> > > >
> > > > 1. 为何要过滤已访问点
> > > >
> > > >    - $\text{Vamana}$图具有建立远程边特性，搜索过程**必定**碰到很多相同点
> > > >    - 如果不过滤邻居会导致大量重复计算，==**实验证明**$\text{Recall}$会下降至原有$0.1$== 
> > > >
> > > > 2. 传统已访问节点追踪方法
> > > >
> > > >    | 描述                                      | 弊端                                                  |
> > > >    | :---------------------------------------- | ----------------------------------------------------- |
> > > >    | 为每个点多划出$\text{1bit}$以标记访问与否 | 对十亿级别点集，会造成百$\text{GB}$级额外存储开销     |
> > > >    | 用优先队列/哈希表存放已访问点             | 队列/哈希表等动态数据结构==不利于$\text{GPU}$的并行== |
> > > >
> > > > :two:$\text{Bloom}$过滤器的原理
> > > >
> > > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241102164205830.png" alt="图片1zxzcfe" width=700 /> 
> > > >
> > > > 1. 组成结构：长为$z$的布尔数组(初始化为全$0$)$+$$k$个哈希函数
> > > >
> > > > 2. 构建过程：给定集合$A$
> > > >
> > > >    - 哈希：对$\forall{}\text{Input}_i\text{∈}A$通过所有$k$个哈希函数，得到$n\text{×}k$个哈希值$f_{Hash}^{i}(\text{Input}_j)\text{ mod }z$
> > > >    - 填充：将布尔数组中$\text{index}=f_{Hash}^{i}(\text{Input}_j)\text{ mod }z$的位由$0$设为$1$ 
> > > >
> > > >    :bulb:如图中：$f_{Hash}^{3}(\text{Input}_4)\text{=}3$所以设$\text{array[3]=1}$ 
> > > >
> > > > 3. 查询过程：给定元素$\text{query}$
> > > >
> > > >    - 哈希：让$\text{query}$通过所有$k$个哈希函数，得到$k$个哈希值$f_{Hash}^{i}(\text{query})\text{ mod }z$ 
> > > >    - 输出：若数组上==所有==索引值为$f_{Hash}^{i}(\text{query})\text{ mod }z$ 的$k$个位置都是$1$，则认为$\text{query∈}A$ 
> > > >
> > > >    :bulb:如图中：如果$\text{query}$的哈希值为$\to\begin{cases}1/3/4\to{}认为\text{query∈}A\\\\1/2/4\to{}不认为\text{query∈A}(源于\text{array[2]=0)}\end{cases}$
> > > >
> > > > :three:本文中$\text{Bloom}$过滤器在的部署
> > > >
> > > > 1. 哈希函数：采用两个$\text{FNV1a}$，为非加密/轻量级
> > > > 2. 工作逻辑：
> > > >    - 构建：将每轮迭代的当前(候选)点${}u_i^*$输入$\text{Bloom}$过滤器以调整布尔数组
> > > >    - 查询：过滤邻居阶段，将$u_i^*$邻居集$N_i$全部输入$\text{Bloom}$过滤器$\to$排除已访问节点
> > >
> > > #### $\textbf{2.2.1.2. }$并行距离计算微内核: 计算${}N_i'$集中所有邻居离查询点的距离
> > >
> > > > :one:距离计算原理：$\text{dist}(q,\textbf{v}_α)$就是表中$\textbf{v}_α$所有<span style="color:red;">子距离</span>相加，对应$M$个子空间共$M$项
> > > >
> > > > | $\text{Dist. Tab.}$ |                          子空间$1$                           |                          子空间$2$                           |   ....   |                          子空间$M$                           |
> > > > | :-----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :------: | :----------------------------------------------------------: |
> > > > |    **.........**    |                        **.........**                         |                        **.........**                         | **....** |                        **.........**                         |
> > > > |   $\textbf{v}_α$    | <span style="color:red;">$\text{dist}^2(\textbf{q}_1,\textbf{Centriod}_{α1})$</span> | <span style="color:red;">$\text{dist}^2(\textbf{q}_2,\textbf{Centriod}_{α2})$</span> | **....** | <span style="color:red;">$\text{dist}^2(\textbf{q}_M,\textbf{Centriod}_{αM})$</span> |
> > > > |    **.........**    |                        **.........**                         |                        **.........**                         | **....** |                        **.........**                         |
> > > >
> > > > :two:距离计算的并行实现 
> > > >
> > > > 1. 线程块的分组结构
> > > >
> > > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241102200735155.png" alt="image-20241102184956227" width=500 /> 
> > > >
> > > >    - 线程块级别：为每个查询$q_i\text{∈}Q_{\rho}$分配一个独立的线程$\text{Block}$ 
> > > >    - 线程组级别：将所有线程分为$g$组，每组线程数为$n\text{:=}\cfrac{\text{{Sum\_Threads\_Num}}}{g}$ 
> > > >
> > > > 2. 关于线程组：负责计算查询点$q_i$与==单个邻居==的距离(故隐式要求$g\text{>}$邻居总数)
> > > >
> > > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241102210607258.png" alt="image-20241102210157060" width=650 />  
> > > >
> > > >    - 计算：将$M$个子空间分为${n}$组，组内每个线程==并行地==将自己组内的$\cfrac{M}{n}$个子距离相加
> > > >
> > > >    - 寄存：组内每个线程将子距离相加的结果，直接放在其本地寄存器==(避免了线程同步开销)==
> > > >
> > > >    - 合并：组内各线程寄存结果相加$\xrightarrow{得到}$查询点与邻居的距离，该步的两种$\text{CUDA}$实现如下
> > > >
> > > >      | $\textbf{CUDA}$函数 | 原理                         | 备注                         |
> > > >      | :-----------------: | ---------------------------- | ---------------------------- |
> > > >      |    `atomicAdd()`    | 将所有结果累加到一个共享变量 | 实现简单，但不适用高并发情况 |
> > > >      |   `WarpReduce()`    | 使用寄存器级规约             | 实验证明当$M=74$时性能略优   |
> > > >
> > > > :three:一些说明$\&$算法分析
> > > >
> > > > 1. 该步骤时最耗时的内核：
> > > >
> > > >    - $\text{WorkList}$中邻居在$\text{GPU}$内存的存储并非连续(未合并)
> > > >    - 计算距离时需要频繁访存$\text{GPU}$内存，而访问显存又极其耗时
> > > >
> > > > 2. 算法分析：$\text{NumNbrs}$即邻居数目
> > > >
> > > >    | $\textbf{Item}$ | 含义                   | 复杂度                                    |
> > > >    | :-------------: | ---------------------- | :---------------------------------------- |
> > > >    |  $\text{Work}$  | 算法串行执行总耗时     | $O(\text{NumNbrs}\cdot M \cdot \rho)$     |
> > > >    |  $\text{Span}$  | 算法最耗时串行步骤耗时 | $O(\log M)$，源于`WarpReduce()`的二分规约 |
> > >
> > > #### $\textbf{2.2.1.3. }$并行归并排序微内核: 为$N_i^{'}$集中所有邻居排序
> > >
> > > > :zero:归并排序流程
> > > >
> > > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241103004334657.png" alt="image-20241103004334657" width=550 /> 
> > > >
> > > > 1. 分割过程：将待排序数组等分为左右子数组，再对左右子数组递归式等分，直至不可分割
> > > > 2. 合并过程：将所有子数组两两递归合并，逐步得到较大有序数组，直到得到完整有序数组
> > > >
> > > > :one:传统的并行归并
> > > >
> > > > 1. 线程映射：为每个合并操作分配一个线程  
> > > >
> > > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241103143638385.png" alt="image-20241103142839348" width=550 /> 
> > > >
> > > > 2. 问题所在：随着归并的进行$\to\begin{cases}同时合并的数组减少\to{}并行工作的线程减少\\\\单次合并的数组更长\to{}单线程运行时间变长\end{cases}\to$导致大量线程排序完成前空闲
> > > >
> > > > :two:并行合并历程
> > > >
> > > > 1. 线程映射：为合并操作中每个列表的每个元素都分配一个线程
> > > >
> > > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241103183520999.png" alt="image-20241103142839348" width=550 /> 
> > > >
> > > > 2. 并行合并历程：对于给定量已排序的待合并表$A$与$B$
> > > >
> > > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241103152045608.png" alt="image-20241103151323537" width=530 /> 
> > > >
> > > >    - 线程索引： 为两列表中每个元素(邻居)分配一线程，线程索引$=$元素在**自己列表**里的位置索引
> > > >    - 位置计算：通过二分查找，找出两列表中每个元素插入**对方列表**后的索引
> > > >    - 列表合并：将每个元素的两个索引相加即得元素在新的合并列表中的索引，由此得到合并列表
> > > >
> > > > :three:基于并行合并历程的归并操作
> > > >
> > > > 1. 线程映射： 
> > > >    - 单线程：为${}N_i'$($u_i^{*}$的未放问邻居集)中每个邻居(即合并时的每个元素)分配一个线程
> > > >    - 线程块：为每个查询(即每个$u_i^{*}$)分配一个线程块，块大小为$\text{Vamana}$图节点==最大邻居数==
> > > > 2. 合并操作：从每个只有单元素的列表开始，依次执行并行合并历程
> > > > 3. 内存分配：由于最大邻居数的限制$\to$排序列表长度较小，全程可将列表放在$\text{GPU}$共享内存
> > > >
> > > > :four:算法分析
> > > >
> > > > 1. 对并行合并历程：令$\ell$为列表长度
> > > >
> > > >    | $\textbf{Item}$ | 含义                   | 复杂度                                 |
> > > >    | :-------------: | ---------------------- | :------------------------------------- |
> > > >    |  $\text{Work}$  | 算法串行执行总耗时     | $O(\ell \cdot \log (\ell))$            |
> > > >    |  $\text{Span}$  | 算法最耗时串行步骤耗时 | $O(\log (\ell))$，源于二分查找不可避免 |
> > > >
> > > > 2. 对整体排序：对长为$n$的数组，由递归得$T(n)=2 \cdot T(n / 2)+n \cdot \log (n)\Rightarrow{}T(n)=O\left(n \cdot \log ^2(n)\right)$，因此
> > > >
> > > >    | $\textbf{Item}$ | 含义                   | 复杂度                                                       |
> > > >    | :-------------: | ---------------------- | :----------------------------------------------------------- |
> > > >    |  $\text{Work}$  | 算法串行执行总耗时     | $O\left(\text{NumNbrs} \cdot \log ^2(\text{NumNbrs}) \cdot \rho\right)$ |
> > > >    |  $\text{Span}$  | 算法最耗时串行步骤耗时 | $O\left(\log ^2(\text{NumNbrs})  \right)$                    |
> > >
> > > #### $\textbf{2.2.1.4. }$列表合并微内核: 合并已排序的$\mathcal{N}_i^{'}$与当前工作列表$\mathcal{L}_i$
> > >
> > > > :one:合并操作：就是并行合并历程，可理解为上一步归并排序的附加步骤
> > > >
> > > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241103160215775.png" alt="image-20241103160215775"  width=730 /> 
> > > >
> > > > :two:复杂度分析：令$\ell$为列表长度
> > > >
> > > > | $\textbf{Item}$ | 含义                   | 复杂度                                 |
> > > > | :-------------: | ---------------------- | :------------------------------------- |
> > > > |  $\text{Work}$  | 算法串行执行总耗时     | $O(\ell \cdot \log (\ell))$            |
> > > > |  $\text{Span}$  | 算法最耗时串行步骤耗时 | $O(\log (\ell))$，源于二分查找不可避免 |
> >
> > ### $\textbf{2.2.2. }$中$\textbf{GPU}$阶段的(非内核)并行优化
> >
> > > #### $\textbf{2.2.2.1. }$全精度向量的异步传输
> > >
> > > > :one:异步传输的基础
> > > >
> > > > 1. 硬件基础：
> > > >
> > > >    - 主存：图在内存以**点全精度向量$+$邻居信息连续且定长存储**组织$\to$使可以==顺序访问==
> > > >
> > > >      <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241102014141202.png" alt="image-20241102014141202" width=550 /> 
> > > >
> > > >    - $\text{GPU}$内存：专设有一段内存，存储每次迭代异步传来的全精度向量，直到最后完成重排
> > > >
> > > > 2. 软件基础：高级的$\text{CUDA}$功能$\to\begin{cases}异步拷贝\text{: }使数据传输时\text{GPU}可执行其它任务\\\\\text{CUDA}流\text{: }即\text{CUDA}核心可并行处理的指令序列\end{cases}$
> > > >
> > > >
> > > > :two:异步传输的实现
> > > >
> > > > 1. 传输的逻辑：
> > > >
> > > >    - 时序逻辑：两次$\text{CPU}\leftrightarrows\text{GPU}$传输间，并行执行$\to{}\begin{cases}\text{CUDA}核心\text{: }执行计算任务\\\\\text{Copy}引擎\text{: }持续传递全精度向量\end{cases}$ 
> > > >
> > > >      <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241102214055217.png" alt="image-20241102020932074" width=450 />  
> > > >
> > > >    - 存取逻辑：得益于顺序存取，故启动异步传输时只需==顺序移动指针==就可获得全精度向量
> > > >
> > > > 2. 相关的$\text{CUDA}$函数：
> > > >
> > > >    | $\text{CUDA}$函数         | 功能                                              |
> > > >    | :------------------------ | ------------------------------------------------- |
> > > >    | `cudaMemcpyAsync()`       | 实现异步拷贝，即使得数据传输/计算同时进行         |
> > > >    | `cudaStreamSynchronize()` | 实现数据依赖，即等必要数据传完后再执行有关计算/流 |
> > >
> > > #### $\textbf{2.2.2.2. }$当前(候选)节点$u_i^{*}$的预取
> > >
> > > > :one:优化的契机：避免$\text{GPU}$与$\text{CPU}$二者间存在过长的空闲
> > > >
> > > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241102220240766.png" alt="image-20241102220240766" width=490 /> 
> > > >
> > > > 1. <span style="color:red;">当$\text{CPU}$在获取邻居时，$\text{GPU}$必定保持空闲</span>
> > > > 2. <span style="color:green;">当$\text{GPU}$在计算并更新$\text{WorkList}$的时候，$\text{CPU}$有可能保持空闲</span> 
> > > >
> > > > :two:候选(当前)节点的预取优化
> > > >
> > > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/图sdrgc片3.png" alt="图sdrgc片3"  width=550 /> 
> > > >
> > > > 1. 时序逻辑：如何减少$\text{GPU}$的等待(空闲)时间
> > > >    - $\text{GPU}$返回真实$u_i^{*}$时$\text{CPU}$就算好了预测$u_i^{*}$的邻居
> > > >    - 一定概率使得下一次迭代开始时，$\text{CPU}$可立即传送邻居信息而不必让$\text{CPU}$等待
> > > > 2. 选取逻辑：如何使(真实$u_i^{*}\text{==}$预测$u_i^{*}$)的概率更大
> > > >    - 预选：选取$N_i^{\prime}$中的最近邻节点$+$当前$\text{WorkList}$中第一个未放问节点
> > > >    - 择优：比较二者离查询点距离$\to$选取之一最优者
> > > >
> > > > :thumbsup:这一优化使召回率增加了$10\%$ 
>
> ## $\textbf{2.3. }$第三阶段: 实现重排的微内核
>
> > :one:重排操作
> >
> > 1. 数据准备：从$\text{GPU}$内存获取所有已异步传输来的全精度邻居
> > 2. 线程映射：
> >    - 线程块：为每个查询(也就是个重排)分配一个线程块
> >    - 单线程：将每个全精度邻居分给一个线程，不同线程并行地进行距离计算
> > 3. 内核操作：
> >    - 距离计算：并行计算查询点$\xleftrightarrow{}$全精度邻居的全精度$L_2$向量
> >    - 归并排序：采用与第二阶段中相同的并行归并历程，对全精度向量排序
> >    - 输出：报告前$k$个最近邻
> >
> > :two:补充说明$\&$算法分析
> >
> > 1. 重充排独立内核：相比搜索阶段的内核，重排操作更为计算密集，建立独立内核有利于线程块大小优化
> >
> > 2. 优化效果：使召回率提升了$10\text{-}15\%$
> >
> > 3. 复杂度分析：$|C|$是查询候选节点最大数量，$d$是全精度向量维度
> >
> >    | $\textbf{Item}$ | 含义                   | 复杂度                                                       |
> >    | :-------------: | ---------------------- | :----------------------------------------------------------- |
> >    |  $\text{Work}$  | 算法串行执行总耗时     | $O\left(\left(d \cdot|C|+|C|\cdot \log ^2(|C|)\right) \cdot \rho\right)$ |
> >    |  $\text{Span}$  | 算法最耗时串行步骤耗时 | $O\left(d+\log ^2(|C|)\right)$                               |

# $\textbf{3. BANG}$的不同版本

> |     $\textbf{BNAG}$版本      | 数据规模 | 优化/改进方法                                                | 效果             |
> | :--------------------------: | :------: | ------------------------------------------------------------ | ---------------- |
> |      原版$\text{BANG}$       |    大    | $\text{NULL}$                                                | $\text{NULL}$    |
> | $\text{IM(In Memory)-BANG}$  |    中    | 将图结构直接放入$\text{GPU}$内存，消除$\text{CPU}\leftrightarrows{}\text{GPU}$通信 | 吞吐量提高$50\%$ |
> | $\text{ED(Exact Dis.)-BANG}$ |    小    | 进一步**直接用精确距离**，省去$\text{PQ}$距离表构建$\&$重排操作 | 召回率提高       |

# <span style="color:red;">$\textbf{PART }$Ⅲ: 实验验证与结论</span> 

# $\textbf{1. }$实验设置

> ## $\textbf{1.1. }$数据/查询集
>
> > :one:数据集的选取：
> >
> > 1. 大型数据集：十亿级
> >
> >    |      数据集       | 描述                                                         | 数据点数量             |     维度     | 分布 |
> >    | :---------------: | ------------------------------------------------------------ | :--------------------- | :----------: | :--: |
> >    |  $\text{DEEP1B}$  | 十亿个图像嵌入，压缩为$\text{96}$维                          | $\text{1,000,000,000}$ | $\text{96}$  | 均匀 |
> >    |  $\text{SIFT1B}$  | 十亿个图像的$\text{128}$维$\text{SIFT}$描述符                | $\text{1,000,000,000}$ | $\text{128}$ | 均匀 |
> >    | $\text{SPACEV1B}$ | 来自$\text{Bing}$的网页文档查询编码，使用$\text{Microsoft SpaceV}$模型 | $\text{1,000,000,000}$ | $\text{100}$ | 均匀 |
> >
> > 2. 中型数据集：亿级
> >
> >    |      数据集       | 描述                                | 数据点数量           |     维度     | 分布 |
> >    | :---------------: | ----------------------------------- | :------------------- | :----------: | :--: |
> >    | $\text{DEEP100M}$ | 从$\text{DEEP1B}$中提取的前一亿个点 | $\text{100,000,000}$ | $\text{128}$ | 均匀 |
> >    | $\text{SIFT100M}$ | 从$\text{SIFT1B}$中提取的前一亿个点 | $\text{100,000,000}$ | $\text{128}$ | 均匀 |
> >
> > 3. 小型数据集：百万级
> >
> >    |      数据集       | 描述                                                       | 数据点数量         |     维度     |  分布  |
> >    | :---------------: | ---------------------------------------------------------- | :----------------- | :----------: | :----: |
> >    | $\text{MNIST8M}$  | $\text{784}$维的手写数字图像数据集，包含变形和平移后的嵌入 | $\text{8,100,000}$ | $\text{784}$ |  均匀  |
> >    |  $\text{GIST1M}$  | 一百万个图像的$\text{960}$维$\text{GIST}$描述符            | $\text{1,000,000}$ | $\text{960}$ |  均匀  |
> >    | $\text{GloVe200}$ | 包含$\text{1,183,514}$个$\text{200}$维的词嵌入             | $\text{1,183,514}$ | $\text{200}$ | 不均匀 |
> >    | $\text{NYTimes}$  | 包含$\text{289,761}$个$\text{256}$维的词嵌入               | $\text{289,761}$   | $\text{256}$ | 不均匀 |
> >
> > :two:查询集的设置：
> >
> > 1. 默认设置：从数据集中随机选取$\text{10000}$查询点
> >
> > 2. 特殊设置：对$\text{SPACEV1B}$选取前$\text{10000}$点，对$\text{GIST1M}$每次选取$\text{1000}$个重复$\text{10}$次
>
> ## $\textbf{1.2.}$ 其它配置
>
> > :one:机器配置
> >
> > 1. 硬件：$\text{Intel Xeon Gold 6326}$处理器，$\text{NVIDIA Ampere A100}$显卡(单个)
> > 2. 软件：$\text{Ubuntu 22.04.01}$系统，$\text{g++ 11.3}$编译器，$\text{nvcc 11.8}$编译器($\text{GPU}$) 
> >
> > :three:$\text{BANG}$的参数设置
> >
> > |         阶段         | 参数                                                         |
> > | :------------------: | ------------------------------------------------------------ |
> > | $\text{DiskANN}$索引 | 最大定点数$R\text{=}64$，工作列表大小$L\text{=}200$，剪枝参数$\alpha\text{=}1.2$ |
> > |   $\text{PQ}$压缩    | 子空间数量$m\text{=}74$(试验确定)                            |
> > |       搜索循环       | 查询批次大小$\rho\text{=}10000$，$\text{WorkList}$大小(结果启发式方法调整为$\text{152}$) |
> >
> > :four:对比方法：
> >
> > 1. 对比基准：$\text{GGNN/SONG/FAISS}$，所有参数均采纳原论文中最优参数
> > 2. 评价指标：$k\text{-recall@}k$召回率，$\text{QPS}$吞吐量($\text{Queries Per Second}$) 

# $\textbf{2. }$实验结果

> ## $\textbf{2.2. }$在不同级别数据集上的表现
>
> > | 数据集 | 相同召回率下的$\text{QPS}$                                   |
> > | :----: | :----------------------------------------------------------- |
> > | 十亿级 | $\textcolor{red}{\text{BANG}}\text{>FAISS>GGNN}$             |
> > |  亿级  | $\textcolor{red}{\text{ED-BANG}}\text{>}\textcolor{red}{\text{IM-BANG}}\text{}\text{≈}\text{GGNN}\text{>}\textcolor{red}{\text{BANG}}$ |
> > | 百万级 | $\text{GGNN}\text{>}\textcolor{red}{\text{ED-BANG}}\text{>}\textcolor{red}{\text{IM-BANG}}\text{>}\textcolor{red}{\text{BANG}}\text{>SONG}$ |
>
> ## $\textbf{2.3. }$其它结果
>
> > :one:$\text{PQ}$压缩比对召回率的影响
> >
> > 1. 关于压缩比：即压缩后数据与原数据的大小比，子空间数$M$越少压缩比就越小
> >
> > 2. 实验结果：
> >
> >    |  $\textbf{M}$  |       压缩比       |  召回率  |
> >    | :------------: | :----------------: | :------: |
> >    |      $74$      |       $0.57$       |   最高   |
> >    | $74\text{→}32$ | $0.57\text{→}0.25$ | 缓慢下降 |
> >    |  $\text{<32}$  |   $\text{<0.25}$   | 快速下降 |
> >
> >    :bulb:可见$M\text{=}74\text{→}32$是召回率和压缩比$\text{Trade-Off}$的最佳区间
> >
> > :two:算法(主循环)迭代次数的研究
> >
> > 1. 查询迭代次数的理论下界：不可少于工作列表$\mathcal{L}$的长度，源于$\mathcal{L}$中每点都必被处理一次
> > 2. 查询迭代次数的实验下界：$95 \%$的查询在$1.1 \mathcal{L}$次迭代内完成

# $\textbf{3. }$结论

> :one:在十亿数据集：$\text{BANG}$在维持超大吞吐量时，依旧保持$\text{Recall}$较高不变
>
> :two:在百万数据集：$\text{BANG}$在大多情况下，优于现有最先进算法