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

# $\textbf{3. BANG}$的总体设计

> ## $\textbf{3.1. BANG}$的索引架构
>
> > ### $\textbf{3.1.1. }$$\textbf{BANG}$索引的硬件布局
> >
> > > |         结构         | 功能                                         |
> > > | :------------------: | -------------------------------------------- |
> > > |     $\text{RAM}$     | 存放$\text{Vamana}$算法构建的图结构$+$数据点 |
> > > |   $\text{GPU}$内存   | 存放$\text{PQ}$压缩后的向量                  |
> > > | $\text{CPU-GPU}$总线 | 传输压缩向量$\&$协调并行                     |
> >
> > ### $\textbf{3.1.2. BANG}$索引构建: $\textbf{Vamana}$图
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
>
> ## $\textbf{3.2. BANG}$的查询架构  
>
> > ### $\textbf{3.2.1. }$第一阶段: 初始化$\textbf{\&PQ}$表的构建
> >
> > > :one:执行的操作
> > >
> > > 1. 并行处理：为查询集$Q_\rho$中的每个查询$\{q_1,q_2,...,q_{\rho}\}$分配一个独立的$\text{CUDA}$线程$\text{Block}$ 
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
> > ### $\textbf{3.2.2. }$第二阶段: 并行$\textbf{GreedySearch}$主循环
> >
> > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/图片的4.png" alt="图片xcxcxcxcxxx3" width=600 /> 
> > >
> > > :one:前$\text{CPU}$阶段：$\text{CPU}$从内存中获取当前在处理节点$u_i^*$的邻居集$N_i$
> > >
> > >  ⇄ <span style="color:orange;">数据传输：$\text{CPU}\xrightarrow{邻居集N_i}\text{GPU}$</span>  
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
> > ### $\textbf{3.2.3. }$第三阶段: 重排与输出
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

# ==$\textbf{4. BANG}$的微内核设计与并行优化==

> ## $\textbf{4.0. }$概览
>
> > :one:微内核设计及其动态线程块
> >
> > 0. 微内核的意义：将距离计算/排序/更新表操作拆分成更原子化的操作，使内核高度优化
> >
> > 1. 设立独立微内核的操作：
> >
> >    |       阶段       | 有独立微内核的操作                           |
> >    | :--------------: | -------------------------------------------- |
> >    |  第一阶段(建表)  | $\text{PQ}$表构建操作                        |
> >    | 第二阶段(主查询) | 过滤邻居，距离计算，邻居(归并)排序，归并列表 |
> >    |  第三阶段(重排)  | 重排操作                                     |
> >
> > 2. 动态线程块的优化：
> >
> >    - 每个查询分配到一线程块执行，查询过程会依次执行多个内核
> >    - 执行不同内核时按经验调整线程块大小(如计算密集型内核的块更大)，以保证$\text{GPU}$的高占有
> >
> > :two:除内核外其它并行优化的大致思路
> >
> > 1. 硬件优化
> >
> >    - 总线优化：减少$\text{CPU-GPU}$间$\text{PCIe}$的通信量$\to$提高吞吐
> >
> >      | 优化思路                 | 具体措施                                                |
> >      | :----------------------- | ------------------------------------------------------- |
> >      | 减少(总共的)总线传输次数 | 负载平衡，预取/流水线(让$\text{CPU/GPU}$尽量没空闲时间) |
> >      | 降低(一次的)总线传输量   | 传输$\text{PQ}$压缩后的向量(而非原始向量)               |
> >
> >    - $\text{GPU}$内存优化：避免存放图结构$+$只存放$\text{PQ}$压缩后的向量
> >
> > 2. 计算优化
> >
> >    - 加速遍历/搜索：使用$\text{Bloom}$过滤器，快速判断$a\text{∈}A$式命题的真伪
> >    - 加速距离计算：使用$\text{PQ}$压缩后的向量计算距离
>
> ## $\textbf{4.1. }$第一阶段: $\textbf{PQ}$表的构建微内核
>
> > ###  $\textbf{4.1.1. }$ 关于向量压缩的$\textbf{PQ}$办法
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
> > > 0. 给定$k$个$D$维向量$\to\begin{cases}
> > >    \textbf{v}_1=[x_{11},x_{12},x_{13},x_{14},...,x_{1D}]\\\\
> > >     \textbf{v}_2=[x_{21},x_{22},x_{23},x_{24},...,x_{2D}]\\\\
> > >     \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> > >     \textbf{v}_k=[x_{k1},x_{k2},x_{k3},x_{k4},...,x_{kD}]
> > >     \end{cases}$ 
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
> > ### $\textbf{4.1.2. }$$\textbf{PQ}$表的构建内核设计
> >
> > > :one:$\text{PQ}$​压缩：将原有$D$维向量分为$M$个子空间，每个子空间$k\text{-Means}$聚类出$k$个簇/质心
> > >
> > > $\to\begin{cases}数据点\text{: }\begin{cases}
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
> > > :three:构建操作：
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
> > > 2. 合并子表：将$\rho$个$M\text{×}k$维的子表合并为最终$\rho\text{×}M\text{×}k$维表，并存储在$\text{GPU}$内存上
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
> ## ${}{}\textbf{4.2. }$第二阶段: 四大内核$\textbf{+}$两大并行优化
>
> > ### $\textbf{4.2.1. }$中$\textbf{GPU}$阶段的四大$\textbf{GPU}$内核
> >
> > > #### $\textbf{4.2.1.1. }$基于$\textbf{Bloom Filter}$的已访问邻居过滤内核
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
> > > #### $\textbf{4.2.1.2. }$并行邻居距离计算内核
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
> > > #### $\textbf{4.2.1.3. }$邻居的并行归并排序
> > >
> > > > :one:归并排序的$\text{GPU}$并行处理版本
> > > >
> > > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241103004334657.png" alt="image-20241103004334657" style="zoom:33%;" /> 
> > > >
> > > > 1. 分割过程：将待排序数组等分为左右子数组，在对左右子数组递归式等分
> > > >
> > > >    
> >
> > ### $\textbf{4.2.2. }$中$\textbf{GPU}$阶段的两大并行优化(非内核)
> >
> > > #### $\textbf{4.2.2.1. }$全精度向量的异步传输
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
> > > #### $\textbf{4.2.2.2. }$当前(候选)节点$u_i^{*}$的预取
> > >
> > > > :one:优化的契机：避免$\text{GPU}$与$\text{CPU}$二者间存在过长的空闲
> > > >
> > > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241102220240766.png" alt="image-20241102220240766" width=490 /> 
> > > >
> > > > 1. <span style="color:red;">当$\text{CPU}$在获取邻居时，$\text{GPU}$必定保持空闲</span>
> > > > 2. <span style="color:green;">当$\text{GPU}$在计算并更新$\text{WorkList}$的时候，$\text{CPU}$有可能保持空闲</span> 
> > > >
> > > > :two:候选(当前)结点的预取优化
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
> > > > :thumbsup:这一优化使召回率增加了$10\%$ ​