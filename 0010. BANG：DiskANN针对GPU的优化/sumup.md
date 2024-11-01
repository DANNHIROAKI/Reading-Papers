# 1. Introduction

> ## 1.1. 关于ANN以及ANN-on-GPU
>
> > :one:高维$k$最邻近查询
> >
> > 1. 精确查询$\text{(NN)}$：
> >    - 含义：找到与给定查询点最近的$k$个数据点
> >    - 困难：由于维度诅咒$\to$难以摆脱暴力扫描$O(n*\text{dimension})$的复杂度
> > 2. 近似查询$\text{(ANN)}$：
> >    - 核心：通过牺牲准确性来换取速度，以减轻维度诅咒
> >    - $\small\text{On GPU}$：大规模并行处理可以提高$\text{ANN}$吞吐量(固定时间内的查询数量)
> > 3. 基于图的$\text{ANN}$：
> >    - 处理大规模数据最为高效的$\text{ANN}$方法
> >    - $\text{Vamana/DiskANN}$是目前最先进的基于图的$\text{ANN}$(详细设计[$\text{Click Here}$](https://blog.csdn.net/qq_64091900/article/details/143091781))   
> >
> > :two:图$\text{ANN on GPU}$：核心问题在如何将图结构存储在有限的$\text{GPU}$内存中
> >
> > |         方法         | 概述                                    | 弊端         | 文献                                                         |
> > | :------------------: | --------------------------------------- | ------------ | ------------------------------------------------------------ |
> > |        分片法        | 将图分片轮流进入$\small\text{GPU}$内存  | 巨量总线传输 | $\small\text{GGNN}$[$\small\text{(TBDATA'22)}$](https://doi.org/10.1109/TBDATA.2022.3161156) |
> > |        压缩法        | 压缩向量到低维                          | 牺牲查询精度 | $\small\text{SONG}$[$\small\text{(ICDE'20)}$](https://doi.org/10.1109/ICDE48307.2020.00094)/$\small\text{FAISS}$[$\small\text{(TBDATA'19)}$](https://doi.org/10.1109/TBDATA.2019.2921572) |
> > | 多$\small\text{GPU}$ | 将数据/吞吐量分给每个$\small\text{GPU}$ | 硬件成本提高 | $\small\text{GGNN}$[$\small\text{(TBDATA'22)}$](https://doi.org/10.1109/TBDATA.2022.3161156) |
>
> ## 1.2. $\text{BANG}$概述：$\text{ANN on single GPU}$ 
>
> > :one:结构(硬件)概览：
> >
> > | 结构                       | 功能                                         |
> >| :------------------------- | -------------------------------------------- |
> > | $\small\text{RAM}$         | 存放$\text{Vamana}$算法构建的图结构$+$数据点 |
> >| $\small\text{GPU}$内存     | 存放$\small\text{PQ}$压缩后的向量            |
> > | $\small\text{CPU-GPU}$总线 | 传输压缩向量$\&$协调并行                     |
> > 
> > :two:优化措施
> > 
> > 1. 硬件优化：
> >
> >    - 总线优化：减少$\text{CPU-GPU}$间$\text{PCIe}$的通信量$\to$提高吞吐
> >
> >      | 优化思路                 | 具体措施                                                |
> >    | :----------------------- | ------------------------------------------------------- |
> >      | 减少(总共的)总线传输次数 | 负载平衡，预取/流水线(让$\text{CPU/GPU}$尽量没空闲时间) |
> >    | 降低(一次的)总线传输量   | 传输$\small\text{PQ}$压缩后的向量(而非原始向量)         |
> > 
> >    - $\text{GPU}$内存优化：避免存放图结构$+$只存放$\text{PQ}$压缩后的向量
> > 
> > 2. 计算优化：
> >
> >    - 加速遍历/搜索：使用$\text{Bloom}$过滤器，快速判断$a\text{∈}A$式命题的真伪
> >    - 加速距离计算：使用$\text{PQ}$压缩后的向量计算距离
> > 
> >3. 软件(内核)优化：将距离计算/排序/更新表操作拆分成更原子化的操作，使内核高度优化
> > 
> > :three:模型性能
> >
> > 1. 在十亿数据集：$\text{BANG}$在维持超大吞吐量时，依旧保持$\text{Recall}$较高不变
> >2. 在百万数据集：大多情况下，优于现有最先进算法

# 2. BACKGROUND  

> ## 2.1. GPU架构与CUDA编程模型
>
> > :one:$\text{GPU}$体系结构
> >
> > 1. 计算单元组织架构：
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241030201057518.png" alt="image-20241030201057518" width=680 />  
> >
> >    |          结构           | 功能                                                         |
> >    | :---------------------: | ------------------------------------------------------------ |
> >    | $\small\text{CUDA}$核心 | 类似$\small\text{ALU}$(但远没$\small\text{CPU}$的灵活)，可执行浮点运算/张量运算/光线追踪(高级核心) |
> >    |   $\small\text{Warp}$   | 多核心共用一个取指/译码器，按$\small\text{SIMT}$工作(所有线程指令相同/数据可不同) |
> >    |    $\small\text{SM}$    | 包含多组$\small\text{Warps}$，所有$\small\text{CUDA}$核心共用一套执行上下文(缓存)$\&$共享内存 |
> >
> > 2. 存储层次架构：
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241031001314018.png" alt="image-20241031001314018" width=640 />  
> >    
> >    - 不同$\text{SM}$能够$\text{Access}
> >    - 显存与缓存之间的带宽极高，但是相比$\text{GPU}$的运算能力仍然有瓶颈
> >
> > :two:$\text{CPU}$与$\text{GPU}$
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241030213159183.png" alt="image-20241030175627888" width=540 /> 
> >
> > 1. $\text{CPU/}\text{GPU}$结构对比    
> >
> >    |                | $\text{GPU}$                                             | $\text{CPU}$               |
> >    | :------------: | -------------------------------------------------------- | -------------------------- |
> >    |  $\text{ALU}$  | 功能强但数量少(只占$\small\text{GPU}$小部)，时钟频率极高 | 功能弱但数量大，时钟频率低 |
> >    | $\text{Cache}$ | 容量大并分级，缓存后续访问数据                           | 容量很小，用于提高线程服务 |
> >    |      控制      | 复杂串行逻辑，如流水/分支预测/乱序执行                   | 简单(但大规模)并行逻辑     |
> >
> > 2. $\text{CPU} \xleftrightarrow[数据/指令传输]{\text{PCIe}} \text{GPU}$交互
> >
> >    |          | $\text{GPU}$               | $\text{CPU}$ |
> >    | :------: | -------------------------- | ------------ |
> >    | 逻辑地位 | 外设                       | 主机         |
> >    | 任务分配 | 控制逻辑和任务调度 | 执行大量并行计算任务 |
> >
> > :three:$\text{CUDA}$编程模型
> >
> > 1. $\text{CUDA}$程序简述：
> >
> >    - $\text{CUDA}$程序的两部分
> >
> >       |     程序     |   运行位置   | 主要职责                               |
> >       | :----------: | :----------: | -------------------------------------- |
> >       |  `Host`程序  | $\text{CPU}$ | 任务管理/数据传输/启动$\text{GPU}$内核 |
> >       | `Device`程序 | $\text{GPU}$ | 执行内核/处理数据                      |
> >
> >    - $\text{Kernel}$即在$\text{GPU}$上运行的函数，如下简单内核定义示例
> >
> >       ```c++
> >       //通过__global__关键字声名内核函数
> >       __global__ void VecAdd(float* A, float* B, float* C)
> >       {
> >          int i = threadIdx.x;
> >          C[i] = A[i] + B[i];
> >       }
> >       int main()
> >       {
> >          //通过<<<...>>>中参数指定执行kernel的CUDA thread数量
> >          VecAdd<<<1, N>>>(A, B, C); 
> >       }
> >       ```
> >
> > 2. 线程并行执行架构：
> >
> >    - 线程层次：
> >
> >      |              结构              | 地位                               | 功能                                                  |
> >      | :----------------------------: | ---------------------------------- | ----------------------------------------------------- |
> >      |     $\small\text{Thread}$      | 并行执行最小单元                   | 执行$\small\text{Kernel}$的一段代码                   |
> >      | $\small\text{Warp(32Threads)}$ | 线程调度的基本单位                 | 所有线程以$\small\text{SIMD}$方式执行相同指令         |
> >      |      $\small\text{Block}$      | $\small\text{GPU}$执行线程基本单位 | 使块内线程==内存共享/指令同步==                       |
> >      |      $\small\text{Grid}$       | 并行执行的最大单元                 | 执行整个内核(启动内核时必启动整个$\small\text{Grid}$) |
> >
> >    - 线程映射：线程层次$\xleftrightarrow{层次对应}\text{GPU}$物理架构，注意$\text{SM}$和$\text{Block}$不必$\text{1-1}$对应也可$\text{N-1}$对应
> >
> >      <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241030233145491.png" alt="image-20241030230857521" width=680 />  
> >
> >    - 内存层次：
> >
> >      |    线程结构     | 可$\textbf{Access}$的内存结构                          |                 访问速度                  |
> >      | :-------------: | ------------------------------------------------------ | :---------------------------------------: |
> >      | $\text{Thread}$ | 每线程唯一的$\text{Local Memory}$                      | <span style="color: #90EE90;">极快</span> |
> >      | $\text{Block}$  | 每块唯一的$\text{Shared Memory}$(块中每个线程都可访问) |  <span style="color:orange;">较快</span>  |
> >      |    所有线程     | 唯一且共享的$\text{Global Memory}$                     |   <span style="color:red;">较慢</span>    |
>
> ## 2.2. 向量压缩$\textbf{PQ}$方法
>
> > :one:$k\text{-Means}$分簇方法
> >
> > 1. 含义：一种无监督学习，用于将数据集分为$k$个簇(每簇一个质心)，使同簇点靠近/异簇点远离
> >
> > 2. 流程：
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241031100601961.png" alt="image-20241031100601961" width=350 />  
> >
> > :two:$\text{PQ}$算法流程
> >
> > 0. 给定$k$个$D$维向量$\to\begin{cases}
> >    \textbf{v}_1=[x_{11},x_{12},x_{13},x_{14},...,x_{1D}]\\\\
> >     \textbf{v}_2=[x_{21},x_{22},x_{23},x_{24},...,x_{2D}]\\\\
> >     \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >     \textbf{v}_k=[x_{k1},x_{k2},x_{k3},x_{k4},...,x_{kD}]
> >     \end{cases}$ 
> > 
> > 1. 分割子空间：将$D$维向量分为$M$个$\cfrac{D}{M}$维向量
> > 
> >    $\small子空间1\begin{cases}
> >    \textbf{v}_{11}=[x_{11},x_{12},x_{13}]\\\\
> >    \textbf{v}_{21}=[x_{21},x_{22},x_{23}]\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \textbf{v}_{k1}=[x_{k1},x_{k2},x_{k3}]
> >     \end{cases}\&子空间2
> >    \small\begin{cases}
> >     \textbf{v}_{12}=[x_{14},x_{15},x_{16}]\\\\
> >    \textbf{v}_{22}=[x_{24},x_{25},x_{26}]\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \textbf{v}_{k2}=[x_{k4},x_{k5},x_{k6}]
> >    \end{cases}\&...\&子空间M
> >    \small\begin{cases}
> >    \textbf{v}_{1M}=[x_{1(D-1)},x_{1(D-1)},x_{1D}]\\\\
> >    \textbf{v}_{2M}=[x_{2(D-1)},x_{2(D-1)},x_{2D}]\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \textbf{v}_{kM}=[x_{k(D-1)},x_{k(D-1)},x_{kD}]
> >    \end{cases}$ 
> > 
> > 2. 生成$\text{PQ}$编码:
> > 
> >    $\small子空间1\begin{cases}
> >    \textbf{v}_{11}\xleftarrow{替代}\text{Centriod}_{11}\\\\
> >    \textbf{v}_{21}\xleftarrow{替代}\text{Centriod}_{21}\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \textbf{v}_{k1}\xleftarrow{替代}\text{Centriod}_{k1}
> >     \end{cases}\&子空间2
> >    \small\begin{cases}
> >     \textbf{v}_{12}\xleftarrow{替代}\text{Centriod}_{12}\\\\
> >    \textbf{v}_{22}\xleftarrow{替代}\text{Centriod}_{22}\\\\
> >     \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \textbf{v}_{k2}\xleftarrow{替代}\text{Centriod}_{k2}
> >     \end{cases}\&...\&子空间M
> >    \small\begin{cases}
> >    \textbf{v}_{1M}\xleftarrow{替代}\text{Centriod}_{1M}\\\\
> >    \textbf{v}_{2M}\xleftarrow{替代}\text{Centriod}_{2M}\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \textbf{v}_{kM}\xleftarrow{替代}\text{Centriod}_{kM}
> >    \end{cases}$ 
> > 
> >    - 聚类：在每个子空间上运行$k\text{-Means}$算法(一般$k\text{=}256$)$\to$每个$\textbf{v}_{ij}$都会分到一个$\cfrac{D}{M}$维的质心
> > 
> >    - 编码：将每个子向量$\textbf{v}_{ij}$所属质心的==索引==作为其$\text{PQ}$编码，并替代原有子向量
> > 
> > 3. 生成最终的压缩向量$\to\begin{cases}
> >    \widetilde{\textbf{v}_{1}}=\{\text{Centriod}_{11},\text{Centriod}_{12},...,\text{Centriod}_{1M}\}\\\\
> >    \widetilde{\textbf{v}_{2}}=\{\text{Centriod}_{21},\text{Centriod}_{22},...,\text{Centriod}_{2M}\}\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \widetilde{\textbf{v}_{k}}=\{\text{Centriod}_{k1},\text{Centriod}_{k2},...,\text{Centriod}_{kM}\}
> >    \end{cases}$ 
> > 
>
> ## 2.3. Vamana图算法与DiskANN
>
> > ### 2.3.1. Vamana图构建算法
> >
> > > :one:算法的基本操作
> > >
> > > 1. 图查询算法：贪心搜索$\text{GreedySearch} \left(s, \mathrm{x}_q, k, L\right)$
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/23b6a1f08a65497fb5a8dfc97d7f0964.png" alt="image-20241019000253117" width=460 /> 
> > >
> > > 2. 图剪枝算法：健壮性剪枝$\text{RobustPrune}(p, R, \alpha, L)$
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/bd587cf172774e6fa390e31940d1a72e.png" alt="图片4"  width=580 /> 
> > >
> > > :two:$\text{Vamana}$图构建流程
> > >
> > > <img src="https://i-blog.csdnimg.cn/direct/f6d574fd5df646438b305b804f314a68.png" alt="图片6" width=500 /> 
> >
> > ### 2.3.1. DiskANN架构
> >
> > > :one:总体设计
> > >
> > > 1. 工作流程：
> > >
> > >    - 索引构建：将数据集$P$加载入内存$\to$在$P$上运行$\text{Vamana}$$\to$将生成图存储在$\text{SSD}$上
> > >    - 查询方式：从$\text{SSD}$加载图信息到内存$\to$获取邻居信息$+$计算/比较距离$\to$迭代搜索
> > >
> > > 2. 索引布局：
> > >
> > >    |     介质     | 每点存储 | 每边(图结构)存储                                  |
> > >    | :----------: | :------: | :------------------------------------------------ |
> > >    |     内存     | 压缩向量 | $\text{NULL}$                                     |
> > >    | $\text{SSD}$ | 原始向量 | 存储每点定长$R$的邻居标识(邻居$\text{<}R$时补$0$) |
> > >
> > > :two:索引构建设计: 将数据集$P$进行重叠分簇$\to$防止内存过载
> > >
> > > 1. 步骤：
> > >    - 划分：用$k\text{-means}$将$P$分为多簇(每簇有一中心)，再将$P$所有点分给$\ell\text{>}1$个中心以构成重叠簇
> > >    - 索引：在每个重叠簇中执行$\text{Vamana}$算法，构建相应有向边
> > >    - 合并：将所有构建的有向边合并在一个图中，完成构建
> > > 2. 重叠分簇：为了保证图的连通性，以及后续搜索的$\text{Navigable}$ 
> > >
> > > :three:查询方法设计: 面向减少$\textbf{SSD}$访问的优化
> > >
> > > 1. 查询算法：$\text{GreedySearch}\xrightarrow{一次性从\text{SSD}获取多点的邻居}\text{BeamSearch}$ 
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/98408f9411f24befaafa5d033c53476a.png" alt="wrfebargsh" width=520 /> 
> > >
> > > 2. $\text{DiskANN}$缓存：将$\text{SSD}$部分节点缓存到$\text{DRAM(Cache)}$中，以超高速访问并避免访问$\text{SSD}$
> > >
> > > :four:查询方法设计: 面向内存空间的优化
> > >
> > > 1. 压缩：用$\text{PQ}$将所有$p\text{∈}P$(以及查询点)压成低维$\widetilde{x_p}$并载入内存$\to$查询时对比近似距离$d\left(\widetilde{x_p}, \mathrm{x}_q\right)$  
> > >
> > > 2. 重排：隐式地发生在$\text{BeamSearch}$阶段
> > >
> > >    - 邻居扩展阶段：用近似距离得到若干候选点
> > >
> > >      :warning:这一过程点的==全精度向量==会随邻居信息一起载入内存，这是由于==二者位于磁盘同一块上== 
> > >
> > >    - 剪裁阶段：按全精度距离将候选点排序，返回离$q$最近的$L$个候选点

# 3. ANN ON GPU的挑战

> ## 3.1. GPU内存有限
>
> > :one:问题与思路
> >
> > 1. 存在的问题：目前主流$\text{GPU}$内存有限，无法将构建好的图结构完整载入
> >
> > 2. 解决方案：
> >
> >    |         方案         | 描述                                                         | 缺陷                        |
> >    | :------------------: | ------------------------------------------------------------ | --------------------------- |
> >    |         分片         | 将图分片$\to$不断在$\small\text{CPU}\leftrightarrows{}\text{GPU}$交换片以处理整个图 | $\small\text{PCIe}$带宽不够 |
> >    | 多$\small\text{GPU}$ | 将图有效分割到所有$\small\text{GPU}$上以容纳并处理整个图     | 硬件成本高                  |
> >    |         压缩         | 压缩图数据维度使图结构能北方进$\small\text{GPU}$内存         | 召回率下降(只适合小数据)    |
> >
> > :two:本文的解决方案
> >
> > 1. 主机(内存/$\text{CPU}$)：存储图结构，并进行$\text{ANN}$搜索
> > 2. $\text{GPU}$：只负责使用压缩数据进行距离计算
> > 3. 总线：传输$\text{PQ}$压缩的向量$+$邻居信息给$\text{GPU}$ 
>
> ## 3.2. 最优硬件使用
>
> > :one:$\text{GPU}\leftrightarrows{}\text{CPU}$负载平衡：确保二者持续并行工作不空闲，并且数据传输量不超过$\text{PCIe}$极限
> >
> > :two:内存占用：基于$\text{GPU}$的$\text{ANN}$搜索占用的内存显著增加

# 4. BANG简述

> <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241031194540440.png" alt="image-20241031194540440" width=520 /> 
>
> :three:重排
>
> 1. 重排流程：
>
>    |    时间    | 操作                                                         |         位置         |
>    | :--------: | ------------------------------------------------------------ | :------------------: |
>    | 搜索过程中 | 用一个数据结构，存储每个$\text{Iter}$中向$\text{CPU}$发送的全精度候选点 | **$\text{CPU/GPU}$** |
>    | 搜索完成后 | 计算所有候选点到查询点距离，按全精度距离排序后选取前若干     |     $\text{GPU}$     |
> 
> 2. 重排目的：用小成本(仅极小部分即候选点以全精度送往$\text{GPU}$)，补偿由压缩距离产生的误差

# 5. BANG详述

> ## 5.1. BANG的搜索算法
>
> > :one:在$\text{Vamana}$图上的并行化$\text{GreedySearch}$  
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/图片的4.png" alt="图片xcxcxcxcxxx3" width=520 />   
> >
> > 1. 初始化：
> >
> >    - 并行处理：为查询集$Q_\rho$中的每个查询$\{q_1,q_2,...,q_{\rho}\}$分配一个独立的$\text{CUDA}$线程$\text{Block}$ 
> >    - 距离表：为每个$q_i$计算并构建$\text{PQ}$距离表
> >    - 搜索起点：对每个$q_i$从图质心开始搜索，即$\small\text{CPU}\xleftarrow{传输给}\text{WorkList}\xleftarrow{放入}\textbf{u}_i\textbf{(当前点)}\xleftarrow{初始化}\text{Centroid}$ 
> >
> > 2. 主循环：当收敛条件$\text{Conv}$为假时循环执行以下操作
> >
> >    - 前$\text{CPU}$阶段：$\text{CPU}$从内存中获取当前在处理节点$u_i^*$的邻居集$N_i$
> >
> >    - <span style="color:orange;">数据传输：$\text{CPU}\xrightarrow{邻居集N_i}\text{GPU}$</span>  
> >
> >    - 中$\text{GPU}$阶段：接收$u_i^*$的邻居集$N_i$按顺序执行以下操作
> >
> >      |   步骤   | 操作                                                         |
> >      | :------: | ------------------------------------------------------------ |
> >      | 过滤邻居 | 用$\small\text{Bloom}$==并行检查==$\small\forall{}n\text{∈}N_i$被访问与否$\small\to$未访问点放入$\small{}N_i'$(未访问集)$\text{+}$更新$\small\text{Bloom}$ |
> >      | 距离计算 | 用$\small\text{PQ}$距离表==并行计算==所有未处理邻居$\small{}n_k\text{∈}N_i'$与查询点$q_i$距离，并存储在$\small\mathcal{D}_i[k]$ |
> >      | 邻居排序 | 将$\small{}N_i'$和$\small\mathcal{D}_i[k]$按与$q_i$的距离执行归并排序，得到排序后的距离$\small\mathcal{D}_i'$和节点$\small\mathcal{N}_i'$ |
> >      | 合并列表 | 合并当前$\small\text{WorkLisk}(\mathcal{L}_i)$与新排序的节点列表$\small\mathcal{N}_i'$形成新的$\small\mathcal{L}_i$ |
> >      | 更新节点 | 又将$\small\mathcal{L}_i$排序后选取最近的未访问点$\small{}u_i^*$作为下一个当前节点 |
> >
> >    - <span style="color:orange;">数据传输：$\text{CPU}\xleftarrow{当前节点u_i^*}\text{GPU}$</span>   
> >
> >    - 后$\text{CPU}$阶段：更新收敛标志，若$\mathcal{L}_i$中所有点都被访问过且$|\mathcal{L}_i|\text{=}t$，则设$\text{Conv}$为真结束循环
> >
> > 3. 返回结果：选取$\mathcal{L}_i$中离$q_i$最近的$k$个节点，作为$k\text{-}$最邻近返回 
> >
> > :two:微内核
> >
> > 1. 微内核设计：为$\text{PQ}$表构建$+\text{GPU}$阶段每步操作设立独立内核
> > 2. 动态线程块：
> >    - 每个查询分配到一线程块执行，查询过程会依次执行多个内核
> >    - 执行不同内核时按经验调整线程块大小(如计算密集型内核的块更大)，以保证$\text{GPU}$的高占有
>
> ## 5.2. $\textbf{PQ}$距离表的并行构建
>
> > :one:构建时序
> >
> > |       时期        | 操作                                   |
> > | :---------------: | -------------------------------------- |
> > | 查询前/初始化阶段 | 在$\text{GPU}$上计算/构建/存储距离表   |
> > |   查询执行阶段    | 保留距离表在$\text{GPU}$上直到查询结束 |
> >
> > :two:构建操作
> >
> > 1. 并行化：
> >
> >    - 操作：对查询集$Q_\rho$中$\rho$个查询，并行构建$\rho$个距离子表$\to$合并为最终距离表
> >
> >    - 映射：为每个查询分配独立的线程块
> >
> > 2. $\text{PQ}$压缩：将原有$D$维向量分为$M$个子空间，每个子空间$k\text{-Means}$聚类出$k$个簇/质心
> >
> >    $\begin{cases}\small数据点\text{: }\begin{cases}
> >    \textbf{v}_1=[x_{11},x_{12},x_{13},x_{14},...,x_{1D}]\\\\
> >    \textbf{v}_2=[x_{21},x_{22},x_{23},x_{24},...,x_{2D}]\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \textbf{v}_α=[x_{α1},x_{α2},x_{α3},x_{α4},...,x_{αD}]\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \textbf{v}_k=[x_{k1},x_{k2},x_{k3},x_{k4},...,x_{kD}]
> >    \end{cases}
> >    \xrightarrow[分割]{\text{PQ}}\small\begin{cases}
> >    \widetilde{\textbf{v}_{1}}=\{\textbf{Centriod}_{11},\textbf{Centriod}_{12},...,\textbf{Centriod}_{1M}\}\\\\
> >    \widetilde{\textbf{v}_{2}}=\{\textbf{Centriod}_{21},\textbf{Centriod}_{22},...,\textbf{Centriod}_{2M}\}\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \widetilde{\textbf{v}_{α}}=\{\textbf{Centriod}_{α1},\textbf{Centriod}_{α2},...,\textbf{Centriod}_{αM}\}\\\\
> >    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
> >    \widetilde{\textbf{v}_{k}}=\{\textbf{Centriod}_{k1},\textbf{Centriod}_{k2},...,\textbf{Centriod}_{kM}\}
> >    \end{cases}\\\\
> >    查询点\text{: }\textbf{q}=[q_{1},q_{2},q_{3},q_{4},...,q_{D}]\xrightarrow[分割]{与\text{PQ}子空间的维度划分对齐}\textbf{q}=\{\textbf{q}_1,\textbf{q}_2,...,\textbf{q}_M\}
> >    \end{cases}$  
> >
> > 3. 构建子表：$M\text{×}k$维
> >
> >    | $\small\text{Dist. Table}$ | 子空间/线程$1$ |子空间/线程$2$                |                             ....                             | 子空间/线程$M$ |
> >    | :------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------: | :----------------------------------------------------------: |
> >    |       $\textbf{v}_1$       | <span style="color:red;">$\small\text{dist}^2(\textbf{q}_1,\textbf{Centriod}_{11})$</span> | <span style="color:red;">$\small\text{dist}^2(\textbf{q}_2,\textbf{Centriod}_{12})$</span> |            **....**             | <span style="color:red;">$\small\text{dist}^2(\textbf{q}_M,\textbf{Centriod}_{1M})$</span> |
> >    |       $\textbf{v}_2$       | <span style="color:red;">$\small\text{dist}^2(\textbf{q}_1,\textbf{Centriod}_{21})$</span> | <span style="color:red;">$\small\text{dist}^2(\textbf{q}_2,\textbf{Centriod}_{22})$</span> |            **....**             | <span style="color:red;">$\small\text{dist}^2(\textbf{q}_M,\textbf{Centriod}_{2M})$</span> |
> >    |       **.........**        |                        **.........**                         |                        **.........**                         |            **....**             |                        **.........**                         |
> >    |       $\textbf{v}_k$       | <span style="color:red;">$\small\text{dist}^2(\textbf{q}_1,\textbf{Centriod}_{k1})$</span> | <span style="color:red;">$\small\text{dist}^2(\textbf{q}_2,\textbf{Centriod}_{k2})$</span> |            **....**             | <span style="color:red;">$\small\text{dist}^2(\textbf{q}_M,\textbf{Centriod}_{kM})$</span> |
> >
> >    - 距离计算：在每个子空间中，计算查询点$q_s \text{∈} \mathbb{R}^{^{\frac{D}{M}}}$$\xleftrightarrow{\text{Euclidean距离平方}}$所有簇的质心
> >    - 线程映射：每个子空间对应一个独立==线程==，依次计算$q_s \text{∈} \mathbb{R}^{^{\frac{D}{M}}}$与每个$k$个质心的距离
> >
> > 4. 合并子表：将$\rho$个$M\text{×}k$维的子表合并为最终$\rho\text{×}M\text{×}k$维表，并存储在$\text{GPU}$内存上
> >
> >
> > :three:一些说明$\&$算法分析
> >
> > 1. 距离计算原理：$\text{dist}(q,\textbf{v}_α)=\displaystyle{}\sum\limits_{i=1}^{M}\text{dist}^2(q_i,\textbf{Centriod}_{αi})$，就是表中$\textbf{v}_α$行所有内容相加
> >
> > 2. 参数设定：更具经验设定$k\text{=}256$，由消融实验确定$m\text{=}74$最优
> >
> > 3. 算法分析：
> >
> >    | $\textbf{Item}$ | 含义                                         |                          复杂度                           |
> >    | :-------------: | -------------------------------------------- | :-------------------------------------------------------: |
> >    |  $\text{Work}$  | 算法串行执行总耗时                           | $O((m \cdot \text{subspace\_size}) \cdot 256 \cdot \rho)$ |
> >    |  $\text{Span}$  | 算法并行执行耗时，即**最耗时串行步骤**的耗时 |         $O(m \cdot \text{subspace\_size}) = O(d)$         |
>
> ## 5.3. 面向减少$\textbf{PCIe}$传输的优化
>
> > :one:优化$1$：只传输最低限度所需信息
> >
> > 1. 每次迭代中传输的内容：$\text{CPU}\xrightarrow[\text{PCIe}]{邻居集N_i}\text{GPU}$与$\text{CPU}\xleftarrow[\text{PCIe}]{当前节点u_i^*}\text{GPU}$ 
> > 2. 结束迭代后传输的内容：最终的近似最邻近列表
> >
> > :two:优化$2$：结合高级的$\text{CUDA}$功能

 
