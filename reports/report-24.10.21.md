:one:$\text{ANN on }\text{GPU}$的一些关键点

1. 内存：如何高效利用$\text{GPU}$内存
   - 不能将整块图结构$\text{+}$数据点置于$\text{GPU}$内存中
   - 利用$\text{PQ}$技术优化距离计算/比较过程的$\text{GPU}$内存占用
2. $\text{CPU-GPU}$通信($\text{PCIe}$)
   - 将数据轮流分片载入$\text{GPU}$处理需要巨量的$\text{PCIe}$通信(轮换处理/未处理的片)，比如$\text{GGNN}$
   - 如何通过重叠通信/计算，减少减少$\text{CPU-GPU}$的通信
   - 只将$\text{PQ}$压缩后的数据传给$\text{GPU}$ 
3. 如何充分调度$\text{GPU}$使之与$\text{CPU}$协作，最大程度的并行运算
   - 将$\text{GPU}$内核，进一步精细化原子化
   - 高效的任务分配策略
4. $\text{GPU}$数量的问题
   - 多$\text{GPU}$可使数据有效分片在所有$\text{GPU}$上，比如$\text{GGNN}$
   - 针对单$\text{GPU}$的优化有利于控制硬件成本





:two:$\text{BANG}$的总体架构：基于单个$\text{CPU}$

1. 在$\text{GPU}$上使用$\text{PQ}$压缩数据进行距离计算
2. 将$\text{Vamana}$图结构保存在$\text{CPU}$(内存/主机?)上
3. 使$\text{CPU-GPU}$更能充分的调度
4. 通过重叠通信与计算，减少$\text{CPU-GPU}$的通信
   - 与$\text{DiskANN}$类似，每次搜索迭代中$\text{CPU}$向$\text{GPU}$传递邻居信息时，都会隐式传递全精度坐标
5. $\text{Bloom}$过滤器(快速检索某元素是否在集合中)

:bulb:$\text{GPU}-\text{CPU/RAM}-\text{SSD Architecture?}$ 

🤔多$\text{GPU}$ 

😕关于$\text{PQ}$：

- 变种诸如优化$\text{PQ}$([$\small\text{CVPR'14}$](https://doi.org/10.1109/CVPR.2014.298))/加性量化([$\small\text{CVPR'14}$](https://doi.org/10.1109/CVPR.2014.124))/复合量化([$\small\text{ICML'14}$](https://dl.acm.org/doi/abs/10.5555/3044805.3044986)) 

- 关于变种$\text{PQ}$不适合在$\text{GPU}$上实现的讨论 ($\text{FAISS}$) 

  ```txt
  在原始的PQ技术上提出了一些改进，但大多数在GPU上难以高效实现。倒排多索引对于高速度/低质量的操作点非常有用，但依赖复杂的“多序列”算法。优化的乘积量化(OPQ)是一种对输入向量进行线性变换的方法，可以提高乘积量化的准确性，并可以作为预处理应用。André等人的SIMD优化实现仅在次优参数(较少的粗量化中心)下运行。许多其他方法，如LOPQ和多义编码，过于复杂，无法在GPU上高效实现。
  ```



:three:其它$\text{ANN on GPU}$

1. $\text{FAISS}$：当前最为主流的基于$\text{GPU}$的相似性检索模型
   - 基于$\text{KNN-Graph}$的图结构
   - 采用$\text{PQ/OPQ}$技术压缩以节省空间
   - 可暴力搜索/近似搜索
2. $\text{GGNN}$：
   - 面向批处理以及多$\text{GPU}$
   - 基于$\text{KNN-Graph}$的图结构
   - 分簇并生成多个子图，将每簇轮换入$\text{CPU}$进行处理
3. $\text{SONG}$：
   - 基于$\text{NSW}$的图索引结构(无分层结构)
   - 将搜索算法解耦为三部分以实现并行
   - 消除动态$\text{GPU}$内存分配





:four:多样化方向图

1. 为何传统$\textbf{KNN}$图连通性较差

   - 最近邻聚集在同一方向，如下**$\textbf{2-NN}$图**中，搜索路径只能$p\text{→}\{a_3,a_4\}$而不能$p\text{→}b$即使$p\xleftrightarrow{}b$很近

     <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240928155307724.png" alt="image-20240928155307724" style="zoom: 15%;" />  

   - 中心性问题，即很多点没有入度，不作为任何点的最邻近

2. 策略

   - 选取邻居(衡量相似度)时将角度纳入考虑
   - 将单向边变成双向边

3. 算法流程

   - 对于$p$，先找出其$K$个最邻近点(组成$\mathcal{L}$列表)

   - 从$\mathcal{L}$列表中选择子集$\mathcal{S}$($\text{|}\mathcal{S}\text{|=}\kappa$)使$\mathcal{S}$中两点**平均角度**最大；选择方法遵循以下贪婪启发式算法

     <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241012235347641.png" alt="image-20241012235132614" style="zoom: 40%;" /> 

   - 将所有边双向化，即$\forall{}u\in\mathcal{S}$将$(p, u)$与$(u, p)$都包括在邻近图中

     