##### Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations  

# 0. 写在前面

> ## 0.1. 预备知识
>
> > :one:最邻近搜索：给定数据库$U$中的$n$个对象(子集)，以及输入的查询点$q$
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240731222415006.png" alt="image-20240731222415006" style="zoom: 20%;" /> 
> >
> > 1. 精确定义($\text{NN}$)：返回$e^*\in{S}$满足$\operatorname{dist}(q, e^*) = \min\limits_{e \in S} \operatorname{dist}(q, e)$    
> > 2. 近似定义($\text{c-ANN}$)：返回$e\in{S}$满足$\operatorname{dist}(q, e)\leq{}c*\operatorname{dist}(q, e^*) $ 
> >
> > :two:倍增维度
> >
> > 1. 直径：点集中距离最远的两点的距离，即$\text{diam}(X)=\sup_\limits{e_1, e_2 \in X} \operatorname{dist}\left(e_1, e_2\right)$ 
> > 2. $2^{\lambda}\text{-}$分割：$X$可被分为$m$个子集$X_1X_2...X_m$，且满足$\begin{cases}m\leq{}2^{\lambda}\\\\\text{diam}(X_i)\leq{}\cfrac{1}{2}\text{diam}(X)\end{cases}$
> > 3. 倍增维度：就是$\lambda_{min}$，即用最少$2^{\lambda_{min}}$个半径不超过$\cfrac{1}{2}r_{_X}$的球体填满$X$ 
> >
> > :three:球体的一个性质：任何球体 $B(e, r)$ 都可被至多 $O\left(k^d\right)$ 个半径为 $\cfrac{r}{c}$ 的球体覆盖
>
> ## 0.2. 一些记号
>
> > |   符号    | 含义                                                 |
> > | :-------: | :--------------------------------------------------- |
> > | $(X, D)$  | 基础度量空间，$X$为点集$D$为距离类型                 |
> > | $D(u, v)$ | $P = \{x_1, \ldots, x_n\}$中的俩点$x_u$和$x_v$的距离 |
> > | $B(p, r)$ | 以$p \in X$为球心$r$为半径的球                       |
> > | $\Delta$  | 点集中最远俩点 / 最近俩点的**比值**                  |
> > |    $d$    | 倍增维度                                             |
>
> ## 0.3. 本文的研究概述
>
> > :one:背景：很多$\text{c-ANN}$算法比如$\text{HNSW/NSG/DiskANN}$在基准数据集上性能不错，但其下界不可知
> >
> > :two:研究成果
> >
> > 1. 最坏情况上界：对于$\text{DiskANN}$慢处理版，可在$O\left(\log _\alpha \cfrac{\Delta}{(\alpha-1) \epsilon}\right)$步返回$\left(\cfrac{\alpha+1}{\alpha-1}+\epsilon\right)\text{-ANN}$ 
> >    - 但值得注意，$\text{DiskANN}$慢处理的图构建复杂度高达$O(n^3)$ 
> > 2. 最坏情况下界：对于$\text{DiskANN}$块预处理版/$\text{HNSW}$/$\text{NSG}$，算法找到$5$个最邻近前至少要走$0.1n$步
> >
> > :three:未来的研究方向：有没有一种算法，预处理和查询都低于线性复杂度$?$ 

# 1. $\text{DiskANN}$回顾

> ## 1.0. Intro
>
> > 🤔是个啥：基于邻近图的贪婪搜索的，一种解决$\text{c-ANN}$的算法
> >
> > 🙉基本思路
> >
> > 1. 在数据库中的点集$P$上创建有向图$G=(V, E)$，其中$V$与$P$关联
> >
> > 2. 给定带查询点$q$，从起始点$s\in{}V$开始对图$G$执行搜索
> >
> > 3. 预期搜索返回$q$的最邻近点
>
> ## 1.1. $\text{DiskANN}$的基本操作
>
> > :one:连接操作$\text{RobustPruning}(v, U, \alpha, R)$，用于图的构建过程
> >
> > 1. 参数含义
> >
> >    |   符号   | 含义                                 |
> >    | :------: | :----------------------------------- |
> >    |   $v$    | 当前处理的图中的顶点                 |
> >    |   $U$    | $v$的备选邻居(预计最终与$v$连接的点) |
> >    | $\alpha$ | 修剪参数，一定大于$1$                |
> >    |   $R$    | 结点出度的限制，即$v$至多有几个邻居  |
> >
> > 2. 连接过程
> >
> >    - 排序：将$U$所有元素按离$v$的距离从近到远排序
> >    - 遍历：开始遍历并处理每个$u\in{}U$
> >      - 修剪：对$u\in{}U$，删除满足$D(u, u^{\prime}) \cdot \alpha < D(v, u^{\prime})$的点$u^{\prime}$，余下的点总体离$v$更近
> >    - 连接：让$v$与所有$U$中所有剩下点连接
> >
> > :two:搜索操作$\text{GreedySearch}(s, q, L)$ 
> >
> > 1. 参数含义
> >
> >    |   符号   | 含义                  |
> >    | :------: | :-------------------- |
> >    |   $s$    | 图$G$中，搜索的起始点 |
> >    |   $q$    |给定带查询点$q$|
> >    | $L$ | 维护队列的最大长度，==这也是搜索算法扫描结点数量的下界== |
> >
> > 2. 辅助数据结构：当前队列$A$，已访问点集$U$ 
> >
> > 3. 搜索过程
> >
> >    - 初始化：$A=\{s\}$，$U=\varnothing$ 
> >    - 扫描与剪裁：循环执行访问$+$剪裁，直到$A$中所有点都被访问
> >      - 访问：选取$A$中距离$q$最近的未访问点$v\to\begin{cases}A=A\cup{}N_{out}(v)\,,将邻居全部加入A\\\\U=U\cup{}v\,,表示v已经被访问\end{cases}$ 
> >      - 裁剪：当$A$队列长度超出$L$后，保留其中$L$个与$q$最近的点
> >    - 排序输出：输出$A$排序后的前$k$个点 
>
> ## 1.2. $\text{DiskANN}$的构建操作
>
> > :one:慢预处理：对所有$v\in{}V$执行$\text{RobustPruning}(v, V, \alpha, |V|)$ 
> >
> > :two:快预处理操作
> >
> > 1. 初始化：对所有结点$V$执行$R$重构建，即每个点随意连接$R$个顶点
> > 2. 第一遍遍历
> >    - 起始：任选一$s$开始，随机访问其后继$v$
> >    - 搜索：对$v$执行$U=\text{GreedySearch}(s, v, L)$得到$U$(可能与$v$最邻近的点集)
> >    - 修剪：对得到的$U$执行$\text{RobustPruning}(v, U, \alpha, n)$修剪其中离$v$较远的点
> >    - 连接：将$v$与修剪后$U$的所有结点相连
> >      - 再修剪：如果$u\in{}U$度数超过$R$，则执行$\text{RobustPruning}(u, N_{\text{out}}(u), \alpha, R)$修剪$u$邻居
> > 3. 第二遍遍历：对第一次遍历所得结果，同样的操作在遍历一次

# 2. 对于慢处理$\text{DiskANN}$的理论分析

> ## 2.1. 预处理分析
>
> > :one:$\alpha\text{-}$捷径可达性
> >
> > 1. 顶点$\alpha\text{-}$捷径可达$\xLeftrightarrow{}$满足二者之一$\forall{}q\in{}V\to{}\begin{cases}直连\text{: }(p, q) \in E\\\\捷径连接\text{: }\exist{}p^{\prime}满足\begin{cases}(p, p') \in E\\\\D(p', q) \leq \cfrac{D(p, q)}{\alpha}\end{cases}\end{cases}$ 
> > 2. $\forall{}p\in{}G$都具有$\alpha\text{-}$捷径可达性，则称$G$具有$\alpha\text{-}$捷径可达性
> >
> > :two:预处理分析
> >
> > 1. 时间复杂度：$O\left(n^3\right)$
> >
> > 2. 捷径可达性：慢处理构建的图具有捷径可达性
> >
> > 3. 稀疏性：
> >
> >    - 记$p$执行$\text{RobustPruning}(p, V, \alpha, n)$后$p$所连接点数量为$|U(p)|$则$|U(p)|\leqslant O\left((4 \alpha)^d \log \Delta\right)$ 
> >
> >      -  证明思路大致为：引入环形区域$\to$覆盖$\to$剩余点范围受限于倍增维度
> >
> >    - 在后续实验中，可以看出慢预处理的$\text{DiskANN}$相当稀疏
> >
> >      <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240811112053924.png" alt="image-20240811112053924" style="zoom: 38%;" /> 
>
> ## 2.2. 查询分析
>
> > :one:结论：从$G$中任一点$s$开始执行$\text{GreedySearch}(s, q, 1)$ 
> >
> > 1. 能在$O\left(\log _\alpha \cfrac{\Delta}{(\alpha-1) \epsilon}\right)$步内返回$\left(\cfrac{\alpha+1}{\alpha-1}+\epsilon\right)\text{-}$近似邻居
> > 2. $\text{P.s. }$每步最多检查$|U|\leqslant O\left((4 \alpha)^d \log \Delta\right)$ 个邻居，即每步最多$O\left((4 \alpha)^d \log \Delta\right)$时间
> >
> > :two:证明思路大致为
> >
> > 1. 通过三角不等式和 $\alpha\text{-}$捷径可达性，得出每一步的距离 $d_i$ 的上界
> >
> > 2. 分析三种情况的查询步数
> >
> >    | $D(s, q)$ | $D(a, q)$ | 分析思路                                                     |
> >    | :-------: | :-------: | ------------------------------------------------------------ |
> >    |    远     |  远$+$近  | 通过初步不等式得出$d_i$与 $D(a, q)$ 关系$\to$算法在$\log _\alpha \cfrac{2}{\epsilon}$步内结束 |
> >    |    近     |    远     | 通过上下界$\to{}$算法在$O\left(\log _\alpha \frac{\Delta}{(\alpha-1) \epsilon}\right)$步内结束 |
> >    |    近     |    近     | 通过不等式结合$D_{\min}$和$D_{\max}$$\to{}$算法在$O\left(\log _\alpha \Delta\right)$步内结束 |
> >
> >    - 注意$D(a, q)$中$a$表示$q$的最邻近点
> >
> > :three:后续实验中，慢处理的$\text{DiskANN}$在难例上，==甚至只需要两步就找到最邻近== 
>
> ## 2.3. 对复杂度的紧致性分析
>
> > :one:收敛率的严格下界：在$O\left(\log _\alpha \cfrac{\Delta}{(\alpha-1) \epsilon}\right)$步中不可将$\log \Delta$换为$\log n$，证明思路如下
> >
> > 1. 构造一个一维度量空间满足$|P|=n=2k-1$且$\Delta=O(\alpha^n)$
> > 2. 证明：给定查询点$q$并执行$\text{GreedySearch}$，找到$q$的$O(1)\text{-ANN}$至少要扫描$\Omega(\log \Delta)$或$O(n)$个点
> >
> > :two:紧密的近似下限：$\cfrac{\alpha+1}{\alpha-1}\text{-ANN}$的比例具有紧，证明思路如下
> >
> > 1. 构建一个简单实例
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240811004613238.png" alt="image-20240811004613238" style="zoom:33%;" /> 
> >
> > 2. 证明在上述实例中，执行$\text{DiskANN}$慢预处理版至少要扫描$n$个点，才能找到一个$\cfrac{\alpha+1}{\alpha-1}\text{-ANN}$ 
> >
> >    - 思路：从$s\in{}P$开始扫描$\to{}n$步贪婪搜索后出不了$P$$\to$无法接近最邻近$a\to$至少扫描$n$点

# 3. 对于快处理$\text{DiskANN}$(及其他算法)的实验

> ## 3.1. 快处理$\text{DiskANN}$的实验
>
> > :one:构建的难实例
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240811095551367.png" alt="image-20240811095551367" style="zoom:45%;" /> 
> >
> > :two:实验结果：执行查询试图输出$5$个最邻近
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240811110820889.png" alt="image-20240811110820889" style="zoom:40%;" /> 
> >
> > 1. 召回率(结果中实际最邻近点的比率)在$L \approx 10 \%$处发生剧变
> > 2. 至少需要扫描$10\%$的点才能使召回率非$0$，即查询的时间复杂度为$O(0.1n)$ 
>
> ## 3.2. $\text{NSG/HNSW}$算法的实验
>
> > :one:构建的难例：
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240811134727506.png" alt="image-20240811134727506" style="zoom: 45%;" /> 
> >
> > :two:同样也是，至少要扫描$10\%$的点才能使召回率达标
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240811141718920.png" alt="image-20240811141718920" style="zoom: 31%;" /> 
>
> ## 3.3. 交叉对比 
>
> > :one:为三个算法构建同样的难例
> >
> > :two:结果：依然是需要扫描至少$0.1n$个点
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240811142950183.png" alt="image-20240811142950183" style="zoom: 32%;" />  
>
> ## 3.4. 其它算法上的实验
>
> > :one:构建难例，给定$L=0.1n$，运行结果(召回率)如下表(截取)
> >
> > | DiskANN | NSG  | HNSW | NGT  | SSG  | KGraph |
> > | :-----: | :--: | :--: | :--: | :--: | :----: |
> > |   0.0   | 0.27 | 0.1  | 0.05 | 0.16 |  0.42  |
> >
> > :two:分析：说明$0.1n$很可能就是这些算法的下界
