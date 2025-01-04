# $\textbf{1. }$一些导论

> ## $\textbf{1.1. }$朴素基于图的$\textbf{ANN}$
>
> > :one:建图：对数据库中所有的点，构建$k\text{-NN}$图(下图以$3\text{-NN}$为例)
> >
> > :two:检索：$\text{GreedySearch}$
> >
> > <img src="https://i-blog.csdnimg.cn/direct/a2d080bae7aa46649d6146c643e562b9.png" alt="image-20250103190133220" width=200 />$\large\boldsymbol{\xrightarrow{\boldsymbol{在图上执行\text{GreedySearch}}}}$<img src="https://i-blog.csdnimg.cn/direct/8b5a3424e10848db861bc49b357a82f9.png" alt="image-20250103190040918" width=200 />
> >
> > 1. 起始操作：从任意点(也可是$\text{Memoid}$中心点)开始
> > 2. 跳跃操作：
> >    - 候选：维护长度为$k\text{+1}$的数据结构$L$，包含当前结点$\&$其$k$个最邻近
> >    - 更新：选择$L$中距离最近的一点作为下一步
> > 3. 终止操作：
> >    - 条件：如果当前结点，比其所有$k$个最邻近都更靠近$q$，则终止跳跃
> >    - 输出：终止处的结点，即为$q$的最邻近
> >
> > :three:存在的问题：缺乏长距离连接导致搜索路径可能过长，可能陷入局部最优
>
> ## $\textbf{1.2. NSW}$算法
>
> > :one:高速通道机制：是$\text{NSW}$的核心机制，即在$\text{NSW}$图中，即使很远的点也可进行长距离连接
> >
> > <img src="https://i-blog.csdnimg.cn/direct/a2d080bae7aa46649d6146c643e562b9.png" alt="image-20250103190133220" width=200 />$\large\boldsymbol{\xrightarrow{\boldsymbol{高速通道\text{ExpressWay}机制}}}$<img src="https://i-blog.csdnimg.cn/direct/eb27ebfe69f841ab8e825288e2b21150.png" alt="image-20250103190040918" width=200 /> 
> >
> > :two:$\text{NSW}$的构图
> >
> > 1. 构图的流程：
> >    - 初始构建：将所有数据库点按随机顺序注意插入图$\text{→}$将插入点与其在图中的$k\text{-NN}$连接
> >    - 后续插入：直接将新的点插入图中$\text{→}$将新插入点与其在图中的$k\text{-NN}$连接
> > 2. 构图的特点：
> >    - 早期插入的点：由于点较少，会被强迫**形成快速通道**
> >    - 后期插入的点：此时点较多，倾向于形成稠密的最邻近连接
> >
> > :three:$\text{NSW}$的查找
> >
> > 1. 有关数据结构：
> >    - 变长废弃表$g$：记录搜索过程中已访问点，避免重复访问
> >    - 定长候选表$c$：记录当前节点下一步要走的候选点，同时也记录上步候选表$c'$(二者相等时收敛)
> > 2. 查找过程：
> >    - 起始：选定图中随机点$/$中心点，加入到候选表$c$中
> >    - 搜索：并行地搜索$c$中所有点的最邻近，用$g$筛掉已访问点后，加入到$c$中并计算与查询的距离
> >    - 更新：对$c$进行去重，按与查询的距离升序排序，并按照$c$的定长(最大长度)截断
> >    - 终止：如果在某一步，更新前的$c^{\prime}$和更新后的$c$相同

# $\textbf{2. HNSW}$

> ## $\textbf{2.1. }$跳表$\textbf{(SkipList)}$的结构与思想
>
> > <img src="https://i-blog.csdnimg.cn/direct/eb62543be5494f2bae57293471a0186e.png" alt="img" width=620 />  
> >
> > :one:跳表的结构
> >
> > 1. 多层有序链表：最底层包含所有元素，上层是下一层的快速通道(包含的元素更稀疏)
> > 2. 层间的关系：通过自上而下的指针，连接相同的元素
> >
> > :two:跳表的操作
> >
> > 1. 构建过程：将所有元素塞入底层($\text{100\%}$)，随机选一半元素升到上一层($\text{50\%}$)，不断随机二分到顶层
> > 2. <font color=red>查找过程</font>：从左到右$/$从上到下，平均复杂度降到$O(\log n)$
> >    - 起始：从顶层的最左边开始
> >    - 下降：从左到右扫描链表内容，如果碰到大于目标值的结点，立马下降
> >    - 迭代：下降后以下降的结点为起点，继续向右扫描$\text{+}$下降
> >    - 终止：到达最底层无法下降时停下，当前结点即为输出结果
>
> ## $\textbf{2.2. HNSW}$
>
> > <img src="https://i-blog.csdnimg.cn/direct/65e8130d648842549e672429cf0e101c.png" alt="image-20250103210857429" width=500 />  
> >
> > :one:$\text{HNSW}$的设计思想
> >
> > 1. 分层结构：最底层是包含所有点的$\text{NSW}$图，往上是含部分点的$\text{NSW}$图(且点数逐渐减少)
> > 2. 层级分配：对每个新插入的点，用概率公式$\text{Floor}(-ln\text{(Uniform(0,1))}\text{×ml)}$决定其最高能到几层
> >
> > :two:$\text{HNSW}$的构建与搜索
> >
> > 1. 构建：对于新插入的点$p$，先算出其最高层$L\text{→}$在$\text{0-L}$都各找出$p$的$k\text{-NN}$并连接
> > 2. 查找：在最高层按$\text{NSW}$方式找到最邻近$\text{→}$将该层最邻近作为下层入口重复搜索过程$\text{→}$迭代到底层

