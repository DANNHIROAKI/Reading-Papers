# $\textbf{Approximate Nearest Neighbor Search}$

# $\textbf{1. Doubling Dimension}$

> ## $\textbf{1.0. Intro}$ 
>
> > :one:度量空间$\text{(metric space)}$：可看作距离二元组即$(U, \operatorname{dist})$  
> >
> > 1. 点集$U$：
> >    - 非空且可能无限，其中的元素称之为对象$\text{(object)}$​ 
> >    - 对象间的距离：$\operatorname{dist}(e_1, e_2)$，其中$e_1, e_2 \in U$ 
> > 2. 距离函数$\operatorname{dist}$：是一个$U\text{×}U\to{}\mathbb{R}_{\geq 0}$(非负实数)的映射，满足以下条件
> >    - 自己到自己距离为$\text{0}$，$e \in U\text{→}\operatorname{dist}(e, e)=0$​ 
> >    - 任意两点距离大于$\text{1}$，$e_1, e_2 \in U\land{}e_1 \neq e_2\text{→}\operatorname{dist}(e_1, e_2) \geq 1$ 
> >    - 两点间互相距离不变，$e_1, e_2 \in U\text{→}\operatorname{dist}(e_1, e_2) = $$\operatorname{dist}(e_2, e_1)$ 
> >    - 满足三角不等式，$e_1, e_2, e_3 \in U\text{→}\operatorname{dist}(e_1, e_2) \leq \operatorname{dist}(e_1, e_3) + \operatorname{dist}(e_3, e_2)$ 
> >
> > :two:$\text{Nearest Neighbor Search}$是个啥​
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240731222415006.png" alt="image-20240731222415006" style="zoom: 20%;" /> 
> >
> > 1. 算法输入：$U$中的$n$个对象(即子集$S$)，以及$S$外的一点$q$
> > 2. 算法输出：$n$个对象中使得$\operatorname{dist}(q, e)$最小的对象$e^*$​
> >    - 即$\operatorname{dist}(q, e^*) = \min\limits_{e \in S} \operatorname{dist}(q, e)$​​  
> >    - 称这个对象 $e^*$ 是 $q$ 的最近邻$\text{(nearest neighbor)}$，$e^*$不一定唯一
> >
> > :three:$\text{Nearest Neighbor Search}$的解决
> >
> > 1. 理想情况：将$S$预处理为一种数据结构，不论度量空间如何，都可高效得到$\min\limits_{e \in S} \operatorname{dist}(q, e)$ 
> > 2. 最坏情况：(朴素方法)计算单个$q\xleftrightarrow{一共n组距离}n$个$e$​之间的距离，选出最大距离
> > 3. 近似情况：$\text{c-}$最近似邻$\text{(c-ANN)}$，即$\begin{cases}\text{NN: }\operatorname{dist}(q, e)=\operatorname{dist}(q, e^*)\\\\\text{c-ANN: }\operatorname{dist}(q, e)\leq{}c*\operatorname{dist}(q, e^*)\end{cases}$
> >    - $q$可能有多个$\text{(c-ANN)}$​点，算法一般返回其中任意一个
> >    - 找到$\text{(c-ANN)}$任然极其困难了，近似条件下的最坏情况仍需计算$n$​次距离
> > 4. $\text{c-ANN}$能够高效解决：
> >    - 必要条件：$(U, \operatorname{dist})$要满足$U=\mathbb{N}^d$ 且维度$d$为常数 ($d$维空间)且$\operatorname{dist}$是欧几里得距离
> >    - 倍增维度：即使$(U, \text{dist})$很难，但特定$S$的$(S, \text{dist})$倍增维度小，其$\text{c-ANN}$也能高效解决
> >
> > :four:$\text{Measure the space and query time of a structure}$​ 
> >
> > 0. 将对象和距离函数视为黑箱
> > 1. 结构的空间复杂度：结构占用的内存单元数$+$存储的对象数
> > 2. 查询的时间复杂度：$\text{RAM}$原子操作的数量$\text{+}$距离函数$\text{dist}$​被调用的次数
> >
> > :five:纵横比$\text{(aspect ratio): }$ $S$ 中最大和最小成对距离之间的比率
> >
> > - 即$\Delta(S)=\left(\sup_\limits{e_1, e_2 \in S} \operatorname{dist}\left(e_1, e_2\right)\right) /\left(\inf_\limits{\text {distinct } e_1, e_2 \in S} \operatorname{dist}\left(e_1, e_2\right)\right)$ 
>
> ## $\textbf{1.1. Doubling Dimension}$ 
>
> > :one:直径：$X$是$U$非空子集，$X$直径是$X$中==距离最远的两点的距离==，即$\text{diam}(X)=\sup_\limits{e_1, e_2 \in X} \operatorname{dist}\left(e_1, e_2\right)$ 
> >
> > :two:$2^{\lambda}\text{-}$分割：$X$可被分为$m$个子集$X_1X_2...X_m$，且满足$\begin{cases}m\leq{}2^{\lambda}\\\\\text{diam}(X_i)\leq{}\cfrac{1}{2}\text{diam}(X)\end{cases}$ 
> >
> > :three:倍增维度
> >
> > 1. 含义：使得每个非空有限子集$X$都可被$2^\lambda$-划分的$\lambda{}$​​ 
> >
> >    - 即倍增维度就是最小的那个$\lambda$，使得数量
> >
> > 2. 实例：当$X$含有八个点，满足$\begin{cases}U\text{=}\mathbb{N}^2\\\\\text{dist}为欧氏距离\end{cases}\to{}$ 该度量空间的倍增维度为 $\log_2 7 < 3$ 
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/3b5c50b6f08ecffabab6639541817158.png" alt="img" style="zoom: 70%;" />   
> >
> >    - $D$为覆盖$X$所有点的最小圆(图中实线)，$\operatorname{diam}(D)=\operatorname{diam}(X)$ 
> >    - 七个直径为$\cfrac{1}{2}\text{diam}(X)$的小圈$D_1D_2...D_7$(图中虚线)，就可实现对$D$​的全覆盖
> >    - 将每个$e$分配到(唯一的)小圆盘$D_i$中，由此将$X$分为$X_1X_2...X_7$
> >    - $\text{max}\{\text{diam}(X_i)\}\leq{}\text{diam}(D_i)=\cfrac{1}{2}\text{diam}(X)$，==所以$X$ 可以被 $2^{\log_2 7}$ 分割==
> >
> > :four:在$X\subseteq{}U$情况下$(X, \operatorname{dist})$的倍增维度$\leq(U, \operatorname{dist})$​的倍增维度 
>
> ## $\textbf{1.2. Two Properties in Metric Space}$ 
>
> > ### $\textbf{1.2.1. Balls}$ 
> >
> > > :one:球体：
> > >
> > > 1. 传统意义：在$\mathbb{R}^d$中位于 $d$​ 维球体内的点集，比如二维圆/三维球
> > > 2. 推广到度量空间：
> > >    - 对于$\forall{}e\in{}U$以及半径$r\geq{}0$，所有满足$\operatorname{dist}\left(e, e^{\prime}\right) \leq r$的$e^{\prime}$集合就是球
> > >    - 记作$B(e, r)=\{e^{\prime}_1e^{\prime}_2...e^{\prime}_k\}$
> > >
> > > :two:球体性质
> > >
> > > 1. 对于传统球体： 半径为 $r$ 的 $d$ 维球体可以被 $2^{O(d)}$ 个半径为 $\Omega(r)$​​ 的球体覆盖
> > > 2. 对度量空间的球体
> > >    - 条件：度量空间$(U, \operatorname{dist})$的倍增维度为$\lambda$，$c$为常数
> > >    - 表述$\text{1: }$任何球体 $B(e, r)$ 都可被至多 $2^{O(\lambda)}$ 个半径为 $\cfrac{r}{c}$​ 的球体覆盖
> > >    - 表述$\text{2: }\exist{}\text{ }e_1, \ldots, e_m \in U$使得$\begin{cases}m \leq 2^{O(\lambda)} \\\\B(e, r) \subseteq \bigcup\limits_{i=1}^m B\left(e_i, \cfrac{r}{c}\right)\end{cases}$ 
> >
> > ### $\text{1.2.2. Constant Aspect-Ratio Object Sets}$​ 
> >
> > > :one:常数纵横比对象集：$r=1$的球体中可放入最多 $2^{O(d)}$ 个点，并确保任意两点间距离$\geq{}\cfrac{1}{2}$ 
> > >
> > > :two:引理：度量空间$(X, \operatorname{dist})$的倍增维度为$\lambda$，$X$的纵横比为常数，则$X$最多只有$2^{O(\lambda)}$​个对象
>
> ## $\textbf{1.3. A 3-ANN Structure}$ 
>
> > ### $\textbf{1.3.0. Inro}$ 
> >
> > > :one:一些要用到的符号
> > >
> > > 1. $(U, \operatorname{dist})$为基础度量空间，$S \subseteq U$为包含$n \geq 2$个对象的$\text{Input}$​ 
> > > 2. $h=\left\lceil\log _2 \operatorname{diam}(S)\right\rceil$​ 
> > > 3. $\lambda$​为$(S, \operatorname{dist})$​的倍增维数
> > >
> > > :two:本节讨论的定理：存在结构可在$\begin{cases}空间复杂度\text{: }2^{O(\lambda)}h\\\\时间复杂度\text{: }2^{O(\lambda)}hn\end{cases}$内回答$\text{3-ANN}$问题
> >
> > ### $\textbf{1.3.1. Sample Nets}$ 
> >
> > > :one:样本网定义：对于$X \subseteq S$，要求$Y$是$X$的$r$-样本网需要满足
> > >
> > > 1. $Y \subseteq X$
> > > 2. $\forall{}y_1, y_2 \in Y$有 $\operatorname{dist}\left(y_1, y_2\right) > r$​ 
> > > 3. $X \subseteq \bigcup\limits_{y \in Y} B(y, r)$即$\forall{}x\in{}X,\,\exists{}y\in{}Y$使得$\operatorname{dist}(x, y) \leq r$ 
> > >
> > > :two:样本网络实例：度量空间为$\left(\mathbb{N}^2,\text{dist=Euclidean})\right.$  
> > >
> > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240802224755982.png" alt="image-20240802224755982" style="zoom: 43%;" /> 
> > >
> > > 1. $Y \subseteq X\Rightarrow\begin{cases}Y=\{黑点\}\\\\X=\{黑点+白点\}\end{cases}$​ 
> > > 2. $\operatorname{dist}\left(y_1, y_2\right) > r\Rightarrow$​单个黑点不能出现在两个圆中
> > > 3. $X \subseteq \bigcup\limits_{y \in Y} B(y, r)\Rightarrow$所有点只出现在圆的重叠平面内
> >
> > ### $\textbf{1.3.2. Structure G}$ 
> >
> > > :one:结构的定义
> > >
> > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240803124118507.png" alt="image-20240803124118507" style="zoom:45%;" />  
> > >
> > > 1. 定义每层结点：$Y_i$层的点即为$S$的$2^i$-样本网$(i=0,1,...,h)$​ 
> > >    - $Y_h$只有一个对象，并且$2^h\geq{}\text{diam}(S)$ 
> > >    - 框定$|Y_i|\leq{}n$后，使得$G$的空间复杂度变为了$O(hn)$​ 
> > > 2. 定义结点的连接
> > >    - 对于$y\in{}Y_{i}$与$z\in{}Y_{i-1}$如果满足$\operatorname{dist}(y, z) \leq 7 \cdot 2^i$则建立有向连接$y\longrightarrow{}z$ ​
> > >    - 用$N_i^{+}(y)$表示 $y$ 的出度$\text{(out-neighbors)}$ 
> > >
> > > :two:结构的性质：$\left|N_i^{+}(y)\right|=2^{O(\lambda)}$即$\left|N_i^{+}(y)\right|$随着$\lambda$​的增加指数级增加
> >
> > ### $\textbf{1.3.3. Query}$  
> >
> > > :one:查询过程
> > >
> > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240731222415006.png" alt="image-20240731222415006" style="zoom: 20%;" />  
> > >
> > > 1. 先将$S$转化为图$G$​的结构
> > > 2. 对于查询对象$q \in U \backslash S$我们在图$G$中沿某条路径下降(称之为$\pi$) 
> > >    - 起始：访问$G$根节点$Y_h$​
> > >    - 下降：$y\in{}Y_{i}$与$z\in{}Y_{i-1}$ 且$i\geq1$时，按照$\operatorname{dist}(q, z)$​ 最小化原则下降，平局时任选
> > > 3. 查询结果即返回$\pi$路径中离$q$最近的一点，即$e^{*}$​ 
> > >
> > > :two:查询性质
> > >
> > > 1. 查询的时间复杂度：$\left|N_i^{+}(y)\right|h=2^{O(\lambda)}h$​​ 
> > > 2. 该查询是$q$的$\text{3-ANN}$，即$\exist{}e\in{}\{y_hy_{h-1}\ldots{}y_0\}$满足$\text{dist}(q,e)\leq{}3*\text{dist}(q,e^*)$  
>
> ## $\textbf{1.4. Remarks}$ 
>
> > :one:结构$G$在度量空间$(U, \operatorname{dist})$倍增维度$\lambda{}$较小时高效，比如以下情况
> >
> > 1. $\left(\mathbb{N}^d,\text{dist=Euclidean})\right.$的倍增维度为$\lambda{}=O(d)\xrightarrow{如果d是常数}O(1)$​ 
> >    - 可将$|S|=n$的集合存储在$\mathbb{N}^d$结构中$\to\begin{cases}空间复杂度\text{: }O(n \log \Delta(S))\\\\时间复杂度\text{: }O(\log \Delta(S))\end{cases}$ 
> > 2. $\left(\mathbb{N}^d,\text{dist=}L_t\text{-Norm})\right.$的倍增维度也为$\lambda{}=O(d)\xrightarrow{如果d是常数}O(1)$ 
> >
> > :two:关于$\lambda$的其它注意事项
> >
> > 1. $\lambda$很大会导致度量空间“困难”，不论输入$S$是什么都无法解决$\text{3-ANN}$​问题
> > 2. $\lambda$只和==输入==度量空间$(S, \text{dist})$而非==基础==度量空间$(U, \text{dist})$，所以只需$S\subseteq{U}$有效就行
> >
> > :three:算法下界：$|S|=n$
> >
> > 1. 最精确的邻近查询：没有结构可以避免计算$q\xleftrightarrow{}\{e_1e_2...e_n\}$即$n$​次距离，下界就是$n$ 
> > 2. $\text{c-ANN}$查询：当$(S, \text{dist})$的倍增维度是$\lambda$时，下界为$2^{\Omega(\lambda)} \log |S|$ 

# $\textbf{2. Locality Sensitive Hashing}$ 

> ## $\textbf{2.0. Intro}$ 
>
> > :one:$\text{LSH}$的优势：在$\lambda{}$较大的度量空间，也可以高效回答$\text{c-ANN}$查询问题
> >
> > :two:一些预备知识
> >
> > 1. 多重集并集$\text{(multi-set union): }$和普通并集相比区别在于保留重复项
> >    - 比如$Z_1 = \{a, b\}和Z_2 = \{b, c\}Z_1 \Rightarrow{}Z_1\cup Z_2 = \{a, b, b,c\}$​  
> > 2. $\text{Markov}$不等式：$\operatorname{Pr}[X \geq t \cdot \mathbf{E}[X]] \leq \frac{1}{t}$ 
>
> ## $\textbf{2.1. }(r,c)\textbf{-Near Neighbor Search}$ 
>
> > :one:$(r,c)\text{-NN}$​​​概念 
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240731222415006.png" alt="image-20240731222415006" style="zoom: 20%;" /> 
> >
> > 1. $r \geq 1$且$c > 1$，$S\subseteq{}U$且$|S|=n$ ，$q \in U$​ 
> >
> > 2. $(r,c)\text{-NN}$查询返回：令==$D=\text{dist}(q,e_i)$==  
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20240803185122335.png" alt="image-20240803185122335" style="zoom: 33%;" /> 
> >
> >    | $\textbf{Case}$ |         $\exist{}e_i使D\in[0,r]$         |       $\exist{}e_i使D\in{}[r,cr]$        |     $\exist{}e_i使D\in[cr,\infin{}]$     | 返回对象               |
> >    | :-------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | ---------------------- |
> >    | $\text{Case 1}$ | <span style="color:#00FF00;">一定</span> | <span style="color:#FF9900;">可能</span> | <span style="color:#FF9900;">可能</span> | 满足$D\leq{cr}$的$e_i$ |
> >    | $\text{Case 2}$ |  <span style="color:red;">不可能</span>  |  <span style="color:red;">不可能</span>  |  <span style="color:red;">不可能</span>  | 返回寂寞               |
> >    | $\text{Case 3}$ |  <span style="color:red;">不可能</span>  | <span style="color:#00FF00;">一定</span> | <span style="color:#FF9900;">可能</span> | 满足$D\leq{cr}$的$e_i$ |
> >
> > :two:引理：按以下步骤，可回答$S$上所有$c^{2}\text{-ANN}$查询
> >
> > 1. 条件：对任意$r \geq 1$和$c > 1$，我们已经知道了如何在$S$上构建结构来回答$(r,c)\text{-NN}$查询
> > 2. 步骤：
> >    - 构建$O(\log \operatorname{diam}(S))$个这样的结构
> >    - 发起$O(\log \operatorname{diam}(S))$个$(r,c)\text{-NN}$查询 ($c$相同但$r$​​不同)
>
> ## $\textbf{2.2. Locality Sensitive Hashing}$ 
>
> > :one:局部敏感哈希函数定义：核心思想就是将相似的点映射进同一桶，不相似的点映射到不同桶
> >
> > 1. 前提
> >    - 设$r/c/p_1/p_2$满足$r\geq{}1/c>1/0 < p_2 < p_1 \leq 1$​ 
> >    - $h$是根据某种分布从函数族$H$中抽取的函数
> > 2. 随机函数$h\text{: }U \rightarrow \mathbb{N}$是$\left(r, cr, p_1, p_2\right)\text{-LSH}$函数，需满足
> >    - $\forall{}x,y\in{}U\to{}\begin{cases}\operatorname{dist}(x, y) \leq r\Rightarrow{}\operatorname{Pr}[h(x) = h(y)] \geq p_1\\\\\operatorname{dist}(x, y) > cr\Rightarrow{}\operatorname{Pr}[h(x) = h(y)] \leq p_2\end{cases}$​​​​ 
> >    - 即两个数据靠得近($\leq{}r$)，哈希冲突到一个桶的概率就大；靠的远($>cr$)则概率就小
> > 3. 此外定义$\left(r, cr, p_1, p_2\right)\text{-LSH}$函数的对数比值为$\rho = \cfrac{\ln \left(\cfrac{1}{p_1}\right)}{\ln \left(\cfrac{1}{p_2}\right)}=\cfrac{\ln{}p_1}{\ln{}p_2}<1$  
> >
> > :two:放大引理：若已知如何获得$\left(r, cr, p_1, p_2\right)\text{-LSH}$函数$h$则$\forall{\text{int }}\ell \geq 1$有$\left(r, cr, p_1^{\ell}, p_2^{\ell}\right)\text{-LSH}$函数$g$使
> >
> > 1. $\forall{}x,g(x)$计算复杂度是$h(x)$的$O(\ell)$倍
> > 2. $g(x)$空间复杂度为$O(\ell)$ 
> >
> > :three:$\text{LHS}$实例：$\left(\mathbb{N}^d,\text{dist=Euclidean})\right.$的$\left(r, cr, p_1, p_2\right)\text{-LSH}$函数
> >
> > 1. 构建
> >    - 生成$d$个随机变量$\alpha_1\alpha_2...\alpha_d$且$\alpha_i\textasciitilde{}N(0,1)$​ 
> >    - 令$\beta > 0$依赖于$c$，$\gamma$在$[0, \beta]$​中均匀随机生成
> >    - $\forall{}x\in\mathbb{N}^d$定义$h(x)=\textbf{[}\cfrac{\gamma+\displaystyle\sum\limits_{i=1}^d\left(\cfrac{\alpha_i \cdot x[i]}{r}\right)}{\beta}\textbf{]}$​ 
> > 2. 性质：$p_2$是一个常数，该函数的对数比值$\rho\leq\cfrac{1}{c}$
>
> ## $\textbf{2.3. A Structure for }(r,c)\textbf{-NN Search}$
>
> > ### $\textbf{2.3.0. Inro}$ 
> >
> > > :one:一些前置条件
> > >
> > > 1. $S\subseteq{}U\,(|S|=n)$
> > > 2. 若能够构建$\rho$的$\left(r, cr, p_1, p_2\right)\text{-LSH}$函数，该结构用于在$S$上回答$(r,c)\text{-NN}$查询
> > > 3. 记$t_{lsh}$为评估$\left(r, cr, p_1, p_2\right)\text{-LSH}$​函数值所需时间
> > >
> > > :two:需要证明的定理：存在这样一种结构
> > >
> > > 1. 复杂度：
> > >    - 空间复杂度：使用$O\left(n^{1+\rho} \cdot \log_{\frac{1}{p_2}} n\right)$个内存单元$+$存储$O\left(n^{1+\rho}\right)$个对象
> > >    - 时间复杂度：查询耗时 $O\left(n^\rho \cdot \log_{\frac{1}{p_2}} n \cdot t_{lsh}\right)+$计算距离耗时$O\left(n^\rho\right)$​ 
> > > 2. 效果：能够至少以$\cfrac{1}{10}$的概率，正确回答一次$(r,c)\text{-NN}$查询
> >
> > ### $\textbf{2.3.1. Structure}$​ 
> >
> > > :one:哈希函数$g_1g_2...g_L$：令$\ell \geq 1$和$L \geq 1$为待定的整数，则
> > >
> > > - 由函数$h\text{:}\left(r, cr, p_1, p_2\right)\text{-LSH}$放大到为$L$个独立函数$\to\begin{cases}g_1\text{:}\left(r, cr, p_1, p_2\right)\text{-LSH}\\g_2\text{:}\left(r, cr, p_1^2, p_2^2\right)\text{-LSH}\\\,\,\,\,\,\,\,\,\text{. . . . . . . }\\g_{\ell}\text{:}\left(r, cr, p_1^{\ell}, p_2^{\ell}\right)\text{-LSH}\\\,\,\,\,\,\,\,\,\text{. . . . . . . }\\g_L\text{:}\left(r, cr, p_1^L, p_2^L\right)\text{-LSH}\end{cases}$​ 
> > >
> > > :two:桶定义：让所有$x\in{}S$通过所有哈希函数$g_i$算出哈希值，==所有==哈希值相同的$x$分到一个桶里
> > >
> > > :three:哈希表：$T_i$收集了由$g_i$哈希出来的若干非空桶，一共$L$张哈希表$T_1, \ldots, T_L$ 构成了我们的结构
> > >
> > > - 空间消耗：$\begin{cases}内存单元\text{: }O(n \cdot L \cdot \ell)\\\\对象\text{: }O(n \cdot L)\end{cases}\to{}$令$\begin{cases}\ell{}=\log_{\frac{1}{p_2}}n\\\\L=n^{\rho}\end{cases}\to{}$空间复杂度符合$\text{Intro}$中的定理
> >
> > ### $\textbf{2.3.2. Query  }$​ 
> >
> > > :one:查询信息：对$q\in{U\text{/}S}$执行$(r,c)\text{-NN}$查询
> > >
> > > :two:查询步骤
> > >
> > > 1. 让$q$分别通过$g_1g_2...g_L$哈希函数，分别被分进桶$g_1(q)g_2(q)...g_L(q)$记作$b_1b_2...b_L$
> > > 2. 让$Z=$ 在$b_1b_2...b_L$的多重集并集中任选$2L+1$个
> > >    - 特殊情况：如果$\displaystyle\sum_{i=1}^L |b_i| \leq 4L+1$，则$Z$​会包括所有桶的所有对象
> > > 3. 在$Z$中找到距$q$最近的对象$e$，若$\operatorname{dist}(q, e) \leq cr$则返回$e$​ 
> > >
> > > :three:查询时间：$\begin{cases}原子操作\text{: }O\left(t_{lsh} \cdot \ell \cdot L\right)\\\\计算距离\text{: }O(L)\end{cases}\to{}$令$\begin{cases}\ell{}=\log_{\frac{1}{p_2}}n\\\\L=n^{\rho}\end{cases}\to{}$时间复杂度符合$\text{Intro}$中的定理 
> >
> > ### $\textbf{2.3.3. Analysis  }$​  
> >
> > > :zero:$\text{Good}$的标准：$x\in{S}$是$\text{good}\xLeftrightarrow{}\operatorname{dist}(q, x) \leq c r$ 否则就为$\text{Bad}$，算法至少返回一个$\text{good}$才成功
> > >
> > > :one:引理$1\text{: }$查询能被正确回答，需要满足以下两个条件
> > >
> > > 1. $\mathbf{C 1：}$$e^*$至少出现在$b_1, \ldots, b_L$中的一个
> > > 2. $\mathbf{C 2：}$$b_1b_2...b_L$的多重集并集中，至少含有$2L$个$\text{bad}$对象
> > >
> > > :two:引理$2$：$\mathbf{C 1}$==不成立==的概率小于$\cfrac{1}{e}$，即$\operatorname{Pr}\left[e^* \notin \displaystyle\bigcup\limits_{i=1}^L b_i\right]\leq{}\cfrac{1}{e}$ ，其中这个$e=2.718...$
> > >
> > > :three:引理$3$：$\mathbf{C 2}$==不成立==的概率小于$\cfrac{1}{2}$
> > >
> > > :face_with_head_bandage:所以$\mathbf{C}1$和$\mathbf{C}2$同时成立的概率至少为$1-(\cfrac{1}{e}+\cfrac{1}{2})>0.1$​ 

# $\text{X. Appendix}$​

> ## $\text{X.1. }O/\Omega{}\text{ Notation}$ 
>
> > :one:渐进上界​
> >
> > | 渐进类型  | 含义 |
> > | :-------- | :--- |
> > | $f(n)=O(1)$ |$\exists{}K$使得无论$n$如何变化，都有$f(n)\leq{}K$|
> > | $f(n)=O(n)$ | $\exists{}K$使得无论$n$如何变化，都有$f(n)\leq{}Kn$ |
> > | $f(n)=O(g(n))$ | $\exists{}K$使得无论$n$如何变化，都有$f(n)\leq{}Kg(n)$ |
> >
> > :two:渐进下界
> >
> > | 渐进类型              | 含义                                                   |
> > | :-------------------- | :----------------------------------------------------- |
> > | $f(n)=\Omega{}(1)$    | $\exists{}K$使得无论$n$如何变化，都有$f(n)\geq{}K$     |
> > | $f(n)=\Omega{}(n)$    | $\exists{}K$使得无论$n$如何变化，都有$f(n)\geq{}Kn$    |
> > | $f(n)=\Omega{}(g(n))$ | $\exists{}K$使得无论$n$如何变化，都有$f(n)\geq{}Kg(n)$ |
>
> ## $\text{X.2. }\text{L}-Norm$ 
>
> > :one:$L_t$-范数：$p \in \mathbb{N}^d$且$p[i]$表示$p$在维度$i$的坐标，则$pq$间范数为$\left(\displaystyle\sum\limits_{i=1}^d |p[i] - q[i]|^t\right)^{1/t}$ 
> >
> > :two:$L_2$范数：==就是欧几里得距离==，即$\begin{cases}二维\text{: }\sqrt{(p_x-q_x)^2+(p_y-q_y)^2}\\\\三维\text{: }\sqrt{(p_x-q_x)^2+(p_y-q_y)^2+(p_z-q_z)^2}\end{cases}$ 
>
> ### $\text{X.3. Adversary Argument}$ 
>
> > :one:对抗性论证：算法分析的技巧，用来寻求最坏情况下的复杂度
> >
> > :two:对抗者：算法的执行者，不仅知道算法的策略，并且还尽可能给出使得算法表现最坏的输入







