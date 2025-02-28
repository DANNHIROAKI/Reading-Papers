# $\textbf{1. QALSH}$原理

> ## $\textbf{1.1. }$查询敏感哈希族
>
> > :one:哈希函数：$h_{\vec{a}}(\vec{o})\text{=}\vec{a·}\vec{o}$，其中$\vec{a}$的每个元素取自标准正态分布
> >
> > 1. 中心桶：使用$h_{\vec{a}}(\vec{q})$作为中心点来定义宽度为$w$的中心桶即$\left[h_{\vec{a}}(\vec{q})–\cfrac{w}{2}, h_{\vec{a}}(\vec{q})\text{+}\cfrac{w}{2}\right]$，其中$w$的大小由$h_{\vec{a}}(\cdot)$定义
> > 2. 碰撞：对象$\vec{o},\vec{q}$之间发生碰撞$\text{⇔}\vec{o}$使得$\left|h_{\vec{a}}(\vec{o})–h_{\vec{a}}(\vec{q})\right|\text{≤}\cfrac{w}{2}$，二者碰撞概率为$\displaystyle{}p(s)\text{=}\int_{-\frac{w}{2 s}}^{\frac{w}{2 s}} \varphi(x)dx$其中$s\text{=}\text{dist}(\vec{o},\vec{q})$
> > 3. 敏感性：$h_{\vec{a}}(\vec{o})$是$\left(1, c, p(1), p(c)\right)\text{-}$敏感的，即相距小于$r$的两点碰撞概率大于$p(1)$，相距大于$c$的两点碰撞概率小于$p(c)$ 
> >
> > :two:查询敏感哈希族：$H_{\vec{a}}^{R}(\vec{o})\text{=}\cfrac{h_{\vec{a}}(\vec{o})}{R}\text{=}\cfrac{\vec{a·}\vec{o}}{R}$
> >
> > 1. 中心桶：使用$H_{\vec{a}}^{R}(\vec{q})$作为中心点来定义宽度为$w$的中心桶即$B^{R}\text{=}\left[H_{\vec{a}}^{R}(\vec{q})–\cfrac{w}{2}, H_{\vec{a}}^{R}(\vec{q})\text{+}\cfrac{w}{2}\right]$
> > 2. 碰撞：对象$\vec{o},\vec{q}$之间发生碰撞$\text{⇔}H_{\vec{a}}^{R}(\vec{o})$落入$\left[H_{\vec{a}}^{R}(\vec{q})–\cfrac{w}{2}, H_{\vec{a}}^{R}(\vec{q})\text{+}\cfrac{w}{2}\right]\text{⇔}h_{\vec{a}}(\vec{o})$落入$\left[h_{\vec{a}}(\vec{q})–\cfrac{wR}{2}, h_{\vec{a}}(\vec{q})\text{+}\cfrac{wR}{2}\right]$
> > 3. 敏感性：$H_{\vec{a}}^{R}(\vec{o})\text{=}\cfrac{h_{\vec{a}}(\vec{o})}{R}$是$\left(R, cR, p(1), p(c)\right)\text{-}$敏感性的
>
> ## $\textbf{1.2.}$ 虚拟重哈希
>
> > :one:虚拟重哈希的操作
> >
> > 1. 当$R\text{=}1$时：所有数据点进行一次真实的哈希，生成以下的物理哈希表，并在物理哈希表上构建$\text{B}^{+}$树
> >
> >    $\begin{array}{|c|c|c|}\hline对象&对象\text{ID}&本轮哈希值\\\hline\vec{o}_1&\text{ID}_{\vec{o}_1}&H_{\vec{a}}^{1}(\vec{o}_1)\text{=}\vec{a·}\vec{o}_1\\\hline\vec{o}_2&\text{ID}_{\vec{o}_2}&H_{\vec{a}}^{1}(\vec{o}_2)\text{=}\vec{a·}\vec{o}_2\\\hline\cdots&\cdots&\cdots\\\hline\vec{o}_n&\text{ID}_{\vec{o}_n}&H_{\vec{a}}^{1}(\vec{o}_n)\text{=}\vec{a·}\vec{o}_n\\\hline\end{array}$ 
> >
> > 2. 当$R\text{=}R$时：基于$R\text{=}1$时的真实哈希缩放生成虚拟哈希，生成以下的逻辑哈希表，并在逻辑哈希表上构建$\text{B}^{+}$树
> >
> >    $\begin{array}{|c|c|c|c|}\hline\text{对象}&\text{对象}\text{ID}&\text{物理哈希值}&\text{本轮哈希值(逻辑)}\\\hline\vec{o}_1&\text{ID}_{\vec{o}_1}&H_{\vec{a}}^{1}(\vec{o}_1)\text{=}\vec{a·}\vec{o}_1&H_{\vec{a}}^{1}(\vec{o}_1)\text{=}\vec{a·}\vec{o}_1\xrightarrow{\text{比例缩放}}H_{\vec{a}}^{R}(\vec{o}_1)\text{=}\cfrac{\vec{a·}\vec{o}_1}{R}\\\hline\vec{o}_2&\text{ID}_{\vec{o}_2}&H_{\vec{a}}^{1}(\vec{o}_2)\text{=}\vec{a·}\vec{o}_2&H_{\vec{a}}^{1}(\vec{o}_2)\text{=}\vec{a·}\vec{o}_2\xrightarrow{\text{比例缩放}}H_{\vec{a}}^{R}(\vec{o}_2)\text{=}\cfrac{\vec{a·}\vec{o}_2}{R}\\\hline\cdots&\cdots&\cdots&\cdots\\\hline\vec{o}_n&\text{ID}_{\vec{o}_n}&H_{\vec{a}}^{1}(\vec{o}_n)\text{=}\vec{a·}\vec{o}_n&H_{\vec{a}}^{1}(\vec{o}_n)\text{=}\vec{a·}\vec{o}_n\xrightarrow{\text{比例缩放}}H_{\vec{a}}^{R}(\vec{o}_n)\text{=}\cfrac{\vec{a·}\vec{o}_n}{R}\\\hline\end{array}$ 
> >
> > :two:虚拟重哈希的应用
> >
> > 1. $(R,c)\text{-NN}$问题：当存在$\vec{o}_2$满足$\text{dist}(\vec{q},\vec{o}_2)\text{≤}R$时，需要找到一个$\vec{o}_1$满足$\text{dist}(\vec{q},\vec{o}_1)\text{≤}cR$
> > 2. $c\text{-ANN}$问题：可将$c\text{-ANN}$问题转化为一系列$R\text{∈}\{1, c, c^{2}, c^{3}, \ldots\}$的$(R,c)\text{-NN}$问题
> >    - 当$R\text{=}1$时，需要$(1, c, p(1), p(c))\text{-}$敏感哈希族来解决$(1,c)\text{-NN}$问题
> >    - 当$R\text{∈}\{c, c^{2}, c^{3}, \ldots\}$时，需从$(1, c, p(1), p(c))\text{-}$敏感哈希重哈希出$(R, Rc, p_{1}, p_{2})\text{-}$敏感哈希族来解决$(R,c)\text{-NN}$问题
>
> ## $\textbf{1.3. }$查询感知的$\textbf{LSH}$方案 
>
> > :one:预处理过程
> >
> > 1. 哈希基：$\mathcal{B}\text{=}\left\{H_{a_1}^R(\cdot), H_{a_2}^R(\cdot), \ldots, H_{a_m}^R(\cdot)\right\}$，其中每个$H_{a_i}^R(\cdot)$独立均匀地取自一个$(R,cR,p_1,p_2)\text{-}$敏感的$\text{LSH}$族
> > 2. 哈希表的构建：对于每个$H_{\vec{a}_i}^R(\cdot)$都构建一个自己的哈希表$T_i\text{=}\left(H_{\vec{a}_i}^R(\vec{o}),\text{ID}_{\vec{o}}\right)$，每个哈希表都用$\text{B}^+$树索引
> >
> > :two:$(R,c)\text{-NN}$过程
> >
> > 1. 中心桶：对所有$H_{a_i}^R(\vec{q})\text{=}\cfrac{h_{\vec{a_i}}(\vec{q})}{R}$计算其中心桶$B^{R}\text{=}\left[H_{\vec{a_i}}^{R}(\vec{q})–\cfrac{w}{2}, H_{\vec{a_i}}^{R}(\vec{q})\text{+}\cfrac{w}{2}\right]$
> > 2. 碰撞点：用$\text{B}^+$树查询各自哈希表，查看哪些$H_{a_i}^R(\vec{q})$落入了各自的中心桶$\left[H_{\vec{a_i}}^{R}(\vec{q})–\cfrac{w}{2}, H_{\vec{a_i}}^{R}(\vec{q})\text{+}\cfrac{w}{2}\right]$
> > 3. 碰撞数：对于每个对象$\vec{o}$，统计其出现在了多少个中心桶中，记为$\text{\#Col}(\vec{o})$ 
> > 4. 频繁对象：设定阈值$l$，如果$\text{\#Col}(\vec{o})$则认为对象$\vec{o}$是频繁的
> > 5. 返回值：计算所有频繁对象到$\vec{q}$的距离，若存在频繁对象与$\vec{q}$距离不超过$cR$，则返回该对象
> >
> > :three:${c}\text{-ANN}$过程
> >
> > 1. 第一种方案：机械地遍历每个搜索半径
> >    - 初始化：从$R\text{=}1$开始执行$(R,c)\text{-NN}$，收集使$h_{\vec{a}_i}(\vec{o})$落入$\left[h_{\vec{a}_i}(\vec{q})–\cfrac{w}{2}, h_{\vec{a}_i}(\vec{q})\text{+}\cfrac{w}{2}\right]$的$\vec{o}$，是为收集频繁对象
> >    - 更新：让$R\text{→}cR$重新执行$(cR,c)\text{-NN}$，收集使$h_{\vec{a}_i}(\vec{o})$落入$\left[h_{\vec{a}_i}(\vec{q})–\cfrac{wR}{2}, h_{\vec{a}_i}(\vec{q})\text{+}\cfrac{wR}{2}\right]$的$\vec{o}$，是为继续收集频繁对象
> >    - 终止：当频繁对象数量达到$\beta{n}$时终止，输出候选集中离$q$最近的点
> > 2. 第二种方案：有策略地跳过一些搜索半径
> >    - 初始化：从$R\text{=}1$开始执行$(R,c)\text{-NN}$过程，收集使得$h_{\vec{a}_i}(\vec{o})$落入$\left[h_{\vec{a}_i}(\vec{q})–\cfrac{w}{2}, h_{\vec{a}_i}(\vec{q})\text{+}\cfrac{w}{2}\right]$的$\vec{o}$，是为收集频繁对象
> >    - 准备更新：对每个哈希表，找到**中心桶外**但**投影离$\boldsymbol{h_{\vec{a}_i}(\vec{q})}$最近**的对象，得到$d_1,...,d_m$共$m$个最近距离取中位数为$d_\text{med}$ 
> >    - 更新：选取使得$\cfrac{wc^k}{2}\text{≥}d_\text{med}$的$R\text{=}c^k$由此跳过一些无用的搜索半径，重新执行$(R,c)\text{-NN}$过程收集频繁对象，并同理更新
> >    - 终止：当频繁对象数量达到$\beta{n}$时终止，输出候选集中离$q$最近的点

# $\textbf{2. }$理论分析

> ## $\textbf{2.1. }$参数设定
>
> > :one:基数$m$的分析：为了保证质量，$m$的设置在尽可能小的情况下必须满足引理$3$的要求
> >
> > 1. 正确性保障：确保以下两个属性能同时以常数概率成立
> >    - $\mathcal{P}_{1}$：如果存在一个对象$o$到$q$的距离在$R$之内，则$o$是一个频繁对象
> >    - $\mathcal{P}_{2}$：假阳性的总数少于$\beta n$(其中$n$为数据库规模)，其中每个假阳性都是到$q$的距离大于$cR$的频繁对象
> > 2. 引理$3$：给定$p_{1}\text{=}p(1)$和$p_{2}\text{=}p(c)$，定义$p_{2}\text{<}\alpha\text{<}p_{1}, 0\text{<}\beta\text{<}1$且$0\text{<}\delta\text{<}\cfrac{1}{2}$
> >    - 前提：令$m=\left\lceil\max\left\{\cfrac{1}{2\left(p_{1}–\alpha\right)^{2}} \ln \cfrac{1}{\delta}, \cfrac{1}{2\left(\alpha–p_{2}\right)^{2}} \ln \cfrac{2}{\beta}\right\}\right\rceil$
> >    - 结论：$\mathcal{P}_{1}$和$\mathcal{P}_{2}$能同时以至少$\cfrac{1}{2}–\delta$的概率成立
> >
> > :two:基数$m$及其它参数的确定
> >
> > 1. 碰撞阈值百分比$\alpha$：令$\eta\text{=}\sqrt{\cfrac{\ln\beta–\ln2}{\ln\delta}}$，于是$\alpha=\cfrac{\eta{}p_{1}\text{+}p_{2}}{\eta\text{+}1}$
> > 2. 基数$m$：当$\max$括号中二者相等时$m$最小，任取其中之一代入$\alpha$则有$m\text{=}\left\lceil\cfrac{\left(\sqrt{\ln \cfrac{2}{\beta}}\text{+}\sqrt{\ln \cfrac{1}{\delta}}\right)^{2}}{2\left(p_{1}–p_{2}\right)^{2}}\right\rceil$
> > 3. 碰撞阈值$l$：$l=\lceil\alpha m\rceil$
> >
> > :three:桶宽$w$的自动设置：令$w\text{=}\sqrt{\cfrac{8 c^{2} \ln c}{c^{2}–1}}$时则可以使得$m$最小化
>
> ## $\textbf{2.2. }$其它
>
> > :one:基于$\text{QALSH}$的$c\text{-ANN}$过程的复杂度
> >
> > 1. 空间复杂度：原始数据储存$O(nd)\text{+}$索引储存$O(n \log n)$
> > 2. 时间复杂度：计算投影$O(d\log n)\text{+}$定位中心桶$O((\log n)^2)\text{+}$碰撞计数$O(n \log n)\text{+}$计算欧氏距离$O(d)$
> >
> > :two:基于$\text{QALSH}$的$c\text{-ANN}$过程的理论保证
> >
> > 1. 可与任意近似比$c$(即使是小数)一起工作
> > 2. 以至少$\cfrac{1}{2}–\delta$的概率返回$c^2\text{-ANN}$