# $\textbf{1. }$导论

:one:$\text{E2LSH}$：让对象$\vec{o}$通过$h_{\vec{a},b}(\vec{o})\text{=}\left\lfloor\cfrac{\vec{a·}\vec{o}\text{+}b}{w}\right\rfloor$

1. 随机投影：用随机向量$\vec{a}$将对象$\vec{o}$变成标量，再将投影结果加上一个随机偏移量$b$
2. 桶划分：将投影值除以桶宽$w$，向下取整就是桶编号了
3. 特点：查询无关的桶划分，即预处理阶段分桶就已经完成并且固定

:two:查询无关分桶的弊端：

1. 例如对象$o_1$离查询$q$更近，但仍然被划分到不同的桶中

   <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250208163552978.png" alt="image-20250208163552978" width=350 /> 

   |    桶     | 落入桶中的点 |
   | :-------: | :----------: |
   | $[0, w)$  |    $o_1$     |
   | $[w, 2w)$ |   $q,o_2$    |

2. 导致查询时无法快速通过桶找到离$q$最近的元素

:three:本文提出的**查询感知分桶**

1. 预处理阶段：将桶宽度设为一个固定值$w$，然后仍然用投影计算哈希函数$h_{\vec{a}}(\vec{o})\text{=}\vec{a·}\vec{o}$ (无需随机偏移)

2. 查询阶段：当$q$到来后将其投影$h_{\vec{a}}(\vec{q})\text{=}\vec{a·}\vec{q}$作为分桶的锚点(中心点)

3. 动态分桶：划定范围$\left[h_{\vec{a}}(q)–\cfrac{w}{2}, h_{\vec{a}}(q)\text{+}\cfrac{w}{2}\right]$为中心位置的桶锚桶

   <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250208173136957.png" alt="image-20250208173136957" width=350 /> 

:four:开销的优化：

1. 预处理阶段：将所有的哈希值$h_{\vec{a}}(\vec{o_i})$索引到$\text{B}^{+}$树(只会存储对象的索引不会存储具体的对象)
2. 查询阶段：计算$h_{\vec{a}}(\vec{q})$，利用$\text{B}^{+}$树快速定位到位于$\left[h_{\vec{a}}(q)–\cfrac{w}{2}, h_{\vec{a}}(q)\text{+}\cfrac{w}{2}\right]$内的$h_{\vec{a}}(\vec{o_i})$对象

 :five:$\text{QALSH}$的优势：再查询时动态分桶，支持任意近似比$c$的$c\text{-ANN}$，可以自动设置桶宽

# $\textbf{2. }$预备知识

:one:问题设定：假设查询$q$的真实最邻近是$o^{*}$

1. $c\text{-ANN}$：返回单个对象$o$，满足$\text{dist}(o,q)\text{≤}c\text{×}\text{dist}(o,q^{*})$ 
2. $c\text{-}k\text{-}\text{ANN}$：返回$k$个对象$o$，每个对象都满足$\text{dist}(o,q)\text{≤}c\text{×}\text{dist}(o,q^{*})$ 

:two:$\text{LSH}$族：对于$\text{LSH}$函数簇$h\text{∈}H$，如果满足以下条件则称$H$为$(r, cr, p_{1}, p_{2})\text{-}$敏感的

1. 如果$\text{dist}(o_1,o_2)\text{≤}r$，则$\text{Pr}[h(o_1)\text{=}h(o_2)]\text{≥}p_1$(正碰撞概率)
2. 如果$\text{dist}(o_1,o_2)\text{＞}cr$，则$\text{Pr}[h(o_1)\text{=}h(o_2)]\text{≤}p_2$(负碰撞概率)

:three:一种查询无关$\text{LSH}$族：$h_{\vec{a},b}(\vec{o})\text{=}\left\lfloor\cfrac{\vec{a·}\vec{o}\text{+}b}{w}\right\rfloor$

1. 其中$\vec{a}$向量的每个元素符合$\mathcal{N}(0,1)$正态分布，标量$b$从区间$[0, w)$中均匀抽取

2. 两对象$o_1,o_2$碰撞的概率：

   - 令$s\text{=}\text{dist}(o_1,o_2)$以及$f_2(x)\text{=}\cfrac{2}{\sqrt{2\pi}} e^{–\frac{x^{2}}{2}}$，碰撞概率为$\displaystyle{}\xi(s)\text{=}P_{\vec{a},b}\left[h_{\vec{a},b}\left(o_{1}\right)\text{=}h_{\vec{a},b}\left(o_{2}\right)\right]\text{=}\int_{0}^{w}\frac{1}{s}f_{2}\left(\frac{t}{s}\right)\left(1–\frac{t}{w}\right)dt\tag{2}$
   - 哈希函数族$h_{\vec{a},b}$是$\left(r, c r, \xi(r), \xi(cr)\right)\text{-}$敏感的
   - 令$r\text{=1}$则希函数族$h_{\vec{a},b}$是$\left(1, c, \xi(1), \xi(c)\right)\text{-}$敏感的

# $\textbf{3. }$查询感知$\textbf{LSH}$族

:one:$\left(1, c, p_{1}, p_{2}\right)\text{-}$敏感$\text{LSH}$族

1. 以查询感知方式构建$\text{LSH}$函数：
   - 预处理：先将所有对象$o$以$h_{\vec{a}}(\vec{o})\text{=}\vec{a·}\vec{o}$投影到$\vec{a}$方向上，其中$\vec{a}$的每个元素取自标准正态分布
   - 查询：使用$h_{\vec{a}}(q)$作为锚点来定义宽度为$w$的锚桶，即$\left[h_{\vec{a}}(q)–\cfrac{w}{2}, h_{\vec{a}}(q)\text{+}\cfrac{w}{2}\right]$
   - 其中$w$的大小由$h_{\vec{a}}(\cdot)$定义
2. 碰撞：
   - 定义：当一个对象$\vec{o}$落在宽度为$w$的桶内即$\left|h_{\vec{a}}(o)–h_{\vec{a}}(q)\right|\text{≤}\cfrac{w}{2}$时，才称$o,q$之间发生碰撞
   - 概率：令$\varphi(x)\text{=}\cfrac{1}{\sqrt{2 \pi}} e^{–\frac{x^{2}}{2}}$以及$s\text{=}\text{dist}(o,q)$，则$o,q$之间发生碰撞的概率为$\displaystyle{}p(s)\text{=}\int_{-\frac{w}{2 s}}^{\frac{w}{2 s}} \varphi(x)dx$
3. 局部敏感性：$h_{\vec{a}}(o)$是$\left(1, c, p(1), p(c)\right)\text{-}$敏感性的

:two:碰撞概率的比较：令$\text{Norm}(x)\text{=}\displaystyle{}\int_{–\infty}^{x}\cfrac{1}{\sqrt{2\pi}} e^{–\frac{x^{2}}{2}}$

1. 一些符号及其表示：

   - 查询感知$\text{LSH}$：$p_1\text{=}p(1)\text{=}1–2 \operatorname{Norm}\left(–\cfrac{w}{2}\right)$以及$p_2\text{=}p(c)\text{=}1–2 \operatorname{Norm}\left(–\cfrac{w}{2c}\right)$
   - 查询无关$\text{LSH}$：$\xi_1\text{=}\xi(r)\text{=}1–2 \operatorname{norm}(–w)–\cfrac{2}{\sqrt{2 \pi} w}\left(1–e^{–\frac{w^{2}}{2}}\right)$以及$\xi_2\text{=}\xi(cr)\text{=}1–2 \operatorname{norm}\left(–\cfrac{w}{2}\right)–\cfrac{2c}{\sqrt{2 \pi} w}\left(1–e^{–\frac{w^{2}}{2c^2}}\right)$ 

2. 实验结果：

   - 所有的概率值都随桶宽$w$增大而增大，直至$w\text{=}10$时概率接近$1$

     <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250208214033118.png" alt=" " width=500 />  

   - 各自概率的差值都会有一个峰值，理论上概率的差值越大越好，所以可以根据峰值的位置自动设置桶宽度

     <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250208214329344.png" alt="image-20250208214329344" width=500 />  

:three:虚拟重哈希

1. $(R,c)\text{-NN}$问题：当存在$o_2$满足$\text{dist}(q,o_2)\text{≤}R$时，需要找到一个$o_1$满足$\text{dist}(q,o_1)\text{≤}cR$
2. 与$c\text{-ANN}$：可将$c\text{-ANN}$问题转化为一系列$R\text{∈}\{1, c, c^{2}, c^{3}, \ldots\}$的$(R,c)\text{-NN}$问题
   - 当$R\text{=}1$时，需要$(1, c, p_{1}, p_{2})\text{-}$敏感哈希族来解决$(1,c)\text{-NN}$问题
   - 当$R\text{∈}\{c, c^{2}, c^{3}, \ldots\}$时，需要从$(1, c, p_{1}, p_{2})\text{-}$敏感哈希族派生出$(R, Rc, p_{1}, p_{2})\text{-}$敏感哈希族来解决$(R,c)\text{-NN}$问题
3. 何谓虚拟：就是指数据点只需要哈希一次并存储在$R\text{=}1$的物理表中，后续只需通过缩放生成虚拟的哈希表即可

:four:$\text{QALSH}$的虚拟重哈希

1. 命题$1$：查询感知哈希族$H_{\vec{a}}^{R}(o)\text{=}\cfrac{h_{\vec{a}}(o)}{R}$是$\left(R, cR, p(1), p(c)\right)\text{-}$敏感性的
   - $h_{\vec{a}}(o)$是$\left(1, c, p(1), p(c)\right)\text{-}$敏感性的
2. 定义第$R$轮的锚(中心)桶为$B^{R}\text{=}\left[H_{\vec{a}}^{R}(q)–\cfrac{w}{2}, H_{\vec{a}}^{R}(q)\text{+}\cfrac{w}{2}\right]$，其中$B^{1}\text{=}\left[h_{\vec{a}}(q)–\cfrac{w}{2}, h_{\vec{a}}(q)\text{+}\cfrac{w}{2}\right]$ 
   - $B^{R}\text{=}\left[\cfrac{h_{\vec{a}}(q)}{R}–\cfrac{w}{2}, \cfrac{h_{\vec{a}}(q)}{R}\text{+}\cfrac{w}{2}\right]$
3. 找到$q$的$(R,c)\text{-NN}$：需要检查特定$R$轮次的中心桶$B^{R}$
4. 找到$q$的$c\text{-ANN}$：需要逐轮检查每一轮次的中心桶$B^{R}$，其中$R\text{∈}\{1, c, c^{2}, c^{3}, \ldots\}$

:five:叠加性质：给定$q$以及$w$

1. $B^R$的范围包含了$B^1$，并且$B^R$的宽度是$B^1$的$R$倍即$wR$，严格来说$B^R$的范围是
   - $B^{R}\text{=}\left[h_{\vec{a}}(q)–\cfrac{wR}{2}, h_{\vec{a}}(q)\text{+}\cfrac{wR}{2}\right]$ 
2. $B^{cR}$的范围包含了$B^R$，并且$B^{cR}$的宽度是$B^R$的$c$倍

:six:划分桶的准备

1. 预处理
   - 哈希表：数据库中的每个对象$o_i$都有一个$\text{ID}_{o_i}$，$h_{\vec{a}}$的哈希表$T\text{=}\left(h_{\vec{a}}(o_i),\text{ID}_{o_i}\right)$
   - 索引：对表$T$中所有条目按照$h_{\vec{a}}(o_i)$升序排序，然后通过$\text{B}^+$索引，每个哈希表对应一个$\text{B}^+$索引
2. 桶划分的动态定位
   - 设定一个桶宽$w$
   - 当查询$q$到来时，执行$(R,c)\text{-NN}$搜索时，需要利用$\text{B}^+$树快速找到$B^{R}\text{=}\left[h_{\vec{a}}(q)–\cfrac{wR}{2}, h_{\vec{a}}(q)\text{+}\cfrac{wR}{2}\right]$ 范围内的对象
   - 执行$c\text{-ANN}$搜索时，先执行$B^{1}\text{=}\left[h_{\vec{a}}(q)–\cfrac{w}{2}, h_{\vec{a}}(q)\text{+}\cfrac{w}{2}\right]$范围内的查询，然后逐步扩展搜索半径，即虚拟重哈希

# $\textbf{4. }$查询感知的$\textbf{LSH}$方案 

## $\textbf{4.1. QALSH}$用于$\boldsymbol{(R,c)}\textbf{-NN}$搜索

:one:$\text{QALSH}$的哈希函数与基

1. 基：$\mathcal{B}\text{=}\left\{H_{a_1}^R(\cdot), H_{a_2}^R(\cdot), \ldots, H_{a_m}^R(\cdot)\right\}$，其中每个$H_{a_i}^R(\cdot)$是查询感知的$\text{LSH}$函数，并且独立均匀地取自一个$(R,cR,p_1,p_2)\text{-}$敏感的$\text{LSH}$族
2. 哈希表的构建：对于每个$H_{a_i}^R(\cdot)$都构建一个自己的哈希表$T_i\text{=}\left(H_{a_i}^R(o),\text{ID}_{o}\right)$，并用$\text{B}^+$树索引

:two:搜索过程

1. 计算哈希值：给定查询对象$q$，计算每个基函数的哈希值$H_{a_i}^R(q)\text{=}\cfrac{h_{\vec{a_i}}(q)}{R}$
2. 定位中心桶：划定每个中心桶的范围为$B^{R}\text{=}\left[H_{\vec{a_i}}^{R}(q)–\cfrac{w}{2}, H_{\vec{a_i}}^{R}(q)\text{+}\cfrac{w}{2}\right]$，用$\text{B}^+$树查询各自哈希表，得到落入各自中心桶的所有元素
3. 收集碰撞数：对于每个对象$o$，统计其出现在了多少个中心桶中，记为$\text{\#Col}(o)$ 
4. 识别频繁对象：设定阈值$l$，如果$\text{\#Col}(o)\text{>}l$则认为对象$o$是频繁的
5. 返回结果：计算所有频繁对象到$q$的距离，若存在频繁对象与$q$距离不超过$cR$，则返回该对象

:three:基数$m$的确定

1. 正确性保障：确保以下两个属性能同时以常数概率成立
   - $\mathcal{P}_{1}$：如果存在一个对象$o$到$q$的距离在$R$之内，则$o$是一个频繁对象
   - $\mathcal{P}_{2}$：假阳性的总数少于$\beta n$(其中$n$为数据库规模)，其中每个假阳性都是到$q$的距离大于$cR$的频繁对象
2. 引理$3$：给定$p_{1}\text{=}p(1)$和$p_{2}\text{=}p(c)$，定义$p_{2}\text{<}\alpha\text{<}p_{1}, 0\text{<}\beta\text{<}1$且$0\text{<}\delta\text{<}\cfrac{1}{2}$
   - 前提：令$m=\left\lceil\max \left(\cfrac{1}{2\left(p_{1}–\alpha\right)^{2}} \ln \cfrac{1}{\delta}, \cfrac{1}{2\left(\alpha–p_{2}\right)^{2}} \ln \cfrac{2}{\beta}\right)\right\rceil$
   - 结论：$\mathcal{P}_{1}$和$\mathcal{P}_{2}$能同时以至少$\cfrac{1}{2}–\delta$的概率成立

## $\textbf{4.1. QALSH}$用于$\boldsymbol{c}\textbf{-ANN}$搜索

:one:算法$1$的流程：从$R\text{=}1$开始在

1. 初始化：设置初始的搜索半径为$R\text{=}1$，以及候选集$C$为空
2. 循环访问：当候选集的大小小于$\beta{n}$时，执行以下步骤
   - 检查每个对象$o$，计算每个$H_{a_i}^R(o)$，并判断$q$是否在$B^{R}\text{=}\left[H_{\vec{a_i}}^{R}(q)–\cfrac{w}{2}, H_{\vec{a_i}}^{R}(q)\text{+}\cfrac{w}{2}\right]$中
   - 最终得到每个对象$o$的碰撞次数$\text{\#Col}(o)$，如果其超过阈值$l$，则将其添加到候选集$c$中
3. 循环更新：
   - 结束：如果候选集$C$中至少有一对象$o$满足$\text{dist}(q,o)\text{≤}cR$，则跳出本次循环
   - 更新：否则更新半径$R$，然后重复下一次循环
4. 输出：候选集$C$中与$q$最近的对象
