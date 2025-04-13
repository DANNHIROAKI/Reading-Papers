[原论文](https://doi.org/10.48550/arXiv.2210.15748)

@[toc]
# $\textbf{1. }$背景与导论

> ## $\textbf{1.1. }$向量集搜索是什么
>
> > :one:向量集搜索的过程
> >
> > 1. 输入：查询向量集$Q\text{=}\{q_1,...,q_{m_q}\}\text{∈}\mathbb{R}^{m_q\text{×}d}$，向量集的集合$D\text{=}\{S_1,...,S_N\}$，其中$S_i\text{=}\{x_{i1},...,x_{im_i}\}\text{∈}\mathbb{R}^{m_i\text{×}d}$
> > 2. 输出：用$F(Q,S_i)$衡量$Q$与$S_i$相似度，要求以$1–\delta$的概率返回与$Q$相似度最高的$S_i$，即$S^*\text{=}\mathop{\operatorname{argmax}}\limits_{{i\in\{1,\ldots{},N\}}}F\left({Q,{S}_{i}}\right)$ 
> >
> > :two:对相似度$F(Q,S_i)$的定义
> >
> > 1. 子相似度：衡量每个$q_r\text{∈}Q$和$x_{ij}\text{∈}S_i$之间的相似度
> >    ```txt
> >    👉举个例子,对于查询Q={q1,q2,q3}考虑下面向量集
> >      S1 = {x11, x12, x13, x14}
> >      S2 = {x21, x22, x23, x24, x25}
> >      S3 = {x31, x32, x33}
> >    👉让Q和S1进行内部聚合,先要算出全部的相似度(比如内积)
> >      Sim(q1,x11) Sim(q1,x12) Sim(q1,x13) Sim(q1,x14)
> >      Sim(q2,x11) Sim(q2,x12) Sim(q2,x13) Sim(q2,x14)
> >      Sim(q3,x11) Sim(q3,x12) Sim(q3,x13) Sim(q3,x14)
> >    ```
> > 2. 内部聚合$\sigma$：让每个$q_r\text{∈}Q$得到一个聚合后的相似度，类似于$\text{ColBERT}$的$\text{MaxSim}$  
> >    ```txt
> >    👉对于每个qi,应用内部聚合函数σ
> >      s(q1,S) = σ{Sim(q1,x11), Sim(q1,x12), Sim(q1,x13),Sim(q1,x14)}
> >      s(q2,S) = σ{Sim(q2,x11), Sim(q2,x12), Sim(q2,x13),Sim(q2,x14)}
> >      s(q3,S) = σ{Sim(q3,x11), Sim(q3,x12), Sim(q3,x13),Sim(q3,x14)}
> >    👉在ColBERT中这个σ就是所谓的MaxSim
> >      s(q1,S) = Max{Sim(q1,x11), Sim(q1,x12), Sim(q1,x13), Sim(q1,x14)}
> >      s(q2,S) = Max{Sim(q2,x11), Sim(q2,x12), Sim(q2,x13), Sim(q2,x14)}
> >      s(q3,S) = Max{Sim(q3,x11), Sim(q3,x12), Sim(q3,x13), Sim(q3,x14)}
> >    ```
> > 3. 外部聚合$A$：将每个$q_r\text{∈}Q$内部聚合的结果进行处理，得到最后评分$F(Q,S)$
> >    ```txt
> >    👉对每个内部聚合应用内部聚合函数A
> >      F(Q,S1) = A{s(q1,S), s(q2,S), s(q3,S)}
> >    👉在ColBERT中这个A就是逐个相加
> >      F(Q,S1) = s(q1,S) + s(q2,S) + s(q3,S)
> >    ```
>
> ## $\textbf{1.2. }$为什么最邻居搜索不够
>
> > :one:向量集搜索：基于单向量最邻近的方案
> >
> > 1. 输入：给定查询$Q\text{=}\{q_1,q_2,...,q_{m_q}\}$和向量集的集合$D\text{=}\{S_1,S_2,...,S_N\}$，其中$S_i\text{=}\{x_{i1},x_{i2},...,x_{im_i}\}$
> > 2. 检索：让每个$q_r\text{∈}Q$在所有$S_1\text{∪}S_2\text{∪}\ldots\text{∪}S_N$中执行$\text{MIPS}$(最大内积搜索)，得到其$\text{Top-}K$的向量
> > 3. 候选：一共得到了$m_q\text{×}K$个向量，分别确定每个向量所属的$S_i$，合并得到候选集$D^\prime{\text{=}}\left\{S_{(1)},S_{(2)},...\right\}$ 
> > 4. 重排：计算$Q$与候选集中向量集的精确距离$F\left(Q,S_{(i)}\right)$，以对候选向量集进行重排
> >
> > :two:存在的问题
> >
> > 1. 检索效果上：假设某个$q_r\text{∈}Q$的最邻近是$x_{ij}\text{∈}S_i$，二者各自的向量邻近和$F(Q,S_i)$会很大没有半毛钱关系
> > 2. 检索成本上：$S_1\text{∪}S_2\text{∪}\ldots\text{∪}S_N$中但向量规模是巨大的，成本会爆掉
>
> ## $\textbf{1.3. }$关于$\textbf{LSH}$
>
> > :one:几种不同的$\text{LSH}$回顾
> >
> > 1. 基于余弦相似度的$\text{LSH}$：
> >    - 原理：一般用随机超平面对分割空间，一个子空间就是一个桶，两向量**内积**越大越可能落入同一桶
> >    - 例如：$\text{Muvera}$中用到的$\text{SimHash}$
> > 2. 基于欧几里得距离的$\text{LSH}$
> >    - 原理：一般将所有向量投影到随机方向的直线上，直线上每$w$步长就是一个桶，两向量**欧式距离**近者更可能落入同一桶
> >    - 例如：经典的如$\text{E2LSH}$，其基于动态分桶的改进$\text{QALSH}$
> > 3. 基于$\text{Jaccard}$相似度的$\text{LSH}$：
> >    - $\text{Jaccard}$相似度：两个集合的距离(交集大小$/$并集大小)作为相似度
> >    - $\text{MinHash}$原理：
> >      - 集合的特征：用$k$个不同哈希函数对集合中所有元素求哈希值，取每个哈希函数的最小哈希值，即为集合的$k$个特征
> >      - 集合相似度：两个集合的$\text{Jaccard}$相似度，理论上就是两个集合的重合特征数$/k$ 
> >
> > :two:本文中对于$\text{LSH}$的假设
> >
> > 1. 假设：存在一个哈希家族$\mathcal{H}{⊂}\left({{\mathbb{R}}^{d} \text{→}\mathbb{Z}}\right)$及其分桶哈希函数$\psi$，使得$\text{Pr}[\psi{(x)}\text{=}\psi{(y)}]\text{=}\text{Sim}(x,y)$ 
> > 2. 含义：两个向量相似度越高，被分到同一桶内的概率就更高

# $\textbf{2. }$算法及理论保证

> ## $\textbf{2.1. }$算法的朴素流程
>
> > :one:索引构建
> >
> > 1. 输入：若干向量集，如$D\text{=}\{S_1,S_2,...,S_N\}$
> > 2. 构建：对于每个$S_i\text{=}\{x_{i1},x_{i2},...,x_{im_i}\}$都执行索引构建操作
> >    - 索引分配：为$S_i$中每个元素分配一个唯一索引，例如$x_{ij}$的索引可以为$j$  
> >    - 哈希分桶：用$L$个$\text{LSH}$函数$\psi_1,\psi_2,...,\psi_L$对$S_i$中所有元素进行$L$次分桶
> >    - 索引存储：利用$\mathcal{D}{\left\lbrack{}i\right\rbrack}_{t,h}$数组结构，对应哈希函数$\psi_t$下哈希值为$h$的桶，存储存储了$S_i$中落入该桶的所有向量的索引
> >
> > :two:查询阶段
> >
> > 1. 输入：查询向量集$Q\text{=}\{q_1,q_2,...,q_{m_q}\}$，以及上一步构建的$\text{DESSERT}$索引
> > 2. 编码：照样用那$L$个$\text{LSH}$函数$\psi_1,\psi_2,...,\psi_L$，对$Q$中所有元素进行$L$次分桶
> > 3. 评分：通过检查$q_r\text{∈}Q$和$x_{ij}\text{∈}S_i$的碰撞次数$\text{Count}(q_r,x_{ij})$，得到二者相似度的一个近似$\hat{\text{Sim}}(q_r,x_{ij})\text{=}\cfrac{\text{Count}(q_r,x_{ij})}{L}$
> >    - 原理：为何$\cfrac{\text{Count}(q_r,x_{ij})}{L}$可作为近似评分
> >      - 对$q_r\text{∈}Q$和$x_{ij}\text{∈}S_i$各自进行$L$次分桶后碰撞$\text{Count}(q_r,x_{ij})$次，故估计碰撞率为$\cfrac{\text{Count}(q_r,x_{ij})}{L}$
> >      - 鉴于$\text{Pr}[\psi{(x)}\text{=}\psi{(y)}]\text{=}\text{Sim}(x,y)$的假设，所以碰撞率就是相似度
> >    - 实现：基于$\mathcal{D}{\left\lbrack{}i\right\rbrack}_{t,h}$数组结构(具体优化则见$\text{TinyTable}$)
> >      - 对于$\forall{}q_{r}\text{∈}Q$用哈希函数$\psi_t$得出其哈希值$h$，按照$t,h$索引直接在找到桶$\mathcal{D}{\left\lbrack{}i\right\rbrack}_{t,h}$，如果$x_{ij}$索引也在这儿就算碰撞一次
> >      - 对$\psi_1,\psi_2,...,\psi_L$都进行如上操作，得到最终碰撞次数$\text{Count}(q_r,x_{ij})$，碰撞率(相似度)为$\cfrac{\text{Count}(q_r,x_{ij})}{L}$
> > 4. 聚合：基于相似度$\hat{\text{Sim}}(q_r,x_{ij})$，得到最终的相似度估值$\hat{F}(Q,S_i)$
> >    - 原始输入：由以上$\text{LSH}$得到的，每个$q_r\text{∈}Q$的近似相似度集$\hat{\mathbf{s}}_{r,i}{=}\{\hat{\text{Sim}}(q_r,x_{i1}),\hat{\text{Sim}}(q_r,x_{i2}),...,\hat{\text{Sim}}(q_r,x_{im})\}$ 
> >    - 内部聚合：$\sigma(q_r,\hat{\mathbf{s}}_{r,i}){=}\sigma\{\hat{\text{Sim}}(q_{r},x_{i1}),\hat{\text{Sim}}(q_{r},x_{i2}),...,\hat{\text{Sim}}(q_{r},x_{im_i})\}$，当$\sigma$为$\text{max}$时左式等于$\hat{\text{MaxSim}}(q_r,S_i)$ 
> >    - 外部聚合：$\hat{F}(Q,S_i)\text{=}A\{\sigma(q_1,\hat{\mathbf{s}}_{1,i}),...,\sigma(q_{m_q},\hat{\mathbf{s}}_{{m_q},i})\}$，$A$可以是加权合如$\hat{F}\left({Q,S_i}\right)\text{=}\displaystyle{}\frac{1}{m_q}\sum_{r=1}^{m_q}w_r\sigma(q_r,{\hat{\mathbf{s}}}_{r,i})$
>
> ## $\textbf{2.2. }$算法的理论保证
>
> > :one:乘性约束函数及其引理
> >
> > 1. 定义$\text{4.1}$：$(α, β)\text{-}$乘性极值约束函数
> >
> >    - 条件：给定参数$\alpha,\beta$满足$0\text{<}β\text{≤}1\text{≤}α$，给定向量$\mathbf{x}$并用$\max{(\mathbf{x})}$表示向量$\mathbf{x}$中最大元素值
> >    - 定义：函数$σ(\mathbf{x})\text{:}ℝ^m\text{→}ℝ$在$U$上是$(α,β)\text{-}$极大的，等价于$\forall{\mathbf{x}}\text{∈}U$有$\beta\max{(\mathbf{x})}\text{≤}σ(\mathbf{x})\text{≤}\alpha\max{(\mathbf{x})}$ 
> >    - 例如：当内部聚合$σ(\mathbf{x})$就是$\text{ColBERT}$的$\text{MaxSim}$函数时，就有$α\text{=}β\text{=}1$即$(1,1)\text{-}$极大 
> >
> > 2. 引理$\text{4.1.1}$：平均极值下界衰减引理
> >    - 条件：$\varphi(x)\text{:}ℝ\text{→}ℝ$在区间$I$上是$(α,β)\text{-}$极大的，$x$为标量时$\max{(x)}$退化为$x$本身，即$\beta{x}\text{≤}\varphi(x)\text{≤}\alpha{x}$
> >    - 结论：$\sigma(\mathbf{x}\text{=}\{x_1,x_2,...,x_m\})\text{=}\displaystyle{}\frac{1}{m}\sum_{i=1}^m\varphi(x_i)$在$U\text{=}I^m$上是$\left(\alpha,\cfrac{\beta}{m}\right)\text{-}$极大的
> >    - 示例：满足该引理的实数域函数
> >      |                 $\boldsymbol{\varphi{(x)}}$                  | $\boldsymbol{\beta}$ | $\boldsymbol{\alpha}$ |
> >      | :----------------------------------------------------------: | :------------------: | :-------------------: |
> >      |                             $x$                              |         $1$          |          $1$          |
> >      |                           $e^x–1$                            |         $1$          |         $e–1$         |
> >      | $\text{Sigmid}(x)\text{=}\cfrac{e^x}{1\text{+}e^x}–\cfrac{1}{2}$ |        $0.23$        |        $0.25$         |
> > 3. 引理$\text{4.1.1}$证明过程：取$\mathbf{x}\text{=}\{x_1,x_2,...,x_m\}{∈}I^m$，其中每个$x_i\text{∈}I$ 
> >    - $\varphi(x_i)\text{:}ℝ\text{→}ℝ$在区间$I$上是$(α,β)\text{-}$极大的，$x_i$为标量是$\max{(x_i)}$退化为$x_i$本身，即$\beta{x_i}\text{≤}\varphi(x_i)\text{≤}\alpha{x_i}$
> >    - 确定$\sigma(\mathbf{x})$的上下界
> >      - 对于上界：$\sigma(\mathbf{x})\text{=}\displaystyle{}\frac{1}{m}\sum_{i=1}^m\varphi(x_i){≤}\frac{1}{m}\sum_{i=1}^m\alpha{x_i}\text{=}\frac{\alpha}{m}\sum_{i=1}^mx_i{≤}\frac{\alpha}{m}(m\max{(\mathbf{x}}))\text{=}\alpha\max{(\mathbf{x})}$ 
> >      - 对于下界：$\sigma(\mathbf{x})\text{=}\displaystyle{}\frac{1}{m}\sum_{i=1}^m\varphi(x_i){≥}\frac{1}{m}\sum_{i=1}^m\beta{x_i}\text{=}\frac{\beta}{m}\sum_{i=1}^mx_i{≥}\frac{\beta}{m}\max{(\mathbf{x})}$ 
> >    - 所以$\cfrac{\beta}{m}\max{(\mathbf{x})}{≤}\sigma(\mathbf{x}){≤}\alpha\max{(\mathbf{x})}$，及$\sigma(\mathbf{x})$是$\left(\alpha,\cfrac{\beta}{m}\right)\text{-}$极大的，证毕
> >
> > :two:引理$\text{4.1.2/4.1.3}$：详见[内部聚合的理论保证](https://dannhiroaki.blog.csdn.net/article/details/146991024)
> >
> > :three:定理$\text{4.2}$及运行时间：详见[外部聚合的理论保证](https://dannhiroaki.blog.csdn.net/article/details/147033621) 

# $\textbf{3. }$实现与优化的细节

> ## $\textbf{3.1. }$过滤操作(类似$\textbf{PLAID}$)
>
> > :one:带有预过滤的构建操作
> >
> > 1. 输入：若干向量集，如$D\text{=}\{S_1,S_2,...,S_N\}$
> > 2. 聚类：对所有向量$\{x_{11},...,x_{1m_1}|x_{21},...,x_{2m_1}|...|x_{N1},...,x_{Nm_1}\}$进行$k\text{-Means}$聚类得质心集$\{c_k\}$ 
> > 3. 倒排：为每个质心$c_k$设置一个桶，完成以下操作以构建倒排索引
> >    - 对每个$S_i\text{=}\{x_{i1},x_{i2},...,x_{im_i}\}\text{∈}D$，确定每个$x_{ij}\text{∈}S_i$所属质心，将$S_i$的$\text{ID}$分到该质心所属的桶中
> >    - 遍历完所有$S_i$后提取每个质心的桶，即可得到每个质心$\text{→}$与该质心有关的向量集$\text{ID}$的倒排索引
> > 4. 构建：对于每个$S_i\text{=}\{x_{i1},x_{i2},...,x_{im_i}\}$都执行索引构建操作，不赘述
> >
> > :two:带有预过滤的查询阶段
> >
> > 1. 输入：查询向量集$Q\text{=}\{q_1,q_2,...,q_{m_q}\}$，以及上一步构建的$\text{DESSERT}$索引$\text{+}$质心到向量集的倒排索引
> > 2. 过滤：对于$Q$中的每一个向量$q_r$，获取与每个$q_r$最近的$n_{\text{probe}}$个质心
> >    - 获得这些质心的倒排索引桶，统计这些桶中每个向量集$\text{ID}$出现的计数
> >    - 认为出现计数越多的向量集与查询越相关，故将基数最高的$\text{Top-}k_{\text{filter}}$作为候选向量集$D^\prime$
> > 3. 编码：照样用那$L$个$\text{LSH}$函数$\psi_1,\psi_2,...,\psi_L$，对$Q$中所有元素进行$L$次分桶
> > 4. 得分：评分和聚合相同不再赘述，不同在于考虑的范围从原来的全部向量集$D$缩小为了候选向量集$D^\prime$  
>
> ## $\textbf{3.2. }$哈希拼接
>
> > :one:动机：为何拼接可以提升分桶效率
> >
> > 1. 事实上：基于$\text{Transformer}$的语言模型由于其注意力偏好，会使得嵌入后的向量彼此邻近
> > 2. 分桶时：以$\text{ColBERT}$为例，其嵌入出来的向量平均余弦相似度为$\text{0.3}$
> >    - 假设$x$落入桶$\psi(x)$，则其它任意$y$由于$\text{Pr}[\psi{(x)}\text{=}\psi{(y)}]\text{=}\text{Sim}(x,y)\text{≈}0.3$也有$\text{30\%}$左右的概率落入桶$\psi(x)$
> >    - 这将导致大量的向量集中于$\psi(x)$一个桶中，即桶过度填充
> >
> > :two:将拼接应用于分桶
> >
> > 1. 原有的：$\psi_t$对$x_{ij}$的分桶结果为$\psi{_t(x_{ij})}$，即$x_{ij}$哈希值是一个<mark>标量</mark>
> > 2. 拼接后：将$\psi_t$分解为子哈希函数$\psi_{t,1},\psi_{t,2},...,\psi_{t,C}$，则$x_{ij}$哈希值为<mark>向量</mark>$\psi_t(x_{ij})\text{=}\{\psi_{t,1}(x_{ij}),\psi_{t,2}(x_{ij}),...,\psi_{t,C}(x_{ij})\}$  
> > 3. 碰撞率：拼接后变为原来的$C$次幂
> >    - 拼接后要求<font color=red>对于每个$\psi_{t,i}$都有$\text{Pr}[\psi_{t,i}{(x)}\text{=}\psi_{t,i}{(y)}]\text{=}\text{Sim}(x,y)$时$xy$才算碰撞</font>，即对$\psi_t$才有$\text{Pr}[\psi_{t}{(x)}\text{=}\psi_{t}{(y)}]$ 
> >    - 故对于$\psi_t$有$\text{Pr}[\psi_{t}{(x)}\text{=}\psi_{t}{(y)}]\text{=}\displaystyle{\prod_{i=1}^C}\text{Pr}[\psi_{t,i}{(x)}\text{=}\psi_{t,i}{(y)}]\text{=}\text{Sim}(x,y)^C$，即$\displaystyle\text{Sim}\left({x,y}\right)\text{=}\exp\left(\frac{\ln\{\text{Pr}[\psi(x)\text{=}\psi(y)]\}}{C}\right)$ 
> >
> > :three:将拼接应用于评分
> >
> > 1. 原始的评分机制：
> >    - 对于$q_r\text{∈}Q$和$x_{ij}\text{∈}S_i$，检查其在$L$个哈希表中的碰撞次数$\text{Count}(q_r,x_{ij})$ 
> >    - 估计碰撞概率为$\hat{\text{Pr}}[\psi(q_r)\text{=}\psi(x_{ij})]\text{=}\cfrac{\text{Count}(q_r,x_{ij})}{L}$
> >    - 基于$\text{Pr}[\psi{(x)}\text{=}\psi{(y)}]\text{=}\text{Sim}(x,y)$，可得$x_{ij}$和$q_r$相似度为$\hat{\text{Sim}}(q_r,x_{ij})\text{=}\cfrac{\text{Count}(q_r,x_{ij})}{L}$
> > 2. 拼接偶的评分机制：
> >    - 对于$q_r\text{∈}Q$和$x_{ij}\text{∈}S_i$，检查其在$L$个哈希表中的碰撞次数$\text{Count}(q_r,x_{ij})$ 
> >    - 估计碰撞概率为$\hat{\text{Pr}}[\psi(q_r)\text{=}\psi(x_{ij})]\text{=}\cfrac{\text{Count}(q_r,x_{ij})}{L}$
> >    - 基于$\displaystyle\text{Sim}\left({x,y}\right)\text{=}\exp\left(\frac{\ln\{\text{Pr}[\psi(x)\text{=}\psi(y)]\}}{C}\right)$，可得$x_{ij}$和$q_r$相似度为$\displaystyle{}\hat{\text{Sim}}(q_r,x_{ij})\text{=}\exp\left(\frac{\ln\left\{\frac{\text{Count}(q_r,x_{ij})}{L}\right\}}{C}\right)$  
>
> ## $\textbf{3.3. }$系统级的优化: $\textbf{TinyTable}$ 
>
> > :one:动机：对于$\mathcal{D}{\left\lbrack{}i\right\rbrack}_{t,h}$的实现(即每个桶的存储)太烂，以标准的`std::vector`实现为例
> >
> > ```txt
> > 👉假定有三个向量集, ψ1/ψ2两个哈希函数, 每个哈希函数有3个桶
> > S1 = {x11, x12, x13}
> > S2 = {x21, x22, x23, x24, x25}
> > S3 = {x31, x32, x33, x34}
> > 👉对S1/S2/S3中所有的向量, 都分别用ψ1/ψ2处理一遍, 得到分桶结果
> > S1的分桶           | S2的分桶           | S3的分桶
> > ψ1-1: {x11}      |  ψ1-1: {x21}      |  ψ1-1: {x31}
> > ψ1-2: {}         |  ψ1-2: {x25}      |  ψ1-2: {}
> > ψ1-3: {}         |  ψ1-3: {x22, x24} |  ψ1-3: {x32}
> > ψ2-1: {x12, x13} |  ψ2-1: {}         |  ψ2-1: {}
> > ψ2-2: {}         |  ψ2-2: {}         |  ψ2-2: {x33}
> > ψ2-3: {}         |  ψ2-3: {x23}      |  ψ2-3: {x34}
> > 👉原始的解决方案会为每个桶创建一个std::vector对象以存储落入该桶的向量索引
> > S1的分桶           | S2的分桶           | S3的分桶
> > ψ1-1: {11}       |  ψ1-1: {21}       |  ψ1-1: {31}
> > ψ1-2: {}         |  ψ1-2: {25}       |  ψ1-2: {}
> > ψ1-3: {}         |  ψ1-3: {22, 24}   |  ψ1-3: {32}
> > ψ2-1: {12, 13}   |  ψ2-1: {}         |  ψ2-1: {}
> > ψ2-2: {}         |  ψ2-2: {}         |  ψ2-2: {33}
> > ψ2-3: {}         |  ψ2-3: {23}       |  ψ2-3: {34}
> > ```
> > 1. 对于每个桶：
> >    - 每个哈希桶都需要单独存放一个`std::vector`对象
> >    - 因此每个桶至少需要三个指针，分别指向已分配内存起始$/$已分配内存终点$/$已使用内存终点
> >    - 因此即使哈希桶为空，每个桶也必须占三指针共$\text{24}$字节
> > 2. 总占用空间：
> >    - 假设有$\text{100w}$个向量集，每个向量集有$L\text{=}64$个哈希函数(哈希表)，每个哈希函数有$r\text{=}128$个桶
> >    - 所有空桶就占用了$\text{196GB}$内存，编码$\text{100w}$个向量集还要超过此
> >
> > :two:$\text{TinyTable}$的优化实现
> >
> > 1. 每个$S_i$都有一个向量$\text{ID}$向量：相当于把该$S_i$所有的桶拍扁放在一个向量中
> >    ```txt
> >    👉原始的分桶实现
> >      S1的分桶           | S2的分桶           | S3的分桶
> >       ψ1-1: {11}       |  ψ1-1: {21}       |  ψ1-1: {31}
> >       ψ1-2: {}         |  ψ1-2: {25}       |  ψ1-2: {}
> >       ψ1-3: {}         |  ψ1-3: {22, 24}   |  ψ1-3: {32}
> >       ψ2-1: {12, 13}   |  ψ2-1: {}         |  ψ2-1: {}
> >       ψ2-2: {}         |  ψ2-2: {}         |  ψ2-2: {33}
> >       ψ2-3: {}         |  ψ2-3: {23}       |  ψ2-3: {34}
> >    👉将原来的每个桶按顺序合并, 得到每个向量集的向量ID表
> >      S1_ID = {11, 12, 13}
> >      S2_ID = {21, 25, 22, 24, 23}
> >      S3_ID = {31, 32, 33, 34}
> >    ```
> > 2. 每个$S_i$同时有一个偏移向量：用于对每个$S_i$的$\text{ID}$列表进行分割，以确定每个桶的边界
> >    ```txt
> >    👉偏移向量的范式
> >      [ψ1-1起, ψ1-2起, ψ1-3起, ψ1-3止, ψ2-1起, ψ2-2起, ψ2-3起, ψ2-3止]
> >    👉S1_offset = {0, 1, 1, 1, 1, 3, 3, 3}
> >         11 12 13           ψ1-1 ψ1-2 ψ1-3 ψ2-1 ψ2-2 ψ2-3
> >       起 0  1  2 ----------> 0    1    1    1    3    3  
> >       止 1  2  3             1    1    1    3    3    3 
> >    👉S2_offset = {0, 1, 2, 4, 4, 4, 4, 5}
> >         21 25 22 24 23     ψ1-1 ψ1-2 ψ1-3 ψ2-1 ψ2-2 ψ2-3
> >       起 0  1  2  3  4 ----> 0    1    2    4    4    4
> >       止 1  2  3  4  5       1    2    4    4    4    5
> >    👉S3_offset = {0, 1, 1, 2, 2, 2, 3, 4}
> >         31 32 33 34        ψ1-1 ψ1-2 ψ1-3 ψ2-1 ψ2-2 ψ2-3
> >       起 0  1  2  3 -------> 0    1    1    2    2    3
> >       止 1  2  3  4          1    1    2    2    3    4
> >    ```
> > 3. 占用内存分析：
> >    - 同样假设有$\text{100w}$个向量集，每个向量集有$L\text{=}64$个哈希函数(哈希表)，每个哈希函数有$r\text{=}128$个桶
> >    - $\text{ID}$向量的长度和其对应的向量集$S_i$长度一样(假设平均长$\text{100}$)，偏移向量的长度则统一为$L(r\text{+}1)$  
> >    - 假设向量集$S_i$长度都不超过$\text{256}$，则每个$\text{ID}$和每个偏移量都是$\text{1}$字节，则总占用内存大小为$\text{14.7GB}$ 

# $\textbf{4. }$实验结果概述

> :one:相比暴力搜索的时间优势
>
> 1. 做法：由随机的$\text{Glove}$生成一系列"人造"向量，对比$\text{DESSERT}$和暴力搜索的性能
> 2. 结果：对比查询时间(而非性能)，$\text{DESSERT}$比暴力方法快$\text{10-50}$倍
>
> :two:段落检索的性能优势
>
> 1. 数据集：$\text{LoTTe}$和$\text{Ms-Marco}$
> 2. 对比结果：检索质量(召回率)略微降低，但是检延迟下降为原来的$\text{1/3}$左右