# $\textbf{1. }$导论与背景

> ## $\textbf{1.1. }$研究背景
>
> > :one:两种文本相似性检索模型
> >
> > |        类型         | 嵌入方式                           | 相似度计算                                              |
> > | :-----------------: | ---------------------------------- | ------------------------------------------------------- |
> > | 单向量$\text{(SV)}$ | 对整个句子生成唯一的嵌入           | $\text{MIPS}$算法，从一堆向量找出与$q$有最大内积的      |
> > | 多向量$\text{(MV)}$ | 对每个$\text{Token}$都生成一个嵌入 | $\text{Chamfer}$相似度，也就是所谓的$\text{MaxSim}$之和 |
> >
> > :two:多向量模型的问题：检索成本还是高过单向量
> >
> > 1. 空间占用上：$\text{Token}$数量过多，需要大量的存储
> > 2. 计算成本上：缺乏对于$\text{Chamfer}$的优化，大多的优化只针对于$\text{MIPS}$而无法用在$\text{Chamfer}$上
> >
> > :three:改进的尝试：将$\text{MV}$改为基于$\text{SV}$的$\text{MIPS}$流水(单向量启发式方法)
> >
> > 1. $\text{SV}$阶段：每个查询$\text{Token}\xrightarrow{\text{MIPS}}$最相似的文档$\text{Token}$
> > 2. $\text{MV}$阶段：收集所有的最相似文档$\text{Token}$，再用原始$\text{Chamfer}$相似度得到最终评分
>
> ## $\textbf{1.2. }$本文工作$\textbf{: MUVERA}$概述
>
> > :one:$\text{Chamfer}$相似度
> >
> > 1. 相似度的定义：对查询$Q/$段落$P$的每个$\text{Token}(q/p)$，相似度为$\displaystyle{}\text{Chamfer}(Q,P)\text{=}\sum_{q \text{∈} Q} \max _{p \text{∈} P}\langle q, p\rangle$ 
> > 2. 扩展到最邻近：在$D\text{=}\left\{P_{1}, \ldots, P_{n}\right\}$中找到与$Q$之间$\text{Chamfer}$相似度最高的文档$P^{*} \text{∈} D$ 
> >
> > :two:$\text{MUVERA}$概述
> >
> > 1. 核心思想：将所向量压缩为单向量，原有的$\text{Chamfer}$搜索也变成$\text{MIPS}$搜索
> >
> >    - 维度压缩：特殊的映射函数$\begin{cases}\mathbf{F}_{\mathrm{que}}: 2^{\mathbb{R}^{d}} \rightarrow \mathbb{R}^{d_{\mathrm{FDE}}}\\\\\mathbf{F}_{\text{doc}}: 2^{\mathbb{R}^{d}} \rightarrow \mathbb{R}^{d_{\mathrm{FDE}}}\end{cases}\text{→}$将多向量压缩为固定$d_{\text{FDE}}$维**单向量编码** 
> >
> >    - 相似度计算：用内积$\left\langle\mathbf{F}_{\mathrm{que}}(Q), \mathbf{F}_{\text{doc}}(P)\right\rangle$作为原有$\displaystyle{}\text{Chamfer}(Q,P)\text{=}\sum_{q \text{∈} Q} \max _{p \text{∈} P}\langle q, p\rangle$的替代
> >
> > 2. 工作流程：
> >
> >    - 预处理：对所有文档进行$\mathbf{F}_{\text{doc}}$映射得到$\mathbf{F}_{\text{doc}}(P_i)$的固定维度编码($\text{FDEs}$)
> >    - 查询初排：对查询$Q$进行$\mathbf{F}_{\text{que}}$映射得到$\mathbf{F}_{\text{que}}(Q)$，计算$\left\langle\mathbf{F}_{\mathrm{que}}(Q), \mathbf{F}_{\text{doc}}(P_i)\right\rangle$得到$\text{Top-}k$文档
> >
> >    - 查询重排：再用完整的$\displaystyle{}\sum_{q \in Q} \max _{p \in P}\langle q, p\rangle$相似度，对$\text{Top-}k$个文档进行重排
> >
> > 3. 备注的点：
> >
> >    - $\mathbf{F}_{\text{doc}}/\mathbf{F}_{\text{que}}$是与数据分布无关，由此对不同分布的处理都有鲁棒性
> >    - $\left\langle\mathbf{F}_{\mathrm{que}}(Q), \mathbf{F}_{\text{doc}}(P_i)\right\rangle$求解过程在高度优化的$\text{MIPS}$求解器中完成

# $\textbf{2. }$固定维度嵌入($\text{FDEs}$)

> ## $\textbf{2.1. FDE}$的生成过程

### 





### 直觉上的做法

:one:最佳映射($\pi\text{: }Q \text{→} P$)：对于每个查询向量$q\text{∈}Q$，$\pi(q)\text{←}P$中与$q$最相似的向量

1. 由此：原来的$\text{Chamfer}$距离变为了$\displaystyle{}\sum_{q \text{∈} Q} \max _{p \text{∈} P}\langle q, p\rangle\text{=}\sum_{q \in Q}\langle q, \pi(q)\rangle$
2. 进一步：让$\vec{q}\text{=}[q_1,q_2,...]$拼接向量，$\vec{p}\text{=}[\pi(q_1),\pi(q_2),...]$拼接向量，则$\displaystyle{}\sum_{q \in Q}\langle q, \pi(q)\rangle\text{=}\langle\vec{q}, \vec{p}\rangle$ 
   - 由此可以直接用内积表示$\text{Chamfer}$相似度

:two:存在的问题：实际中难以找到最佳映射$\pi$，对不同的[文档$\xleftrightarrow{}$查询]对有不同的$\pi$ 

:three:本文的做法：避免显式地求解$\pi$，而是一种近似方法

1. 先使用一种==随机化的顺序排列==，使相似的向量能在某种排序下相近
2. 将向量在该顺序下拼接成单向量
3. 最后再用单向量计算点积近似距离

### 对直觉的进一步思考

:one:空间分区：先将用分区函数$\varphi\text{: }\mathbb{R}^{d}\text{→}[B]$将==潜在空间==分簇

1. 用$\text{K-Means}$或者$\text{LSH}$，使得较近的向量被分为同一簇

2. 类似于$\text{LSH}$的思想，查询$q \text{∈} Q$应该落入与其最邻近$p \text{∈} P$同一个簇，即$\varphi(q)\text{=}\varphi(p)$

3. 相似度就变为了每簇内$\text{MaxSim}$总和再求和$\displaystyle{}\text{Chamfer}(Q, P)=\sum_{k=1}^{B} \sum_{\substack{q \in Q\text{∧}\varphi(q)=k}} \max _{\substack{p \in P\text{∧}\varphi(p)=k}}\langle q, p\rangle$  

   ```txt
   查询嵌入: {q1,q2,q3}
   段落嵌入: {d1,d2,d3,d4}
   
   假设最邻近对应关系为: q1<->d2,q2<->d4,q3<->d3
   
   不分簇的方法
   q1 <-> {d1,d2,d3,d4} -> MaxSim-1(q1<->d2)
   q2 <-> {d1,d2,d3,d4} -> MaxSim-2(q2<->d4)
   q3 <-> {d1,d2,d3,d4} -> MaxSim-3(q3<->d3)
   
   分簇的方法: 假设分簇为{q1,q2,d1,d2,d4}和{q3,d3}
   q1 <-> {d1,d2,d4} -> MaxSim-11(q1<->d2)
   q2 <-> {d1,d2,d4} -> MaxSim-12(q2<->d4)
   q3 <-> {d3}       -> MaxSim-21(q3<->d3)
   
   
   只要同一簇内必定有q的最邻近,分簇前后的距离计算就可划等号
   ```

:two:对碰撞的分析

<img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250107004939032.png" alt="image-20250107004939032" width=650 /> 

1. 只一个$p \text{∈} P$与$q$碰撞，即桶内只有$\{q_1,q_2,...|p\}$，可以得到精确的相似度
   - 将所有落入第$k$簇的所有$q \text{∈} Q$累加成$\vec{q}_{(k)}$，所有落入第$k$簇的所有$p \text{∈} P$累加成$\vec{p}_{(k)}$ 
     - $\displaystyle{}\vec{q}_{(k)}=\sum_{\substack{q \in Q \text{∧} \varphi(q)=k}} q$
     - $\displaystyle{}\vec{p}_{(k)}=\sum_{\substack{q \in Q \text{∧} \varphi(p)=k}} p$ 
   - 对所有$B$个簇进行拼接，形成$\vec{q}\text{=}\left(\vec{q}_{(1)}, \ldots, \vec{q}_{(B)}\right)$和$\vec{p}\text{=}\left(\vec{p}_{(1)}, \ldots, \vec{p}_{(B)}\right)$ 
   - 所以$\text{Chamfer}$相似度直接就是$\langle\vec{q}, \vec{p}\rangle$ 
2. 有多个$p \text{∈} P$与$q$碰撞，即桶内为$\{q_1,q_2,...|p_1,p_2,...\}$，只能得到近似的相似度
   - 原有$\langle\vec{q}, \vec{p}\rangle$就偏离$\text{Chamfer}$相似度了，$\vec{q}_{(k)}$还是一个$q$的累加，但$\vec{p}_{(k)}$就变成多个$p$的累加了
   - 解决办法就是改变$\vec{p}_{(k)}$的定义，即变成桶内$\{p_1,p_2,...\}$的质心，或者说变成多个$p$的平均
     - $\displaystyle{}\vec{q}_{(k)}\text=\sum_{\substack{q \in Q \text{∧} \varphi(q)=k}} q$
     - $\displaystyle{}\vec{p}_{(k)}=\cfrac{1}{\left|P \cap \boldsymbol{\varphi}^{-1}(k)\right|} \sum_{\substack{p \in P \text{∧} \varphi(p)=k}} p$
     - 其中$\left|P \cap \boldsymbol{\varphi}^{-1}(k)\right|$表示映射到桶内$p_i$的数量，所谓的质心就是求所有$p_i$坐标的平均值
   - 对所有$B$个簇进行拼接，形成$\vec{q}\text{=}\left(\vec{q}_{(1)}, \ldots, \vec{q}_{(B)}\right)$和$\vec{p}\text{=}\left(\vec{p}_{(1)}, \ldots, \vec{p}_{(B)}\right)$ 
     - 由此$\displaystyle{}\langle\vec{q}, \vec{p}\rangle=\sum_{k=1}^{B}\vec{q}_{(k)}\vec{p}_{(k)}\text{=}\sum_{k=1}^{B}\left(\sum_{\substack{q \in Q \text{∧} \varphi(q)=k}} q\right)\left(\cfrac{1}{\left|P \cap \boldsymbol{\varphi}^{-1}(k)\right|} \sum_{\substack{p \in P \text{∧} \varphi(p)=k}} p\right)$ 
     - 即$\displaystyle{}\langle\vec{q}, \vec{p}\rangle=\sum_{k=1}^{B} \sum_{\substack{q \in Q \text{∧} \varphi(q)=k}} \frac{1}{\left|P \cap \varphi^{-1}(k)\right|} \sum_{\substack{p \in P \text{∧} \varphi(p)=k}}\langle q, p\rangle$
     - 通过如下的降维方法，使得$\langle\vec{q}, \vec{p}\rangle$最终可以作为$\text{Chamfer}$相似度的近似

### 降维优化

:one:向量$\vec{q}, \vec{p}$的维度：每个$\vec{q}_{(k)}$固定维度为$d$，然后一共$B$个向量合在一起，所以是$dB$维

- 实际上$dB$有点高了，需要降维

:two:降维方法：对$\vec{q}_{(k)}$应用随机线性投影$\psi\text{: }\mathbb{R}^{d} \text{→} \mathbb{R}^{d_{\text{proj}}}(d_{\text{proj}} \text{<} d)$ 

1. 具体定义：$\boldsymbol{\psi}(x)\text{=}\left(\cfrac{1}{\sqrt{d_{\text{proj}}}}\right)\mathbf{S}\text{×}x$，其中$\mathbf{S}\text{∈}\mathbb{R}^{d_{\text{proj}} \times d}$为随机矩阵，每个元素均匀分布在$\pm 1$间
   - 但注意：$d\text{=}d_{\text{proj}}$时候，$\boldsymbol{\psi}(x)\text{=}x$
2. 投影后记作：$\vec{q}_{(k), \psi}\text{=}\psi\left(\vec{q}_{(k)}\right)$还有$\vec{p}_{(k), \psi}\text{=}\psi\left(\vec{p}_{(k)}\right)$  
3. 得到$\text{FDE}$的最终定义：$\vec{q}_{\psi}=\left(\vec{q}_{(1), \psi}, \ldots, \vec{q}_{(B), \psi}\right)$还有$\vec{p}_{\psi}=\left(\vec{p}_{(1), \psi}, \ldots, \vec{p}_{(B), \psi}\right)$
   - 维度为$B \text{×} d_{\text{proj}}$ 

:three:多次重复：

1. 不断重复上述降维过程$R_{\text{reps}} \text{≥} 1$次，得到如下的重复结果
   - $\vec{q}_{1, \psi},\vec{q}_{2, \psi}, \ldots, \vec{q}_{R_{\text{reps}}, \psi}$
   - $\vec{p}_{1, \psi},\vec{p}_{2, \psi}, \ldots, \vec{p}_{R_{\text{reps}}, \psi}$ 
2. 最终再对每一次的降维进行拼接，得到修改后的$\text{FDE}$最终定义
   - $\mathbf{F}_{\mathrm{que}}(Q)\text{=}\left(\vec{q}_{1, \psi},\vec{q}_{2, \psi}, \ldots, \vec{q}_{R_{\text{reps}}, \psi}\right)$
   - $\mathbf{F}_{\mathrm{doc}}(P)\text{=}\left(\vec{p}_{1, \psi},\vec{p}_{2, \psi}, \ldots, \vec{p}_{R_{\text{reps}}, \psi}\right)$ 
   - 最终的维度为$d_{\text{FDE}}\text{=}B \text{×} d_{\text{proj}} \text{×} R_{\text{reps}}$

### 如何进行空间划分

:zero:空间划分函数$\varphi$

- 所期望$\varphi$达到的特性：距离近的点被分到同一簇，较远的点分到不同簇，也就是$\text{LSH}$的特性

:one:$\text{SimHash}$的做法

1. 核心思想：
   - 将原始向量映射为定长的二进制哈希值(例如$\text{128bit}$的布尔向量)
   - 计算两个二进制哈希值的海明距离，相似的海明距离代表着原始向量在空间中的相邻
2. 详细定义：
   - 先随机从高斯分布中抽取$k_{\text{sim}} \text{≥} 1$个向量$g_{1}, \ldots, g_{k_{\text{sim}}} \text{∈} \mathbb{R}^{d}$作为$k_{\text{sim}}$个超平面的法向量
   - 定义划分函数为$\varphi(x)\text{=}\left(\mathbf{1}\left(\left\langle g_{1}, x\right\rangle\text{>}0\right), \ldots, \mathbf{1}\left(\left\langle g_{k_{\text{sim}}}, x\right\rangle\text{>}0\right)\right)$
     - $\mathbf{1}\left(\left\langle g_{i}, x\right\rangle\text{>}0\right)$表示$\langle g_{i}, x\rangle\text{>}0$成立，即$x$投影在超平面的正侧时，将该位设为$1$ 
     - 对每位按此设置，最终得到长度为$k_{\text{sim}}$的二进制串
   - 最后再将该二进制串，转化为十进制数，这个十进制数就是$x$所属簇的索引
3. 对定义的几何理解：
   - 选取$k_{\text{sim}}$个超平面的法向量，其实就是选取$k_{\text{sim}}$个超平面，将空间分为$2^{k_{\mathrm{sim}}}$个区域
     - $2^{k_{\mathrm{sim}}}$个区域的结论只在高维空间中成立，低维空间中实际少于这个数
     - 再二维空间中以$k_{\text{sim}}\text{=3}$为例，实际上代表三条直线，对应的最大区域数就是$6$
   - 每个区域对应一个簇，所以共有$B\text{=}2^{k_{\mathrm{sim}}}$个簇

:two:$k\text{-Means}$的做法：先对空间$k\text{-Means}$出$k_{\text{CENTER}}$个质心，$\varphi(x)\text{←}$与$x$最接近的质心

### 填补空簇 

:one:最大的误差来源

1. $q \text{∈} Q$的真实最邻近$p \text{∈} P$被分到了不同的簇
2. 这样一来$q \text{∈} Q$在自己的簇中，无论如何都得不到真实的$\text{MaxSim}$

:two:存在一个$\text{Trade-Off}$

1. 增大$B$的值

   - 会使得$q \text{∈} Q$真实最邻近$p \text{∈} P$被分到不同簇的概率变大

   - 最极端的情况下，所有$q \text{∈} Q$都$\text{Miss}$，即不予任何的$p \text{∈} P$碰撞

- 减小$B$的值

  - 分簇错误的概率变小了，但是一个簇中$p_i$数量就会增多

  - 导致质心$\vec{p}_{(k)}$严重偏离真实最邻近$p$ 

:three:解决办法：填补空簇

1. 针对的情况，当某个簇$k$中没有任何的$p$落入时
2. 原有的方案：$\displaystyle{}\vec{p}_{(k)}=\sum p\text{}=\vec{0}$ 
3. 现在的方案：$\vec{p}_{(k)}$为距离簇最近的一点$p \text{∈} P$
   - 关于与簇的距离
     - 在$\text{SimHash}$中每个簇都有一二进制表示，看的是该二进制表示与$p$点二进制表示的海明距离最小值
     - 在$k\text{-Means}$中每个簇都有一质心，看的是与质心最近的点$p$

### 最终投影

:one:最终的$\text{FDE}$之前已经得到了

- $\mathbf{F}_{\mathrm{que}}(Q)\text{=}\left(\vec{q}_{1, \psi},\vec{q}_{2, \psi}, \ldots, \vec{q}_{R_{\text{reps}}, \psi}\right)$
- $\mathbf{F}_{\mathrm{doc}}(P)\text{=}\left(\vec{p}_{1, \psi},\vec{p}_{2, \psi}, \ldots, \vec{p}_{R_{\text{reps}}, \psi}\right)$ 
- 最终的维度为$d_{\text{FDE}}\text{=}B \text{×} d_{\text{proj}} \text{×} R_{\text{reps}}$

:two:最终投影就是，还要再对以上结构投影一次

- $\psi^{\prime}\text{: }\mathbb{R}^{d_{\text{FDE}}} \text{→} \mathbb{R}^{d_{\text{final}}}$  
- 显得很多余，但是还是可以小小地提升$\text{Recall}$

### 理论保证

:one:一些预备

1. 归一化的$\text{Chamfer}$相似度：$\text{NChamfer}(Q, P)=\cfrac{1}{|Q|} \text{Chamfer}(Q, P)\text{∈}[-1,1]$，即$\text{MaxSim}$平均值
2. 假设所有嵌入都是归一化的：即$\|q\|_{2}\text{=}\sqrt{q_1^2\text{+}q_2^2\text{+}\cdots\text{+}q_n^2}\text{=}1$，$p$也同理

:two:定理$1$：$\text{FDE}$对$\text{Chamfer}$相似度的近似程度，可以达到$ε\text{-}$加相似性，即定理$1$的结论

1. 给定条件：$\forall\varepsilon, \delta \text{>} 0$，所有向量均为单位向量，$m\text{=}|Q|\text{+}|P|$ 
2. 设定参数：$k_{\text {sim }}\text{=}O\left(\cfrac{\log \left(m \delta^{-1}\right)}{\varepsilon}\right)$，$d_{\text {proj }}\text{=}O\left(\cfrac{1}{\varepsilon^{2}} \log \left(\cfrac{m}{\varepsilon \delta}\right)\right)$，$R_{\text {reps }}\text{=}1$
   - 由此得到：$d_{F D E}=(\cfrac{m}{\delta})^{O(\frac{1} { \varepsilon})}$ 
3. 结论：$\text{NChamfer}(Q, P)-\varepsilon \leqslant \cfrac{1}{|Q|}\left\langle\mathbf{F}_{q}(Q), \mathbf{F}_{d o c}(P)\right\rangle \leqslant \text{NChamfer}(Q, P)+\varepsilon$ 
   - $ε$的含义即，在$\{q_1,q_2,...|p_1,p_2,...\}$桶内，$\text{dist}(q_i,p_j)\text{≤}ε$ 

:three:定理$2$：将原有的$P$便变成$P_i$，即有不止一个段落

1. 给定条件：$\forall\varepsilon \text{>} 0$，$m\text{=}|Q|\text{+}|P_i|_{\text{max}}$ 

2. 设定参数：$k_{\text {sim }}\text{=}O\left(\cfrac{\log m}{\varepsilon}\right)$，$d_{\text {proj }}\text{=}O\left(\cfrac{1}{\varepsilon^{2}} \log (\cfrac{m} {\varepsilon})\right)$，$R_{\text {reps }}\text{=}O\left(\cfrac{1}{\varepsilon^{2}} \log n\right)$ 

   - 由此得到：$d_{F D E}=m^{O(\frac{1} { \varepsilon})} \text{×} \log n$

3. 结论：假设$P_i$是用本文方法找到的近似最相似文档，$P_j$是真实的最相似文档

   - 二者的$\text{Chamfer}$距离的差距不会超过$\varepsilon$，即$\text{NChamfer}\left(Q, P_{i^{*}}\right) \text{≥} \max _{i \in[n]} \text{NChamfer}\left(Q, P_{i}\right)-\varepsilon$ 
   - 这一结论以极高的概率$1-\cfrac{1}{\text{poly}(n)}$满足

4. 结论：从$\{P_{1}, \ldots, P_{n}\}$中找出最$Q$的最相似文档的时间复杂度为$\displaystyle{}O\left(|Q| \max \{d, n\} \frac{1}{\varepsilon^{4}} \log \left(\frac{m}{\varepsilon}\right) \log n\right)$

   





 比如我现在有一个文档D，嵌入之后得到了多向量{d1,d2,.....} 模型中所用的嵌入模型BERT其实是基于Transformer的，有研究指出Transformer-based模型，倾向于把相邻的两个Token嵌入地非常非常相似 比如说d_i和d_i+1就会非常相似 这导致了一个问题，即一个文档D经过LSH之后，d_i碰撞的概率太高了，也就使得在很多情况下只能用d_i集合的质心作为最邻近的近似替代，导致距离的误差变大