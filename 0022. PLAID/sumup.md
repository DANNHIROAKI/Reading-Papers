:point_right:前情提要：

1. [神经网络自然语言模型概述](https://dannhiroaki.blog.csdn.net/article/details/143985271)
2. [$\text{Transformer}$与注意力机制概述](https://dannhiroaki.blog.csdn.net/article/details/144035750)

:books:相关论文：

1. [$\text{BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding}$](https://doi.org/10.48550/arXiv.1810.04805)
   - 提出了基于双向深度$\text{Transformer}$的$\text{BERT}$交叉编码器
   - [$\text{BERT}$的总结](https://blog.csdn.net/qq_64091900/article/details/144120987)
2. [$\text{ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT}$](https://doi.org/10.48550/arXiv.2004.12832)
   - 提出了基于$\text{BERT}$编码的后期$\text{Token}$级交互模式
   - [$\text{ColBERTv1}$的总结](https://dannhiroaki.blog.csdn.net/article/details/144157480)
3. [$\text{ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction}$](https://doi.org/10.48550/arXiv.2112.01488)
   - 保留了$\text{ColBERT}$的后期交互架构，但从训练策略$/$嵌入压缩$/$数据集上优化
   - [$\text{ColBERTv2}$的总结](https://dannhiroaki.blog.csdn.net/article/details/144378880)
4. [$\text{PLAID: An Efficient Engine for Late Interaction Retrieval}$](https://doi.org/10.48550/arXiv.2205.09707)
   - 在$\text{ColBERTv2}$的基础上，进一步改进了检索策略
   - [$\text{PLAID}$的总结](https://blog.csdn.net/qq_64091900/article/details/144410178)
5. [$\text{EMVB: Efficient Multi-Vector Dense Retrieval Using Bit Vectors}$](https://doi.org/10.48550/arXiv.2404.02805)  

---

[TOC]

---

# $\textbf{1. }$背景与导论

> ## $\textbf{1.1. }$研究背景: 后期交互
>
> > :one:后期交互的含义
> >
> > 1. 概念：将查询$/$段落编码为**词级向量**，让其之间互相交互评分，突破了单向量模型排序效果的瓶颈
> > 2. 改进：基于监督调优$+$残差压缩的$\text{ColBERTv2}$交互，优于大多稀疏(词袋)$/$稠密(深度学习)模型
> >
> > :two:后期交互的挑战
> >
> > 1. 计算成本：需要将每个查询$/$段落表示为一个完整的向量矩阵，计算开销大
> > 2. 硬件实现：后期交互需要专门的设施和优化，难以部署
> > 3. 可扩展性：后期交互是词一级的，所以成熟的单向量模型无法直接进行后期交互
> >
> > :three:后期交互的优化
> >
> > 1. 现状：有一定的工作尝试优化了后期交互的部分组件(存储$/$计算....)
> > 2. 空白：缺乏对后期交互$\text{End-to-end}$(即整个过程)的优化，而这正是本文$\text{PLAID}$致力于的
>
> ## $\textbf{1.2. }$本文研究的概述
>
> > ### $\textbf{1.2.1. }$提出$\textbf{PLAID}$的动机
> >
> > > :one:研究基础：全盘采纳$\text{ColBERTv1}$后期交互架构，以及$\text{ColBERTv2}$优化(去噪监督$+$残差压缩)
> > >
> > > :two:研究目的：优化后期交互，以降低模型在大规模数据集上的搜索延迟
> > >
> > > :three:基本策略：在排序阶段，跳过全面评分$\text{→}$减小解压的规模
> > >
> > > |        策略        | 重排: 残差解压前                                             | 重排: 残差解压后                             |
> > > | :----------------: | ------------------------------------------------------------ | -------------------------------------------- |
> > > | $\text{ColBERTv2}$ | 通过$\text{ANN}$筛选出初步候选段落                           | 解压**所有**候选段落的精确嵌入$\text{+}$评分 |
> > > |   $\text{PLAID}$   | 候选段落$\xrightarrow[识别并筛选]{(已有的)质心组件}$潜在高分段落 | 解压**部分**候选段落的精确嵌入$\text{+}$评分 |
> >
> > ### $\textbf{1.2.2. }\textbf{PLAID}$的创新点
> >
> > > :one:$\text{PLAID}$的机制：
> > >
> > > 1. 质心交互机制：在不解压段落嵌入的残差压缩情况下，通过质心排除较弱的段落候选项
> > >
> > >    - 预处理阶段：
> > >
> > >      |   操作   |  阶段  | 描述                                        | 备注                  |
> > >      | :------: | :----: | :------------------------------------------ | :-------------------- |
> > >      | 段落近似 | 查询前 | 将段落全部嵌入替换为其所属质心的$\text{ID}$ | $\text{ID}$为紧凑整数 |
> > >      | 质心距离 | 查询时 | 预计算查询与所有质心的距离                  | 这么作源于质心固定    |
> > >
> > >    - 交互阶段：不对段落嵌入进行任何解压，转而用其近似表示，与查询交互得到近似得分
> > >
> > > 2. 质心剪枝机制：直接屏蔽与查询距离太远的质心，在本次查询之后的操作中不再涉及
> > >
> > > :two:$\text{PLAID}$引擎的实现
> > >
> > > 1. 功能集成：实现了可适配$\text{GPU/CPU}$的质心交互内核$+$质心剪枝设计，并集成到引擎当中
> > > 2. 内核模块：分别为$\text{ColBERTv2}$中的数据移动$/$解压$/$评分构建独立内核，并分别优化
> >
> > ### $\textbf{1.2.3. PLAID}$的评估
> >
> > > :one:评估操作
> > >
> > > 1. 数据集：域内$\text{(MS MARCOv1/MS MARCOv2)}$，域外$(\text{Wikipedia/LoTTE})$  
> > > 2. 超参数：搜索深度$k\text{=}10/100/1000$
> > > 3. 环境配置：多线程$\text{CPU/}$单线程$\text{CPU/GPU}$
> > >
> > > :two:评估结果：$\text{PLAID}$更具大规模低延迟检索的能力
> > >
> > > |   方面   | 性能相比$\textbf{ColBERTv2}$                                 |
> > > | :------: | ------------------------------------------------------------ |
> > > | 搜索延迟 | 在$\text{GPU}$上延迟减少$2.5–7$倍，$\text{CPU}$上则是$\text{9–45}$倍 |
> > > | 搜索质量 | 差不多，都保持极高水平                                       |
>
> ## $\textbf{1.3. }$有关工作
>
> > :one:$\text{IR}$系统的演变
> >
> > 1. 早期–交叉编码模型：直接将[查询$\xleftrightarrow{合并}$段落]为整体，再输入$\text{BERT}$处理，以得到相似性评分
> >
> > 2. 后期–独立表示模型：
> >
> >    |  模型形式  | 描述                                                         | 空间$/$性能 | 实例              |
> >    | :--------: | ------------------------------------------------------------ | :---------: | ----------------- |
> >    |  稀疏词权  | 查询$/$段落$\xrightarrow[表示]{词项权重}$稀疏向量$\text{→}$相关性 |  很小$/$弱  | $\text{BM25}$     |
> >    | 单向量模型 | 查询$/$段落$\xrightarrow[编码]{神经网络}$单稠密向量$\xrightarrow{互相点积}$相似度 |   小$/$弱   | $\text{RocketQA}$ |
> >    | 多向量模型 | 查询$/$段落$\xrightarrow[编码]{神经网络}$多稠密向量$\xrightarrow{细颗粒度交互}$相似度 |   大$/$强   | $\text{ColBERT}$  |
> >
> > 3. 当下–进一步优化：负样本训练$/$蒸馏训练$/$去噪监督.....
> >
> > :two:检索过程中的剪枝：快速跳过无关段落，只保留有潜力成为$\text{Top-}k$的候选段落进入下一步处理
> >
> > 1. 稀疏$\text{IR}$中的剪枝：
> >
> >    |     时机      | 操作                                                         |
> >    | :-----------: | ------------------------------------------------------------ |
> >    | 索引$/$嵌入时 | 在得段落嵌入的同时，会同时计算并保存每个段落的得分上限$\text{Meta-Data}$ |
> >    |    查询时     | 直接跳过得分上限小于阈值的段落                               |
> >
> >    :thinking:分析：该模式无法直接用于后期交互模型，缘于后期交互**不可能**在嵌入时得到得分上限
> >
> > 2. 密集$\text{IR}$中的剪枝：
> >
> >    - 单向量：为两向量的交互，故可直接用$\text{ANN(}$如$\text{HNSW)}$找到与查询最近的$\text{Top-}k$，再剪枝
> >    - 多向量：为两矩阵的交互(后期交互)，本文关注的即是<font color=red>可否将$\text{ANN}$方法扩展到两矩阵的交互$?$</font> 

# $\textbf{2. }$对$\textbf{ColBERTv2}$的进一步分析

> ## $\textbf{2.1. ColBERTv2}$架构
>
> > :one:后期交互模式：==$\text{PLAID}$全盘采纳==
> >
> > <img src="https://i-blog.csdnimg.cn/direct/3212d3ce98f14538a0663d517e2240ef.png" alt="image-20241129235249305" width=500 /> 
> >
> > 1. 编码系统：
> >
> >    | 阶段 | 操作                                                       | 备足                                        |
> >    | :--: | :--------------------------------------------------------- | :------------------------------------------ |
> >    | 离线 | 编码数据库中所有段落$d$为词级嵌入$D_{M\text{×}k^{\prime}}$ | $M$为段落词数，$k^{\prime}$为(压缩)嵌入维度 |
> >    | 在线 | 在给出查询$q$后再将其编码为词级嵌入$Q_{N\text{×}k}$        | $N$为查询词数，$k$为嵌入维度                |
> >
> > 2. 交互系统：$\displaystyle{}S_{q, d}=\sum_{i=1}^N \max _{j=1}^M Q_i \cdot D_j^T$ 
> >
> >    - $\text{MaxSim}$：让$d$每个嵌入和$q$所有嵌入计算相似度并取最大值，再经重排最终得到$N$个$\text{MaxSim}$ 
> >    - 最终得分：将所有$\text{MaxSim}$求和，即得到查询对一个段落的最终相似度
> >
> > :two:残差压缩策略：==$\text{PLAID}$全盘采纳==
> >
> > 1. 运行聚类：对所有段落的所有嵌入组成的空间执行聚类，每个嵌入$t$分配到其最近的质心$C_t$
> > 2. 残差编码：计算每个嵌入的$t$的残差$r\text{=}t\text{–}C_t$，并将其近似量化编码为$\tilde{r}\text{≈}t\text{–}C_t$ 
> > 3. 压缩表示：
> >    - 存储时：$t$的嵌入$\xrightarrow{编码为}$(离$t$最近质心$C_t$的索引)$+$(残差向量$r\text{=}t\text{–}C_t$的量化近似$\tilde{r}\text{≈}t\text{–}C_t$)
> >    - 检索时：近似地将$t$还原为$\tilde{t}\text{=}C_t\text{+}\tilde{r}$ 
> >
> > :three:查询策略：==$\text{PLAID}$改进的核心点==，本文称原有查询方案为$\text{Vanilla}$方案
> >
> > 1. 段落初排：
> >
> >    |   过程   | 描述                                                         |
> >    | :------: | ------------------------------------------------------------ |
> >    | 查询编码 | 将查询$q$的原始文本，编码为嵌入向量集合                      |
> >    | 候选生成 | 查找与$q$的最邻近质心，再由质心==定位==到其所含所有的嵌入向量(即候选嵌入的==索引==) |
> >    | 索引查找 | 索引到并收集候选集中所有嵌入的残差压缩向量(质心$\text{ID}+$量化残差表示) |
> >    | 残差解压 | 将所有收集到的残差向量执行解压操作，得到(近似的)完整嵌入表示 |
> >    | 相似计算 | 计算$\text{MaxSim}$并加和为，作为段落的初步得分$/$排名       |
> >
> > 2. 段落重排：
> >
> >    - 段落选取：仅保留初排结果的前若干段落，解压其所有嵌入
> >    - 相似计算：对若干段落执行完整的$\text{MaxSim}$计算与加和，得到最终的最相似段落
>
> ## $\textbf{2.2. ColBERT}$查询过程的分析
>
> > ### $\textbf{2.2.1. }$查询延迟分解: [索引查找$+$候选生成]是瓶颈
> >
> > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241212155844854.png" alt="image-20241212155844854" width=400 /> 
> > >
> > > :one:索引查找：涉及对候选嵌入的压缩向量的传输，候选规模一般不小
> > >
> > > 1. 硬件占用：需占据大量内存带宽$+\text{CPU}$就绪等待
> > >
> > > 2. 动态填充：候选集中属于每个段落的嵌入数量可能不一，为方便批处理需要进行==批内填充对齐==
> > >
> > > :two:差解残压：即[索引得质心向量$+$还原残差每位$\text{→}$二者相加]，其本身就非常耗时(即使缩小候选集)
> >
> > ### $\textbf{2.2.2. }$改进思路: 初排可否不涉及残差$\textbf{?}$✅
> >
> > > :one:基本做法
> > >
> > > |        模型        | 候选段落的嵌入 | $\boldsymbol{\xRightarrow{操作}}$ | 初排时的段落嵌入   |
> > > | :----------------: | :------------: | :-------------------------------: | ------------------ |
> > > | $\text{ColBERTv2}$ |  残差压缩向量  | 索引到质心$+$还原残差$+$二者加和  | 压缩还原的完整嵌入 |
> > > |   $\text{PLAID}$   |  残差压缩向量  |            索引到质心             | 嵌入的质心代替之   |
> > >
> > > :two:初步评估：采纳此改进后的搜索延迟$/$搜索质量
> > >
> > > 1. 搜索延迟：毫无疑问大幅减少
> > >
> > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241212164845596.png" alt="image-20241212164845596" width=400 /> 
> > >
> > > 2. 搜索质量：与原有方案并无明显下滑
> > >
> > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241212165258595.png" alt="image-20241212165258595" width=350 /> 
> > >
> > >    :bulb:图的含义：$\text{Vanilla}$方案前$k\text{=100}$个段落，出现在$\text{PLAID}$方案(仅质心)前$k\text{=100}$个段落之比
> >
> > ### $\textbf{2.2.3. }$进一步的思考: 所有质心都对查询有用吗$\textbf{?}$❌
> >
> > > :one:质心假设：​对一个查询，仅少部分质心对计算段落相关性起作用，以至于其余质心可直接忽略
> > >
> > > :two:实验验证：
> > >
> > > 1. 实验设置：随机抽取$15$个$\text{MS MARCO v1}$查询，计算每个查询与所有质心的得分(示例如下)
> > >
> > >    ```txt
> > >    查询: Q -> 词元{q1,q2,q3,q4....}
> > >    质心: C -> 质心{c1,c2,c3,c4,c5,c6....}
> > >    ```
> > >
> > >    ```txt
> > >    Q↔c1得分: {q1↔c1, q2↔c1, q3↔c1, q4↔c1, .....}距离的最大值(最大质心得分)
> > >    Q↔c2得分: {q1↔c2, q2↔c2, q3↔c2, q4↔c2, .....}距离的最大值(最大质心得分)
> > >    Q↔c3得分: {q1↔c3, q2↔c3, q3↔c3, q4↔c3, .....}距离的最大值(最大质心得分)
> > >    .......
> > >    ```
> > >
> > > 2. 实验结果：大多质心对与查询的相关性极低，以得分$0.2$为界就能刷掉近$90\%$质心
> > >
> > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241212180029243.png" alt="image-20241212180029243" width=350 />   

# $\textbf{3. PLAID}$的流程与实现

> <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241212180339399.png" alt="image-20241212180339399" width=720 /> 
>
> | 阶段概览 | 目的                                                         |
> | :------: | ------------------------------------------------------------ |
> | 候选生成 | 通过质心计算初步筛选潜在相关段落，生成初始候选集             |
> | 质心剪枝 | 在候选生成和质心交互之前，剔除与查询相关性较低的质心         |
> | 质心交互 | 使用质心代替完整嵌入进行轻量化相似度计算，快速筛选出高相关性段落 |
> | 评分重排 | 通过残差解压缩重构最终候选段落的完整嵌入，计算$\text{MaxSim}$与得分，随后重排 |
>
> ## $\textbf{3.1. }$候选生成
>
> > ### $\textbf{3.1.1.  }$候选生成的过程
> >
> > > |   步骤   | 描述                                                         |
> > > | :------: | ------------------------------------------------------------ |
> > > | 查询输入 | 查询矩阵$Q$(所有$\text{Token}$的嵌入向量)$+$质心列表矩阵$C$(所有质心的向量) |
> > > | 得分计算 | 直接计算$S_{c,q}\text{=}C\text{⸱}Q^{T}$，其中$S_{c,q}[i][j]$是质心$c_i$与查询词元$q_j$的相关性得分 |
> > > | 质心候选 | 对每个$q_j$选取其排名从高到低前$t_{\text{nprobe}}$个质心$\text{→}$合并所有$q$的质心选集为最终选集$C^{\prime}$ |
> > > | 段落候选 | 若一个段落中==存在$q$==被聚类到(属于)$C^{\prime}$，则将该段落候选之 |
> > >
> > > :thinking:==黄标==部分：只能是<font color=gree>存在$q$</font>而不能是<font color=red>超过多少个$q$</font>，因为$\text{PLAID}$的倒排索引只能判断存在而非数量
> >
> > ### $\textbf{3.1.2. }$与$\textbf{ColBERTv2}$的对比
> >
> > > :one:倒排索引
> > >
> > > 1. 索引方式：$\text{ColBERTv2}$直接由质心映射到嵌入，而$\text{PLAID}$则映射到与嵌入有关段落
> > >
> > >    ```txt
> > >    ColBERTv2做法:
> > >    c1 -> {Doc1-Token1, Doc3-Token2, Doc3-Token3}
> > >    c2 -> {Doc2-Token1, Doc2-Token2}
> > >    c3 -> {Doc1-Token2, Doc3-Token4, Doc1-Token4, Doc2-Token3}
> > >    c4 -> {Doc1-Token3, Doc2-Token4, Doc3-Token1}
> > >    ```
> > >
> > >    ```txt
> > >    PLAID做法:
> > >    c1 -> {Doc1, Doc3}
> > >    c2 -> {Doc2}
> > >    c3 -> {Doc1, Doc2, Doc3}
> > >    c4 -> {Doc1, Doc2, Doc3}
> > >    ```
> > >
> > > 2. 索引使用：
> > >
> > >    |        模型        | 候选生成流程                                                 |
> > >    | :----------------: | ------------------------------------------------------------ |
> > >    | $\text{ColBERTv2}$ | 输入查询$+$质心$\text{→}$候选质心$\xrightarrow{质心\text{→}嵌入的倒排索引}$候选嵌入(再对应到候选段落) |
> > >    |   $\text{PLAID}$   | 输入查询$+$质心$\text{→}$候选质心$\xrightarrow{质心\text{→}段落的倒排索引}$候选段落 |
> > >
> > > 3. 索引实现：由于段落数$\text{≪}$嵌入向量数，故改进后节省了大幅空间($\text{71GB}\xrightarrow{2.7×}\text{27GB}$)
> > >
> > > :two:候选集剪枝
> > >
> > > 1. $\text{ColBERTv2}$：会对候选集剪枝<font color=red>($\text{ColBERTv2}$原文也没说，难道是隐式的$?$)</font> 
> > > 2. $\text{PLAID}$：对候选集大小不做任何限制，**而是在后续进行更为链接高效的质心剪枝** 
>
> ## $\textbf{3.2. }$质心交互$\textbf{\&}$剪枝 
>
> > :one:目的：使用基于质心的近似距离快速筛掉评分低的段落
> >
> > $\quad$:thinking:==类似于很多模型中$\text{BM25}$的作用==，质心相关性得分$\xleftrightarrow{相当于}\text{BM25}$的词项相关性得分
> >
> > :two:流程
> >
> > 1. 倒排索引：对于某一段落$D$，由[质心$\xrightarrow{索引}$文档]的索引，得到与其有关的质心(==注意可能不止一个==)
> >
> >    ```txt
> >    查询Q -> {q1, q2, q3, q4}
> >    段落D -> {c1, c2, c3, c4, c5, c6, c7, c8,.......}
> >    ```
> >
> > 2. 质心剪枝：计算每个质心对于查询的分数(规则如下示例)，筛掉小于阈值$t_{c s}$的质心
> >
> >    ```txt
> >    Q↔c1得分: {q1↔c1, q2↔c1, q3↔c1, q4↔c1}距离的最大值(最大质心得分)
> >    Q↔c2得分: {q1↔c2, q2↔c2, q3↔c2, q4↔c2}距离的最大值(最大质心得分)
> >    Q↔c3得分: {q1↔c3, q2↔c3, q3↔c3, q4↔c3}距离的最大值(最大质心得分)
> >    .......
> >    筛掉对于查询评分过低的质心
> >    
> >    段落D -> {c1, c2, c3}(剪枝后)
> >    ```
> >
> > 3. 质心交互：完全类似于后期交互，并且具体实现上也==与$\text{MaxSim}$共享一套无填充内核优化==(见后)
> >
> >    ```txt
> >    计算q1↔{c1, c2, c3}距离, 取最大值作为MaxSim1
> >    计算q2↔{c1, c2, c3}距离, 取最大值作为MaxSim2
> >    计算q3↔{c1, c2, c3}距离, 取最大值作为MaxSim3
> >    计算q4↔{c1, c2, c3}距离, 取最大值作为MaxSim4
> >    
> >    最终q↔d的近似相似度: MaxSim1+MaxSim2+MaxSim3+MaxSim4
> >    ```
> >
> > 4. 排序筛选：对所有段落执行此操作$\text{→}$依据得到的近似相似度排序$\text{→}$以供截取前若干个以筛选
>
> ## $\textbf{3.3. }$评分(重排): 快速内核的实现
>
> > ### $\textbf{3.3.1. }$无填充的$\textbf{MaxSim}$计算内核
> >
> > > :one:传统$\text{ColBERTv1/v2}$的$\text{MaxSim}$实现
> > >
> > > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241130043629711.png" alt="image-20241130035316438" width=500 /> 
> > >
> > > 1. 预处理：
> > >    - 填充：对$\text{GPU/CPU}$一批内的文档，都填充至与批内最长文档对齐
> > >    - 构建：将所有批内所有的文档嵌入集(二维张量)堆叠，构成三维张量后再移入$\text{RAM/}$显存
> > > 2. 改进思路：避免填充这一耗时操作，就要放弃三维对齐，故直接在无填充二维张量上$\text{MaxSim}$ 
> > >
> > > :two:无填充$\text{MaxSim}$的[$\text{CPU}$内核](https://github.com/stanford-futuredata/ColBERT/blob/7067ef598b5011edaa1f4a731a2c269dbac864e4/colbert/modeling/segmented_maxsim.cpp)(暂无$\text{GPU}$版本)
> > >
> > > 1. 数据结构：直接将二维的段落嵌入张量作为输入
> > >
> > >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241212223511146.png" alt="image-20241212221655326" width=400 /> 
> > >
> > > 2. 计算流程：(内核的核心代码)
> > >
> > >    ```C++
> > >    // 遍历所有2D文档张量
> > >    for (int doc = 0; doc < ndocs; doc++) 
> > >    {
> > >        float max_score[nquery_vectors] = {0}; 
> > >        int offset = offsets[doc];        
> > >        // 遍历当前文档的所有1D嵌入向量
> > >        for (int vec = 0; vec < lengths[doc]; vec++) 
> > >        {
> > >            // 遍历当前文档的嵌入时, 试图更新其得分最大值(就是MaxSim)
> > >            for (int q = 0; q < nquery_vectors; q++) 
> > >            {
> > >                max_score[q] = max(max_score[q], score[offset + vec][q]);
> > >            }
> > >        }
> > >        // 对每个MaxSim进行累加, 以得到段落的最终得分
> > >        float total_score = accumulate(max_scores, max_scores + nquery);
> > >        result[doc] = total_score; 
> > >    }
> > >    ```
> > >
> > > 3. 流程分析：
> > >
> > >    - 并行计算：每个段落的得分都是独立更新并得到的，有助于实现段落间的并行计算
> > >    - 内存优化：空间复杂度从$O(\text{BatchSize×}N)$降到了$O(N)$
> >
> > ### $\textbf{3.3.2. }$优化的解压缩内核
> >
> > > :one:$\text{ColBERTv2}$对于解压的传统实现
> > >
> > > 1. 方法：从量化残差($\text{bit}$流)中按位提取$+$偏移$\text{→}$与质心向量相加，如下为$b\text{=}2$的例子
> > >
> > >    ```txt
> > >    压缩残差: 11001001 10100111 .....
> > >            👆
> > >    索引为11(残差第1位为+Δ), 与质心相加后为(c1+Δ,c2,c3,c4,...)
> > >    
> > >    压缩残差: 11001001 10100111 ..... 
> > >              👆
> > >      索引为00(残差第2位为-Δ), 与质心相加后为(c1+Δ,c2-Δ,c3,c4,...)
> > >    
> > >    压缩残差: 11001001 10100111 ..... 
> > >                👆
> > >        索引为10(残差第3位为+Δ/2), 与质心相加后为(c1+Δ,c2-Δ,c3+Δ/2,c4,...)
> > >         
> > >    压缩残差: 11001001 10100111 ..... 
> > >                  👆
> > >          索引为01(残差第4位为-Δ/2), 与质心相加后为(c1+Δ,c2-Δ,c3+Δ/2,c4-Δ/2,...)
> > >    ```
> > >
> > > 2. 分析：为何按位$+$偏移操作会很耗时$?$
> > >
> > >    - 按位提取本身就很费时：
> > >
> > >      ```txt
> > >      提取第一个2位: 0b11 & [压缩值] & 
> > >      提取第一个2位: 0b11 & ([压缩值] >> 2)
> > >      ```
> > >
> > >    - 质心与残差的相加为逐维推进(串行)，当嵌入维度过高时会很耗时
> > >
> > > :two:$\text{PLAID}$的改进与实现
> > >
> > > 1. 改进描述：预先计算并存储所有$2^b$种可能的索引值(如下以$b\text{=2}$为例)，解压时直接查表无需再算
> > >
> > >    | 残差编码 | $\textbf{00}$ | $\textbf{01}$ | $\textbf{10}$ | $\textbf{11}$ |
> > >    | :------: | :-----------: | :-----------: | :-----------: | :-----------: |
> > >    |  残差值  |   $-\Delta$   |  $-\Delta/2$  |  $\Delta/2$   |   $\Delta$    |
> > >
> > > 2. 内核实现：
> > >
> > >    - $\text{GPU}$：定义的$\text{CUDA}$，为每个$\text{Byte}$的残差编码分配一个大小为$\cfrac{b\text{⸱}嵌入维度}{8}$的线程块
> > >    - $\text{CPU}$：以单个段落为单位进行解压

# $\textbf{4. }$评估

> ## $\textbf{4.1. }$实验的设置
>
> > :one:实现：直接在原有$\text{ColBERTv2}$上改进，加入少量$\text{C}$++代码($\text{CPU}$内核)$/$和$\text{CUDA}$代码($\text{GPU}$内核)
> >
> > :two:数据集
> >
> > 1. 域外：$\text{MS MARCO v1/Wikipedia Open QA}$
> > 2. 域内：$\text{StackExchange/LoTTE/MS MARCO v2}$
> >
> > :three:评估模型：$\text{ColBERTv2/PLAID/ColBERT/BM25/SPLADEv2/DPR}$ 
> >
> > :four:超参数：与$\text{ColBERTv2}$共享的保持一致，令$\text{PLAID}$的新增的重排文档数$k\text{=10/100/1000}$ 
> >
> > :five:硬件配置
> >
> > 1. $\text{CPU}$：$\text{28}$个$\text{Intel Xeon Gold 6132 2.6 GHz}$
> > 2. $\text{CPU}$：$4$个$\text{NVIDIA TITAN V}$ 
> > 3. 内存：$\text{72 GBps/33 GBps}$带宽$\text{92 ns/142 ns}$延迟，$\text{NUMA}$加`numactl --membind 0`确保$\text{I/O}$ ​
>
> ## $\textbf{4.2. }$实验结果
>
> > :one:端到端结果
> >
> > 1. 域内结果：在$\text{MS MARCO v1/Wikipedia OpenQA}$上
> >
> >    | 质量($\text{MRR/Recall}$) | $\text{CPU}$提速 | $\text{GPU}$提速 |
> >    | :-----------------------: | :--------------: | :--------------: |
> >    |  不损失($k\text{=1000}$)  |    $20–40$倍     |     $4–6$倍      |
> >    |         略微下降          |    $50–150$倍    |    $10–20$倍     |
> >
> > 2. 域外结果：在$\text{LoTTE/MS MARCO v2}$上
> >
> >    | 质量($\text{MRR/Recall}$) | $\text{CPU}$提速 | $\text{GPU}$提速 |
> >    | :-----------------------: | :--------------: | :--------------: |
> >    |  不损失($k\text{=1000}$)  |    $10–20$倍     |     $3–4$倍      |
> >    |         略微下降          |    $20–40$倍     |     $3–7$倍      |
> >
> > :two:消融分析：
> >
> > | <font color=red>质心交互</font> | <font color=red>质心剪枝</font> | <font color=red>优化内核</font> | $\textbf{GPU}$加速 | $\textbf{CPU}$加速 |
> > | :-----------------------------: | :-----------------------------: | :-----------------------------: | :----------------: | :----------------: |
> > |                ✅                |                ✅                |                ✅                |    $\text{N/A}$    |   $\text{42.4}$    |
> > |                ✅                |                ✅                |                ❌                |    $\text{6.7}$    |   $\text{42.1}$    |
> > |                ✅                |                ❌                |                ❌                |    $\text{5.2}$    |    $\text{8.6}$    |
> > |                ❌                |                ❌                |                ✅                |    $\text{N/A}$    |     $\text{3}$     |
> > |                ❌                |                ❌                |                ❌                |     $\text{1}$     |     $\text{1}$     |
> >
> > :three:可扩展性：
> >
> > $\quad$😕并没有实现并行$/$数据规模上的可扩展性，或许源于为解决负载不平衡的问题