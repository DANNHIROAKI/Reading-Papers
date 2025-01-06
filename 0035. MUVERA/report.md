# 基于多向量的文本相似性检索

## $\textbf{1. }$从词袋到$\textbf{ColBERT}$: 文本相似性计算方法概览

> ### $\textbf{1.1. }$如何计算两个文本的相似性$\textbf{?}$ 
>
> > :one:特征方法：从文本中提取人工设计的特征$\text{→}$通过特征来计算文本相似度，也可叫稀疏方法$/$词频方法
> >
> > |        方式         | 描述                                                         |
> > | :-----------------: | ------------------------------------------------------------ |
> > |      基于词频       | 认为[查询$\xleftrightarrow{}$段落]共同词多$\text{+}$共同词的词频($\text{TF}$)分布类似，表明相似度高 |
> > | 基于$\text{TF-IDF}$ | 在词频的基础上，引入$\text{IDF}$以惩罚常见词$/$奖励关键词    |
> > |    $\text{BM25}$    | $\text{TF-IDF}$的改进版，平滑化词频$\text{+}$考虑文档长度，<font color=red>**最为经典的特征方法没有之一**</font> |
> >
> > :two:神经方法：将文本转化为稠密的嵌入$\text{→}$通过嵌入来计算文本相似度
> >
> > |            方法            |                            示意图                            | 描述                                                         |            查询质量            |            查询延迟            |
> > | :------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ | :----------------------------: | :----------------------------: |
> > | 段落级<br />**(大颗粒度)** | <img src="https://i-blog.csdnimg.cn/direct/353bea1c5ccf4308a0ba258099b54456.png" alt="image-20241129205331629" width=170 /> | ①查询$/$段落各自生成单嵌入<br />②相似度$\text{←}$两向量的内积 | <font color=red>**低**</font>  | <font color=gree>**小**</font> |
> > | 词元级<br />**(细颗粒度)** | <img src="https://i-blog.csdnimg.cn/direct/f0eac169821246578e432747a212b39f.png" alt="image-20241129212017809" width=170 /> | ①拼接将查询和段落<br />②输入$\text{BERT}$得到相似度          | <font color=gree>**高**</font> | <font color=red>**大**</font>  |
> >
> > $\quad$:thinking:是否有方法，能在保留$\text{Token}$级的细颗粒度交互同时，降低计算量$?$
> >
> > $\quad$:bulb:**查询前(离线)**就算好所有**段落**的词级嵌入$\text{→}$**查询时(在线)**即时得到**查询**的词级嵌入$\text{→}$交互得相似度
>
> ### $\textbf{1.2. ColBERT}$概述 
>
> > :one:组成结构上 
> >
> > <img src="https://i-blog.csdnimg.cn/direct/3212d3ce98f14538a0663d517e2240ef.png" alt="image-20241129235249305" width=540 /> 
> >
> > 1. 编码模块：
> >    - 对段落：原始段落$\xrightarrow[\text{CPU}]{\text{WordPiece}}$段落$\text{Token}\xrightarrow[\text{GPU}]{\text{BERT}(段落编码器f_P)}$每个$\text{Token}$都生成一个嵌入向量
> >    - 对查询：原始查询$\xrightarrow[\text{CPU}]{\text{WordPiece}}$查询$\text{Token}\xrightarrow[\text{GPU}]{\text{BERT}(查询编码器f_Q)}$每个$\text{Token}$都生成一个嵌入向量
> > 2. 交互模块：后期交互$\displaystyle{}S_{q, d}\text{:=}\sum_{i \in\left[\left|E_q\right|\right]} \max _{j \in\left[\left|E_d\right|\right]} E_{q_i} \cdot E_{d_j}^T$ 
> >    - 第一步：计算每个$v \text{∈} E_q$与$E_d$中向量的最大相似度($\text{MaxSim}$)
> >    - 第二步：将所有$\text{MaxSim}$相加，即为最终的相似度
> >
> > :two:使用流程上​
> >
> > 1. 离线阶段：对数据库中所有段落进行操作，生成全部其嵌入
> > 2. 在线阶段：即时地生成查询的嵌入$\text{→}$与段落嵌入后期交互得到相似度$\text{→}$结合$\text{IVF}$返回$\text{Top-}k/$重排
> >
> > :three:性能上：在于原始$\text{BERT}$性能差不多的情况下大大降低了延迟
> >
> > <img src="https://i-blog.csdnimg.cn/direct/359eb79a3b1b4adab35defe026010771.png" alt="image-20250106034946943" width=530 /> 

## $\textbf{2. }$从$\textbf{ColBERT}$到$\textbf{PLAID}$: 对原始模型的改进

> ### $\textbf{2.1. ColBERTv2}$: 针对训练$\textbf{\&}$存储的优化
>
> > :one:改进动机
> >
> > 1. 基础：全盘采纳$\text{ColBERT}$的结构，不作任何改变
> > 2. 动机：训练策略高度优化的单向量模型反而优于$\text{ColBERT}$，对原始嵌入整个存储开销太大
> >
> > :two:改进思路：
> >
> > 1. 模型训练：借鉴很多单向量模型的<font color=red>**高度监督调优**</font>，即蒸馏优化$\text{+}$降噪训练
> >
> >    - 蒸馏优化：将原有大型$\text{BERT}$的参数蒸馏到小型$\text{MiniLM}$中，用来给段落评分以获取负样本
> >    - 降噪训练：借助上一步，在训练过程中引入更多的负样本
> >
> > 2. 存储优化：对离线阶段所得的嵌入进行<font color=red>**残差压缩**</font>，以减小离线存储需求$\&$在线查询时的总线带宽
> >
> >    |    模型     | 离线嵌入阶段                                                | 在线查询阶段               |
> >    | :---------: | ----------------------------------------------------------- | -------------------------- |
> >    | $\text{v1}$ | 计算段落嵌入$\text{→}$存储完整嵌入                          | 直接计算                   |
> >    | $\text{v2}$ | 计算段落嵌入$\text{+}$质心$/$残差压缩$\text{→}$存储压缩嵌入 | 先要解压查询附近的段落嵌入 |
>
> ### $\textbf{2.2. PLAID}$: 针对后期交互机制的优化
>
> > :one:改进动机
> >
> > 1. 基础：全盘采纳$\text{ColBERT}$的结构，对$\text{ColBERTv2}$的降噪监督$\text{+}$残差压缩也予以保留
> > 2. 动机：基于对$\text{ColBERTv2}$进一步分析所得的两个结论
> >    - [查询嵌入$\xleftrightarrow{距离}$段落所属的质心向量]$\text{≈}$[查询$\xleftrightarrow{距离}$段落嵌入]$\text{→}$以该近似距离进行初排可避免解压
> >    - 对一个查询只有少部分质心起作用$\text{→}$可进行大刀阔斧的质心剪枝
> >
> > :two:改进思路
> >
> > 1. 段落初排：使用质心代替的阉割版
> >    - 候选生成：计算与查询邻近的质心集
> >    - 质心剪枝：计算[查询$\xleftrightarrow{}$质心]的近似距离，筛掉评分低的质心
> >    - 质心交互：和后期交互一致，只不过参与交互的段落嵌入集，变成了段落嵌入所属质心的集合
> > 2. 段落重排：使用嵌入解压的完整版

## $\textbf{3. }$后$\textbf{PLAID}$: 对多向量检索优化的探索

> ### $\textbf{3.1. EMVB: }$沿原有思路对$\textbf{PLAID}$的进一步改进
>
> > :one:模型方法上
> >
> > 1. 质心交互：抛弃原有的残差压缩改为$\text{PQ}$压缩
> > 2. 质心剪枝：对已生成的候选集进一步"预过滤"，减少参与排序的质心数目
> >
> > :two:工程实现上：按位运算快速判断质心相似度，垂直堆叠压缩存储，$\text{SIMD}$加速向量计算......
>
> ### $\textbf{3.2. BGE-M3: }$从嵌入方式上下手
>
> > :one:三种检索任务：密集$\mathcal{/}$多向量$\mathcal{/}$稀疏(词汇)
> >
> > |    模型    | 嵌入方式                     | 相似度计算(得分)                                   |     模型实例     |
> > | :--------: | ---------------------------- | -------------------------------------------------- | :--------------: |
> > |  密集检索  | 编码段落为单个段落级稠密向量 | $s_{\text {dense}} \leftarrow$两个向量间的点积计算 |  $\text{BERT}$   |
> > | 多向量检索 | 编码段落为多个词汇级稠密向量 | $s_{\text {mul}} \leftarrow$两组向量间的复杂交互   | $\text{ColBERT}$ |
> > |  稀疏检索  | 计算段落中词的词项权重       | $s_{\text {lex}} \leftarrow$词匹配得分             |  $\text{BM25}$   |
> >
> > :two:$\text{BGE-M3}$：实现对三种嵌入方式的通用性
> >
> > 1. 训练：三种检索任务的优化方向可能存在冲突$\text{→}$通过自蒸馏$\text{+}$集成学习整合进统一损失函数
> >
> > 2. 部署：针对不同任务使用不同的检索$/$重排策略
> >
> >    - 单独的：
> >
> >      |    检索方式     | 描述                                                         |
> >      | :-------------: | :----------------------------------------------------------- |
> >      | $\text{Dense}$  | 生成语料库的稠密嵌入，建立$\text{Faiss}$索引以进行$\text{Top-1000}$检索 |
> >      | $\text{Sparse}$ | 生成语料库的稀疏表示，建立$\text{Lucene}$索引以进行$\text{Top-1000}$检索 |
> >
> >    - 整合的：
> >
> >      |       检索方式        |             检索内容             | 重排依据                                                     |
> >      | :-------------------: | :------------------------------: | ------------------------------------------------------------ |
> >      | $\text{Dense+Sparse}$ |    并合各自$\text{Top-1000}$     | $w_1s_{\text {dense}}\text{+}w_2s_{\text {lex}}$             |
> >      |   $\text{MultiVec}$   | $\text{Dense}$的$\text{Top-200}$ | $s_{\text {mul}}$                                            |
> >      |     $\text{All}$      | $\text{Dense}$的$\text{Top-200}$ | $w_1s_{\text {dense}}\text{+}w_2  s_{\text {lex}}\text{+}w_3s_{\text {mul}}$ |
>
> ### $\textbf{3.3. MuVERA: }$将多向量压缩为固定维度的单向量
>
> > :one:核心思想
> >
> > 1. 维度压缩：设计特殊的映射函数$\begin{cases}\mathbf{F}_{\mathrm{que}}: 2^{\mathbb{R}^{d}} \rightarrow \mathbb{R}^{d_{\mathrm{FDE}}}\\\\\mathbf{F}_{\text{doc}}: 2^{\mathbb{R}^{d}} \rightarrow \mathbb{R}^{d_{\mathrm{FDE}}}\end{cases}\text{→}$将多向量压缩为固定$d_{\text{FDE}}$维的单向量 
> > 2. 相似度计算：直接用内积$\left\langle\mathbf{F}_{\mathrm{que}}(Q), \mathbf{F}_{\text{doc}}(P)\right\rangle$作为原有$\displaystyle{}\sum_{q \in Q} \max _{p \in P}\langle q, p\rangle$(即$\text{MaxSim}$)的替代
> >
> > :two:工作流程：
> >
> > 1. 预处理：对所有文档进行$\mathbf{F}_{\text{doc}}$映射得到$\mathbf{F}_{\text{doc}}(P_i)$
> > 2. 查询时：对给定查询$Q$进行$\mathbf{F}_{\text{que}}$映射得到$\mathbf{F}_{\text{que}}(Q)$，计算$\left\langle\mathbf{F}_{\mathrm{que}}(Q), \mathbf{F}_{\text{doc}}(P_i)\right\rangle$得到$\text{Top-}k$文档
> >    - 最后再用完整的$\displaystyle{}\sum_{q \in Q} \max _{p \in P}\langle q, p\rangle$相似度，对$\text{Top-}k$个文档进行重排