# $\textbf{1. }$导论与综述

> ## $\textbf{1.1. }$研究的背景
>
> > :one:$\text{IR}$(信息检索)：相似性检索方法
> >
> > 1. 特征方法：基于人工设计的特征(词频$/\text{TF-IDF}$)来计算文档相似性
> >
> > 2. 神经方法：
> >
> >    | 方法 | 含义                                                      | 模型实例           |
> >    | :--: | --------------------------------------------------------- | ------------------ |
> >    | 嵌入 | 将文档嵌入为词向量集，在**词$/$句级**细颗粒度上计算相似性 | $\text{DRMM/KNRM}$ |
> >    | 微调 | 在上游预训练模型，在下游微调以适应相似性检索任务          | $\text{BERT/ELMo}$ |
> >
> > :two:不同方法的评估：检索效果($\text{MRR@10}$)$\xleftrightarrow{}$延迟
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241129195020633.png" alt="image-20241129195020633" width=530 /> 
> >
> > 1. 关于$\text{MS MARCO Ranking}$：
> >
> >    - 是啥：包含$\text{900}$万段落与$\text{100}$万查询，源于$\text{Bing}$查询日志
> >
> >    - 任务：文档检索(从数据库中返回最相关的文档)$+$问答任务
> >
> >    - 指标：$\text{MRR@10}$，即前$10$个返回的文档中给出正确答案的排名的平均倒数，示例如下
> >
> >      ```txt
> >      查询1 应返回A 实际返回[A,D,E,F,G,H,I,J,K,L] 倒数排名为1/1
> >      查询2 应返回B 实际返回[D,B,E,F,G,H,I,J,K,L] 倒数排名为1/2
> >      查询3 应返回C 实际返回[D,E,F,G,C,H,I,J,K,L] 倒数排名为1/5 ⇒ MRR@10=0.567
> >      ```
> >
> > 2. 评估的结论：$\text{BERT}$显著提高了搜索精度，但计算成本也急剧增加(而$\text{ColBERT}$解决了这一问题)
> >
> > :three:现行主要的神经排序范式
> >
> > |    方式    |                            示意图                            | 原理                                                         |
> > | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------- |
> > |  基于表示  | <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241129205331629.png" alt="image-20241129205331629" width=200 /> | ①分别将$q/d$映射到==单一(文档级)==嵌入<br />②以嵌入向量的相似度作为$q/d$相似度 |
> > |  基于交互  | <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241129205923992.png" alt="image-20241129205923992" width=200 /> | ①分别将$q/d$映射到==多个(短语级)==嵌入<br />②计算每对嵌入的相似度$\text{→}$得到交互矩阵<br />③输入交互矩阵到$\text{CNN/MLP}$以得到最终相似度 |
> > | 注意力交互 | <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241129212017809.png" alt="image-20241129212017809" width=200 /> | ①双向深度$\text{Transformer}$架构($\text{BERT}$)<br />②同时关注==${qd}$互相之间==$\&$==${}qd$各自内部==词级关系<br />③下游进行微调以适应相似性计算任务 |
>
> ## $\textbf{1.2. ColBERT}$的提出
>
> > :one:提出的动机
> >
> > 1. 两种神经排序范式的特点：
> >
> >    |   范式   | 特点(优势)                                                   |
> >    | :------: | ------------------------------------------------------------ |
> >    | 基于表示 | 可以在离线情况下预先计算文档的表示$\text{→}$大大降低每个查询的计算成本 |
> >    | 基于交互 | 具有细颗粒度(词级)匹配能力$\text{→}$在检索任务中往往性能更好 |
> >
> > 2. $\text{ColBERT}$的基本思想：通过推后[文档$\xleftrightarrow{交互}$查询]，来结合两种范式的优势(离线计算$+$细颗粒度)
> >
> > :two:$\text{ColBERT}$概述
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241129215149324.png" alt="image-20241129215149324" width=300 /> 
> >
> > 1. 文档编码：在查询开始前，事先通过离线方式完成
> > 2. 后期交互：执行[每个查询的词级嵌入$\xleftrightarrow{交互}$每个文档的词级嵌入]
> > 3. 相似度：求得每个查询词级嵌入的$\text{MaxSim}\text{→}$求和所有查询词级嵌入的$\text{MaxSim}$$\text{→}q/d$最终相似度
>
> ## $\textbf{1.3. }$有关工作的综述
>
> > :one:基于神经的匹配模型
> >
> > |            模型            | 概述                                                    | 类型 |
> > | :------------------------: | :------------------------------------------------------ | :--: |
> > |       $\text{KNRM}$        | 提出了可微分的核池化技术，从交互矩阵中提取匹配信号      | 交互 |
> > |       $\text{Duet}$        | 结合了基于精确匹配和嵌入的相似性信号进行排序            | 交互 |
> > |     $\text{ConvKNRM}$      | 能够学习查询和文档中$\text{N-gram}$的匹配，提升排序效果 | 交互 |
> > | $\text{fastText+ConvKNRM}$ | 采用子词嵌入，解决常见词嵌入列表中稀有词缺失的问题      | 交互 |
> > |       $\text{SNRM}$        | 用稀疏高维潜在词$+$倒排索引，加快检索速度               | 表示 |
> >
> > :two:用于信息检索($\text{IR}$)的预训练模型
> >
> > |       模型       | 概述                                                         |
> > | :--------------: | ------------------------------------------------------------ |
> > |  $\text{BERT}$   | 双向深度$\text{Transformer}$，通过微调以适应不同任务         |
> > |  $\text{ELMo}$   | 对词级的上下文进行建模，提升下游任务性能。                   |
> > | $\text{duoBERT}$ | 一种微调的$\text{BERT}$模型，提升了性能的同时也增加了计算成本 |
> >
> > :three:对$\text{BERT}$的优化
> >
> > | 优化方法 | 概述                                                         |
> > | :------: | ------------------------------------------------------------ |
> > |   蒸馏   | 迁移$\text{BERT}$模型的知识到小模型中，从而提升推理效率$+$减少计算成本 |
> > |   压缩   | 减小$\text{BERT}$模型的大小，同时尽可能保持性能              |
> > |   剪枝   | 去除$\text{BERT}$中不重要的网络连接，减少计算量              |
> >
> > :four:对$\text{NLU}$计算的离线执行方法
> >
> > |            模型             | 概述                                                         |
> > | :-------------------------: | ------------------------------------------------------------ |
> > |     $\text{doc2query}$      | 使用$\text{seq2seq}$模型将文档转换为查询$\text{→}$用$\text{BM25}$检索扩展后的文档 |
> > |       $\text{DeepCT}$       | 使用$\text{BERT}$为$\text{BM25}$的词频组件提供上下文感知的支持 |
> > | $\text{Transformer-Kernel}$ | 使用$\text{Transformer}$对查询和文档进行上下文编码，改进池化技术 |
> >
> > - 关于$\text{BM25}$：$\text{IR}$中的一种排名函数，分数越高代表[文档$\xleftrightarrow{相似度}$查询]越高

# $\textbf{2. ColBERT}$总论 

> ## $\textbf{2.1. ColBERT}$的结构单元
>
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241129235249305.png" alt="image-20241129235249305" width=540 /> 
> >
> > :thinking:$\text{→}\begin{cases}
> > 查询\xrightarrow[\text{BERT}]{查询编码器f_Q}\textcolor{red}{固定大小}的\textcolor{red}{上下文}嵌入集合E_q\\\\
> > 文档\xrightarrow[\text{BERT}]{文档编码器f_D}\textcolor{red}{固定大小}的\textcolor{red}{上下文}嵌入集合E_d
> > \end{cases}\text{→}\displaystyle{}S_{q, d}\text{:=}\sum_{i \in\left[\left|E_q\right|\right]} \max _{j \in\left[\left|E_d\right|\right]} E_{q_i} \cdot E_{d_j}^T$ 
> >
> > ### $\textbf{2.1.1. }$查询编码结构
> >
> > > :one:分词操作：查询$q\xrightarrow{分词}(\text{[Q]}q_0q_1...q_l\#\#...\#)$
> > >
> > > 1. 初始分词：查询文本$q\xrightarrow{\text{WordPiece}}$词元($\text{Token}$)集$\{q_1,q_2,...,q_l\}$ 
> > >
> > > 2. 查询标志：在`<cls>`$\text{Token}$后添加`<Q>`表明此序列为查询，即`<cls><Q><q1><q2><q3>...`
> > >
> > > 3. 查询增强：将$\text{Token}$序列控制在定长$N_q$
> > >
> > >    - 增强方法：两种情况
> > >
> > >      |        情况         | 操作                                                         |
> > >      | :-----------------: | ------------------------------------------------------------ |
> > >      | 查询序列长小于$N_q$ | 往$\text{Token}$序列末尾不断填充`<MASK>`(掩码$\text{Token}$)直到$N_q$长度 |
> > >      | 查询序列长大于$N_q$ | 在序列$N_q$长度处截断，即只保留前$N_q$个$\text{Token}$       |
> > >
> > >    - 增强意义：使$\text{BERT}$可在`<MASK>`处也生成上下文表示$\xrightarrow[使可微分]{软性扩展查询}$是$\text{ColBERT}$性能的关键
> > >
> > > :two:嵌入操作：查询$q\xrightarrow{分词}\xrightarrow{嵌入}\text{E}_q\text{:=Norm(CNN(BERT([Q]}q_0q_1...q_l\#\#...\#)))\textcolor{red}{\text{∈}\mathbb{R}^{N_d\text{×}m}}$ 
> > >
> > > 1. 初始嵌入：输入$\text{Token}$序列到$\text{BERT→}$得到每个$\text{Token}$的上下文嵌入向量
> > >
> > > 2. 维度控制：
> > >
> > >    - 操作方法：初始嵌入$\xrightarrow{没有激活函数的线性层}$固定维度($m$)的嵌入，其中$m\text{≪BERT}$的隐藏层维度
> > >
> > >    - 关于维度：对以下方面的影响
> > >
> > >      |   方面   | 影响                                                         |
> > >      | :------: | ------------------------------------------------------------ |
> > >      | 编码效率 | 几乎无影响，因为只是一个矩阵变换的功夫                       |
> > >      | 空间占用 | ==只管重要==，维度直接决定了空间占用的大小                   |
> > >      | 查询延时 | 影响显著，这是因为[$\text{CPU}\xleftrightarrow[总线]{嵌入向量}\text{CPU}$]是重排中最耗时的步骤 |
> > >
> > > 3. 归一化：使每个嵌入的模($L_2$范数)归一化为$1$，从向量点积直接就是$\text{Cosine}$相似度
> >
> > ### $\textbf{2.1.2. }$文档编码结构
> >
> > > :one:分词操作：文档$d\xrightarrow{分词}(\text{[D]}d_0d_1...d_n)$ 
> > >
> > > 1. 初始分词$/$查询标志与查询编码基本相同
> > > 2. 但不进行任何填充
> > >
> > > :two:嵌入操作：文档$d\xrightarrow{分词}\xrightarrow{嵌入}\text{E}_d\text{:=Filter(Norm(CNN(BERT([D]}d_0d_1...d_n))))\textcolor{red}{\text{∈}\mathbb{R}^{\text{}n\text{×}m}}$
> > >
> > > 1. 输入$\text{NERT}/$通过线性层控制维度为$m/$归一化，与查询编码基本相同
> > > 2. 多一个过滤操作，即滤掉==对应于标点符号==的嵌入
> >
> > ### $\textbf{2.1.3. }$后期交互结构
> >
> > > :one:整体思路：$\displaystyle{}S_{q, d}\text{:=}\sum_{i \in\left[\left|E_q\right|\right]} \max _{j \in\left[\left|E_d\right|\right]} E_{q_i} \cdot E_{d_j}^T$ 
> > >
> > > 1. $\text{MaxSim}$：计算每个$v \text{∈} E_q$与$E_d$中向量的最大相似度($\text{Consine}$距离$/$平方$L_2$距离)
> > > 2. 求和：将所有$\text{MaxSim}$的和作为最终的相似度
> > >
> > > :two:一些事项
> > >
> > > 1. 软性匹配：并不要求[查询词$\xleftrightarrow{完全相同}$文档词]，语义具有相似也会返回一定分数(相似度)
> > > 2. 关于求和：当然也可[$\textbf{MaxSim}\xrightarrow[得到]{\text{CNN/注意力}}$相似性打分]，但求和廉价$+$还能打$+$==可前$k$项剪枝== 
>
> ## $\textbf{2.2. ColBERT}$的训练方法
>
> > :one:$\text{ColBERT}$特性：端到端可微分，即可以反向传播以进行梯度下降
> >
> > :two:训练任务
> >
> > 1. 微调$\text{BERT}$
> > 2. 训练额外参数：包括`<P><Q>`的嵌入$\text{Token}/$线性层(交互机制中无参数)，使用$\text{Adam}$优化
> >
> > :three:训练过程
> >
> > 1. 训练参数：三元组$\left\langle q, d^{+}, d^{-}\right\rangle$，其中$q$是查询，$d^{+}$是正样本文档，$d^{-}$是负样本文档
> > 2. 损失函数：$\mathcal{L}\text{=}-\log \left(\cfrac{\exp \left(\text{score}\left(q, d^{+}\right)\right)}{\exp \left(\text{score}\left(q, d^{+}\right)\right)\text{+}\exp \left(\text{score}\left(q, d^{-}\right)\right)}\right)$并对其用交叉熵来优化
>
> ## $\textbf{2.3. ColBERT}$的运行机制
>
> > ### $\textbf{2.3.1. }$文档嵌入的离线预计算
> >
> > > :one:$\text{CPU}$阶段：运行$\text{WordPiece}$得到每个文档的$\text{Token}$集
> > >
> > > :two:$\text{GPU}$阶段：
> > >
> > > 1. 分批处理：将待嵌入的文档分批，一次输入一批到$\text{GPU}$(例如一次$\text{10w}$个文档)
> > >
> > > 2. 分组处理：对输入的一批文档执行以下步骤
> > >
> > >    | 操作 | 描述                                   | 备注                                                   |
> > >    | :--: | -------------------------------------- | ------------------------------------------------------ |
> > >    | 排序 | 将所有文$\text{10w}$个档按照其长度排序 | $\text{N/A}$                                           |
> > >    | 分组 | 将长度相近的文档分为一组               | 如$0\text{→}127$名为一组$/128\text{→}255$名为一组..... |
> > >    | 填充 | 使每个文档==与当前组中最长文档对齐==   | 使以组为单位在==$\text{GPU}$上并行处理==成为可能       |
> > >
> > > 3. 文档嵌入：给每一组分配相应的线程$\text{→}$并行运行$f_D$将每组中文档编码为一个上下文嵌入集合$E_d$
> > >
> > > :three:存储阶段：将得到的嵌入向量以$\text{16/32}$位$\text{Float}$存于磁盘中$\text{→}$可构建索引$\text{→}$查询时再加载出来
> >
> > ### $\textbf{2.3.2. }$基于$\textbf{ColBERT}$的$\textbf{Top-}\boldsymbol{k}$重排
> >
> > > :one:目的
> > >
> > > 1. 给定：一个查询$q+$一小组的$k$个文档(例如$k\text{=1000}$)
> > > 2. 需要：按照[查询$\xleftrightarrow{相似度}$文档]对$k$个文档进行排序，找到最相似的文档
> > >
> > > :two:流程
> > >
> > > 1. 预处理与数据准备
> > >
> > >    - 查询处理：运行$f_Q$以计算其上下文嵌入集$E_q\textcolor{red}{\text{∈}\mathbb{R}^{N_d\text{×}m}}$
> > >
> > >    - 文档载入：从磁盘中加载一批($k\text{=}1000$)文档的嵌入进入内存
> > >
> > >    - 向量构建：将所有的文档嵌入组织成三维张量$\textbf{D}\textcolor{red}{\text{∈}\mathbb{R}^{L\text{×}m\text{×}k}}$
> > >
> > >      <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241130043629711.png" alt="image-20241130035316438" width=500 />  
> > >
> > >    - 数据迁移：将三维张量$\textbf{D}$移入$\text{GPU}$显存，==后续操作都在$\text{GPU}$上完成==
> > >
> > > 2. 批量点积(余弦距离)计算
> > >
> > >    - 点积：得到交叉匹配矩阵$\textbf{M}\text{∈}\textcolor{red}{\mathbb{R}^{N_d\text{×}L}}$ 
> > >
> > >      $\textbf{M}\text{=}\begin{bmatrix}
> > >      \text{M}_{11} & \text{M}_{12} & \cdots & \text{M}_{1L} \\
> > >      \text{M}_{21} & \text{M}_{22} & \cdots & \text{M}_{2L} \\
> > >      \vdots  & \vdots  & \ddots & \vdots  \\
> > >      \text{M}_{N_d1} & \text{M}_{N_d2} & \cdots & \text{M}_{N_dL} \\
> > >      \end{bmatrix}
> > >      \text{←}M_{ij} \text{=} E_{q_i} \text{×} E_{d_j}^T(点积)\text{←}
> > >      \begin{cases}查询E_q中第i个\text{Token}\\\\文档E_d中第j个\text{Token}\end{cases}$ 
> > >
> > >    - 批处理：多个文档为一批依次执行点积$\text{→}$合并所有文档的交叉匹配矩阵$\text{→}\textbf{D}^{\prime}\text{∈}\textcolor{red}{\mathbb{R}^{N_d\text{×}L\text{×}k}}$ 
> > >
> > > 3. 计算得分：
> > >
> > >    - 每个$\text{MaxSim}$：对所有$k$个交叉匹配矩阵进行最大池化
> > >
> > >      $\textbf{M}\text{=}\begin{bmatrix}
> > >      \text{M}_{11} & \text{M}_{12} & \cdots & \text{M}_{1L} \\
> > >      \text{M}_{21} & \text{M}_{22} & \cdots & \text{M}_{2L} \\
> > >      \vdots  & \vdots  & \ddots & \vdots  \\
> > >      \text{M}_{N_d1} & \text{M}_{N_d2} & \cdots & \text{M}_{N_dL} \\
> > >      \end{bmatrix}\xrightarrow{最大值池化}
> > >      \begin{bmatrix}
> > >      (\textbf{M}_{1})_{\text{max}}  \\
> > >      (\textbf{M}_{2})_{\text{max}}  \\
> > >      \vdots    \\
> > >      (\textbf{M}_{N_d})_{\text{max}} \\
> > >      \end{bmatrix}$
> > >
> > >    - 每个文档的得分：对经过最大池化的结果进行求和(规约)$\displaystyle\text{score}(d)=\sum_{i=1}^{N_d}\left(\mathbf{M}_i\right)_{\max }$
> > >
> > > 4. 文档重排：将所有文档按得分从高到低排序
> > >
> > > :three:与传统基于$\text{BERT}$对比
> > >
> > > |       模型       | 每次输入$\textbf{BERT}$的长度 | 每次自注意力机制复杂度 |
> > > | :--------------: | --------------------------- | ---- |
> > > | $\text{ColBERT}$ | $\|q\|$(只需要查询)         | $O(\|q\|^2)$ |
> > > |  $\text{BERT}$   | $\|q\|\text{+}\|d_i\|$(查询$\xleftrightarrow{分别拼接}$每个文档) | $O\left((\|q\|\text{+}\|d_i\|)^2\right)$ |
> > >
> >
> > ### $\textbf{2.3.3. }$基于$\textbf{ColBERT}$的端到端$\textbf{Top-}\boldsymbol{k}$检索
> >
> > > :one:目的
> > >
> > > 1. 给定：一个查询$q+$超大规模的文档集合(例如一千万个)
> > > 2. 需要：避免直接对每个文档都进行评估和排序，但仍能快速返回==前$k$个==最相关文档
> > >
> > > :two:索引构建
> > >
> > > 1. 构建操作：
> > >    - 在离线嵌入时：记录并维护每个[(词级)嵌入$\xrightarrow{映射}$所属文档]关系
> > >    - 在此刻的操作：使用$\text{FAISS}$构建嵌入的快速向量相似性索引，并保留[嵌入$\xrightarrow{映射}$所属文档]
> > >
> > > 2. 关于$\text{FAISS}$：此处所采用的是$\text{IVF+PQ}$，流程和方法如下
> > >    - 构建操作：先用$\text{IVF}$将所有文档嵌入向量划分为$P$个簇$\text{→}$对每个簇内向量应用$\text{PQ}$压缩
> > >    - 查询过程：由[查询向量$\xleftrightarrow{距离}$簇中心]得最近的$p$个簇$\text{→}$再计算[查询向量$\xleftrightarrow{相似度}$簇内$\text{PQ}$压缩]
> > >
> > > :two:查询的流程
> > >
> > > 1. 第一阶段(近似过滤)：用==近似检索==方法，从所有文档中快速找出最相关的小部分文档
> > >
> > >    |   操作   | 描述                                                         |
> > >    | :------: | ------------------------------------------------------------ |
> > >    | 查询发起 | 让$E_q\textcolor{red}{\text{∈}\mathbb{R}^{N_d\text{×}m}}$的每行(每个嵌入)，都并行发起对$\text{FAISS}$索引的相似性查询 |
> > >    | 查询结果 | 得到每个查询嵌入的前$k^{\prime}$(例如$k^{\prime}\text{=}\cfrac{k}{2}$)个最匹配的文档嵌入 |
> > >    | 文档映射 | 对所有得到的$N_q \text{×} k^{\prime}$个最匹配文档，往回映射得到$N_q \text{×} k^{\prime}$个文档$\text{ID}$ |
> > >    | 去重处理 | 得到$N_q \text{×} k^{\prime}$个文档$\text{ID}$中$K \text{≤} N_q \text{×} k^{\prime}$个文档，==即为过滤得到的文档== |
> > >
> > > 2. 第二阶段(精细排序)：按照$\text{2.3.2.}$中所描述的方式，对小规模$K$个文档进行重排

# $\textbf{3. }$实验评估

> ## $\textbf{3.1. }$数据集与配置
>
> > :one:数据集概述
> >
> > |      数据集       | 描述                                              | 本文的处理                 |
> > | :---------------: | ------------------------------------------------- | -------------------------- |
> > | $\text{MS MARCO}$ | $\text{8.8}$百万段网页文本，$\text{Bing}$查询结果 | 使用开发集和评估集进行评估 |
> > | $\text{TREC CAR}$ | 维基百科合成数据集，约$2900$万段文本              | 使用前四折训练，第五折验证 |
> >
> > :two:实现与超参数
> >
> > 1. 框架与工具：使用$\text{Python3/PyTorch1}$实现，预训练$\text{BERT}$模型通过`transformers`库加载
> >
> > 2. 模型设置：
> >
> >    |             参数类型             | 参数设置                                                    |
> >    | :------------------------------: | ----------------------------------------------------------- |
> >    |             训练参数             | 学习率$3 \text{×} 10^{-6}/\text{BatchSize=32}$              |
> >    |      $\text{ColBERT}$超参数      | $N_q\text{=32}/m\text{=128}$                                |
> >    | $\text{FAISS}$配置($\text{IVF}$) | $P\text{=2000}/p\text{=}10/k^{\prime}\text{=}k\text{=}1000$ |
> >    | $\text{FAISS}$配置($\text{PQ}$)  | 子向量数$s\text{=16}$                                       |
> >
> > 3. 预训练模型：
> >
> >    - 在$\text{MS MARCO}$上：用$\text{Google}$官方的预训练$\text{BERT}$模型初始化$\text{ColBERT}$查询和文档编码器
> >    - 在$\text{TREC CAR}$上：使用$\text{Nogueira/Cho}$发布的预训练$\text{BERT}_{\text{LARGE}}$模型
> >
> > :three:硬件配置：
> >
> > 1. 延迟评估：单个$\text{Tesla V100 GPU}$，两个$\text{Intel Xeon Gold 6132 CPU}$
> > 2. 索引实验：四个$\text{Titan V GPU}$，两个$\text{Intel Xeon Gold 6132 CPU}$
>
> ## $\textbf{3.2. }$实验结论
>
> > ### $\textbf{3.2.1. Top-}\boldsymbol{k}$重排的[质量$\boldsymbol{\xleftrightarrow{权衡}}$成本]
> >
> > > :one:实验设置
> > >
> > > 1. 比较的模型：$\text{ColBERT/KNRM/DuetfastText+ConvKNR}$以及两种$\text{BERT}$
> > > 2. 评估的指标：
> > >    - 排序效果：$\text{MS MARCO}$上的$\text{MRR@10}$或者$\text{MAP}$
> > >    - 查询成本：延迟$\&\text{FLOPs}$(每个查询的浮点运算次数)
> > >
> > > :two:结果分析：$\text{ColBERT}$用极低的代价达到了与$\text{BERT}$差不多的效果
> > >
> > > 1. 排序效果：采用$\text{MAP/MRR@10}$指标时，$\text{ColBERT≈BERT}_{\text{BASE}}\text{<BERT}_{\text{LARGE}}$ 
> > > 2. 开销成本：$\text{ColBERT}$在延迟$/\text{FLOPS}$上远低于$\text{BERT}$，并且仅仅只略高于非$\text{BERT}$模型
> >
> > ### $\textbf{3.2.2. }$端到端的$\textbf{Top-}\boldsymbol{k}$重排
> >
> > > :one:实验设置
> > >
> > > 1. 比较的模型：$\text{ColBERT/BM25/doc2query/DeepCT/docTTTTquery}$ 
> > >
> > > 2. 评估指标：
> > >
> > >    |     评估的方面     | 指标                                      |
> > >    | :----------------: | :---------------------------------------- |
> > >    |      排序质量      | $\text{MS MARCO}$上的$\text{MRR@10}$      |
> > >    | 找出相关文档的能力 | $\text{Recall@50/Recall@200/Recall@1000}$ |
> > >    |      开销成本      | 延迟                                      |
> > >
> > > :three:结果分析：$\text{ColBERT}$通过端到端检索在效果和召回率上都有显著提升
> > >
> > > 1. 总体来讲：端到端的$\text{ColBERT}$，性能还要优于与仅进行后期重排的$\text{ColBERT}$ 
> > > 2. 延迟：$\text{ColBERT}$超过其它所有高度优化的廉价模型
> > > 3. 召回率：也超过了其它所有模型
> >
> > ### $\textbf{3.2.3. }$其它实验验证
> >
> > > :one:消融实验：$\text{BERT}$的每个组成部分对其质量分别有何贡献$?$
> > >
> > > |    模型    | 模型的描述                    | 相比$\textbf{ColBERT}$ | 结论                          |
> > > | :--------: | ----------------------------- | :--------------------: | ----------------------------- |
> > > | $\text{A}$ | 只为查询$/$文档生成单一向量   |  $\text{MRR@10}$较低   | 后期交互很重要                |
> > > | $\text{B}$ | 替换$\text{MaxSim}$为平均池化 |  $\text{MRR@10}$很低   | 最大相似度优于平均相似度      |
> > > | $\text{C}$ | 去除查询增强机制              |  $\text{MRR@10}$极低   | 查询增强机制对于模型至关重要  |
> > > | $\text{D}$ | 只是用$\text{BERT}$前五层训练 |  $\text{MRR@10}$较低   | 更深的$\text{BERT}$能提高性能 |
> > > | $\text{E}$ | 完整的$\text{BERT}$           |      $\text{N/A}$      | $\text{N/A}$                  |
> > >
> > > :two:离线预计算的开销
> > >
> > > 1. 索引吞吐量：
> > >
> > >    - 索引优化的消融实验：说明$\text{ColBERT}$索引的相对高效
> > >
> > >      <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241130163054201.png" alt="image-20241130163054201" width=540 /> 
> > >
> > >    - 此外较$\text{BERT}$而言$\text{ColBERT}$减少了查询对文档的重复计算
> > >
> > > 2. 空间占用：减小$m$的大小可直接减小索引大小，且对性能降低的效果不甚明显

