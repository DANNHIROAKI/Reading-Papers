[TOC]

原文章：[$\text{BGE M3-Embedding}$](https://doi.org/10.48550/arXiv.2402.03216) 

# $\textbf{1. }$背景与导论

> ## $\textbf{1.1. }$研究背景
>
> > :one:嵌入与检索
> >
> > 1. 嵌入模型：一种深度学习模型，将文本转化为向量，以捕捉文本的意义信息
> > 2. 检索方法：密集$\mathcal{/}$多向量$\mathcal{/}$稀疏(词汇)
> >    |    模型    | 嵌入方式                         | 相似度计算           |     模型实例     |
> >    | :--------: | -------------------------------- | -------------------- | :--------------: |
> >    |  密集检索  | 整一段编码为单个段落级稠密向量   | 两个向量间的点积计算 |  $\text{BERT}$   |
> >    | 多向量检索 | 整一段编码为多个词汇级稠密向量   | 两组向量间的复杂交互 | $\text{ColBERT}$ |
> >    |  稀疏检索  | 整一段中词的重要性分布(词项权重) | 词匹配得分           |  $\text{BM25}$   |
> >
> > :two:当前嵌入模型的局限：通用性不足
> >
> > |      局限      | 描述                                                       |
> > | :------------: | ---------------------------------------------------------- |
> > |   语言局限性   | 大多数模型针对英语开发，在其他语言上表现差                 |
> > |   功能单一性   | 嵌入模型只针对单一检索功能**训练**                         |
> > | 无法处理长文本 | 大多数模型只能处理短文本输入，缘于训练长文档检索器成本太高 |
>
> ## $\textbf{1.2. }$本文的研究
>
> > :one:$\text{M3-Embedding}$的功能：克服通用性不足的问题
> >
> > |   功能   | 描述                                                         |
> > | :------: | ------------------------------------------------------------ |
> > |  多语言  | 支持$\text{100}$多种语言，学习不同语言的共同语义空间，支持语言内$/$跨语言的检索 |
> > |  多功能  | 能生成三种不同类型的嵌入，以同时支持密集检索$\mathcal{/}$稀疏检索$\mathcal{/}$多向量检索 |
> > | 多颗粒度 | 处理不同长度的输入，从细颗粒度的短输入$\text{→}$最长$\text{8192}$个$\text{Token}$ |
> >
> > :two:$\text{M3-Embedding}$的训练：如何整合三种嵌入方式的不同训练目标
> >
> > 1. 高效的数据策划：
> >    - 数据源：无监督数据$/$监督数据$/$合成数据
> >    - 目的：互为补充，应用在不同训练阶段
> > 2. 一种新的自我蒸馏框架：
> >    - 结构：`<CLS>`结构嵌入$\xrightarrow{用于}$密集检索，其它$\text{Token}$嵌入$\xrightarrow{用于}$稀疏检索和多向量检索
> >    - 原理：整合不同检索功能产生的相关性分数为教师信号$\xrightarrow{知识蒸馏}$反馈给模型自己，不断增强循环
> > 3. 优化了批处理策略：实现大批量$\text{+}$高吞吐的训练，以提高嵌入模型的区分能力
>
> ## $\textbf{1.3. }$有关工作
>
> > :one:一般的文本嵌入
> >
> > 1. 文本嵌入的进展：预训练模型(有效编码将数据的潜在语义)，对比学习(负采样和知识蒸馏的运用)
> > 2. 文本嵌入的趋势：多功能的嵌入模型(统一支持多种场景)，如$\text{E5/BGE/SGPT/Contriever... }$
> >
> > :two:$\text{IR}$的文本嵌入：密集检索$\mathcal{/}$多向量检索$\mathcal{/}$稀疏(词汇)检索
> >
> > :three:多语言的文本嵌入：
> >
> > 1. 实现多文本嵌入：
> >    |         方向(思路)         | 模型                          |
> >    | :------------------------: | ----------------------------- |
> >    |        多语言数据集        | $\text{MIRACL/mMARCO/MKQA}$   |
> >    |  多语言编码器**(预训练)**  | $\text{mBERT/mT5/XLM-R}$      |
> >    | 多语言嵌入模型**(微调后)** | $\text{mDPR/mContriever/mE5}$ |
> > 2. 当前的挑战：其他语种与英语的固有差距，其它语种语料库稀少

# $\textbf{2. M3-Embedding}$ 

> ## $\textbf{2.1. }$模型核心: 混合检索方式
>
> > ### $\textbf{2.1.1. }$三种不同的检索方式
> >
> > > #### $\textbf{2.1.1.1. }$稠密检索
> > >
> > > > :one:分词：保留`<cls>`标记
> > > >
> > > > 1. 查询：$q\text{=}$`<q-cls> <q-Token-1> <q-Token-2> ....` 
> > > > 2. 文档：$p\text{=}$`<p-cls> <p-Token-1> <p-Token-2> ....` 
> > > >
> > > > :two:嵌入：词级嵌入，但重点关注$\mathbf{H}_{\mathbf{}}[0]$ 
> > > >
> > > > 1. 查询：$q\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}$，其中`<q-cls>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}[0]$，`<q-Token-1>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}[1]$ ....
> > > > 2. 文档：$p\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{p}}$，其中`<p-cls>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{p}}[0]$，`<p-Token-1>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{p}}[1]$ ....
> > > >
> > > > :three:归一化：为方便计算内积
> > > >
> > > > 1. 查询：`<q-cls>`$\xrightarrow[\text{Norm}]{\text{Encoder}}\text{norm}\left(\mathbf{H}_q[0]\right)$，作为$q$的最终嵌入表示
> > > > 2. 文档：`<p-cls>`$\xrightarrow[\text{Norm}]{\text{Encoder}}\text{norm}\left(\mathbf{H}_p[0]\right)$，作为$p$的最终嵌入表示
> > > >
> > > > :four:相似度：$s_{\text{dense}} \xleftarrow{内积} \langle e_p, e_q \rangle$ 
> > >
> > > #### $\textbf{2.1.1.2. }$词汇(稀疏)检索
> > >
> > > > :one:分词：可以不保留`<cls>`标记
> > > >
> > > > 1. 查询：$q\text{=}$`<q-Token-1> <q-Token-2> <q-Token-3> ....` 
> > > > 2. 文档：$p\text{=}$`<p-Token-1> <p-Token-2> <q-Token-3> ....` 
> > > >
> > > > :two:嵌入：词级嵌入
> > > >
> > > > 1. 查询：$q\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}$，其中`<q-Token-1>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}[1]$，`<q-Token-2>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}[2]$...
> > > > 2. 文档：$p\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{p}}$，其中`<p-Token-1>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{p}}[1]$，`<q-Token-2>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}[2]$...
> > > >
> > > > :three:权值：将所有嵌入映射为$\mathbf{W}_{\mathrm{lex}}^T \mathbf{H}_q[i]$标量，注意$\text{Token}$相同时取权值最大值
> > > >
> > > > 1. 查询：`<q-Token-i>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}[i]\xrightarrow[\text{ReLU}激活函数]{\mathbf{W}_{\text{lex}}权重矩阵}w_{q_i}\text{=}\text{ReLU}\left(\mathbf{W}_{\mathrm{lex}}^T \mathbf{H}_q[i]\right)$ 
> > > > 2. 文档：`<p-Token-i>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{p}}[i]\xrightarrow[\text{ReLU}激活函数]{\mathbf{W}_{\text{lex}}权重矩阵}w_{p_i}\text{=}\text{ReLU}\left(\mathbf{W}_{\mathrm{lex}}^T \mathbf{H}_p[i]\right)$ 
> > > >
> > > > :four:得分：先过滤得查询$q/p$共同$\text{Token}$，再将相同$\text{Token}$的权值相乘相加$\displaystyle{}s_{\text {lex}} \text{←} \sum_{t \in q \cap p}\left(w_{q_t} \text{×} w_{p_t}\right)$ 
> > >
> > > #### $\textbf{2.1.1.3. }$多向量检索
> > >
> > > > :one:分词：可以不保留`<cls>`标记
> > > >
> > > > 1. 查询：$q\text{=}$`<q-Token-1> <q-Token-2> <q-Token-3> ....` 
> > > > 2. 文档：$p\text{=}$`<p-Token-1> <p-Token-2> <q-Token-3> ....` 
> > > >
> > > > :two:嵌入：词级嵌入
> > > >
> > > > 1. 查询：$q\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}$，其中`<q-Token-1>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}[1]$，`<q-Token-2>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}[2]$...
> > > > 2. 文档：$p\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{p}}$，其中`<p-Token-1>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{p}}[1]$，`<q-Token-2>`$\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}[2]$...
> > > >
> > > > :three:嵌入后：投影到权值$\text{+}$归一化
> > > >
> > > > 1. 查询：$q\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{q}}\xrightarrow{权重\mathbf{W}_{\mathrm{mul}}^T}\left(\mathbf{W}_{\mathrm{mul}}^T \mathbf{H}_q\right)\xrightarrow{归一化}\text{norm}\left(\mathbf{W}_{\mathrm{mul}}^T \mathbf{H}_q\right)$ 
> > > > 2. 文档：$p\xrightarrow{\text{Encoder}}\mathbf{H}_{\mathbf{p}}\xrightarrow{权重\mathbf{W}_{\mathrm{mul}}^T}\left(\mathbf{W}_{\mathrm{mul}}^T \mathbf{H}_p\right)\xrightarrow{归一化}\text{norm}\left(\mathbf{W}_{\mathrm{mul}}^T \mathbf{H}_p\right)$ 
> > > >
> > > > :four:得分：就是$\text{ColBERT}$的后期交互，$\displaystyle{}s_{m u l} \text{ ←} \frac{1}{N} \sum_{i=1}^N \max _{j=1}^M \left\{E_q[i] \text{×} E_p^T[j]\right\}$ 
> >
> > ### $\textbf{2.1.1. }$三种检索方式的
> >
> > > :one:集成方法
> > >
> > > 1. 分立：用三种不同的方法，分别单独就行检索，得到三个检索结果$s_{\text {dense}}/s_{\text {lex}}/s_{\text {mul}}$ 
> > > 2. 整合：最终得分$s_{\text {rank}} \leftarrow w_1  s_{\text {dense}}\text{+}w_2  s_{\text {lex}}\text{+}w_3  s_{\text {mul}}$，三个系数由不同下游场景确定
> > >
> > > :two:集成表示：$d^y \leftarrow \mathrm{fn}^*\left(q^x, D^y\right)$
> > >
> > > |     参数      | 含义                                                         |
> > > | :-----------: | ------------------------------------------------------------ |
> > > |     $q^x$     | 由语言$x$给出的查询$q$                                       |
> > > |     $D^y$     | 由语言$y$组成的语料库$D$                                     |
> > > | $\text{fn}^*$ | 可以是密集检索$/$词汇检索$/$多向量检索中的任何一种函数，或者组合 |
> > > |     $d^y$     | 由语言$x$给出的查询$q$，在由语言$y$组成的语料库$D$中，采取$\text{fn}^*$方式检索，所得的结果 |
>
> ## $\textbf{2.2. }$模型训练: 一种新的自蒸馏框架
>
> > ### $\textbf{2.2.1. }$训练数据: 多样化的数据策划
> >
> > > :one:大规模预训练数据：源于无标签语料库的无监督数据，一般为简单清洗后的“标题-正文”结构文本
> > >
> > > 1. 普通的：维基百科$\text{/S2ORC/xP3/mC4/CC-News/MTP}$
> > > 2. 特殊的：翻译数据集的平行句子$\text{NLLB/CCMatrix}$，用于学习跨语言的统一嵌入
> > >
> > > :two:中规模微调数据：源于有标签语料库的监督数据
> > >
> > > | 语种 | 来源                                                 |
> > > | :--: | ---------------------------------------------------- |
> > > | 英文 | $\text{HotpotQA/TriviaQA/NQ/MS MARCO.... }$          |
> > > | 中文 | $\text{DuReader/mMARCO-ZH/T2-Ranking/CMedQAv2.... }$ |
> > > | 其它 | $\text{Mr. TyDi/MIRACL}$                             |
> > >
> > > :three:小规模合成数据：本文额外生成的数据集，称之为$\text{MultiLongDoc}$
> > >
> > > 1. 来源：抽取长篇文章的随机段落$\text{→}$用$\text{ChatGPT}$生成段落的对应问题$\text{→}$将二者整合成文本对
> > > 2. 目的：缓解长文档检索任务的不足
> >
> > ### $\textbf{2.3.1. }$损失函数: $\boldsymbol{\mathcal{L}_{\textbf {final}} \textbf{←} \cfrac{\mathcal{L}+\mathcal{L}^{\prime}}{2}}$ 
> >
> > > :one:基础损失函数$\mathcal{L}$：基于集成学习原理，直接将不同模型的预测结果加权合并
> > >
> > > 1. 原始损失：$\text{InfoNCE}$损失函数$\mathcal{L}_{s(\cdot)}=-\log \cfrac{\exp (\cfrac{s(q, p^*)}{\tau})}{\displaystyle{}\sum_{p \in\{p^*, P^{\prime}\}} \exp (\cfrac{s(q, p)}{\tau})}$ 
> > >    |         参数          | 含义                                                         |
> > >    | :-------------------: | :----------------------------------------------------------- |
> > >    |         $p^*$         | 查询$q$的正样本，即与查询最相关的段落或文档                  |
> > >    |         $P'$          | 查询$q$的负样本**集**，包含与查询不相关的段落或文档          |
> > >    | $s(q, p)$和$s(\cdot)$ | 查询$q$和段落$p$之间的相似度得分，可以通过$s_{\text{dense}}/s_{\text{lex}}/s_{\text{mul}}$任一个得到 |
> > >    |        $\tau$         | 温度参数，控制得分的平滑度                                   |
> > > 2. $\mathcal{L}$的构成：
> > >    - 得分的线性加权：$s_{\text {inter}} \leftarrow w_1  s_{\text {dense}}\text{+}w_2  s_{\text {lex}}\text{+}w_3  s_{\text {mul}}$ 
> > >    - 损失的线性加权：$\mathcal{L} \leftarrow \cfrac{\lambda_1  \mathcal{L}_{\text {dense}}\text{+}\lambda_2  \mathcal{L}_{\text {lex}}\text{+}\lambda_3  \mathcal{L}_{\text {mul}}\text{+}\mathcal{L}_{\text {inter}}}{4}$ 
> > > 3. 一些思考：
> > >    - 问题：不同检索方法的训练目标**相互冲突**，对$\mathcal{L}_{\text {dense}}/\mathcal{L}_{\text {lex}}/\mathcal{L}_{\text {mul}}$分别优化再加和行不通
> > >    - 解决：通过对$s_{\text{inter}}$进行蒸馏，以统一优化目标
> > >
> > > :two:蒸馏损失$\mathcal{L}^{\prime}$：基于自蒸馏的框架
> > >
> > > 1. 原始损失：一种改进的交叉熵
> > >    |                 损失                  | 损失公式                                                     |
> > >    | :-----------------------------------: | ------------------------------------------------------------ |
> > >    | $\mathcal{L}_{\text{dense}}^{\prime}$ | $-\left(\text{Softmax}\left(s_{\text{inter}}\right)\right) \text{×} \log \left(\text{Softmax}\left(s_{\text{dense}}\right)\right)$ |
> > >    |  $\mathcal{L}_{\text{lex}}^{\prime}$  | $-\left(\text{Softmax}\left(s_{\text{inter}}\right)\right) \text{×} \log \left(\text{Softmax}\left(s_{\text{lex}}\right)\right)$ |
> > >    |  $\mathcal{L}_{\text{mul}}^{\prime}$  | $-\left(\text{Softmax}\left(s_{\text{inter}}\right)\right) \text{×} \log \left(\text{Softmax}\left(s_{\text{mul}}\right)\right)$ |
> > > 2. 蒸馏框架：将综合得分$s_{\text{inter}}$作为教师模型，每种方法的得分$\mathcal{s}_{\text {dense}}/\mathcal{s}_{\text {lex}}/\mathcal{s}_{\text {mul}}$作为学生模型
> > > 3. $\mathcal{L}^{\prime}$的构成：$\mathcal{L}^{\prime} \leftarrow \cfrac{\lambda_1 \cdot \mathcal{L}_{\text {dense}}^{\prime}\text{+}\lambda_2 \cdot \mathcal{L}_{\text {lex}}^{\prime}\text{+}\lambda_3 \cdot \mathcal{L}_{\text {mul}}^{\prime}}{3}$ 
> >
> > ### $\textbf{2.3.2. }$训练流程: 自我蒸馏的优化
> >
> > > <img src="https://i-blog.csdnimg.cn/direct/059e9e26adc94e2695fb3b516a137384.png" alt="image-20250105191622155" height=210 /><img src="https://i-blog.csdnimg.cn/direct/4cf979e8e3f84c4386f26b881cfed774.png" alt="image-20250105191656146" height=210 /> 
> > >
> > > :one:第一阶段：无监督的预训练，让嵌入模型具备基本智能
> > >
> > > 1. 预训练数据：收集的非监督数据
> > > 2. 预训练模型：用$\text{RetroMAE}$方法调整过的$\text{XLM-RoBERTa}$模型
> > > 3. 预训练流程：反复执行稠密检索$\text{→}$根据检索结果通过对比学习调整参数，不断重复这一过程
> > >    - **稠密检索**：即预训练任务，在此处还不涉及其它检索方法
> > >    - **对比学习**：即预训练策略，通过区分正负样本对的相似度，学习嵌入的表示方法
> > >
> > > :two:第二阶段：使用自知识蒸馏进行微调，嵌入模型被微调以建立三种检索功能
> > >
> > > 1. 权重参数：$w_1=1, w_2=0.3, w_3=1, \lambda_1=1, \lambda_2=0.1, \lambda_3=1$
> > >    - 由于$\mathbf{W}_{\text{lex}}$是随机初始化的，所以一开始$s_{\text{lex}}$准确率很低，故有意降低其权重
> > > 2. 微调数据：收集的标注数据$\text{+}$合成数据
> > > 3. 微调策略：$\text{ANCE}$方法，即通过$\text{ANN}$加速寻找正负样本对
>
> ## $\textbf{2.4. }$训练优化: 高效批处理
>
> > :one:所面临的问题
> >
> > 1. 一般模型的训练：一方面保持批足够大(含足够多批内负样本)，另一方面对太长的输入**直接截断**
> > 2. $\text{BGE-M3}$的训练：万万不可直接截断长输入，不然就丧失队长序列的学习能力
> >
> > :two:解决方法：优化的批处理
> >
> > <img src="https://i-blog.csdnimg.cn/direct/b8ca5cc8a36d43a6946c4e4d46b6bcf4.png" alt="image-2024874" width=500 /> 
> >
> > 1. 分组：训练数据按序列长度分到不同小组(如图中$\text{128/1024/4096...}$)
> > 2. 采样：训练实例从**同一组**中抽取一部分作为训练的$\text{Batch}$进入$\text{GPU}$，这一过程有如下两种优化
> >    - 负载平衡：不同$\text{GPU}$采样数据时都保持固定随机种子，使得抽到每批数据(句子)长短分布一致
> >      ```txt
> >      👉不固定随机种子:
> >         GPU1: 可能随机抽到很多长序列 -> 负载重(计算慢) -> 执行计算
> >         GPU2: 可能随机抽到很多短序列 -> 负载重(计算慢) -> 空闲等待
> >      ```
> >    - 填充优化：一批数据进入到$\text{GPU}$仍需填充至最长，但由于分组$+$平衡策略，实际填充数很低
> > 3. 拆分：对序列长度较长的$\text{Batch}$(如图中右边的$\text{4096/8192...}$)，再分割成子批
> >    - 流程(附录$\text{B.3}$)：启用梯度检查点$\text{→}$分子批$\text{→}$逐个处理子批得到嵌入$\text{→}$合并得到原始批的嵌入
> > 4. 广播：$\text{GPU}$会将自己的嵌入结果广播给其它$\text{GPU}$，以扩大批内负样本
> >    ```txt
> >    广播前: 
> >      GPU1处理: [A1, A2, A3]
> >      GPU2处理: [B1, B2, B3]
> >    广播后：
> >      GPU1获得: [A1, A2, A3, B1, B2, B3]
> >      GPU2获得: [A1, A2, A3, B1, B2, B3]
> >    ```

# $\textbf{3. }$实验验证

> ## $\textbf{3.1. }$实验设置
>
> > :one:实验配置
> >
> > 1. 数据集：$\text{MIRACL}$(包含$\text{18}$种语言)
> > 2. 评估指标：$\text{nDCG@10}$，其越高表明相关文档被排到了越靠前的位置
> >
> > :two:检索实现
> >
> > 1. 单独的：
> >    |    检索方式     |      目的      |     索引构建     |       检索任务        |
> >    | :-------------: | :------------: | :--------------: | :-------------------: |
> >    | $\text{Dense}$  | 生成语料库嵌入 | $\text{Faiss}$库 | 检索$\text{Top-1000}$ |
> >    | $\text{Sparse}$ | 生成语料库权重 | $\text{Lucene}$  | 检索$\text{Top-1000}$ |
> > 2. 整合的：
> >    |      检索方式      | 检索内容                         | 重排依据(任务$\textbf{1\&2}$)                                | 重排依据(任务$\textbf{3}$)                                   |
> >    | :----------------: | -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
> >    |    $\text{D+S}$    | 并合各自$\text{Top-1000}$        | $s_{\text {dense}}\text{+}0.3s_{\text {lex}}$                | $0.2s_{\text {dense}}\text{+}0.8s_{\text {lex}}$             |
> >    | $\text{Multi-vec}$ | $\text{Dense}$的$\text{Top-200}$ | $s_{\text {mul}}$                                            | $s_{\text {mul}}$                                            |
> >    |    $\text{All}$    | $\text{Dense}$的$\text{Top-200}$ | $s_{\text {dense}}\text{+}0.3  s_{\text {lex}}\text{+}s_{\text {mul}}$ | $0.15s_{\text {dense}}\text{+}0.5  s_{\text {lex}}\text{+}0.35s_{\text {mul}}$ |
>
> ## $\textbf{3.2. }$实验结果
>
> > ### $\textbf{3.2.1. }$在不同任务上
> >
> > > :one:任务$1$：多语言检索
> > >
> > > 1. 基准：词汇检索($\text{BM25}$)，密集检索($\text{mDPR/mCont./mE5/E5-7b}$)，其它($\text{Text-Emb.-L3}$)
> > > 2. 结果：$\text{Dense}$(性能最优$\text{+}$在语言以外语言提升更大)，$\text{Sparse}$(碾压$\text{BM25}$)，$\text{ALL}$(效果最佳)
> > >
> > > :two:任务$2$：跨语言检索
> > >
> > > 1. 数据：采用$\text{MKQA}$基准数据集，用非英语检索英文维基百科
> > > 2. 结果：$\text{Dense}$就已超过基准，从$\text{Dense+Sparse}$到$\text{ALL}$性能进一步提升
> > >    - 但$\text{Sparse}$的性能不行：源于$\text{Spaese}$主要基于词分布，而跨语言背景下，共同词汇有限
> > >    - 在小语种上表现优异：归功于广泛的无监督训练
> > >
> > > :three:任务$3$：长文档检索
> > >
> > > 1. 数据：$\text{MLDR}$(多语言长文档)，$\text{NarrativeQA}$(英语长文档)，$\text{JinaEmbeddingv2}$(额外基准)
> > > 2. 结果：在$\text{MLDR}$上，$\text{Sparse/MutiVec}$效果突出，但还是$\text{ALL}$得分最高
> > > 3. 消融：去除长文档的限制，$\text{Dense}$依旧超越很多基准
> >
> > ### $\textbf{3.2.2. }$消融实验
> >
> > > :one:实验$1$：自我知识蒸馏的消融
> > >
> > > 1. 消融方式：禁用自知识蒸馏，每种检索方式独立训练，然后再整合
> > > 2. 实验结果：所有自蒸馏的$\text{Dense/Sparse/MultiVec}$效果，都比消融整合模型好
> > >
> > > :two:实验$2$：多阶段训练的消融
> > >
> > > 1. 消融方式：
> > >    |    模型    | $\textbf{XLM-RoBERTa}$模型 | $\textbf{RetroMAE}$调整 | 无监督数据预训练 |
> > >    | :--------: | :------------------------: | :---------------------: | :--------------: |
> > >    | $\text{1}$ |             ✅              |            ❌            |        ❌         |
> > >    | $\text{2}$ |             ✅              |            ✅            |        ❌         |
> > >    | $\text{3}$ |             ✅              |            ✅            |        ✅         |
> > > 2. 结果：$\text{RetroMAE}$显著提高了检索性能，无监督训练进一步提高了嵌入模型的质量