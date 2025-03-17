@[toc]

[原文](https://doi.org/10.48550/arXiv.2211.01267)，另外这篇文章被$\text{ICLR'23}$拒稿了，公开处刑见[$\text{OpenReview}$](https://openreview.net/forum?id=2EFQ_QlcPs8)(几个审稿人说的都挺好的)

# $\textbf{1. }$导论

> :one:多向量文本检索的定义：给定一个查询$Q$和一个包含$N$个段落的段落集$\mathscr{P}\text{=}\left\{P^{(1)},P^{(2)},\ldots,P^{(N)}\right\}$ 
>
> 1. 嵌入：为$Q$与$\mathscr{P}\text{=}\left\{P^{(1)},P^{(2)},\ldots,P^{(N)}\right\}$的每个$\text{Token}$都生成一向量，即$Q\text{=}\left\{q_{1},q_2,\ldots,q_{n}\right\}$和$P^{(j)}\text{=}\{p_{1}^{(j)},p_{2}^{(j)},\ldots,p_{m}^{(j)}\}$  
> 2. 检索：让每个$q_i\text{∈}Q$在所有段落子向量集$P^{(1)}\text{∪}P^{(2)}\text{∪}\ldots\text{∪}P^{(N)}$中执行$\text{MIPS}$(最大内积搜索)，得到$\text{Top-}K$的段落子向量
> 3. 回溯：合并$n$个$q_i$的$\text{MIPS}$结果得到$n\text{×}K$个段落子向量，将每个子向量回溯到其所属段落得到$\mathscr{P}^\prime\text{=}\left\{P^{(1)},P^{(2)},\ldots,P^{(M)}\right\}$ 
> 4. 重排：用基于$\text{MaxSim}$的后期交互计算精确相似度评分$\displaystyle{}(Q,P)\text{=}\sum_{i=1}^{n} \max _{j=1 \ldots m} q_{i}^{\top} p_{j}$，从而对$\mathscr{P}^\prime$进行重排得到最相似段落 
>
> :two:$\text{ColBERT}$中$\text{MaxSim}$的缺陷 
>
> 1. 检索性能上：$\text{MaxSim}$操作为每个查询向量$q_i$找到**单个**段落向量$p_{q_i}$，放宽单个的约束可能可以提升检索性能
> 2. 检索效率上：在检索阶段，传统的$\text{ColBERT}$并不会对向量个数$|P|$进行剪枝，但存储计算开销$\text{∝}|P|$ 
>
> :two:改进的直觉：引入稀疏矩阵$\textbf{A}$来扩展原有的$\text{MaxSim}$操作
>
> 1. 数据结构：相似度矩阵$\textbf{S}$，和稀疏矩阵$\textbf{A}$
>    - 对于$\textbf{S}$：令查询$Q\text{=}\left\{q_{1},q_2,\ldots,q_{n}\right\}$以及文档$P\text{=}\{p_1,p_2,...,p_m\}$中，记子内积为$s_{ij}\text{=}{q}_{i}^{\top}{p}_{j}$，由此构成$\textbf{S}\text{∈}\mathbb{R}^{n\text{×}m}$ 
>    - 对于$\textbf{A}$：让每个元素$a_{ij}\text{∈}[0,1]$来对$\textbf{S}$中的元素进行不同强度的选择，由此构成$\textbf{A}\text{∈}\mathbb{R}^{n\text{×}m}$ 
> 2. 相似评分：$\displaystyle\text{Sim}(Q,D)\text{=}\frac{1}{Z} \sum_{i=1}^{n} \sum_{j=1}^{m}(\textbf{S}\text{∘}\textbf{A})_{ij}$ 
>    - $\textbf{S}\text{∘}\textbf{A}$表示两矩阵的$\text{Hadamard}$积，即两形状相同的矩阵按位相乘，构成新的形状相同的矩阵
>    - $\displaystyle\sum_{i=1}^{n} \sum_{j=1}^{m}(\textbf{S}\text{∘}\textbf{A})_{ij}$表示将$\textbf{S}\text{∘}\textbf{A}$矩阵的每位相加，最后再进行$Z$归一化即除以$\textbf{A}$中所有$m\text{×}n$个元素的总和 
> 3. 对比：稠密检索和多向量检索，可以看出稀疏矩阵$\textbf{A}$的特殊情况
>    - 稠密检索：计算$P$和$Q$嵌入后二者`<cls>`的内积，相当于用$\textbf{A}$屏蔽掉其它子向量的相似度 
>      <img src="https://i-blog.csdnimg.cn/direct/0d5f8f1bce124dd396e456a1ccd7de13.png" alt="image-20250306233440478" width=800 /> 
>    - 多向量检索：找到每个$q_i$找到最相似的段落子向量$p$，相当于设置$\textbf{A}$以选取$\textbf{S}$中每行最大元素
>      <img src="https://i-blog.csdnimg.cn/direct/b16995c1a9e54a9cbc6d865928c76300.png" alt="image-20250306233530006" width=800 /> 

# $\textbf{3. Aliger}$原理

> ## $\textbf{3.0. }$模型的假设
>
> > :one:不同任务有不同的最优对齐方式，例如
> >
> > 1. 事实型：例如查询`Who-is-the-president-of-USA`的答案通常集中在$P$的某一部分，因此只需要少量且集中的对齐即可
> > 2. 论点型：例如查询`Who-is-next-president-of-USA`的答案需要综合多个文档$P$进行分析，影刺需要较多且分散的对齐
> >
> > :two:一个段落$P$中大多$\text{Token}$不贡献语义
> >
> > 1. 实验上：只有$1/10$左右的文档在$\text{MaxSim}$时被检索到过，即大多的$\text{Token}$不具备检索的价值
> > 2. 如何作：可以对大多$\text{Token}$进行剪枝
>
> ## $\textbf{3.1. }$稀疏矩阵
>
> > ### $\textbf{3.1.1. }$稀疏矩阵的分解$\boldsymbol{\textbf{A}\text{=}\tilde{\textbf{A}}\text{∘}(u^q\text{⊗}u^p)\text{∈}\mathbb{R}^{n\text{×}m}}$  
> >
> > > <img src="https://i-blog.csdnimg.cn/direct/38bd974be74443d28041205889fb0066.png" alt="image-20250311010730856" width=666 />  
> > >
> > > |  数据结构  | 维度                                                | 范围                                                       | 意义                                                         |
> > > | :--------: | :-------------------------------------------------- | :--------------------------------------------------------- | ------------------------------------------------------------ |
> > > | 稀疏性矩阵 | $\tilde{\textbf{A}}\text{∈}\mathbb{R}^{n\text{×}m}$ | $a_{ij}\text{∈}\{0,1\}$                                    | $a_{ij}$决定了$s_{ij}\text{=}{q}_{i}^{\top}{p}_{j}$是否参与最终评分 |
> > > | 显著性向量 | $u^q,u^p\text{∈}\mathbb{R}^{n}$                     | $u^q_{i},u^p_{j}\text{∈}[0,1]$                             | $u^q_{i}/u^p_{j}$相当于$q_i/p_j$的一个重要性权重             |
> > > | 显著性矩阵 | $(u^q\text{⊗}u^p)\text{∈}\mathbb{R}^{n\text{×}m}$   | $(u^q\text{⊗}u^p)_{ij}\text{=}u^q_{i}u^p_{j}\text{∈}[0,1]$ | $u^q_{i}u^p_{j}$决定了$s_{ij}\text{=}{q}_{i}^{\top}{p}_{j}$参与最终评分的权重 |
> >
> > ## $\textbf{3.1.2. }$如何确定显著性矩阵$\boldsymbol{(u^q\text{⊗}u^p)}$: 显著性参数的学习 
> >
> > > :one:显著性的定义：以段落子向量$p_i$为例，其显著性定义为$u_{i}^{p}\text{=}\lambda_{i}^{p}\text{×ReLU}\left(\textbf{W}^{p} {p}_{i}\text{+}b^{p}\right)$ 
> > >
> > > 1. 神经网络部分：$\text{ReLU}\left(\textbf{W}^{p}p_i \text{+}b^{p}\right)$为一前馈网络，$\textbf{W}^{p}$和$b^{p}$是可学习参数，最终给出${p}_{i}$一个显著性得分
> > > 2. 稀疏性门控变量：$\lambda_{i}^{p}\text{∈}[0,1]$用于控制该${p}_{i}$被激活的程度，当$\lambda_{i}^{p}\text{=}0$时与${p}_{i}$有关的相似度全被屏蔽 
> > >    <img src=https://i-blog.csdnimg.cn/direct/b42a14efeb0c489096d4c47fc788cd6b.png alt="image-20250311222358185" width=765 />   
> > >
> > > :two:训练方法：基于熵正则化线性规划的训练
> > >
> > > 1. 有关符号：令$s_i^p\text{=ReLU}\left(\textbf{W}^{p} {p}_{i}\text{+}b^{p}\right)$构成向量$s^p\text{=}\{s^p_1,s^p_2,...,s^p_m\}$，以及门控向量${\lambda}^p\text{=}\{\lambda_{1}^{p},\lambda_{1}^{p},...,\lambda_{m}^{p}\}$ 
> > > 2. 训练目标：$\displaystyle\max\langle{s^p,\lambda^p}\rangle\text{ – }\varepsilon{\mathop{\sum}\limits_{{i=1}}^{m}{\lambda^p}_{i}\log{\lambda^p}_{i}}$，并约束$\lambda_{i}^{p}\text{∈}[0,1]$及$\lambda^p$的非零元素数$\left\|\lambda^d\right\|_0\text{=}\left\lceil\alpha^pm\right\rceil$
> > >    - 加权项：最大化$s^p$与$\lambda^p$内积，使所有段落$\text{Token}$的显著性合$\displaystyle{}\sum_{i=1}^{m}u_{i}^{p}$最大，鼓励模型选择使得得分更高的$\text{Token}$
> > >    - 熵项：$\lambda^p$实际对于$s^p$进行类$\text{0/1}$二元选择，会导致结果离散不可微，故加入熵项以进行$\varepsilon$平滑化以便梯度优化 
> > > 3. 训练迭代：初始化辅助变量$a^p/b_1^p,b_2^p,...,n_m^p$为全$0$
> > >    - 更新方式：$\displaystyle{}{a}^{p\text{ }\prime}\text{=}\varepsilon\ln{k}\text{ – }\varepsilon\ln\left\{{\mathop{\sum}\limits_{i}\exp\left(\frac{{s}_{i}^p\text{+}{b}_{i}^p}{\varepsilon}\right)}\right\}$以及${b}_{i}^{\prime}\text{=}\min\left({–{s}_{i}–{a}^{\prime},0}\right)$ 
> > >    - 最终输出：只需几轮迭代后，即可输出结果$\lambda_i^p\text{=}\exp\left(\cfrac{s_i^p\text{+}b_i^p\text{+}a^p}{\varepsilon}\right)$ 
> >
> > ### $\textbf{3.1.3. }$如何确定稀疏性矩阵$\tilde{\textbf{A}}$: 在小样本上的对齐适配
> >
> > > :one:一些稀疏对齐策略
> > >
> > > 1. $\text{Top-}k$：对每个$q_i$选取相关性评分最高的$k$个文档向量，当$k\text{=1}$时退化为$\text{ColBERT}$ 
> > > 2. $\text{Top-}p$：对每个$q_i$选取相关性评分最高的$\max\left(\lfloor pm\rfloor,1\right)$个文档向量，其中$m$为文档长度$p$为对齐比例
> > >
> > > :two:让稀疏对齐策略适应特定目标任务
> > >
> > > 1. 训练阶段：在源域(通用检索语料库)上使用一个固定的对齐策略(如$\text{Top-1}$)训练$\text{Aligner}$
> > > 2. 适应阶段：在不改参数的前提下，基于目标任务完成以下步骤的调整
> > >    - 输入：目标域(目标任务语料库)$\left\{P^{(1)},P^{(2)},\ldots,P^{(N)}\right\}$及少量标注数据$\left\{\left(Q^1, P_{+}^1\right),\left(Q^2,P_{+}^2\right), \ldots,\left(Q^K, P_{+}^K\right)\right\}$ 
> > >    - 检索：用预训练模型为每个查询$Q^i$检索得到候选段落$\left\{P^{\left(i_1\right)}, P^{\left(i_2\right)}, \ldots\right\}$ 
> > >    - 评估：用不同的对齐策略($\text{Top-0.1/Top-0.2/.../Top-1/Top-2/...}$)重新计算候选文档集中每个段落的得分并排序
> > >    - 适应：基于标注数据，选择评估阶段中排序效果(如$\text{nDCG@10}$)最佳的对齐策略，将其作为是用于该任务的对齐策略
> > >
> > > :three:适配后的检索：以$\text{Top-2}$为例，用$\tilde{\textbf{A}}$去选择$(u^q\text{⊗}u^p)$矩阵每行显著性积$(u^q\text{⊗}u^p)_{ij}\text{=}u^q_{i}u^p_{j}$最大的两个元素
> > >
> > > <img src="https://i-blog.csdnimg.cn/direct/b6d230bed12149c1831014017019d22e.png" alt="image-20250312032834301" width=666 /> 
>
> ## $\textbf{3.2. }$基于显著性的剪枝
>
> > :one:原始方法
> >
> > 1. 嵌入：为$Q$与$\mathscr{P}\text{=}\left\{P^{(1)},P^{(2)},\ldots,P^{(N)}\right\}$每个$\text{Token}$都生成向量，即$Q\text{=}\left\{q_{1},q_2,\ldots,q_{n}\right\}$和$P^{(j)}\text{=}\{p_{1}^{(j)},p_{2}^{(j)},\ldots,p_{m}^{(j)}\}$  
> > 2. 检索：让每个$q_i\text{∈}Q$在所有段落子向量集$P^{(1)}\text{∪}P^{(2)}\text{∪}\ldots\text{∪}P^{(N)}$中执行$\text{MIPS}$(最大内积搜索)，得到$\text{Top-}K$的段落子向量
> > 3. 回溯：合并$n$个$q_i$的$\text{MIPS}$结果得到$n\text{×}K$个段落子向量，将每个子向量回溯到其所属段落得到$\mathscr{P}^\prime\text{=}\left\{P^{(1)},P^{(2)},\ldots,P^{(M)}\right\}$ 
> > 4. 重排：用基于$\text{MaxSim}$的后期交互计算精确相似度评分$\displaystyle{}(Q,P)\text{=}\sum_{i=1}^{n} \max _{j=1 \ldots m} q_{i}^{\top} p_{j}$，从而对$\mathscr{P}^\prime$进行重排得到最相似段落 
> >
> > :two:剪枝方法
> >
> > 1. 段落剪枝：用$u_{i}^{p}\text{=}\lambda_{i}^{p}\text{×ReLU}\left(\textbf{W}^{p} {p}_{i}\text{+}b^{p}\right)$计算$P^{(1)}\text{∪}\ldots\text{∪}P^{(N)}$中每个$\text{Token}$的显著性，留下显著性排前$β^p\%$的$\text{Token}$ 
> > 2. 查询剪枝：用$u_{i}^{q}\text{=}\lambda_{i}^{q}\text{×ReLU}\left(\textbf{W}^{q} {q}_{i}\text{+}b^{q}\right)$计算$Q$中每个$\text{Token}$的显著性，同样只留下显著性排前$β^q\%$的$\text{Token}$  
> > 3. 检索：让剩下的查询子向量$q_i$集，对剩下的段落子向量$p_j$集进行$\text{MIPS}$搜索，后续回溯步骤不变
> > 4. 重排：将精确的距离评分从$\text{MaxSim}$，变成基于稀疏矩阵的相似度评分$\displaystyle\text{Sim}(Q,D)\text{=}\frac{1}{Z} \sum_{i=1}^{n} \sum_{j=1}^{m}(\textbf{S}\text{∘}\textbf{A})_{ij}$  

# $\textbf{4. }$实验

> :one:实验的关键设置
>
> 1. 模型配置：采用$\text{Top-1}$任务进行训练$\text{Transformer}$，在$\text{Ms-Marco}$上进行微调
> 2. 检索过程：采用$\text{ScaNN}$最邻近查询进行$\text{MIPS}$，为每个$q_i$查找$\text{Top-4000}$最邻近
> 3. 对齐适配：采样$\text{8}$个标记数据，在$k\text{∈}\{1,2,4,6,8\}$和$p\text{∈}\{0.5\%,1\%,1.5\%,2\%\}$上选择能最大化$\text{nDCG@10}$的策略
>
> :two:实验的关键结果
>
> 1. 检索性能：$\text{Aligner}$的性能优于单向量模型和$\text{ColBERTv2}$，并且仅需$\text{8}$个标记样本就可使$\text{Aligner}$完成对特定任务的适配
> 2. 检索开销：高度剪枝后($β^q\text{=}50\%,β^d\text{=}40\%$)性能仍接近未剪枝，更高程度的剪枝后性能仍然衰减不大，然而索引大大减小
> 3. 可解释性：在具体的任务当中，模型能够自动识别关键的名词$\text{/}$动词短语，并在不同任务中有不同的识别模式