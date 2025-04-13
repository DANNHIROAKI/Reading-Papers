[原文章](https://doi.org/10.48550/arXiv.2304.01982)

@[toc]
# $\textbf{1. XTR}$原理

> ## $\textbf{1.1. }$导论
>
> > :one:多向量相似度的问题定义：$\displaystyle\text{ColBERT}(Q,P)\text{=}\frac{1}{Z} \sum_{i=1}^{n} \sum_{j=1}^{m}(\textbf{S}\text{∘}\textbf{A})_{ij}$
> >
> > 1. 数据结构
> >    - 评分矩阵$\textbf{S}$：令查询$Q\text{=}\left\{q_{1},q_2,\ldots,q_{n}\right\}$以及段落$P\text{=}\{p_1,p_2,...,p_m\}$，记子内积为$s_{ij}\text{=}{q}_{i}^{\top}{p}_{j}$，由此构成$\textbf{S}\text{∈}\mathbb{R}^{n\text{×}m}$ 
> >    - 对齐矩阵$\textbf{A}$：让每个元素$a_{ij}\text{∈}\{0,1\}$来对$\textbf{S}$中的元素进行不同强度的选择，由此构成$\textbf{A}\text{∈}\mathbb{R}^{n\text{×}m}$ 
> > 2. $\text{ColBERT}$版本，通过调整对齐矩阵$\textbf{A}$，让其选择评分矩阵$\textbf{S}$每行最大的一个值，最后除以$Z$归一化
> >    <img src="https://i-blog.csdnimg.cn/direct/c9aa508343f84d7d8c7fb7c73393c418.png" alt="image-20250306233530006" width=800 />   
> > 3. 传统的训练方式：最大化批内正样本${P}^{+}\text{∈}{P}_{1:B}\text{=}\left\lbrack{{P}_{1},\ldots ,{P}_{B}}\right\rbrack$的得分，即最小化${\mathcal{L}}_{\mathrm{{CE}}}\textbf{= }–\log\cfrac{e^{\text{ColBERT}\left( {Q,{P}_{b}}\right)}}{\displaystyle{}\sum_{{b\textbf{=}1}}^{B}e^{\text{ColBERT}\left( {Q,{P}_{b}}\right)}}$ 
> >
> > :two:传统$\text{ColBERT}$的流程
> >
> > <img src="https://i-blog.csdnimg.cn/direct/acf6d1caa7a14872b6db3d6034254900.png" alt="image-20250313215730217" width=800 />   
> >
> > 1. $\text{Token}$检索：用查询单向量集中每个$q_i$检索$k^\prime$个段落$\text{Token}$，最多产生$n\text{×}k^\prime$个候选$\text{Token}$
> > 2. 收集向量：加载$n\text{×}k^\prime$个候选$\text{Token}$所属的段落，收集这些段落中所有的$\text{Token}$向量
> > 3. 评分与重排：对这些段落应用全精度的$\text{ColBERT}$非线性相似度以进行重排
> >
> > :three:研究的动机
> >
> > 1. 传统$\text{ColBERT}$面临的问题
> >    - 训练上：与推理不一致，传统$\text{ColBERT}$的旨在优化最终$\text{ColBERT}$评分，而推理过程旨在获得$\text{Top-}k$的$\text{Token}$
> >    - 开销上：收集$\text{Top-}k$候选段落的多有$\text{Token}$空间开销巨大，由此后续精确距离的计算成本也巨大
> >    - 泛化上：$\text{ColBERT}$的评分函数是非线性的，阻碍了使用$\text{MIPS}$进行检索
> > 2. $\text{XTR}$的改进
> >    - 训练阶段：重新设计了训练目标函数，使得模型能优先检索出最有价值的段落$\text{Token}$
> >    - 重排阶段：完全省去回溯(收集)步骤，直接只用检索到的段落$\text{Token}$来构成
> >    - 缺失补充：只考虑检索到的$\text{Token}$难免漏掉相关的$\text{Token}$，故$\text{XTR}$还会对缺失$\text{Token}$进行自动评分
>
> ## $\textbf{1.2. XTR}$的训练和推理
>
> > :one:举例说明：为何传统$\text{ColBERT}$训练方式对$\text{Token}$检索效果不佳
> >
> > 1. 例子的详细情况：
> >    - 正段落$P^+$：所有$\text{Token}$与$q_i$的相似度均为$0.8$，则最终段落$P^+$评分为$0.8$ 
> >    - 负段落$P^-$：大多$\text{Token}$与$q_i$的相似度均为$0$，但少数几个$\text{Token}$与$q_i$相似度超过$0.8$，导致最终段落得分仍然有$0.2$ 
> > 2. 训练和推理的偏差：
> >    - 训练时：认为得分越高者为越好，即会选择$P^+$
> >    - 推理时：推理的任务是检索与$q_i$最相似的段落$\text{Token}$，如果进行$\text{Top-1}$检索则最终会回溯到选择$P^-$
> >
> > :two:$\text{XTR}$的训练
> >
> > 1. 批内$\text{Token}$检索的训练策略：
> >    - 给定一个查询$Q\text{=}\{q_1,...,q_n\}$和一批共$B$个段落向量$P^{(i)}\text{=}\{p_1^{(i)},...,p_m^{(i)}\}$ 
> >      <img src="https://i-blog.csdnimg.cn/direct/6e92a77acd8b435a80dff8cefdc940ce.png" width=550 />  
> >      
> >    - 为每个$q_i$在所有的段落向量集中执行$\text{Top-K}$搜索，将每个$q_i$的$\text{Top-K}$段落向量相应位设为$1$  
> >    
> >      <img src="https://i-blog.csdnimg.cn/direct/3159c71bf0d94227b1e8412280a7e6f7.png" width=665 /> 
> >    
> >    - 将矩阵按段落拆分，就得到了段落的对齐矩阵 
> >      <img src="https://i-blog.csdnimg.cn/direct/e04a20e32dee4d3db4922de7e2f7b296.png" width=565 />   
> >      
> >      
> >      
> >    - 将每行被激活的子相似度的最大值相加，再除以归一化参数$Z$(即有几行有被激活的相似度)，得到最终的相似度评分 
> >      <img src="https://i-blog.csdnimg.cn/direct/865a5b343b2347a6b53b7768b1220c22.png" width=490 />   
> >      
> >    - 零处理机制：当一个段落所有$\text{Token}$都没有足够高相似度(对齐矩阵全$0$)，会将归一化参数$Z$设为很小的一个数避免除以$0$ 
> >    
> > 2. 与传统$\text{ColBERT}$训练的对比：还是回到原来的例子
> >    - $\text{ColBERT}$：不论$P^+$被选择与否，都会被给予很高的得分，导致模型最终无法正确选出$P^+$
> >    - $\text{XTR}$：极端情况如$P^+$的每个$\text{Token}$都不是$q_i$的$\text{Top-K}$，导致$P^+$被打零分造成高损失，迫使模型调整以能正确选择$P^+$
> >
> > :three:推理阶段：使用检索到的$\text{Token}$对段落进行评分
> >
> > 1. 获取候选段落：
> >    - $\text{MIPS}$检索：对所有$n$个查询向量$q_i$执行$\text{Top-}k^\prime$检索，得到$k^\prime$个最相似的段落$\text{Token}$
> >    - 回溯(但不收集)：回溯这$nk^\prime$个$\text{Token}$所属的段落，确定$C$个候选段落
> >    
> > 2. 相似度填充：
> >    - 排序：其检索$q_i$的$\text{Top-}k$为$p_{(1)},p_{(2)},...,p_{(k)}$，假设这些$\text{Token}$与$q_i$的相似度从高到低为$\left\langle{q_i,p_{(1)}}\rangle,...,\langle{q_i,p_{(k)}}\right\rangle$ 
> >    
> >      <img src="https://i-blog.csdnimg.cn/direct/e7c71aac337b4951a785d7d4c5ca55d3.png" width=663 />  
> >    
> >    - 填充：令被检索到的$\text{Token}$中的相似度最低者为$m_i\text{=}\langle{q_i,p_{(k)}}\rangle$，直接用$m_i$去填充一切其它(未被检索到$\text{Token}$)的相似度
> >    
> >       <img src="https://i-blog.csdnimg.cn/direct/cc78f7dc4b79476c8cc05a9908c0fa76.png" width=655 /> 
> >    
> >    - 得分：填充后再取每个段落相似度矩阵每行相似度的最大值相加，然后除以行数归一化，避免了某一行贡献的相似度为$0$ 
> >    
> >      <img src="https://i-blog.csdnimg.cn/direct/19fe0c8a759f42d1b2b56b5da2fd2b53.png" alt="image-20250323015940873" width=785 />   

# $\textbf{2. }$实验与分析

> ## $\textbf{2.1. }$实验配置与结果
>
> > :one:训练设置：在$\text{MS-MARCO}$上微调$\text{XTR}$，使用来自$\text{Rocket-QA}$的一组难负样本
> >
> > :two:检索结果：
> >
> > 1. 域内检索：$\text{XTR}$在$\text{MS-MARCO}$上$\text{nDCG@10}$优于绝大部分模型
> > 2. 零样本检索：即在未接触过的数据集上测试泛化能力，$\text{XTR}$表现优异，如在$\text{BEIR}$上达到了$\text{SOTA}$
> > 3. 多语言检索：无需二次训练，就能在$\text{MIRACL}$上取得良好的性能
>
> ## $\textbf{2.2. }$结果分析
>
> > :one:对$\text{Token}$检索的分析
> >
> > 1. 黄金$\text{Token}$：即来自$\text{Groumd Truth}$段落中的$\text{Token}$，实验证明$\text{XTR}$的检索结果要包含更多黄金$\text{Token}$
> > 2. 词法$\text{Token}$：即与查询$\text{Token}$相匹配的$\text{Token}$，而$\text{XTR}$检索结果中包含更少的词法$\text{Token}$，即降低了对词法匹配的依赖
> >
> > :two:对评分高效性的分析
> >
> > 1. 评分函数：传统的$\text{ColBERT}$模式需要加载所有的候选段落$\text{Token}$，而$\text{XTR}$只需检索到的$\text{Token}$是为一种简化方法
> > 2. 实验结果：让$\text{XTR}$的简化评分模式应用在传统$\text{ColBERT}$模型上时性能雪崩，但应用在$\text{XTR}$上时性能几乎没有损失
> >
> > :three:训练超参数分析：
> >
> > 1. 训练时的$\text{Top-}k_{\text{train}}$和推理时的$\text{Top-}{k^\prime}$
> >    - $\text{Top-}{k^\prime}$：当${k^\prime}$增大时$\text{XTR}$模型性能必将提升，因为更多$\text{Token}$能提供更完整的信息
> >    - $\text{Top-}k_{\text{train}}$：当然也是$k_{\text{train}}$越大越好，在于训练时适应了处理更多$\text{Token}$的能力
> >    - 二者间：用较小的$k_{\text{train}}$训练模型时，能在较小的${k^\prime}$下取得更好的推理效果，源于模型能学习如何高效利用有限$\text{Token}$ 
> > 2. 训练批次大小：$\text{XTR}$总体上偏好更大的训练批次，因为大批次能增加负样本数量，从而提升模型表现力
> >
> > :four:定性分析：一言以蔽之就是$\text{XTR}$更关注上下文相关性，$\text{ColBERT}$反而更依赖词汇匹配

# $\textbf{3. }$其它分析

> :one:复杂度分析：$\text{XTR}$和$\text{ColBERT}$都面临$O(nk^\prime)$个候选段落
>
> 1. $\text{ColBERT}$运行的复杂度
>    - 假设段落平均长度为$\bar{m}$，则需加载$O(\bar{m}nk^\prime)$个段落$\text{Token→}$进行$O(\bar{m}n^2k^\prime)$次内积
>    - 考虑到每次内积要$O(d)$次运算，所以总共需要$O(\bar{m}n^2dk^\prime)$次运算
> 2. $\text{XTR}$运行的复杂度：比$\text{ColBERT}$降低了约$\text{4000}$倍
>    - 不需要收集候选段落的所有$\text{Token}$，只需要收集每个段落中被检索到的$\text{Token}$，设平均每个段落中有$\bar{r}$个被检索到$\text{Token}$
>    - 然而考虑到这些被检索到的$\text{Token}$其内积相似度已经得到了，故不用管$O(d)$，最终需要进行$O(\bar{r}n^2k^\prime)$次运算
>
> :two:梯度分析：对于一批样本中的正负样本${P}^{-},{P}^{+}\text{∈}{P}_{1:B}\text{=}\left\lbrack{{P}_{1},\ldots ,{P}_{B}}\right\rbrack$
>
> 1. 梯度计算：对于正样本$P^+$中最大的$\text{Token}$相似度$P^{+}_{\hat{ij}}$，以及负样本$P^{-}$中最大的$\text{Token}$相似度$P^{-}_{\hat{ij}}$ 
>    |       模型       |                          正样本梯度                          |                          负样本梯度                          | 参数                                    |
>    | :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------------------------- |
>    | $\text{ColBERT}$ | $\cfrac{\partial\mathcal{L}_{CE}}{\partial{}P_{i\hat{\jmath}}^{+}}\text{=}–\cfrac{1}{m}\text{Pr}\left\{P^{+}\mid{}Q,P_{1:B}\right\}$ | $\cfrac{\partial\mathcal{L}_{CE}}{\partial{}P_{i\hat{\jmath}}^{-}}\text{=}\cfrac{1}{m}\text{Pr}\left\{P^{-}\mid{}Q,P_{1:B}\right\}$ | $m$是段落中所有$\text{Token}$的数量     |
>    |   $\text{XTR}$   | $\cfrac{\partial\mathcal{L}_{CE}}{\partial{}P_{i\hat{\jmath}}^{+}}\text{=}–\cfrac{1}{Z^{+}}\text{Pr}\left\{P^{+}\mid{}Q,P_{1:B}\right\}$ | $\cfrac{\partial\mathcal{L}_{CE}}{\partial{}P_{i\hat{\jmath}}^{-}}\text{=}\cfrac{1}{Z^{-}}\text{Pr}\left\{P^{-}\mid{}Q,P_{1:B}\right\}$ | $Z$是段落中被检索到$\text{Token}$的数量 |
> 2. 梯度分析：传统$\text{ColBERT}$正样本$\text{Token}$得分会很低，$\text{XTR}$改变了归一化策略，使得正样本的$\text{Token}$被迫获得更高的得分