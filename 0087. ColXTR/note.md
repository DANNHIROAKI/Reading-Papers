# $\textbf{1. ColXTR}$原理

> ## $\textbf{1.1. ColBERTv2}$概述
>
> > ### $\textbf{1.1.1. }$训练优化
> >
> > > :one:难负样本生成
> > >
> > > 1. 初筛：基于$\text{BM-25}$找到可能的负样本
> > > 2. 重排：使用$\text{KL}$散度将大型交叉编码器蒸馏进$\text{MiniLM}$，再用$\text{MiniLM}$保留负样本中的难负样本
> > >
> > > :two:高效训练结构
> > >
> > > 1. 多元组结构：从原来的结构$\langle q, {d}^+,{d}^\text{–}\rangle$，变成$\langle q, \textbf{d}_{w}\rangle\text{=}\langle q, {d}^+_{1},{d}^\text{–}_{2},...,{d}^\text{–}_{w}\rangle$结构
> > > 2. 批内负样本：除自身负样本外，还将批内其他查询的所有段落视为负样本，用来统一优化训练函数
> > >
> > > :three:降噪训练优化：
> > >
> > > 1. 样本刷新：定期用训练到一半的模型，重新生成训练样本
> > > 2. 防过拟合：定期刷新训练样本，以防止陷入局部最优
> >
> > ### $\textbf{1.1.2. }$整体流程
> >
> > > :one:残差压缩的原理
> > >
> > > 1. 聚类：对全部段落全部向量的集合$\{p_j\}$进行聚类，为每个向量$p_j$分配了一个质心$C_{p_j}$ 
> > > 2. 残差：就是$p_j$与其质心$C_{p_j}$的距离$r\text{=}p_j\text{–}C_{p_j}$ 
> > > 3. 压缩：将浮点数压缩为二进制表示，如[$r$每个维度的连续分布$\xrightarrow{分割为}4$个离散状态]$\xLeftrightarrow{对应}2$个$\text{bit}$所表示的四种状态
> > > 4. 编码：将每个$p_j$表示为质心$C_{p_j}\text{+}$压缩的残差，通过反向解压缩残差即可得到近似的$p_j$ 
> > >
> > > :two:离线索引流程
> > >
> > > 1. 近似质心：抽取一部分段落$\text{Token}$执行嵌入，再对部分的嵌入向量执行$\sqrt{n_{\text{embeddings}}}\text{-Means}$聚类，得到$\sqrt{n_{\text{embeddings}}}$个质心
> > > 2. 压缩编码：对所有的段落进行全精度嵌入==(但是不存储)==，基于上一步得到的质心进行残差压缩==(此时才对残差编码进行存储)==
> > > 3. 倒排索引：构建质心$\text{→}$质心所包含的嵌入的$\text{ID}$的列表，存储到磁盘中
> > >    ```txt
> > >    质心a -> 属于质心a的嵌入ID=a1, a2, a3, ...
> > >    质心b -> 属于质心b的嵌入ID=b1, b2, b3, ...
> > >    质心c -> 属于质心c的嵌入ID=c1, c2, c3, ...
> > >    ```
> > >
> > > :three:在线查询流程
> > >
> > > 1. 查询嵌入：原始查询$Q\xrightarrow[预处理(嵌入)]{\text{BERT}}$多向量表示$\{q_1, q_2, \dots, q_n\}$
> > > 2. 候选生成：查找与每个$q_i$最近的$n_{\text{probe}}\text{≥}1$个质心，收集每个质心下属的$\{p_j\}$集合，再收集与$\{p_j\}$有关的段落是为候选段落
> > > 3. 初筛流程：解压$\{p_j\}$中所有的向量，利用解压向量计算$Q$与候选段落的近似距离，这个近似距离时则为一个下界
> > >    ```txt
> > >    👉举个简单例子
> > >      Q: {q1, q2, q3}
> > >      P: {p1, p2, p3, p4} → {pj}集合中仅有{p1, p2, p3}
> > >    👉完整的距离计算
> > >      Maxsim-1-full = Max{<q1,p1>,<q1,p2>,<q1,p3>,<q1,p4>}
> > >      Maxsim-2-full = Max{<q2,p1>,<q2,p2>,<q2,p3>,<q2,p4>}
> > >      Maxsim-3-full = Max{<q3,p1>,<q3,p2>,<q3,p3>,<q3,p4>}
> > >    👉近似的距离计算
> > >      Maxsim-1-part = Max{<q1,p1>,<q1,p2>,<q1,p3>} ≤ Maxsim-1-full
> > >      Maxsim-2-part = Max{<q2,p1>,<q2,p2>,<q2,p3>} ≤ Maxsim-2-full
> > >      Maxsim-3-part = Max{<q3,p1>,<q3,p2>,<q3,p3>} ≤ Maxsim-3-full
> > >    👉所以一定是下界
> > >    ```
> > > 4. 重排流程：根据初筛结果选取若干最相似的段落，解压得到这些段落的全部向量，计算精确的相似度以得到最终结果
>
> ## $\textbf{1.2. XTR}$概述
>
> > ### $\textbf{1.2.1. }$研究的动机
> >
> > > :one:重新定义多向量相似度问题：$\displaystyle\text{ColBERT}(Q,P)\text{=}\frac{1}{Z} \sum_{i=1}^{n} \sum_{j=1}^{m}(\textbf{P}\text{∘}\textbf{A})_{ij}$
> > >
> > > 1. 数据结构：
> > >    - 评分矩阵$\textbf{S}$：令查询$Q\text{=}\left\{q_{1},q_2,\ldots,q_{n}\right\}$以及文档$P\text{=}\{p_1,p_2,...,p_m\}$，记子内积为$s_{ij}\text{=}{q}_{i}^{\top}{p}_{j}$，由此构成$\textbf{S}\text{∈}\mathbb{R}^{n\text{×}m}$ 
> > >    - 对齐矩阵$\textbf{A}$：让每个元素$a_{ij}\text{∈}\{0,1\}$来对$\textbf{P}$中的元素进行不同强度的选择，由此构成$\textbf{A}\text{∈}\mathbb{R}^{n\text{×}m}$ 
> > > 2. $\text{ColBERT}$版本，通过调整对齐矩阵$\textbf{A}$，让其选择评分矩阵$\textbf{S}$每行最大的一个值，最后除以$Z$归一化
> > >    <img src="https://i-blog.csdnimg.cn/direct/c9aa508343f84d7d8c7fb7c73393c418.png" alt="image-20250306233530006" width=800 />   
> > > 3. 传统的训练方式：最大化批内正样本${P}^{+}\text{∈}{P}_{1:B}\text{=}\left\lbrack{{P}_{1},\ldots ,{P}_{B}}\right\rbrack$的得分，即最小化${\mathcal{L}}_{\mathrm{{CE}}}\textbf{= }–\log\cfrac{e^{\text{ColBERT}\left( {Q,{P}_{b}}\right)}}{\displaystyle{}\sum_{{b\textbf{=}1}}^{B}e^{\text{ColBERT}\left( {Q,{P}_{b}}\right)}}$ 
> > >
> > > :two:传统$\text{ColBERT}$的流程
> > >
> > > <img src="https://i-blog.csdnimg.cn/direct/acf6d1caa7a14872b6db3d6034254900.png" alt="image-20250313215730217" width=800 />   
> > >
> > > 1. $\text{Token}$检索：用查询单向量集中每个$q_i$检索$k^\prime$个段落$\text{Token}$，最多产生$n\text{×}k^\prime$个候选$\text{Token}$
> > > 2. 收集向量：加载$n\text{×}k^\prime$个候选$\text{Token}$所属的段落，收集这些段落中所有的$\text{Token}$向量
> > > 3. 评分与重排：对这些段落应用全精度的$\text{ColBERT}$非线性相似度以进行重排
> > >
> > > :three:$\text{XTR}$的动机
> > >
> > > 1. 传统$\text{ColBERT}$面临的问题
> > >    - 训练上：与推理不一致，传统$\text{ColBERT}$的旨在优化最终$\text{ColBERT}$评分，而推理过程旨在获得$\text{Top-}k$的$\text{Token}$
> > >    - 开销上：收集$\text{Top-}k$候选段落的多有$\text{Token}$空间开销巨大，由此后续精确距离的计算成本也巨大
> > >    - 泛化上：$\text{ColBERT}$的评分函数是非线性的，阻碍了使用$\text{MIPS}$进行检索
> > > 2. $\text{XTR}$的改进
> > >    - 训练阶段：重新设计了训练目标函数，使得模型能优先检索出最有价值的段落$\text{Token}$
> > >    - 重排阶段：完全省去回溯(收集)步骤，直接只用检索到的段落$\text{Token}$来构成
> > >    - 缺失补充：只考虑检索到的$\text{Token}$难免漏掉相关的$\text{Token}$，故$\text{XTR}$还会对缺失$\text{Token}$进行自动评分 
> >
> > ### $\textbf{1.2.2. }$模型训练
> >
> > > :one:批内$\text{Token}$检索的训练策略
> > >
> > > 1. 给定一个查询$Q\text{=}\{q_1,...,q_n\}$和一批共$B$个段落向量$P^{(i)}\text{=}\{p_1^{(i)},...,p_m^{(i)}\}$ 
> > >    <img src="https://i-blog.csdnimg.cn/direct/6e92a77acd8b435a80dff8cefdc940ce.png" width=550 /> 
> > > 2. 为每个$q_i$在所有的段落向量集中执行$\text{Top-K}$搜索，将每个$q_i$的$\text{Top-K}$段落向量相应位设为$1$  
> > >    <img src="https://i-blog.csdnimg.cn/direct/3159c71bf0d94227b1e8412280a7e6f7.png" width=665 /> 
> > > 3. 将矩阵按段落拆分，就得到了段落的对齐矩阵 
> > >    <img src="https://i-blog.csdnimg.cn/direct/e04a20e32dee4d3db4922de7e2f7b296.png" width=565 /> 
> > > 4. 将每行被激活的子相似度的最大值相加，再除以归一化参数$Z$(即有几行有被激活的相似度)，得到最终的相似度评分 
> > >    <img src="https://i-blog.csdnimg.cn/direct/865a5b343b2347a6b53b7768b1220c22.png" width=490 />   
> > > 5. 零处理机制：当一个段落所有$\text{Token}$都没有足够高相似度(对齐矩阵全$0$)，会将归一化参数$Z$设为很小的一个数避免除以$0$ 
> > >
> > > :two:与传统$\text{ColBERT}$训练的对比：还是回到原来的例子
> > >
> > > 1. $\text{ColBERT}$：不论$P^+$被选择与否，都会被给予很高的得分，导致模型最终无法正确选出$P^+$
> > >2. $\text{XTR}$：极端情况如$P^+$的每个$\text{Token}$都不是$q_i$的$\text{Top-K}$，导致$P^+$被打零分造成高损失，迫使模型调整以能正确选择$P^+$ 
> >
> > ### $\textbf{1.2.3. }$推理阶段
> >
> > > :one:获取候选文档：
> > >
> > > 1. $\text{MIPS}$检索：对所有$n$个查询向量$q_i$执行$\text{Top-}k^\prime$检索，得到$k^\prime$个最相似的段落$\text{Token}$
> > > 2. 回溯(但不收集)：回溯这$nk^\prime$个$\text{Token}$所属的文档，确定$C$个候选文档
> > >
> > > :two:相似度填充：
> > >
> > > 1. 排序：其检索$q_i$的$\text{Top-}k$为$p_{(1)},p_{(2)},...,p_{(k)}$，假设这些$\text{Token}$与$q_i$的相似度从高到低为$\left\langle{q_i,p_{(1)}}\rangle,...,\langle{q_i,p_{(k)}}\right\rangle$ 
> > >    <img src="https://i-blog.csdnimg.cn/direct/e7c71aac337b4951a785d7d4c5ca55d3.png" width=663 /> 
> > > 2. 填充：令被检索到的$\text{Token}$中的相似度最低者为$m_i\text{=}\langle{q_i,p_{(k)}}\rangle$，直接用$m_i$去填充一切其它(未被检索到$\text{Token}$)的相似度
> > >    <img src="https://i-blog.csdnimg.cn/direct/cc78f7dc4b79476c8cc05a9908c0fa76.png" width=655 /> 
> > > 3. 评分：填充后再取每个段落相似度矩阵每行相似度的最大值相加，然后除以行数归一化，避免了某一行贡献的相似度为$0$ 
> > >
> > >    <img src="https://i-blog.csdnimg.cn/direct/19fe0c8a759f42d1b2b56b5da2fd2b53.png" alt="image-20250323015940873" width=785 /> 
>
> ## $\textbf{1.3. ColXTR}$原理
>
> > :one:$\text{ColXTR}$概述：集成$\text{ColBERTv2}$的优化来增强$\text{XTR}$
> >
> > 1. 训练阶段：
> >    - 保留$\text{XTR}$训练目标：即仍然采用$\text{Token}$检索级优化，而非段落级评分的优化
> >    - 增加降维投影层：训练过程中引入降维投影，试图降低每个$\text{Token}$的向量的维度 
> > 2. 推理阶段：
> >    - $\text{XTR}$：直接使用$\text{ScaNN}$库存储精确的向量，不进行任何压缩，从而使得空间需求巨大
> >    - $\text{ColXTR}$：引入了$\text{ColBERTv2}$的残差压缩机制，大幅降低存储需求
> >
> > :two:$\text{ColXTR}$索引：全盘采纳$\text{ColBERTv2}$的三阶段索引，并用$\text{T5}$编码器(和[$\text{Aligner}$](https://dannhiroaki.blog.csdn.net/article/details/146194192)一样)嵌入
> >
> > 1. 近似质心：抽取一部分段落$\text{Token}$执行嵌入，再对部分的嵌入向量执行$\sqrt{n_{\text{embeddings}}}\text{-Means}$聚类，得到$\sqrt{n_{\text{embeddings}}}$个质心
> > 2. 压缩编码：对所有的段落进行全精度嵌入==(但是不存储)==，基于上一步得到的质心进行残差压缩==(此时才对残差编码进行存储)==
> > 3. 倒排索引：构建质心$\text{→}$质心所包含的嵌入的$\text{ID}$的列表，存储到磁盘中
> >
> >    ```txt
> >    质心a -> 属于质心a的嵌入ID=a1, a2, a3, ...
> >    质心b -> 属于质心b的嵌入ID=b1, b2, b3, ...
> >    质心c -> 属于质心c的嵌入ID=c1, c2, c3, ...
> >    ```
> >
> > :three:$\text{ColXTR}$查询：假设查询$Q$被编码为多向量表示$\{q_1,q_2,...,q_n\}$ 
> >
> > 1. 候选生成：原始$\text{XTR}$直接对$\{p_j\}$进行$\text{MIPS}$搜索生成候选段落，$\text{ColXTR}$先对质心进行$\text{MIPS}$搜索，再倒排索引回候选段落
> >    <img src="https://i-blog.csdnimg.cn/direct/e474bcbc35854a8eb570cd17af44e2de.png" width=550 /> 
> >    - 质心搜索：计算所有$q_i$与所有质心$c_j$的相似度，确定每个$q_i$的$\text{Top-}k$个质心，合并后得到候选质心集$\{c_j\}$
> >    - 倒排索引：通过倒排索引，将候选质心回溯得到与每个质心有关的段落嵌入，是为候选向量集$\{p_j\}$
> >    - 候选生成：找到所有与$\{p_j\}$种向量有关的段落，从而构成候选段落集$\{P_1,P_2,...,P_N\}$  
> > 2. 内积计算：解压以获得$\{p_j\}$中所有向量的全精度表示，让查询向量集$\{q_i\}$和段落向量集$\{p_j\}$中向量两两算内积$\langle{q_i,p_j}\rangle$ 
> >    <img src="https://i-blog.csdnimg.cn/direct/30edd682e8d74597b21790937b810609.png" width=550 />  
> > 3. 相似填充：用每行最小的相似度值，填充每行剩余的相似度值，最终输出每个候选段落每行最大值的平均(即为近似相似度)
> >    <img src="https://i-blog.csdnimg.cn/direct/9a2adbda000f496085bcf6efd84689ee.png" width=550 />  
> > 4. 段落重排：选取若干近似评分最高的段落，解压其所有的向量从而计算精确的相似度，即可得到精确的最相似段落

# $\textbf{2. }$实验及结果

> :one:实验设置
>
> 1. 模型设置：在$\text{MS-MARCO}$上对$\text{T5-base}$编码器进行微调，并在顶层设置一个$\text{768→128}$的投影层
> 2. 训练设置：设置$\text{XTR}$中训练的$k\text{=}320$，一批样本$\text{48}$个(其中难负样本由$\text{BM25}$挖掘出)，训练$\text{50K}$步
> 3. 检索设置：让每个$q_i$先探测$\text{10}$个质心，对小索引设置倒排索引到$k\text{=}500$个$p_j/$大索引则$k\text{=}10000$个
>
> :two:实验结果
>
> 1. 性能上：$\text{ColXTR}$比$\text{COlBERT}$和$\text{XTR}$都要差，我不理解这篇文章为什么要重新训练$\text{ColBERT}$，他妈性能当然比不过啊
> 2. 开销上：你都他妈压缩了开销当然小啊，性能又打不过，这种文章还能发$\text{COLING'25}$奶奶的
>
> :three:优化分析
>
> 1. 这一部分内容也相当无聊
> 2. 就是啊我们的优化，哎哟和$\text{XTR}$一样降低了推理开销，那个和$\text{ColBERTv2}$一样降低了存储开销
> 3. 所以$\text{ColXTR}$性能连传统$\text{ColBERT}$都不如，但我们就是牛逼
