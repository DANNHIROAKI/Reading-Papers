# 1. 导论

:one:所面临的问题：$\text{ColBERT}$的评分函数是非线性的，阻碍了使用$\text{MIPS}$进行检索

:two:传统$\text{ColBERT}$的流程

<img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250313215730217.png" alt="image-20250313215730217" width=760 /> 

1. $\text{Token}$检索：用查询单向量集中每个$q_i$检索$k^\prime$个段落$\text{Token}$，最多产生$n\text{×}k^\prime$个候选$\text{Token}$
2. 收集向量：加载$n\text{×}k^\prime$个候选$\text{Token}$所属的段落，收集这些段落中所有的向量
3. 评分与重排：对这些段落应用全精度的$\text{ColBERT}$相似度以进行重排

:three:以上流程所面临的问题

1. 运行开销上：收集$\text{Top-K}$候选段落的多有$\text{Token}$空间开销巨大，由此后续精确距离的计算成本也巨大
2. 检索性能上：传统$\text{ColBERT}$的训练目标是为最终$\text{ColBERT}$评分优化的，而不是为获得$\text{Top-K}$候选文档(或者说$\text{Token}$)而优化的 

:three:本文$\text{XTR}$的改进

1. 检索阶段：通过一种新的训练目标，训练一个模型，专门用于检索出最有价值的段落$\text{Token}$
2. 重排阶段：完全省去回溯(收集)步骤，直接只用检索到的段落$\text{Token}$来构成
3. 该方法相当激进，只考虑检索到的$\text{Token}$难免漏掉相关的$\text{Token}$，因此本文还有一种缺失相似度插补方法，考虑缺失$\text{Token}$对总体得分的贡献

# 2. 背景

:one:多向量相似度的问题定义：$\displaystyle\text{ColBERT}(Q,P)\text{=}\frac{1}{Z} \sum_{i=1}^{n} \sum_{j=1}^{m}(\textbf{P}\text{∘}\textbf{A})_{ij}$

1. 数据结构

   - 评分矩阵$\textbf{P}$：令查询$Q\text{=}\left\{q_{1},q_2,\ldots,q_{n}\right\}$以及文档$P\text{=}\{p_1,p_2,...,p_m\}$中，记子内积为$s_{ij}\text{=}{q}_{i}^{\top}{p}_{j}$，由此构成$\textbf{P}\text{∈}\mathbb{R}^{n\text{×}m}$ 
   - 对齐矩阵$\textbf{A}$：让每个元素$a_{ij}\text{∈}\{0,1\}$来对$\textbf{P}$中的元素进行不同强度的选择，由此构成$\textbf{A}\text{∈}\mathbb{R}^{n\text{×}m}$ 

2. $\text{ColBERT}$版本，通过调整对齐矩阵$\textbf{A}$，让其选择评分矩阵$\textbf{P}$每行最大的一个值，最后除以$Z$归一化

   <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250311002609291.png" alt="image-20250306233530006" width=800 />

3. 传统的训练方式：最大化批内正样本得分，即最小化${\mathcal{L}}_{\mathrm{{CE}}}\textbf{= }–\log \cfrac{e^{\text{ColBERT}\left( {Q,{P}_{b}}\right)}}{\displaystyle{}\sum_{{b\textbf{=}1}}^{B}e^{\text{ColBERT}\left( {Q,{P}_{b}}\right)}}$其中${P}^{+}\text{∈}{P}_{1:B}\text{=}\left\lbrack{{P}_{1},\ldots ,{P}_{B}}\right\rbrack$

:two:多向量的三阶段推理：上面已经讲的很清楚了

# 3. XTR

:one:为何传统$\text{ColBERT}$训练方式对$\text{Token}$检索效果不佳

1. 传统训练方法力求得到最大化相关样本的$\text{ColBERT}(Q,P^+)$得分，关注的只是段落级别的得分，而非$\text{Token}$级别的检索质量
2. 举例说明：
   - 假设正段落$P^+$所有$\text{Token}$与$q_i$的相似度均为$0.8$，则最终段落$P^+$评分为$0.8$ 
   - 假设负段落$P^-$大多$\text{Token}$与$q_i$的相似度均为$0$，但少数几个$\text{Token}$与$q_i$相似度超过$0.8$，导致最终文档得分仍然有$0.2$ 
3. 训练和推理的偏差
   - 训练时：认为得分越高者为越好，即会选择$P^+$
   - 推理是：推理的任务是检索与$q_i$最相似的段落$\text{Token}$，如果进行$\text{Top-1}$检索则最终会选择$P^-$

:two:训练阶段：批内$\text{Token}$检索的训练策略

1. 给定一个查询$Q\text{=}\{q_1,...,q_n\}$和一批共$B$个段落向量$P^{(i)}\text{=}\{p_1^{(i)},...,p_m^{(i)}\}$ 

   <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250314004847495.png" alt="image-20250314004847495" width=600 />    

2. 为每个$q_i$在所有的段落向量集中执行$\text{Top-K}$搜索，将每个$q_i$的$\text{Top-K}$段落向量相应位设为$1$

   <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250314005036691.png" alt="image-20250314005036691" width=728 />  

3. 将矩阵按段落拆分，就得到了段落的对齐矩阵

   <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250314005634798.png" alt="image-20250314005634798" width=620 />  

4. 将每行被激活的子相似度的最大值相加，再除以归一化参数$Z$(即有几行有被激活的相似度)，得到最终的相似度评分

   <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250314010150750.png" alt="image-20250314010150750" width=530 /> 

5. 零检索处理机制：当一个段落的所有$\text{Token}$都没有足够高相似度，以至于其对齐矩阵全$0$时，会将归一化参数$Z$设为很小的一个数避免除以$0$

6. 回到之前的例子

   - 假设正段落$P^+$所有$\text{Token}$与$q_i$的相似度均为$0.8$，则最终段落$P^+$评分为$0.8$ 
   - 假设负段落$P^-$大多$\text{Token}$与$q_i$的相似度均为$0$，但少数几个$\text{Token}$与$q_i$相似度超过$0.8$，导致最终文档得分仍然有$0.2$ 
   - 原始的$\text{ColBERT}$评分中不论$P^+$被选择与否，都会被给予很高的得分，导致模型最终无法正确选出$P^+$
   - 在$\text{XTR}$评分中，考虑一种情况就是$P^+$的每个$\text{Token}$都不是$q_i$的$\text{Top-K}$，最终导致$P^+$被打零分。从而在训练时产生高损失，从而迫使模型调整，以至于最后能正确地选择$P^+$

:three:推理阶段：使用检索到的$\text{Token}$对文档进行评分

1. 获取候选文档：对所有$n$个查询向量$q_i$执行$\text{Top-}k^\prime$检索，得到$k^\prime$个最相似的段落$\text{Token}$，回溯这$nk^\prime$个$\text{Token}$所属的文档，是为$C$个候选文档

2. 相似度填充：

   - 对于$q_i$，其检索到的$\text{Top-}k$为$p_{(1)},p_{(2)},...,p_{(k)}$，假设这些$\text{Token}$与$q_i$的相似度从高到低为$\langle{q_i,p_{(1)}}\rangle,...,\langle{q_i,p_{(k)}}\rangle$ 
   - 令被检索到的$\text{Token}$中的相似度最低者为$m_i\text{=}\langle{q_i,p_{(k)}}\rangle$  

   - 区别于传统方法调集所有的段落$\text{Token}$再去计算除了$\langle{q_i,p_{(1)}}\rangle,...,\langle{q_i,p_{(k)}}\rangle$以外的相似度
   - 这里的做法是用$m_i$去填充一切其它的相似度
   - 填充后再取每行相似度的最大值相加，然后除以行数归一化，避免了某一行贡献的相似度为$0$
   - 这与训练策略中忽略掉没有$\text{Top-k}$的行有所不同，可以放心大胆的直接除以固定的行数
