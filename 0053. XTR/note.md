# 1. 导论

:two:传统$\text{ColBERT}$的流程

<img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250313215730217.png" alt="image-20250313215730217" width=760 /> 

1. $\text{Token}$检索：用查询单向量集中每个$q_i$检索$k^\prime$个段落$\text{Token}$，最多产生$n\text{×}k^\prime$个候选$\text{Token}$
2. 收集向量：加载$n\text{×}k^\prime$个候选$\text{Token}$所属的段落，收集这些段落中所有的$\text{Token}$向量
3. 评分与重排：对这些段落应用全精度的$\text{ColBERT}$非线性相似度以进行重排

:three:以上流程所面临的问题

1. 训练上：训练推理不一致，传统$\text{ColBERT}$的训练目标是为最终$\text{ColBERT}$评分优化的，而推理过程旨在获得$\text{Top-K}$的$\text{Token}$
2. 开销上：收集$\text{Top-K}$候选段落的多有$\text{Token}$空间开销巨大，由此后续精确距离的计算成本也巨大
3. 泛化上：$\text{ColBERT}$的评分函数是非线性的，阻碍了使用$\text{MIPS}$进行检索

:three:本文$\text{XTR}$的改进

1. 训练阶段：重新设计了训练目标函数，使得模型能优先检索出最有价值的段落$\text{Token}$
2. 重排阶段：完全省去回溯(收集)步骤，直接只用检索到的段落$\text{Token}$来构成
3. 缺失补充：只考虑检索到的$\text{Token}$难免漏掉相关的$\text{Token}$，因此本文还有一种缺失相似度插补方法，考虑缺失$\text{Token}$对总体得分的贡献

# 2. 背景

:one:多向量相似度的问题定义：$\displaystyle\text{ColBERT}(Q,P)\text{=}\frac{1}{Z} \sum_{i=1}^{n} \sum_{j=1}^{m}(\textbf{P}\text{∘}\textbf{A})_{ij}$

1. 数据结构

   - 评分矩阵$\textbf{P}$：令查询$Q\text{=}\left\{q_{1},q_2,\ldots,q_{n}\right\}$以及文档$P\text{=}\{p_1,p_2,...,p_m\}$，记子内积为$s_{ij}\text{=}{q}_{i}^{\top}{p}_{j}$，由此构成$\textbf{P}\text{∈}\mathbb{R}^{n\text{×}m}$ 
   - 对齐矩阵$\textbf{A}$：让每个元素$a_{ij}\text{∈}\{0,1\}$来对$\textbf{P}$中的元素进行不同强度的选择，由此构成$\textbf{A}\text{∈}\mathbb{R}^{n\text{×}m}$ 

2. $\text{ColBERT}$版本，通过调整对齐矩阵$\textbf{A}$，让其选择评分矩阵$\textbf{S}$每行最大的一个值，最后除以$Z$归一化

   <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250311002609291.png" alt="image-20250306233530006" width=800 />

3. 传统的训练方式：最大化批内正样本${P}^{+}\text{∈}{P}_{1:B}\text{=}\left\lbrack{{P}_{1},\ldots ,{P}_{B}}\right\rbrack$的得分，即最小化${\mathcal{L}}_{\mathrm{{CE}}}\textbf{= }–\log\cfrac{e^{\text{ColBERT}\left( {Q,{P}_{b}}\right)}}{\displaystyle{}\sum_{{b\textbf{=}1}}^{B}e^{\text{ColBERT}\left( {Q,{P}_{b}}\right)}}$

:two:多向量的三阶段推理：上面已经讲的很清楚了

# 3. XTR

:one:为何传统$\text{ColBERT}$训练方式对$\text{Token}$检索效果不佳

1. 传统训练方法力求得到最大化相关样本的$\text{ColBERT}(Q,P^+)$得分，关注的只是段落级别的得分，而非$\text{Token}$级别的检索质量
2. 举例说明：
   - 假设正段落$P^+$所有$\text{Token}$与$q_i$的相似度均为$0.8$，则最终段落$P^+$评分为$0.8$ 
   - 假设负段落$P^-$大多$\text{Token}$与$q_i$的相似度均为$0$，但少数几个$\text{Token}$与$q_i$相似度超过$0.8$，导致最终文档得分仍然有$0.2$ 
3. 训练和推理的偏差
   - 训练时：认为得分越高者为越好，即会选择$P^+$
   - 推理时：推理的任务是检索与$q_i$最相似的段落$\text{Token}$，如果进行$\text{Top-1}$检索则最终会回溯到选择$P^-$

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

1. 获取候选文档：对所有$n$个查询向量$q_i$执行$\text{Top-}k^\prime$检索，得到$k^\prime$个最相似的段落$\text{Token}$，回溯这$nk^\prime$个$\text{Token}$所属的文档，确定$C$个候选文档

2. 相似度填充：

   - 对于$q_i$，其检索到的$\text{Top-}k$为$p_{(1)},p_{(2)},...,p_{(k)}$，假设这些$\text{Token}$与$q_i$的相似度从高到低为$\left\langle{q_i,p_{(1)}}\rangle,...,\langle{q_i,p_{(k)}}\right\rangle$ 
   - 令被检索到的$\text{Token}$中的相似度最低者为$m_i\text{=}\langle{q_i,p_{(k)}}\rangle$  

   - 区别于传统方法调集所有的段落$\text{Token}$再去计算除了$\langle{q_i,p_{(1)}}\rangle,...,\langle{q_i,p_{(k)}}\rangle$以外的相似度
   - 这里的做法是用$m_i$去填充一切其它的相似度
   - 填充后再取每行相似度的最大值相加，然后除以行数归一化，避免了某一行贡献的相似度为$0$
   - 这与训练策略中忽略掉没有$\text{Top-k}$的行有所不同，可以放心大胆的直接除以固定的行数

# 4. 实验

:one:训练：在$\text{MS-MARCO}$上微调$\text{XTR}$，使用来自$\text{Rocket-QA}$的一组难负样本​

:two:域内检索：$\text{XTR}$在$\text{MS-MARCO}$上$\text{nDCG@10}$优于绝大部分模型

:three:零样本检索：即在未接触过的数据集上测试泛化能力，$\text{XTR}$表现优异，如在$\text{BEIR}$上达到了$\text{SOTA}$

:four:多语言检索：无需二次训练，就能在$\text{MIRACL}$上取得良好的性能

# 5. 分析

:one:对$\text{Token}$检索的分析

1. 黄金$\text{Token}$：即来自$\text{Groumd Truth}$段落中的$\text{Token}$，实验证明$\text{XTR}$的检索结果要包含更多黄金$\text{Token}$
2. 词法$\text{Token}$：即与查询$\text{Token}$相匹配的$\text{Token}$，相比之下$\text{XTR}$检索结果中包含更少的词法$\text{Token}$，即降低了对词法匹配的依赖

:two:对评分高效性的分析

1. 评分函数：传统的$\text{ColBERT}$模式需要加载所有的候选段落$\text{Token}$，而$\text{XTR}$只需检索到的$\text{Token}$是为一种简化方法
2. 实验结果：让$\text{XTR}$的简化评分模式应用在传统$\text{ColBERT}$模型上时性能雪崩，但应用在$\text{XTR}$上时性能几乎没有损失

:three:训练超参数分析：

1. 训练时的$\text{Top-}k_{\text{train}}$和推理时的$\text{Top-}{k^\prime}$
   - $\text{Top-}{k^\prime}$：当${k^\prime}$增大时$\text{XTR}$模型性能必将提升，因为更多$\text{Token}$能提供更完整的信息
   - $\text{Top-}k_{\text{train}}$：当然也是$k_{\text{train}}$越大越好，在于训练时适应了处理更多$\text{Token}$的能力
   - 二者间：当用较小的$k_{\text{train}}$下训练模型时，能在较小的${k^\prime}$下取得更好的推理效果，源于模型能学习如何高效利用有限$\text{Token}$ 
2. 训练批次大小：$\text{XTR}$总体上偏好更大的训练批次，因为大批次能增加负样本数量，从而提升模型表现力

:four:定性分析：一言以蔽之就是$\text{XTR}$更关注上下文相关性，$\text{ColBERT}$反而更依赖词汇匹配

# 6. 其它

:one:复杂度分析：$\text{XTR}$和$\text{ColBERT}$都面临$O(nk^\prime)$个候选文档

1. $\text{ColBERT}$运行的复杂度
   - 假设文档平均长度为$\bar{m}$，则需加载$O(\bar{m}nk^\prime)$个段落$\text{Token→}$进行$O(\bar{m}n^2k^\prime)$次内积
   - 考虑到每次内积要$O(d)$次运算，所以总共需要$O(\bar{m}n^2dk^\prime)$次运算
2. $\text{XTR}$运行的复杂度：比$\text{ColBERT}$降低了约$\text{4000}$倍
   - 不需要收集候选段落的所有$\text{Token}$，只需要收集每个段落中被检索到的$\text{Token}$，设平均每个段落中有$\bar{r}$个被检索到$\text{Token}$
   - 然而考虑到这些被检索到的$\text{Token}$其内积相似度已经得到了，故不用管$O(d)$，最终需要进行$O(\bar{r}n^2k^\prime)$次运算

:two:梯度分析：对于一批样本中的正负样本${P}^{-},{P}^{+}\text{∈}{P}_{1:B}\text{=}\left\lbrack{{P}_{1},\ldots ,{P}_{B}}\right\rbrack$

1. 梯度计算：对于正样本$P^+$中最大的$\text{Token}$相似度$P^{+}_{\hat{ij}}$，以及负样本$P^{-}$中最大的$\text{Token}$相似度$P^{-}_{\hat{ij}}$ 

   |       模型       |                          正样本梯度                          |                          负样本梯度                          | 参数                                    |
   | :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | --------------------------------------- |
   | $\text{ColBERT}$ | $\cfrac{\partial\mathcal{L}_{CE}}{\partial{}P_{i\hat{\jmath}}^{+}}\text{=}–\cfrac{1}{m}\text{Pr}\left\{P^{+}\mid{}Q,P_{1:B}\right\}$ | $\cfrac{\partial\mathcal{L}_{CE}}{\partial{}P_{i\hat{\jmath}}^{-}}\text{=}\cfrac{1}{m}\text{Pr}\left\{P^{-}\mid{}Q,P_{1:B}\right\}$ | $m$是段落中所有$\text{Token}$的数量     |
   |   $\text{XTR}$   | $\cfrac{\partial\mathcal{L}_{CE}}{\partial{}P_{i\hat{\jmath}}^{+}}\text{=}–\cfrac{1}{Z^{+}}\text{Pr}\left\{P^{+}\mid{}Q,P_{1:B}\right\}$ | $\cfrac{\partial\mathcal{L}_{CE}}{\partial{}P_{i\hat{\jmath}}^{-}}\text{=}\cfrac{1}{Z^{-}}\text{Pr}\left\{P^{-}\mid{}Q,P_{1:B}\right\}$ | $Z$是段落中被检索到$\text{Token}$的数量 |

2. 梯度分析：传统$\text{ColBERT}$正样本$\text{Token}$得分会很低，$\text{XTR}$改变了归一化策略，使得正样本的$\text{Token}$被迫获得更高的得分
