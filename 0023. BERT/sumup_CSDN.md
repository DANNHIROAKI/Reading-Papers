
@[toc]
[原论文: BERT, Pre-training of Deep Bidirectional Transformers for Language Understanding](https://doi.org/10.48550/arXiv.1810.04805) 

[预备知识: Transformer与注意力机制的概述](https://blog.csdn.net/qq_64091900/article/details/144035750)

---
# $\textbf{1. }$概述与综述

> ## $\textbf{1.1. }$研究背景
>
> > :one:上下文无关$\textbf{\&}$上下文敏感 
> >
> > |      模式      | 含义                                                         | 有关模型                 |
> > | :------------: | :----------------------------------------------------------- | :----------------------- |
> > | 上下文无关表示 | 词元$x\xleftrightarrow{一一对应}$预训练向量$f(x)$            | $\text{word2vec/GloVe}$  |
> > | 上下文敏感表示 | 词元$x\xleftrightarrow[上下文c(x)]{一一对应}$预训练向量$f(x,c(x))$ | $\text{TagLM/CoVe/ELMo}$ |
> >
> > :two:语言模型的预训练
> >
> > 1. 含义：在大量未标注的文本数据上训练模型$\xrightarrow{}$对语言结构$/$词义$/$上下文有初步的理解
> > 2. 方式：常见的预训练任务
> >    |           方式           | 含义                                               |     示例      |
> >    | :----------------------: | :------------------------------------------------- | :-----------: |
> >    |      自回归语言模型      | (从左到右)给定一部分文本时预测下一词的概率         | $\text{GPT}$  |
> >    | 完形填空($\text{Cloze}$) | 通过遮蔽部分输入词(`[MASK]`标记)用于后续预测与填充 | $\text{BERT}$ |
> >
> > :three:预训练模型如何用于下游任务
> >
> > 1. 基于特征的方法：例如$\text{ELMo}$ 
> >    <img src="https://i-blog.csdnimg.cn/direct/c5941d2ab9474d82a022536575d4d075.png" alt="image-20241127195139710" width=550 /> 
> >    - 原理：预训练时用$\text{BiLSTM}$生成动态上下文表示$\text{→}$在下游与$\text{word2vec}$生成的静态特征拼接
> >    - 特点：需要为每个下游任务设计专门的模型结构
> > 2. 微调($\text{Fine-tuning}$)方法：例如$\text{GPT}$ 
> >    <img src="https://i-blog.csdnimg.cn/direct/b3cc16e40338437ab4f1c8bc2d719f5f.png" alt="image-20241127195228188" width=250 /> 
> >    - 原理：预训练阶段只添加很少特定参数$\text{→}$在下游任务上对整个预训练模型进行训练
> >    - 特点：为单向模型(只能考虑前文信息从而由左到右生成)
>
> ## $\textbf{1.2. BERT}$的主要贡献
>
> >  <img src="https://i-blog.csdnimg.cn/direct/505d9419b1684e2492108550b2e5a8ea.png" alt="qewfrgdf片1" width=600 /> 
> >
> >  :one:引入掩码语言模型：
> >
> >  1. 原理：预训练过程中，将部分词随机替换为`<MASK>`，然后预测这些被掩盖的原始词汇
> >  2. 优势：许模型同时融合左侧和右侧的上下文$\text{→}$从而预训练深度双向的$\text{Transformer}$ 
> >
> >  :two:引入下一句预测($\text{NSP}$)：
> >
> >  1. 原理：预训练时，模型需要预测两句话是否在原始文本中相邻
> >  2. 优势：使$\text{BERT}$能更好地理解句子间的关系(如$\text{Q\&A}$间的关系)
> >
> >  :three:证明了广泛适用性：
> >
> >  1. 原理：$\text{BERT}$预训练的深度双向表示，可以通过微调应用于各种下游任务
> >  2. 优势：$\text{BERT}$在大量句级$/$词级任务性能，超过许多为特定任务专门设定的架构

# $\textbf{2. BERT}$原理: 预训练$\textbf{\&}$微调  

> ## $\textbf{2.1. BERT}$的结构
>
> > :one:模型的两个阶段  
> >
> > |  阶段  | 主要任务                                                     |
> > | :----: | :----------------------------------------------------------- |
> > | 预训练 | 在不同任务上用无标注数据训练模型                             |
> > |  微调  | 用==相同的==预训练参数初始化$\text{→}$用下游任务中的有标注数据调整所有参数$\text{→}$适应特定任务 |
> >
> > :two:模型的总体结构
> >
> > 1. 多层双向$\text{Transformer}$编码器：
> >    <img src="https://i-blog.csdnimg.cn/direct/cf2949cbdc244b4aae7abc11baa49b16.png" alt="image-20241127203048377" width=280 /> 
> >    | 特点 | 结构                                               | 意义                       |
> >    | :--: | :------------------------------------------------- | :------------------------- |
> >    | 双向 | 每个$\text{Transformer}$自注意力同时关注左右上下文 | 捕捉更全面的全局上下文关系 |
> >    | 多层 | 堆叠很多层的$\text{Transformer}$编码层             | 捕捉更深层次的特诊和语义   |
> > 2. 模型参数：
> >    | 参数 | 含义                                             | $\text{BERT}_{\text{BASE}}$ | $\text{BERT}_{\text{LARGE}}$ |
> >    | :--: | :----------------------------------------------- | :-------------------------: | :--------------------------: |
> >    | $L$  | $\text{Transformer}$编码器的数量(即竖直上的层数) |            $12$             |             $24$             |
> >    | $H$  | 每层$\text{Transformer}$隐藏状态$/$词嵌入的维度  |            $768$            |            $1024$            |
> >    | $A$  | 每个$\text{Transformer}$自注意力机制的头数       |            $12$             |             $16$             |
> >
> > :three:模型的输入表示
> >
> > 1. 基本概念：
> >    |           结构           | 含义                                                         |
> >    | :----------------------: | :----------------------------------------------------------- |
> >    |   词元$\text{(Token)}$   | $\text{BERT}$处理的最小单元，为$\text{WordPiece}$分词得到的词$/$子词$\text{+}$特殊标记 |
> >    | 句子$\text{(Senetence)}$ | 任意长度的连续文本片段，不一定要有语言学意义                 |
> >    | 序列$\text{(Sequence)}$  | 输入到$\text{BERT}$种的$\text{Token}$序列，可以包含一个$/$多个句子 |
> > 2. 序列的表征：
> >    - 单句子：`<cls>`<font color=purple>句子$1$</font>`<sep>` 
> >    - 多句子：`<cls>`<font color=purple>句子$1$</font>`<sep>`<font color=purple>句子$2$</font>`<sep>`$\text{.......}$`<sep>`<font color=purple>句子$\text{N}$</font>`<sep>`
> > 3. 输入的构建：三种嵌入的相加
> >     <img src="https://i-blog.csdnimg.cn/direct/73e8cb63550f423fb4a3d04c32dde1bf.png" alt="WRAETHJTDKYJJwefvd" width=690 /> 
> >    |     嵌入类型      | 含义                                 | 维度 |
> >    | :---------------: | :----------------------------------- | :--: |
> >    |  $\text{Token}$   | 表示$\text{Token}$本身的词汇信息     | $H$  |
> >    | $\text{Segment}$  | 指示$\text{Token}$属于哪个句子       | $H$  |
> >    | $\text{Position}$ | 表示$\text{Token}$在序列中的位置信息 | $H$  |
> >
>
> ## $\textbf{2.2. BERT}$的预训练: 两种无监督任务
>
> > <img src="https://i-blog.csdnimg.cn/direct/f4db269158824104af3562d3253b61f1.png" alt="areshjfkj4" width=470 /> 
> >
> > :one:第一种任务: $\text{Masked}$语言模型
> >
> > 1. 采用$\text{Mask}$的动机：
> >    - 原有弊端：传统语言模型只能单向预测，而如果同时从两边开始训练会"泄露"词本身
> >    - 改进思路：在从两边开始训练前，先==随机==遮蔽一部分$\text{Token}$，再进行完形填空($\text{Cloze}$) 
> > 2. 屏蔽操作：选定$\text{15\%}$的$\text{Token}$进行替换，对需要替换的$\text{Token}$ 
> >    | 执行本操作的概率 | 执行的操作                                             |
> >    | :--------------: | :----------------------------------------------------- |
> >    |  $\text{80\%}$   | 直接用`<MASK>`替换原有$\text{Token}$                   |
> >    |  $\text{10\%}$   | 从词汇表中任选一个$\text{Token}$替换原有$\text{Token}$ |
> >    |  $\text{10\%}$   | 不做任何改变                                           |
> >    
> >    :bulb:这样做是为了弥合(预训练$\xleftrightarrow{不匹配}$微调)，因为微调的输入$\text{Token}$中不包含`<MASK>`
> > 3. 预测操作：不论是否被执行替换都要求模型预测其原始$\text{Token}\text{→}$用交叉熵计算与真实值的差异
> >
> > :two:第二种任务: 预测下一个句子($\text{NSP}$)，即两句子$\text{AB}$中$\text{B}$是否是$\text{A}$的下一句
> >
> > 1. 引入$\text{NSP}$的动机：
> >    - 对很多下游任务：需要判断句间关系，如$\text{Q\&A}/$自然语言推理$\text{NLI}$
> >    - 传统语言模型：只关注对单句进行建模从而预测下一个词，无法捕捉句间关系
> > 2. $\text{NSP}$的实现：
> >    - 数据构建：
> >      |         样本类型         | 构建方法                                                  |     比例      |
> >      | :----------------------: | :-------------------------------------------------------- | :-----------: |
> >      | 正样本$\text{(IsNext)}$  | 从语料库中选择两个连续的句子$\text{AB}$                   | $\text{50\%}$ |
> >      | 负样本$\text{(NotNext)}$ | 从语料库中选择一个句子$\text{A}/$再随机选择一个$\text{B}$ | $\text{50\%}$ |
> >    - 输入构建：`<cls>`<font color=purple>字句$\text{A}$</font>`<swp>`<font color=purple>字句$\text{B}$</font>`<swp>`$\xrightarrow{输入}\text{Transformer}$ 
> >    - 预测输出：在`<cls>`的最终隐向量上添加一个二分类器，将$\text{AB}$之间的关系分类为$\text{Is/NotNext}$ 
> >    - 损失函数：用交叉熵来计算预测$/$真实值之间的误差
> >
> > :three:整体的预训练目标：最小化$\text{MaskedLM/NSP}$两个过程中的交叉熵损失
>
> ## $\textbf{2.3. BERT}$微调的概述
>
> > :one:自注意力在微调中的优势
> >
> > |   方法   | 描述                                                         |
> > | :------: | :----------------------------------------------------------- |
> > | 传统方法 | 先独立对两段文本进行编码$\text{→}$再通过双向交叉注意力机制捕获其间的关系 |
> > | 自注意力 | 直接对连接后的文本进行自注意力编码，<font color=red>编码的同时就捕获了两句子间的交互信息</font> |
> >
> > :two:微调的原理
> >
> > 1. 输入结构
> >    - 输入格式：`<cls>`<font color=purple>字句$\text{A}$</font>`<swp>`<font color=purple>字句$\text{B}$</font>`<swp>` 
> >    - 聚合表示：用一个序列开头处`<CLS>`标记的最终隐向量作为整个序列的表示
> > 2. 输出结构：
> >    - 微调实现：根据特定任务，通过在$\text{BERT}$上添加一个额外输出层，以满足不同任务
> >    - 输出示例：将`<CLS>`的最终隐向量传给$\text{MLP}$全连接层进行分类，从而给出分类结果
> >
> > :three:一些$\text{NLP}$任务的微调概览
> >
> > | $\textbf{NLP}$任务 | $\textbf{A}$相当于 | $\textbf{B}$相当于 | 要干啥                                             |
> > | :----------------: | :----------------: | :----------------: | :------------------------------------------------- |
> > |      释义识别      | 句子$1$(地位相同)  | 句子$2$(地位相同)  | 判断$\text{AB}$意思是否相同                        |
> > |    自然语言推理    |        前提        |        假设        | 判断$\text{A}\xrightarrow{推导出}\text{B}$是否成立 |
> > |        问答        |        问题        |        段落        | 找到$\text{B}$中能回答$\text{A}$的片段             |
> > |      情感分析      |        文本        |   $\text{NULL}$    | 判断$\text{A}$的情感极性                           |
> >
> > - 相比预训练实现这些任务，微调要快多得多得多

# $\textbf{4. BERT}$实验: 微调$\textbf{\&}$消融

> ## $\textbf{4.1. BERT}$在不同任务上的微调
>
> > ### $\textbf{4.1.1 }$序列级任务
> >
> > > :one:通用语言理解评估$\text{(GLUE)}$
> > >
> > > 1. 关于$\text{GLUE}$：包含多种$\text{NLU}$任务的结合，部分数据集如下
> > >    |     数据集      | 描述                                   |    任务类型     |              备注               |
> > >    | :-------------: | :------------------------------------- | :-------------: | :-----------------------------: |
> > >    |  $\text{MNLI}$  | 判断句子对是否蕴涵$/$矛盾$/$中性       |     多分类      | $\text{GLUE}$中最大规模最常使用 |
> > >    |  $\text{QQP}$   | 判断两条问题是否语义等价               |     二分类      |          $\text{N/A}$           |
> > >    | $\text{SST-2}$  | 判断句子的情感极性(正面$/$负面)        |     二分类      |          $\text{N/A}$           |
> > >    |  $\text{CoLA}$  | 判断句子是否在语言学上“可接受”         |     二分类      |          $\text{N/A}$           |
> > >    | $\text{STS-B}$  | 预测句子对的语义相似度($1\text{→}5$分) |      回归       |          $\text{N/A}$           |
> > >    | $\text{......}$ | $\text{............}$                  | $\text{......}$ |         $\text{......}$         |
> > > 2. 微调的架构：
> > >    <img src="https://i-blog.csdnimg.cn/direct/2008edcd1ef84fbbba0386808ff0b7e9.png" alt="image-20241128012740164" width=400 /> 
> > >    - 输入序列：`<cls>`<font color=purple>句子$1$</font>`<sep>`<font color=purple>句子$2$</font>`<sep>`$\text{.......}$`<sep>`<font color=purple>句子$\text{N}$</font>`<sep>`
> > >    - 聚合表示：将`<cls>`的最终隐藏向量$C \text{∈} \mathbb{R}^H$，作为输入序列的表示
> > >    - 分类输出：新增一个分类层，其权重$W \text{∈} \mathbb{R}^{K \text{×} H}$($K$为标签数量)是唯一新增参数
> > >    - 损失函数：即$\log \left(\text{Softmax}\left(C W^T\right)\right)$，用于衡量分类的偏差
> > > 3. 微调的性能：在$\text{BatchSize=32}$，微调$\text{Epoch=3}$，学习率$\text{5/4/3/2e-5}$情况下
> > >    - $\text{BERT}_{\text{BASE}}/\text{BERT}_{\text{LARGE}}$优于现有模型，$\text{BERT}_{\text{LARGE}}$在所有数据集上优于$\text{BERT}_{\text{BASE}}$  
> > >    - $\text{BERT}$在$\text{GLUE}$上有提示，并拿下(当时的)[$\text{GLUE}$榜](https://gluebenchmark.com/leaderboard/)榜首 
> > >
> > > :two:情景对抗生成数据集($\text{SWAG}$)
> > >
> > > 1. 关于$\text{SWAG}$：
> > >    - 是什么：评估<font color=red>基于常识推理任务</font>的数据集，包含大量$\text{113000}$句子对
> > >    - 任务是：给定句子$A$，从四个句子$B_1B_2B_3B_4$中，选取最匹配的句子$B$作为$A$的延续
> > > 2. 微调的架构：
> > >    - 输入构造：共四个序列，即`<cls>`<font color=purple>句子$A$</font>`<sep>`<font color=purple>句子$B_{1/2/3/4}$($\text{A}$的四个可能延续)</font>`<sep>` 
> > >    - 分类参数：引入一个$\text{Task-specific}$向量$E$，其与$C$(`<cls>`的隐含表示)用于给候选项打分
> > >    - 得分计算：选取$\text{Softmax}(E\cdot C)$最大的项，作为四个中最可能的延续之一
> > > 3. 微调的性能：在$\text{BatchSize=16}/$学习率为$5 \text{×} 10^{-5}/\text{Epoch=3}$的情况下，$\text{BERT}_{\text{LARGE}}$最牛逼
> >
> > ### $\textbf{4.1.2. Token}$级任务
> >
> > > :one:斯坦福问答数据集$\text{(SQuAD v1.1)}$ 
> > >
> > > 1. 关于$\text{SQuAD v1.1}$：
> > >    - 是什么：包含$\text{10}$万对问答对的阅读理解数据集
> > >    - 任务是：给定(问题$+$包含答案的维基百科段落)$\text{→}$在段落中定位$\text{→}$获取正确答案片段
> > > 2. 微调的架构：
> > >    <img src="https://i-blog.csdnimg.cn/direct/46c3702323d34d51a53393b8d6b19f2d.png" alt="erzthyjgkvghgret" width=400 /> 
> > >    - 输入序列：`<cls>`<font color=purple>问题(使用$\text{A}$类型嵌入)</font>`<sep>`<font color=purple>段落(使用$\text{B}$类型嵌入)</font>`<sep>` 
> > >    - 新增参数：<font color=red>起始</font>$/$<font color=green>终止</font>向量<font color=red>$S\text{∈}\mathbb{R}^H$</font>$/$<font color=green>$E \text{∈} \mathbb{R}^H$</font>，作用分别是预测答案片段的<font color=red>起始</font>$/$<font color=green>终止</font>位置
> > > 3. 答案位置预测机制：
> > >    - <font color=red>起</font>$/$<font color=green>止</font>位置的概率：计算每个$\text{Token}$与<font color=red>$S$</font>$/$<font color=green>$E$</font>的点积<font color=red>$S\cdot {}T_i$</font>$/$<font color=green>$E\cdot {}T_i$</font>并将二者都$\text{Softmax}$成概率
> > >    - 答案片段选取：以最大化$\text{Score}(i, j)\text{=}\textcolor{red}{S \cdot  T_i}\text{+}\textcolor{green}{E \cdot  T_j}$的$\textcolor{red}{i}/\textcolor{green}{j}$间片段作为输出
> > > 4. 微调结果：在$\text{BatchSize=32}$，学习率为$5 \text{×} 10^{-5}$，$\text{Epoch=3}$的情况下
> > >    - 最佳系统$\text{BERT}_{\text{LARGE}}(\text{Ens.+TriviaQA})$在$\text{F1}$高出顶尖模型$\text{1.5}$
> > >    - 单个$\text{BERT}$模型在$\text{F1}$高出顶尖模型$\text{1.3}$
> > >    - 不用$\text{TriviaQA}$预训练的模型仍然超过现有模型
> > >
> > > :two:斯坦福问答扩展数据集$\text{(SQuAD v2.0)}$ 
> > >
> > > 1. 关于$\text{SQuAD v2.0}$
> > >    - 是什么：相比$\text{SQuAD v1.1}$引入了无法回答的问题，即回答不存在于提供段落中
> > >    - 目标是：先分类出段落中有无答案$\text{→}$再预测答案是什么
> > > 2. 扩展的规则：
> > >    - 对有答案$/$无答案的处理：
> > >      |  情况  | 描述                                                         | 得分                                                         |
> > >      | :----: | :----------------------------------------------------------- | :----------------------------------------------------------- |
> > >      | 有答案 | <font color=red>起</font>$/$<font color=green>止</font>向量都位于`<cls>` | $\text{S}_{\text{null}}\text{=}\textcolor{red}{S \cdot  C}\text{+}\textcolor{green}{E \cdot  C}$($C$为`<cls>`隐含表示)，为定值 |
> > >      | 无答案 | $\text{Otherwise}$                                           | $\widehat{s_{i, j}}\text{=}\max _{j \geq i} \textcolor{red}{S \cdot  T_i}\text{+}\textcolor{green}{E \cdot  T_j}$($T_{\textcolor{red}{i}/\textcolor{green}{j}}$为词$\textcolor{red}{i}/\textcolor{green}{j}$隐含表示) |
> > >    - 预测规则：考虑阈值$\tau$，若满足$\widehat{s_{i, j}}\text{>}s_{\text {null }}\text{+}\tau$，则认为$ij$间的非空片段为答案
> > > 3. 扩展结果：在$\text{BatchSize=48}/$学习率为$5 \text{×} 10^{-5}/\text{Epoch=2}$的情况下，照样是$\text{BERT}$吊打其它
> >
>
> ## $\textbf{4.2. BERT}$的消融实验
>
> > :one:预训练任务是否有效$?$ 
> >
> > 1. 三种模型：
> >    |         模型         |                    训练方向                     | 是否使用下一句预测 |
> >    | :------------------: | :---------------------------------------------: | :----------------: |
> >    |   $\text{No-NSP}$    | <font color=gree>$\text{MaskedLM}$(双向)</font> |         ❌          |
> >    | $\text{LTR\&No-NSP}$ |         <font color=red>从左到右</font>         |         ❌          |
> >    |    $\text{BERT}$     | <font color=gree>$\text{MaskedLM}$(双向)</font> |         ✅          |
> > 2. 实验结果：
> >    |       机制        | 影响                        | 机理                                       |
> >    | :---------------: | :-------------------------- | :----------------------------------------- |
> >    |   $\text{NSP}$    | 在推理$/$问答任务上性能提升 | $\text{NSP}$可帮助模型捕捉上下文之间的逻辑 |
> >    | $\text{MaskedLM}$ | 在所有任务上性能提升        | 使模型同时利用左右上下文                   |
> >
> > :two:模型规模对微调性能的影响$?$ 
> >
> > | 下游任务 | 实验结果                                                     |                结论                |
> > | :------: | ------------------------------------------------------------ | :--------------------------------: |
> > | 任意规模 | $\text{BERT}_{\text{LARGE}}$优于$\text{BERT}_{\text{BASE}}$及其它小规模$\text{Transformer}$ |        模型参数越多性能越好        |
> > |  小规模  | 将模型规模扩展到极限也能显著提升小规模任务的表现             | :point_up:同样适用小任务:point_up: |
> >
> > :three:基于特征的方法$\text{\&BERT}$
> >
> > 1. 两种模型：
> >    |        模型         | 对预训练$\textbf{BERT}$的处理 | 如何适应下游任务                                  |
> >    | :-----------------: | :---------------------------- | :------------------------------------------------ |
> >    | 微调版$\text{BERT}$ | 额外加一个分类层              | 联合微调所有参数                                  |
> >    | 特征版$\text{BERT}$ | 提取其上下文嵌入(特征)        | 输入特征到额外$\text{BiLSTM}$并训练，在接入分类器 |
> > 2. 实验任务：$\text{NER}$ 
> >    <img src="https://i-blog.csdnimg.cn/direct/855d43f2e12d477194bf2d34a75307a9.png" alt="image-20241128204144203" width=400 /> 
> >    - 数据集：$\text{CoNLL-2003 NER}$
> >    - 任务：为每个$\text{Token}$分配一个命名实体标签(不使用$\text{CRF}$) 
> > 3. 实验结果：
> >    - 微调方法：性能最佳
> >    - 特征方法：性能也与微调方法接近，也说明了$\text{BERT}$的通用性(也可以无需微调)