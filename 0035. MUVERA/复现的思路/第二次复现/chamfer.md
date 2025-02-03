# Chamfer的实现`chamfer.py`

## 1. `chamfer_sim()`函数

> :one:函数的封装
>
> 1. 输入：查询多向量$Q$，段落多向量$P$
>    - 查询多向量$Q$，包含有若干个嵌入向量$Q\{q_1,q_2,...,q_n\}$，每个嵌入向量的维度固定
>    - 段落多向量$P$，包含有若干个嵌入向量$P\{p_1,p_2,...,p_m\}$，每个嵌入向量的维度固定
> 2. 输出：查询多向量$Q$和段落多向量的$P$之间的相似度
>
> :two:函数的逻辑：$\displaystyle{}\text{Chamfer}(Q,P)\text{=}\sum_{q \text{∈} Q} \max _{p \text{∈} P}\langle q, p\rangle$也就是所谓的后期交互
>
> - 遍历所有的$q\in{}Q$，假设当前正在处理的是$q_i$
>   - 对于当前的$q_i$，遍历所有的$p\in{}P$，假设当前正在处理的是$p_j$ 
>     - 计算$\langle q_i, p_j\rangle$内积，也就是二者的相似度
>     - 更新$\text{MaxSim}_i$为$q_i$当前得到过的最大内积
>   - 得到最终的$\text{MaxSim}_i$ 
> - 得到所有的$\text{MaxSim}_1,\text{MaxSim}_2,...,\text{MaxSim}_n$，将其全部相加即为最终的相似度

## 2. 对`Chamfer.py`

> :one:数据准备阶段
>
> 1. 加载查询`m3_emb_query.csv`文件，表头为`query_id,query_emb`
>    - `query_id`即查询的主键
>    - `query_emb`是每个查询的嵌入，张量的形状是`(N,1024)`即共`N`个单向量，每个单向量`1024`维
> 2. 加载查询`m3_emb_passage.csv`文件，表头为`passage_id,passage_emb`
>    - `passage_id`即查询的主键
>    - `passage_emb`是每个查询的嵌入，张量的形状是`(M,1024)`即共`M`个单向量，每个单向量`1024`维
>
> :two:相似度计算
>
> - 遍历每个`query_id`，假设当前`query_id=i`
>   - 对于当前`query_id=i`，遍历每个`passage_id`，假设当前`passage_id=j`
>     - 计算$\text{Chamfer}$距离`chamfer_sim(query_id=i,passage_id=j)`
>   - 建立索引`query_id=i -> query_id=i`的$\text{Top-1000}$，存储到合适的目录和文件格式中
> - 得到最终的索引文件，每个`query_id`都能索引到$\text{Top-1000}$的`passage_id` 