<img src="https://i-blog.csdnimg.cn/direct/d2ca3bae2f6f41e8a39675ed1527bd77.png" alt="image-20250305212409686" width=800 />   

😂图放这了，大道至简的$\text{idea}$不愧是$\text{ECIR}$

# $\textbf{1. ConstBERT}$的原理

> :one:模型的改进点：相较于$\text{ColBERT}$为每个$\text{Token}$生成一个向量，$\text{ConstBERT}$只为段落生成固定$C$个向量
>
> 1. 嵌入阶段：为查询$Q$和段落$P$的每个$\text{Token}$都生成一个$d$维向量，是为$\{q_{1},\ldots,q_{N}\}$和$\{p_{1},\ldots,p_{M}\}$
> 2. 线性变换：拼接所有段落单向量为$\left[p_{1},\cdots,p_{M}\right]\text{∈}\mathbb{R}^{dM}$，进行$\mathbf{W}\text{∈}\mathbb{R}^{Mk\text{×}Ck}$投影得$\left[\delta_{1},\cdots, \delta_{C}\right]\text{=}\mathbf{W}^{T}\left[p_{1},\cdots,p_{M}\right]\text{∈}\mathbb{R}^{dC}$
> 3. 后期交互：同$\text{ColBERT}$，为每个$q_i$找到与其内积最大的$\text{MaxSim}(q_i,\delta)\text{=}\delta_{p_i}$，最后将所有$\text{MaxSim}$相加得到相似度评分
>
> :two:改进的动机：为何非要固定数目的段落向量
>
> 1. 存储效率上：设定$C\text{<}M$后，能降低段落嵌入所占的空间
> 2. 计算效率上：设定$C\text{<}M$后，将原有$O(MN)$的查询复杂度降为了$O(CN)$
> 3. 系统级优化：使得内存对齐，规避了变长文档表示导致内存碎片化，从而降低了$\text{Cache Miss}$ 

# $\textbf{2. ConstBERT}$的实验结果

> :one:效果：当$C\text{=}32$时，在$\text{MsMarco/BEIR}$等数据集上，查询效果与$\text{ColBERT}$相当(用$\text{MRR@10/nDCG@10}$衡量)
>
> :two:效率：相比$\text{ColBERT}$对段落的存储空间需求减少了一半多，端到端检索响应速度也显著加快