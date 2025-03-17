# 1. 导论$\textbf{\&}$方法

> :one:要干啥：在$\text{ColBERT}$方法中，限制每个段落要保留的$\text{Token}$的数量，或者说对段落$\text{Token}$进行剪枝
>
> :two:怎么干：注意以下方法都是整合进$\text{ColBERT}$训练的顶层池化层，而非在后期交互中进行改进
>
> 1. 前$k$位置$\text{Token}$：只保留每个段落的前$k$个$\text{Token}$
>
> 2. 前$k$罕见$\text{Token}$：选择段落中最罕见的$k$个$\text{Token}$，所谓罕见的$\text{Token}$即$\text{IDF}$高的$\text{Token}$ 
>
> 3. 前$k$闲置$\text{Token}$：在段落前添加$k$个特殊$\text{Token}$，这些$\text{Token}$在$\text{BERT}$词汇表中标为闲置(`unused`)，最终只保留这$k$个$\text{Token}$ 
>
> 4. 前$k$得分$\text{Token}$：用预训练模型的最后一层注意力机制给所有$\text{Token}$一个注意力评分，选取注意力机制最高的$k$个$\text{Token}$
>
>    - 注意力张量：$P\text{=}\{p_1,p_2,...,p_m\}$的注意力为三维张量$A(h,i,j)$，表示在$h$头注意力机制中$p_i$与$p_j$二者的注意力相关性
>
>      <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250312200743542.png" alt="image-20250312200743542" width=530 />  
>
>    - 注意力评分：以$p_i$为例，其注意力评分为每个注意力头中与$p_i$有关行的总和，即$a(q_i)\text{=}\displaystyle{}\sum_{h=0}^{h_\max}\sum_{j=0}^{m}A(h,i,j)$ 
>

# $\textbf{2. }$实验概要

> :one:训练方法：$\text{ColBERT}$使用$\text{Mini-LM}$时无需归一化和查询扩展，大幅降低计算成本​
>
> :two:检索性能：当$k\text{=}50$时，剪枝可减少$\text{30\%}$的段落索引，并且性能减少极小($\text{nDCG@10}$减小$\text{0.01}$)
>
> :three:方法对比：当普通剪枝($k\text{=50}$)时方法$\text{1\&3}$最佳，剧烈剪枝($k\text{=10}$)时方法$3$显著优于其它方法