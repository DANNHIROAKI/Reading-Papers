[toc]
:point_right:相关论文

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://doi.org/10.48550/arXiv.1810.04805)
   - [BERT的总结](https://blog.csdn.net/qq_64091900/article/details/144120987)
2. [ColBERTv1: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://doi.org/10.48550/arXiv.2004.12832)
   - [ColBERTv1的总结](https://dannhiroaki.blog.csdn.net/article/details/144157480)
3. [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://doi.org/10.48550/arXiv.2112.01488)
4. [PLAID: An Efficient Engine for Late Interaction Retrieval](https://doi.org/10.48550/arXiv.2205.09707)
5. [EMVB: Efficient Multi-Vector Dense Retrieval Using Bit Vectors](https://doi.org/10.48550/arXiv.2404.02805) 

# $\textbf{1. }$研究背景与综述

> ## $\textbf{1.1. }$本研究的背景
>
> > :one:神经信息检索的范式
> >
> > |     方式     | 编码方式                                    | 相似度计算                                  |
> > | :----------: | ------------------------------------------- | ------------------------------------------- |
> > | 单向量相似性 | 将查询$/$文档编码为单个高维向量             | 查询$/$文档向量的点积                       |
> > | 多向量交互式 | 将查询$/$文档的每个$\text{Token}$编码为嵌入 | 查询$/$文档所有$\text{Token}$嵌入的丰富交互 |
> >
> > :two:关于后期交互
> >
> > 1. 含义：多向量($\text{Token}$级)交互的一种
> >
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20241204161715584.png" alt="image-20241204161715584" width=450 /> 
> >
> >    - 后期：在查询前就算好所有文档的嵌入，查询时只需编码查询
> >    - 交互：计算查询每个嵌入与文章所有嵌入的最大值，最终再将所有最大值相加得到相似度 
> >
> > 2. 优$/$缺点：
> >
> >    - 优点：将相关性拆解到$\text{Token}$级$\text{→}$更能捕捉语义，减轻了每次查询时$\text{Encoder}$的压力
> >    - 却点：相比文档级相似度需要存储大量$\text{Token}$的嵌入，存在固有的$\text{Token}$级偏差
>
> ## $\textbf{1.2. ColBERTv2}$的主要贡献
>
> > 
> 
> ## $\textbf{1.3. }$文献综述
> 
> > 
> >

