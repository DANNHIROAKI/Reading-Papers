# $\textbf{1. }$实验设置

> :one:实验数据：$\text{MS-MARCO v1.1}$，大约$1\text{w}$条查询和$8.5\text{w}$条段落
>
> :two:嵌入模型：$\text{ColBERTv2}$，设置嵌入维度固定为$\text{128}$，查询嵌入数量为$30$，段落嵌入数量不固定
>
> :three:超参数：$(R_{\text{reps}},k_{\text{sim}},d_{\text{proj}})\text{∈}\{(20, 3, 8), (20, 4, 8)(20, 5, 8), (20, 5, 16)  \}$ 

# $\textbf{2. Recall}$ 

> :one:方法：
>
> 1. 分计算$\text{Chamfer}$方法的查询$\text{Top-1}$段落$/\text{Muvera}$方法的查询$\text{Top-N}$段落
> 2. 得到$\text{1Recall@N}$，即前者$\text{Top-1}$在后者$\text{Top-N}$中情况的比率
>
> :two:结果：
>
> |                        | $\textbf{1Recall@100}$ | $\textbf{1Recall@200}$ | $\textbf{1Recall@500}$ |
> | :--------------------: | :--------------------: | :--------------------: | :--------------------: |
> | $\text{(20, 3, 8)  }$  |   $\text{0.810677}$    |   $\text{0.838314}$    |   $\text{0.875163}$    |
> | $\text{(20, 4, 8)  }$  |   $\text{0.865951}$    |   $\text{0.884375}$    |   $\text{0.902800}$    |
> | $\text{(20, 5, 8)  }$  |   $\text{0.884375}$    |   $\text{0.893587}$    |   $\text{0.907406}$    |
> | $\text{(20, 5, 16)  }$ |   $\text{0.893587}$    |   $\text{0.902800}$    |   $\text{0.912012}$    |

# $\textbf{3. }$碰撞率实验

> ## $\textbf{3.1. }$实验流程
>
> > :one:对每个[查询$\xleftrightarrow{}$段落]$\text{Pairs}$计算$\text{Muvera}$相似度，同时记录如下信息
> >
> > ```json
> >{
> >    "num_of_emb": {
> >       "num_of_qemb": 000,
> >          "num_of_pemb": 000
> >       },
> >       "bucket_distribution": {
> >          "p_bucket_100": 000,
> >          "p_bucket_110": 000,
> >          "p_bucket_101": 000,
> >          "p_bucket_000": 000,
> >          "q_bucket_010": 000,
> >          "q_bucket_100": 000,
> >          "q_bucket_101": 000,
> >          "q_bucket_111": 000,
> >          "q_bucket_000": 000
> >       },
> >       "bucket_stats": {
> >          "case_0": 00,
> >          "case_1": 00,
> >          "case_n": 00
> >       }
> >    }
> >    ```
> > 
> > 1. `num_of_emb`：[查询$\xleftrightarrow{}$段落]$\text{Pairs}$中嵌入数量的统计
> >   - `num_of_qemb`：对查询进行$\text{ColBERTv2}$多向量嵌入后的嵌入个数
> >    - `num_of_pemb`：对段落进行$\text{ColBERTv2}$多向量嵌入后的嵌入个数
> > 2. `bucket_distribution`：进行$\text{SimHash}$后嵌入在桶中的分布
> >    - `q_bucket_XXX`：二进制编码为`XXX`的桶中，查询嵌入向量的个数
> >    - `p_bucket_XXX`：二进制编码为`XXX`的桶中，段落嵌入向量的个数
> > 3. `bucket_stats`：进行$\text{SimHash}$后桶的状态分布
> >    - `case_0`：有多少个桶，含有$0$个来自段落的嵌入
> >    - `case_1`：有多少个桶，含有$1$个来自段落的嵌入
> >    - `case_n`：有多少个桶，含有多个来自段落的嵌入
>
> ## $\textbf{3.2. }$实验的结果
>
> > :one:`bucket_stats`的分布：`case_0`/`case_1`/`case_n`的占比平均
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250114042614227.png" alt="image-20250114042614227" width=350 />  <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250114042647029.png" alt="image-20250114042647029" width=350 /> 
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250114053259115.png" alt="image-20250114053259115" width=350 /> <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250114083523628.png" alt="image-20250114082606218" width=350 />  
> >
> > :two:碰撞率
> >
> > 1. 如何定义碰撞率
> >
> >    - 将段落的嵌入分为两类，一类"独占"一个桶，一类与其他段落嵌入”共享“一个桶，后者算作碰撞
> >    - 在每次计算中，$\#$不碰撞段落嵌入$\text{ =  \#Case1}$
> >    - $\text{碰撞率} = 1 - \left( \cfrac{\text{\#Case1}}{\text{\#PasssageEmb}} \right)$
> >
> > 2. 实验结果
> >
> >    |  参数  |  $\textbf{(20,3,8)}$  | $\textbf{(20, 4, 8)}$ | $\textbf{(20, 5, 8)}$ | $\textbf{(20, 5, 16)}$ |
> >    | :----: | :---------------: | :-------------------: | :--: | :--: |
> >    | 碰撞率 |     $98.38\%$     |       $98.66\%$       | $\text{97.55\%}$ | $\text{97.20\%}$ |
> >
> > :two:`bucket_distribution`的分布：落入每个桶的元素的占比
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250114043659726.png" alt="image-20250114043659726" width=600 /> 
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250114043722259.png" alt="image-20250114043722259" width=600 /> 
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250114053613007.png" alt="image-20250114053539659" width=600 /> 
> >
> > <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250114082648993.png" alt="image-20250114082648993" width=600 />   
