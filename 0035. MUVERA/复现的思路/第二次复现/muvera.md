# Muvera的实现`muvera.py`

## 1. `Muvera_sim()`计算单向量相似度

> ### 1.1. 函数的封装
>
> > :one:输入：
> >
> > 1. 数据部分：查询多向量`q_embedding`，段落多向量`p_embedding` 
> >    - 形状为`(N,1024)`的`q_embedding`，每个嵌入`q_i`向量的维度固定为`1024`，一共有`N`个向量
> >    - 形状为`(M,1024)`的`p_embedding`，每个嵌入`p_j`向量的维度固定为`1024`，一共有`M`个向量
> > 2. 超参数部分：$\text{SimHash}$参数`k_sim`，投影后的维度`d_proj`，投影重复的次数`R_reps` 
> >
> > :two:输出：
> >
> > 1. 相似度：查询多向量`q_embedding`和段落多向量的`p_embedding`之间的相似度
> > 2. 碰撞率：在分桶阶段中，`case_0/case_1/case_n`出现的次数
>
> ### 1.2. 函数的逻辑
>
> > :one:调用`fix_dimension(q_embedding,p_embedding)`
> >
> > 1. 得到单向量表示`q_final,p_final`
> > 2. 得到参数`case_0/case_1/case_n`的值
> >
> > :three:计算`q_final`和`p_final`的内积，得到相似度

---

## 2. `fix_dimension()`转为原始单向量

> ### 2.1. 函数的封装
>
> > :one:输入：
> >
> > 1. 数据部分：查询多向量`q_embedding`，段落多向量`p_embedding` 
> > 2. 超参数：
> >    - 超平面法向量数`k_sim`，生成此处哈希桶的数量`B=2^(k_sim)` 
> >    - 投影后的维度`d_proj`
> >    - 投影重复的次数`R_reps` 
> >
> > :two:输出：
> >
> > 1. 向量上：
> >    - `q_embedding`生成的最终单向量`q_final`
> >    - `p_embedding`生成的最终单向量`p_final`
> > 2. 参数上：`case_0/case_1/case_n`各自的数目
>
> ### 2.2. 函数的逻辑
>
> > :one:调用`sim_hash()`，处理多向量`q_embedding`和`p_embedding` ，一共`M+N`个嵌入
> >
> > 1. 将`M+N`个嵌入映射到`B`个桶中
> >
> > :two:调用`sub_vector()`，对所有`B`个桶执行操作
> >
> > 1. 得到所有由`q_embedding`生成的子向量：`q_bucket_1,q_bucket_2,...,q_bucket_B`
> > 2. 得到所有由`p_embedding`生成的子向量：`p_bucket_1,p_bucket_2,...,p_bucket_B`
> > 3. 返回三类桶出现的次数，即`case_0/case_1/case_n`各自的数目
> >
> > :three:调用`vec_repeat()`，生成最终的单向量表示`q_final`和`p_final`

## 3. `sim_hash()`对嵌入集进行哈希

> ### 3.1. 函数的封装
>
> > :one:输入：若干个嵌入向量，在这里一般为`M+N`个嵌入
> >
> > :two:输出：一共`B`组桶结构，每组桶都可能有相应的`q_i`和`p_j`被放入
>
> ### 3.2. 函数的逻辑
>
> > :one:从高斯分布中随机抽取`k_sim`个法向量，并对所有法向量作归一化操作
> >
> > 1. 记归一化后的向量为$g_{1}, \ldots, g_{k_{\text{sim}}} \text{∈} \mathbb{R}^{d}$ 
> >
> > 2. 由此生成了`2^(k_sim)`个划分空间，也就是桶，每个桶对应一个长为`k_sim`的二进制向量
> >
> > :two:对所有`M+N`个嵌入执行划分，每个嵌入都分配到一个空间中去，具体做法如下
> >
> > 1. 遍历所有的嵌入，假设当前嵌入为$x$，则作如下的处理
> >    - $\varphi(x)\text{=}\left(\mathbf{1}\left(\left\langle g_{1}, x\right\rangle\text{>}0\right), \ldots, \mathbf{1}\left(\left\langle g_{k_{\text{sim}}}, x\right\rangle\text{>}0\right)\right)$ 
> >    - $\mathbf{1}\left(\left\langle g_{i}, x\right\rangle\text{>}0\right)$表示$\langle g_{i}, x\rangle\text{>}0$成立，即$x$投影在超平面的正侧时，将该位设为$1$ 
> >    - 将$x$分配到与$\varphi(x)$编码一致的桶中

## 4. `sub_vector()`生成每个桶的子向量

> ### 4.1. 函数的封装
>
> > :one:输入：`B`组桶结构，每组桶都可能有相应的`q_i`和`p_j`被放入，其实就是`sim_hash()`的输出
> >
> > :two:输出：每桶输出两个子向量，一共`2B`个子向量
> >
> > 1. 得到所有由`q_embedding`生成的子向量：`q_bucket_1,q_bucket_2,...,q_bucket_B`
> > 2. 得到所有由`p_embedding`生成的子向量：`p_bucket_1,p_bucket_2,...,p_bucket_B`
> > 3. 三类桶出现的次数，即`case_0/case_1/case_n`各自的数目
>
> ### 4.2. 函数的逻辑
>
> > :one:遍历每个桶，假设当前处理的是`bucket_n`
> >
> > 1. 如果该桶内只有==一个==嵌入`p_j`来自于`p_embedding`
> >    - 子向量`p_bucket_n`等于`p_j` 
> >    - 子向量`q_bucket_n`等于所有`q_i`的嵌入的总和
> >    - `case_1`加一
> >
> > 2. 如果该桶内有==多个==嵌入`p_j`来自于`p_embedding`
> >    - 子向量`p_bucket_n`等于所有`p_j` 的总和除以嵌入的数量，即质心
> >    - 子向量`q_bucket_n`等于所有`q_i`的嵌入的总和
> >    - `case_n`加一
> >
> > 3. 如果该桶内==没有==嵌入`p_j`来自于`p_embedding`
> >    - 选择离`bucket_n`最近的一个`p_j` ，作为子向量`p_bucket_n`
> >      - 距离的计算基于：比较"二进制表示位差"(Hamming distance)
> >    - 子向量`q_bucket_n`等于所有`q_i`的嵌入的总和
> >    - `case_0`加一
> >
> > :two:按照以上所述，正确返回必要的详细

---

## 5. `vec_compress()`压缩每个原始单向量

> ### 5.1. 函数的封装
>
> > :one:输入：
> >
> > 1. 数据上：注意此处记每个子向量为原始维度`d`
> >    - 所有由`q_embedding`生成的子向量：`q_bucket_1,q_bucket_2,...,q_bucket_B`
> >    - 所有由`p_embedding`生成的子向量：`p_bucket_1,p_bucket_2,...,p_bucket_B`
> > 2. 超参数：降维后的维度`d_proj`
> >
> > :two:输出：压缩后的子向量集的拼接，注意此处每个子向量维度变为`d_proj`
> >
> > 1. `q_compress = <q_compress_1,q_compress_2,...,q_compress_B>`
> > 2. `p_compress = <p_compress_1,p_compress_2,...,p_compress_B>`
>
> ### 5.2. 函数的逻辑
>
> > :one:压缩函数的形式化表达：
> >
> > 1. $\psi: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d_{\text{proj}}}$
> > 2. $\psi(x)=\left(1 / \sqrt{d_{\text{proj}}}\right) S x$
> > 3. 其中 $S \in \mathbb{R}^{d_{\text{proj}} \times d}$ 是随机矩阵，其元素在 $\pm 1$ 之间均匀分布
> >
> > :two:压缩的逻辑，直接应用`vec_compress_i = phi(vec_bucket_i)`即可

## 6. `vec_repeat()`生成最终的单向量

> ### 6.1. 函数的封装
>
> > :one:输入：
> >
> > 1. 数据上：和`vec_compress()`的一样
> >    - 所有由`q_embedding`生成的子向量：`q_bucket_1,q_bucket_2,...,q_bucket_B`
> >    - 所有由`p_embedding`生成的子向量：`p_bucket_1,p_bucket_2,...,p_bucket_B`
> > 2. 超参数：压缩向量`vec_compress`的重复次数`R_reps` 
> >
> > :two:输出：`R_reps` 次重复压缩后的向量，注意每次压缩的时候压缩矩阵$S$都要更新
> >
> > 1. `q_final = <q_compress_1st,q_compress_2nd,...,q_compress_Bth>`
> > 2. `p_final = <p_compress_1st,p_compress_2nd,...,p_compress_Bth>`
>
> ### 6.2. 函数的逻辑
>
> > :one:执行`R_reps` 次`vec_compress()`，得到`R_reps`个向量
> >
> > :two:将`R_reps`个向量左右拼接，得到最终向量

---

## 7. `main()`端到端的实现

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
> 1. 遍历每个`query_id`，假设当前`query_id=i`
>
>    - 对于当前`query_id=i`，遍历每个`passage_id`，假设当前`passage_id=j`
>
>      - 计算$\text{Muvera}$距离`Muvera_sim(query_id=i,passage_id=j)`
>
>      - 得到了单向量的近似距离
>
>      - 得到了`query_id=i`和`passage_id=j`之间的`case_0/case_1/case_n`各自的数目
>
>      - 将所得到的数据组织成数据结构：`result_muvera.csv`
>
>        ```txt
>        query_id  passage_id  muvera_sim  case_0_num  case_1_num  case_n_num 
>        ```
>
> 2. 最终得到完整的`result_muvera.csv`