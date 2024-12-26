:one:$k\text{-Means}$分簇方法

1. 含义：一种无监督学习，用于将数据集分为$k$个簇(每簇一个质心)，使同簇点靠近/异簇点远离

2. 流程：

   <img src="https://img-blog.csdnimg.cn/img_convert/0054ea133264842f19b0ccbcac7f266a.png" alt="image-20241031100601961" width=320 />  

:two:$\text{PQ}$算法流程

0. 给定$k$个$D$维向量

    $\small\begin{cases}
    \textbf{v}_1=[x_{11},x_{12},x_{13},x_{14},...,x_{1D}]\\\\
    \textbf{v}_2=[x_{21},x_{22},x_{23},x_{24},...,x_{2D}]\\\\
    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
    \textbf{v}_k=[x_{k1},x_{k2},x_{k3},x_{k4},...,x_{aD}]
    \end{cases}\xleftrightarrow{}
    \begin{cases}
    \textbf{v}_{1}=\{[x_{11},x_{12},x_{13}],[x_{14},x_{15},x_{16}],...,[x_{1(D-1)},x_{1(D-1)},x_{1D}]\}\\\\
    \textbf{v}_{2}=\{[x_{21},x_{22},x_{23}],[x_{24},x_{25},x_{26}],...,[x_{2(D-1)},x_{2(D-1)},x_{2D}]\}\\\\
    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
    \textbf{v}_{k}=\{[x_{k1},x_{k2},x_{k3}],[x_{k4},x_{k5},x_{k6}],...,[x_{k(D-1)},x_{k(D-1)},x_{kD}]\}
    \end{cases}$ 

1. 分割子空间：将$D$维向量分为$M$个$\cfrac{D}{M}$维向量

   $\small子空间1\begin{cases}
   \textbf{v}_{11}=[x_{11},x_{12},x_{13}]\\\\
   \textbf{v}_{21}=[x_{21},x_{22},x_{23}]\\\\
   \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
   \textbf{v}_{k1}=[x_{k1},x_{k2},x_{k3}]
    \end{cases}\&子空间2
   \begin{cases}
    \textbf{v}_{12}=[x_{14},x_{15},x_{16}]\\\\
   \textbf{v}_{22}=[x_{24},x_{25},x_{26}]\\\\
   \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
   \textbf{v}_{k2}=[x_{k4},x_{k5},x_{k6}]
   \end{cases}\&...\&子空间M
   \begin{cases}
   \textbf{v}_{1M}=[x_{1(D-1)},x_{1(D-1)},x_{1D}]\\\\
   \textbf{v}_{2M}=[x_{2(D-1)},x_{2(D-1)},x_{2D}]\\\\
   \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
   \textbf{v}_{kM}=[x_{k(D-1)},x_{k(D-1)},x_{kD}]
   \end{cases}$ 

2. 生成$\text{PQ}$编码:

   $\small子空间1\begin{cases}
   \textbf{v}_{11}\xleftarrow{替代}\text{Centriod}_{11}\\\\
   \textbf{v}_{21}\xleftarrow{替代}\text{Centriod}_{21}\\\\
   \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
   \textbf{v}_{k1}\xleftarrow{替代}\text{Centriod}_{k1}
    \end{cases}\&子空间2
   \begin{cases}
    \textbf{v}_{12}\xleftarrow{替代}\text{Centriod}_{12}\\\\
   \textbf{v}_{22}\xleftarrow{替代}\text{Centriod}_{22}\\\\
    \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
   \textbf{v}_{k2}\xleftarrow{替代}\text{Centriod}_{k2}
    \end{cases}\&...\&子空间M
   \begin{cases}
   \textbf{v}_{1M}\xleftarrow{替代}\text{Centriod}_{1M}\\\\
   \textbf{v}_{2M}\xleftarrow{替代}\text{Centriod}_{2M}\\\\
   \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
   \textbf{v}_{kM}\xleftarrow{替代}\text{Centriod}_{kM}
   \end{cases}$ 

   - 聚类：在每个子空间上运行$k\text{-Means}$算法(一般$k\text{=}256$)$\to$每个$\textbf{v}_{ij}$都会分到一个$\cfrac{D}{M}$维的质心

   - 编码：将每个子向量$\textbf{v}_{ij}$所属质心的==索引==作为其$\text{PQ}$编码，并替代原有子向量

3. 生成最终的压缩向量$\small\to\begin{cases}
   \widetilde{\textbf{v}_{1}}=\{\text{Centriod}_{11},\text{Centriod}_{12},...,\text{Centriod}_{1M}\}\\\\
   \widetilde{\textbf{v}_{2}}=\{\text{Centriod}_{21},\text{Centriod}_{22},...,\text{Centriod}_{2M}\}\\\\
   \,\,\,\,\,\,\,\,\,\,\,\,.........\\\\
   \widetilde{\textbf{v}_{k}}=\{\text{Centriod}_{k1},\text{Centriod}_{k2},...,\text{Centriod}_{kM}\}
   \end{cases}$  

   - 存储阶段：存储的内容实质上是质心索引，每个向量只占用$M$维
   - 使用阶段：所有的质心索引被解压为质心，每个向量维度又恢复$M\text{×}\cfrac{D}{M}\text{=}D$维

:three:$\text{IVF+PQ}$原理

1. 离线索引阶段：
   - 构建$\text{IVF}$：使用$\text{K-Menas}$将原始向量集合划分为$n$簇(即$n$个质心)
   - 簇内压缩：对每个簇执行$\text{PQ}$压缩，即将每个簇内向量替换为质心索引
2. 在线查询阶段：
   - $\text{IVF}$部分：计算与查询$q$与所有簇质心的距离，由此选定前$n_{\text{probe}}$个簇的所有向量
   - $\text{PQ}$部分：由质心索引还原选定向量$\text{→}$计算$q$之的距离(遍历子空间)$\displaystyle{}\text{dist}(q, v) \text{≈} \sum_{i=1}^M \text{dist}\left(q_i, c_{j i}\right)$