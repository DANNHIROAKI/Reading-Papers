[原论文](https://doi.org/10.48550/arXiv.2210.15748)

@[toc]



# $\textbf{1. }$定理$\textbf{4.2}$的内容

> ## $\textbf{1.1. }$一些符号
>
> > :one:一些基础的符号
> >
> > 1. 基本符号：
> >
> >    | 符号 | 含义                                                         |
> >    | :--: | :----------------------------------------------------------- |
> >    | $Q$  | 查询集，此处只考虑一个$Q$并且$Q{=}\{q_1,q_2,...,q_{m_q}\}$(此处假设$m_q$为常数)，记其子向量为$q_r{∈}Q$ |
> >    | $S$  | 目标集，此处假定每个${\mid}S{\mid}{=}m$(常数)，$S^*$与$Q$评分最大(其元素为$x^*_j{∈}S^*$)，其余都记为$S_i$(其元素为$x_{ij}{∈}S_i$) |
> >    | $N$  | 目标集的集即$D\text{=}\{S_1,...,S_N\}$，记其元素为$S_i{∈}N$  |
> >
> > 2. 相似度集合
> >
> >    |          符号           | 含义                                                         |
> >    | :---------------------: | :----------------------------------------------------------- |
> >    |   ${\mathbf{s}}_{ri}$   | $q_r$与$\forall{x_{ij}}{∈}S_i$的精确相似度的集合，即${\mathbf{s}}_{ri}{=}\{\text{Sim}(q_r,x_{i1}),...,\text{Sim}(q_r,x_{im})\}$ |
> >    | $\hat{\mathbf{s}}_{ri}$ | $q_r$与$\forall{x_{ij}}{∈}S_i$的近似相似度的集合，即$\hat{\mathbf{s}}_{ri}{=}\{\hat{\text{Sim}}(q_r,x_{i1}),...,\hat{\text{Sim}}(q_r,x_{im})\}$ |
> >    |   ${\mathbf{s}}_r^*$    | $q_r$与$\forall{}x^*_j{∈}S^*$的精确相似度的集合，即${\mathbf{s}}_{ri}{=}\{\text{Sim}(q_r,x_1^*),...,\text{Sim}(q_r,x_m^*)\}$ |
> >    | $\hat{\mathbf{s}}_r^*$  | $q_r$与$\forall{}x^*_j{∈}S^*$的近似相似度的集合，即${\mathbf{\hat{s}}}_{ri}{=}\{\hat{\text{Sim}}(q_r,x_1^*),...,\hat{\text{Sim}}(q_r,x_m^*)\}$ |
> >
> > 3. 相似度及其最大值
> >
> >    |                 符号                  | 含义                                                         |
> >    | :-----------------------------------: | :----------------------------------------------------------- |
> >    |        $s_{rij}$和$s_{ri\max}$        | $q_r$与$\forall{x_{ij}}{∈}S_i$中的$s_{rij}{=}\text{Sim}(q_r,x_{ij})$，其最大值记作$s_{ri\max}{=}\max({\mathbf{s}}_{ri})$且此时向量记为$x_{ij}^*$ |
> >    |  $\hat{s}_{rij}$和$\hat{s}_{ri\max}$  | $q_r$与$\forall{x_{ij}}{∈}S_i$中的$\hat{s}_{rij}{=}\hat{\text{Sim}}(q_r,x_{ij})$，其最大值记作$\hat{s}_{ri\max}{=}\max(\hat{{\mathbf{s}}}_{ri})$且此时向量记为$x_{ij}^*$ |
> >    |      ${s}_{rj}^*$和$s_{r\max}^*$      | $q_r$与$\forall{}x^*_j{∈}S^*$中的${s}_{rj}^*{=}\text{Sim}(q_r,x_j^*)$，其最大值记作$s_{r\max}^*{=}\max({\mathbf{s}}_r^*)$且此时向量记为$x_j^{**}$ |
> >    | $\hat{s}_{rj}^*$和$\hat{s}_{r\max}^*$ | $q_r$与$\forall{}x^*_j{∈}S^*$中的$\hat{s}_{rj}^*{=}\hat{\text{Sim}}(q_r,x_j^*)$，其最大值记作$\hat{s}_{r\max}^*{=}\max({\mathbf{\hat{s}}}_r^*)$且此时向量记为$x_j^{**}$ |
> >
> > 4. 几种聚合：($w_r$为权值)
> >
> >    - 内部聚合：即$σ$或$\max$(最大值聚合)，以$\sigma$为例对不同集合的聚合记作$σ({\mathbf{s}}_{ri})/σ(\hat{\mathbf{s}}_{ri})/σ({\mathbf{s}}_r^*)/σ(\hat{\mathbf{s}}_r^*)$
> >    - 外部聚合：聚合$σ({\mathbf{s}}_{ri})/σ(\hat{\mathbf{s}}_{ri})/σ({\mathbf{s}}_r^*)/σ(\hat{\mathbf{s}}_r^*)$得到评分
> >      - 对$S_i$有：$F\left({Q,S_i}\right)\text{=}\displaystyle{}\frac{1}{m_q}\sum_{r=1}^{m_q}w_rσ({\mathbf{s}}_{ri})$和$\hat{F}\left({Q,S_i}\right)\text{=}\displaystyle{}\frac{1}{m_q}\sum_{r=1}^{m_q}w_rσ({\mathbf{\hat{s}}}_{ri})$
> >      - 对$S^*$有：${F}\left({Q,S^*}\right)\text{=}\displaystyle{}\frac{1}{m_q}\sum_{r=1}^{m_q}w_rσ({\mathbf{s}}_r^*)$和$\hat{F}\left({Q,S^*}\right)\text{=}\displaystyle{}\frac{1}{m_q}\sum_{r=1}^{m_q}w_rσ(\hat{\mathbf{s}}_r^*)$
> >
> > :two:定理参数与结论
> >
> > 1. 界限参数：对于$0\text{<}β\text{≤}1\text{≤}α$ 
> >
> >    |          符号          | 含义                                                         |
> >    | :--------------------: | :----------------------------------------------------------- |
> >    | $F(Q, S^*)$的下界$B^*$ | $B^*\text{=}\displaystyle{}\frac{\beta}{m_q}\sum_{r=1}^{m_q}w_rs_{r\max}^*$即$S^*$的保守估计 |
> >    | $F(Q, S_i)$的上界$B_i$ | $B_i\text{=}\displaystyle{}\fracα{m_q}\sum_{r=1}^{m_q}w_r\hat{s}_{ri\max}$即$S_i$的乐观估计，$S^*{\notin}\{S_i\}$时$B_i$最大值为$B_{i\max}$，及$\Delta{\text{=}}\cfrac{B^*–B_{i\max}}{3}$ |
> >
> > 2. 其它参数：(有些是证明过程中的)
> >
> >    |         符号         | 含义                                                         |
> >    | :------------------: | :----------------------------------------------------------- |
> >    |   失败概率$δ$   | 算法的失败概率，目标是以至少$1{-}δ$的概率正确返回$S^*$  |
> >    |     哈希数量$L$      | $\text{DESSERT}$中对每个$q_r{∈}Q$和$x_{ij}{∈}S_i$进行$L$次分桶，此处设为$L\text{=}\displaystyle{}O\left({\log\left(\frac{N{m}_{q}m}{δ}\right)}\right)$ |
> >    |  上界$γ_{ri}$   | 是一个关于$\Delta_{ir}/s_{ri\max}/\tau_{ir}$的函数($\tau_{ri}{=}αs_{ri\max}{+}\Delta_{ri}$)，当$\Delta_{ri}$固定时其最大值记作$(γ_{ri})_{\max}$ |
> >    | 指示随机$\mathbb{1}$ | 例如事件$A$发生了则有$\mathbb{1}_A{=}1$，而本文中$\mathbb{1}{=}1$表示所有上界和下界条件同时满足 |
>
> ## $\textbf{1.2. }$定理内容
>
> > :one:定理结论：设定$\Delta{＞}0$
> >
> > 1. 第一种表述：$\Pr\left[\forall{i}{,}\hat{F}(Q,S^*){>}\hat{F}(Q,S_i)\right]{≥}1{-}δ$
> >
> > 2. 第一种表述：$\text{DESSERT}$算法结构能以$1{-}δ$的概率，返回与$Q$相似度最高的$S^*\text{=}\mathop{\operatorname{argmax}}\limits_{{i{∈}\{1,\ldots,N\}}}F\left( {Q,S_{i}}\right)$   
> >
> > :two:证明思路：要证$\Pr\left[\forall{i}{,}\hat{F}(Q,S^*){>}\hat{F}(Q,S_i)\right]{≥}1{-}δ$ 
> >
> > 1. 上界控制：对于所有$S_i{≠}S^*$，确保其估计得分$\hat{F}(Q,S_i)$不超过某个阈值
> > 2. 下界控制：对于$S^*$，确保其估计得分$\hat{F}(Q,S^*)$不低于某个阈值
> > 3. 联合界限：找到一个$L$，使得上述条件同时以高概率成立，从而保证$\hat{F}(Q,S^*){>}\hat{F}(Q,S_i)$ 

# $\textbf{2. }$上下界控制

> ## $\textbf{2.1. }$对$\boldsymbol{γ}$函数的重新分析
>
> > :one:函数$γ_{ri}{=}\left(\cfrac{α\left({1{-}s_{ri\max}}\right)}{α{-}τ_{ri}}\right){\left(\cfrac{s_{ri\max}\left({α{-}τ_{ri}}\right)}{τ_{ri}\left({1{-}s_{ri\max}}\right)}\right)}^{\frac{τ_{ri}}{α}}$ 
> >
> > 1. 用$τ_{ri}{=}\Delta_{ri}{+}α{}s_{ri\max}$替换$\tau_{ri}$，于是有${γ}_{ri}{=}\left(\cfrac{α\left({1{-}s_{ri\max}}\right)}{α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)}\right){\left(\cfrac{\left({\Delta_{ri}{+}αs_{ri\max}}\right)\left({1{-}s_{ri\max}}\right)}{s_{ri\max}\left({α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)}\right)}\right)}^{-\frac{\Delta_{ri}{+}αs_{ri\max}}{α}}$ 
> >    <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250411233848876.png" alt="image-20250411233848876" width=520 />  
> > 2. 定义域的分析：
> >    - 由前提可知$α{>}1$，并且分数介于$(0,1)$间故${s}_{ri\max}{∈}(0,1)$
> >    - 由引理$\text{4.1.2}$的证明过程可知$γ_{ri}{>}0$故$α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)\text{＞}0$，即$s_{ri\max}{∈}\left(0,1{-}\cfrac{\Delta_{ri}}{α}\right)$  
> > 3. 在引理$\text{4.1.2}$的证明过程中，已经说明了$\lim\limits_{{τ_{ri}\searrowα{s}_{ri\max}}}{γ}_{ri}{=}\lim\limits_{{\Delta_{ri}\searrow0}}{γ}_{ri}{=}1$，下面分析其它两个边界的极限
> >
> > :two:极限分析$\lim\limits_{{s_{ri\max}{\searrow}0}}γ_{ri}{=}0$
> >
> > 1. 线性部分：$\lim\limits_{{s_{ri\max}{\searrow}0}}\left(\cfrac{α\left({1{-}s_{ri\max}}\right)}{α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)}\right){=}\cfrac{α}{α{-}\Delta_{ri}}$
> > 2. 指数部分：$\lim\limits_{{s_{ri\max}{\searrow}0}}{\left(\cfrac{\left({\Delta_{ri}{+}αs_{ri\max}}\right)\left({1{-}s_{ri\max}}\right)}{s_{ri\max}\left({α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)}\right)}\right)}^{-\frac{\Delta_{ri}{+}αs_{ri\max}}{α}}$
> >    - 底数部分：$\lim\limits_{{s_{ri\max}{\searrow}0}}{\left(\cfrac{\left({\Delta_{ri}{+}αs_{ri\max}}\right)\left({1{-}s_{ri\max}}\right)}{s_{ri\max}\left({α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)}\right)}\right)}{=}\lim\limits_{{s_{ri\max}{\searrow}0}}{\left(\cfrac{\Delta_{ri}}{s_{ri\max}\left({α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)}\right)}\right)}{=}{+}∞$ 
> >    - 幂部分：$\lim\limits_{{s_{ri\max}{\searrow}0}}\left(-\cfrac{\Delta_{ri}{+}αs_{ri\max}}{α}\right){=}\left(-\cfrac{\Delta_{ri}}{α}\right)$
> >    - 原式即为：$\lim\limits_{{s_{ri\max}{\searrow}0}}{\left(\cfrac{s_{ri\max}\left({α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)}\right)}{\Delta_{ri}}\right)}^{\frac{\Delta_{ri}}{α}}{=}0^{{\Delta_{ri}}/{α}}{=}0$，故二者结合起来极限为$\lim\limits_{{s_{ri\max}{\searrow}0}}γ_{ri}{=}0$   
> >
> > :three:极限分析$\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}γ_{ri}{=}1{-}\cfrac{\Delta_{ri}}{α}$ 
> >
> > 1. 令$y{=}α{-}\Delta_{ri}{-}αs_{ri\max}$，进行如下代换
> >    - 先代入$\Delta_{ri}{+}αs_{ri\max}{=}α{-}y$，于是有${γ}_{ri}{=}\left(\cfrac{α\left({1{-}s_{ri\max}}\right)}{y}\right){\left(\cfrac{(α{-}y)\left({1{-}s_{ri\max}}\right)}{ys_{ri\max}}\right)}^{-\frac{α{-}y}{α}}$ 
> >    - 再代入${}s_{ri\max}{=}\cfrac{α{-}\Delta_{ri}{-}y}{α}$，于是有${γ}_{ri}{=}\left(\cfrac{\Delta_{ri}{+}y}{y}\right)\left(\cfrac{(\Delta_{ri}{+}y)(α{-}y)}{y(α{-}\Delta_{ri}{-}y)}\right)^{\frac{y}{α}{-}1}{=}\left(\cfrac{\Delta_{ri}{+}y}{y}\right)^{\frac{y}{α}}\left(\cfrac{α{-}y}{α{-}\Delta_{ri}{-}y}\right)^{\frac{y}{α}{-}1}$ 
> > 2. 代回原极限，可得$\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}γ_{ri}{=}\lim\limits_{y{\searrow}0}γ_{ri}{=}\lim\limits_{y{\searrow}0}\left(\cfrac{\Delta_{ri}{+}y}{y}\right)^{\frac{y}{α}}\lim\limits_{y{\searrow}0}\left(\cfrac{α{-}y}{α{-}\Delta_{ri}{-}y}\right)^{\frac{y}{α}{-}1}$ 
> >    - 其中$\lim\limits_{y{\searrow}0}\left(\cfrac{α{-}y}{α{-}\Delta_{ri}{-}y}\right)^{\frac{y}{α}{-}1}{=}\left(\cfrac{α}{α{-}\Delta_{ri}}\right)^{{-}1}{=}1{-}\cfrac{\Delta_{ri}}{α}$ 
> >    - 其中$\lim\limits_{y{\searrow}0}\left(\cfrac{\Delta_{ri}{+}y}{y}\right)^{\frac{y}{α}}$，令$D{=}\left(\cfrac{\Delta_{ri}{+}y}{y}\right)^{\frac{y}{α}}$
> >      - 于是$\ln{D}{=}\cfrac{y}{α}\ln{\left(\cfrac{\Delta_{ri}{+}y}{y}\right)}{=}\cfrac{\ln{\left(\cfrac{\Delta_{ri}{+}y}{y}\right)}}{\left(\cfrac{α}{y}\right)}$，这是一个$\cfrac{{+}∞}{{+}∞}$极限
> >      - 所以$\lim\limits_{y{\searrow}0}\ln\left(\cfrac{\Delta_{ri}{+}y}{y}\right)^{\frac{y}{α}}{=}\lim\limits_{y{\searrow}0}\left(\cfrac{\cfrac{1}{\Delta_{ri}{+}y}{-}\cfrac{1}{y}}{{-}\cfrac{α}{y^2}}\right){=}\lim\limits_{y{\searrow}0}\left(\cfrac{y^2{-}y(\Delta_{ri}{+}y)}{{-}α(\Delta_{ri}{+}y)}\right){=}0$，所以$\lim\limits_{y{\searrow}0}\left(\cfrac{\Delta_{ri}{+}y}{y}\right)^{\frac{y}{α}}{=}1$ 
> >    - 合并后$\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}γ_{ri}{=}1{-}\cfrac{\Delta_{ri}}{α}$ 
> >
> > :four:最值的分析
> >
> > 1. 进行代换$A{=}\left(\cfrac{α\left({1{-}s_{ri\max}}\right)}{α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)}\right)$和$B{=}{\left(\cfrac{\left({\Delta_{ri}{+}αs_{ri\max}}\right)\left({1{-}s_{ri\max}}\right)}{s_{ri\max}\left({α{-}\left({\Delta_{ri}{+}αs_{ri\max}}\right)}\right)}\right)}$以及$k{=}{-}\cfrac{\Delta_{ri}{+}αs_{ri\max}}{α}{=}{-}\cfrac{\tau_{ri}}{α}$ 
> > 2. 于是$γ_{ri}{=}AB^k$即$\lnγ_{ri}{=}\ln{A}{+}k\ln{B}$，现对$γ_i$的偏导进行拆解
> >    - 首先$\cfrac{\partialγ_{ri}}{\partial{}s_{ri\max}}{=}\cfrac{\partial\lnγ_{ri}}{\partial{}s_{ri\max}}γ_{ri}$，其中$\cfrac{\partial\lnγ_{ri}}{\partial{}s_{ri\max}}{=}\cfrac{\partial\ln{A}}{\partial{}s_{ri\max}}{+}\cfrac{\partial{}(k\ln{}B)}{\partial{}s_{ri\max}}$  
> >    - 然而$\cfrac{\partial{}(k\ln{}B)}{\partial{}s_{ri\max}}{=}\cfrac{\partial{}k}{\partial{}s_{ri\max}}\ln{}B{+}k\cfrac{\partial{}\ln{}B}{\partial{}s_{ri\max}}$，其中$\cfrac{\partial{}k}{\partial{}s_{ri\max}}{=}\cfrac{\partial{}\left({-}\cfrac{\Delta_{ri}{+}αs_{ri\max}}{α}\right)}{\partial{}s_{ri\max}}{=}{-}1$ 
> >    - 所以$\cfrac{\partialγ_{ri}}{\partial{}s_{ri\max}}{=}γ_{ri}\left(\cfrac{\partial\ln{A}}{\partial{}s_{ri\max}}{+}k\cfrac{\partial{}\ln{}B}{\partial{}s_{ri\max}}{-}\ln{B}\right)$
> >      - 其中$\cfrac{\partial\ln{A}}{\partial{}s_{ri\max}}{=}\cfrac{\Delta_{ri}}{(α{-}(\Delta_{ri}{+}α{}s_{ri\max}))(1{-}s_{ri\max})}$
> >      - 以及$\cfrac{\partial{}\ln{}B}{\partial{}s_{ri\max}}{=}\cfrac{α}{\Delta_{ri}{+}α{}s_{ri\max}}{+}\cfrac{α}{α{-}(\Delta_{ri}{+}α{}s_{ri\max})}{-}\cfrac{1}{s_{ri\max}}{-}\cfrac{1}{1{-}s_{ri\max}}$ 
> > 3. 综上最终得到偏导数$\cfrac{\partialγ_{ri}}{\partial{}s_{ri\max}}{=}γ_{ri}\left( \cfrac{\Delta_{ri}}{α s_{ri\max}(1{-}s_{ri\max})}{-}\ln\left(\cfrac{(\Delta_{ri}{+}αs_{ri\max})(1{-}s_{ri\max})}{s_{ri\max}\left(α{-}(\Delta_{ri}{+}αs_{ri\max}\right)} \right)\right)$  
> >    - 引理$\text{4.1.2}$的证明中已说明了$γ_{ri}{∈}\left(s_{ri\max},1\right)$恒大于$0$，故只需考虑后面一堆的正负
> >    - $G(α,s_{ri\max}){=}\left(\left(\cfrac{\Delta_{ri}}{α s_{ri\max}(1{-}s_{ri\max})}\right){-}\ln\left(\cfrac{(\Delta_{ri}{+}αs_{ri\max})(1{-}s_{ri\max})}{s_{ri\max}\left(α{-}(\Delta_{ri}{+}αs_{ri\max}\right)}\right)\right)$，且$α{>}1$和$s_{ri\max}{∈}\left(0,1{-}\cfrac{\Delta_{ri}}{α}\right)$ 
> >      <img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/image-20250412220907160.png" alt="image-20250412220907160" width=529 /> 
> > 4. 对$G(α,s_{ri\max})$的分析：
> >    - 求极限$\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}G(α,s_{ri\max})$，原式可写作$A{-}B{+}C$
> >      - $A{=}\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}\left(\cfrac{\Delta_{ri}}{αs_{ri\max}(1{-}s_{ri\max})}\right){=}\cfrac{α}{α{-}\Delta_{ri}}$ 
> >      - $B{=}\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}\ln\left((\Delta_{ri}{+}αs_{ri\max})(1{-}s_{ri\max})\right){=}\ln{\Delta_{ri}}$ 
> >      - $C{=}\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}\ln\left(s_{ri\max}\left(α{-}(\Delta_{ri}{+}αs_{ri\max}\right)\right){=}\ln\left(1{-}\cfrac{\Delta_{ri}}{α}\right){+}\lim\limits_{{α{\searrow}(\Delta_{ri}{+}αs_{ri\max})}}\ln\left(α{-}(\Delta_{ri}{+}αs_{ri\max}\right){=}{-}{∞}$
> >    - 所以$\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}G(α,s_{ri\max}){=}{-}∞$
> > 5. 回到对$γ_{ri}$的分析
> >    - 已证$\cfrac{\partialγ_{ri}}{\partial{}s_{ri\max}}$在${s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}$时为负无穷，而且$\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}γ_{ri}{=}1{-}\cfrac{\Delta_{ri}}{α}$大于$\lim\limits_{{s_{ri\max}{\searrow}0}}γ_{ri}{=}0$ 
> >    - 所以在$\Delta_{ri}$固定的情况下$γ_{ri}$必定不是单调递增的，并在$s_{ri\max}{∈}\left(0,1{-}\cfrac{\Delta_{ri}}{α}\right)$上取到一个最大值$(γ_{ri})_{\max}$  
> >    - 考虑到最大值一定大于$\lim\limits_{{s_{ri\max}{\nearrow}1{-}\frac{\Delta_{ri}}{α}}}γ_{ri}{=}1{-}\cfrac{\Delta_{ri}}{α}$并且引理$\text{4.1.2}$已证$γ_{ri}{∈}(s_{ri\max},1)$，所以$(γ_{ri})_{\max}{∈}\left(1{-}\cfrac{\Delta_{ri}}{α},1\right)$
>
> ## $\textbf{2.2. }$如何控制上界
>
> > :one:先假设一个$\cfrac{δ}{2}$的界：希望对所有$S_i{≠}S^*$中的$q_r$有$\Pr\left[\forall{q_r}{,}{σ(\hat{\mathbf{s}}_{ri}){≥}α s_{ri\max}{+}\Delta_{ri}}\right]{≤}\cfrac{δ}{2}$ 
> >
> > 1. 对所有$N{-}1$个$S_i$，每个$S_i$中有$m_q$个$q_r$，所有$S_i$中一共有$(N{-}1)m_q$个$q_r$
> > 2. 设定事件$E_r$为$E_r{=}\{q_r|{σ(\hat{\mathbf{s}}_{ri}){≥}α s_{ri\max}{+}\Delta_{ri}}\}$，于是原假设即为$\Pr\left[E_1{∪}E_1{∪}...{∪}E_{(N{-}1)m_q}\right]{≤}\cfrac{δ}{2}$ 
> > 3. 考虑到$\Pr\left[E_1{∪}E_2{∪}...{∪}E_{(N{-}1)m_q}\right]{≤}\Pr[E_1]{+}\Pr[E_2]{+}{\cdots}{+}E\left[E_{(N{-}1)m_q}\right]$   
> > 4. 所以更进一步的，我们希望对单个$q_r$有$(N{-}1)m_q\Pr[E_r]{≤}\cfrac{δ}{2}$，即$\Pr\left[{σ(\hat{\mathbf{s}}_{ri}){≥}α s_{ri\max}{+}\Delta_{ri} }\right]{≤}\cfrac{δ}{2(N{-}1)m_q}$ 
> >
> > :two:应用引理$\text{4.1.2}$：求出使得该界成立的$L$的范围
> >
> > 1. 由于引理可得$\Pr\left[{σ(\hat{\mathbf{s}}_{ri}){≥}α s_{ri\max}{+}\Delta_{ri} }\right]{≤}mγ_{ri}^L$，更进一步假定有$mγ_{ri}^L{≤}m((γ_{ri})_{\max})^L{≤}\cfrac{δ}{2(N{-}1)m_q}$
> > 2. 所以由$m((γ_{ri})_{\max})^L{≤}\cfrac{δ}{2(N{-}1)m_q}$可得$L\ln{\left((γ_{ri})_{\max}\right)}{≤}\ln\left(\cfrac{δ}{2(N{-}1)m_qm}\right)$ 
> > 3. 由于$(γ_{ri})_{\max}{∈}(0,1)$所以$\ln{\left((γ_{ri})_{\max}\right)}{<}0$，解得$L{≥}\cfrac{\ln\left(\cfrac{δ}{2(N{-}1)m_qm}\right)}{\ln{\left((γ_{ri})_{\max}\right)}}$，即此时有$\Pr\left[\forall{q_r}{,}{σ(\hat{\mathbf{s}}_{ri}){≥}α s_{ri\max}{+}\Delta_{ri}}\right]{≤}\cfrac{δ}{2}$  
>
> ## $\textbf{2.3. }$如何控制下界
>
> > :one:也假设一个$\cfrac{δ}{2}$的界：希望对所有$S^*$中的$q_r$有$\Pr\left[\forall{q_r}{,}{σ(\hat{\mathbf{s}}_r^*){≤}βs_{r\max}^*{-}\Delta_{ri}}\right]{≤}\cfrac{δ}{2}$ 
> >
> > 1. 只有一个$S^*$，$S^*$中有$m_q$个$q_r$，即所有$S^*$中一共有$m_q$个$q_r$
> > 2. 设定事件$F_r$为$F_r{=}\{q_r|σ(\hat{\mathbf{s}}_r^*){≤}βs_{r\max}^*{-}\Delta_{ri}\}$，于是原假设即为$\Pr\left[F_1{∪}E_1{∪}...{∪}F_{m_q}\right]{≤}\cfrac{δ}{2}$ 
> > 3. 考虑到$\Pr\left[F_1{∪}F_2{∪}...{∪}F_{(N{-}1)m_q}\right]{≤}\Pr[F_1]{+}\Pr[F_2]{+}{\cdots}{+}F\left[F_{(N{-}1)m_q}\right]$  
> > 4. 所以更进一步的，我们希望对单个$q_r$有$m_q\Pr[E_r]{≤}\cfrac{δ}{2}$，即$\Pr\left[{σ(\hat{\mathbf{s}}_r^*){≤}βs_{r\max}^*{-}\Delta_{ri}}\right]{≤}\cfrac{δ}{2m_q}$ 
> >
> > :two:应用引理$\text{4.1.2}$：求出使得该界成立的$L$的范围
> >
> > 1. 由于引理可得$\Pr[{σ\left(\hat{\mathbf{s}}_r^*\right){≤}βs_{r\max}^*{-}\Delta_{ri}}]{≤}2{e}^{{-}{2L}{\Delta_{ri}}^{2}/{β}^{2}}$，所以假定有$2{e}^{{-}{2L}{\Delta_{ri}}^{2}/{β}^{2}}{≤}\cfrac{δ}{2m_q}$
> > 2. 解得$L{≥}\cfrac{β^2\ln\left(\cfrac{4m_q}{\delta}\right)}{2\Delta_{ri}^2}$，即此时有$\Pr\left[\forall{q_r}{,}{σ(\hat{\mathbf{s}}_r^*){≤}βs_{r\max}^*{-}\Delta_{ri}}\right]{≤}\cfrac{δ}{2}$  
> >
> > :three:取最小的$L$，让两个界依然成立
> >
> > 1. $\min{L}{=}\min\left\{L{∈}\left[\cfrac{\ln\left(\cfrac{δ}{2(N{-}1)m_qm}\right)}{\ln{\left((γ_{ri})_{\max}\right)}},{+}∞\right){∩}\left[\cfrac{β^2\ln\left(\cfrac{4m_q}{\delta}\right)}{2\Delta_{ri}^2},{+}∞\right)\right\}{=}\max\left\{\cfrac{\ln\left(\cfrac{δ}{2(N{-}1)m_qm}\right)}{\ln{\left((γ_{ri})_{\max}\right)}},\cfrac{β^2\ln\left(\cfrac{4m_q}{\delta}\right)}{2\Delta_{ri}^2}\right\}$
> > 2. 此时$\Pr\left[\forall{q_r}{,}{σ(\hat{\mathbf{s}}_{ri}){≥}αs_{ri\max}{+}\Delta_{ri}}\right]{≤}\cfrac{δ}{2}$和$\Pr\left[\forall{q_r}{,}{σ(\hat{\mathbf{s}}_r^*){≤}βs_{r\max}^*{-}\Delta_{ri}}\right]{≤}\cfrac{δ}{2}$同时成立
> > 3. 即在这个$L$的设置下，$σ(\hat{\mathbf{s}}_{ri})$高概率在上界$αs_{ri\max}{+}\Delta_{ri}$下，$σ(\hat{\mathbf{s}}_r^*)$高概率在下界$βs_{r\max}^*{-}\Delta_{ri}$上