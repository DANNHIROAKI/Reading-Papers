[原论文](https://doi.org/10.48550/arXiv.2210.15748)

@[toc]
# $\textbf{1. }$引理$\textbf{4.1.2/4.1.3}$的内容

> :one:一些基础的符号
>
> 1. 基本符号：
>    | 符号 | 含义                                                         |
>    | :--: | :----------------------------------------------------------- |
>    | $Q$  | 查询集，此处只考虑一个$Q$并且$Q{=}\{q_1,q_2,...,q_{m_q}\}$，记其子向量为$q_r{∈}Q$ |
>    | $S$  | 目标集，此处有$N$个$S$并设每个${\mid}S{\mid}{=}m$(常数)，$S^*$与$Q$评分最大(其元素为$x^*_j{∈}S^*$)，其余都记为$S_i$(其元素为$x_{ij}{∈}S_i$) |
>    | $N$  | 目标集的集即$D\text{=}\{S_1,...,S_N\}$，记其元素为$S_i{∈}N$  |
> 2. 相似度及其集合
>    |                符号                 | 含义                                                         |
>    | :---------------------------------: | :----------------------------------------------------------- |
>    |         ${\mathbf{s}}_{ri}$         | $q_r$与$\forall{x_{ij}}{∈}S_i$的精确相似度的集合，即${\mathbf{s}}_{ri}{=}\{\text{Sim}(q_r,x_{i1}),...,\text{Sim}(q_r,x_{im})\}$ |
>    |       $\hat{\mathbf{s}}_{ri}$       | $q_r$与$\forall{x_{ij}}{∈}S_i$的近似相似度的集合，即$\hat{\mathbf{s}}_{ri}{=}\{\hat{\text{Sim}}(q_r,x_{i1}),...,\hat{\text{Sim}}(q_r,x_{im})\}$ |
>    |       $s_{rij}$和$s_{ri\max}$       | $q_r$与$\forall{x_{ij}}{∈}S_i$中的$s_{rij}{=}\text{Sim}(q_r,x_{ij})$，其最大值记作$s_{ri\max}{=}\max({\mathbf{s}}_{ri})$且此时向量记为$x_{ij}^*$ |
>    | $\hat{s}_{rij}$和$\hat{s}_{ri\max}$ | $q_r$与$\forall{x_{ij}}{∈}S_i$中的$\hat{s}_{rij}{=}\hat{\text{Sim}}(q_r,x_{ij})$，其最大值记作$\hat{s}_{ri\max}{=}\max(\hat{{\mathbf{s}}}_{ri})$且此时向量记为$x_{ij}^*$ |
>    - 由于内部聚合仅考虑固定的$q_r{∈}Q$和$S_i{∈}D$，${\mathbf{s}}_{ri}/\hat{\mathbf{s}}_{ri}/s_{rij}/\hat{s}_{rij}/s_{ri\max}/\hat{s}_{ri\max}$分别简写为$\mathbf{s}/\hat{\mathbf{s}}/s_{j}/\hat{s}_{j}/s_{\max}/\hat{s}_{\max}$
> 3. 聚合符号：
>    - 即$σ$或$\max$(最大值聚合)，以$\sigma$为例对不同集合的聚合记作$σ({\mathbf{s}})/σ(\hat{\mathbf{s}})$ 
>    - 这里还需一个条件就是满足$σ$是$(α,β)$-极大的
>
> :two:$\gamma_{r,i}$函数：基于同样考量去掉角标，则$γ{=}γ(s_{\max},τ){=}\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α{-}τ}\right){\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}^{\frac{τ}{α}}$  
>
> 1. 单调：处于$\left({{s_{\max}},1}\right)$区间中，并且随$s_{\max}$递增随$τ$递减
> 2. 极限：$γ$存在单侧极限，$τ$从高处接近$α{s_{\max}}$时$\mathop{\lim}\limits_{{τ\searrowα{s}_{\max}}}γ\text{=}1$，$τ$从低处接近$α$时$\mathop{\lim}\limits_{{τ\nearrowα}}γ\text{=}{s}_{\max}$ 
>
> :three:结论：令$τ$表示阈值满足$τ\text{∈}(α{s_{\max}},α)$，其差值记为$\Delta{=}τ{-}α{}s_{\max}$
>
> 1. 引理$\text{4.1.2}$：$\Pr\left\lbrack{σ\left(\widehat{\mathbf{s}}\right)\text{≥}α{s}_{\max}\text{+}\Delta}\right\rbrack\text{≤}m{γ}^{L}$即$\Pr\left\lbrack{σ\left(\widehat{\mathbf{s}}\right)\text{≥}τ}\right\rbrack\text{≤}m{γ}^{L}$，对近似相似度聚合后，大概率不超过理论上界
> 2. 引理$\text{4.1.3}$：$\Pr\left\lbrack{σ\left(\widehat{\mathbf{s}}\right)\text{≤}β{s}_{\max}\text{{-}}\Delta}\right\rbrack\text{≤}2{e}^{{-}{2L}{\Delta}^{2}/{β}^{2}}$，对近似相似度聚合后，大概率不低于理论下界

# $\textbf{2. }$引理$\textbf{4.1.2}$的证明

> ## $\textbf{2.1. }\boldsymbol{γ}$函数是怎么来的
>
> > :one:对$\Pr\left\lbrack{σ\left(\widehat{\mathbf{s}}\right)\text{≥}τ}\right\rbrack\text{≤}m{γ}^{L}$的变换
> >
> > 1. 应用$\text{Chernoff}$界限$\text{Pr}\left[σ\left(\hat{\mathbf{s}}\right){≥}τ\right]{=}\Pr\left[e^{tσ\left(\hat{\mathbf{s}}\right)}{≥}e^{tτ}\right]{≤}\cfrac{\mathbb{E}\left[e^{tσ\left(\hat{\mathbf{s}}\right)}\right]}{e^{tτ}}(t{>}0)$，做如下变换
> >    - 考虑到$σ$是$(α,β)\text{-}$极大的，即$\displaystyleβ\max\left(\hat{\mathbf{s}}\right){≤}σ\left(\widehat{\mathbf{s}}\right){≤}α\max\left(\hat{\mathbf{s}}\right)$，所以分子$\mathbb{E}\left[e^{tσ\left(\hat{\mathbf{s}}\right)}\right]{≤}\mathbb{E}\left[e^{tα\max\left(\hat{\mathbf{s}}\right)}\right]$  
> >    - 利用$\displaystyle\max_jX_j{≤}\sum_{j}X_j$，于是$\displaystyle\mathbb{E}\left[e^{tα\max\left(\hat{\mathbf{s}}\right)}\right]{=}\mathbb{E}\left[\max_j{\left(e^{tα\hat{s}_j}\right)}\right]{≤}\mathbb{E}\left[\sum_{j=1}^{m}\left(e^{tα\hat{s}_j}\right)\right]{=}\sum_{j=1}^{m}\mathbb{E}\left[e^{tα\hat{s}_j}\right]$ 
> >    
> > 2. 可以将$\hat{s}_{\max}/\hat{s}_j$作为${s}_{\max}/{s}_j$的无偏二项估计，即$\hat{s}_{\max}{=}\cfrac{1}{L} \mathcal{B}\left(s_{\max },L\right)$和$\hat{s}_j{=}\cfrac{1}{L} \mathcal{B}\left(s_j,L\right)$
> >    - <mark>为何偏偏是二项估计</mark>：
> >      - 视$q_r,x_{i j}$每次分桶为一次独立伯努利试验，二者碰撞与否为实验的两个结果
> >      - 基于本文给出的$\text{LSH}$的定义，单次分桶过程中$q_r$和$x_{i j}$碰撞的概率为$s_j{=}\text{Sim}\left(q_r,x_{ij}\right)$ 
> >      - 而一共需要进行$L$次分桶，所以总的碰撞次数服从分布$\text{Count}\left(q_r, x_{ij}\right){\sim}\mathcal{B}\left(L, s_j\right)$ 
> >      
> >    - 为何一定是无偏的：
> >
> >      - 用碰撞率估计相似度，则$\hat{s}_j{=}\cfrac{\text{Count}\left(q_r, x_{i j}\right)}{L}$
> >      - 于是$\mathbb{E}\left[\hat{s}_j\right]{=}\cfrac{1}{L}\mathbb{E}\left[\text{Count}\left(q_r,x_{i j}\right)\right]{=}\cfrac{1}{L}Ls_j{=}s_j$，即$\hat{s}_j$为$s_j$无偏估计
> >    
> >    - 几种无偏估计的含义：
> >      |              随机变量               | 含义                                                         |
> >      | :---------------------------------: | :----------------------------------------------------------- |
> >      |             $\hat{s}_j$             | 真实相似度${{s}_j{∈}{\mathbf{s}}}$的无偏估计，即$\hat{s_{j}}{=}\hat{{\text{Sim}}}(q_{r},x_{ij})$ |
> >      | $\max\left(\hat{\mathbf{s}}\right)$ | 先对所有$\forall{{s}_j{∈}{\mathbf{s}}}$求估计值$\forall{\hat{s}_j{∈}\hat{\mathbf{s}}}$，再求最大值$\max\left(\hat{\mathbf{s}}\right){=}\max\{\hat{s}_1,\hat{s}_2,...,\hat{s}_{m}\}$ |
> >      |          $\hat{s}_{\max}$           | 先对所有$\forall{{s}_j{∈}{\mathbf{s}}}$求最大值$s_{\max}$，再求估计值$\hat{s}_{\max}$ |
> >    
> > 3. 基于对二项分布的假设和分析，继续改写
> >    - 由二项分布的矩生成函数函数得$\mathbb{E}\left[e^{tα\hat{s}_j}\right]{=}\left(1{-}s_j{+}s_je^{\frac{tα}{L}}\right)^L$以及$\mathbb{E}\left[e^{tα\hat{s}_{\max}}\right]{=}\left(1{-}s_{\max}{+}s_{\max}{}e^{\frac{tα}{L}}\right)^L$
> >    - 由于$e^{\frac{tα}{L}}{>}1$，所以$\mathbb{E}\left[e^{tα\hat{s}_j}\right]$关于$s_j$递增而递增，所以由$s_{\max}{≥}s_j$可得$\mathbb{E}\left[e^{tα\hat{s}_j}\right]{≤}\mathbb{E}\left[e^{tα\hat{s}_{\max}}\right]$
> >    - 由此求和后$\displaystyle\sum_{j=1}^{m}\mathbb{E}\left[e^{tα\hat{s}_j}\right]{≤}\displaystyle\sum_{j=1}^{m}\mathbb{E}\left[e^{tα\hat{s}_{\max}}\right]{=}m\mathbb{E}\left[e^{tα\hat{s}_{\max}}\right]{=}m\left(1{-}s_{\max}{+}s_{\max}{}e^{\frac{tα}{L}}\right)^L$ 
> >    
> > 4. 全部代回原式得$\text{Pr}\left[σ\left(\hat{\mathbf{s}}\right){≥}τ\right]{≤}\cfrac{m\left(1{-}s_{\max}{+}s_{\max}{}e^{\frac{tα}{L}}\right)^L}{e^{tτ}}(t{>}0)$
> >
> > :two:确定$f(t){=}\cfrac{m\left(1{-}s_{\max}{+}s_{\max}{}e^{\frac{tα}{L}}\right)^L}{e^{tτ}}$在$t{=}t^*{>}0$时取得的下界
> >
> > 1. 令$g(t){=}\cfrac{\ln{f(t)}}{m}{=}L\ln{\left(1{-}s_{\max}{+}s_{\max}{}e^{\frac{tα}{L}}\right)}{-}tτ$，则$\cfrac{dg(t)}{dt}{=}\cfrac{α{s_{\max}}e^{\frac{tα}{L}}}{1{-}s_{\max}{+}s_{\max}e^{\frac{tα}{L}}}{-}τ$
> > 2. 当$\cfrac{dg(t)}{dt}{=}0$时$\cfrac{α{s_{\max}}e^{\frac{tα}{L}}}{1{-}s_{\max}{+}s_{\max}e^{\frac{tα}{L}}}{=}τ$，即$e^{\frac{tα}{L}}{=}\cfrac{τ(1{-}s_{\max})}{s_{\max}(α{-}τ)}$，所以$t^*{=}\cfrac{L}{α}\ln{\left(\cfrac{τ(1{-}s_{\max})}{s_{\max}(α{-}τ)}\right)}$ 
> > 3. 让指数项分子减分母则为$τ{-}α{s_{\max}}$，而$τ\text{∈}(α{s_{\max}},α)$所以$τ{-}α{s_{\max}}{>}0$，于是$t^*{>}0$符合条件
> > 4. 代回$f(t)$则有$\Pr\left\lbrack{σ\left(\widehat{\mathbf{s}}\right)\text{≥}τ}\right\rbrack\text{≤}m\left(\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α{-}τ}\right){\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}^{\frac{τ}{α}}\right){=}mγ^L$ 
> >
> > :three:在此基础上还需说明$γ{<}1$恒成立，因为此时$mγ^L$才收敛，引理(概率的界)才有意义
>
> ## $\textbf{2.2. }$对$\boldsymbol{γ}$函数的分析
>
> > <img src="https://i-blog.csdnimg.cn/direct/3e7d62075c144b44b8cd692381330134.png"  width=500 /> 
> >
> > :one:当${τ{\searrow}α{s}_{\max}}$时$γ$的极限：$\mathop{\lim}\limits_{{τ\searrowα{s}_{\max}}}\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α{-}τ}\right){\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}^{\frac{τ}{α}}\text{=}1$ 
> >
> > 1. 线性部分：$\mathop{\lim}\limits_{{τ\searrowα{s}_{\max}}}\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α{-}τ}\right){=}1$
> >    - 用$\Delta{=}τ{-}α{}s_{\max}$替换，则${τ{\searrow}α{s}_{\max}}$也变为了$\Delta{\searrow}0$
> >    - 则原极限为$\mathop{\lim}\limits_{\Delta{\searrow}0}\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α{-}τ}\right){=}\mathop{\lim}\limits_{\Delta{\searrow}0}\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α(1{-}s_{\max}){-}\Delta}\right){=}\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α\left({1{-}{s_{\max}}}\right)}\right){=}1$ 
> > 2. 指数部分：$\mathop{\lim}\limits_{{τ\searrowα{s}_{\max}}}{\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}^{\frac{τ}{α}}\text{=}1$ 
> >    - 同样用$\Delta{=}τ{-}α{}s_{\max}$替换，则${τ{\searrow}α{s}_{\max}}$也变为了$\Delta{\searrow}0$
> >    - 则原极限为$\mathop{\lim}\limits_{\Delta{\searrow}0}{\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}\text{=}\mathop{\lim}\limits_{\Delta{\searrow}0}{\left(\cfrac{{s_{\max}}\left(α(1{-}s_{\max}){-}\Delta\right)}{(α{s_{\max}{+}\Delta})\left({1{-}{s_{\max}}}\right)}\right)}\text{=}{\left(\cfrac{α{s_{\max}}\left(1{-}s_{\max}\right)}{α{s_{\max}}\left({1{-}{s_{\max}}}\right)}\right)}{=}1$ 
> >    - 另外$\mathop{\lim}\limits_{{τ\searrowα{s}_{\max}}}\left(\cfrac{τ}{α}\right){=}\mathop{\lim}\limits_{\Delta{\searrow}0}\left(\cfrac{α{s_{\max}}\text{+}\Delta}{α}\right){=}s_{\max}{+}\mathop{\lim}\limits_{\Delta{\searrow}0}\left(\cfrac{\Delta}{α}\right){=}s_{\max}$(非无穷)
> >    - 所以最后原极限$\mathop{\lim}\limits_{{τ\searrowα{s}_{\max}}}{\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}^{\frac{τ}{α}}{=}1^{s_{\max}}{=}1$ 
> >
> > :two:当${τ{\nearrow}α}$时$γ$的极限：$\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α{-}τ}\right){\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}^{\frac{τ}{α}}{=}s_{\max}$  
> >
> > 1. 拆分极限：原极限拆为$\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{α^{\frac{α}{τ}}}{τ(1{-}s_{\max})^{1{-}\frac{α}{τ}}}\right)$和$\mathop{\lim}\limits_{{τ{\nearrow}α}}{\left((α{-}τ)^{1{-}\frac{α}{τ}}\right)}$，还有常数项$s_{\max}$
> >    - 合并原式为大指数项，即$\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α{-}τ}\right)^{\frac{α}{τ}}\right)^{\frac{τ}{α}}{\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}^{\frac{τ}{α}}{=}\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{s_{\max}(α{-}τ)^{1{-}\frac{α}{τ}}α^{\frac{α}{τ}}}{τ(1{-}s_{\max})^{1{-}\frac{α}{τ}}}\right)^{\frac{τ}{α}}$ 
> >    - 考虑到$\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{τ}{α}\right){=}1$，所以$\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{s_{\max}(α{-}τ)^{1{-}\frac{α}{τ}}α^{\frac{α}{τ}}}{τ(1{-}s_{\max})^{1{-}\frac{α}{τ}}}\right)^{\frac{τ}{α}}{=}\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{s_{\max}(α{-}τ)^{1{-}\frac{α}{τ}}α^{\frac{α}{τ}}}{τ(1{-}s_{\max})^{1{-}\frac{α}{τ}}}\right)$ 
> >    - 进一步拆分为两部分，则$\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{s_{\max}(α{-}τ)^{1{-}\frac{α}{τ}}α^{\frac{α}{τ}}}{τ(1{-}s_{\max})^{1{-}\frac{α}{τ}}}\right){=}s_{\max}\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{α^{\frac{α}{τ}}}{τ(1{-}s_{\max})^{1{-}\frac{α}{τ}}}\right)\mathop{\lim}\limits_{{τ{\nearrow}α}}{\left((α{-}τ)^{1{-}\frac{α}{τ}}\right)}$ 
> > 2. 左边部分：$\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{α^{\frac{α}{τ}}}{τ(1{-}s_{\max})^{1{-}\frac{α}{τ}}}\right){=}1$
> >    - 令$\varepsilon{=}α{-}τ$，则${τ{\nearrow}α}$变为$\varepsilon{\searrow}0$
> >    - 于是$\mathop{\lim}\limits_{{τ{\nearrow}α}}\left(\cfrac{α^{\frac{α}{τ}}}{τ(1{-}s_{\max})^{1{-}\frac{α}{τ}}}\right){=}\mathop{\lim}\limits_{{\varepsilon{\searrow}0}}\left(\cfrac{α^{\frac{α}{α{-}\varepsilon}}}{(α{-}\varepsilon)(1{-}s_{\max})^{1{-}\frac{α}{α{-}\varepsilon}}}\right){=}\mathop{\lim}\limits_{{\varepsilon{\searrow}0}}\left(\cfrac{α^{\frac{α}{α}}}{α(1{-}s_{\max})^{1{-}\frac{α}{α}}}\right){=}1$ 
> > 3. 右边部分：$\mathop{\lim}\limits_{{τ{\nearrow}α}}{\left((α{-}τ)^{1{-}\frac{α}{τ}}\right)}{=}1$
> >    - 同样令$\varepsilon{=}α{-}τ$则${τ{\nearrow}α}$变为$\varepsilon{\searrow}0$，则原极限为$\mathop{\lim}\limits_{{t{\searrow}0}}\left({t^{1{-}\frac{α}{α{-}\varepsilon}}}\right){=}\mathop{\lim}\limits_{{t{\searrow}0}}\left(t^{\frac{t}{t{-}α}}\right)$，再令$y{=}t^{\frac{t}{t{-}α}}$则原极限变为$\mathop{\lim}\limits_{{t{\searrow}0}}y$
> >    - 由于$\mathop{\lim}\limits_{{t{\searrow}0}}y{=}\mathop{\lim}\limits_{{t{\searrow}0}}\left(e^{\ln{y}}\right){=}\exp{\left(\mathop{\lim}\limits_{{t{\searrow}0}}\left(\ln{y}\right)\right)}$，所以不妨先求$\mathop{\lim}\limits_{{t{\searrow}0}}\left(\ln{y}\right){=}\mathop{\lim}\limits_{{t{\searrow}0}}\left(\cfrac{t}{t{-}α}\ln{t}\right){=}\mathop{\lim}\limits_{{t{\searrow}0}}\left(\cfrac{\ln{t}}{1{-}\frac{α}{t}}\right)$
> >    - 该极限为$\cfrac{{-}{∞}}{{-}{∞}}$型，故洛必达法得$\mathop{\lim}\limits_{{t{\searrow}0}}\left(\cfrac{\ln{t}}{1{-}\frac{α}{t}}\right){=}\mathop{\lim}\limits_{{t{\searrow}0}}\left(\cfrac{\frac{1}{t}}{\frac{α}{t^2}}\right){=}\mathop{\lim}\limits_{{t{\searrow}0}}\left(\cfrac{t}{α}\right){=}0$，由此$\mathop{\lim}\limits_{{t{\searrow}0}}\left(\ln{y}\right){=}0$ 
> >    - 所以$\mathop{\lim}\limits_{{t{\searrow}0}}y{=}e^0{=}1$，即原极限为$\mathop{\lim}\limits_{{τ{\nearrow}α}}{\left((α{-}τ)^{1{-}\frac{α}{τ}}\right)}{=}1$
> >
> > :three:求$γ{=}γ(s_{\max},τ){=}\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α{-}τ}\right){\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}^{\frac{τ}{α}}$的偏导
> >
> > 1. 令$γ{=}AB^{\frac{τ}{α}}$，其中$A{=}\left(\cfrac{α\left({1{-}{s_{\max}}}\right)}{α{-}τ}\right)$以及$B{=}{\left(\cfrac{{s_{\max}}\left({α{-}τ}\right)}{τ\left({1{-}{s_{\max}}}\right)}\right)}^{\frac{τ}{α}}$ 
> > 2. 求偏导$\cfrac{\deltaγ}{\delta{s}_{\max}}{=}\cfrac{\left({τ{-}α{s}_{\max}}\right){\left(\cfrac{{s}_{\max}(α{-}τ)}{τ(1{-}{s}_{\max})}\right)}^{\frac{τ}{α}}}{{s}_{\max}\left({α{-}τ}\right)}$
> >    - 首先取自然对数$\lnγ{=}\ln{A}{+}\cfrac{τ}{α}\ln{B}$，让$\ln{γ}$对$s_{\max}$求偏导得$\cfrac{\partial{}\ln{}γ}{\partial{}s_{\max}}{=}\cfrac{\partial{}\ln{}A}{\partial{}s_{\max}}{+}\cfrac{τ}{α}\cfrac{\partial{}\ln{}B}{\partial{}s_{\max}}$ 
> >      - 其中有$\cfrac{\partial{}\ln{}A}{\partial{}s_{\max}}{=}\cfrac{\partial{}{\left(\lnα{+}\ln\left(1{-}s_{\max}\right){-}\ln(α{-}τ)\right)}}{\partial{}s_{\max}}{=}\cfrac{1}{s_{\max}{-}1}$ 
> >      - 其中有$\cfrac{\partial{}\ln{}B}{\partial{}s_{\max}}{=}\cfrac{\partial{}{\left(\ln{s_{\max}}{+}\ln\left(α{-}τ\right){-}\ln{τ}{-}\ln(1{-}s_{\max})\right)}}{\partial{}s_{\max}}{=}\cfrac{1}{s_{\max}}{+}\cfrac{1}{1{-}s_{\max}}$ 
> >      - 合并得$\cfrac{\partial{}\ln{}γ}{\partial{}s_{\max}}{=}\cfrac{τ}{α}\cfrac{1}{s_{\max }\left(1{-}s_{\max }\right)}\text{{-}}\cfrac{1}{1{-}s_{\max}}{=}\cfrac{τ{-}α s_{\max }}{α s_{\max }\left(1{-}s_{\max }\right)}$ 
> >    - 应用链式法则得$\cfrac{\partial{γ}}{\partial{s_{\max}}}{=}\cfrac{\partial{γ}}{\partial\ln{γ}}\cfrac{\partial\lnγ}{\partial{s_{\max}}}{=}γ{}\cfrac{τ{-}α s_{\max }}{α s_{\max }\left(1{-}s_{\max }\right)}$，带入$γ{=}AB^{\frac{τ}{α}}$及$AB$的值即得到结果
> > 3. 求偏导$\cfrac{\deltaγ}{\deltaτ}{=}\cfrac{\left({1{-}{s}_{\max}}\right){\left(\cfrac{{s}_{\max}(α{-}τ)}{τ(1{-}{s}_{\max})}\right)}^{\frac{τ}{α}}\ln\left(\cfrac{s_{\max}(α{-}τ)}{τ(1{-}{s}_{\max})}\right)}{α{-}τ}$<mark>(原文偏导有误)</mark> 
> >    - 同样$\lnγ{=}\ln{A}{+}\cfrac{τ}{α}\ln{B}$，让$\ln{γ}$对$τ$求偏导得$\cfrac{\partial{}\ln{}γ}{\partial{}τ}{=}\cfrac{\partial{}\ln{}A}{\partial{}τ}{+}\cfrac{\partial{}\left(\cfrac{τ}{α}\ln{}B\right)}{\partial{}τ}{=}\cfrac{\partial{}\ln{}A}{\partial{}τ}{+}\cfrac{1}{α}\ln{}B{+}\cfrac{τ}{α}\cfrac{\partial\ln{}B}{\partialτ}$ 
> >      - 其中有$\cfrac{\partial{}\ln{}A}{\partial{}τ}{=}\cfrac{\partial{}{\left(\lnα{+}\ln\left(1{-}s_{\max}\right){-}\ln(α{-}τ)\right)}}{\partial{}τ}{=}\cfrac{1}{α{-}τ}$  
> >      - 其中有$\cfrac{\partial{}\ln{}B}{\partial{}τ}{=}\cfrac{\partial{}{\left(\ln{s_{\max}}{+}\ln\left(α{-}τ\right){-}\ln{τ}{-}\ln(1{-}s_{\max})\right)}}{\partial{}τ}{=}\cfrac{1}{τ{-}α}{{-}}\cfrac{1}{τ}$  
> >      - 代回得$\cfrac{\partial{}\ln{}γ}{\partial{}τ}{=}\cfrac{1}{α{-}τ}{+}\cfrac{1}{α}\ln{}B{+}\cfrac{τ}{α}\left(\cfrac{1}{τ{-}α}{{-}}\cfrac{1}{τ}\right){=}\cfrac{1}{α}\ln{}B$ 
> >    - 应用链式法则得$\cfrac{\partial{γ}}{\partial{τ}}{=}\cfrac{\partial{γ}}{\partial\ln{γ}}\cfrac{\partial\lnγ}{\partial{τ}}{=}\cfrac{γ{}}{α}\ln{}B$，代入$γ{=}AB^{\frac{τ}{α}}$及$AB$的值即得到结果
> >
> > :four:对$γ$单调性的分析
> >
> > 1. 考虑到对阈值的定义，直接就有$τ$的范围$τ\text{∈}(α{s_{\max}},α)$，以及对于$s_{\max}$有$s_{\max}\text{∈}\left(0,\cfrac{τ}{α}\right)$ 
> > 2. 偏导$\cfrac{\deltaγ}{\delta{s}_{\max}}{=}\cfrac{\left({τ{-}α{s}_{\max}}\right){\left(\cfrac{{s}_{\max}(α{-}τ)}{τ(1{-}{s}_{\max})}\right)}^{\frac{τ}{α}}}{{s}_{\max}\left({α{-}τ}\right)}$中
> >    - 有$τ{>}α{s_{\max}}$即$\left({τ{-}α{s}_{\max}}\right){>}0$，和$α{>}τ/1{>}s_{\max}$即${\left(\cfrac{{s}_{\max}(α{-}τ)}{τ(1{-}{s}_{\max})}\right)}^{\frac{τ}{α}}{>}0$，所以$\cfrac{\deltaγ}{\delta{s}_{\max}}{>}0$
> >    - 所以$γ{=}γ(s_{\max},τ)$随$s_{\max}$递增而递增
> > 3. 偏导$\cfrac{\deltaγ}{\deltaτ}{=}\cfrac{\left({1{-}{s}_{\max}}\right){\left(\cfrac{{s}_{\max}(α{-}τ)}{τ(1{-}{s}_{\max})}\right)}^{\frac{τ}{α}}\ln\left(\cfrac{s_{\max}(α{-}τ)}{τ(1{-}{s}_{\max})}\right)}{α{-}τ}$中<mark>(原文偏导错了但是不影响结论)</mark>
> >    - 同样有$1{>}s_{\max}$即$\left({1{-}{s}_{\max}}\right){>}0$，和$α{>}τ/1{>}s_{\max}$即${\left(\cfrac{{s}_{\max}(α{-}τ)}{τ(1{-}{s}_{\max})}\right)}^{\frac{τ}{α}}{>}0$
> >    - 但是$α{}s_{\max}{-}τ{}s_{\max}{-}τ{+}τ{}s_{\max}{=}α{}s_{\max}{-}τ{}$，鉴于$s_{\max}\text{∈}\left(0,\cfrac{τ}{α}\right)$所以$α{}s_{\max}{-}τ{<}0$，所以$\ln\left(\cfrac{s_{\max}(α{-}τ)}{τ(1{-}{s}_{\max})}\right){<}0$
> >    - 所以综上$\cfrac{\deltaγ}{\deltaτ}{<}0$，即$γ{=}γ(s_{\max},τ)$随$τ$递增而递减
> >
> > :five:总结与扩展
> >
> > 1. $γ$在$τ$上单调递减，在$τ$的两端有两个极限值$1$和$s_{\max}$，所以$γ{∈}\left(s_{\max},1\right)$故引理证毕
> > 2. 另外可以让$τ{=}\cfrac{α(k{+}s_{\max})}{k{+}1}$其中$k{>}0$，则$γ{=}(k{+}1){\left(\cfrac{{s}_{\max}}{k{+}{s}_{\max}}\right)}^{\frac{k{+}{s}_{\max}}{k{+}1}}$ 

# $\textbf{3. }$引理$\textbf{4.1.3}$证明

> :one:不断对$\Pr\left[σ{\left(\hat{\mathbf{s}}\right)}{≤}β{s_{\max}}{-}\Delta\right]$"松绑"
>
> 1. 由于$σ$是$(α,β)\text{-}$极大的，所以$β\max{\left(\widehat{\mathbf{s}}\right)}{≤}σ(\widehat{\mathbf{s}})$即$\Pr\left[σ{\left(\hat{\mathbf{s}}\right)}{≤}β{s_{\max}}{-}\Delta\right]{≤}\Pr\left[β\max{\left(\widehat{\mathbf{s}}\right)}{≤}β{s_{\max}}{-}\Delta\right]$
> 2. 由于对$\forall\hat{s}_j{∈}\hat{\mathbf{s}}$都有$\max{\left(\hat{\mathbf{s}}\right)}{≥}\hat{s}_j$，以及$\hat{s}_{\max}{∈}\hat{\mathbf{s}}$，故$\max{\left(\widehat{\mathbf{s}}\right)}{≥}\hat{s}_{\max}$即$\Pr\left[β\max{\left(\widehat{\mathbf{s}}\right)}{≤}β{s_{\max}}{-}\Delta\right]{≤}\Pr\left[β\hat{s}_{\max}{≤}β{s_{\max}}{-}\Delta\right]$
> 3. 应用绝对值，则有$\Pr\left[β\hat{s}_{\max}{≤}β{s_{\max}}{-}\Delta\right]{=}\Pr\left[β\left({s_{\max}}{-}\hat{s}_{\max}\right){≥}\Delta\right]{≤}\Pr\left[β\left|{s_{\max}}{-}\hat{s}_{\max}\right|{≥}\Delta\right]$  
>
> :two:应用$\text{Hoeffding}$不等式
>
> 1. 分桶随机变量：
>    - 假设$x_{ij}^*$就是那个与$q_r$一起构成最大真实相似度的向量，即$s_{\max}{=}\text{Sim}\left(q_r,x_{ij}^*\right)$
>    - 让$X_i$指示$q_r$和$x_{ij}^*$第$i$次分桶的伯努利随机变量，$X_i{∈}\{0,1\}$表示$q_r$和$x_{ij}^*$碰撞与否，即$\hat{s}_{\max}{=}\hat{\text{Sim}}\left(q_r,x_{ij}^*\right){=}\cfrac{1}{L}\displaystyle{}\sum_{i{=}1}^LX_i$
> 2. 带入$\text{Hoeffding}$不等式：
>    - 对$n$个独立的$X_i{∈}\{0,1\}{\subset}{(0,1)}$，有$\displaystyle\Pr\left[\left|\frac{1}{L}\sum_{i{=}1}^LX_i{-}\mathbb{E}\left[\frac{1}{L}\sum_{i{=}1}^LX_i\right]\right|{≥}t\right]{≤}2e^{{-}2Lt^2}$ 
>    - 其中$\hat{s}_{\max}{=}\hat{\text{Sim}}\left(q_r,x_{ij}^*\right){=}\cfrac{1}{L}\displaystyle{}\sum_{i{=}1}^LX_i$，而$\hat{s}_j/\hat{s}_{\max}$都是${s}_j/{s}_{\max}$的无偏估计，所以$\displaystyle\mathbb{E}\left[\frac{1}{L}\sum_{i{=}1}^LX_i\right]{=}\mathbb{E}\left[\hat{s}_{\max}\right]{=}{s}_{\max}$
>    - 所以$\Pr\left[\left|{s_{\max}}{-}\hat{s}_{\max}\right|{≥}t\right]{≤}2e^{{-}2tL^2}$，为与原式一致只需令$t{=}\cfrac{\Delta}{β}$，则$\Pr\left[β\left|{s_{\max}}{-}\hat{s}_{\max}\right|{≥}\Delta\right]{≤}2e^{{-}2L\Delta^2{/}β^2}$(证毕)

