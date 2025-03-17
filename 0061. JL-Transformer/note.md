# $\textbf{1. Johnson-Lindenstrauss}$引理

> ## $\textbf{1.1. }$引理内容
>
> > :one:原始$\text{JL}$引理：存在一种方法，能将高维数据急剧压缩，并且各点间距离信息几乎不变
> >
> > 1. 前提：对维度$d\text{∈}\{1,2,...\}$，小常数$\varepsilon\text{∈}\left(0,1\right)$，有限点集$X\text{⊆}{\mathbb{R}}^{d}$，投影后维度$m\text{=}\Theta\left( {\cfrac{\log\left|X\right|}{{\varepsilon}^{2}}}\right)$ 
> > 2. 结论：存在函数$\boldsymbol{\psi}\text{:}X\text{→}{\mathbb{R}}^{m}$，使得$\forall{u,v}\text{∈}X$有$\|\boldsymbol{\psi}(u)–\boldsymbol{\psi}(v)\|_{\ell_2}^2\text{∈}(1\text{±}\varepsilon)\|u–v\|_{\ell_2}^2$ 
> >
> > :two:分布型$\text{JL}$引理：存在分布$\Psi$，从中抽取的线性投影$\boldsymbol{\psi}$能将高维数据急剧压缩，且各点间距离大概率几乎不变
> >
> > 1. 前提$1$：对维度$d\text{∈}\{1,2,...\}$，小常数$\varepsilon,\delta\text{∈}\left(0,1\right)$，投影后维度$m\text{=}\Theta\left( {\cfrac{1}{{\varepsilon}^{2}}\log\left(\cfrac{1}{\delta}\right)}\right)$  
> > 2. 前提$2$：可从随机分布$\Psi$($\text{JL}$分布)中采样随机线性映射$\boldsymbol{\psi}\text{:}{\mathbb{R}}^{d}\text{→}{\mathbb{R}}^{m}$($\text{JL}$变换)，即$\boldsymbol{\psi}(x)\text{=}\textbf{A}x$其中$\textbf{A}\text{∈}{\mathbb{R}}^{m\text{×}d}$  
> > 3. 结论：分为对于单向量$u$的界，以及双向量$u,v$的联合界
> >    - 单向量：对$\forall{u}\text{∈}{\mathbb{R}}^{d}$，有$\text{Pr}\left[\|\boldsymbol{\psi}(u)\|_{\ell_2}^2\text{∈}(1\text{±}\varepsilon)\|u\|_{\ell_2}^2\right]\text{≥}1–\delta$  
> >    - 双向量：对$\forall{u,v}\text{∈}X\text{⊆}{\mathbb{R}}^{d}$，另设$\delta\text{=}\cfrac{1}{|X|^2}$则有$\text{Pr}\left[\|\boldsymbol{\psi}(u)–\boldsymbol{\psi}(v)\|_{\ell_2}^2\text{∈}(1\text{±}\varepsilon)\|u–v\|_{\ell_2}^2\right]\text{≥}\displaystyle{}1–\delta\binom{|X|}2\text{≥}1–\cfrac{1}{2|X|}$ 
>
> ## $\textbf{1.2. }$紧致性定理
>
> > :three:狭义$\text{JL}$紧性定理：存在某些高维点集$X$天然抗拒压缩，无论如何优化都只能压到$m\text{=}\Omega\left({\cfrac{\log\left(\varepsilon^2|X|\right)}{{\varepsilon}^{2}}}\right)$维
> >
> > 1. 前提：对$d\text{∈}\{1,2,...\}$，存在有限点集$X\text{⊆}{\mathbb{R}}^{d}$ ，以及小常数$\varepsilon\text{∈}\left(\cfrac{{\lg }^{0.5001}|X|}{\sqrt{\min\{d,|X|\}}},1\right)$ 
> > 2. 结论：对任何满足原始$\text{JL}$引理的函数$\boldsymbol{\psi}\text{:}X\text{→}{\mathbb{R}}^{m}$，必须满足维度下界$m\text{=}\Omega\left({\cfrac{\log\left(\varepsilon^2|X|\right)}{{\varepsilon}^{2}}}\right)$ 
> > 3. 备注：当$\varepsilon$过小即$\varepsilon\text{≤}\sqrt{\cfrac{\lg{|X|}}{\min\{d,|X|\}}}$时，造成”降维“后的$m\text{=}\Omega\left({\cfrac{1}{{\varepsilon}^{2}}\log\left(\varepsilon^2|X|\right)}\right)$反而大于降维前的$\min\{d,|X|\}$
> >
> > :four:广义$\text{JL}$紧性定理：即使允许更大的误差$\varepsilon$，依然有高维点集$X$抗拒压缩，最低只能压到$m\text{=}\Omega\left({\cfrac{\log\left(2\text{+}\varepsilon^2|X|\right)}{{\varepsilon}^{2}}}\right)$维
> >
> > 1. 前提：存在有限点集$X\text{⊆}{\mathbb{R}}^{d}$ ，给定常数$0\text{<}c\text{<}1$使得有$m\text{≤}cd\text{<}d\text{≤}n$，以及小常数$\varepsilon{≥}\cfrac{2}{\sqrt{|X|}}$ 
> > 2. 结论：对任何满足原始$\text{JL}$引理的函数$\boldsymbol{\psi}\text{:}X\text{→}{\mathbb{R}}^{m}$，必须满足维度下界$m\text{=}\Omega\left({\cfrac{\log\left(2\text{+}\varepsilon^2|X|\right)}{{\varepsilon}^{2}}}\right)$ 
>
> ## $\textbf{1.3. }$近似点积推论
>
> > :one:基于原始$\text{JL}$引理的表述：压缩前后不仅两点间距离变化不大，两点互相的内积也变化不大
> >
> > 1. 前提：对维度$d\text{∈}\{1,2,...\}$，小常数$\varepsilon\text{∈}\left(0,1\right)$，有限点集$X\text{⊆}{\mathbb{R}}^{d}$
> > 2. 结论：存在线性函数$\boldsymbol{\psi}\text{:}X\text{→}{\mathbb{R}}^{m}$，使得$\forall{u,v}\text{∈}X$有$\langle{}\boldsymbol{\psi}\left(u\right),\boldsymbol{\psi}\left(v\right)\rangle{}\text{∈}\langle{}u,v\rangle{}\text{±}\varepsilon\|{u}{\|}_{\ell_2}\|v{\|}_{\ell_2}$
> >
> > :two:基于分布型$\text{JL}$引理的表述：两点互相的内积也变化不大，以高概率成立
> >
> > 1. 前提$1$：对维度$d\text{∈}\{1,2,...\}$，小常数$\varepsilon,\delta\text{∈}\left(0,1\right)$，有限点集$X\text{⊆}{\mathbb{R}}^{d}$ 
> > 2. 前提$2$：可从$\text{JL}$分布$\Psi$中采样线性$\text{JL}$变换$\boldsymbol{\psi}\text{:}{\mathbb{R}}^{d}\text{→}{\mathbb{R}}^{m}$，即$\boldsymbol{\psi}(x)\text{=}\textbf{A}x$其中$\textbf{A}\text{∈}{\mathbb{R}}^{m\text{×}d}$  
> > 3. 结论：对$\forall{u,v}\text{∈}X$有$\text{Pr}\left[\langle{}\boldsymbol{\psi}\left(u\right),\boldsymbol{\psi}\left(v\right)\rangle{}\text{∈}\langle{}u,v\rangle{}\text{±}\varepsilon\|{u}{\|}_{\ell_2}\|v{\|}_{\ell_2}\right]\text{≥}1–\delta$ 

# $\textbf{2. Johnson-Lindenstrauss}$引理的应用

> ## $\textbf{2.1. JL}$引理与聚类
>
> > :one:$k\text{-Means}$方差双射引理：将簇内所有点到质心的距离平方的总和，转化为簇内所有点对距离平方之和(的一半)
> >
> > 1. 结论：令$k,d\text{∈}\{1,2,...\}$及$i\text{∈}\{1,2,...,k\}$，对簇$X_i\text{⊆}{\mathbb{R}}^{d}$有$\displaystyle{}\sum_{{i=1}}^{k}\sum_{{x\in{X}_{i}}}{\begin{Vmatrix}x–\cfrac{1}{\left|{X}_{i}\right|}\displaystyle{}\sum_{{y\in{X}_{i}}}y\end{Vmatrix}}_{\ell_2}^{2}\text{=}\sum_{{i=1}}^{k}\cfrac{1}{2\left|{X}_{i}\right|}\sum_{{x,y\in{X}_{i}}}\|{x}–y{\|}_{\ell_2}^{2}$  
> > 2. 含义：$\displaystyle{}\sum_{{x\in{X}_{i}}}{\begin{Vmatrix}x–\cfrac{1}{\left|{X}_{i}\right|}\displaystyle{}\sum_{{y\in{X}_{i}}}y\end{Vmatrix}}_{\ell_2}^{2}$为所有簇内点离质心距离的和(簇内方差)，另$\displaystyle{}\sum_{{x,y\in{X}_{i}}}\|{x}–y{\|}_{\ell_2}^{2}$为簇内所有点对距离的和
> >
> > :two:降维聚类成本传递命题：把高维空间的点$\text{JL}$降维后再聚类，结果与直接在高维空间中聚类相差不大
> >
> > 1. 前提$1$：给定聚类数$k\text{∈}\{1,2,...\}$，高维点集$X\text{⊆}{\mathbb{R}}^{d}$其中$d\text{∈}\{1,2,...\}$，以及误差$\varepsilon\text{≤}\cfrac{1}{2}$ 
> > 2. 前提$2$：投影$\boldsymbol{\psi}\text{:}X\text{→}{\mathbb{R}}^{m}$为$\text{JL}$变换，其中$m\text{=}\Theta\left( {\cfrac{\log\left|X\right|}{{\varepsilon}^{2}}}\right)$ 
> > 3. 前提$2$：令${X}_{1},\ldots,{X}_{k}\text{⊆}X$经过$\boldsymbol{\psi}$投影后生成${Y}_{1},\ldots,{Y}_{k}\text{⊆}Y$，以及所有点到各自质心距离的总和为所谓==$k\text{-Means}$成本==
> >    - ${\kappa}_{d}$为将$X$划分为${X}_{1},X_2,\ldots,{X}_{k}$的成本，${\kappa}_{d}^{*}$为将$X$划分为最优$k\text{-Means}$的成本
> >    - ${\kappa}_{m}$为将$Y$划分为${Y}_{1},Y_2,\ldots,{Y}_{k}$的成本，${\kappa}_{m}^{*}$为将$Y$划分为最优$k\text{-Means}$的成本
> > 4. 结论：当某个$\gamma\text{∈}\mathbb{R}$使得有${\kappa}_{m}\text{≤}\left({1\text{+}\gamma}\right){\kappa}_{m}^{*}$时，有${\kappa}_{d}\text{≤}\left({1\text{+}{4\varepsilon}}\right)\left({1\text{+}\gamma}\right){\kappa}_{d}^{*}$ 
>
> ## $\textbf{2.1. JL}$引理与流算法
>
> > :one:流算法及有关概念
> >
> > 1. 流的特点：
> >
> >    - 流查询：对持续到达并快速流失的数据进行查询
> >    - 流特点：只能对部分数据实现存储$\text{\&}$只能对数据遍历一次，因此只进行近似查询
> >
> > 2. 流的描述：
> >
> >    - 流的组成：即一系列更新操作，每个操作形如$(i_j,\Delta_j)$即在$t\text{=}j$时在$x$向量的第$i$维加上$\Delta$，因此$t$时刻有$x\text{=}\displaystyle{\sum_{j=1}^t{}\Delta_j\textbf{e}_{i_j}}$
> >    - 查询项目：$t$时刻关于$x$的统计量，例如$\|x\|_{\ell_2}$或频繁项(超过阈值的维度)等
> >
> > 3. 流的变体：
> >
> >    |    模型    | 特点                                | 实例             |
> >    | :--------: | ----------------------------------- | ---------------- |
> >    | 收银机模型 | 对于$x$向量的每一维，只允许递增操作 | 统计网页的点击量 |
> >    | 旋转门模型 | 对于$x$向量的每一维，允许增或减操作 | 统计银行账户余额 |
> >
> > :two:基于$\text{JL}$引理的流处理
> >
> > 1. 存储：不显式存储原始高维的$x$，而是存储$\text{JL}$线性降维投影后的$\boldsymbol{\psi}(x)$ 
> > 2. 更新：鉴于$\boldsymbol{\psi}(x)$是线性的，所以有$\boldsymbol{\psi}(x)\text{=}\displaystyle{\sum_{j=1}^t{}\Delta_j\boldsymbol{\psi}(\textbf{e}_{i_j})}$，即流$(i_j,\Delta_j)$到达后只需更新其投影$\Delta_j\boldsymbol{\psi}(\textbf{e}_{i_j})$ 
> > 3. 查询：基于分布型$\text{JL}$引理，在引入一定失败概率的情况下，直接输出投影后向量$\boldsymbol{\psi}(x)$的统计量



















