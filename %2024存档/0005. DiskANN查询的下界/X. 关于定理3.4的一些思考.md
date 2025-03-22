# X. 关于定理$\textbf{3.4}$的一些思考 

> ## X.1. 预备知识回顾
>
> > :one:$\alpha{\text{-}}$可捷达性
> > 1. 点$mn$的$\alpha{\text{-}}$可捷达性：$mn$要么直连，要么$mm^{\prime}$直连但$D(m', n) \leq \cfrac{1}{\alpha}*D(m, n)$ 
> >    <img src="https://i-blog.csdnimg.cn/direct/884f62aa4df34345847fd9b95d3a8779.png" alt="image-20240810214535083" width=300 /> 
> > 2. 图的$\alpha{\text{-}}$可捷达性：要求图中任意两点都是$\alpha{\text{-}}$可捷达的
> >
> > :two:$\text{DiskANN}$的$\text{GreedySearch}(s, q, L)$ 
> >
> > 1. 参数含义
> >    - 符号：$s$(搜索起点)，$q$(待查询点)，$L$(队列$A$最大长度)
> >      $\text{Ps. }$ 当$L=1$时意味着贪心搜索不会回溯，即每走到下一个结点后，离最邻近必定更近
> >    - 辅助数据结构：$A$(当前队列)，$U$(已访问点的集合)
> > 2. 算法过程
> >    <img src="https://i-blog.csdnimg.cn/direct/54acfe085ddc4b7aaedb5d407134b959.png" width=350 /> 
> >
> > :three:定理内容
> >
> > 1. 前提
> >    - 图：$G=(V, E)$时经过慢预处理构建的$\alpha\text{-}$捷径可达图
> >    - 搜索：令$L=1$，从任意$s \in V$开始执行$\text{GreedySearch}(s, q, L)$ 
> > 2. 结论：算法在$O\left(\log _\alpha \cfrac{\Delta}{(\alpha-1) \epsilon}\right)$步内找到$\left(\cfrac{\alpha+1}{\alpha-1}+\epsilon\right)$的近似最近邻
>
> ## X.2. 证明$\textbf{Pipeline}$ 
>
> > :one:有关符号
> > 1. 结点
> >    | 符号  | 含义                                                    |
> >    | :---: | :------------------------------------------------------ |
> >    |  $q$  | 给定的待查询点                                          |
> >    |  $a$  | $q$的最邻近                                             |
> >    | $v_i$ | 图$G=(V,E)$中第$i$个被扫描到的点，==$i$就是算法的步数== |
> > 2. 距离：$d_i=D(v_i,q)$ 
> > 3. ==近似比：$c_i=\cfrac{d_i}{D(a, q)}$== 
> > 4. 纵横比：$\Delta=\cfrac{D_{\max }}{D_{\min }}$，$D_{\text{max}}$为点集$V$中相聚最远两点的距离
> >
> > :two:初步分析：关于$d_i$的递归式
> >
> > <img src="https://i-blog.csdnimg.cn/direct/995c7f57350143769c808beba754c22a.png" alt="image-20240817144002003" width=400 />  
> >
> > 1. 递归式通项的推导  
> >    - $\textcolor{green}{D(a,v^{\prime})\leq{}\cfrac{d_i+D(a, q)}{\alpha}}\Leftarrow{}
> >      \begin{cases}
> >      D(a,v_i)\leq{}d_i+D(a,q)(三角定理)\\\\
> >      D(a,v^{\prime})\leq{}\cfrac{1}{\alpha}*D(a, v_i)(\alpha{}\text{-}可捷达性)
> >      \end{cases}$ 
> >      - 直连：$a\xleftrightarrow{}v_i$直接连接
> >      - $\alpha\text{-}$捷达性：$a\xleftrightarrow{}v_i$不直接连接，$v'\xleftrightarrow{}v_i$直接连接但$D(a,v^{\prime})\leq{}\cfrac{1}{\alpha}*D(a, v_i)$ 
> >    - $\textcolor{orange}{D(q,v^{\prime})\leq{}\cfrac{d_i+D(a, q)}{\alpha}+D(a,q)}\Leftarrow{}
> >      \begin{cases}
> >      \textcolor{green}{D(a,v^{\prime})\leq{}\cfrac{d_i+D(a, q)}{\alpha}}\\\\
> >      D(q,v^{\prime})\leq{}D(a,v^{\prime})+D(a,q)(三角定理)
> >      \end{cases}$  
> >    - $\textcolor{red}{d_{i+1}\leq{}\cfrac{d_i+D(a, q)}{\alpha}+D(a,q)
> >      }\Leftarrow{}
> >      \begin{cases}
> >      \textcolor{orange}{D(q,v^{\prime})\leq{}\cfrac{d_i+D(a, q)}{\alpha}+D(a,q)}\\\\
> >      d_{i+1}\leq{}D(q,v^{\prime})(必定从v_i邻居(含v^{\prime})选一离q更近的v_{i+1})
> >      \end{cases}$ 
> > 2. 归纳法解递归
> >     - 初始条件：$d_0 = D(s, q)$
> >     - 归纳假设：$d_i \leq \cfrac{D(s, q)}{\alpha^i} + \cfrac{\alpha+1} {\alpha-1} D(a, q)$ 
> >     - 归纳推到：$d_{i+1} \leq \cfrac{D(s, q)}{\alpha^{i+1}} + \cfrac{\alpha+1}{\alpha-1} D(a, q)\Leftarrow{}\begin{cases}存在d_{i+1} \leq \cfrac{d_i + D(a, q)}{\alpha} + D(a, q)\\\\带入d_i \leq \cfrac{D(s, q)}{\alpha^i} + \cfrac{\alpha+1} {\alpha-1} D(a, q)\end{cases}$ 
> >3. 递归解：==$d_i \leq \cfrac{D(s, q)}{\alpha^i} + \cfrac{\alpha+1}{\alpha-1} D(a, q)$==   
> > 
> >:three:三种情况
> > 
> ><img src="https://i-blog.csdnimg.cn/direct/ca3aa839994a45eca441e3fee09a0f76.png" alt="image-20240817145143282" width=400 />  
> > 
> > |  情况  |        $D(s, q)$         |                           $D(a,q)$                           |
> > | :----: | :----------------------: | :----------------------------------------------------------: |
> > | 情况一 | $(2 D_{\max },\infin{})$ |                   $[-\infin{},+\infin{}]$                    |
> > | 情况二 |    $(0,2 D_{\max }]$     | $\left[\cfrac{\alpha-1}{4(\alpha+1)} D_{\min },+\infin{}\right)$ |
> >| 情况三 |    $(0,2 D_{\max }]$     |   $\left(0,\cfrac{\alpha-1}{4(\alpha+1)} D_{\min }\right)$   |
> > 
> > 1. 情况一：当起始点离最邻近很远时，**同时推导出邻近度$+$步数**    
> >       - $\textcolor{green}{D(a,q)>D(s,q)-D_{\text{max}}} \Leftarrow{} \begin{cases} D(a,q)>D(s,q)-D(a,s)(三角定理)\\\\ D(a,s)<D_{\text{max}}(D_{\text{max}}的定义) \end{cases}$
> >       - $\textcolor{purple}{c_i\leqslant \cfrac{D(s, q)}{\alpha^i D(a, q)}+\cfrac{\alpha+1}{\alpha-1} }\Leftarrow{} \begin{cases} d_i \leq \cfrac{D(s, q)}{\alpha^i} + \cfrac{\alpha+1}{\alpha-1} D(a, q)(递归解)\\\\ c_i=\cfrac{d_i}{D(a, q)}(近似比定义) \end{cases}$
> >       - $\textcolor{orange}{D(a,q)>\cfrac{D(s,q)}{2}} \Leftarrow{} {D(a,q)>D(s,q)-\cfrac{D(s,q)}{2}} \Leftarrow{} \begin{cases} \textcolor{green}{D(a,q)>D(s,q)-D_{\text{max}}}\\\\ D(s,q)>2D_{\text{max}}(前提) \end{cases}$
> >       -  $\textcolor{red}{\begin{cases} c_i \leqslant \cfrac{\alpha+1}{\alpha-1}+\epsilon\\\\ i\geq{}\log_{\alpha}\left(\cfrac{2}{\epsilon}\right) \end{cases}} \xLeftarrow[]{\large\frac{2}{\alpha{}^{i}}\leq{}\epsilon\\} {c_i\leqslant \cfrac{2}{\alpha^i }+\cfrac{\alpha+1}{\alpha-1}} \Leftarrow{} \begin{cases} \textcolor{purple}{c_i\leqslant \cfrac{D(s, q)}{\alpha^i D(a, q)}+\cfrac{\alpha+1}{\alpha-1} }\\\\ \textcolor{orange}{{D(s,q)}<{2}*D(a,q)} \end{cases}$ 
> > 2. 情况二：起始点离最邻近适中，但最邻近离查询点又很远，**假设邻近度→推导出步数** 
> >       - 假设算法达到了$\textcolor{green}{c_i \leq \cfrac{\alpha+1}{\alpha-1} + \epsilon}$ 邻近度 
> >       - $\textcolor{purple}{\cfrac{D(s, q)}{\alpha^i}<\epsilon D(a, q)}
> >         \Leftarrow{}
> >         \begin{cases}
> >         c_i\leqslant \cfrac{D(s, q)}{\alpha^i D(a, q)}+\cfrac{\alpha+1}{\alpha-1}(情况1中已证明)\\\\
> >         \textcolor{green}{c_i \leq \cfrac{\alpha+1}{\alpha-1} + \epsilon }
> >         \end{cases}$ 
> >      - $\textcolor{orange}{\cfrac{2 D_{\max}}{\alpha^i} \leq \epsilon \times \cfrac{\alpha-1}{4(\alpha+1)} D_{\min}}
> >         \xLeftarrow[用D(a,q)下界将其替代]{用D(s,q)上限将其替代}
> >         \begin{cases}
> >         D(s, q) \leq 2 D_{\max}\\\\
> >         D(a, q) \geq \cfrac{\alpha-1}{4(\alpha+1)} D_{\min}\\\\
> >        \textcolor{purple}{\cfrac{D(s, q)}{\alpha^i}<\epsilon D(a, q)}
> >         \end{cases}$  
> >       - $O\left(\log_\alpha \cfrac{\Delta}{(\alpha-1) \epsilon}\right)\xLeftarrow{渐进上界}
> >          \textcolor{red}{i \geq \log_\alpha \cfrac{8(\alpha+1) \Delta}{(\alpha-1) \epsilon}}
> >         \xLeftarrow{}
> >          \begin{cases}
> >          \textcolor{orange}{\cfrac{2 D_{\max}}{\alpha^i} \leq \epsilon \times \cfrac{\alpha-1}{4(\alpha+1)} D_{\min}}\\\\
> >          \Delta=\cfrac{D_{\max }}{D_{\min }} (定义)
> >          \end{cases}$   
> > 3. 情况三：起始点离最邻近适中，最邻近离查询点很近，**推导出找到==确切最邻近==的步数上界 **  
> >    - 证明：$d_i\geq{}\cfrac{1}{2}{D_{\text{min}}}$ 
> >      - $\textcolor{green}{D(a,q)\leq{}\cfrac{1}{2}D(v_i,a)}
> >        \Leftarrow{}
> >        \begin{cases}{D(a,q)<\cfrac{1}{4}D_{\text{min}}}\xLeftarrow{\frac{\alpha{}-1}{4(\alpha{}+1)}{<0.25}}
> >        D(a,q)<{}\cfrac{(\alpha-1)D_{\min }}{4(\alpha+1)} (条件)\\\\
> >        D_{\text{min}}<D(v_i,a)(D_{\text{min}}的定义)
> >        \end{cases}$  
> >      - $\textcolor{orange}{d_i\geq{}\cfrac{1}{2}D(v_i,a)}
> >        \Leftarrow{}
> >        \begin{cases}
> >        \textcolor{green}{\cfrac{1}{2}D(v_i,a)\geq{}D(a,q)}\\\\
> >        d_i\geq{}D(v_i,a)-D(a,q) (三角定理)
> >        \end{cases}$   
> >      - $\textcolor{red}{d_i\geq{}\cfrac{1}{2}{D_{\text{min}}}}
> >        \Leftarrow{}
> >        \begin{cases}
> >        \textcolor{orange}{d_i\geq{}\cfrac{1}{2}D(v_i,a)}\\\\
> >        D(v_i,a)>D_{\text{min}}(D_{\text{min}}的定义)
> >        \end{cases}$ 
> >     - 证明：$i\leq\log_{\alpha}8\Delta$ 
> >         - $\textcolor{green}{\cfrac{1}{2}{D_{\text{min}}}\leq{}\cfrac{D(s, q)}{\alpha^i} + \cfrac{\alpha+1}{\alpha-1} D(a, q)}
> >          \Leftarrow{}
> >           \begin{cases}
> >           d_i \leq \cfrac{D(s, q)}{\alpha^i} + \cfrac{\alpha+1}{\alpha-1} D(a, q)(递归解)\\\\
> >           d_i\geq{}\cfrac{1}{2}{D_{\text{min}}}(上述已证明)
> >           \end{cases}$
> >         - $\textcolor{orange}{\alpha^i \leqslant \cfrac{8 D_{\max }}{D_{\min }}}\Leftarrow{}\cfrac{D_{\min }}{2} \leqslant \cfrac{2 D_{\max }}{\alpha^i}+\cfrac{D_{\min }}{4}
> >          \Leftarrow{}
> >           \begin{cases}
> >           \textcolor{green}{\cfrac{1}{2}{D_{\text{min}}}\leq{}\cfrac{D(s, q)}{\alpha^i} + \cfrac{\alpha+1}{\alpha-1} D(a, q)}\\\\
> >           D(a,q)<{}\cfrac{\alpha-1}{4(\alpha+1)} D_{\min }(条件)\\\\
> >           D(s,q)\leq{}2D_{\text{max}}(条件)
> >           \end{cases}$ 
> >         - $O\left(\log_\alpha \Delta\right)
> >          \xLeftarrow{渐进上界}\textcolor{red}{i\leq\log_{\alpha}8\Delta}
> >           \Leftarrow{}
> >           \begin{cases}
> >           \textcolor{orange}{\alpha^i \leqslant \cfrac{8 D_{\max }}{D_{\min }}}\\\\
> >           \Delta=\cfrac{D_{\max }}{D_{\min }} (定义)
> >           \end{cases}$ 
> > 
> 
> ## X.3. 对证明过程的思考
>
> > :one:证明的关键何在
>>
> > |     关键点      | 解释                                                         |
> > | :-------------: | ------------------------------------------------------------ |
> > |   三角不等式    | 是推导过程的基石，几乎出现在所有推导的过程                   |
> > | ==$d_i$递归式== | 推导过程中最为核心的一步，因为其解释了算法在每一步的收敛速度 |
> > |    三种情况     | 本质上是算法在不同距离下==有三种收敛方式==，*<span style="color:#FF3300;">不同收敛方式受(局部)倍增维度控制?</span>* |
> >
> > :two:三种情况的一些分析：~~本想尝试解答三种情况划分界限的内在因果，但最后也没有分析出什么🤣~~ 
> >
> > |  情况  |        $D(s, q)$         |                           $D(a,q)$                           |
> > | :----: | :----------------------: | :----------------------------------------------------------: |
> > | 情况一 | $(2 D_{\max },\infin{})$ |                   $[-\infin{},+\infin{}]$                    |
> > | 情况二 |    $(0,2 D_{\max }]$     | $\left[\cfrac{\alpha-1}{4(\alpha+1)} D_{\min },+\infin{}\right)$ |
> > | 情况三 |    $(0,2 D_{\max }]$     |   $\left(0,\cfrac{\alpha-1}{4(\alpha+1)} D_{\min }\right)$   |
> >
> > 1. $D(a,q)\leq{}3D_{\text{max}}
> >    \Leftarrow{}
> >    \begin{cases}
> >    D(a,q)\leq{}D(s,q)+D_{\text{max}}
> >    \Leftarrow{}
> >    \begin{cases}
> >    D(a,q)\leq{}D(s,q)+D(a,s) (三角)\\\\
> >    D(a,s)\leq{}D_{\text{max}}(D_{\text{max}}定义)
> >    \end{cases}\\\\
> >    D(s,q)\leq{}2 D_{\max }(情况2/3条件)
> >    \end{cases}$
> > 2. $D(a,q)\geq{}D_{\text{max}}
> >    \Leftarrow{}
> >    \begin{cases}
> >    D(a,q)\geq{}D(s,q)-D_{\text{max}}
> >    \Leftarrow{}
> >    \begin{cases}
> >    D(a,q)\geq{}D(s,q)-D(a,s) (三角)\\\\
> >    D(a,s)\leq{}D_{\text{max}}(D_{\text{max}}定义)
> >    \end{cases}\\\\
> >    D(s,q)\geq{}2 D_{\max }(情况1条件)
> >    \end{cases}$ 
