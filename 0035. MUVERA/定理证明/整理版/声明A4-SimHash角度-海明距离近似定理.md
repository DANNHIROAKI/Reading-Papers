# $\textbf{Muvera}$的前两步

**1️⃣**文本嵌入：对查询文本和段落文本分别应用嵌入器(如$\text{ColBERTv2}$)，得到各自的多向量嵌入
1. 查询嵌入$Q$：$\{q_1,q_2,...,q_m\}$，其中$q_i\text{⊆}\mathbb{R}^{d}$即为固定$d$维
2. 段落嵌入$P$：$\{p_1,p_2,...,p_n\}$，其中$p_i\text{⊆}\mathbb{R}^{d}$即为固定$d$维

**2️⃣**向量分桶：用$\text{SimHash}$将原有空间分为$2^{k_{\text{sim}}}$个桶，每个桶用长为$k_{\text{sim}}$的定长二进制向量编码

1. 法向抽取：从高斯分布中抽取$k_{\text{sim}}\text{≥}1$个向量$g_{1},\ldots,g_{k_{\text{sim}}}\text{∈}\mathbb{R}^{d}$，作为$k_{\text{sim}}$个超平面的法向量
2. 空间划分：$\varphi(x)\text{=}\left(\mathbf{1}\left(\left\langle{}g_{1},x\right\rangle{}\text{>}0\right),\ldots,\mathbf{1}\left(\left\langle{}g_{k_{\text{sim}}},x\right\rangle{}\text{>}0\right)\right)$
   - $\mathbf{1}\left(\left\langle{}g_{i},x\right\rangle{}\text{>}0\right)$：当$\langle{}g_{i},x\rangle{}\text{>}0$成立(即$x$投影在超平面$g_i$的正侧)时，将该位设为$1$
3. 向量分桶：让所有的$m\text{+}n$个嵌入通过$\varphi(\cdot)$得到长$k_{\text{sim}}$的二进制编码，相同编码者(即桶编码)放入同一桶

# 声明的内容

👉记号：记$\forall{}x,y\text{∈}\mathbb{R}^b$夹角为$\theta(x,y)\text{∈}[0,\pi]$，二进制编码$x,y$的海明距离为$\|x–y\|_{0}$

👉前提$1$：对$\forall{}q_i\text{∈}Q$以及$\forall{}p_j\text{∈}P$，给定$\forall{}\varepsilon{}\text{≤}\cfrac{1}{2}$(与定理$\text{2.1}$统一)与$\forall{}\delta{≤}\varepsilon$

👉前提$2$：令$k_{\text{sim}}\text{=}O\left(\cfrac{1}{\varepsilon}\log{\left(\cfrac{m}{\delta}\right)}\right)$，其中$m\text{=}|Q|\text{+}|P|$

👉结论：$\left|\|\varphi(q_i)–\varphi(p_j)\|_0\text{ – }\cfrac{k_{\mathrm{sim}}\theta(q_i,p_j)}{\pi}\right|\text{≤}\sqrt{\varepsilon}k_{\mathrm{sim}}$以$\text{Pr}\text{≥}1–\left(\cfrac{\varepsilon{}\delta{}}{m^2}\right)$的概率成立

# 声明的证明

➡️由以下分析可得$\text{Pr}[\mathbf{1}(\langle{}g_k,q_i\rangle\text{>}0)\text{≠}\mathbf{1}(\langle{}g_k, p_j\rangle\text{>}0)]\text{=}\cfrac{\theta(q_i,p_j)}{\pi}$  

1. 通过$\text{Gram-Schmidt}$过程，对$q_i,p_j$进行$\mathbf{R}$旋转

   - 令$d$维空间基为$B\text{=}\{\vec{e}_1,\vec{e}_2,...,\vec{e}_d\}$满足$\forall{\vec{e}_i,\vec{e}_j}\text{∈}B$有$\vec{e}_i\text{⊥}\vec{e}_j$，让旋转矩阵$\mathbf{R}$将$q_i$旋转到$\vec{e}_1$方向即$\mathbf{R}q_i\text{=}(1,0,...,0)^d$

   - 现在只考虑$\vec{e}_1$以及$\vec{e}_2$组成的二维平面，则$\mathbf{R}p_j\text{=}\vec{e}_1\cos{(\mathbf{R}q_i,\mathbf{R}p_j)}\text{+}\vec{e}_2\sin{(\mathbf{R}q_i,\mathbf{R}p_j)}$ 

     <img src="https://i-blog.csdnimg.cn/direct/add07e98bef94bf8b77221759004d44d.png" alt="image-20250305154857408" width=400 />   

   - 考虑到$\theta{(\mathbf{R}q_i,\mathbf{R}p_j)}\text{=}\theta{(q_i,p_j)}$，则有$\mathbf{R}p_j\text{=}e_1\cos{\theta(q_i,p_j)}\text{+}e_2\sin\theta{(q_i,p_j)}\text{=}(\cos{\theta(q_i,p_j)},\sin{\theta(q_i,p_j)},0,...,0)^d$ 

2. 考虑高斯分布的旋转不变性，对$g_k$进行$\mathbf{R}$旋转

   - 旋转不变性，即对于高斯向量$g_k\text{∈}\mathbb{R}^{d\text{×}1}\text{∼}\mathcal{N}(\textbf{0},\boldsymbol{\text{I}_d})$以及旋转矩阵$\mathbf{R}\text{∈}\mathbb{R}^{d\text{×}d}$，则有$\mathbf{R}g_k\text{∼}\mathcal{N}(\textbf{0},\boldsymbol{\text{I}_d})$ 
   - 由此可得$\mathbf{R}g_k\text{=}(g_{k_1},g_{k_2},g_{k_3},...,g_{k_d})\text{∼}\mathcal{N}(\textbf{0},\boldsymbol{I_d})$，其中有$g_{k_1}\text{∼}{\mathcal{N}(0,1)}$和$g_{k_2}\text{∼}{\mathcal{N}(0,1)}$ 
   - 考虑到$\mathbf{R}$在对$x,y$进行旋转时仅前两位在起作用，故不妨令$\mathbf{R}g_k\text{=}(g_{k_1},g_{k_2},0,...,0)^d$，或者写作$\mathbf{R}g\text{=}(\cos\phi,\sin\phi,0,...,0)^d$  

3. 求解相应的内积

   - $\langle{}g_k,q_i\rangle\text{=}\langle{}\mathbf{R}g_k,\mathbf{R}q_i\rangle\text{=}(g_{k_1},g_{k_2},...,g_{k_d})(1,0,...,0)^{dT}\text{=}\cos\phi$ 
   - $\langle{}g_k,p_j\rangle\text{=}\langle{}\mathbf{R}g_k,\mathbf{R}p_j\rangle\text{=}(\cos\phi,\sin\phi,0,...,0)^d(\cos{\theta(q_i,p_j)},\sin{\theta(q_i,p_j)},0,...,0)^{dT}\text{=}\cos(\phi–\theta(q_i,p_j))$   

4. 整理$\text{Pr}[\mathbf{1}(\langle{}g_k,q_i\rangle\text{>}0)\text{≠}\mathbf{1}(\langle{}g_{k},p_j\rangle\text{>}0)]$，变为$\text{Pr}\left[\mathbf{1}\left(\cos\phi\text{>}0\right)\text{≠}\mathbf{1}\left(\cos(\phi–\theta(q_i,p_j))\text{>}0\right)\right]$ 

   - 情形$1$：$\cos\phi\text{>}0$且$\cos(\phi–\theta(q_i,p_j))\text{≤}0$，则$\phi\text{∈}\left(–\cfrac{\pi}{2},\cfrac{\pi}{2}\right)$及$\phi\text{∈}\left(\theta(q_i,p_j)\text{+}\cfrac{\pi}{2},\theta(q_i,p_j)\text{+}\cfrac{3\pi}{2}\right)$，二者交集长$|\phi|\text{=}\theta(q_i,p_j)$
   - 情形$2$：$\cos\phi\text{<}0$且$\cos(\phi–\theta(q_i,p_j))\text{≥}0$，则$\phi\text{∈}\left(\cfrac{\pi}{2},\cfrac{3\pi}{2}\right)$及$\phi\text{∈}\left(\theta(q_i,p_j)\text{–}\cfrac{\pi}{2},\theta(q_i,p_j)\text{+}\cfrac{\pi}{2}\right)$，二者交集长$|\phi|\text{=}\theta(q_i,p_j)$
   - 而由于$\phi$总范围为$|\phi|_{\max}\text{=}2\pi$，所以$\text{Pr}[\mathbf{1}(\langle{}g_k,q_i\rangle\text{>}0)\text{≠}\mathbf{1}(\langle{}g_{k},p_j\rangle\text{>}0)]\text{=}\cfrac{2\theta(q_i,p_j)}{2\pi}\text{=}\cfrac{\theta(q_i,p_j)}{\pi}$ 

➡️构造变量$Z_k$

1. 对每个$k\text{∈}\{1,2,...,k_{\text{sim}}\}$及对应的高斯向量$g_k$，定义$Z_k\text{=}\mathbf{1}\left(\left\langle g_k, q_i\right\rangle\text{>}0\right)\text{⊕}\mathbf{1}\left(\left\langle g_k, p_j\right\rangle\text{>}0\right)$，即二者不相等时$Z_k\text{=}1$

   <img src="https://i-blog.csdnimg.cn/direct/039efc73090e4deba46bbd5eec9511bf.png" alt="erzthyjgdkvghvvzvgret" width=420 />     

2. 由海明距离的定义(两二进制编码上下对齐后有多少对应位不同)，则有$\|\varphi(q_i)–\varphi(p_j)\|_0\text{=}\displaystyle\sum_{k\text{=}1}^{k_{\text{sim}}}Z_k$ 

3. 考虑到$\text{Pr}[\mathbf{1}(\langle{}g_k,q_i\rangle\text{>}0)\text{≠}\mathbf{1}(\langle{}g_k, p_j\rangle\text{>}0)]\text{=}\cfrac{\theta(q_i,p_j)}{\pi}$于是$\mathbb{E}\left[Z_k\right]\text{=}1\left(\cfrac{\theta(q_i,p_j)}{\pi}\right)\text{+}0\left(1–\cfrac{\theta(q_i,p_j)}{\pi}\right)\text{=}\cfrac{\theta(q_i,p_j)}{\pi}$

4. 由此$\left|\|\varphi(q_i)–\varphi(p_j)\|_0\text{ – }\cfrac{k_{\mathrm{sim}}\theta(q_i,p_j)}{\pi}\right|\text{=}\left|\displaystyle\sum_{k\text{=}1}^{k_{\text{sim}}}Z_k\text{ – }\mathbb{E}\left[\sum_{k\text{=}1}^{k_{\text{sim}}}Z_k\right]\right|$，故需证$\text{Pr}\left[\left|\displaystyle\sum_{k\text{=}1}^{k_{\text{sim}}}Z_k\text{ – }\mathbb{E}\left[\sum_{k\text{=}1}^{k_{\text{sim}}}Z_k\right]\right|\text{≥}\sqrt{\varepsilon}k_{\text{sim}}\right]\text{≤}\left(\cfrac{\varepsilon\delta}{m^2}\right)$ 

➡️进一步转换所要证明的结论

1. 根据$\text{Hoeffding}$不等式，即对于独立有界变量$Z_k\text{∈}[0,1]$有$\text{Pr}\left[\left|\displaystyle\sum_{k\text{=}1}^{n}Z_k\text{ – }\mathbb{E}\left[\sum_{k\text{=}1}^{n}Z_k\right]\right|\text{≥}t\right]\text{≤}2e^{–\frac{2t^2}{n}}$ 
2. 令$t\text{=}\sqrt{\varepsilon}k_{\text{sim}}$与$n\text{=}k_{\text{sim}}$，则$\text{Pr}\left[\left|\displaystyle\sum_{k\text{=}1}^{k_{\text{sim}}}Z_k\text{ – }\mathbb{E}\left[\sum_{k\text{=}1}^{k_{\text{sim}}}Z_k\right]\right|\text{≥}\sqrt{\varepsilon}k_{\text{sim}}\right]\text{≤}2e^{–2\varepsilon{}k_{\text{sim}}}$，于是只需证$2e^{–2\varepsilon{}k_{\text{sim}}}\text{≤}\cfrac{\varepsilon\delta}{m^2}$即$k_{\mathrm{sim}}\text{≥}\cfrac{1}{2\varepsilon}\ln\left(\cfrac{2m^2}{\varepsilon \delta}\right)$ 
3. 集合前提故只需验证$k_{\text{sim}}\text{=}O\left(\cfrac{1}{\varepsilon}\log{\left(\cfrac{m}{\delta}\right)}\right)\text{≥}\cfrac{1}{2\varepsilon}\ln\left(\cfrac{2m^2}{\varepsilon \delta}\right)$ 
   - 令$\cfrac{C}{\varepsilon}\ln\left(\cfrac{m}{\delta}\right)\text{≥}k_{\text{sim}}\text{≥}\cfrac{1}{2\varepsilon}\ln\left(\cfrac{2m^2}{\varepsilon \delta}\right)$，即需要验证$\cfrac{C}{\varepsilon}\ln\left(\cfrac{m}{\delta}\right)\text{≥}\cfrac{1}{2\varepsilon}\ln\left(\cfrac{2m^2}{\varepsilon \delta}\right)$，即上界足以覆盖下界
   - 稍作变形即可得$(2C–2)\ln{m}\text{+}(2C–1)\ln{\left(\cfrac{1}{\delta}\right)}\text{≥}\ln2\text{+}\ln{\left(\cfrac{1}{\varepsilon}\right)}$
   - 不妨令$\begin{cases}2C–2\text{≥}1\\\\2C–1\text{≥}1\end{cases}$以及$\begin{cases}m\text{≥}2\\\\\cfrac{1}{\delta}\text{≥}\cfrac{1}{\varepsilon}\end{cases}$则上式成立，解得$C\text{≥}\cfrac{3}{2}$以及$m\text{≥}2,\delta{≤}\varepsilon$，其中$m\text{=}|Q|\text{+}|P|\text{≥}2$隐性成立
   - 故只需让$\delta{≤}\varepsilon$，结论$O\left(\cfrac{1}{\varepsilon}\log{\left(\cfrac{m}{\delta}\right)}\right)\text{≥}\cfrac{1}{2\varepsilon}\ln\left(\cfrac{2m^2}{\varepsilon \delta}\right)$就成立，故证毕







