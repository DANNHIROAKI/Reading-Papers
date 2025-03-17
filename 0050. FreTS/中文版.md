### 基于频域的多层感知机在时间序列预测中更具学习效果

## 摘要

时间序列预测在金融、交通、能源和医疗等多个工业领域中扮演着关键角色。尽管现有文献提出了许多基于循环神经网络（RNN）、图神经网络（GNN）或Transformer的复杂架构，但另一种基于多层感知机（MLP）的方法因其结构简单、复杂度低且性能优越而受到关注。然而，大多数基于MLP的预测方法受限于逐点映射和信息瓶颈，这极大地阻碍了其预测性能。为了解决这一问题，我们探索了一种新颖的方向，即在频域中应用MLP进行时间序列预测。我们研究了频域MLP的学习模式，并发现了其两个固有的特性对预测有益：（i）全局视角：频谱使MLP能够全面观察信号，并更容易学习全局依赖关系；（ii）能量压缩：频域MLP集中于较小的关键频率分量，压缩信号能量。基于此，我们提出了FreTS，一种基于频域MLP的简单而有效的时间序列预测架构。FreTS主要包括两个阶段：（i）域转换，将时域信号转换为频域的复数形式；（ii）频率学习，通过重新设计的MLP学习频率分量的实部和虚部。上述阶段在序列间和序列内尺度上操作，进一步促进了通道间和时间依赖关系的学习。在13个真实世界基准数据集（包括7个短期预测和6个长期预测数据集）上的广泛实验表明，我们的方法在性能上始终优于现有最先进的方法。代码可在以下仓库获取：https://github.com/aikunyi/FreTS。

## 1 引言

时间序列预测在多个现实世界的行业中发挥着关键作用，例如气候条件估计[1, 2]、交通状态预测[3, 4]、经济分析[5, 6]等。在早期阶段，许多传统的统计预测方法被提出，例如指数平滑[7]和自回归移动平均（ARMA）[8]。近年来，深度学习的快速发展催生了许多深度预测模型，包括基于循环神经网络的方法（如DeepAR[9]、LSTNet[10]）、基于卷积神经网络的方法（如TCN[11]、SCINet[12]）、基于Transformer的方法（如Informer[13]、Autoformer[14]）以及基于图神经网络的方法（如MTGNN[15]、StemGNN[16]、AGCRN[17]）等。

尽管这些深度模型在某些场景中取得了令人瞩目的预测性能，但其复杂的网络架构通常会在训练或推理阶段带来昂贵的计算负担。此外，这些模型的鲁棒性可能会因大量参数而受到影响，尤其是在可用训练数据有限的情况下[13, 18]。因此，最近提出了基于多层感知机（MLP）的方法，因其结构简单、复杂度低且预测性能优越而受到关注，例如N-BEATS[19]、LightTS[20]、DLinear[21]等。然而，这些基于MLP的方法依赖于逐点映射来捕捉时间映射，无法处理时间序列的全局依赖关系。此外，它们还会因时间序列的波动性和冗余的局部动量而遇到信息瓶颈，这极大地限制了其在时间序列预测中的表现。

---

为了解决上述问题，我们探索了一种新颖的方向，即在频域中应用MLP进行时间序列预测。我们研究了频域MLP在预测中的学习模式，并发现了其两个关键优势：（i）全局视角：通过对从序列变换中获取的频谱分量进行操作，频域MLP能够捕捉到更全面的信号视角，从而更容易学习全局的空间/时间依赖关系。（ii）能量压缩：频域MLP集中于具有紧凑信号能量的较小关键频率分量，因此可以在过滤噪声影响的同时保留更清晰的模式。通过实验，我们观察到频域MLP比时域MLP捕捉到更明显的全局周期性模式（如图1(a)所示），这突显了其识别全局信号的能力。此外，从图1(b)中可以看出，频域MLP学习到的权重中出现了更清晰的对角线依赖关系，而时域MLP学习到的依赖关系则更为分散。这表明频域MLP在处理复杂和噪声信息时识别最重要特征和关键模式的巨大潜力。

- 图1：时域和频域中MLP学习模式的可视化（见附录B.4）。(a) 全局视角：频域中学习的模式比时域中表现出更明显的全局周期性模式；(b) 能量压缩：频域中的学习比时域中能够识别更清晰的对角线依赖关系和关键模式。

---

为了充分利用这些优势，我们提出了FreTS，一种简单而有效的频域MLP架构，用于时间序列预测。FreTS的核心思想是在频域中学习时间序列预测映射。具体来说，FreTS主要包括两个阶段：（i）域转换：首先通过离散傅里叶变换（DFT）[22]将原始的时域序列信号转换为频域频谱，其中频谱由多个复数形式的频率分量组成，包括实系数和虚系数。（ii）频率学习：在给定实/虚系数的基础上，我们重新设计了用于复数的频域MLP，分别考虑实映射和虚映射。通过两个独立的MLP分别学习输出的实部和虚部，然后将它们堆叠在一起，从而从频率分量恢复为最终的预测结果。此外，FreTS在序列间和序列内尺度上执行上述两个阶段，这进一步促进了频域中通道间和时间依赖关系的学习，从而提升预测性能。我们在13个基准数据集上进行了广泛的实验，涵盖7个短期预测和6个长期预测数据集，实验结果证明了我们相比现有最先进方法的一致优越性。

# 2 相关工作

##### 时间域中的预测

传统上，时间域中的预测主要依赖于统计方法，包括自回归移动平均模型（ARMA）[8]、向量自回归模型（VAR）[23]和自回归积分滑动平均模型（ARIMA）[24]。近年来，基于深度学习的方法由于能够提取非线性和复杂的相关性，在时间序列预测中得到了广泛应用[25, 26]。这些方法通过循环神经网络（RNN）（如DeepAR [9]、LSTNet [10]）和卷积神经网络（CNN）（如TCN [11]、SCINet [12]）学习时间域中的依赖关系。此外，基于图神经网络（GNN）的模型因其在建模时间域中变量之间的序列依赖关系方面表现出色，提出了如TAMP-S2GCNets [4]、AGCRN [17]、MTGNN [15]和GraphWaveNet [27]等模型。同时，基于Transformer的预测方法因其注意力机制在建模长程依赖关系方面的能力而被引入，如Reformer [18]和Informer [13]。

---

##### 频域中的预测

最近的一些时间序列预测方法通过提取频域知识进行预测[28]。具体而言，SFM [29]通过离散傅里叶变换（DFT）将LSTM的隐藏状态分解为频率。StemGNN [16]基于图傅里叶变换（GFT）进行图卷积，并基于离散傅里叶变换计算序列相关性。Autoformer [14]通过提出基于快速傅里叶变换（FFT）的自相关机制替代了自注意力机制。FEDformer [30]提出了一种基于DFT的频率增强注意力机制，通过查询和键的频谱获得注意力权重，并在频域中计算加权和。CoST [31]使用DFT将中间特征映射到频域，以实现表示中的交互。FiLM [32]利用傅里叶分析保留历史信息并去除噪声信号。与这些通过频域技术改进原始架构（如Transformer和GNN）的研究不同，本文提出了一种新的频域学习架构，能够在频域中同时学习通道间和时间上的依赖关系。

---

##### 基于MLP的预测模型

一些研究探索了基于多层感知机（MLP）的网络在时间序列预测中的应用。N-BEATS [19]利用堆叠的MLP层和双重残差学习处理输入数据，以迭代地预测未来。DEPTS [33]应用傅里叶变换提取周期，并使用MLP进行单变量预测中的周期性依赖建模。LightTS [20]使用轻量级的采样导向MLP结构，在保持准确性的同时降低复杂性和计算时间。N-HiTS [34]将多速率输入采样和分层插值与MLP结合，用于单变量预测。LTSF-Linear [35]提出了一组极其简单的一层线性模型，用于学习输入和输出序列之间的时间关系。这些研究展示了基于MLP的网络在时间序列预测任务中的有效性，并启发了本文中频域MLP的研发。

# 3 FreTS

在本节中，我们将详细阐述我们提出的基于频率域重新设计的MLPs（多层感知器）的新方法——FreTS，用于时间序列预测。首先，我们在第3.1节中介绍FreTS的详细频率学习架构，主要包括带有域转换的双重频率学习器。然后，我们在第3.2节中详细介绍上述频率学习器所采用的重新设计的频率域MLPs。此外，我们还从理论上分析了它们的全局视角和能量压缩的优越性，如第1节所述。

---

**问题定义**。令 $\left[X_{1}, X_{2}, \cdots, X_{T}\right] \in \mathbb{R}^{N \times T}$ 表示具有 $N$ 个序列和 $T$ 个时间戳的规则采样的多变量时间序列数据集，其中 $X_{t} \in \mathbb{R}^{N}$ 表示在时间戳 $t$ 处 $N$ 个不同序列的多变量值。我们考虑在时间戳 $t$ 处长度为 $L$ 的时间序列回看窗口作为模型输入，即 $\mathbf{X}_{t}=\left[X_{t-L+1}, X_{t-L+2}, \cdots, X_{t}\right] \in \mathbb{R}^{N \times L}$；同时，我们考虑在时间戳 $t$ 处长度为 $\tau$ 的预测窗口作为预测目标，记为 $\mathbf{Y}_{t}=$ $\left[X_{t+1}, X_{t+2}, \cdots, X_{t+\tau}\right] \in \mathbb{R}^{N \times \tau}$。那么，时间序列预测的公式化表示是使用历史观测值 $\mathbf{X}_{t}$ 来预测未来值 $\hat{\mathbf{Y}}_{t}$，典型的预测模型 $f_{\theta}$ 由参数 $\theta$ 参数化，通过 $\hat{\mathbf{Y}}_{t}=f_{\theta}\left(\mathbf{X}_{t}\right)$ 生成预测结果。

## 3.1 频率学习架构

FreTS的频率学习架构如图2所示，主要涉及域转换/反转换阶段、频率域MLPs以及相应的两个学习器，即频率通道学习器和频率时间学习器。此外，在输入到学习器之前，我们具体应用了一个维度扩展块来增强模型的能力。具体来说，输入的回看窗口 $\mathbf{X}_{t} \in \mathbb{R}^{N \times L}$ 与一个可学习的权重向量 $\phi_{d} \in \mathbb{R}^{1 \times d}$ 相乘，以获得更具表达性的隐藏表示 $\mathbf{H}_{t} \in \mathbb{R}^{N \times L \times d}$，即 $\mathbf{H}_{t}=\mathbf{X}_{t} \times \phi_{d}$，以引入更多的语义信息，灵感来源于词嵌入[36]。

- 图2：FreTS的框架概览：频率通道学习器专注于通过频率域MLPs在通道维度上建模序列间的依赖关系；频率时间学习器通过在时间维度上执行频率域MLPs来捕捉时间依赖关系。

---

**域转换/反转换**。傅里叶变换的使用能够将时间序列信号分解为其组成的频率分量。这对于时间序列分析特别有利，因为它有助于识别数据中的周期性或趋势模式，这些模式在预测任务中通常非常重要。正如前文图1(a)所述，在频域中学习有助于捕捉更多的周期性模式。鉴于此，我们将输入 $\mathbf{H}$ 转换为频域 $\mathcal{H}$，公式如下：

$$
\begin{equation*}
\mathcal{H}(f)=\int_{-\infty}^{\infty} \mathbf{H}(v) e^{-j 2 \pi f v} \mathrm{~d} v=\int_{-\infty}^{\infty} \mathbf{H}(v) \cos (2 \pi f v) \mathrm{d} v+j \int_{-\infty}^{\infty} \mathbf{H}(v) \sin (2 \pi f v) \mathrm{d} v \tag{1}
\end{equation*}
$$

其中，$f$ 是频率变量，$v$ 是积分变量，$j$ 是虚数单位，定义为 $-1$ 的平方根；$\int_{-\infty}^{\infty} \mathbf{H}(v) \cos (2 \pi f v) \mathrm{d} v$ 是 $\mathcal{H}$ 的实部，简记为 $\operatorname{Re}(\mathcal{H})$；$\int_{-\infty}^{\infty} \mathbf{H}(v) \sin (2 \pi f v) \mathrm{d} v$ 是虚部，简记为 $\operatorname{Im}(\mathcal{H})$。因此，我们可以将公式(1)中的 $\mathcal{H}$ 重写为：$\mathcal{H}=\operatorname{Re}(\mathcal{H})+j \operatorname{Im}(\mathcal{H})$。需要注意的是，在FreTS中，我们分别在通道维度和时间维度上执行域转换。在频域中完成学习后，我们可以使用以下反转换公式将 $\mathcal{H}$ 转换回时域：

$$
\begin{equation*}
\mathbf{H}(v)=\int_{-\infty}^{\infty} \mathcal{H}(f) e^{j 2 \pi f v} \mathrm{~d} f=\int_{-\infty}^{\infty}\left(\operatorname{Re}(\mathcal{H}(f))+j \operatorname{Im}(\mathcal{H}(f)) e^{j 2 \pi f v} \mathrm{~d} f\right. \tag{2}
\end{equation*}
$$

其中，我们将频率 $f$ 作为积分变量。实际上，频谱 $\mathcal{H}$ 表示为具有不同频率和振幅的 $\cos$ 和 $\sin$ 波的组合，这些频率和振幅反映了时间序列信号中的不同周期性特征。因此，检查频谱可以更好地识别时间序列中的显著频率和周期性模式。在接下来的章节中，为简洁起见，我们将公式(1)称为DomainConversion，公式(2)称为DomainInversion。

---

**频率通道学习器**。在时间序列预测中考虑通道依赖关系非常重要，因为它允许模型捕捉不同变量之间的交互和相关性，从而提高预测的准确性。频率通道学习器实现了不同通道之间的通信；它在每个时间戳上操作，通过在 $L$ 个时间戳之间共享相同的权重来学习通道依赖关系。具体来说，频率通道学习器以 $\mathbf{H}_{t} \in \mathbb{R}^{N \times L \times d}$ 作为输入。给定第 $l$ 个时间戳 $\mathbf{H}_{t}^{:,(l)} \in \mathbb{R}^{N \times d}$，我们通过以下方式执行频率通道学习器：

$$
\begin{align*}
\mathcal{H}_{c h a n}^{:,(l)} & =\text { DomainConversion }_{(\text {chan })}\left(\mathbf{H}_{t}^{;,(l)}\right) \\
\mathcal{Z}_{\text {chan }}^{:,(l)} & =\operatorname{FreMLP}\left(\mathcal{H}_{\text {chan }}^{:,(l)}, \mathcal{W}^{\text {chan }}, \mathcal{B}^{\text {chan }}\right)  \tag{3}\\
\mathbf{Z}^{:,(l)} & =\text { DomainInversion }_{(\text {chan })}\left(\mathcal{Z}_{\text {chan }}^{:,(l)}\right)
\end{align*}
$$

其中，$\mathcal{H}_{c h a n}^{;,(l)} \in \mathbb{C}^{\frac{N}{2} \times d}$ 是 $\mathbf{H}_{t}^{;,(l)}$ 的频率分量；DomainConversion ${ }_{(c h a n)}$ 和 DomainInversion ${ }_{(\text {chan })}$ 表示这些操作是沿着通道维度执行的。FreMLP 是第3.2节中提出的频率域MLPs，它以 $\mathcal{W}^{\text {chan }}=\left(\mathcal{W}_{r}^{\text {chan }}+\right.$ $\left.j \mathcal{W}_{i}^{\text {chan }}\right) \in \mathbb{C}^{d \times d}$ 作为复数权重矩阵，其中 $\mathcal{W}_{r}^{\text {chan }} \in \mathbb{R}^{d \times d}$ 和 $\mathcal{W}_{i}^{\text {chan }} \in \mathbb{R}^{d \times d}$，以及 $\mathcal{B}^{\text {chan }}=\left(\mathcal{B}_{r}^{\text {chan }}+j \mathcal{B}_{i}^{\text {chan }}\right) \in \mathbb{C}^{d}$ 作为偏置，其中 $\mathcal{B}_{r}^{\text {chan }} \in \mathbb{R}^{d}$ 和 $\mathcal{B}_{i}^{\text {chan }} \in \mathbb{R}^{d}$。$\mathcal{Z}_{\text {chan }}^{:,(l)} \in \mathbb{C}^{\frac{N}{2} \times d}$ 是FreMLP的输出，同样位于频域，它被转换回时域为 $\mathbf{Z}^{;,(l)} \in \mathbb{R}^{N \times d}$。最后，我们将 $L$ 个时间戳的 $\mathbf{Z}^{;,(l)}$ 整合为一个整体，并输出 $\mathbf{Z}_{t} \in \mathbb{R}^{N \times L \times d}$。

---

**频率时间学习器**。频率时间学习器旨在频域中学习时间模式；同时，它基于频率域MLPs构建，并在每个通道上执行，同时在 $N$ 个通道之间共享权重。具体来说，它以频率通道学习器的输出 $\mathbf{Z}_{t} \in \mathbb{R}^{N \times L \times d}$ 作为输入，对于第 $n$ 个通道 $\mathbf{Z}_{t}^{(n),:} \in \mathbb{R}^{L \times d}$，我们通过以下方式应用频率时间学习器：

$$
\begin{align*}
& \mathcal{Z}_{\text {temp }}^{(n),:}=\operatorname{DomainConversion}_{(\text {tem } p)}\left(\mathbf{Z}_{t}^{(n),:}\right) \\
& \mathcal{S}_{\text {temp }}^{(n),:}=\operatorname{FreMLP}\left(\mathcal{Z}_{\text {temp }}^{(n),:}, \mathcal{W}^{\text {temp }}, \mathcal{B}^{\text {temp }}\right)  \tag{4}\\
& \mathbf{S}^{(n),:}=\operatorname{DomainInversion}_{(\text {temp })}\left(\mathcal{S}_{\text {temp }}^{(n),:}\right)
\end{align*}
$$

其中，$\mathcal{Z}_{\text {temp }}^{(n),:} \in \mathbb{C}^{\frac{L}{2} \times d}$ 是 $\mathbf{Z}_{t}^{(n),:}$ 对应的频谱；DomainConversion $_{(\text {temp })}$ 和 DomainInversion $_{(\text {temp })}$ 表示这些计算是沿着时间维度执行的。$\mathcal{W}^{\text {temp }}=\left(\mathcal{W}_{r}^{\text {temp }}+j \mathcal{W}_{i}^{\text {temp }}\right) \in \mathbb{C}^{d \times d}$ 是复数权重矩阵，其中 $\mathcal{W}_{r}^{\text {temp }} \in \mathbb{R}^{d \times d}$ 和 $\mathcal{W}_{i}^{\text {temp }} \in \mathbb{R}^{d \times d}$，$\mathcal{B}^{\text {temp }}=\left(\mathcal{B}_{r}^{\text {temp }}+j \mathcal{B}_{i}^{\text {temp }}\right) \in \mathbb{C}^{d}$ 是复数偏置，其中 $\mathcal{B}_{r}^{\text {temp }} \in \mathbb{R}^{d}$ 和 $\mathcal{B}_{i}^{\text {temp }} \in \mathbb{R}^{d}$。$\mathcal{S}_{\text {temp }}^{(n),:} \in \mathbb{C}^{\frac{L}{2} \times d}$ 是FreMLP的输出，并被转换回时域为 $\mathbf{S}^{(n),:} \in \mathbb{R}^{L \times d}$。最后，我们将所有通道整合并输出 $\mathbf{S}_{t} \in \mathbb{R}^{N \times L \times d}$。

---

**投影**。最后，我们利用学习到的通道和时间依赖关系，通过一个两层前馈网络（FFN）对未来 $\tau$ 个时间戳 $\hat{\mathbf{Y}}_{t} \in \mathbb{R}^{N \times \tau}$ 进行预测。这种一步前向的方式可以避免误差累积，公式如下：

$$
\begin{equation*}
\hat{\mathbf{Y}}_{t}=\sigma\left(\mathbf{S}_{t} \phi_{1}+\mathbf{b}_{1}\right) \phi_{2}+\mathbf{b}_{2} \tag{5}
\end{equation*}
$$

其中，$\mathbf{S}_{t} \in \mathbb{R}^{N \times L \times d}$ 是频率时间学习器的输出，$\sigma$ 是激活函数，$\phi_{1} \in \mathbb{R}^{(L * d) \times d_{h}}$ 和 $\phi_{2} \in \mathbb{R}^{d_{h} \times \tau}$ 是权重，$\mathbf{b}_{1} \in \mathbb{R}^{d_{h}}$ 和 $\mathbf{b}_{2} \in \mathbb{R}^{\tau}$ 是偏置，$d_{h}$ 是内层维度大小。

## 3.2 频率域MLPs

如图3所示，我们详细阐述了在FreTS中重新设计的用于复数频率分量的频率域MLPs，以有效捕捉时间序列的关键模式，并具有全局视角和能量压缩特性，如第1节所述。

**定义1（频率域MLPs）**。形式上，对于一个复数输入 $\mathcal{H} \in \mathbb{C}^{m \times d}$，给定一个复数权重矩阵 $\mathcal{W} \in \mathbb{C}^{d \times d}$ 和一个复数偏置 $\mathcal{B} \in \mathbb{C}^{d}$，则频率域MLPs可以表示为：

$$
\begin{align*}
\mathcal{Y}^{\ell} & =\sigma\left(\mathcal{Y}^{\ell-1} \mathcal{W}^{\ell}+\mathcal{B}^{\ell}\right)  \tag{6}\\
\mathcal{Y}^{0} & =\mathcal{H}
\end{align*}
$$

其中，$\mathcal{Y}^{\ell} \in \mathbb{C}^{m \times d}$ 是最终输出，$\ell$ 表示第 $\ell$ 层，$\sigma$ 是激活函数。

---

由于 $\mathcal{H}$ 和 $\mathcal{W}$ 都是复数，根据复数的乘法规则（详见附录C），我们将公式(6)进一步扩展为：
$$
\begin{equation*}
\mathcal{Y}^{\ell}=\sigma\left(\operatorname{Re}\left(\mathcal{Y}^{\ell-1}\right) \mathcal{W}_{r}^{\ell}-\operatorname{Im}\left(\mathcal{Y}^{\ell-1}\right) \mathcal{W}_{i}^{\ell}+\mathcal{B}_{r}^{\ell}\right)+j \sigma\left(\operatorname{Re}\left(\mathcal{Y}^{\ell-1}\right) \mathcal{W}_{i}^{\ell}+\operatorname{Im}\left(\mathcal{Y}^{\ell-1}\right) \mathcal{W}_{r}^{\ell}+\mathcal{B}_{i}^{\ell}\right) \tag{7}
\end{equation*}
$$

其中，$\mathcal{W}^{\ell}=\mathcal{W}_{r}^{\ell}+j \mathcal{W}_{i}^{\ell}$，$\mathcal{B}^{\ell}=\mathcal{B}_{r}^{\ell}+j \mathcal{B}_{i}^{\ell}$。根据该公式，我们通过对频率分量的实部和虚部分别计算来实现频率域MLPs（简称为FreMLP），然后将它们组合成复数以获得最终结果。具体实现过程如图3所示。

---

**定理1**。假设 $\mathbf{H}$ 是原始时间序列的表示，$\mathcal{H}$ 是其频谱对应的频率分量，那么时间序列在时域中的能量等于其在频域中表示的能量。形式上，我们可以用上述符号表示为：

$$
\begin{equation*}
\int_{-\infty}^{\infty}|\mathbf{H}(v)|^{2} \mathrm{~d} v=\int_{-\infty}^{\infty}|\mathcal{H}(f)|^{2} \mathrm{~d} f \tag{8}
\end{equation*}
$$

其中，$\mathcal{H}(f)=\int_{-\infty}^{\infty} \mathbf{H}(v) e^{-j 2 \pi f v} \mathrm{~d} v$，$v$ 是时间/通道维度，$f$ 是频率维度。

---

我们在附录D.1中包含了证明。该定理表明，如果时间序列的大部分能量集中在少量频率分量中，那么仅使用这些分量就可以准确地表示时间序列。因此，丢弃其他分量不会显著影响信号的能量。如图1(b)所示，在频域中，能量集中在较少的频率分量上，因此在频谱中学习有助于保留更清晰的模式。

---

**定理2**。给定时间序列输入 $\mathbf{H}$ 及其对应的频域转换 $\mathcal{H}$，频率域MLP对 $\mathcal{H}$ 的操作可以表示为时域中对 $\mathbf{H}$ 的全局卷积。这可以表示为：

$$
\begin{equation*}
\mathcal{H W}+\mathcal{B}=\mathcal{F}(\mathbf{H} * W+B) \tag{9}
\end{equation*}
$$

其中，$*$ 是循环卷积，$\mathcal{W}$ 和 $\mathcal{B}$ 是复数权重和偏置，$W$ 和 $B$ 是时域中的权重和偏置，$\mathcal{F}$ 是离散傅里叶变换（DFT）。

---

证明见附录D.2。因此，FreMLP的操作，即 $\mathcal{H W}+\mathcal{B}$，等同于时域中的操作 $(\mathbf{H} * W+B)$。这意味着频率域MLPs的操作可以被视为时域中的全局卷积。

## 4 实验

为了评估FreTS的性能，我们在13个真实世界的时间序列基准上进行了广泛的实验，涵盖短期预测和长期预测设置，并与相应的最先进方法进行比较。

---

**数据集**。我们的实验结果基于多个领域的数据集，包括交通、能源、网络、心电图和医疗等。具体来说，对于短期预测任务，我们采用了Solar ${ }^{2}$、Wiki [37]、Traffic [37]、Electricity ${ }^{3}$、ECG [16]、METR-LA [38] 和 COVID-19 [4] 数据集，遵循之前的预测文献[16]。对于长期预测任务，我们采用了Weather [14]、Exchange [10]、Traffic [14]、Electricity [14] 和 ETT 数据集[13]，遵循之前的长时序预测工作[13, 14, 30, 39]。我们按照[16, 13, 14]的方法对所有数据集进行预处理，并使用最小-最大归一化进行标准化。我们将数据集按7:2:1的比例划分为训练集、验证集和测试集，除了COVID-19数据集按6:2:2划分。更多数据集细节见附录B.1。

---

**基线模型**。我们将FreTS与短期和长期预测任务中的代表性及最先进模型进行比较，以评估其有效性。对于短期预测，我们将FreTS与VAR [23]、SFM [29]、LSTNet [10]、TCN [11]、GraphWaveNet [27]、DeepGLO [37]、StemGNN [16]、MTGNN [15] 和 AGCRN [17] 进行比较。我们还纳入了需要预定义图结构的TAMP-S2GCNets [4]、DCRNN [38] 和 STGCN [40] 进行比较。对于长期预测，我们纳入了Informer [13]、Autoformer [14]、Reformer [18]、FEDformer [30]、LTSF-Linear [35] 以及最近的PatchTST [39] 进行比较。关于基线模型的更多细节见附录B.2。

---

**实现细节**。我们的模型使用Pytorch 1.8 [41]实现，所有实验均在单个NVIDIA RTX 3080 10GB GPU上进行。我们采用均方误差（MSE）作为损失函数，并报告平均绝对误差（MAE）和均方根误差（RMSE）作为评估指标。更多实现细节请参考附录B.3。

## 4.1 主要结果

**短期时间序列预测**  

表1展示了我们的FreTS与13个基线模型在六个数据集上的预测准确性比较，输入长度为12，预测长度为12。最佳结果以粗体显示，次佳结果以下划线标注。从表中可以看出，FreTS在所有数据集的MAE和RMSE指标上均优于所有基线模型，平均在MAE上提升了$9.4\%$，在RMSE上提升了$11.6\%$。我们认为这归功于FreTS显式地建模了通道和时间依赖性，并在频域中灵活地统一了通道和时间建模，从而能够有效地捕捉具有全局视角和能量压缩的关键模式。我们还在附录F.1中报告了不同步长下不同数据集（包括METR-LA数据集）的短期预测完整基准结果。

- 表1：短期预测比较。最佳结果以粗体显示，次佳结果以下划线标注。短期预测的完整基准结果见附录F.1。

---

**长期时间序列预测**  

表2展示了FreTS与六个代表性基线模型在六个基准数据集上不同预测长度下的长期预测结果。对于交通数据集，我们选择48作为回看窗口大小$L$，预测长度为$\tau \in \{48,96,192,336\}$。对于其他数据集，输入回看窗口长度设置为96，预测长度设置为$\tau \in \{96,192,336,720\}$。结果表明，FreTS在所有数据集上均优于所有基线模型。定量来看，与基于Transformer模型的最佳结果相比，FreTS在MAE和RMSE上平均降低了超过$20\%$。与最近的LSTF-Linear [35] 和 SOTA PatchTST [39] 相比，FreTS在总体上仍然表现更优。此外，我们在附录F.2中提供了FreTS与其他基线模型的进一步比较，并报告了不同回看窗口大小下的性能。结合表1和表2，我们可以得出结论，FreTS在短期和长期预测任务中均取得了具有竞争力的性能。

- 表2：长期预测比较。我们设置回看窗口大小$L$为96，预测长度为$\tau \in \{96,192,336,720\}$，除了交通数据集的预测长度设置为$\tau \in \{48,96,192,336\}$。最佳结果以粗体显示，次佳结果以下划线标注。长期预测的完整结果见附录F.2。

## 4.2 模型分析

##### 频率通道和时间学习器

我们在短期和长期实验设置中分析了频率通道和时间学习器的影响，结果如表3所示。我们考虑了两个变体：FreCL（从FreTS中移除频率时间学习器）和FreTL（从FreTS中移除频率通道学习器）。通过对比，我们观察到频率通道学习器在短期预测中扮演了更重要的角色。而在长期预测中，频率时间学习器比频率通道学习器更为有效。在附录E.1中，我们还进行了其他数据集的实验并报告了性能。有趣的是，我们发现通道学习器在某些长期预测案例中会导致性能下降。一个潜在的解释是，通道独立策略[39]为预测带来了更多的好处。

- 表3：频率通道和时间学习器在短期和长期预测中的消融研究。'I/O'表示回看窗口大小/预测长度。

---

##### FreMLP vs. MLP

我们进一步研究了FreMLP在时间序列预测中的有效性。我们使用FreMLP替换现有基于MLP的SOTA模型（即DLinear和NLinear [35]）中的原始MLP组件，并在相同的实验设置下比较它们的性能。实验结果如表4所示。从表中可以很容易地观察到，对于任何预测长度，DLinear和NLinear模型在将相应的MLP组件替换为我们的FreMLP后，性能均有所提升。定量来看，将FreMLP引入DLinear模型在Exchange数据集上带来了平均6.4%的MAE提升和11.4%的RMSE提升，在Weather数据集上带来了4.9%的MAE提升和3.5%的RMSE提升。根据表4，NLinear模型在这两个数据集上也实现了类似的提升。这些结果再次证实了FreMLP相比于MLP的有效性，我们在附录B.5中包含了更多实现细节和分析。

- 表4：在Exchange和Weather数据集上的消融研究，回看窗口大小为96，预测长度$\tau \in\{96,192,336,720\}$。DLinear (FreMLP)/NLinear (FreMLP)表示我们将DLinear/NLinear中的MLP替换为FreMLP。最佳结果以粗体显示。

## 4.3 效率分析

我们提出的FreTS的复杂度为$\mathcal{O}(N \log N+L \log L)$。我们分别在不同变量数量$N$和预测长度$\tau$下与一些基于GNN的SOTA方法和基于Transformer的模型进行了效率对比。在Wiki数据集上，我们在回看窗口大小为12、预测长度为12的条件下，对$N \in\{1000,2000,3000,4000,5000\}$进行了实验，结果如图4(a)所示。从图中可以发现：(1) FreTS的参数数量与$N$无关。(2) 与AGCRN相比，FreTS平均减少了30%的参数数量和20%的训练时间。在Exchange数据集上，我们在输入长度为96的条件下，对不同预测长度$\tau \in\{96,192,336,480\}$进行了实验，结果如图4(b)所示。实验表明：(1) 与基于Transformer的方法（FEDformer [30]、Autoformer [14]和Informer [13]）相比，FreTS减少了至少3倍的参数数量。(2) FreTS的训练时间平均比Informer快3倍，比Autoformer快5倍，比FEDformer快10倍以上。这些结果展示了我们在实际部署中的巨大潜力。

- 图4：在Wiki和Exchange数据集上的效率分析（模型参数和训练时间）。(a) 不同变量数量下的效率对比：在Wiki数据集上，变量数量从1000增加到5000，输入窗口大小为12，预测长度为12。(b) 不同预测长度下的效率对比：在Exchange数据集上，在相同窗口大小为96的条件下，预测长度从96延长到480。

---

## 4.4 可视化分析

在图5中，我们在Traffic数据集上可视化了FreMLP中学习到的权重$\mathcal{W}$，回看窗口大小为48，预测长度为192。由于权重$\mathcal{W}$是复数，我们分别提供了实部$\mathcal{W}_{r}$（如图(a)所示）和虚部$\mathcal{W}_{i}$（如图(b)所示）的可视化。从图中可以观察到，实部和虚部在学习过程中都起到了关键作用：实部或虚部的权重系数表现出能量聚集特征（清晰的对角线模式），这有助于学习显著特征。在附录E.2中，我们进一步详细分析了实部和虚部在不同预测场景中的作用，以及它们在FreMLP中的影响。我们分别检验了它们的贡献，并研究了它们如何影响最终性能。更多在不同数据集和设置下的权重可视化，以及全局周期性模式的可视化，分别可以在附录G.1和附录G.2中找到。

- 图5：在Traffic数据集上可视化FreMLP学习到的权重。$\mathcal{W}_{r}$表示$\mathcal{W}$的实部，$\mathcal{W}_{i}$表示$\mathcal{W}$的虚部。

# 5 总结

在本文中，我们探索了一个新的方向，并尝试将频域MLP应用于时间序列预测。我们重新设计了频域中的MLP，使其能够从全局视角和能量压缩的角度有效捕捉时间序列的潜在模式。随后，我们通过一个简单而有效的架构——FreTS，验证了这一设计。FreTS基于频域MLP构建，用于时间序列预测。我们在七个短期预测基准和六个长期预测基准上进行了全面的实验，验证了我们所提出方法的优越性。简单的MLP具有诸多优势，并为现代深度学习奠定了基础，其在高效率下实现满意性能的潜力巨大。我们希望这项工作能够促进未来更多关于MLP在时间序列建模中的研究。

# B 实验细节

## B.1 数据集

我们在实验中采用了十三个真实世界基准数据集，以评估短期和长期预测的准确性。数据集的详细信息如下：

---

**Solar**${ }^{4}$：该数据集由美国国家可再生能源实验室收集的太阳能发电数据组成。我们选择佛罗里达州的发电厂数据点作为数据集，其中包含593个数据点。数据采集时间为2006年1月1日至2016年12月31日，采样间隔为每小时一次。

**Wiki [37]**：该数据集包含不同维基百科文章的每日浏览量，采集时间为2015年7月1日至2016年12月31日。它包含约$145k$条时间序列，我们从中随机选择$5k$条作为实验数据集。

**Traffic [37]**：该数据集包含旧金山963条高速公路车道的每小时交通数据，用于短期预测设置；而长期预测设置中包含862条车道。数据采集从2015年1月1日开始，采样间隔为每小时一次。

**ECG**${ }^{5}$：该数据集来自UCR时间序列分类档案中的心电图（ECG）数据。它包含140个节点，每个节点的长度为5000。

---

**Electricity**：该数据集包含370个客户的电力消耗数据，用于短期预测；而长期预测设置中包含321个客户的电力消耗数据。数据采集从2011年1月1日开始，采样间隔为每15分钟一次。

**COVID-19 [4]**：该数据集由约翰斯·霍普金斯大学提供的美国加利福尼亚州（CA）的COVID-19住院数据，采集时间为2020年1月2日至2020年12月31日，采样间隔为每天一次。

**METR-LA**${ }^{7}$：该数据集包含从洛杉矶县高速公路的环形检测器收集的交通信息。它包含207个传感器，数据采集时间为2012年3月1日至2012年6月30日，采样间隔为每5分钟一次。

---

**Exchange**${ }^{8}$：该数据集包含八个国家（澳大利亚、英国、加拿大、瑞士、中国、日本、新西兰和新加坡）的每日汇率数据，时间跨度为1990年至2016年，采样间隔为每天一次。

**Weather**${ }^{9}$：该数据集收集了德国马克斯·普朗克生物地球化学研究所气象站在2020年的21个气象指标（如湿度和气温）。采样间隔为每10分钟一次。

**ETT**${ }^{10}$：该数据集从两个不同的电力变压器（标记为1和2）收集，每个变压器包含两种不同的分辨率（15分钟和1小时），分别标记为$m$和$h$。我们使用ETTh1和ETTm1作为长期预测基准。

## B.2 基线模型

我们采用了十八个具有代表性且处于前沿的基线模型进行比较，包括基于LSTM的模型、基于GNN的模型和基于Transformer的模型。以下是这些模型的介绍：

**VAR [23]**：VAR是一种经典的线性自回归模型。我们使用Statsmodels库（https://www.statsmodels.org）来实现VAR，该库是一个提供统计计算的Python包。

**DeepGLO [37]**：DeepGLO通过矩阵分解建模变量之间的关系，并采用时间卷积神经网络引入非线性关系。我们从https://github.com/rajatsen91/deepglo下载了源代码。对于Wiki、Electricity和Traffic数据集，我们使用推荐的配置作为实验设置。对于COVID-19数据集，垂直和水平批次大小设置为64，全局模型的秩设置为64，通道数设置为$[32,32,32,1]$，周期设置为7。

---

**LSTNet [10]**：LSTNet使用CNN捕捉变量间的关系，并使用RNN发现长期模式。我们从https://github.com/laiguokun/LSTNet下载了源代码。在实验中，我们使用推荐的配置：CNN隐藏单元数为100，CNN层的核大小为4，dropout为0.2，RNN隐藏单元数为100，RNN隐藏层数为1，学习率为0.001，优化器为Adam。

**TCN [11]**：TCN是一种用于回归预测的因果卷积模型。我们从https://github.com/locuslab/TCN下载了源代码。我们使用与开源代码中多音音乐任务示例相同的配置：dropout为0.25，核大小为5，隐藏单元数为150，层数为4，优化器为Adam。

---

**Informer [13]**：Informer利用高效的自注意力机制编码变量之间的依赖关系。我们从https://github.com/zhouhaoyi/Informer2020下载了源代码。我们使用推荐的配置作为实验设置：dropout为0.05，编码器层数为2，解码器层数为1，学习率为0.0001，优化器为Adam。

**Reformer [18]**：Reformer结合了Transformer的建模能力与一种可以在长序列上高效执行且内存占用较小的架构。我们从https://github.com/thuml/Autoformer下载了源代码。我们使用推荐的配置作为实验设置。

---

**Autoformer [14]**：Autoformer提出了一种分解架构，通过将序列分解块嵌入为内部操作符，可以从中间预测中逐步聚合长期趋势部分。我们从https://github.com/thuml/Autoformer下载了源代码。我们使用推荐的配置作为实验设置。

**FEDformer [30]**：FEDformer提出了一种在频率上进行低秩近似的注意力机制，并通过专家分解的混合来控制分布偏移。我们从https://github.com/MAZiqing/FEDformer下载了源代码。我们使用FEB-f作为频率增强块，并选择随机模式（64）作为实验模式。

---

**SFM [29]**：在LSTM模型的基础上，SFM在细胞状态中引入了一系列不同的频率分量。我们从https://github.com/z331565360/State-Frequency-Memory-stock-prediction下载了源代码。我们遵循推荐的配置作为实验设置：学习率为0.01，频率维度为10，隐藏维度为10，优化器为RMSProp。

**StemGNN [16]**：StemGNN利用GFT和DFT在频域中捕捉变量之间的依赖关系。我们从https://github.com/microsoft/StemGNN下载了源代码。我们使用StemGNN的推荐配置作为实验设置：优化器为RMSProp，学习率为0.0001，堆叠层数为5，dropout率为0.5。

---

**MTGNN [15]**：MTGNN提出了一种有效的方法来利用多个时间序列之间的固有依赖关系。我们从https://github.com/nnzhan/MTGNN下载了源代码。由于实验数据集没有静态特征，我们将参数`load_static_feature`设置为false。我们通过自适应邻接矩阵构建图并添加图卷积层。对于其他参数，我们遵循推荐的设置。

**GraphWaveNet [27]**：GraphWaveNet引入了自适应依赖矩阵学习来捕捉隐藏的空间依赖关系。我们从https://github.com/nnzhan/Graph-WaveNet下载了源代码。由于我们的数据集没有预先定义的图结构，我们仅使用自适应邻接矩阵。我们添加了一个图卷积层并随机初始化邻接矩阵。我们采用推荐的设置作为实验配置：学习率为0.001，dropout为0.3，epoch数为50，优化器为Adam。

---

**AGCRN [17]**：AGCRN提出了一种数据自适应的图生成模块，用于从数据中发现空间相关性。我们从https://github.com/LeiBAI/AGCRN下载了源代码。我们遵循推荐的设置：嵌入维度为10，学习率为0.003，优化器为Adam。

**TAMP-S2GCNets [4]**：TAMP-S2GCNets探索了在时间感知深度学习范式中利用MP增强知识表示机制的效用。我们从https://www.dropbox.com/sh/n0ajd510tdeyb80/AABGn-ejfV1YtRwjf_L0AOsNa?dl=0下载了源代码。TAMP-S2GCNets需要一个预定义的图拓扑，我们使用源代码提供的加利福尼亚州拓扑作为输入。我们采用推荐的设置作为COVID-19的实验配置。

---

**DCRNN [38]**：DCRNN使用双向图随机游走建模空间依赖性，并通过循环神经网络捕捉时间动态。我们从https://github.com/liyaguang/DCRNN下载了源代码。我们使用推荐的配置作为实验设置：批次大小为64，学习率为0.01，输入维度为2，优化器为Adam。DCRNN需要一个预定义的图结构，我们使用METR-LA数据集提供的邻接矩阵作为预定义结构。

**STGCN [40]**：STGCN通过时空卷积块整合了图卷积和门控时间卷积。我们从https://github.com/VeritasYin/STGCN_IJCAI-18下载了源代码。我们遵循推荐的设置作为实验配置：批次大小为50，学习率为0.001，优化器为Adam。STGCN需要一个预定义的图结构，我们使用METR-LA数据集提供的邻接矩阵作为预定义结构。

---

**LTSF-Linear [35]**：LTSF-Linear提出了一组极其简单的单层线性模型，用于学习输入和输出序列之间的时间关系。我们从https://github.com/cure-lab/LTSF-Linear下载了源代码。我们将其作为长期预测的基线，并遵循推荐的设置作为实验配置。

**PatchTST [39]**：PatchTST通过引入两个关键组件（分块和通道独立结构），提出了一种基于Transformer的时间序列预测模型的有效设计。我们从https://github.com/PatchTST下载了源代码。我们将其作为长期预测的基线，并遵循推荐的设置作为实验配置。

## B.3 实现细节

默认情况下，频率通道和时间学习器均包含一层FreMLP，嵌入大小$d$为128，隐藏大小$d_{h}$设置为256。对于短期预测，Solar、METR-LA、ECG、COVID-19和Electricity数据集的批次大小设置为32，而Wiki和Traffic数据集的批次大小设置为4。对于长期预测，除了回看窗口大小外，我们遵循LTSF-Linear [35]的大部分实验设置。回看窗口大小设置为96，这是FEDformer [30]和Autoformer [14]推荐的。在附录F.2中，我们还使用192和336作为回看窗口大小进行实验，结果表明FreTS同样优于其他基线模型。对于较长的预测长度（例如336、720），我们使用通道独立策略，并在模型中仅包含频率时间学习器。对于某些数据集，我们在验证集上仔细调整了超参数（包括批次大小和学习率），并选择了性能最佳的设置。我们在$\{4,8,16,32\}$范围内调整批次大小。

## B.4 可视化设置

**全局视图的可视化方法**：我们遵循LTSF-Linear [35]的可视化方法，在时域中对输入学习到的权重进行可视化（对应于图1(a)的左侧）。对于在频域中学习到的权重的可视化，我们首先将输入转换到频域，并选择输入频谱的实部来替换原始输入。然后，我们以与时域相同的方式学习并可视化权重。图1(a)的右侧展示了在Traffic数据集上学习到的权重，回看窗口为96，预测长度为96。图9展示了在Traffic数据集上学习到的权重，回看窗口为72，预测长度为336。图10展示了在Electricity数据集上学习到的权重，回看窗口为96，预测长度为96。

---

**能量压缩的可视化方法**：由于频域MLP学习到的权重$\mathcal{W}=\mathcal{W}_{r}+j \mathcal{W}_{i} \in \mathbb{C}^{d \times d}$是复数，我们分别可视化其实部$\mathcal{W}_{r}$和虚部$\mathcal{W}_{i}$。我们通过计算$1 / \max (\mathcal{W}) * \mathcal{W}$对其进行归一化，并可视化归一化后的值。图1(b)的右侧展示了在Traffic数据集上学习到的$\mathcal{W}$的实部，回看窗口为48，预测长度为192。为了可视化在时域中学习到的相应权重，我们将输入$\mathcal{Z}_{\text {temp }} \in \mathbb{C}^{N \times L \times d}$的频谱替换为原始时域输入$\mathbf{H}_{t} \in \mathbb{R}^{N \times L \times d}$，并在时域中使用权重$W \in \mathbb{R}^{d \times d}$进行计算，如图1(b)的左侧所示。

## B.5 消融实验设置

DLinear将原始数据输入分解为趋势分量和季节分量，并对每个分量应用两个单层线性层。在消融研究部分，我们将这两个线性层替换为两个不同的频域MLP（对应于表4中的DLinear (FreMLP)），并使用LTSF-Linear [35]推荐的相同实验设置比较其准确性。NLinear通过减去序列的最后一个值来调整输入，然后输入通过一个线性层，并在最终预测前将减去的部分加回。我们将线性层替换为频域MLP（对应于表4中的NLinear (FreMLP)），并使用LTSF-Linear [35]推荐的相同实验设置比较其准确性。

# C 复数乘法

对于两个复数值$\mathcal{Z}_{1}=(a+j b)$和$\mathcal{Z}_{2}=(c+j d)$，其中$a$和$c$分别是$\mathcal{Z}_{1}$和$\mathcal{Z}_{2}$的实部，$b$和$d$分别是$\mathcal{Z}_{1}$和$\mathcal{Z}_{2}$的虚部。则$\mathcal{Z}_{1}$和$\mathcal{Z}_{2}$的乘积计算如下：

$$
\begin{equation*}
\mathcal{Z}_{1} \mathcal{Z}_{2}=(a+j b)(c+j d)=a c+j^{2} b d+j a d+j b c=(a c-b d)+j(a d+b c) \tag{10}
\end{equation*}
$$

其中$j^{2}=-1$。

## D 证明

## D.1 定理1的证明

**定理1**：假设$\mathbf{H}$是原始时间序列的表示，$\mathcal{H}$是其频谱的对应频率分量，则时间序列在时域中的能量等于其在频域中表示的能量。形式上，我们可以用上述符号表示为：

$$
\begin{equation*}
\int_{-\infty}^{\infty}|\mathbf{H}(v)|^{2} \mathrm{~d} v=\int_{-\infty}^{\infty}|\mathcal{H}(f)|^{2} \mathrm{~d} f \tag{11}
\end{equation*}
$$

其中$\mathcal{H}(f)=\int_{-\infty}^{\infty} \mathbf{H}(v) e^{-j 2 \pi f v} \mathrm{~d} v$，$v$是时间/通道维度，$f$是频率维度。

---

**证明**：给定原始时间序列的表示$\mathbf{H} \in \mathbb{R}^{N \times L \times d}$，我们考虑在$N$维度（通道维度）或$L$维度（时间维度）上进行积分，记为对$v$的积分，则

$$
\int_{-\infty}^{\infty}|\mathbf{H}(v)|^{2} \mathrm{~d} v=\int_{-\infty}^{\infty} \mathbf{H}(v) \mathbf{H}^{*}(v) \mathrm{d} v
$$

其中$\mathbf{H}^{*}(v)$是$\mathbf{H}(v)$的共轭。根据逆离散傅里叶变换（IDFT），$\mathbf{H}^{*}(v)=\int_{-\infty}^{\infty} \mathcal{H}^{*}(f) e^{-j 2 \pi f v} \mathrm{~d} f$，我们可以得到

$$
\begin{aligned}
\int_{-\infty}^{\infty}|\mathbf{H}(v)|^{2} \mathrm{~d} v & =\int_{-\infty}^{\infty} \mathbf{H}(v)\left[\int_{-\infty}^{\infty} \mathcal{H}^{*}(f) e^{-j 2 \pi f v} \mathrm{~d} f\right] \mathrm{d} v \\
& =\int_{-\infty}^{\infty} \mathcal{H}^{*}(f)\left[\int_{-\infty}^{\infty} \mathbf{H}(v) e^{-j 2 \pi f v} \mathrm{~d} v\right] \mathrm{d} f \\
& =\int_{-\infty}^{\infty} \mathcal{H}^{*}(f) \mathcal{H}(f) \mathrm{d} f \\
& =\int_{-\infty}^{\infty}|\mathcal{H}(f)|^{2} \mathrm{~d} f
\end{aligned}
$$

证毕。

---

因此，时间序列在时域中的能量等于其在频域中表示的能量。

## D.2 定理2的证明

**定理2**：给定时间序列输入$\mathbf{H}$及其对应的频域转换$\mathcal{H}$，频域MLP对$\mathcal{H}$的操作可以表示为时域中对$\mathbf{H}$的全局卷积。具体表示为：

$$
\begin{equation*}
\mathcal{H W}+\mathcal{B}=\mathcal{F}(\mathbf{H} * W+B) \tag{12}
\end{equation*}
$$

其中，$*$表示循环卷积，$\mathcal{W}$和$\mathcal{B}$是复数权重和偏置，$W$和$B$是时域中的权重和偏置，$\mathcal{F}$表示离散傅里叶变换（DFT）。

---

**证明**：假设我们在$N$维度（即通道维度）或$L$维度（即时间维度）上进行操作，则

$$
\mathcal{F}(\mathbf{H}(v) * W(v))=\int_{-\infty}^{\infty}(\mathbf{H}(v) * W(v)) e^{-j 2 \pi f v} \mathrm{~d} v
$$

根据卷积定理，$\mathbf{H}(v) * W(v)=\int_{-\infty}^{\infty}(\mathbf{H}(\tau) W(v-\tau)) \mathrm{d} \tau$，则

$$
\begin{aligned}
\mathcal{F}(\mathbf{H}(v) * W(v)) & =\int_{-\infty}^{\infty} \int_{-\infty}^{\infty}(\mathbf{H}(\tau) W(v-\tau)) e^{-j 2 \pi f v} \mathrm{~d} \tau \mathrm{~d} v \\
& =\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} W(v-\tau) e^{-j 2 \pi f v} \mathrm{~d} v \mathbf{H}(\tau) \mathrm{d} \tau
\end{aligned}
$$

令$x=v-\tau$，则

$$
\begin{aligned}
\mathcal{F}(\mathbf{H}(v) * W(v)) & =\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} W(x) e^{-j 2 \pi f(x+\tau)} \mathrm{d} x \mathbf{H}(\tau) \mathrm{d} \tau \\
& =\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} W(x) e^{-j 2 \pi f x} e^{-j 2 \pi f \tau} \mathrm{~d} x \mathbf{H}(\tau) \mathrm{d} \tau \\
& =\int_{-\infty}^{\infty} \mathbf{H}(\tau) e^{-j 2 \pi f \tau} \mathrm{~d} \tau \int_{-\infty}^{\infty} W(x) e^{-j 2 \pi f x} \mathrm{~d} x \\
& =\mathcal{H}(f) \mathcal{W}(f)
\end{aligned}
$$

因此，时域中的$(\mathbf{H}(v) * W(v))$等于频域中的$(\mathcal{H}(f) \mathcal{W}(f))$。因此，频域MLP在通道维度（即$v=N$）或时间维度（即$v=L$）上的操作$(\mathcal{H W}+\mathcal{B})$，等于时域中的操作$(\mathbf{H} * W+B)$。这表明频域MLP可以被视为时域中的全局卷积。证毕。

# E 进一步分析

## E.1 消融研究

在本节中，我们进一步分析了不同预测长度下频率通道和时间学习器在ETTm1和ETTh1数据集上的效果。结果如表6所示。结果表明，随着预测长度的增加，频率时间学习器比通道学习器更为有效。特别是当预测长度较长时（例如336、720），通道学习器会导致性能下降。原因是当预测长度变长时，带有通道学习器的模型在训练过程中容易过拟合。因此，对于具有较长预测长度的长期预测，通道独立策略可能更为有效，如PatchTST [39]中所述。

## E.2 实部/虚部的影响

为了研究实部和虚部的影响，我们在Exchange和ETTh1数据集上进行了实验，预测长度$L \in\{96,192\}$，回看窗口为96。此外，我们分析了FreMLP权重$\mathcal{W}=\mathcal{W}_{r}+j \mathcal{W}_{i}$中$\mathcal{W}_{r}$和$\mathcal{W}_{i}$的影响。在本实验中，我们仅在模型中使用频率时间学习器。结果如表7所示。在表中，Input $_{\text {real }}$表示我们仅将输入的实部输入网络，Input ${ }_{\text {imag }}$表示我们仅将输入的虚部输入网络。$\mathcal{W}\left(\mathcal{W}_{r}\right)$表示我们将$\mathcal{W}_{i}$设置为0，$\mathcal{W}\left(\mathcal{W}_{i}\right)$表示我们将$\mathcal{W}_{r}$设置为0。从表中可以看出，输入的实部和虚部都是不可或缺的，且实部比虚部更为重要，而$\mathcal{W}$的实部对模型性能的影响更为显著。

## E.3 参数敏感性

我们在ECG数据集上进行了大量实验，以评估输入长度$L$和嵌入维度大小$d$的敏感性。(1) 输入长度：我们在ECG数据集上调整输入长度，取值为$\{6,12,18,24,30,36,42,50,60\}$，预测长度为12，结果如图6(a)所示。从图中可以看出，随着输入长度的增加，性能首先变好，因为较长的输入长度可能包含更多的模式信息，然后由于数据冗余或过拟合而下降。(2) 嵌入大小：我们在ECG数据集上选择嵌入大小为$\{32,64,128,256,512\}$。结果如图6(b)所示。结果表明，随着嵌入大小的增加，性能先增加后下降，因为较大的嵌入大小提高了FreTS的拟合能力，但也容易导致过拟合，特别是当嵌入大小过大时。

# F 附加结果

## F.1 多步预测

为了进一步评估FreTS在多步预测中的性能，我们在METR-LA和COVID-19数据集上进行了更多实验，输入长度为12，预测长度为$\{3,6,9,12\}$，结果分别如表8和表9所示。在本实验中，我们仅选择最先进的模型（即基于GNN和基于Transformer的模型）作为基线，因为它们比其他模型（如RNN和TCN）表现更好。在这些基线中，STGCN、DCRNN和TAMP-S2GCNets需要预定义的图结构。结果表明，FreTS在所有步骤中均优于其他基线，包括那些具有预定义图结构的模型。这进一步证实了FreTS在捕捉通道间和时间依赖关系方面的强大能力。

- 表8：在METR-LA数据集上的多步短期预测结果比较，输入长度为12，预测长度为$\tau \in\{3,6,9,12\}$。最佳结果以粗体显示，次佳结果以下划线标出。

- 表9：在COVID-19数据集上的多步短期预测结果比较，输入长度为12，预测长度为$\tau \in\{3,6,9,12\}$。最佳结果以粗体显示，次佳结果以下划线标出。

## F.2 不同回看窗口下的长期预测

在表10中，我们展示了FreTS与其他基线模型（PatchTST [39]、LTSF-linear [35]、FEDformer [30]、Autoformer [14]、Informer [13]和Reformer [18]）在Exchange数据集上不同回看窗口长度$L \in\{96,192,336\}$下的长期预测结果。预测长度为$\{96,192,336,720\}$。从表中可以看出，我们的FreTS在所有设置中均优于所有基线模型，并且相比FEDformer [30]、Autoformer [14]、Informer [13]和Reformer [18]实现了显著改进。这验证了FreTS在不同回看窗口下学习信息表示的有效性。

- 表10：不同回看窗口长度$L \in\{96,192,336\}$下的长期预测结果比较。预测长度为$\tau \in\{96,192,336,720\}$。最佳结果以粗体显示，次佳结果以下划线标出。

# G 可视化

## G.1 能量压缩的权重可视化

我们进一步可视化了在不同设置下（包括不同的回看窗口大小和预测长度）频率时间学习器中的权重$\mathcal{W}=\mathcal{W}_{r}+j \mathcal{W}_{i}$，实验在Traffic和Electricity数据集上进行。结果如图7和图8所示。这些图表明，权重系数的实部或虚部表现出能量聚集特征（清晰的对角线模式），这有助于频域MLP学习显著特征。

- 图7：在Traffic数据集上频率时间学习器中权重$\mathcal{W}$的可视化。'I/O'表示回看窗口大小/预测长度。$\mathcal{W}_{r}$和$\mathcal{W}_{i}$分别是$\mathcal{W}$的实部和虚部。

- 图8：在Electricity数据集上频率时间学习器中权重$\mathcal{W}$的可视化。'I/O'表示回看窗口大小/预测长度。$\mathcal{W}_{r}$和$\mathcal{W}_{i}$分别是$\mathcal{W}$的实部和虚部。

## G.2 全局视角的权重可视化

为了验证频域学习中全局视角的特性，我们在Traffic和Electricity数据集上进行了额外实验，并比较了在时域输入上学习到的权重与在输入频谱上学习到的权重。结果如图9和图10所示。图的左侧显示了在时域输入上学习到的权重，而右侧显示了在输入频谱实部上学习到的权重。从图中可以看出，与时域相比，在输入频谱上学习到的模式表现出更明显的周期性特征。这归因于频域的全局视角特性。此外，我们可视化了FreTS在Traffic和Electricity数据集上的预测结果，如图11和图12所示，这些图表明FreTS在拟合周期性模式方面表现出良好的能力。总的来说，这些结果证明了FreTS在捕捉全局周期性模式方面具有强大的能力，这得益于频域的全局视角特性。

- 图9：在Traffic数据集上权重$(L \times \tau)$的可视化，回看窗口大小为72，预测长度为336。

- 图10：在Electricity数据集上权重$(L \times \tau)$的可视化，回看窗口大小为96，预测长度为96。

---

- 图11：在Traffic数据集上预测结果（预测值 vs. 实际值）的可视化。'I/O'表示回看窗口大小/预测长度。

- 图12：在Electricity数据集上预测结果（预测值 vs. 实际值）的可视化。'I/O'表示回看窗口大小/预测长度。
