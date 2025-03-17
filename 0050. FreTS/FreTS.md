# Frequency-domain MLPs are More Effective Learners in Time Series Forecasting

# 频域多层感知器在时间序列预测中是更有效的学习者

Kun ${\mathrm{{Yi}}}^{1}$ ,Qi Zhang ${}^{2}$ ,Wei Fan ${}^{3}$ ,Shoujin Wang ${}^{4}$ ,Pengyang Wang ${}^{5}$ ,Hui He ${}^{1}$ Defu Lian ${}^{6}$ , Ning An ${}^{7}$ , Longbing Cao ${}^{8}$ , Zhendong Niu ${}^{1 * }$

Kun ${\mathrm{{Yi}}}^{1}$ ，张琦（Qi Zhang） ${}^{2}$ ，范伟（Wei Fan） ${}^{3}$ ，王守进（Shoujin Wang） ${}^{4}$ ，王鹏阳（Pengyang Wang） ${}^{5}$ ，何辉（Hui He） ${}^{1}$ ，连德富（Defu Lian） ${}^{6}$ ，安宁（Ning An） ${}^{7}$ ，曹龙兵（Longbing Cao） ${}^{8}$ ，牛振东（Zhendong Niu） ${}^{1 * }$

${}^{1}$ Beijing Institute of Technology, ${}^{2}$ Tongji University, ${}^{3}$ University of Oxford ${}^{4}$ University of Technology Sydney, ${}^{5}$ University of Macau, ${}^{6}$ USTC ${}^{7}$ HeFei University of Technology, ${}^{8}$ Macquarie University \{ yikun, hehui617, zniu \} @bit.edu.cn, zhangqi_cs@tongji.edu.cn, weifan.oxford@gmail.com pywang@um.edu.mo, liandefu@ustc.edu.cn, ning.g.an@acm.org, longbing.cao@mq.edu.au

${}^{1}$ 北京理工大学（Beijing Institute of Technology），${}^{2}$ 同济大学（Tongji University），${}^{3}$ 牛津大学（University of Oxford） ${}^{4}$ 悉尼科技大学（University of Technology Sydney），${}^{5}$ 澳门大学（University of Macau），${}^{6}$ 中国科学技术大学（USTC） ${}^{7}$ 合肥工业大学（HeFei University of Technology），${}^{8}$ 麦考瑞大学（Macquarie University） \{ yikun, hehui617, zniu \} @bit.edu.cn, zhangqi_cs@tongji.edu.cn, weifan.oxford@gmail.com pywang@um.edu.mo, liandefu@ustc.edu.cn, ning.g.an@acm.org, longbing.cao@mq.edu.au

## Abstract

## 摘要

Time series forecasting has played the key role in different industrial, including finance, traffic, energy, and healthcare domains. While existing literatures have designed many sophisticated architectures based on RNNs, GNNs, or Transformers, another kind of approaches based on multi-layer perceptrons (MLPs) are proposed with simple structure, low complexity, and superior performance. However, most MLP-based forecasting methods suffer from the point-wise mappings and information bottleneck, which largely hinders the forecasting performance. To overcome this problem, we explore a novel direction of applying MLPs in the frequency domain for time series forecasting. We investigate the learned patterns of frequency-domain MLPs and discover their two inherent characteristic benefiting forecasting, (i) global view: frequency spectrum makes MLPs own a complete view for signals and learn global dependencies more easily, and (ii) energy compaction: frequency-domain MLPs concentrate on smaller key part of frequency components with compact signal energy. Then, we propose FreTS, a simple yet effective architecture built upon Frequency-domain MLPs for Time Series forecasting. FreTS mainly involves two stages, (i) Domain Conversion, that transforms time-domain signals into complex numbers of frequency domain; (ii) Frequency Learning, that performs our redesigned MLPs for the learning of real and imaginary part of frequency components. The above stages operated on both inter-series and intra-series scales further contribute to channel-wise and time-wise dependency learning. Extensive experiments on 13 real-world benchmarks (including 7 benchmarks for short-term forecasting and 6 benchmarks for long-term forecasting) demonstrate our consistent superiority over state-of-the-art methods. Code is available at this repository: https://github.com/aikunyi/FreTS

时间序列预测在不同行业中发挥着关键作用，包括金融、交通、能源和医疗保健领域。虽然现有文献已经基于循环神经网络（RNNs）、图神经网络（GNNs）或Transformer设计了许多复杂的架构，但另一类基于多层感知机（MLPs）的方法也被提出，它们结构简单、复杂度低且性能优越。然而，大多数基于MLP的预测方法存在逐点映射和信息瓶颈的问题，这在很大程度上阻碍了预测性能。为了克服这个问题，我们探索了一种在频域应用MLP进行时间序列预测的新方向。我们研究了频域MLP学习到的模式，并发现了它们有利于预测的两个固有特性：（i）全局视角：频谱使MLP对信号有完整的视图，更容易学习全局依赖关系；（ii）能量集中：频域MLP专注于具有紧凑信号能量的频率分量的较小关键部分。然后，我们提出了FreTS，这是一种基于频域MLP的简单而有效的时间序列预测架构。FreTS主要包括两个阶段：（i）域转换，将时域信号转换为频域复数；（ii）频率学习，对频率分量的实部和虚部执行我们重新设计的MLP。上述在序列间和序列内尺度上操作的阶段进一步有助于通道和时间依赖关系的学习。在13个真实世界基准数据集（包括7个短期预测基准和6个长期预测基准）上的大量实验表明，我们的方法始终优于现有最先进的方法。代码可在以下仓库获取：https://github.com/aikunyi/FreTS

## 1 Introduction

## 1 引言

Time series forecasting has been a critical role in a variety of real-world industries, such as climate condition estimation [1, 2], traffic state prediction [3, 4], economic analysis [5, 6], etc. In the early stage, many traditional statistical forecasting methods have been proposed, such as exponential smoothing [7] and auto-regressive moving averages (ARMA) [8]. Recently, the emerging development of deep learning has fostered many deep forecasting models, including Recurrent Neural Network-based methods (e.g., DeepAR [9], LSTNet [10]), Convolution Neural Network-based methods (e.g., TCN [11], SCINet [12]), Transformer-based methods (e.g., Informer [13], Autoformer [14]), and Graph Neural Network-based methods (e.g., MTGNN [15], StemGNN [16], AGCRN [17]), etc.

时间序列预测在各种现实世界的行业中一直发挥着关键作用，例如气候条件估计[1, 2]、交通状态预测[3, 4]、经济分析[5, 6]等。早期，人们提出了许多传统的统计预测方法，如指数平滑法[7]和自回归移动平均模型（ARMA）[8]。最近，深度学习的新兴发展催生了许多深度预测模型，包括基于循环神经网络的方法（如DeepAR[9]、LSTNet[10]）、基于卷积神经网络的方法（如TCN[11]、SCINet[12]）、基于Transformer的方法（如Informer[13]、Autoformer[14]）以及基于图神经网络的方法（如MTGNN[15]、StemGNN[16]、AGCRN[17]）等。

---

<!-- Footnote -->

*Corresponding author

*通信作者

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: (a) Left: time domain. Right: frequency domain. (b) Left: time domain. Right: frequency domain -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_1.jpg?x=316&y=215&w=1149&h=282&r=0"/>

Figure 1: Visualizations of the learned patterns of MLPs in the time domain and the frequency domain (see Appendix B.4). (a) global view: the patterns learned in the frequency domain exhibits more obvious global periodic patterns than the time domain; (b) energy compaction: learning in the frequency domain can identify clearer diagonal dependencies and key patterns than the time domain.

图1：多层感知器（MLP）在时域和频域中学习到的模式可视化（见附录B.4）。（a）全局视图：在频域中学习到的模式比在时域中表现出更明显的全局周期性模式；（b）能量集中：在频域中学习比在时域中能够识别出更清晰的对角依赖关系和关键模式。

<!-- Media -->

While these deep models have achieved promising forecasting performance in certain scenarios, their sophisticated network architectures would usually bring up expensive computation burden in training or inference stage. Besides, the robustness of these models could be easily influenced with a large amount of parameters, especially when the available training data is limited [13, 18]. Therefore, the methods based on multi-layer perceptrons (MLPs) have been recently introduced with simple structure,low complexity,and superior forecasting performance,such as N-BEATS [19], LightTS [20], DLinear [21], etc. However, these MLP-based methods rely on point-wise mappings to capture temporal mappings, which cannot handle global dependencies of time series. Moreover, they would suffer from the information bottleneck with regard to the volatile and redundant local momenta of time series, which largely hinders their performance for time series forecasting.

虽然这些深度模型在某些场景中取得了不错的预测性能，但它们复杂的网络架构通常会在训练或推理阶段带来高昂的计算负担。此外，这些模型的鲁棒性很容易受到大量参数的影响，尤其是在可用训练数据有限的情况下 [13, 18]。因此，近年来引入了基于多层感知器（MLP）的方法，这些方法结构简单、复杂度低且具有出色的预测性能，如N - BEATS [19]、LightTS [20]、DLinear [21]等。然而，这些基于MLP的方法依赖逐点映射来捕捉时间映射，无法处理时间序列的全局依赖关系。此外，它们会受到时间序列波动且冗余的局部矩信息瓶颈的影响，这在很大程度上阻碍了它们在时间序列预测中的性能。

To overcome the above problems, we explore a novel direction of applying MLPs in the frequency domain for time series forecasting. We investigate the learned patterns of frequency-domain MLPs in forecasting and have discovered their two key advantages: (i) global view: operating on spectral components acquired from series transformation, frequency-domain MLPs can capture a more complete view of signals, making it easier to learn global spatial/temporal dependencies. (ii) energy compaction: frequency-domain MLPs concentrate on the smaller key part of frequency components with the compact signal energy, and thus can facilitate preserving clearer patterns while filtering out influence of noises. Experimentally, we have observed that frequency-domain MLPs capture much more obvious global periodic patterns than the time-domain MLPs from Figure 1(a), which highlights their ability to recognize global signals. Also, from Figure 1(b), we easily note a much more clear diagonal dependency in the learned weights of frequency-domain MLPs, compared with the more scattered dependency learned by time-domain MLPs. This illustrates the great potential of frequency-domain MLPs to identify most important features and key patterns while handling complicated and noisy information.

为克服上述问题，我们探索了在频域应用多层感知器（MLPs）进行时间序列预测的新方向。我们研究了频域多层感知器在预测中学习到的模式，并发现了它们的两个关键优势：（i）全局视角：频域多层感知器基于序列变换得到的频谱分量进行操作，能够捕捉到更完整的信号视图，从而更易于学习全局空间/时间依赖关系。（ii）能量集中：频域多层感知器专注于具有紧凑信号能量的较小频率分量关键部分，因此能够在过滤噪声影响的同时保留更清晰的模式。通过实验，从图1（a）中我们观察到，频域多层感知器比时域多层感知器捕捉到的全局周期性模式要明显得多，这凸显了它们识别全局信号的能力。此外，从图1（b）中我们可以轻易注意到，与时域多层感知器学习到的更分散的依赖关系相比，频域多层感知器学习到的权重具有更清晰的对角依赖关系。这表明频域多层感知器在处理复杂且含噪信息时，在识别最重要特征和关键模式方面具有巨大潜力。

To fully utilize these advantages, we propose FreTS, a simple yet effective architecture of Frequency-domain MLPs for Time Series forecasting. The core idea of FreTS is to learn the time series forecasting mappings in the frequency domain. Specifically, FreTS mainly involves two stages: (i) Domain Conversion: the original time-domain series signals are first transformed into frequency-domain spectrum on top of Discrete Fourier Transform (DFT) [22], where the spectrum is composed of several complex numbers as frequency components, including the real coefficients and the imaginary coefficients. (ii) Frequency Learning: given the real/imaginary coefficients, we redesign the frequency-domain MLPs originally for the complex numbers by separately considering the real mappings and imaginary mappings. The respective real/imaginary parts of output learned by two distinct MLPs are then stacked in order to recover from frequency components to the final forecasting. Also, FreTS performs above two stages on both inter-series and intra-series scales, which further contributes to the channel-wise and time-wise dependencies in the frequency domain for better forecasting performance. We conduct extensive experiments on 13 benchmarks under different settings, covering 7 benchmarks for short-term forecasting and 6 benchmarks for long-term forecasting, which demonstrate our consistent superiority compared with state-of-the-art methods.

为了充分利用这些优势，我们提出了FreTS（频域多层感知器时间序列预测架构，Frequency-domain MLPs for Time Series），这是一种简单而有效的用于时间序列预测的频域多层感知器（MLPs）架构。FreTS的核心思想是在频域中学习时间序列预测映射。具体而言，FreTS主要包括两个阶段：（i）域转换：首先基于离散傅里叶变换（DFT）[22]将原始时域序列信号转换为频域频谱，其中频谱由几个作为频率分量的复数组成，包括实系数和虚系数。（ii）频率学习：给定实/虚系数，我们通过分别考虑实映射和虚映射，重新设计了最初用于复数的频域多层感知器。然后将由两个不同的多层感知器学习到的输出的实/虚部分分别堆叠起来，以便从频率分量恢复到最终预测结果。此外，FreTS在序列间和序列内尺度上都执行上述两个阶段，这进一步增强了频域中的通道依赖和时间依赖，从而获得更好的预测性能。我们在不同设置下的13个基准数据集上进行了广泛的实验，其中包括7个短期预测基准数据集和6个长期预测基准数据集，结果表明与现有最先进的方法相比，我们的方法具有持续的优越性。

## 2 Related Work

## 2 相关工作

Forecasting in the Time Domain Traditionally, statistical methods have been proposed for forecasting in the time domain, including (ARMA) [8], VAR [23], and ARIMA [24]. Recently, deep learning based methods have been widely used in time series forecasting due to their capability of extracting nonlinear and complex correlations [25, 26]. These methods have learned the dependencies in the time domain with RNNs (e.g., deepAR [9], LSTNet [10]) and CNNs (e.g., TCN [11], SCINet [12]). In addition, GNN-based models have been proposed with good forecasting performance because of their good abilities to model series-wise dependencies among variables in the time domain, such as TAMP-S2GCNets [4], AGCRN [17], MTGNN [15], and GraphWaveNet [27]. Besides, Transformer-based forecasting methods have been introduced due to their attention mechanisms for long-range dependency modeling ability in the time domain, such as Reformer [18] and Informer [13].

时域预测 传统上，已经提出了用于时域预测的统计方法，包括自回归滑动平均模型（ARMA）[8]、向量自回归模型（VAR）[23]和自回归积分滑动平均模型（ARIMA）[24]。最近，基于深度学习的方法因其能够提取非线性和复杂的相关性而被广泛应用于时间序列预测[25, 26]。这些方法使用循环神经网络（RNNs，例如深度自回归模型（deepAR）[9]、长短期记忆网络（LSTNet）[10]）和卷积神经网络（CNNs，例如时间卷积网络（TCN）[11]、自相关网络（SCINet）[12]）学习了时域中的依赖关系。此外，由于基于图神经网络（GNN）的模型在时域中对变量之间的序列依赖关系进行建模的能力较强，因此已经提出了具有良好预测性能的基于图神经网络的模型，如时间感知多尺度图卷积网络（TAMP - S2GCNets）[4]、自适应图卷积循环网络（AGCRN）[17]、多尺度图神经网络（MTGNN）[15]和图波网络（GraphWaveNet）[27]。此外，由于基于Transformer的预测方法在时域中具有用于长距离依赖建模能力的注意力机制，因此已经引入了此类方法，如改革者模型（Reformer）[18]和告密者模型（Informer）[13]。

Forecasting in the Frequency Domain Several recent time series forecasting methods have extracted knowledge of the frequency domain for forecasting [28]. Specifically, SFM [29] decomposes the hidden state of LSTM into frequencies by Discrete Fourier Transform (DFT). StemGNN [16] performs graph convolutions based on Graph Fourier Transform (GFT) and computes series correlations based on Discrete Fourier Transform. Autoformer [14] replaces self-attention by proposing the auto-correlation mechanism implemented with Fast Fourier Transforms (FFT). FEDformer [30] proposes a DFT-based frequency enhanced attention, which obtains the attentive weights by the spectrums of queries and keys, and calculates the weighted sum in the frequency domain. CoST [31] uses DFT to map the intermediate features to frequency domain to enables interactions in representation. FiLM [32] utilizes Fourier analysis to preserve historical information and remove noisy signals. Unlike these efforts that leverage frequency techniques to improve upon the original architecture such as Transformer and GNN, in this paper, we propose a new frequency learning architecture that learns both channel-wise and time-wise dependencies in the frequency domain.

频域预测 近期有几种时间序列预测方法提取了频域知识用于预测[28]。具体而言，频谱特征模型（SFM）[29]通过离散傅里叶变换（DFT）将长短期记忆网络（LSTM）的隐藏状态分解为不同频率。图卷积神经网络（StemGNN）[16]基于图傅里叶变换（GFT）进行图卷积，并基于离散傅里叶变换计算序列相关性。自相关变换器（Autoformer）[14]通过提出基于快速傅里叶变换（FFT）实现的自相关机制来替代自注意力机制。频域增强变换器（FEDformer）[30]提出了一种基于离散傅里叶变换的频域增强注意力机制，该机制通过查询和键的频谱获得注意力权重，并在频域中计算加权和。跨尺度时间序列预测模型（CoST）[31]使用离散傅里叶变换将中间特征映射到频域，以实现表示中的交互。基于傅里叶变换的线性调制（FiLM）[32]利用傅里叶分析来保留历史信息并去除噪声信号。与这些利用频域技术改进如变换器（Transformer）和图神经网络（GNN）等原始架构的工作不同，在本文中，我们提出了一种新的频域学习架构，该架构在频域中学习通道维度和时间维度的依赖关系。

MLP-based Forecasting Models Several studies have explored the use of MLP-based networks in time series forecasting. N-BEATS [19] utilizes stacked MLP layers together with doubly residual learning to process the input data to iteratively forecast the future. DEPTS [33] applies Fourier transform to extract periods and MLPs for periodicity dependencies for univariate forecasting. LightTS [20] uses lightweight sampling-oriented MLP structures to reduce complexity and computation time while maintaining accuracy. N-HiTS [34] combines multi-rate input sampling and hierarchical interpolation with MLPs for univariate forecasting. LTSF-Linear [35] proposes a set of embarrassingly simple one-layer linear model to learn temporal relationships between input and output sequences. These studies demonstrate the effectiveness of MLP-based networks in time series forecasting tasks, and inspire the development of our frequency-domain MLPs in this paper.

基于多层感知器（MLP）的预测模型 多项研究探索了基于多层感知器（Multilayer Perceptron，MLP）的网络在时间序列预测中的应用。N - BEATS [19]将堆叠的MLP层与双重残差学习相结合，对输入数据进行处理，以迭代方式预测未来。DEPTS [33]应用傅里叶变换（Fourier transform）提取周期，并使用MLP处理单变量预测中的周期性依赖关系。LightTS [20]使用轻量级的面向采样的MLP结构，在保持准确性的同时降低复杂度和计算时间。N - HiTS [34]将多速率输入采样和分层插值与MLP相结合，用于单变量预测。LTSF - Linear [35]提出了一组极其简单的单层线性模型，用于学习输入和输出序列之间的时间关系。这些研究证明了基于MLP的网络在时间序列预测任务中的有效性，并启发了本文频域MLP的发展。

## 3 FreTS

## 3 频率时间序列模型（FreTS）

In this section, we elaborate on our proposed novel approach, FreTS, based on our redesigned MLPs in the frequency domain for time series forecasting. First, we present the detailed frequency learning architecture of FreTS in Section 3.1, which mainly includes two-fold frequency learners with domain conversions. Then, we detailedly introduce our redesigned frequency-domain MLPs adopted by above frequency learners in Section 3.2. Besides, we also theoretically analyze their superior nature of global view and energy compaction, as aforementioned in Section 1.

在本节中，我们详细阐述我们提出的新颖方法FreTS（频域时间序列预测方法），该方法基于我们在频域重新设计的多层感知器（MLPs）进行时间序列预测。首先，我们在3.1节介绍FreTS的详细频率学习架构，其主要包括带有域转换的双重频率学习器。然后，我们在3.2节详细介绍上述频率学习器采用的我们重新设计的频域多层感知器。此外，正如第1节所述，我们还从理论上分析了它们具有全局视野和能量压缩的优越特性。

Problem Definition Let $\left\lbrack  {{X}_{1},{X}_{2},\cdots ,{X}_{T}}\right\rbrack   \in  {\mathbb{R}}^{N \times  T}$ stand for the regularly sampled multi-variate time series dataset with $N$ series and $T$ timestamps,where ${X}_{t} \in  {\mathbb{R}}^{N}$ denotes the multi-variate values of $N$ distinct series at timestamp $t$ . We consider a time series lookback window of length- $L$ at timestamp $t$ as the model input,namely ${\mathbf{X}}_{t} = \left\lbrack  {{X}_{t - L + 1},{X}_{t - L + 2},\cdots ,{X}_{t}}\right\rbrack   \in  {\mathbb{R}}^{N \times  L}$ ; also,we consider a horizon window of length- $\tau$ at timestamp $t$ as the prediction target,denoted as ${\mathbf{Y}}_{t} =$ $\left\lbrack  {{X}_{t + 1},{X}_{t + 2},\cdots ,{X}_{t + \tau }}\right\rbrack   \in  {\mathbb{R}}^{N \times  \tau }$ . Then the time series forecasting formulation is to use historical observations ${\mathbf{X}}_{t}$ to predict future values ${\widehat{\mathbf{Y}}}_{t}$ and the typical forecasting model ${f}_{\theta }$ parameterized by $\theta$ is to produce forecasting results by ${\widehat{\mathbf{Y}}}_{t} = {f}_{\theta }\left( {\mathbf{X}}_{t}\right)$ .

问题定义 设 $\left\lbrack  {{X}_{1},{X}_{2},\cdots ,{X}_{T}}\right\rbrack   \in  {\mathbb{R}}^{N \times  T}$ 表示具有 $N$ 个序列和 $T$ 个时间戳的规则采样多元时间序列数据集，其中 ${X}_{t} \in  {\mathbb{R}}^{N}$ 表示在时间戳 $t$ 处 $N$ 个不同序列的多元值。我们将时间戳 $t$ 处长度为 $L$ 的时间序列回溯窗口作为模型输入，即 ${\mathbf{X}}_{t} = \left\lbrack  {{X}_{t - L + 1},{X}_{t - L + 2},\cdots ,{X}_{t}}\right\rbrack   \in  {\mathbb{R}}^{N \times  L}$；此外，我们将时间戳 $t$ 处长度为 $\tau$ 的预测范围窗口作为预测目标，记为 ${\mathbf{Y}}_{t} =$ $\left\lbrack  {{X}_{t + 1},{X}_{t + 2},\cdots ,{X}_{t + \tau }}\right\rbrack   \in  {\mathbb{R}}^{N \times  \tau }$。那么，时间序列预测公式就是使用历史观测值 ${\mathbf{X}}_{t}$ 来预测未来值 ${\widehat{\mathbf{Y}}}_{t}$，并且由 $\theta$ 参数化的典型预测模型 ${f}_{\theta }$ 通过 ${\widehat{\mathbf{Y}}}_{t} = {f}_{\theta }\left( {\mathbf{X}}_{t}\right)$ 产生预测结果。

### 3.1 Frequency Learning Architecture

### 3.1 频率学习架构

The frequency learning architecture of FreTS is depicted in Figure 2, which mainly involves Domain Conversion/Inversion stages, Frequency-domain MLPs, and the corresponding two learners, i.e., the Frequency Channel Learner and the Frequency Temporal Learner. Besides, before taken to learners, we concretely apply a dimension extension block on model input to enhance the model capability. Specifically,the input lookback window ${\mathbf{X}}_{t} \in  {\mathbb{R}}^{N \times  L}$ is multiplied with a learnable weight vector ${\phi }_{d} \in  {\mathbb{R}}^{1 \times  d}$ to obtain a more expressive hidden representation ${\mathbf{H}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ ,yielding ${\mathbf{H}}_{t} = {\mathbf{X}}_{t} \times  {\phi }_{d}$ to bring more semantic information,inspired by word embeddings [36].

FreTS的频率学习架构如图2所示，主要包括域转换/反转阶段、频域多层感知机（MLPs）以及相应的两个学习器，即频率通道学习器和频率时间学习器。此外，在将输入送入学习器之前，我们具体在模型输入上应用了一个维度扩展块来增强模型能力。具体而言，受词嵌入 [36] 的启发，将输入回望窗口 ${\mathbf{X}}_{t} \in  {\mathbb{R}}^{N \times  L}$ 与一个可学习的权重向量 ${\phi }_{d} \in  {\mathbb{R}}^{1 \times  d}$ 相乘，以获得更具表现力的隐藏表示 ${\mathbf{H}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ ，从而得到 ${\mathbf{H}}_{t} = {\mathbf{X}}_{t} \times  {\phi }_{d}$ 以带来更多语义信息。

<!-- Media -->

<!-- figureText: $\mathcal{{HW}} + \mathcal{B}$ -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_3.jpg?x=308&y=201&w=1174&h=432&r=0"/>

Figure 2: The framework overview of FreTS: the Frequency Channel Learner focuses on modeling inter-series dependencies with frequency-domain MLPs operating on the channel dimensions; the Frequency Temporal Learner is to capture the temporal dependencies by performing frequency-domain MLPs on the time dimensions.

图2：FreTS（频率时间序列学习框架，Frequency Time Series learning framework）的框架概述：频率通道学习器（Frequency Channel Learner）专注于通过在通道维度上运行频域多层感知器（MLPs）来对序列间的依赖关系进行建模；频率时间学习器（Frequency Temporal Learner）则是通过在时间维度上执行频域多层感知器来捕捉时间依赖关系。

<!-- Media -->

Domain Conversion/Inversion The use of Fourier transform enables the decomposition of a time series signal into its constituent frequencies. This is particularly advantageous for time series analysis since it benefits to identify periodic or trend patterns in the data, which are often important in forecasting tasks. As aforementioned in Figure 1(a), learning in the frequency spectrum helps capture a greater number of periodic patterns. In view of this,we convert the input $\mathbf{H}$ into the frequency domain $\mathcal{H}$ by:

域转换/反转 傅里叶变换的使用能够将时间序列信号分解为其组成频率。这对于时间序列分析特别有利，因为它有助于识别数据中的周期性或趋势模式，而这些模式在预测任务中通常很重要。如前文图1(a)所示，在频谱中进行学习有助于捕捉更多的周期性模式。鉴于此，我们通过以下方式将输入$\mathbf{H}$转换到频域$\mathcal{H}$：

$$
\mathcal{H}\left( f\right)  = {\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) {e}^{-{j2\pi fv}}\mathrm{\;d}v = {\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) \cos \left( {2\pi fv}\right) \mathrm{d}v + j{\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) \sin \left( {2\pi fv}\right) \mathrm{d}v \tag{1}
$$

where $f$ is the frequency variable, $v$ is the integral variable,and $j$ is the imaginary unit,which is defined as the square root of -1 ; ${\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) \cos \left( {2\pi fv}\right) \mathrm{d}v$ is the real part of $\mathcal{H}$ and is abbreviated as $\operatorname{Re}\left( \mathcal{H}\right) ;{\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) \sin \left( {2\pi fv}\right) \mathrm{d}v$ is the imaginary part and is abbreviated as $\operatorname{Im}\left( \mathcal{H}\right)$ . Then we can rewrite $\mathcal{H}$ in Equation (1) as: $\mathcal{H} = \operatorname{Re}\left( \mathcal{H}\right)  + j\operatorname{Im}\left( \mathcal{H}\right)$ . Note that in FreTS we operate domain conversion on both the channel dimension and time dimension, respectively. Once completing the learning in the frequency domain,we can convert $\mathcal{H}$ back into the the time domain using the following inverse conversion formulation:

其中 $f$ 是频率变量，$v$ 是积分变量，$j$ 是虚数单位，定义为 -1 的平方根；${\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) \cos \left( {2\pi fv}\right) \mathrm{d}v$ 是 $\mathcal{H}$ 的实部，缩写为 $\operatorname{Re}\left( \mathcal{H}\right) ;{\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) \sin \left( {2\pi fv}\right) \mathrm{d}v$，$\operatorname{Im}\left( \mathcal{H}\right)$ 是虚部。然后我们可以将方程 (1) 中的 $\mathcal{H}$ 重写为：$\mathcal{H} = \operatorname{Re}\left( \mathcal{H}\right)  + j\operatorname{Im}\left( \mathcal{H}\right)$。请注意，在频域时间序列（FreTS）中，我们分别对通道维度和时间维度进行域转换。一旦完成频域中的学习，我们可以使用以下逆转换公式将 $\mathcal{H}$ 转换回时域：

$$
\mathbf{H}\left( v\right)  = {\int }_{-\infty }^{\infty }\mathcal{H}\left( f\right) {e}^{j2\pi fv}\mathrm{\;d}f = {\int }_{-\infty }^{\infty }(\operatorname{Re}\left( {\mathcal{H}\left( f\right) }\right)  + j\operatorname{Im}\left( {\mathcal{H}\left( f\right) }\right) {e}^{j2\pi fv}\mathrm{\;d}f \tag{2}
$$

where we take frequency $f$ as the integral variable. In fact,the frequency spectrum is expressed as a combination of cos and sin waves in $\mathcal{H}$ with different frequencies and amplitudes inferring different periodic properties in time series signals. Thus examining the frequency spectrum can better discern the prominent frequencies and periodic patterns in time series. In the following sections, we use DomainConversion to stand for Equation (1), and DomainInversion for Equation (2) for brevity.

我们将频率 $f$ 作为积分变量。实际上，频谱在 $\mathcal{H}$ 中表示为不同频率和振幅的余弦波和正弦波的组合，这意味着时间序列信号具有不同的周期性特征。因此，检查频谱可以更好地识别时间序列中的显著频率和周期性模式。在接下来的章节中，为简洁起见，我们用“域转换”（DomainConversion）表示方程 (1)，用“域反演”（DomainInversion）表示方程 (2)。

Frequency Channel Learner Considering channel dependencies for time series forecasting is important because it allows the model to capture interactions and correlations between different variables, leading to a more accurate predictions. The frequency channel learner enables communications between different channels; it operates on each timestamp by sharing the same weights between $L$ timestamps to learn channel dependencies. Concretely,the frequency channel learner takes ${\mathbf{H}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ as input. Given the $l$ -th timestamp ${\mathbf{H}}_{t}^{ : ,\left( l\right) } \in  {\mathbb{R}}^{N \times  d}$ ,we perform the frequency channel learner by:

频率通道学习器 在时间序列预测中考虑通道依赖关系非常重要，因为这能让模型捕捉不同变量之间的相互作用和相关性，从而实现更准确的预测。频率通道学习器可实现不同通道之间的通信；它在每个时间戳上进行操作，通过在 $L$ 个时间戳之间共享相同的权重来学习通道依赖关系。具体而言，频率通道学习器以 ${\mathbf{H}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ 作为输入。给定第 $l$ 个时间戳 ${\mathbf{H}}_{t}^{ : ,\left( l\right) } \in  {\mathbb{R}}^{N \times  d}$ ，我们通过以下方式执行频率通道学习器：

$$
{\mathcal{H}}_{\text{chan }}^{ : ,\left( l\right) } = {\operatorname{DomainConversion}}_{\left( \text{chan }\right) }\left( {\mathbf{H}}_{t}^{ : ,\left( l\right) }\right) 
$$

$$
{\mathcal{Z}}_{\text{chan }}^{ : ,\left( l\right) } = \operatorname{FreMLP}\left( {{\mathcal{H}}_{\text{chan }}^{ : ,\left( l\right) },{\mathcal{W}}^{\text{chan }},{\mathcal{B}}^{\text{chan }}}\right)  \tag{3}
$$

$$
{\mathbf{Z}}^{ : ,\left( l\right) } = {\operatorname{DomainInversion}}_{\left( \text{chan }\right) }\left( {\mathcal{Z}}_{\text{chan }}^{ : ,\left( l\right) }\right) 
$$

where ${\mathcal{H}}_{\text{chan }}^{ : ,\left( l\right) } \in  {\mathbb{C}}^{\frac{N}{2} \times  d}$ is the frequency components of ${\mathbf{H}}_{t}^{ : ,\left( l\right) }$ ; DomainConversion ${}_{\left( \text{chan }\right) }$ and DomainInversion ${}_{\left( chan\right) }$ indicates such operations are performed along the channel dimension. FreMLP are frequency-domain MLPs proposed in Section 3.2,which takes ${\mathcal{W}}^{\text{chan }} = \left( {\mathcal{W}}_{r}^{\text{chan }}\right.  +$ $\left. {j{\mathcal{W}}_{i}^{\text{chan }}}\right)  \in  {\mathbb{C}}^{d \times  d}$ as the complex number weight matrix with ${\mathcal{W}}_{r}^{\text{chan }} \in  {\mathbb{R}}^{d \times  d}$ and ${\mathcal{W}}_{i}^{\text{chan }} \in  {\mathbb{R}}^{d \times  d}$ , and ${\mathcal{B}}^{\text{chan }} = \left( {{\mathcal{B}}_{r}^{\text{chan }} + j{\mathcal{B}}_{i}^{\text{chan }}}\right)  \in  {\mathbb{C}}^{d}$ as the biases with ${\mathcal{B}}_{r}^{\text{chan }} \in  {\mathbb{R}}^{d}$ and ${\mathcal{B}}_{i}^{\text{chan }} \in  {\mathbb{R}}^{d}$ . And ${\mathcal{Z}}_{\text{chan }}^{ : ,\left( l\right) } \in  {\mathbb{C}}^{\frac{N}{2} \times  d}$ is the output of FreMLP,also in the frequency domain,which is conversed back to time domain as ${\mathbf{Z}}^{ : ,\left( l\right) } \in  {\mathbb{R}}^{N \times  d}$ . Finally,we ensemble ${\mathbf{Z}}^{ : ,\left( l\right) }$ of $L$ timestamps into a whole and output ${\mathbf{Z}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ .

其中 ${\mathcal{H}}_{\text{chan }}^{ : ,\left( l\right) } \in  {\mathbb{C}}^{\frac{N}{2} \times  d}$ 是 ${\mathbf{H}}_{t}^{ : ,\left( l\right) }$ 的频率分量；域转换 ${}_{\left( \text{chan }\right) }$ 和域反转 ${}_{\left( chan\right) }$ 表示这些操作是沿着通道维度执行的。频域多层感知器（FreMLP）是在3.2节中提出的频域多层感知器，它将 ${\mathcal{W}}^{\text{chan }} = \left( {\mathcal{W}}_{r}^{\text{chan }}\right.  +$ $\left. {j{\mathcal{W}}_{i}^{\text{chan }}}\right)  \in  {\mathbb{C}}^{d \times  d}$ 作为复数权重矩阵，其中 ${\mathcal{W}}_{r}^{\text{chan }} \in  {\mathbb{R}}^{d \times  d}$ 和 ${\mathcal{W}}_{i}^{\text{chan }} \in  {\mathbb{R}}^{d \times  d}$ ，并将 ${\mathcal{B}}^{\text{chan }} = \left( {{\mathcal{B}}_{r}^{\text{chan }} + j{\mathcal{B}}_{i}^{\text{chan }}}\right)  \in  {\mathbb{C}}^{d}$ 作为偏置，其中 ${\mathcal{B}}_{r}^{\text{chan }} \in  {\mathbb{R}}^{d}$ 和 ${\mathcal{B}}_{i}^{\text{chan }} \in  {\mathbb{R}}^{d}$ 。并且 ${\mathcal{Z}}_{\text{chan }}^{ : ,\left( l\right) } \in  {\mathbb{C}}^{\frac{N}{2} \times  d}$ 是频域多层感知器（FreMLP）的输出，同样在频域中，它被转换回时域为 ${\mathbf{Z}}^{ : ,\left( l\right) } \in  {\mathbb{R}}^{N \times  d}$ 。最后，我们将 $L$ 个时间戳的 ${\mathbf{Z}}^{ : ,\left( l\right) }$ 组合成一个整体并输出 ${\mathbf{Z}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ 。

Frequency Temporal Learner The frequency temporal learner aims to learn the temporal patterns in the frequency domain; also, it is constructed based on frequency-domain MLPs conducting on each channel and it shares the weights between $N$ channels. Specifically,it takes the frequency channel learner output ${\mathbf{Z}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ as input and for the $n$ -th channel ${\mathbf{Z}}_{t}^{\left( n\right) , : } \in  {\mathbb{R}}^{L \times  d}$ ,we apply the frequency temporal learner by:

频率时间学习器 频率时间学习器旨在学习频域中的时间模式；此外，它基于对每个通道进行操作的频域多层感知器（MLP）构建，并且在 $N$ 个通道之间共享权重。具体而言，它将频率通道学习器的输出 ${\mathbf{Z}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ 作为输入，对于第 $n$ 个通道 ${\mathbf{Z}}_{t}^{\left( n\right) , : } \in  {\mathbb{R}}^{L \times  d}$，我们通过以下方式应用频率时间学习器：

$$
{\mathcal{Z}}_{\text{temp }}^{\left( n\right) , : } = {\operatorname{DomainConversion}}_{\left( \text{temp }\right) }\left( {\mathbf{Z}}_{t}^{\left( n\right) , : }\right) 
$$

$$
{\mathcal{S}}_{\text{temp }}^{\left( n\right) , : } = \operatorname{FreMLP}\left( {{\mathcal{Z}}_{\text{temp }}^{\left( n\right) , : },{\mathcal{W}}^{\text{temp }},{\mathcal{B}}^{\text{temp }}}\right)  \tag{4}
$$

$$
{\mathbf{S}}^{\left( n\right) , : } = {\operatorname{DomainInversion}}_{\left( \text{ temp }\right) }\left( {\mathcal{S}}_{\text{ temp }}^{\left( n\right) , : }\right) 
$$

where ${\mathcal{Z}}_{\text{temn }}^{\left( n\right) , : } \in  {\mathbb{C}}^{\frac{L}{2} \times  d}$ is the corresponding frequency spectrum of ${\mathbf{Z}}_{t}^{\left( n\right) , : }$ ; DomainConversion(temp) and DomainInversion ${}_{\left( \text{temp }\right) }$ indicates the calculations are applied along the time dimension. ${\mathcal{W}}^{\text{temp }} = \left( {{\mathcal{W}}_{r}^{\text{temp }} + j{\mathcal{W}}_{i}^{\text{temp }}}\right)  \in  {\mathbb{C}}^{d \times  d}$ is the complex number weight matrix with ${\mathcal{W}}_{r}^{\text{temp }} \in  {\mathbb{R}}^{d \times  d}$ and ${\mathcal{W}}_{i}^{\text{temp }} \in  {\mathbb{R}}^{d \times  d}$ ,and ${\mathcal{B}}^{\text{temp }} = \left( {{\mathcal{B}}_{r}^{\text{temp }} + j{\mathcal{B}}_{i}^{\text{temp }}}\right)  \in  {\mathbb{C}}^{d}$ are the complex number biases with ${\mathcal{B}}_{r}^{\text{temp }} \in  {\mathbb{R}}^{d}$ and ${\mathcal{B}}_{i}^{\text{temp }} \in  {\mathbb{R}}^{d}.{\mathcal{S}}_{\text{temp }}^{\left( n\right) , : } \in  {\mathbb{C}}^{\frac{L}{2} \times  d}$ is the output of FreMLP and is converted back to the time domain as ${\mathbf{S}}^{\left( n\right) , : } \in  {\mathbb{R}}^{L \times  d}$ . Finally,we incorporate all channels and output ${\mathbf{S}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ .

其中 ${\mathcal{Z}}_{\text{temn }}^{\left( n\right) , : } \in  {\mathbb{C}}^{\frac{L}{2} \times  d}$ 是 ${\mathbf{Z}}_{t}^{\left( n\right) , : }$ 对应的频谱；DomainConversion(temp) 和 DomainInversion ${}_{\left( \text{temp }\right) }$ 表示这些计算是沿着时间维度进行的。${\mathcal{W}}^{\text{temp }} = \left( {{\mathcal{W}}_{r}^{\text{temp }} + j{\mathcal{W}}_{i}^{\text{temp }}}\right)  \in  {\mathbb{C}}^{d \times  d}$ 是复数权重矩阵，其中 ${\mathcal{W}}_{r}^{\text{temp }} \in  {\mathbb{R}}^{d \times  d}$ 和 ${\mathcal{W}}_{i}^{\text{temp }} \in  {\mathbb{R}}^{d \times  d}$ ，并且 ${\mathcal{B}}^{\text{temp }} = \left( {{\mathcal{B}}_{r}^{\text{temp }} + j{\mathcal{B}}_{i}^{\text{temp }}}\right)  \in  {\mathbb{C}}^{d}$ 是复数偏置，其中 ${\mathcal{B}}_{r}^{\text{temp }} \in  {\mathbb{R}}^{d}$ ，${\mathcal{B}}_{i}^{\text{temp }} \in  {\mathbb{R}}^{d}.{\mathcal{S}}_{\text{temp }}^{\left( n\right) , : } \in  {\mathbb{C}}^{\frac{L}{2} \times  d}$ 是 FreMLP 的输出，并作为 ${\mathbf{S}}^{\left( n\right) , : } \in  {\mathbb{R}}^{L \times  d}$ 转换回时域。最后，我们合并所有通道并输出 ${\mathbf{S}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ 。

Projection Finally, we use the learned channel and temporal dependencies to make predictions for the future $\tau$ timestamps ${\widehat{\mathbf{Y}}}_{t} \in  {\mathbb{R}}^{N \times  \tau }$ by a two-layer feed forward network (FFN) with one forward step which can avoid error accumulation, formulated as follows:

投影 最后，我们利用学习到的通道和时间依赖关系，通过一个具有一步前向传播的两层前馈网络（FFN）对未来 $\tau$ 个时间戳 ${\widehat{\mathbf{Y}}}_{t} \in  {\mathbb{R}}^{N \times  \tau }$ 进行预测，该网络可以避免误差累积，具体公式如下：

$$
{\widehat{\mathbf{Y}}}_{t} = \sigma \left( {{\mathbf{S}}_{t}{\phi }_{1} + {\mathbf{b}}_{1}}\right) {\phi }_{2} + {\mathbf{b}}_{2} \tag{5}
$$

where ${\mathbf{S}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ is the output of the frequency temporal learner, $\sigma$ is the activation function, ${\phi }_{1} \in  {\mathbb{R}}^{\left( {L * d}\right)  \times  {d}_{h}},{\phi }_{2} \in  {\mathbb{R}}^{{d}_{h} \times  \tau }$ are the weights, ${\mathbf{b}}_{1} \in  {\mathbb{R}}^{{d}_{h}},{\mathbf{b}}_{2} \in  {\mathbb{R}}^{\tau }$ are the biases,and ${d}_{h}$ is the inner-layer dimension size.

其中 ${\mathbf{S}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ 是频率时间学习器的输出，$\sigma$ 是激活函数，${\phi }_{1} \in  {\mathbb{R}}^{\left( {L * d}\right)  \times  {d}_{h}},{\phi }_{2} \in  {\mathbb{R}}^{{d}_{h} \times  \tau }$ 是权重，${\mathbf{b}}_{1} \in  {\mathbb{R}}^{{d}_{h}},{\mathbf{b}}_{2} \in  {\mathbb{R}}^{\tau }$ 是偏置，${d}_{h}$ 是内层维度大小。

### 3.2 Frequency-domain MLPs

### 3.2 频域多层感知器（Frequency-domain MLPs）

As shown in Figure 3, we elaborate our novel frequency-domain MLPs in FreTS that are redesigned for the complex numbers of frequency components, in order to effectively capture the time series key patterns with global view and energy compaction, as aforementioned in Section 1,

如图3所示，我们详细阐述了在频域时间序列学习系统（FreTS）中提出的新型频域多层感知器，这些感知器是针对频率分量的复数重新设计的，以便如第1节所述，从全局视角有效捕捉时间序列的关键模式并实现能量压缩。

Definition 1 (Frequency-domain MLPs). Formally,for a complex number input $\mathcal{H} \in  {\mathbb{C}}^{m \times  d}$ , given a complex number weight matrix $\mathcal{W} \in  {\mathbb{C}}^{d \times  d}$ and a complex number bias $\mathcal{B} \in  {\mathbb{C}}^{d}$ ,then the frequency-domain MLPs can be formulated as:

定义1（频域多层感知机）。形式上，对于复数输入 $\mathcal{H} \in  {\mathbb{C}}^{m \times  d}$ ，给定复数权重矩阵 $\mathcal{W} \in  {\mathbb{C}}^{d \times  d}$ 和复数偏置 $\mathcal{B} \in  {\mathbb{C}}^{d}$ ，则频域多层感知机可以表示为：

$$
{\mathcal{Y}}^{\ell } = \sigma \left( {{\mathcal{Y}}^{\ell  - 1}{\mathcal{W}}^{\ell } + {\mathcal{B}}^{\ell }}\right)  \tag{6}
$$

$$
{\mathcal{Y}}^{0} = \mathcal{H}
$$

where ${\mathcal{Y}}^{\ell } \in  {\mathbb{C}}^{m \times  d}$ is the final output, $\ell$ denotes the $\ell$ -th layer,and $\sigma$ is the activation function.

其中 ${\mathcal{Y}}^{\ell } \in  {\mathbb{C}}^{m \times  d}$ 是最终输出， $\ell$ 表示第 $\ell$ 层， $\sigma$ 是激活函数。

As both $\mathcal{H}$ and $\mathcal{W}$ are complex numbers,according to the rule of multiplication of complex numbers (details can be seen in Appendix C), we further extend the Equation (6) to:

由于 $\mathcal{H}$ 和 $\mathcal{W}$ 均为复数，根据复数乘法规则（具体细节见附录C），我们进一步将方程（6）扩展为：

$$
{\mathcal{Y}}^{\ell } = \sigma \left( {\operatorname{Re}\left( {\mathcal{Y}}^{\ell  - 1}\right) {\mathcal{W}}_{r}^{\ell } - \operatorname{Im}\left( {\mathcal{Y}}^{\ell  - 1}\right) {\mathcal{W}}_{i}^{\ell } + {\mathcal{B}}_{r}^{\ell }}\right)  + {j\sigma }\left( {\operatorname{Re}\left( {\mathcal{Y}}^{\ell  - 1}\right) {\mathcal{W}}_{i}^{\ell } + \operatorname{Im}\left( {\mathcal{Y}}^{\ell  - 1}\right) {\mathcal{W}}_{r}^{\ell } + {\mathcal{B}}_{i}^{\ell }}\right)  \tag{7}
$$

where ${\mathcal{W}}^{\ell } = {\mathcal{W}}_{r}^{\ell } + j{\mathcal{W}}_{i}^{\ell }$ and ${\mathcal{B}}^{\ell } = {\mathcal{B}}_{r}^{\ell } + j{\mathcal{B}}_{i}^{\ell }$ . According to the equation,we implement the MLPs in the frequency domain (abbreviated as FreMLP) by the separate computation of the real and imaginary parts of frequency components. Then, we stack them to form a complex number to acquire the final results. The specific implementation process is shown in Figure 3.

其中 ${\mathcal{W}}^{\ell } = {\mathcal{W}}_{r}^{\ell } + j{\mathcal{W}}_{i}^{\ell }$ 和 ${\mathcal{B}}^{\ell } = {\mathcal{B}}_{r}^{\ell } + j{\mathcal{B}}_{i}^{\ell }$ 。根据该方程，我们通过分别计算频率分量的实部和虚部，在频域中实现多层感知器（Multilayer Perceptrons，缩写为 FreMLP）。然后，我们将它们堆叠形成一个复数以获得最终结果。具体实现过程如图 3 所示。

Theorem 1. Suppose that $\mathbf{H}$ is the representation of raw time series and $\mathcal{H}$ is the corresponding frequency components of the spectrum, then the energy of a time series in the time domain is equal to the energy of its representation in the frequency domain. Formally, we can express this with above notations by:

定理 1。假设 $\mathbf{H}$ 是原始时间序列的表示，$\mathcal{H}$ 是其频谱对应的频率分量，那么时间序列在时域中的能量等于其在频域中表示的能量。形式上，我们可以用上述符号表示为：

$$
{\int }_{-\infty }^{\infty }{\left| \mathbf{H}\left( v\right) \right| }^{2}\mathrm{\;d}v = {\int }_{-\infty }^{\infty }{\left| \mathcal{H}\left( f\right) \right| }^{2}\mathrm{\;d}f \tag{8}
$$

where $\mathcal{H}\left( f\right)  = {\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) {e}^{-{j2\pi fv}}\mathrm{\;d}v,v$ is the time/channel dimension, $f$ is the frequency dimension.

其中 $\mathcal{H}\left( f\right)  = {\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) {e}^{-{j2\pi fv}}\mathrm{\;d}v,v$ 是时间/通道维度，$f$ 是频率维度。

We include the proof in Appendix D.1. The theorem implies that if most of the energy of a time series is concentrated in a small number of frequency components, then the time series can be accurately represented using only those components. Accordingly, discarding the others would not significantly affect the signal's energy. As shown in Figure 1(b), in the frequency domain, the energy concentrates on the smaller part of frequency components, thus learning in the frequency spectrum can facilitate preserving clearer patterns.

我们将证明过程放在附录 D.1 中。该定理表明，如果一个时间序列的大部分能量集中在少数几个频率分量上，那么仅使用这些分量就可以准确表示该时间序列。因此，舍弃其他分量不会显著影响信号的能量。如图 1(b) 所示，在频域中，能量集中在频率分量的较小部分，因此在频谱中进行学习有助于保留更清晰的模式。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_5.jpg?x=1100&y=485&w=336&h=363&r=0"/>

Figure 3: One layer of the frequency-domain MLPs.

图 3：频域多层感知器（MLPs）的一层。

<!-- Media -->

Theorem 2. Given the time series input $\mathbf{H}$ and its corresponding frequency domain conversion $\mathcal{H}$ ,the operations of frequency-domain MLP on $\mathcal{H}$ can be represented as global convolutions on $\mathbf{H}$ in the time domain. This can be given by:

定理2。给定时间序列输入$\mathbf{H}$及其对应的频域转换$\mathcal{H}$，频域多层感知器（MLP）对$\mathcal{H}$的操作可以表示为在时域中对$\mathbf{H}$的全局卷积。这可以表示为：

$$
\mathcal{H}\mathcal{W} + \mathcal{B} = \mathcal{F}\left( {\mathbf{H} * W + B}\right)  \tag{9}
$$

where $*$ is a circular convolution, $\mathcal{W}$ and $\mathcal{B}$ are the complex number weight and bias, $W$ and $B$ are the weight and bias in the time domain,and $\mathcal{F}$ is DFT.

其中$*$是循环卷积，$\mathcal{W}$和$\mathcal{B}$分别是复数权重和偏置，$W$和$B$分别是时域中的权重和偏置，$\mathcal{F}$是离散傅里叶变换（DFT）。

The proof is shown in Appendix D.2. Therefore,the operations of FreMLP,i.e., $\mathcal{{HW}} + \mathcal{B}$ ,are equal to the operations $\left( {\mathbf{H} * W + B}\right)$ in the time domain. This implies that the operations of frequency-domain MLPs can be viewed as global convolutions in the time domain.

证明见附录D.2。因此，频域多层感知器（FreMLP）的操作，即$\mathcal{{HW}} + \mathcal{B}$，等同于时域中的操作$\left( {\mathbf{H} * W + B}\right)$。这意味着频域多层感知器的操作可以看作是时域中的全局卷积。

## 4 Experiments

## 4 实验

To evaluate the performance of FreTS, we conduct extensive experiments on thirteen real-world time series benchmarks, covering short-term forecasting and long-term forecasting settings to compare with corresponding state-of-the-art methods.

为了评估FreTS（频域时间序列预测系统）的性能，我们在十三个真实世界的时间序列基准数据集上进行了广泛的实验，涵盖短期预测和长期预测场景，并与相应的最先进方法进行比较。

Datasets Our empirical results are performed on various domains of datasets, including traffic, energy, web, traffic, electrocardiogram, and healthcare, etc. Specifically, for the task of short-term forecasting, we adopt Solar 2, Wiki [37], Traffic [37], Electricity 3, ECG [16], METR-LA [38], and COVID-19 [4] datasets, following previous forecasting literature [16]. For the task of long-term forecasting, we adopt Weather [14], Exchange [10], Traffic [14], Electricity [14], and ETT datasets [13], following previous long time series forecasting works [13, 14, 30, 39]. We preprocess all datasets following [16, 13, 14] and normalize them with the min-max normalization. We split the datasets into training, validation, and test sets by the ratio of 7:2:1 except for the COVID-19 datasets with 6:2:2. More dataset details are in Appendix B.1.

数据集 我们的实证结果是在多个领域的数据集上得出的，包括交通、能源、网络、心电图和医疗保健等。具体而言，对于短期预测任务，我们遵循先前的预测文献[16]，采用了Solar 2、Wiki [37]、Traffic [37]、Electricity 3、ECG [16]、METR - LA [38]和COVID - 19 [4]数据集。对于长期预测任务，我们遵循先前的长时间序列预测研究[13, 14, 30, 39]，采用了Weather [14]、Exchange [10]、Traffic [14]、Electricity [14]和ETT数据集[13]。我们按照文献[16, 13, 14]对所有数据集进行预处理，并使用最小 - 最大归一化方法对其进行归一化。除了COVID - 19数据集按6:2:2的比例划分外，我们将其他数据集按7:2:1的比例划分为训练集、验证集和测试集。更多数据集细节见附录B.1。

Baselines We compare our FreTS with the representative and state-of-the-art models for both short-term and long-term forecasting to evaluate their effectiveness. For short-term forecasting, we compre FreTS against VAR [23], SFM [29], LSTNet [10], TCN [11], GraphWaveNet [27], DeepGLO [37], StemGNN [16], MTGNN [15], and AGCRN [17] for comparison. We also include TAMP-S2GCNets [4], DCRNN [38] and STGCN [40], which require pre-defined graph structures, for comparison. For long-term forecasting, we include Informer [13], Autoformer [14], Reformer [18], FEDformer [30], LTSF-Linear [35], and the more recent PatchTST [39] for comparison. Additional details about the baselines can be found in Appendix B.2

基线模型 我们将我们的FreTS与用于短期和长期预测的代表性和最先进的模型进行比较，以评估它们的有效性。对于短期预测，我们将FreTS与向量自回归模型（VAR）[23]、结构因子模型（SFM）[29]、长短期记忆网络（LSTNet）[10]、时间卷积网络（TCN）[11]、图波网络（GraphWaveNet）[27]、深度全局预测模型（DeepGLO）[37]、茎图神经网络（StemGNN）[16]、多尺度时空图神经网络（MTGNN）[15]和自适应图卷积循环网络（AGCRN）[17]进行比较。我们还纳入了需要预定义图结构的时间感知多尺度时空图卷积网络（TAMP - S2GCNets）[4]、扩散卷积循环神经网络（DCRNN）[38]和时空图卷积网络（STGCN）[40]进行比较。对于长期预测，我们纳入了信息变压器（Informer）[13]、自动变压器（Autoformer）[14]、改革变压器（Reformer）[18]、联邦变压器（FEDformer）[30]、长序列时间序列预测线性模型（LTSF - Linear）[35]以及最近的补丁时间序列变压器（PatchTST）[39]进行比较。关于基线模型的更多详细信息可在附录B.2中找到。

---

<!-- Footnote -->

${}^{2}$ https://www.nrel.gov/grid/solar-power-data.html

${}^{2}$ https://www.nrel.gov/grid/solar-power-data.html

${}^{3}$ https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

${}^{3}$ https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

<!-- Footnote -->

---

Implementation Details Our model is implemented with Pytorch 1.8 [41], and all experiments are conducted on a single NVIDIA RTX 3080 10GB GPU. We take MSE (Mean Squared Error) as the loss function and report MAE (Mean Absolute Errors) and RMSE (Root Mean Squared Errors) results as the evaluation metrics. For additional implementation details, please refer to Appendix B.3

实现细节 我们的模型使用Pytorch 1.8 [41]实现，所有实验均在单张NVIDIA RTX 3080 10GB GPU上进行。我们采用均方误差（Mean Squared Error，MSE）作为损失函数，并报告平均绝对误差（Mean Absolute Errors，MAE）和均方根误差（Root Mean Squared Errors，RMSE）结果作为评估指标。有关更多实现细节，请参考附录B.3

### 4.1 Main Results

### 4.1 主要结果

<!-- Media -->

Table 1: Short-term forecasting comparison. The best results are in bold, and the second best results are underlined. Full benchmarks of short-term forecasting are in Appendix F.1

表1：短期预测比较。最佳结果以粗体显示，次佳结果加下划线。短期预测的完整基准见附录F.1

<table><tr><td rowspan="2">Models</td><td colspan="2">Solar</td><td colspan="2">Wiki</td><td colspan="2">Traffic</td><td colspan="2">ECG</td><td colspan="2">Electricity</td><td colspan="2">COVID-19</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>VAR</td><td>0.184</td><td>0.234</td><td>0.052</td><td>0.094</td><td>0.535</td><td>1.133</td><td>0.120</td><td>0.170</td><td>0.101</td><td>0.163</td><td>0.226</td><td>0.326</td></tr><tr><td>SFM</td><td>0.161</td><td>0.283</td><td>0.081</td><td>0.156</td><td>0.029</td><td>0.044</td><td>0.095</td><td>0.135</td><td>0.086</td><td>0.129</td><td>0.205</td><td>0.308</td></tr><tr><td>LSTNet</td><td>0.148</td><td>0.200</td><td>0.054</td><td>0.090</td><td>0.026</td><td>0.057</td><td>0.079</td><td>0.115</td><td>0.075</td><td>0.138</td><td>0.248</td><td>0.305</td></tr><tr><td>TCN</td><td>0.176</td><td>0.222</td><td>0.094</td><td>0.142</td><td>0.052</td><td>0.067</td><td>0.078</td><td>0.107</td><td>0.057</td><td>0.083</td><td>0.317</td><td>0.354</td></tr><tr><td>DeepGLO</td><td>0.178</td><td>0.400</td><td>0.110</td><td>0.113</td><td>0.025</td><td>0.037</td><td>0.110</td><td>0.163</td><td>0.090</td><td>0.131</td><td>0.169</td><td>0.253</td></tr><tr><td>Reformer</td><td>0.234</td><td>0.292</td><td>0.047</td><td>0.083</td><td>0.029</td><td>0.042</td><td>0.062</td><td>0.090</td><td>0.078</td><td>0.129</td><td>0.152</td><td>0.209</td></tr><tr><td>Informer</td><td>0.151</td><td>0.199</td><td>0.051</td><td>0.086</td><td>0.020</td><td>0.033</td><td>0.056</td><td>0.085</td><td>0.074</td><td>0.123</td><td>0.200</td><td>0.259</td></tr><tr><td>Autoformer</td><td>0.150</td><td>0.193</td><td>0.069</td><td>0.103</td><td>0.029</td><td>0.043</td><td>0.055</td><td>0.081</td><td>0.056</td><td>0.083</td><td>0.159</td><td>0.211</td></tr><tr><td>FEDformer</td><td>0.139</td><td>0.182</td><td>0.068</td><td>0.098</td><td>0.025</td><td>0.038</td><td>0.055</td><td>0.080</td><td>0.055</td><td>0.081</td><td>0.160</td><td>0.219</td></tr><tr><td>GraphWaveNet</td><td>0.183</td><td>0.238</td><td>0.061</td><td>0.105</td><td>0.013</td><td>0.034</td><td>0.093</td><td>0.142</td><td>0.094</td><td>0.140</td><td>0.201</td><td>0.255</td></tr><tr><td>StemGNN</td><td>0.176</td><td>0.222</td><td>0.190</td><td>0.255</td><td>0.080</td><td>0.135</td><td>0.100</td><td>0.130</td><td>0.070</td><td>0.101</td><td>0.421</td><td>0.508</td></tr><tr><td>MTGNN</td><td>0.151</td><td>0.207</td><td>0.101</td><td>0.140</td><td>0.013</td><td>0.030</td><td>0.090</td><td>0.139</td><td>0.077</td><td>0.113</td><td>0.394</td><td>0.488</td></tr><tr><td>AGCRN</td><td>0.123</td><td>0.214</td><td>0.044</td><td>0.079</td><td>0.084</td><td>0.166</td><td>0.055</td><td>0.080</td><td>0.074</td><td>0.116</td><td>0.254</td><td>0.309</td></tr><tr><td>FreTS (Ours)</td><td>0.120</td><td>0.162</td><td>0.041</td><td>0.074</td><td>0.011</td><td>0.023</td><td>0.053</td><td>0.078</td><td>0.050</td><td>0.076</td><td>0.123</td><td>0.167</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">太阳能；太阳的</td><td colspan="2">维基（Wiki）</td><td colspan="2">交通；流量</td><td colspan="2">心电图（ECG）</td><td colspan="2">电力；电</td><td colspan="2">新冠肺炎（COVID-19）</td></tr><tr><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td></tr><tr><td>向量自回归（VAR）</td><td>0.184</td><td>0.234</td><td>0.052</td><td>0.094</td><td>0.535</td><td>1.133</td><td>0.120</td><td>0.170</td><td>0.101</td><td>0.163</td><td>0.226</td><td>0.326</td></tr><tr><td>结构因子模型（SFM）</td><td>0.161</td><td>0.283</td><td>0.081</td><td>0.156</td><td>0.029</td><td>0.044</td><td>0.095</td><td>0.135</td><td>0.086</td><td>0.129</td><td>0.205</td><td>0.308</td></tr><tr><td>长短期时间序列网络（LSTNet）</td><td>0.148</td><td>0.200</td><td>0.054</td><td>0.090</td><td>0.026</td><td>0.057</td><td>0.079</td><td>0.115</td><td>0.075</td><td>0.138</td><td>0.248</td><td>0.305</td></tr><tr><td>时间卷积网络（Temporal Convolutional Network，TCN）</td><td>0.176</td><td>0.222</td><td>0.094</td><td>0.142</td><td>0.052</td><td>0.067</td><td>0.078</td><td>0.107</td><td>0.057</td><td>0.083</td><td>0.317</td><td>0.354</td></tr><tr><td>深度全局预测模型（Deep Global Prediction Model，DeepGLO）</td><td>0.178</td><td>0.400</td><td>0.110</td><td>0.113</td><td>0.025</td><td>0.037</td><td>0.110</td><td>0.163</td><td>0.090</td><td>0.131</td><td>0.169</td><td>0.253</td></tr><tr><td>改革者模型（Reformer）</td><td>0.234</td><td>0.292</td><td>0.047</td><td>0.083</td><td>0.029</td><td>0.042</td><td>0.062</td><td>0.090</td><td>0.078</td><td>0.129</td><td>0.152</td><td>0.209</td></tr><tr><td>告密者模型（Informer）</td><td>0.151</td><td>0.199</td><td>0.051</td><td>0.086</td><td>0.020</td><td>0.033</td><td>0.056</td><td>0.085</td><td>0.074</td><td>0.123</td><td>0.200</td><td>0.259</td></tr><tr><td>自动变压器模型（Autoformer）</td><td>0.150</td><td>0.193</td><td>0.069</td><td>0.103</td><td>0.029</td><td>0.043</td><td>0.055</td><td>0.081</td><td>0.056</td><td>0.083</td><td>0.159</td><td>0.211</td></tr><tr><td>联邦变压器模型（FEDformer）</td><td>0.139</td><td>0.182</td><td>0.068</td><td>0.098</td><td>0.025</td><td>0.038</td><td>0.055</td><td>0.080</td><td>0.055</td><td>0.081</td><td>0.160</td><td>0.219</td></tr><tr><td>图波网络（GraphWaveNet）</td><td>0.183</td><td>0.238</td><td>0.061</td><td>0.105</td><td>0.013</td><td>0.034</td><td>0.093</td><td>0.142</td><td>0.094</td><td>0.140</td><td>0.201</td><td>0.255</td></tr><tr><td>茎图神经网络（StemGNN）</td><td>0.176</td><td>0.222</td><td>0.190</td><td>0.255</td><td>0.080</td><td>0.135</td><td>0.100</td><td>0.130</td><td>0.070</td><td>0.101</td><td>0.421</td><td>0.508</td></tr><tr><td>多任务图神经网络（MTGNN）</td><td>0.151</td><td>0.207</td><td>0.101</td><td>0.140</td><td>0.013</td><td>0.030</td><td>0.090</td><td>0.139</td><td>0.077</td><td>0.113</td><td>0.394</td><td>0.488</td></tr><tr><td>自适应图卷积循环网络（AGCRN）</td><td>0.123</td><td>0.214</td><td>0.044</td><td>0.079</td><td>0.084</td><td>0.166</td><td>0.055</td><td>0.080</td><td>0.074</td><td>0.116</td><td>0.254</td><td>0.309</td></tr><tr><td>频率时间序列模型（FreTS，我们的方法）</td><td>0.120</td><td>0.162</td><td>0.041</td><td>0.074</td><td>0.011</td><td>0.023</td><td>0.053</td><td>0.078</td><td>0.050</td><td>0.076</td><td>0.123</td><td>0.167</td></tr></tbody></table>

Table 2: Long-term forecasting comparison. We set the lookback window size $L$ as 96 and the prediction length as $\tau  \in  \{ {96},{192},{336},{720}\}$ except for traffic dataset whose prediction length is set as $\tau  \in  \{ {48},{96},{192},{336}\}$ . The best results are in bold and the second best are underlined. Full results of long-term forecasting are included in Appendix F.2

表2：长期预测比较。我们将回溯窗口大小$L$设为96，预测长度设为$\tau  \in  \{ {96},{192},{336},{720}\}$，交通数据集除外，其预测长度设为$\tau  \in  \{ {48},{96},{192},{336}\}$。最佳结果用粗体显示，次佳结果用下划线标注。长期预测的完整结果见附录F.2

<table><tr><td colspan="2" rowspan="2">Models Metrics</td><td colspan="2">FreTS</td><td colspan="2">PatchTST</td><td colspan="2">LTSF-Linear</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td><td colspan="2">Reformer</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.032</td><td>0.071</td><td>0.034</td><td>0.074</td><td>0.040</td><td>0.081</td><td>0.050</td><td>0.088</td><td>0.064</td><td>0.104</td><td>0.101</td><td>0.139</td><td>0.108</td><td>0.152</td></tr><tr><td>192</td><td>0.040</td><td>0.081</td><td>$\underline{0.042}$</td><td>$\underline{0.084}$</td><td>0.048</td><td>0.089</td><td>0.051</td><td>0.092</td><td>0.061</td><td>0.103</td><td>0.097</td><td>0.134</td><td>0.147</td><td>0.201</td></tr><tr><td>336</td><td>0.046</td><td>0.090</td><td>0.049</td><td>0.094</td><td>0.056</td><td>0.098</td><td>0.057</td><td>0.100</td><td>0.059</td><td>0.101</td><td>0.115</td><td>0.155</td><td>0.154</td><td>0.203</td></tr><tr><td>720</td><td>0.055</td><td>0.099</td><td>$\underline{0.056}$</td><td>$\underline{0.102}$</td><td>0.065</td><td>0.106</td><td>0.064</td><td>0.109</td><td>0.065</td><td>0.110</td><td>0.132</td><td>0.175</td><td>0.173</td><td>0.228</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.037</td><td>0.051</td><td>0.039</td><td>0.052</td><td>0.038</td><td>0.052</td><td>0.050</td><td>0.067</td><td>0.050</td><td>0.066</td><td>0.066</td><td>0.084</td><td>0.126</td><td>0.146</td></tr><tr><td>192</td><td>0.050</td><td>0.067</td><td>0.055</td><td>0.074</td><td>0.053</td><td>0.069</td><td>0.064</td><td>0.082</td><td>0.063</td><td>0.083</td><td>0.068</td><td>0.088</td><td>0.147</td><td>0.169</td></tr><tr><td>336</td><td>0.062</td><td>0.082</td><td>0.071</td><td>0.093</td><td>$\underline{0.064}$</td><td>$\underline{0.085}$</td><td>0.080</td><td>0.105</td><td>0.075</td><td>0.101</td><td>0.093</td><td>0.127</td><td>0.157</td><td>0.189</td></tr><tr><td>720</td><td>0.088</td><td>0.110</td><td>0.132</td><td>0.166</td><td>0.092</td><td>$\underline{0.116}$</td><td>0.151</td><td>0.183</td><td>0.150</td><td>0.181</td><td>0.117</td><td>0.170</td><td>0.166</td><td>0.201</td></tr><tr><td rowspan="4">Traffic</td><td>48</td><td>0.018</td><td>0.036</td><td>0.016</td><td>0.032</td><td>0.020</td><td>0.039</td><td>0.022</td><td>0.036</td><td>0.026</td><td>0.042</td><td>0.023</td><td>0.039</td><td>0.035</td><td>0.053</td></tr><tr><td>96</td><td>0.020</td><td>0.038</td><td>0.018</td><td>0.035</td><td>0.022</td><td>0.042</td><td>0.023</td><td>0.044</td><td>0.033</td><td>0.050</td><td>0.030</td><td>0.047</td><td>0.035</td><td>0.054</td></tr><tr><td>192</td><td>0.019</td><td>0.038</td><td>0.020</td><td>0.039</td><td>0.020</td><td>0.040</td><td>0.022</td><td>0.042</td><td>0.035</td><td>0.053</td><td>0.034</td><td>0.053</td><td>0.035</td><td>0.054</td></tr><tr><td>336</td><td>0.020</td><td>0.039</td><td>$\underline{0.021}$</td><td>$\underline{0.040}$</td><td>$\underline{0.021}$</td><td>0.041</td><td>0.021</td><td>0.040</td><td>0.032</td><td>0.050</td><td>0.035</td><td>0.054</td><td>0.035</td><td>0.055</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.039</td><td>0.065</td><td>0.041</td><td>0.067</td><td>0.045</td><td>0.075</td><td>0.049</td><td>0.072</td><td>0.051</td><td>0.075</td><td>0.094</td><td>0.124</td><td>0.095</td><td>0.125</td></tr><tr><td>192</td><td>0.040</td><td>0.064</td><td>0.042</td><td>0.066</td><td>0.043</td><td>0.070</td><td>0.049</td><td>0.072</td><td>0.072</td><td>0.099</td><td>0.105</td><td>0.138</td><td>0.121</td><td>0.152</td></tr><tr><td>336</td><td>0.046</td><td>0.072</td><td>0.043</td><td>0.067</td><td>0.044</td><td>0.071</td><td>0.051</td><td>0.075</td><td>0.084</td><td>0.115</td><td>0.112</td><td>0.144</td><td>0.122</td><td>0.152</td></tr><tr><td>720</td><td>0.052</td><td>0.079</td><td>0.055</td><td>0.081</td><td>$\underline{0.054}$</td><td>$\underline{0.080}$</td><td>0.055</td><td>0.077</td><td>0.088</td><td>0.119</td><td>0.116</td><td>0.148</td><td>0.120</td><td>0.151</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.061</td><td>0.087</td><td>0.065</td><td>0.091</td><td>0.063</td><td>0.089</td><td>0.072</td><td>0.096</td><td>0.079</td><td>0.105</td><td>0.093</td><td>0.121</td><td>0.113</td><td>0.143</td></tr><tr><td>192</td><td>0.065</td><td>0.091</td><td>0.069</td><td>0.094</td><td>0.067</td><td>0.094</td><td>0.076</td><td>0.100</td><td>0.086</td><td>0.114</td><td>0.103</td><td>0.137</td><td>0.120</td><td>0.148</td></tr><tr><td>336</td><td>0.070</td><td>0.096</td><td>0.073</td><td>0.099</td><td>0.070</td><td>0.097</td><td>0.080</td><td>0.105</td><td>0.088</td><td>0.119</td><td>0.112</td><td>0.145</td><td>0.124</td><td>0.155</td></tr><tr><td>720</td><td>0.082</td><td>0.108</td><td>0.087</td><td>0.113</td><td>0.082</td><td>0.108</td><td>0.090</td><td>0.116</td><td>0.102</td><td>0.136</td><td>0.125</td><td>0.157</td><td>0.126</td><td>0.155</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.052</td><td>0.077</td><td>0.055</td><td>0.082</td><td>0.055</td><td>0.080</td><td>0.063</td><td>0.087</td><td>0.081</td><td>0.109</td><td>0.070</td><td>0.096</td><td>0.065</td><td>0.089</td></tr><tr><td>192</td><td>0.057</td><td>0.083</td><td>0.059</td><td>0.085</td><td>0.060</td><td>0.087</td><td>0.068</td><td>0.093</td><td>0.083</td><td>0.112</td><td>0.082</td><td>0.107</td><td>0.081</td><td>0.108</td></tr><tr><td>336</td><td>0.062</td><td>0.089</td><td>0.064</td><td>0.091</td><td>0.065</td><td>0.093</td><td>0.075</td><td>0.102</td><td>0.091</td><td>0.125</td><td>0.090</td><td>0.119</td><td>0.100</td><td>0.128</td></tr><tr><td>720</td><td>0.069</td><td>0.096</td><td>0.070</td><td>0.097</td><td>0.072</td><td>0.099</td><td>0.081</td><td>0.108</td><td>0.093</td><td>0.126</td><td>0.115</td><td>0.149</td><td>0.132</td><td>0.163</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">模型指标</td><td colspan="2">FreTS（原词：FreTS）</td><td colspan="2">PatchTST（原词：PatchTST）</td><td colspan="2">LTSF - 线性模型（原词：LTSF - Linear）</td><td colspan="2">FEDformer（原词：FEDformer）</td><td colspan="2">Autoformer（原词：Autoformer）</td><td colspan="2">告密者</td><td colspan="2">改革者</td></tr><tr><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td></tr><tr><td rowspan="4">天气</td><td>96</td><td>0.032</td><td>0.071</td><td>0.034</td><td>0.074</td><td>0.040</td><td>0.081</td><td>0.050</td><td>0.088</td><td>0.064</td><td>0.104</td><td>0.101</td><td>0.139</td><td>0.108</td><td>0.152</td></tr><tr><td>192</td><td>0.040</td><td>0.081</td><td>$\underline{0.042}$</td><td>$\underline{0.084}$</td><td>0.048</td><td>0.089</td><td>0.051</td><td>0.092</td><td>0.061</td><td>0.103</td><td>0.097</td><td>0.134</td><td>0.147</td><td>0.201</td></tr><tr><td>336</td><td>0.046</td><td>0.090</td><td>0.049</td><td>0.094</td><td>0.056</td><td>0.098</td><td>0.057</td><td>0.100</td><td>0.059</td><td>0.101</td><td>0.115</td><td>0.155</td><td>0.154</td><td>0.203</td></tr><tr><td>720</td><td>0.055</td><td>0.099</td><td>$\underline{0.056}$</td><td>$\underline{0.102}$</td><td>0.065</td><td>0.106</td><td>0.064</td><td>0.109</td><td>0.065</td><td>0.110</td><td>0.132</td><td>0.175</td><td>0.173</td><td>0.228</td></tr><tr><td rowspan="4">交换；交流；交易</td><td>96</td><td>0.037</td><td>0.051</td><td>0.039</td><td>0.052</td><td>0.038</td><td>0.052</td><td>0.050</td><td>0.067</td><td>0.050</td><td>0.066</td><td>0.066</td><td>0.084</td><td>0.126</td><td>0.146</td></tr><tr><td>192</td><td>0.050</td><td>0.067</td><td>0.055</td><td>0.074</td><td>0.053</td><td>0.069</td><td>0.064</td><td>0.082</td><td>0.063</td><td>0.083</td><td>0.068</td><td>0.088</td><td>0.147</td><td>0.169</td></tr><tr><td>336</td><td>0.062</td><td>0.082</td><td>0.071</td><td>0.093</td><td>$\underline{0.064}$</td><td>$\underline{0.085}$</td><td>0.080</td><td>0.105</td><td>0.075</td><td>0.101</td><td>0.093</td><td>0.127</td><td>0.157</td><td>0.189</td></tr><tr><td>720</td><td>0.088</td><td>0.110</td><td>0.132</td><td>0.166</td><td>0.092</td><td>$\underline{0.116}$</td><td>0.151</td><td>0.183</td><td>0.150</td><td>0.181</td><td>0.117</td><td>0.170</td><td>0.166</td><td>0.201</td></tr><tr><td rowspan="4">交通</td><td>48</td><td>0.018</td><td>0.036</td><td>0.016</td><td>0.032</td><td>0.020</td><td>0.039</td><td>0.022</td><td>0.036</td><td>0.026</td><td>0.042</td><td>0.023</td><td>0.039</td><td>0.035</td><td>0.053</td></tr><tr><td>96</td><td>0.020</td><td>0.038</td><td>0.018</td><td>0.035</td><td>0.022</td><td>0.042</td><td>0.023</td><td>0.044</td><td>0.033</td><td>0.050</td><td>0.030</td><td>0.047</td><td>0.035</td><td>0.054</td></tr><tr><td>192</td><td>0.019</td><td>0.038</td><td>0.020</td><td>0.039</td><td>0.020</td><td>0.040</td><td>0.022</td><td>0.042</td><td>0.035</td><td>0.053</td><td>0.034</td><td>0.053</td><td>0.035</td><td>0.054</td></tr><tr><td>336</td><td>0.020</td><td>0.039</td><td>$\underline{0.021}$</td><td>$\underline{0.040}$</td><td>$\underline{0.021}$</td><td>0.041</td><td>0.021</td><td>0.040</td><td>0.032</td><td>0.050</td><td>0.035</td><td>0.054</td><td>0.035</td><td>0.055</td></tr><tr><td rowspan="4">电力</td><td>96</td><td>0.039</td><td>0.065</td><td>0.041</td><td>0.067</td><td>0.045</td><td>0.075</td><td>0.049</td><td>0.072</td><td>0.051</td><td>0.075</td><td>0.094</td><td>0.124</td><td>0.095</td><td>0.125</td></tr><tr><td>192</td><td>0.040</td><td>0.064</td><td>0.042</td><td>0.066</td><td>0.043</td><td>0.070</td><td>0.049</td><td>0.072</td><td>0.072</td><td>0.099</td><td>0.105</td><td>0.138</td><td>0.121</td><td>0.152</td></tr><tr><td>336</td><td>0.046</td><td>0.072</td><td>0.043</td><td>0.067</td><td>0.044</td><td>0.071</td><td>0.051</td><td>0.075</td><td>0.084</td><td>0.115</td><td>0.112</td><td>0.144</td><td>0.122</td><td>0.152</td></tr><tr><td>720</td><td>0.052</td><td>0.079</td><td>0.055</td><td>0.081</td><td>$\underline{0.054}$</td><td>$\underline{0.080}$</td><td>0.055</td><td>0.077</td><td>0.088</td><td>0.119</td><td>0.116</td><td>0.148</td><td>0.120</td><td>0.151</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.061</td><td>0.087</td><td>0.065</td><td>0.091</td><td>0.063</td><td>0.089</td><td>0.072</td><td>0.096</td><td>0.079</td><td>0.105</td><td>0.093</td><td>0.121</td><td>0.113</td><td>0.143</td></tr><tr><td>192</td><td>0.065</td><td>0.091</td><td>0.069</td><td>0.094</td><td>0.067</td><td>0.094</td><td>0.076</td><td>0.100</td><td>0.086</td><td>0.114</td><td>0.103</td><td>0.137</td><td>0.120</td><td>0.148</td></tr><tr><td>336</td><td>0.070</td><td>0.096</td><td>0.073</td><td>0.099</td><td>0.070</td><td>0.097</td><td>0.080</td><td>0.105</td><td>0.088</td><td>0.119</td><td>0.112</td><td>0.145</td><td>0.124</td><td>0.155</td></tr><tr><td>720</td><td>0.082</td><td>0.108</td><td>0.087</td><td>0.113</td><td>0.082</td><td>0.108</td><td>0.090</td><td>0.116</td><td>0.102</td><td>0.136</td><td>0.125</td><td>0.157</td><td>0.126</td><td>0.155</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.052</td><td>0.077</td><td>0.055</td><td>0.082</td><td>0.055</td><td>0.080</td><td>0.063</td><td>0.087</td><td>0.081</td><td>0.109</td><td>0.070</td><td>0.096</td><td>0.065</td><td>0.089</td></tr><tr><td>192</td><td>0.057</td><td>0.083</td><td>0.059</td><td>0.085</td><td>0.060</td><td>0.087</td><td>0.068</td><td>0.093</td><td>0.083</td><td>0.112</td><td>0.082</td><td>0.107</td><td>0.081</td><td>0.108</td></tr><tr><td>336</td><td>0.062</td><td>0.089</td><td>0.064</td><td>0.091</td><td>0.065</td><td>0.093</td><td>0.075</td><td>0.102</td><td>0.091</td><td>0.125</td><td>0.090</td><td>0.119</td><td>0.100</td><td>0.128</td></tr><tr><td>720</td><td>0.069</td><td>0.096</td><td>0.070</td><td>0.097</td><td>0.072</td><td>0.099</td><td>0.081</td><td>0.108</td><td>0.093</td><td>0.126</td><td>0.115</td><td>0.149</td><td>0.132</td><td>0.163</td></tr></tbody></table>

<!-- Media -->

Short-Term Time Series Forecasting Table 1 presents the forecasting accuracy of our FreTS compared to thirteen baselines on six datasets, with an input length of 12 and a prediction length of 12. The best results are highlighted in bold and the second-best results are underlined. From the table, we observe that FreTS outperforms all baselines on MAE and RMSE across all datasets, and on average it makes improvement of 9.4% on MAE and 11.6% on RMSE. We credit this to the fact that FreTS explicitly models both channel and temporal dependencies, and it flexibly unifies channel and temporal modeling in the frequency domain, which can effectively capture the key patterns with the global view and energy compaction. We further report the complete benchmarks of short-term forecasting under different steps on different datasets (including METR-LA dataset) in Appendix F.1

短期时间序列预测 表1展示了我们的FreTS（频域时间序列模型，Frequency-based Temporal Series model）与13种基线模型在6个数据集上的预测准确性，输入长度为12，预测长度也为12。最佳结果以粗体突出显示，次佳结果加下划线。从表中可以看出，在所有数据集上，FreTS在平均绝对误差（MAE）和均方根误差（RMSE）方面均优于所有基线模型，平均而言，它在MAE上提高了9.4%，在RMSE上提高了11.6%。我们认为这得益于FreTS明确地对通道和时间依赖关系进行建模，并在频域中灵活地统一了通道和时间建模，这能够从全局视角和能量压缩的角度有效捕捉关键模式。我们在附录F.1中进一步报告了不同数据集（包括METR - LA数据集）在不同步长下的短期预测完整基准结果。

Long-term Time Series Forecasting Table 2 showcases the long-term forecasting results of FreTS compared to six representative baselines on six benchmarks with various prediction lengths. For the traffic dataset,we select 48 as the lookback window size $L$ with the prediction lengths $\tau  \in$ $\{ {48},{96},{192},{336}\}$ . For the other datasets,the input lookback window length is set to 96 and the prediction length is set to $\tau  \in  \{ {96},{192},{336},{720}\}$ . The results demonstrate that FreTS outperforms all baselines on all datasets. Quantitatively, compared with the best results of Transformer-based models, FreTS has an average decrease of more than 20% in MAE and RMSE. Compared with more recent LSTF-Linear [35] and the SOTA PathchTST [39], FreTS can still outperform them in general. In addition, we provide further comparison of FreTS and other baselines and report performance under different lookback window sizes in Appendix F.2. Combining Tables 1 and 2, we can conclude that FreTS achieves competitive performance in both short-term and long-term forecasting task.

长期时间序列预测 表2展示了FreTS与六个具有代表性的基线模型在六个基准数据集上进行不同预测长度的长期预测结果。对于交通数据集，我们选择48作为回溯窗口大小（$L$），预测长度为（$\tau  \in$ $\{ {48},{96},{192},{336}\}$）。对于其他数据集，输入回溯窗口长度设置为96，预测长度设置为（$\tau  \in  \{ {96},{192},{336},{720}\}$）。结果表明，FreTS在所有数据集上的表现均优于所有基线模型。从量化角度来看，与基于Transformer的模型的最佳结果相比，FreTS的平均绝对误差（MAE）和均方根误差（RMSE）平均降低了20%以上。与近期的LSTF-Linear [35]和最优的PatchTST [39]相比，FreTS总体上仍能超越它们。此外，我们在附录F.2中进一步比较了FreTS和其他基线模型，并报告了不同回溯窗口大小下的性能。结合表1和表2，我们可以得出结论，FreTS在短期和长期预测任务中均取得了具有竞争力的性能。

### 4.2 Model Analysis

### 4.2 模型分析

Frequency Channel and Temporal Learners We analyze the effects of frequency channel and temporal learners in Table 3 in both short-term and long-term experimental settings. We consider two variants: FreCL: we remove the frequency temporal learner from FreTS, and FreTL: we remove the frequency channel learner from FreTS. From the comparison, we observe that the frequency channel learner plays a more important role in short-term forecasting. In long-term forecasting, we note that the frequency temporal learner is more effective than the frequency channel learner. In Appendix E. 1, we also conduct the experiments and report performance on other datasets. Interestingly, we find out the channel learner would lead to the worse performance in some long-term forecasting cases. A potential explanation is that the channel independent strategy [39] brings more benefit to forecasting.

频率通道和时间学习器 我们在表3中分析了频率通道和时间学习器在短期和长期实验设置中的效果。我们考虑了两种变体：FreCL：我们从FreTS中移除频率时间学习器；FreTL：我们从FreTS中移除频率通道学习器。通过比较，我们观察到频率通道学习器在短期预测中发挥着更重要的作用。在长期预测中，我们注意到频率时间学习器比频率通道学习器更有效。在附录E.1中，我们还进行了实验并报告了在其他数据集上的性能。有趣的是，我们发现通道学习器在某些长期预测情况下会导致更差的性能。一个可能的解释是通道独立策略[39]为预测带来了更多好处。

<!-- Media -->

Table 3: Ablation studies of frequency channel and temporal learners in both short-term and long-term forecasting. 'I/O' indicates lookback window sizes/prediction lengths.

表3：频率通道和时间学习器在短期和长期预测中的消融研究。“输入/输出”表示回溯窗口大小/预测长度。

<table><tr><td>Tasks</td><td colspan="4">Short-term</td><td colspan="4">Long-term</td></tr><tr><td>Dataset I/O</td><td colspan="2">Electricity 12/12</td><td colspan="2">METR-LA 12/12</td><td colspan="2">Exchange 96/336</td><td colspan="2">Weather 96/336</td></tr><tr><td>Metrics</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>FreCL</td><td>0.054</td><td>0.080</td><td>0.086</td><td>0.168</td><td>0.067</td><td>0.086</td><td>0.051</td><td>0.094</td></tr><tr><td>FreTL</td><td>0.058</td><td>0.086</td><td>0.085</td><td>0.167</td><td>0.065</td><td>0.085</td><td>0.047</td><td>0.091</td></tr><tr><td>FreTS</td><td>0.050</td><td>0.076</td><td>0.080</td><td>0.166</td><td>0.062</td><td>0.082</td><td>0.046</td><td>0.090</td></tr></table>

<table><tbody><tr><td>任务</td><td colspan="4">短期</td><td colspan="4">长期</td></tr><tr><td>数据集输入/输出</td><td colspan="2">电力数据 12/12</td><td colspan="2">洛杉矶都市高速公路交通流量数据集（METR - LA） 12/12</td><td colspan="2">交换96/336</td><td colspan="2">天气96/336</td></tr><tr><td>指标</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td></tr><tr><td>频率约束损失（FreCL）</td><td>0.054</td><td>0.080</td><td>0.086</td><td>0.168</td><td>0.067</td><td>0.086</td><td>0.051</td><td>0.094</td></tr><tr><td>弗雷特尔（FreTL）</td><td>0.058</td><td>0.086</td><td>0.085</td><td>0.167</td><td>0.065</td><td>0.085</td><td>0.047</td><td>0.091</td></tr><tr><td>弗雷茨（FreTS）</td><td>0.050</td><td>0.076</td><td>0.080</td><td>0.166</td><td>0.062</td><td>0.082</td><td>0.046</td><td>0.090</td></tr></tbody></table>

<!-- Media -->

FreMLP vs. MLP We further study the effectiveness of FreMLP in time series forecasting. We use FreMLP to replace the original MLP component in the existing SOTA MLP-based models (i.e., DLinear and NLinear [35]), and compare their performances with the original DLinear and NLinear under the same experimental settings. The experimental results are presented in Table 4 . From the table, we easily observe that for any prediction length, the performance of both DLinear and NLinear models has been improved after replacing the corresponding MLP component with our FreMLP. Quantitatively, incorporating FreMLP into the DLinear model brings an average improvement of 6.4% in MAE and 11.4% in RMSE on the Exchange dataset, and 4.9% in MAE and 3.5% in RMSE on the Weather dataset. A similar improvement has also been achieved on the two datasets with regard to NLinear, according to Table 4. These results confirm the effectiveness of FreMLP compared to MLP again and we include more implementation details and analysis in Appendix B.5

频率多层感知机（FreMLP）与多层感知机（MLP）的比较 我们进一步研究了频率多层感知机（FreMLP）在时间序列预测中的有效性。我们使用频率多层感知机（FreMLP）替换现有基于多层感知机（MLP）的最优模型（即DLinear和NLinear [35]）中的原始多层感知机（MLP）组件，并在相同的实验设置下将它们的性能与原始的DLinear和NLinear进行比较。实验结果如表4所示。从表中我们可以很容易地观察到，对于任何预测长度，在将相应的多层感知机（MLP）组件替换为我们的频率多层感知机（FreMLP）后，DLinear和NLinear模型的性能都得到了提升。从量化角度来看，在Exchange数据集上，将频率多层感知机（FreMLP）融入DLinear模型后，平均绝对误差（MAE）平均提升了6.4%，均方根误差（RMSE）平均提升了11.4%；在Weather数据集上，平均绝对误差（MAE）提升了4.9%，均方根误差（RMSE）提升了3.5%。根据表4，在NLinear模型的两个数据集上也取得了类似的提升。这些结果再次证实了与多层感知机（MLP）相比，频率多层感知机（FreMLP）的有效性，我们在附录B.5中提供了更多的实现细节和分析。

<!-- Media -->

Table 4: Ablation study on the Exchange and Weather datasets with a lookback window size of 96 and the prediction length $\tau  \in  \{ {96},{192},{336},{720}\}$ . DLinear (FreMLP)/NLinear (FreMLP) means that we replace the MLPs in DLinear/NLinear with FreMLP. The best results are in bold.

表4：在Exchange和Weather数据集上进行的消融实验，回溯窗口大小为96，预测长度为$\tau  \in  \{ {96},{192},{336},{720}\}$。DLinear (FreMLP)/NLinear (FreMLP) 表示我们用FreMLP替换了DLinear/NLinear中的多层感知机（MLP）。最佳结果以粗体显示。

<table><tr><td>Datasets</td><td colspan="8">Exchange</td><td colspan="8">Weather</td></tr><tr><td>Lengths</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td></tr><tr><td>Metrics</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>DLinear</td><td>0.037</td><td>0.051</td><td>0.054</td><td>0.072</td><td>0.071</td><td>0.095</td><td>0.095</td><td>0.119</td><td>0.041</td><td>0.081</td><td>0.047</td><td>0.089</td><td>0.056</td><td>0.098</td><td>0.065</td><td>0.106</td></tr><tr><td>DLinear (FreMLP)</td><td>0.036</td><td>0.049</td><td>0.053</td><td>0.071</td><td>0.063</td><td>0.071</td><td>0.086</td><td>0.101</td><td>0.038</td><td>0.078</td><td>0.045</td><td>0.086</td><td>0.055</td><td>0.097</td><td>0.061</td><td>0.100</td></tr><tr><td>NLinear</td><td>0.037</td><td>0.051</td><td>0.051</td><td>0.069</td><td>0.069</td><td>0.093</td><td>0.115</td><td>0.146</td><td>0.037</td><td>0.081</td><td>0.045</td><td>0.089</td><td>0.052</td><td>0.098</td><td>0.058</td><td>0.106</td></tr><tr><td>NLinear (FreMLP)</td><td>0.036</td><td>0.050</td><td>0.049</td><td>0.067</td><td>0.067</td><td>0.091</td><td>0.109</td><td>0.139</td><td>0.035</td><td>0.076</td><td>0.043</td><td>0.084</td><td>0.050</td><td>0.094</td><td>0.057</td><td>0.103</td></tr></table>

<table><tbody><tr><td>数据集</td><td colspan="8">交换；交易</td><td colspan="8">天气</td></tr><tr><td>长度</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td></tr><tr><td>指标；度量</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td></tr><tr><td>深度线性模型（DLinear）</td><td>0.037</td><td>0.051</td><td>0.054</td><td>0.072</td><td>0.071</td><td>0.095</td><td>0.095</td><td>0.119</td><td>0.041</td><td>0.081</td><td>0.047</td><td>0.089</td><td>0.056</td><td>0.098</td><td>0.065</td><td>0.106</td></tr><tr><td>深度线性模型（频率多层感知机版）（DLinear (FreMLP)）</td><td>0.036</td><td>0.049</td><td>0.053</td><td>0.071</td><td>0.063</td><td>0.071</td><td>0.086</td><td>0.101</td><td>0.038</td><td>0.078</td><td>0.045</td><td>0.086</td><td>0.055</td><td>0.097</td><td>0.061</td><td>0.100</td></tr><tr><td>非线性线性模型（NLinear）</td><td>0.037</td><td>0.051</td><td>0.051</td><td>0.069</td><td>0.069</td><td>0.093</td><td>0.115</td><td>0.146</td><td>0.037</td><td>0.081</td><td>0.045</td><td>0.089</td><td>0.052</td><td>0.098</td><td>0.058</td><td>0.106</td></tr><tr><td>非线性线性模型（频率多层感知机版）（NLinear (FreMLP)）</td><td>0.036</td><td>0.050</td><td>0.049</td><td>0.067</td><td>0.067</td><td>0.091</td><td>0.109</td><td>0.139</td><td>0.035</td><td>0.076</td><td>0.043</td><td>0.084</td><td>0.050</td><td>0.094</td><td>0.057</td><td>0.103</td></tr></tbody></table>

<!-- Media -->

### 4.3 Efficiency Analysis

### 4.3 效率分析

The complexity of our proposed FreTS is $\mathcal{O}\left( {N\log N + L\log L}\right)$ . We perform efficiency comparisons with some state-of-the-art GNN-based methods and Transformer-based models under different numbers of variables $N$ and prediction lengths $\tau$ ,respectively. On the Wiki dataset,we conduct experiments over $N \in  \{ {1000},{2000},{3000},{4000},{5000}\}$ under the same lookback window size of 12 and prediction length of 12 , as shown in Figure 4(a). From the figure, we can find that: (1) The amount of FreTS parameters is agnostic to $N$ . (2) Compared with AGCRN,FreTS incurs an average ${30}\%$ reduction of the number of parameters and ${20}\%$ reduction of training time. On the Exchange dataset,we conduct experiments on different prediction lengths $\tau  \in  \{ {96},{192},{336},{480}\}$ with the same input length of 96 . The results are shown in Figure 4(b). It demonstrates: (1) Compared with Transformer-based methods (FEDformer [30], Autoformer [14], and Informer [13]), FreTS reduces the number of parameters by at least 3 times. (2) The training time of FreTS is averagely 3 times faster than Informer, 5 times faster than Autoformer, and more than 10 times faster than FEDformer. These show our great potential in real-world deployment.

我们提出的FreTS（频率变换序列模型，Frequency Transformed Sequence model）的复杂度为$\mathcal{O}\left( {N\log N + L\log L}\right)$。我们分别在不同的变量数量$N$和预测长度$\tau$下，将其与一些最先进的基于图神经网络（GNN）的方法和基于Transformer的模型进行了效率比较。在Wiki数据集上，我们在相同的回溯窗口大小为12和预测长度为12的条件下，对$N \in  \{ {1000},{2000},{3000},{4000},{5000}\}$进行了实验，如图4(a)所示。从图中我们可以发现：(1) FreTS的参数数量与$N$无关。(2) 与AGCRN（自适应图卷积循环网络，Adaptive Graph Convolutional Recurrent Network）相比，FreTS的参数数量平均减少了${30}\%$，训练时间减少了${20}\%$。在Exchange数据集上，我们在相同输入长度为96的情况下，对不同的预测长度$\tau  \in  \{ {96},{192},{336},{480}\}$进行了实验。结果如图4(b)所示。这表明：(1) 与基于Transformer的方法（FEDformer [30]、Autoformer [14]和Informer [13]）相比，FreTS的参数数量至少减少了3倍。(2) FreTS的训练时间平均比Informer快3倍，比Autoformer快5倍，比FEDformer快10倍以上。这些结果显示了我们在实际应用中的巨大潜力。

<!-- Media -->

<!-- figureText: different variable numbers Prediction lengths (a) Parameters (left) and training time (right) under (b) Parameters (left) and training time (right) under different prediction lengths -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_8.jpg?x=310&y=219&w=1167&h=289&r=0"/>

Figure 4: Efficiency analysis (model parameters and training time) on the Wiki and Exchange dataset. (a) The efficiency comparison under different number of variables: the number of variables is enlarged from 1000 to 5000 with the input window size as 12 and the prediction length as 12 on Wiki dataset. (b) The efficiency comparison under the prediction lengths: we conduct experiments with prediction lengths prolonged from 96 to 480 under the same window size of 96 on the Exchange dataset.

图4：在维基（Wiki）和外汇（Exchange）数据集上的效率分析（模型参数和训练时间）。(a) 不同变量数量下的效率比较：在维基数据集上，输入窗口大小为12，预测长度为12，变量数量从1000增加到5000。(b) 不同预测长度下的效率比较：在外汇数据集上，我们在相同的窗口大小96下进行实验，预测长度从96延长到480。

<!-- Media -->

### 4.4 Visualization Analysis

### 4.4 可视化分析

In Figure 5, we visualize the learned weights $\mathcal{W}$ in FreMLP on the Traffic dataset with a lookback window size of 48 and prediction length of 192. As the weights $\mathcal{W}$ are complex numbers,we provide visualizations of the real part ${\mathcal{W}}_{r}$ (presented in (a)) and the imaginary part ${\mathcal{W}}_{i}$ (presented in (b)) separately. From the figure, we can observe that both the real and imaginary parts play a crucial role in learning process: the weight coefficients of the real or imaginary part exhibit energy aggregation characteristics (clear diagonal patterns) which can facilitate to learn the significant features. In Appendix E.2, we further conduct a detailed analysis on the effects of the real and imaginary parts in different contexts of forecasting, and the effects of the two parts in the FreMLP. We examine their individual contributions and investigate how they influence the final performance. Additional visualizations of the weights on different datasets with various settings, as well as visualizations of global periodic patterns, can be found in Appendix G.1 and Appendix G.2, respectively.

在图5中，我们可视化了FreMLP（频域多层感知器，Frequency Multi-Layer Perceptron）在交通数据集上学习到的权重$\mathcal{W}$，该数据集的回溯窗口大小为48，预测长度为192。由于权重$\mathcal{W}$是复数，我们分别展示了实部${\mathcal{W}}_{r}$（如图(a)所示）和虚部${\mathcal{W}}_{i}$（如图(b)所示）的可视化结果。从图中可以观察到，实部和虚部在学习过程中都起着至关重要的作用：实部或虚部的权重系数呈现出能量聚集特征（明显的对角模式），这有助于学习重要特征。在附录E.2中，我们进一步详细分析了实部和虚部在不同预测场景下的影响，以及它们在FreMLP中的作用。我们研究了它们各自的贡献，并探讨了它们如何影响最终性能。不同数据集在各种设置下的权重的额外可视化结果，以及全局周期性模式的可视化结果，分别可以在附录G.1和附录G.2中找到。

<!-- Media -->

<!-- figureText: (a) The real part ${\mathcal{W}}_{r}$ -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_8.jpg?x=860&y=1107&w=603&h=283&r=0"/>

Figure 5: Visualizing learned weights of FreMLP on the Traffic dataset. ${\mathcal{W}}_{r}$ represents the real part of $\mathcal{W}$ , and ${\mathcal{W}}_{i}$ represents the imaginary part.

图5：可视化FreMLP在交通数据集上学习到的权重。${\mathcal{W}}_{r}$表示$\mathcal{W}$的实部，${\mathcal{W}}_{i}$表示其虚部。

<!-- Media -->

## 5 Conclusion Remarks

## 5 结论

In this paper, we explore a novel direction and make a new attempt to apply frequency-domain MLPs for time series forecasting. We have redesigned MLPs in the frequency domain that can effectively capture the underlying patterns of time series with global view and energy compaction. We then verify this design by a simple yet effective architecture, FreTS, built upon the frequency-domain MLPs for time series forecasting. Our comprehensive empirical experiments on seven benchmarks of short-term forecasting and six benchmarks of long-term forecasting have validated the superiority of our proposed methods. Simple MLPs have several advantages and lay the foundation of modern deep learning, which have great potential for satisfied performance with high efficiency. We hope this work can facilitate more future research of MLPs on time series modeling.

在本文中，我们探索了一个新的方向，并进行了一次新的尝试，即将频域多层感知机（MLP）应用于时间序列预测。我们重新设计了频域中的MLP，它能够以全局视角和能量压缩的方式有效捕捉时间序列的潜在模式。然后，我们通过一个简单而有效的架构FreTS来验证这一设计，该架构基于频域MLP进行时间序列预测。我们在七个短期预测基准和六个长期预测基准上进行的全面实证实验验证了我们所提出方法的优越性。简单的MLP具有多个优点，并为现代深度学习奠定了基础，它们在实现高效且令人满意的性能方面具有巨大潜力。我们希望这项工作能够促进未来更多关于MLP在时间序列建模方面的研究。

## Acknowledgments and Disclosure of Funding

## 致谢与资金披露

The work was supported in part by the National Key Research and Development Program of China under Grant 2020AAA0104903 and 2019YFB1406300, and National Natural Science Foundation of China under Grant 62072039 and 62272048.

本研究得到了中国国家重点研发计划（项目编号：2020AAA0104903、2019YFB1406300）和中国国家自然科学基金（项目编号：62072039、62272048）的部分资助。

## References

## 参考文献

[1] Edward N Lorenz. Empirical orthogonal functions and statistical weather prediction, volume 1. Massachusetts Institute of Technology, Department of Meteorology Cambridge, 1956.

[1] 爱德华·N·洛伦兹（Edward N Lorenz）。经验正交函数与统计天气预报，第1卷。麻省理工学院（Massachusetts Institute of Technology）气象系，剑桥，1956年。

[2] Yu Zheng, Xiuwen Yi, Ming Li, Ruiyuan Li, Zhangqing Shan, Eric Chang, and Tianrui Li. Forecasting fine-grained air quality based on big data. In ${KDD}$ ,pages 2267-2276,2015.

[2] 郑宇、易秀文、李明、李瑞源、单章庆、埃里克·张（Eric Chang）和田瑞。基于大数据的细粒度空气质量预测。见${KDD}$，第2267 - 2276页，2015年。

[3] Hui He, Qi Zhang, Simeng Bai, Kun Yi, and Zhendong Niu. CATN: cross attentive tree-aware network for multivariate time series forecasting. In AAAI, pages 4030-4038. AAAI Press, 2022.

[3] 何辉、张琪、白思萌、易坤和牛振东。CATN：用于多变量时间序列预测的交叉注意力树感知网络。见AAAI会议论文集，第4030 - 4038页。AAAI出版社，2022年。

[4] Yuzhou Chen, Ignacio Segovia-Dominguez, Baris Coskunuzer, and Yulia Gel. TAMP-s2GCNets: Coupling time-aware multipersistence knowledge representation with spatio-supra graph convolutional networks for time-series forecasting. In International Conference on Learning Representations, 2022.

[4] 陈宇舟（Yuzhou Chen）、伊格纳西奥·塞戈维亚 - 多明格斯（Ignacio Segovia-Dominguez）、巴里斯·科斯库努泽尔（Baris Coskunuzer）和尤利娅·格尔（Yulia Gel）。TAMP - s2GCNets：将时间感知多持久性知识表示与时空超图卷积网络相结合用于时间序列预测。发表于《国际学习表征会议》，2022 年。

[5] Benjamin F King. Market and industry factors in stock price behavior. the Journal of Business, 39(1):139-190, 1966.

[5] 本杰明·F·金（Benjamin F King）。股票价格行为中的市场和行业因素。《商业杂志》，39(1):139 - 190，1966 年。

[6] Adebiyi A Ariyo, Adewumi O Adewumi, and Charles K Ayo. Stock price prediction using the arima model. In 2014 UKSim-AMSS 16th international conference on computer modelling and simulation, pages 106-112. IEEE, 2014.

[6] 阿德比伊·A·阿利约（Adebiyi A Ariyo）、阿德武米·O·阿德武米（Adewumi O Adewumi）和查尔斯·K·阿约（Charles K Ayo）。使用自回归积分滑动平均模型（ARIMA）进行股票价格预测。发表于《2014 年英国计算机模拟学会 - 美国数学与统计学会第 16 届计算机建模与仿真国际会议》，第 106 - 112 页。电气与电子工程师协会（IEEE），2014 年。

[7] Charles C Holt. Forecasting trends and seasonal by exponentially weighted moving averages. ONR Memorandum, 52(2), 1957.

[7] 查尔斯·C·霍尔特（Charles C Holt）。通过指数加权移动平均法进行趋势和季节性预测。海军研究办公室备忘录，52(2)，1957 年。

[8] Peter Whittle. Prediction and regulation by linear least-square methods. English Universities Press, 1963.

[8] 彼得·惠特尔（Peter Whittle）。用线性最小二乘法进行预测和调节。英国大学出版社，1963 年。

[9] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3):1181-1191, 2020.

[9] 大卫·萨利纳斯（David Salinas）、瓦伦丁·弗伦克特（Valentin Flunkert）、扬·加施豪斯（Jan Gasthaus）和蒂姆·亚努绍夫斯基（Tim Januschowski）。《Deepar：基于自回归循环网络的概率预测》。《国际预测期刊》（International Journal of Forecasting），36(3):1181 - 1191，2020年。

[10] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-term temporal patterns with deep neural networks. In SIGIR, pages 95-104, 2018.

[10] 赖国坤（Guokun Lai）、张维正（Wei - Cheng Chang）、杨一鸣（Yiming Yang）和刘瀚霄（Hanxiao Liu）。《利用深度神经网络对长短期时间模式进行建模》。发表于《信息检索研究与发展会议》（SIGIR），第95 - 104页，2018年。

[11] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. CoRR, abs/1803.01271, 2018.

[11] 白少杰（Shaojie Bai）、J. 齐科·科尔特（J. Zico Kolter）和弗拉德连·科尔图恩（Vladlen Koltun）。《对用于序列建模的通用卷积和循环网络的实证评估》。《计算机研究存储库》（CoRR），编号abs/1803.01271，2018年。

[12] Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet: time series modeling and forecasting with sample convolution and interaction. Advances in Neural Information Processing Systems, 35:5816-5828, 2022.

[12] 刘敏浩（Minhao Liu）、曾爱玲（Ailing Zeng）、陈慕溪（Muxi Chen）、徐志坚（Zhijian Xu）、赖秋霞（Qiuxia Lai）、马灵娜（Lingna Ma）和徐强（Qiang Xu）。《Scinet：基于样本卷积和交互的时间序列建模与预测》。《神经信息处理系统进展》（Advances in Neural Information Processing Systems），35:5816 - 5828，2022年。

[13] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In AAAI, pages 11106-11115, 2021.

[13] 周浩毅（Haoyi Zhou）、张上航（Shanghang Zhang）、彭杰奇（Jieqi Peng）、张帅（Shuai Zhang）、李建新（Jianxin Li）、熊辉（Hui Xiong）和张万才（Wancai Zhang）。《Informer：超越高效Transformer的长序列时间序列预测方法》。发表于《AAAI会议论文集》，第11106 - 11115页，2021年。

[14] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In NeurIPS, pages 22419- 22430, 2021.

[14] 吴海旭（Haixu Wu）、徐杰辉（Jiehui Xu）、王建民（Jianmin Wang）和龙明盛（Mingsheng Long）。《Autoformer：用于长期序列预测的具有自相关特性的分解Transformer》。发表于《神经信息处理系统大会（NeurIPS）论文集》，第22419 - 22430页，2021年。

[15] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi Zhang. Connecting the dots: Multivariate time series forecasting with graph neural networks. In ${KDD}$ , pages 753-763, 2020.

[15] 吴宗翰（Zonghan Wu）、潘世瑞（Shirui Pan）、龙国栋（Guodong Long）、蒋静（Jing Jiang）、常小军（Xiaojun Chang）和张成奇（Chengqi Zhang）。《连接数据点：基于图神经网络的多变量时间序列预测》。发表于${KDD}$，第753 - 763页，2020年。

[16] Defu Cao, Yujing Wang, Juanyong Duan, Ce Zhang, Xia Zhu, Congrui Huang, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, and Qi Zhang. Spectral temporal graph neural network for multivariate time-series forecasting. In NeurIPS, 2020.

[16] 曹德富（Defu Cao）、王雨静（Yujing Wang）、段娟勇（Juanyong Duan）、张策（Ce Zhang）、朱霞（Xia Zhu）、黄聪睿（Congrui Huang）、童云海（Yunhai Tong）、徐必熊（Bixiong Xu）、白静（Jing Bai）、童杰（Jie Tong）和张琪（Qi Zhang）。用于多变量时间序列预测的谱时图神经网络。发表于《神经信息处理系统大会》（NeurIPS），2020年。

[17] Lei Bai, Lina Yao, Can Li, Xianzhi Wang, and Can Wang. Adaptive graph convolutional recurrent network for traffic forecasting. In NeurIPS, 2020.

[17] 白磊（Lei Bai）、姚丽娜（Lina Yao）、李灿（Can Li）、王先志（Xianzhi Wang）和王灿（Can Wang）。用于交通预测的自适应图卷积循环网络。发表于《神经信息处理系统大会》（NeurIPS），2020年。

[18] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In ICLR, 2020.

[18] 尼基塔·基塔耶夫（Nikita Kitaev）、卢卡斯·凯泽（Lukasz Kaiser）和安塞尔姆·列夫斯卡亚（Anselm Levskaya）。改革者：高效的Transformer模型。发表于《国际学习表征会议》（ICLR），2020年。

[19] Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basis expansion analysis for interpretable time series forecasting. arXiv preprint arXiv:1905.10437, 2019.

[19] 鲍里斯·N·奥列什金（Boris N Oreshkin）、德米特里·卡尔波夫（Dmitri Carpov）、尼古拉斯·查帕多斯（Nicolas Chapados）和约书亚·本吉奥（Yoshua Bengio）。N - beats：用于可解释时间序列预测的神经基扩展分析。预印本arXiv:1905.10437，2019年。

[20] Tianping Zhang, Yizhuo Zhang, Wei Cao, Jiang Bian, Xiaohan Yi, Shun Zheng, and Jian Li. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186, 2022.

[20] 张天平（Tianping Zhang）、张一卓（Yizhuo Zhang）、曹伟（Wei Cao）、边江（Jiang Bian）、易晓晗（Xiaohan Yi）、郑顺（Shun Zheng）和李建（Jian Li）。少即是多：采用轻采样多层感知器（MLP）结构实现快速多元时间序列预测。预印本 arXiv:2207.01186，2022年。

[21] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? arXiv preprint arXiv:2205.13504, 2022.

[21] 曾爱玲（Ailing Zeng）、陈慕溪（Muxi Chen）、张磊（Lei Zhang）和徐强（Qiang Xu）。Transformer 对时间序列预测有效吗？预印本 arXiv:2205.13504，2022年。

[22] Duraisamy Sundararajan. The discrete Fourier transform: theory, algorithms and applications. World Scientific, 2001.

[22] 杜赖萨米·桑达拉扬（Duraisamy Sundararajan）。离散傅里叶变换：理论、算法与应用。世界科学出版社，2001年。

[23] Mark W. Watson. Vector autoregressions and cointegration. Working Paper Series, Macroeconomic Issues, 4, 1993.

[23] 马克·W·沃森（Mark W. Watson）。向量自回归与协整。工作论文系列，宏观经济问题，第4期，1993年。

[24] Dimitros Asteriou and Stephen G Hall. Arima models and the box-jenkins methodology. Applied Econometrics, 2(2):265-286, 2011.

[24] 迪米特罗斯·阿斯泰里奥（Dimitros Asteriou）和斯蒂芬·G·霍尔（Stephen G Hall）。自回归积分滑动平均（ARIMA）模型与博克斯 - 詹金斯方法。应用计量经济学，第2卷第2期：265 - 286页，2011年。

[25] Bryan Lim and Stefan Zohren. Time-series forecasting with deep learning: a survey. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 379(2194):20200209, feb 2021.

[25] 布莱恩·林（Bryan Lim）和斯特凡·佐伦（Stefan Zohren）。深度学习时间序列预测综述。《英国皇家学会哲学汇刊A：数学、物理与工程科学》，379(2194):20200209，2021年2月。

[26] José Torres, Dalil Hadjout, Abderrazak Sebaa, Francisco Martínez-Álvarez, and Alicia Troncoso. Deep learning for time series forecasting: A survey. Big Data, 9, 122020.

[26] 何塞·托雷斯（José Torres）、达利勒·哈茹特（Dalil Hadjout）、阿卜杜勒拉扎克·塞巴（Abderrazak Sebaa）、弗朗西斯科·马丁内斯 - 阿尔瓦雷斯（Francisco Martínez - Álvarez）和阿莉西亚·特龙科索（Alicia Troncoso）。深度学习时间序列预测综述。《大数据》，9，122020。

[27] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, and Chengqi Zhang. Graph wavenet for deep spatial-temporal graph modeling. In IJCAI, pages 1907-1913, 2019.

[27] 吴宗翰（Zonghan Wu）、潘世瑞（Shirui Pan）、龙国栋（Guodong Long）、蒋静（Jing Jiang）和张成奇（Chengqi Zhang）。用于深度时空图建模的图波网络。见《国际人工智能联合会议论文集》，第1907 - 1913页，2019年。

[28] Kun Yi, Qi Zhang, Longbing Cao, Shoujin Wang, Guodong Long, Liang Hu, Hui He, Zhendong Niu, Wei Fan, and Hui Xiong. A survey on deep learning based time series analysis with frequency transformation. CoRR, abs/2302.02173, 2023.

[28] 易坤（Kun Yi）、张琪（Qi Zhang）、曹龙兵（Longbing Cao）、王首进（Shoujin Wang）、龙国栋（Guodong Long）、胡亮（Liang Hu）、何辉（Hui He）、牛振东（Zhendong Niu）、樊伟（Wei Fan）和熊辉（Hui Xiong）。基于深度学习的频域变换时间序列分析综述。《计算机研究与发展预印本》，abs/2302.02173，2023年。

[29] Liheng Zhang, Charu C. Aggarwal, and Guo-Jun Qi. Stock price prediction via discovering multi-frequency trading patterns. In ${KDD}$ ,pages 2141-2149,2017.

[29] 张立恒（Liheng Zhang）、查鲁·C·阿加瓦尔（Charu C. Aggarwal）和齐国军（Guo-Jun Qi）。通过发现多频率交易模式进行股票价格预测。见${KDD}$，第2141 - 2149页，2017年。

[30] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In ${ICML},{2022}$ .

[30] 周天（Tian Zhou）、马子清（Ziqing Ma）、文青松（Qingsong Wen）、王雪（Xue Wang）、孙亮（Liang Sun）和金榕（Rong Jin）。FEDformer：用于长期序列预测的频率增强分解变压器。见${ICML},{2022}$。

[31] Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven C. H. Hoi. Cost: Contrastive learning of disentangled seasonal-trend representations for time series forecasting. In ICLR. OpenReview.net, 2022.

[31] 杰拉尔德·吴（Gerald Woo）、刘承浩（Chenghao Liu）、多延·萨胡（Doyen Sahoo）、阿克沙特·库马尔（Akshat Kumar）和史蒂文·C·H·霍伊（Steven C. H. Hoi）。Cost：用于时间序列预测的解纠缠季节性 - 趋势表示的对比学习。见国际学习表征会议（ICLR）。OpenReview.net，2022年。

[32] Tian Zhou, Ziqing Ma, Xue Wang, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, and Rong Jin. Film: Frequency improved legendre memory model for long-term time series forecasting. 2022.

[32] 周天（Tian Zhou）、马子清（Ziqing Ma）、王雪（Xue Wang）、文青松（Qingsong Wen）、孙亮（Liang Sun）、姚涛（Tao Yao）、尹沃涛（Wotao Yin）和金榕（Rong Jin）。Film：用于长期时间序列预测的频率改进勒让德记忆模型。2022年。

[33] Wei Fan, Shun Zheng, Xiaohan Yi, Wei Cao, Yanjie Fu, Jiang Bian, and Tie-Yan Liu. DEPTS: deep expansion learning for periodic time series forecasting. In ICLR. OpenReview.net, 2022.

[33] 樊伟（Wei Fan）、郑顺（Shun Zheng）、易晓晗（Xiaohan Yi）、曹伟（Wei Cao）、傅彦杰（Yanjie Fu）、边江（Jiang Bian）和刘铁岩（Tie-Yan Liu）。DEPTS：用于周期性时间序列预测的深度扩展学习。发表于国际学习表征会议（ICLR）。OpenReview.net，2022年。

[34] Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler, and Artur Dubrawski. N-hits: Neural hierarchical interpolation for time series forecasting. CoRR, abs/2201.12886, 2022.

[34] 克里斯蒂安·查卢（Cristian Challu）、金·G·奥利瓦雷斯（Kin G. Olivares）、鲍里斯·N·奥列什金（Boris N. Oreshkin）、费德里科·加尔萨（Federico Garza）、马克斯·默根塔勒（Max Mergenthaler）和阿图尔·杜布拉夫斯基（Artur Dubrawski）。N - hits：用于时间序列预测的神经分层插值法。计算机研究报告（CoRR），编号abs/2201.12886，2022年。

[35] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? 2023.

[35] 曾爱玲（Ailing Zeng）、陈慕溪（Muxi Chen）、张磊（Lei Zhang）和徐强（Qiang Xu）。Transformer模型对时间序列预测有效吗？2023年。

[36] Tomás Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. In ICLR (Workshop Poster), 2013.

[36] 托马斯·米科洛夫（Tomás Mikolov）、陈凯（Kai Chen）、格雷格·科拉多（Greg Corrado）和杰弗里·迪恩（Jeffrey Dean）。向量空间中词表征的高效估计。发表于国际学习表征会议（ICLR）（研讨会海报），2013年。

[37] Rajat Sen, Hsiang-Fu Yu, and Inderjit S. Dhillon. Think globally, act locally: A deep neural network approach to high-dimensional time series forecasting. In NeurIPS, pages 4838-4847, 2019.

[37] 拉贾特·森（Rajat Sen）、余翔富（Hsiang - Fu Yu）和英德吉特·S·狄隆（Inderjit S. Dhillon）。全局思考，局部行动：一种用于高维时间序列预测的深度神经网络方法。发表于神经信息处理系统大会（NeurIPS），第4838 - 4847页，2019年。

[38] Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. In ICLR (Poster), 2018.

[38] 李雅光（Yaguang Li）、于露丝（Rose Yu）、赛勒斯·沙哈比（Cyrus Shahabi）和刘燕（Yan Liu）。扩散卷积循环神经网络：数据驱动的交通流量预测。发表于国际学习表征会议（ICLR）（海报），2018年。

[39] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023.

[39] 聂玉琪（Yuqi Nie）、南·H·阮（Nam H. Nguyen）、潘瓦迪·辛通（Phanwadee Sinthong）和贾扬特·卡拉格纳纳姆（Jayant Kalagnanam）。一个时间序列价值64个字：基于Transformer的长期预测。发表于国际学习表征会议，2023年。

[40] Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In IJCAI, pages 3634-3640, 2018.

[40] 于冰（Bing Yu）、尹浩腾（Haoteng Yin）和朱占星（Zhanxing Zhu）。时空图卷积网络：一种用于交通流量预测的深度学习框架。发表于国际人工智能联合会议（IJCAI），第3634 - 3640页，2018年。

[41] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, pages 8024-8035, 2019.

[41] 亚当·帕斯兹克（Adam Paszke）、山姆·格罗斯（Sam Gross）、弗朗西斯科·马萨（Francisco Massa）、亚当·勒雷尔（Adam Lerer）、詹姆斯·布拉德伯里（James Bradbury）、格雷戈里·查南（Gregory Chanan）、特雷弗·基林（Trevor Killeen）、林泽明（Zeming Lin）、娜塔莉亚·吉梅尔申（Natalia Gimelshein）、卢卡·安蒂加（Luca Antiga）、阿尔班·德斯梅森（Alban Desmaison）、安德里亚斯·科普夫（Andreas Köpf）、杨爱德华（Edward Z. Yang）、扎卡里·德维托（Zachary DeVito）、马丁·赖森（Martin Raison）、阿利汗·特贾尼（Alykhan Tejani）、萨桑克·奇拉姆库尔蒂（Sasank Chilamkurthy）、贝努瓦·施泰纳（Benoit Steiner）、方璐（Lu Fang）、白俊杰（Junjie Bai）和苏米特·钦塔拉（Soumith Chintala）。PyTorch：一种命令式风格的高性能深度学习库。发表于《神经信息处理系统大会论文集》（NeurIPS），第8024 - 8035页，2019年。

## A Notations

## A 符号说明

<!-- Media -->

Table 5: Notation.

表5：符号说明。

${\mathbf{X}}_{t}$ multivariate time series with a lookback window of $L$ at timestamps $\mathfrak{t},{\mathbf{X}}_{t} \in  {\mathbb{R}}^{N \times  L}$

${\mathbf{X}}_{t}$ 在时间戳 $\mathfrak{t},{\mathbf{X}}_{t} \in  {\mathbb{R}}^{N \times  L}$ 处具有回溯窗口 $L$ 的多变量时间序列

${X}_{t}$ the multivariate values of $N$ distinct series at timestamp $t,{X}_{t} \in  {\mathbb{R}}^{N}$ the prediction target with a horizon window of length $\tau$ at timestamps $t,{\mathbf{Y}}_{t} \in  {\mathbb{R}}^{N \times  \tau }$

${X}_{t}$ 时间戳 $t,{X}_{t} \in  {\mathbb{R}}^{N}$ 处 $N$ 个不同序列的多变量值；时间戳 $t,{\mathbf{Y}}_{t} \in  {\mathbb{R}}^{N \times  \tau }$ 处预测目标的长度为 $\tau$ 的预测窗口

${\mathbf{H}}_{t}$ the hidden representation of ${\mathbf{X}}_{t},{\mathbf{H}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$

${\mathbf{H}}_{t}$ ${\mathbf{X}}_{t},{\mathbf{H}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ 的隐藏表示

${\mathbf{Z}}_{t}$ the output of the frequency channel learner, ${\mathbf{Z}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$

${\mathbf{Z}}_{t}$ 频率通道学习器的输出，${\mathbf{Z}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$

${\mathbf{S}}_{t}$ the output of the frequency temporal learner, ${\mathbf{S}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ ${\mathcal{H}}_{\text{chan }}$ the domain conversion of ${\mathbf{H}}_{t}$ on channel dimensions, ${\mathcal{H}}_{\text{chan }} \in  {\mathbb{C}}^{N \times  L \times  d}$ the FreMLP output of ${\mathcal{H}}_{\text{chan }},{\mathcal{Z}}_{\text{chan }} \in  {\mathbb{C}}^{N \times  L \times  d}$ the domain conversion of ${\mathbf{Z}}_{t}$ on temporal dimensions, ${\mathcal{Z}}_{\text{temp }} \in  {\mathbb{C}}^{N \times  L \times  d}$

${\mathbf{S}}_{t}$ 频率时间学习器的输出，${\mathbf{S}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ ${\mathcal{H}}_{\text{chan }}$ ${\mathbf{H}}_{t}$在通道维度上的域转换，${\mathcal{H}}_{\text{chan }} \in  {\mathbb{C}}^{N \times  L \times  d}$ ${\mathcal{H}}_{\text{chan }},{\mathcal{Z}}_{\text{chan }} \in  {\mathbb{C}}^{N \times  L \times  d}$ ${\mathbf{Z}}_{t}$在时间维度上的域转换的频率多层感知器（FreMLP）输出，${\mathcal{Z}}_{\text{temp }} \in  {\mathbb{C}}^{N \times  L \times  d}$

${\mathcal{S}}_{\text{temp }}$ the FreMLP output of ${\mathcal{Z}}_{\text{temp }},{\mathcal{S}}_{\text{temp }} \in  {\mathbb{C}}^{N \times  L \times  d}$ Wchan the complex number weight matrix of FreMLP in the frequency channel learner, ${\mathcal{W}}^{\text{chan }} \in  {\mathbb{C}}^{d \times  d}$ Bchan the complex number bias of FreMLP in the frequency channel learner, ${\mathcal{B}}^{\text{chan }} \in  {\mathbb{C}}^{d}$

${\mathcal{S}}_{\text{temp }}$ ${\mathcal{Z}}_{\text{temp }},{\mathcal{S}}_{\text{temp }} \in  {\mathbb{C}}^{N \times  L \times  d}$的FreMLP输出 Wchan（频率通道学习器中FreMLP的复数权重矩阵），${\mathcal{W}}^{\text{chan }} \in  {\mathbb{C}}^{d \times  d}$ Bchan（频率通道学习器中FreMLP的复数偏置），${\mathcal{B}}^{\text{chan }} \in  {\mathbb{C}}^{d}$

Wtemp the complex number weight matrix of FreMLP in the frequency temporal learner, ${\mathcal{W}}^{\text{temp }} \in  {\mathbb{C}}^{d \times  d}$

Wtemp为频率时间学习器中频率多层感知器（FreMLP）的复数权重矩阵，${\mathcal{W}}^{\text{temp }} \in  {\mathbb{C}}^{d \times  d}$

Btemp the complex number bias of FreMLP in the frequency temporal learner, ${\mathcal{B}}^{\text{temp }} \in  {\mathbb{C}}^{d}$

Btemp为频率时间学习器中频率多层感知器（FreMLP）的复数偏置，${\mathcal{B}}^{\text{temp }} \in  {\mathbb{C}}^{d}$

<!-- Media -->

## B Experimental Details

## B 实验细节

### B.1 Datasets

### B.1 数据集

We adopt thirteen real-world benchmarks in the experiments to evaluate the accuracy of short-term and long-term forecasting. The details of the datasets are as follows:

我们在实验中采用了十三个真实世界的基准数据集，以评估短期和长期预测的准确性。数据集的详细信息如下：

Solar ${}^{4}$ . It is about the solar power collected by National Renewable Energy Laboratory. We choose the power plant data points in Florida as the data set which contains 593 points. The data is collected from 01/01/2006 to 31/12/2016 with the sampling interval of every 1 hour.

太阳能 ${}^{4}$。该数据集与美国国家可再生能源实验室（National Renewable Energy Laboratory）收集的太阳能发电数据有关。我们选择佛罗里达州（Florida）的发电厂数据点作为数据集，其中包含593个数据点。数据收集时间为2006年1月1日至2016年12月31日，采样间隔为每1小时一次。

Wiki [37]: It contains a number of daily views of different Wikipedia articles and is collected from $1/7/{2015}$ to ${31}/{12}/{2016}$ . It consists of approximately ${145k}$ time series and we randomly choose ${5k}$ from them as our experimental data set.

维基百科数据集[37]：它包含了不同维基百科文章的每日浏览量数据，数据收集时间从$1/7/{2015}$到${31}/{12}/{2016}$。该数据集大约由${145k}$个时间序列组成，我们从中随机选取了${5k}$个作为实验数据集。

Traffic [37]: It contains hourly traffic data from 963 San Francisco freeway car lanes for short-term forecasting settings while it contains 862 car lanes for long-term forecasting. It is collected since 01/01/2015 with a sampling interval of every 1 hour.

交通数据集[37]：在短期预测场景下，它包含了旧金山963条高速公路车道的每小时交通数据；在长期预测场景下，包含862条车道的数据。数据从2015年1月1日开始收集，采样间隔为每小时一次。

ECG ${}^{5}$ : It is about Electrocardiogram(ECG) from the UCR time-series classification archive. It contains 140 nodes and each node has a length of 5000 .

心电图数据集${}^{5}$：该数据集来自加州大学河滨分校（UCR）时间序列分类存档库，包含心电图（ECG）数据。它有140个节点，每个节点的长度为5000。

---

<!-- Footnote -->

https://www.nrel.gov/grid/solar-power-data.html

shtp://www.timeseriesclassification.com/description.php?Dataset=ECG5000

shtp://www.timeseriesclassification.com/description.php?Dataset=ECG5000

<!-- Footnote -->

---

Electricity ${}^{6}$ : It contains electricity consumption of 370 clients for short-term forecasting while it contains electricity consumption of 321 clients for long-term forecasting. It is collected since 01/01/2011. The data sampling interval is every 15 minutes.

电力数据集${}^{6}$：在短期预测中，它包含370个客户的电力消耗数据；在长期预测中，包含321个客户的电力消耗数据。数据从2011年1月1日开始收集，采样间隔为每15分钟一次。

COVID-19 [4]: It is about COVID-19 hospitalization in the U.S. state of California (CA) from 01/02/2020 to 31/12/2020 provided by the Johns Hopkins University with the sampling interval of every day.

新冠疫情（COVID - 19）[4]：该数据是约翰·霍普金斯大学（Johns Hopkins University）提供的2020年2月1日至2020年12月31日美国加利福尼亚州（CA）的新冠住院情况，采样间隔为每天一次。

METR-LA7: It contains traffic information collected from loop detectors in the highway of Los Angeles County. It contains 207 sensors which are from 01/03/2012 to 30/06/2012 and the data sampling interval is every 5 minutes.

洛杉矶县交通数据（METR - LA7）：该数据包含从洛杉矶县高速公路环形探测器收集的交通信息。它涵盖了207个传感器的数据，时间范围为2012年3月1日至2012年6月30日，数据采样间隔为每5分钟一次。

Exchange ${}^{8}$ . It contains the collection of the daily exchange rates of eight foreign countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand, and Singapore ranging from 1990 to 2016 and the data sampling interval is every 1 day.

汇率数据 ${}^{8}$：该数据包含1990年至2016年澳大利亚、英国、加拿大、瑞士、中国、日本、新西兰和新加坡这八个国家的每日汇率信息，数据采样间隔为每天一次。

Weather ${}^{9}$ . It collects 21 meteorological indicators,such as humidity and air temperature,from the Weather Station of the Max Planck Biogeochemistry Institute in Germany in 2020. The data sampling interval is every 10 minutes.

气象数据 ${}^{9}$：该数据收集了2020年德国马克斯·普朗克生物地球化学研究所（Max Planck Biogeochemistry Institute）气象站的21项气象指标，如湿度和气温等。数据采样间隔为每10分钟一次。

ETT ${}^{10}$ . It is collected from two different electric transformers labeled with 1 and 2,and each of them contains 2 different resolutions ( 15 minutes and 1 hour) denoted with $\mathrm{m}$ and $\mathrm{h}$ . We use ETTh1 and ETTm1 as our long-term forecasting benchmarks.

电力变压器时间序列数据（ETT） ${}^{10}$ 。它从标记为1和2的两个不同电力变压器（Electric Transformer）中收集，每个变压器包含两种不同分辨率（15分钟和1小时），分别用 $\mathrm{m}$ 和 $\mathrm{h}$ 表示。我们使用ETTh1和ETTm1作为长期预测基准。

### B.2 Baselines

### B.2 基线模型

We adopt eighteen representative and state-of-the-art baselines for comparison including LSTM-based models, GNN-based models, and Transformer-based models. We introduce these models as follows:

我们采用了18种具有代表性的先进基线模型进行比较，包括基于长短期记忆网络（LSTM）的模型、基于图神经网络（GNN）的模型和基于Transformer的模型。我们对这些模型介绍如下：

VAR [23]: VAR is a classic linear autoregressive model. We use the Statsmodels library (https: //www.statsmodels.org) which is a Python package that provides statistical computations to realize the VAR.

向量自回归模型（VAR） [23]：VAR是一种经典的线性自回归模型。我们使用Statsmodels库（https://www.statsmodels.org）来实现VAR，这是一个提供统计计算功能的Python包。

DeepGLO [37]: DeepGLO models the relationships among variables by matrix factorization and employs a temporal convolution neural network to introduce non-linear relationships. We download the source code from: https://github.com/rajatsen91/deepglo.We use the recommended configuration as our experimental settings for Wiki, Electricity, and Traffic datasets. For the COVID- 19 dataset, the vertical and horizontal batch size is set to 64, the rank of the global model is set to 64, the number of channels is set to $\left\lbrack  {{32},{32},{32},1}\right\rbrack$ ,and the period is set to 7 .

DeepGLO [37]：DeepGLO（深度全局学习模型）通过矩阵分解对变量之间的关系进行建模，并采用时间卷积神经网络引入非线性关系。我们从以下网址下载源代码：https://github.com/rajatsen91/deepglo。我们将推荐配置作为维基百科（Wiki）、电力（Electricity）和交通（Traffic）数据集的实验设置。对于新冠疫情（COVID - 19）数据集，垂直和水平批量大小设置为64，全局模型的秩设置为64，通道数设置为$\left\lbrack  {{32},{32},{32},1}\right\rbrack$，周期设置为7。

LSTNet [10]: LSTNet uses a CNN to capture inter-variable relationships and an RNN to discover long-term patterns. We download the source code from: https://github.com/laiguokun/ LSTNet. In our experiment, we use the recommended configuration where the number of CNN hidden units is 100, the kernel size of the CNN layers is 4, the dropout is 0.2, the RNN hidden units is 100, the number of RNN hidden layers is 1, the learning rate is 0.001 and the optimizer is Adam.

LSTNet [10]：LSTNet使用卷积神经网络（CNN）来捕捉变量间的关系，并使用循环神经网络（RNN）来发现长期模式。我们从以下网址下载源代码：https://github.com/laiguokun/ LSTNet。在我们的实验中，我们使用推荐的配置，其中卷积神经网络的隐藏单元数量为100，卷积神经网络层的核大小为4，丢弃率为0.2，循环神经网络的隐藏单元数量为100，循环神经网络的隐藏层数为1，学习率为0.001，优化器为Adam。

TCN [11]: TCN is a causal convolution model for regression prediction. We download the source code from: https://github.com/locuslab/TCN.We utilize the same configuration as the polyphonic music task exampled in the open source code where the dropout is 0.25 , the kernel size is 5 , the number of hidden units is 150 , the number of levels is 4 and the optimizer is Adam.

TCN [11]：TCN是一种用于回归预测的因果卷积模型。我们从以下网址下载源代码：https://github.com/locuslab/TCN。我们采用与开源代码中复调音乐任务示例相同的配置，其中丢弃率为0.25，核大小为5，隐藏单元数量为150，层数为4，优化器为Adam。

Informer [13]: Informer leverages an efficient self-attention mechanism to encode the dependencies among variables. We download the source code from: https://github.com/zhouhaoyi/ Informer2020. We use the recommended configuration as the experimental settings where the dropout is 0.05 , the number of encoder layers is 2 , the number of decoder layers is 1 , the learning rate is 0.0001 , and the optimizer is Adam.

Informer [13]：Informer（信息者模型）利用一种高效的自注意力机制对变量之间的依赖关系进行编码。我们从以下链接下载源代码：https://github.com/zhouhaoyi/ Informer2020。我们采用推荐的配置作为实验设置，其中丢弃率（dropout）为0.05，编码器层数为2，解码器层数为1，学习率为0.0001，优化器为Adam。

Reformer [18]: Reformer combines the modeling capacity of a Transformer with an architecture that can be executed efficiently on long sequences and with small memory use. We download the source code from: https://github.com/thuml/Autoformer.We use the recommended configuration as the experimental settings.

Reformer [18]：Reformer（改革者模型）将Transformer（变换器模型）的建模能力与一种能够在长序列上高效执行且内存使用量小的架构相结合。我们从以下链接下载源代码：https://github.com/thuml/Autoformer。我们采用推荐的配置作为实验设置。

---

<!-- Footnote -->

${}^{6}$ https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

${}^{6}$ https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

https://github.com/liyaguang/DCRNN

https://github.com/laiguokun/multivariate-time-series-data

https://www.bgc-jena.mpg.de/wetter/

10 https://github.com/zhouhaoyi/ETDataset

10 https://github.com/zhouhaoyi/ETDataset

<!-- Footnote -->

---

Autoformer [14]: Autoformer proposes a decomposition architecture by embedding the series decomposition block as an inner operator, which can progressively aggregate the long-term trend part from intermediate prediction. We download the source code from: https://github.com/thuml/ Autoformer. We use the recommended configuration as the experimental settings.

Autoformer [14]：Autoformer（自动变换器模型）通过将序列分解块嵌入为内部算子来提出一种分解架构，该架构可以从中间预测中逐步聚合长期趋势部分。我们从以下链接下载源代码：https://github.com/thuml/ Autoformer。我们采用推荐的配置作为实验设置。

FEDformer [30]: FEDformer proposes an attention mechanism with low-rank approximation in frequency and a mixture of expert decomposition to control the distribution shifting. We download the source code from: https://github.com/MAZiqing/FEDformer We use FEB-f as the Frequency Enhanced Block and select the random mode with 64 as the experimental mode.

FEDformer [30]：FEDformer（频率增强分解变换器）提出了一种在频域进行低秩近似的注意力机制，以及一种专家混合分解方法来控制分布偏移。我们从以下链接下载源代码：https://github.com/MAZiqing/FEDformer 我们使用FEB - f作为频率增强模块，并选择随机模式，以64作为实验模式。

SFM [29]: On the basis of the LSTM model, SFM introduces a series of different frequency components in the cell states. We download the source code from: https://github.com/z331565360/ State-Frequency-Memory-stock-prediction We follow the recommended configuration as the experimental settings where the learning rate is 0.01 , the frequency dimension is 10 , the hidden dimension is 10 and the optimizer is RMSProp.

SFM [29]：SFM（状态频率记忆模型）在长短期记忆网络（LSTM）模型的基础上，在单元状态中引入了一系列不同的频率分量。我们从以下链接下载源代码：https://github.com/z331565360/ State - Frequency - Memory - stock - prediction 我们按照推荐配置进行实验设置，其中学习率为0.01，频率维度为10，隐藏维度为10，优化器为均方根传播（RMSProp）。

StemGNN [16]: StemGNN leverages GFT and DFT to capture dependencies among variables in the frequency domain. We download the source code from: https://github.com/microsoft/ StemGNN. We use the recommended configuration of stemGNN as our experiment setting where the optimizer is RMSProp, the learning rate is 0.0001 , the number of stacked layers is 5 , and the dropout rate is 0.5 .

StemGNN [16]：StemGNN利用图傅里叶变换（GFT）和离散傅里叶变换（DFT）在频域中捕捉变量之间的依赖关系。我们从以下网址下载源代码：https://github.com/microsoft/StemGNN。我们使用StemGNN推荐的配置作为实验设置，其中优化器为RMSProp，学习率为0.0001，堆叠层数为5，丢弃率为0.5。

MTGNN [15]: MTGNN proposes an effective method to exploit the inherent dependency relationships among multiple time series. We download the source code from: https://github.com/ nnzhan/MTGNN. Because the experimental datasets have no static features, we set the parameter load_static_feature to false. We construct the graph by the adaptive adjacency matrix and add the graph convolution layer. Regarding other parameters, we follow the recommended settings.

多变量时间序列图神经网络（MTGNN）[15]：MTGNN提出了一种有效的方法来挖掘多个时间序列之间的内在依赖关系。我们从以下网址下载源代码：https://github.com/ nnzhan/MTGNN。由于实验数据集没有静态特征，我们将参数“load_static_feature”设置为“false”。我们通过自适应邻接矩阵构建图，并添加图卷积层。对于其他参数，我们遵循推荐设置。

GraphWaveNet [27]: GraphWaveNet introduces an adaptive dependency matrix learning to capture the hidden spatial dependency. We download the source code from: https://github.com/ nnzhan/Graph-WaveNet. Since our datasets have no prior defined graph structures, we use only adaptive adjacent matrix. We add a graph convolutional layer and randomly initialize the adjacent matrix. We adopt the recommended setting as its experimental configuration where the learning rate is 0.001 , the dropout is 0.3 , the number of epochs is 50 , and the optimizer is Adam.

图波网络（GraphWaveNet）[27]：GraphWaveNet引入了自适应依赖矩阵学习来捕捉隐藏的空间依赖关系。我们从以下网址下载源代码：https://github.com/ nnzhan/Graph-WaveNet。由于我们的数据集没有预先定义的图结构，我们仅使用自适应邻接矩阵。我们添加一个图卷积层并随机初始化邻接矩阵。我们采用推荐设置作为其实验配置，其中学习率为0.001，丢弃率为0.3，训练轮数为50，优化器为Adam。

AGCRN [17]: AGCRN proposes a data-adaptive graph generation module for discovering spatial correlations from data. We download the source code from: https://github.com/LeiBAI/AGCRN.We follow the recommended settings where the embedding dimension is 10 , the learning rate is 0.003 , and the optimizer is Adam.

自适应图卷积循环网络（AGCRN）[17]：自适应图卷积循环网络（AGCRN）提出了一个数据自适应图生成模块，用于从数据中发现空间相关性。我们从以下链接下载源代码：https://github.com/LeiBAI/AGCRN。我们遵循推荐设置，其中嵌入维度为10，学习率为0.003，优化器为Adam。

TAMP-S2GCNets [4]: TAMP-S2GCNets explores the utility of MP to enhance knowledge representation mechanisms within the time-aware DL paradigm. We download the source code from: https: //www.dropbox.com/sh/n0ajd510tdeyb80/AABGn-ejfV1YtRwjf_LOAOsNa?dl=0 TAMP-S2GCNets require a pre-defined graph topology and we use the California State topology provided by the source code as input. We adopt the recommended settings as the experimental configuration for COVID-19.

基于时间感知注意力机制的时空图卷积网络（TAMP - S2GCNets）[4]：基于时间感知注意力机制的时空图卷积网络（TAMP - S2GCNets）探索了消息传递（MP）在时间感知深度学习范式中增强知识表示机制的效用。我们从以下链接下载源代码：https: //www.dropbox.com/sh/n0ajd510tdeyb80/AABGn - ejfV1YtRwjf_LOAOsNa?dl=0 基于时间感知注意力机制的时空图卷积网络（TAMP - S2GCNets）需要一个预定义的图拓扑结构，我们使用源代码提供的加利福尼亚州拓扑结构作为输入。我们采用推荐设置作为新冠肺炎（COVID - 19）的实验配置。

DCRNN [38]: DCRNN uses bidirectional graph random walk to model spatial dependency and recurrent neural network to capture the temporal dynamics. We download the source code from: https://github.com/liyaguang/DCRNN.We use the recommended configuration as our experimental settings with the batch size is 64 , the learning rate is 0.01 , the input dimension is 2 and the optimizer is Adam. DCRNN requires a pre-defined graph structure and we use the adjacency matrix as the pre-defined structure provided by the METR-LA dataset.

动态卷积循环神经网络（DCRNN）[38]：DCRNN使用双向图随机游走对空间依赖性进行建模，并使用循环神经网络捕捉时间动态。我们从以下链接下载源代码：https://github.com/liyaguang/DCRNN。我们采用推荐的配置作为实验设置，批量大小为64，学习率为0.01，输入维度为2，优化器为Adam。DCRNN需要一个预定义的图结构，我们使用METR - LA数据集提供的邻接矩阵作为预定义结构。

STGCN [40]: STGCN integrates graph convolution and gated temporal convolution through spatial-temporal convolutional blocks. We download the source code from: https://github.com/ VeritasYin/STGCN_IJCAI-18. We follow the recommended settings as our experimental configuration where the batch size is 50 , the learning rate is 0.001 and the optimizer is Adam. STGCN requires a pre-defined graph structure and we leverage the adjacency matrix as the pre-defined structure provided by the METR-LA dataset.

时空图卷积网络（STGCN）[40]：STGCN通过时空卷积块将图卷积和门控时间卷积相结合。我们从以下链接下载源代码：https://github.com/ VeritasYin/STGCN_IJCAI - 18。我们遵循推荐的设置作为实验配置，其中批量大小为50，学习率为0.001，优化器为Adam。STGCN需要一个预定义的图结构，我们利用METR - LA数据集提供的邻接矩阵作为预定义结构。

LTSF-Linear [35]: LTSF-Linear proposes a set of embarrassingly simple one-layer linear models to learn temporal relationships between input and output sequences. We download the source code from: https://github.com/cure-lab/LTSF-Linear.We use it as our long-term forecasting baseline and follow the recommended settings as experimental configuration.

LTSF-Linear [35]：LTSF-Linear提出了一组极其简单的单层线性模型，用于学习输入和输出序列之间的时间关系。我们从以下链接下载源代码：https://github.com/cure-lab/LTSF-Linear。我们将其用作长期预测的基线模型，并按照推荐设置进行实验配置。

PatchTST [39]: PatchTST proposes an effective design of Transformer-based models for time series forecasting tasks by introducing two key components: patching and channel-independent structure. We download the source code from: https://github.com/PatchTST We use it as our long-term forecasting baseline and adhere to the recommended settings as the experimental configuration.

PatchTST [39]：PatchTST通过引入两个关键组件——分块（patching）和通道独立结构，为时间序列预测任务提出了一种基于Transformer的有效模型设计。我们从以下链接下载源代码：https://github.com/PatchTST。我们将其用作长期预测的基线模型，并遵循推荐设置进行实验配置。

### B.3 Implementation Details

### B.3 实现细节

By default, both the frequency channel and temporal learners contain one layer of FreMLP with the embedding size $d$ of 128,and the hidden size ${d}_{h}$ is set to 256 . For short-term forecasting,the batch size is set to 32 for Solar, METR-LA, ECG, COVID-19, and Electricity datasets. And for Wiki and Traffic datasets, the batch size is set to 4 . For the long-term forecasting, except for the lookback window size, we follow most of the experimental settings of LTSF-Linear [35]. The lookback window size is set to 96 which is recommended by FEDformer [30] and Autoformer [14]. In Appendix F.2 we also use 192 and 336 as the lookback window size to conduct experiments and the results demonstrate that FreTS outperforms other baselines as well. For the longer prediction lengths (e.g., 336, 720), we use the channel independence strategy and contain only the frequency temporal learner in our model. For some datasets, we carefully tune the hyperparameters including the batch size and learning rate on the validation set, and we choose the settings with the best performance. We tune the batch size over $\{ 4,8,{16},{32}\}$ .

默认情况下，频率通道学习器和时间学习器均包含一层嵌入维度 $d$ 为 128 的频率多层感知机（FreMLP），隐藏层维度 ${d}_{h}$ 设置为 256。对于短期预测，Solar、METR - LA、心电图（ECG）、新冠疫情（COVID - 19）和电力（Electricity）数据集的批量大小（batch size）设置为 32。对于维基百科（Wiki）和交通（Traffic）数据集，批量大小设置为 4。对于长期预测，除了回溯窗口大小外，我们遵循了长序列时间序列预测线性模型（LTSF - Linear）[35] 的大部分实验设置。回溯窗口大小设置为 96，这是联邦变换器（FEDformer）[30] 和自动变换器（Autoformer）[14] 所推荐的。在附录 F.2 中，我们还使用 192 和 336 作为回溯窗口大小进行实验，结果表明频率时间序列预测模型（FreTS）同样优于其他基线模型。对于更长的预测长度（例如 336、720），我们采用通道独立策略，并且模型中仅包含频率时间学习器。对于一些数据集，我们在验证集上仔细调整包括批量大小和学习率在内的超参数，并选择性能最佳的设置。我们在 $\{ 4,8,{16},{32}\}$ 范围内调整批量大小。

### B.4 Visualization Settings

### B.4 可视化设置

The Visualization Method for Global View. We follow the visualization methods in LTSF-Linear [35] to visualize the weights learned in the time domain on the input (corresponding to the left side of Figure 1(a)). For the visualization of the weights learned on the frequency spectrum, we first transform the input into the frequency domain and select the real part of the input frequency spectrum to replace the original input. Then, we learn the weights and visualize them in the same manner as in the time domain. The right side of Figure 1(a) shows the weights learned on the Traffic dataset with a lookback window of 96 and a prediction length of 96 , Figure 9 displays the weights learned on the Traffic dataset with a lookback window of 72 and a prediction length of 336, and Figure 10 is the weights learned on the Electricity dataset with a lookback window of 96 and a prediction length of 96 .

全局视图的可视化方法。我们采用LTSF - Linear [35]中的可视化方法，对在时域中学习到的输入权重进行可视化（对应图1(a)的左侧）。对于在频谱上学习到的权重的可视化，我们首先将输入转换到频域，然后选择输入频谱的实部来替换原始输入。接着，我们学习权重并以与时域相同的方式对其进行可视化。图1(a)的右侧展示了在回溯窗口为96、预测长度为96的交通数据集上学习到的权重，图9展示了在回溯窗口为72、预测长度为336的交通数据集上学习到的权重，图10展示了在回溯窗口为96、预测长度为96的电力数据集上学习到的权重。

The Visualization Method for Energy Compaction. Since the learned weights $\mathcal{W} = {\mathcal{W}}_{r} + j{\mathcal{W}}_{i} \in$ ${\mathbb{C}}^{d \times  d}$ of the frequency-domain MLPs are complex numbers,we visualize the corresponding real part ${\mathcal{W}}_{r}$ and imaginary part ${\mathcal{W}}_{i}$ ,respectively. We normalize them by the calculation of $1/\max \left( \mathcal{W}\right)  * \mathcal{W}$ and visualize the normalization values. The right side of Figure 1(b) is the real part of $\mathcal{W}$ learned on the Traffic dataset with a lookback window of 48 and a prediction length of 192. To visualize the corresponding weights learned in the time domain, we replace the frequency spectrum of input ${\mathcal{Z}}_{\text{temp }} \in  {\mathbb{C}}^{N \times  L \times  d}$ with the original time domain input ${\mathbf{H}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$ and perform calculations in the time domain with a weight $W \in  {\mathbb{R}}^{d \times  d}$ ,as depicted in the left side of Figure 1(b).

能量压缩的可视化方法。由于频域多层感知器（MLPs）学习到的权重 $\mathcal{W} = {\mathcal{W}}_{r} + j{\mathcal{W}}_{i} \in$ ${\mathbb{C}}^{d \times  d}$ 是复数，我们分别对相应的实部 ${\mathcal{W}}_{r}$ 和虚部 ${\mathcal{W}}_{i}$ 进行可视化。我们通过计算 $1/\max \left( \mathcal{W}\right)  * \mathcal{W}$ 对它们进行归一化处理，并可视化归一化后的值。图1（b）的右侧是在交通数据集上学习到的 $\mathcal{W}$ 的实部，回溯窗口为48，预测长度为192。为了可视化在时域中学习到的相应权重，我们将输入 ${\mathcal{Z}}_{\text{temp }} \in  {\mathbb{C}}^{N \times  L \times  d}$ 的频谱替换为原始时域输入 ${\mathbf{H}}_{t} \in  {\mathbb{R}}^{N \times  L \times  d}$，并使用权重 $W \in  {\mathbb{R}}^{d \times  d}$ 在时域中进行计算，如图1（b）左侧所示。

### B.5 Ablation Experimental Settings

### B.5 消融实验设置

DLinear decomposes a raw data input into a trend component and a seasonal component, and two one-layer linear layers are applied to each component. In the ablation study part, we replace the two linear layers with two different frequency-domain MLPs (corresponding to DLinear (FreMLP) in Table 4, and compare their accuracy using the same experimental settings recommended in LTSF-Linear [35]. NLinear subtracts the input by the last value of the sequence. Then, the input goes through a linear layer, and the subtracted part is added back before making the final prediction. We replace the linear layer with a frequency-domain MLP (corresponding to NLinear (FreMLP) in Table 4), and compare their accuracy using the same experimental settings recommended in LTSF-Linear [35].

DLinear将原始数据输入分解为趋势分量和季节性分量，并对每个分量应用两个单层线性层。在消融研究部分，我们用两个不同的频域多层感知机（MLP）替换这两个线性层（对应表4中的DLinear (FreMLP)），并使用LTSF-Linear [35]中推荐的相同实验设置比较它们的准确性。NLinear从输入中减去序列的最后一个值。然后，输入经过一个线性层，并在进行最终预测之前将减去的部分加回。我们用一个频域MLP替换线性层（对应表4中的NLinear (FreMLP)），并使用LTSF-Linear [35]中推荐的相同实验设置比较它们的准确性。

## C Complex Multiplication

## C 复数乘法

For two complex number values ${\mathcal{Z}}_{1} = \left( {a + {jb}}\right)$ and ${\mathcal{Z}}_{2} = \left( {c + {jd}}\right)$ ,where $a$ and $c$ is the real part of ${\mathcal{Z}}_{1}$ and ${\mathcal{Z}}_{2}$ respectively, $b$ and $d$ is the imaginary part of ${\mathcal{Z}}_{1}$ and ${\mathcal{Z}}_{2}$ respectively. Then the multiplication of ${\mathcal{Z}}_{1}$ and ${\mathcal{Z}}_{2}$ is calculated by:

对于两个复数 ${\mathcal{Z}}_{1} = \left( {a + {jb}}\right)$ 和 ${\mathcal{Z}}_{2} = \left( {c + {jd}}\right)$，其中 $a$ 和 $c$ 分别是 ${\mathcal{Z}}_{1}$ 和 ${\mathcal{Z}}_{2}$ 的实部，$b$ 和 $d$ 分别是 ${\mathcal{Z}}_{1}$ 和 ${\mathcal{Z}}_{2}$ 的虚部。那么 ${\mathcal{Z}}_{1}$ 和 ${\mathcal{Z}}_{2}$ 的乘法计算如下：

$$
{\mathcal{Z}}_{1}{\mathcal{Z}}_{2} = \left( {a + {jb}}\right) \left( {c + {jd}}\right)  = {ac} + {j}^{2}{bd} + {jad} + {jbc} = \left( {{ac} - {bd}}\right)  + j\left( {{ad} + {bc}}\right)  \tag{10}
$$

where ${j}^{2} =  - 1$ .

其中 ${j}^{2} =  - 1$。

## D Proof

## D 证明

### D.1 Proof of Theorem 1

### D.1 定理 1 的证明

Theorem 1. Suppose that $\mathbf{H}$ is the representation of raw time series and $\mathcal{H}$ is the corresponding frequency components of the spectrum, then the energy of a time series in the time domain is equal to the energy of its representation in the frequency domain. Formally, we can express this with above notations by:

定理1. 假设$\mathbf{H}$是原始时间序列的表示，$\mathcal{H}$是频谱的相应频率分量，那么时间序列在时域中的能量等于其在频域中表示的能量。形式上，我们可以用上述符号表示为：

$$
{\int }_{-\infty }^{\infty }{\left| \mathbf{H}\left( v\right) \right| }^{2}\mathrm{\;d}v = {\int }_{-\infty }^{\infty }{\left| \mathcal{H}\left( f\right) \right| }^{2}\mathrm{\;d}f \tag{11}
$$

where $\mathcal{H}\left( f\right)  = {\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) {e}^{-{j2\pi fv}}\mathrm{\;d}v,v$ is the time/channel dimension, $f$ is the frequency dimension.

其中$\mathcal{H}\left( f\right)  = {\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) {e}^{-{j2\pi fv}}\mathrm{\;d}v,v$是时间/通道维度，$f$是频率维度。

Proof. Given the representation of raw time series $\mathbf{H} \in  {\mathbb{R}}^{N \times  L \times  d}$ ,let us consider performing integration in either the $N$ dimension (channel dimension) or the $L$ dimension (temporal dimension), denoted as the integral over $v$ ,then

证明。给定原始时间序列$\mathbf{H} \in  {\mathbb{R}}^{N \times  L \times  d}$的表示，让我们考虑在$N$维度（通道维度）或$L$维度（时间维度）上进行积分，记为对$v$的积分，则

$$
{\int }_{-\infty }^{\infty }{\left| \mathbf{H}\left( v\right) \right| }^{2}\mathrm{\;d}v = {\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) {\mathbf{H}}^{ * }\left( v\right) \mathrm{d}v
$$

where ${\mathbf{H}}^{ * }\left( v\right)$ is the conjugate of $\mathbf{H}\left( v\right)$ . According to IDFT, ${\mathbf{H}}^{ * }\left( v\right)  = {\int }_{-\infty }^{\infty }{\mathcal{H}}^{ * }\left( f\right) {e}^{-{j2\pi fv}}\mathrm{\;d}f$ ,we can obtain

其中 ${\mathbf{H}}^{ * }\left( v\right)$ 是 $\mathbf{H}\left( v\right)$ 的共轭。根据离散傅里叶逆变换（IDFT） ${\mathbf{H}}^{ * }\left( v\right)  = {\int }_{-\infty }^{\infty }{\mathcal{H}}^{ * }\left( f\right) {e}^{-{j2\pi fv}}\mathrm{\;d}f$ ，我们可以得到

$$
{\int }_{-\infty }^{\infty }{\left| \mathbf{H}\left( v\right) \right| }^{2}\mathrm{\;d}v = {\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) \left\lbrack  {{\int }_{-\infty }^{\infty }{\mathcal{H}}^{ * }\left( f\right) {e}^{-{j2\pi fv}}\mathrm{\;d}f}\right\rbrack  \mathrm{d}v
$$

$$
 = {\int }_{-\infty }^{\infty }{\mathcal{H}}^{ * }\left( f\right) \left\lbrack  {{\int }_{-\infty }^{\infty }\mathbf{H}\left( v\right) {e}^{-{j2\pi fv}}\mathrm{\;d}v}\right\rbrack  \mathrm{d}f
$$

$$
 = {\int }_{-\infty }^{\infty }{\mathcal{H}}^{ * }\left( f\right) \mathcal{H}\left( f\right) \mathrm{d}f
$$

$$
 = {\int }_{-\infty }^{\infty }{\left| \mathcal{H}\left( f\right) \right| }^{2}\mathrm{\;d}f
$$

Proved.

证明完毕。

Therefore, the energy of a time series in the time domain is equal to the energy of its representation in the frequency domain.

因此，一个时间序列在时域中的能量等于其在频域中表示的能量。

### D.2 Proof of Theorem 2

### D.2 定理2的证明

Theorem 2. Given the time series input $\mathbf{H}$ and its corresponding frequency domain conversion $\mathcal{H}$ , the operations of frequency-domain MLP on $\mathcal{H}$ can be represented as global convolutions on $\mathbf{H}$ in the time domain. This can be given by:

定理2。给定时间序列输入 $\mathbf{H}$ 及其对应的频域转换 $\mathcal{H}$ ，频域多层感知器（MLP）对 $\mathcal{H}$ 的操作可以表示为时域中对 $\mathbf{H}$ 的全局卷积。这可以表示为：

$$
\mathcal{H}\mathcal{W} + \mathcal{B} = \mathcal{F}\left( {\mathbf{H} * W + B}\right)  \tag{12}
$$

where $*$ is a circular convolution, $\mathcal{W}$ and $\mathcal{B}$ are the complex number weight and bias, $W$ and $B$ are the weight and bias in the time domain,and $\mathcal{F}$ is DFT.

其中 $*$ 是循环卷积（circular convolution），$\mathcal{W}$ 和 $\mathcal{B}$ 分别是复数权重（complex number weight）和偏置（bias），$W$ 和 $B$ 分别是时域中的权重（weight）和偏置（bias），$\mathcal{F}$ 是离散傅里叶变换（DFT，Discrete Fourier Transform）。

Proof. Suppose that we conduct operations in the $N$ (i.e.,channel dimension) or $L$ (i.e.,temporal dimension) dimension, then

证明。假设我们在 $N$（即通道维度，channel dimension）或 $L$（即时间维度，temporal dimension）维度上进行运算，那么

$$
\mathcal{F}\left( {\mathbf{H}\left( v\right)  * W\left( v\right) }\right)  = {\int }_{-\infty }^{\infty }\left( {\mathbf{H}\left( v\right)  * W\left( v\right) }\right) {e}^{-{j2\pi fv}}\mathrm{\;d}v
$$

According to convolution theorem, $\mathbf{H}\left( v\right)  * W\left( v\right)  = {\int }_{-\infty }^{\infty }\left( {\mathbf{H}\left( \tau \right) W\left( {v - \tau }\right) }\right) \mathrm{d}\tau$ ,then

根据卷积定理，$\mathbf{H}\left( v\right)  * W\left( v\right)  = {\int }_{-\infty }^{\infty }\left( {\mathbf{H}\left( \tau \right) W\left( {v - \tau }\right) }\right) \mathrm{d}\tau$ ，那么

$$
\mathcal{F}\left( {\mathbf{H}\left( v\right)  * W\left( v\right) }\right)  = {\int }_{-\infty }^{\infty }{\int }_{-\infty }^{\infty }\left( {\mathbf{H}\left( \tau \right) W\left( {v - \tau }\right) }\right) {e}^{-{j2\pi fv}}\mathrm{\;d}\tau \mathrm{d}v
$$

$$
 = {\int }_{-\infty }^{\infty }{\int }_{-\infty }^{\infty }W\left( {v - \tau }\right) {e}^{-{j2\pi fv}}\mathrm{\;d}v\mathbf{H}\left( \tau \right) \mathrm{d}\tau 
$$

Let $x = v - \tau$ ,then

设 $x = v - \tau$ ，那么

$$
\mathcal{F}\left( {\mathbf{H}\left( v\right)  * W\left( v\right) }\right)  = {\int }_{-\infty }^{\infty }{\int }_{-\infty }^{\infty }W\left( x\right) {e}^{-{j2\pi f}\left( {x + \tau }\right) }\mathrm{d}x\mathbf{H}\left( \tau \right) \mathrm{d}\tau 
$$

$$
 = {\int }_{-\infty }^{\infty }{\int }_{-\infty }^{\infty }W\left( x\right) {e}^{-{j2\pi fx}}{e}^{-{j2\pi f\tau }}\mathrm{d}x\mathbf{H}\left( \tau \right) \mathrm{d}\tau 
$$

$$
 = {\int }_{-\infty }^{\infty }\mathbf{H}\left( \tau \right) {e}^{-{j2\pi f\tau }}\mathrm{d}\tau {\int }_{-\infty }^{\infty }W\left( x\right) {e}^{-{j2\pi fx}}\mathrm{\;d}x
$$

$$
 = \mathcal{H}\left( f\right) \mathcal{W}\left( f\right) 
$$

Accordingly, $\left( {\mathbf{H}\left( v\right)  * W\left( v\right) }\right)$ in the time domain is equal to $\left( {\mathcal{H}\left( f\right) \mathcal{W}\left( f\right) }\right)$ in the frequency domain. Therefore,the operations of FreMLP $\left( {\mathcal{{HW}} + \mathcal{B}}\right)$ in the channel (i.e., $v = N$ ) or temporal dimension (i.e., $v = L$ ),are equal to the operations $\left( {\mathbf{H} * W + B}\right)$ in the time domain. This implies that frequency-domain MLPs can be viewed as global convolutions in the time domain. Proved.

因此，时域中的$\left( {\mathbf{H}\left( v\right)  * W\left( v\right) }\right)$等于频域中的$\left( {\mathcal{H}\left( f\right) \mathcal{W}\left( f\right) }\right)$。因此，FreMLP在信道（即$v = N$）或时间维度（即$v = L$）上的操作$\left( {\mathcal{{HW}} + \mathcal{B}}\right)$，等同于时域中的操作$\left( {\mathbf{H} * W + B}\right)$。这意味着频域多层感知器（Multilayer Perceptron，MLP）可以被视为时域中的全局卷积。证明完毕。

## E Further Analysis

## E 进一步分析

### E.1 Ablation Study

### E.1 消融研究

In this section, we further analyze the effects of the frequency channel and temporal learners with different prediction lengths on ETTm1 and ETTh1 datasets. The results are shown in Table 6 . It demonstrates that with the prediction length increasing, the frequency temporal learner shows more effective than the channel learner. Especially, when the prediction length is longer (e.g., 336, 720), the channel learner will lead to worse performance. The reason is that when the prediction lengths become longer, the model with the channel learner is likely to overfit data during training. Thus for long-term forecasting with longer prediction lengths, the channel independence strategy may be more effective, as described in PatchTST [39].

在本节中，我们进一步分析了具有不同预测长度的频率通道和时间学习器对ETTm1和ETTh1数据集的影响。结果如表6所示。结果表明，随着预测长度的增加，频率时间学习器比通道学习器更有效。特别是当预测长度较长时（例如336、720），通道学习器会导致性能变差。原因是当预测长度变长时，采用通道学习器的模型在训练过程中可能会出现数据过拟合的情况。因此，对于预测长度较长的长期预测，如PatchTST [39]中所述，通道独立策略可能更有效。

<!-- Media -->

Table 6: Ablation studies of the frequency channel and temporal learners in long-term forecasting. 'I/O' indicates lookback window sizes/prediction lengths.

表6：长期预测中频率通道和时间学习器的消融研究。“输入/输出”表示回溯窗口大小/预测长度。

<table><tr><td>Dataset</td><td colspan="8">ETTm1</td><td colspan="8">ETTh1</td></tr><tr><td>I/O</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/336</td><td colspan="2">96/720</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/336</td><td colspan="2">96/720</td></tr><tr><td>Metrics</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>FreCL</td><td>0.053</td><td>0.078</td><td>0.059</td><td>0.085</td><td>0.067</td><td>0.095</td><td>0.097</td><td>0.125</td><td>0.063</td><td>0.089</td><td>0.067</td><td>0.093</td><td>0.071</td><td>0.097</td><td>0.087</td><td>0.115</td></tr><tr><td>FreTL</td><td>0.053</td><td>0.078</td><td>0.058</td><td>0.084</td><td>0.062</td><td>0.089</td><td>0.069</td><td>0.096</td><td>0.061</td><td>0.087</td><td>0.065</td><td>0.091</td><td>0.070</td><td>0.096</td><td>0.082</td><td>0.108</td></tr><tr><td>FreTS</td><td>0.052</td><td>0.077</td><td>0.057</td><td>0.083</td><td>0.064</td><td>0.092</td><td>0.071</td><td>0.099</td><td>0.063</td><td>0.089</td><td>0.066</td><td>0.092</td><td>0.072</td><td>0.098</td><td>0.086</td><td>0.113</td></tr></table>

<table><tbody><tr><td>数据集</td><td colspan="8">ETTm1（ETTm1）</td><td colspan="8">ETTh1（ETTh1）</td></tr><tr><td>输入/输出</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/336</td><td colspan="2">96/720</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/336</td><td colspan="2">96/720</td></tr><tr><td>指标</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td></tr><tr><td>频率对比学习（FreCL）</td><td>0.053</td><td>0.078</td><td>0.059</td><td>0.085</td><td>0.067</td><td>0.095</td><td>0.097</td><td>0.125</td><td>0.063</td><td>0.089</td><td>0.067</td><td>0.093</td><td>0.071</td><td>0.097</td><td>0.087</td><td>0.115</td></tr><tr><td>频率迁移学习（FreTL）</td><td>0.053</td><td>0.078</td><td>0.058</td><td>0.084</td><td>0.062</td><td>0.089</td><td>0.069</td><td>0.096</td><td>0.061</td><td>0.087</td><td>0.065</td><td>0.091</td><td>0.070</td><td>0.096</td><td>0.082</td><td>0.108</td></tr><tr><td>频率时间序列（FreTS）</td><td>0.052</td><td>0.077</td><td>0.057</td><td>0.083</td><td>0.064</td><td>0.092</td><td>0.071</td><td>0.099</td><td>0.063</td><td>0.089</td><td>0.066</td><td>0.092</td><td>0.072</td><td>0.098</td><td>0.086</td><td>0.113</td></tr></tbody></table>

<!-- Media -->

### E.2 Impacts of Real/Imaginary Parts

### E.2 实部/虚部的影响

To investigate the effects of real and imaginary parts, we conduct experiments on Exchange and ETTh1 datasets under different prediction lengths $L \in  \{ {96},{192}\}$ with the lookback window of 96 . Furthermore,we analyze the effects of ${\mathcal{W}}_{r}$ and ${\mathcal{W}}_{i}$ in the weights $\mathcal{W} = {\mathcal{W}}_{r} + j{\mathcal{W}}_{i}$ of FreMLP. In this experiment, we only use the frequency temporal learner in our model. The results are shown in Table 7. In the table,Input ${}_{real}$ indicates that we only feed the real part of the input into the network, and ${\operatorname{Input}}_{imag}$ indicates that we only feed the imaginary part of the input into the network. $\mathcal{W}\left( {\mathcal{W}}_{r}\right)$ denotes that we set ${\mathcal{W}}_{i}$ to 0 and $\mathcal{W}\left( {\mathcal{W}}_{i}\right)$ denotes that we set ${\mathcal{W}}_{r}$ to 0 . From the table,we can observe that both the real part and imaginary part of input are indispensable and the real part is more important to the imaginary part,and the real part of $\mathcal{W}$ plays a more significant role for the model performances.

为了研究实部和虚部的影响，我们在Exchange和ETTh1数据集上，以96的回溯窗口，针对不同预测长度$L \in  \{ {96},{192}\}$进行了实验。此外，我们分析了FreMLP权重$\mathcal{W} = {\mathcal{W}}_{r} + j{\mathcal{W}}_{i}$中${\mathcal{W}}_{r}$和${\mathcal{W}}_{i}$的影响。在本实验中，我们仅使用模型中的频率时间学习器。结果如表7所示。在表中，输入${}_{real}$表示我们仅将输入的实部输入到网络中，${\operatorname{Input}}_{imag}$表示我们仅将输入的虚部输入到网络中。$\mathcal{W}\left( {\mathcal{W}}_{r}\right)$表示我们将${\mathcal{W}}_{i}$设为0，$\mathcal{W}\left( {\mathcal{W}}_{i}\right)$表示我们将${\mathcal{W}}_{r}$设为0。从表中我们可以观察到，输入的实部和虚部都是不可或缺的，并且实部比虚部更重要，$\mathcal{W}$的实部对模型性能起着更重要的作用。

<!-- Media -->

Table 7: Investigation the impacts of real/imaginary parts

表7：研究实部/虚部的影响

<table><tr><td>Dataset</td><td colspan="4">Exchange</td><td colspan="4">ETTh1</td></tr><tr><td>I/O</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/96</td><td colspan="2">96/192</td></tr><tr><td>Metrics</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>${\text{Input}}_{\text{real }}$</td><td>0.048</td><td>0.062</td><td>0.058</td><td>0.074</td><td>0.080</td><td>0.111</td><td>0.083</td><td>0.113</td></tr><tr><td>${\operatorname{Input}}_{imag}$</td><td>0.143</td><td>0.185</td><td>0.143</td><td>0.184</td><td>0.130</td><td>0.156</td><td>0.130</td><td>0.156</td></tr><tr><td>$\mathcal{W}\left( {\mathcal{W}}_{r}\right)$</td><td>0.039</td><td>0.053</td><td>0.051</td><td>0.067</td><td>0.063</td><td>0.089</td><td>0.067</td><td>0.093</td></tr><tr><td>$\mathcal{W}\left( {\mathcal{W}}_{i}\right)$</td><td>0.143</td><td>0.184</td><td>0.142</td><td>0.184</td><td>0.116</td><td>0.138</td><td>0.117</td><td>0.139</td></tr><tr><td>FreTS</td><td>0.037</td><td>0.051</td><td>0.050</td><td>0.067</td><td>0.061</td><td>0.087</td><td>0.065</td><td>0.091</td></tr></table>

<table><tbody><tr><td>数据集</td><td colspan="4">交换；交流；交易</td><td colspan="4">ETTh1（原文未明确含义，保留原文）</td></tr><tr><td>输入/输出</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/96</td><td colspan="2">96/192</td></tr><tr><td>指标</td><td>平均绝对误差（Mean Absolute Error）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（Mean Absolute Error）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（Mean Absolute Error）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（Mean Absolute Error）</td><td>均方根误差（RMSE）</td></tr><tr><td>${\text{Input}}_{\text{real }}$</td><td>0.048</td><td>0.062</td><td>0.058</td><td>0.074</td><td>0.080</td><td>0.111</td><td>0.083</td><td>0.113</td></tr><tr><td>${\operatorname{Input}}_{imag}$</td><td>0.143</td><td>0.185</td><td>0.143</td><td>0.184</td><td>0.130</td><td>0.156</td><td>0.130</td><td>0.156</td></tr><tr><td>$\mathcal{W}\left( {\mathcal{W}}_{r}\right)$</td><td>0.039</td><td>0.053</td><td>0.051</td><td>0.067</td><td>0.063</td><td>0.089</td><td>0.067</td><td>0.093</td></tr><tr><td>$\mathcal{W}\left( {\mathcal{W}}_{i}\right)$</td><td>0.143</td><td>0.184</td><td>0.142</td><td>0.184</td><td>0.116</td><td>0.138</td><td>0.117</td><td>0.139</td></tr><tr><td>荧光共振能量转移（FreTS）</td><td>0.037</td><td>0.051</td><td>0.050</td><td>0.067</td><td>0.061</td><td>0.087</td><td>0.065</td><td>0.091</td></tr></tbody></table>

<!-- Media -->

### E.3 Parameter Sensitivity

### E.3 参数敏感性

We further perform extensive experiments on the ECG dataset to evaluate the sensitivity of the input length $L$ and the embedding dimension size $d$ . (1) Input length: We tune over the input length with the value $\{ 6,{12},{18},{24},{30},{36},{42},{50},{60}\}$ on the ECG dataset and the prediction length is 12,and the result is shown in Figure 6(a). From the figure, we can find that with the input length increasing, the performance first becomes better because the long input length may contain more pattern information, and then it decreases due to data redundancy or overfitting. (2) Embedding size: We choose the embedding size over the set $\{ {32},{64},{128},{256},{512}\}$ on the ECG dataset. The results are shown in Figure 6(b). It shows that the performance first increases and then decreases with the increase of the embedding size because a large embedding size improves the fitting ability of our FreTS but may easily lead to overfitting especially when the embedding size is too large.

我们进一步在心电图（ECG）数据集上进行了大量实验，以评估输入长度 $L$ 和嵌入维度大小 $d$ 的敏感性。(1) 输入长度：我们在心电图数据集上对输入长度进行调优，取值为 $\{ 6,{12},{18},{24},{30},{36},{42},{50},{60}\}$，预测长度为 12，结果如图 6(a) 所示。从图中可以发现，随着输入长度的增加，性能首先变好，因为较长的输入长度可能包含更多的模式信息，然后由于数据冗余或过拟合而下降。(2) 嵌入大小：我们在心电图数据集上从集合 $\{ {32},{64},{128},{256},{512}\}$ 中选择嵌入大小。结果如图 6(b) 所示。结果表明，随着嵌入大小的增加，性能先上升后下降，因为较大的嵌入大小提高了我们的 FreTS 模型的拟合能力，但可能容易导致过拟合，尤其是当嵌入大小过大时。

<!-- Media -->

<!-- figureText: 0.0545 0.0800 0.05290 0.0790 0.05285 0.0788 0.0786 0.05270 0.05265 0.0784 0.05260 400 (b) Embedding size 0.0540 0.0795 0.0535 0.0790 0.0785 0.0525 0.0780 0.0520 0.0775 (a) Input window length -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_18.jpg?x=437&y=1343&w=913&h=331&r=0"/>

Figure 6: The parameter sensitivity analyses of FreTS.

图 6：FreTS 的参数敏感性分析。

<!-- Media -->

## F Additional Results

## F 额外结果

### F.1 Multi-Step Forecasting

### F.1 多步预测

To further evaluate the performance of our FreTS in multi-step forecasting, we conduct more experiments on METR-LA and COVID-19 datasets with the input length of 12 and the prediction lengths of $\{ 3,6,9,{12}\}$ ,and the results are shown in Tables 8 and 9,respectively. In this experiment, we only select the state-of-the-art (i.e., GNN-based and Transformer-based) models as the baselines since they perform better than other models, such as RNN and TCN. Among these baselines, STGCN, DCRNN, and TAMP-S2GCNets require pre-defined graph structures. The results demonstrate that

为了进一步评估我们的FreTS在多步预测中的性能，我们在METR - LA（洛杉矶交通流量数据集）和COVID - 19数据集上进行了更多实验，输入长度为12，预测长度为$\{ 3,6,9,{12}\}$，结果分别如表8和表9所示。在本实验中，我们仅选择最先进的（即基于图神经网络（GNN）和基于Transformer的）模型作为基线，因为它们的表现优于其他模型，如循环神经网络（RNN）和时间卷积网络（TCN）。在这些基线模型中，时空图卷积网络（STGCN）、扩散卷积循环神经网络（DCRNN）和TAMP - S2GCNets需要预定义的图结构。结果表明

FreTS outperforms other baselines, including those models with pre-defined graph structures, at all steps. This further confirms that FreTS has strong capabilities in capturing channel-wise and time-wise dependencies.

在所有步骤中，FreTS的表现都优于其他基线模型，包括那些具有预定义图结构的模型。这进一步证实了FreTS在捕捉通道维度和时间维度依赖关系方面具有强大的能力。

<!-- Media -->

Table 8: Multi-step short-term forecasting results comparison on the METR-LA dataset with the input length of 12 and the prediction length of $\tau  \in  \{ 3,6,9,{12}\}$ . We highlight the best results in bold and the second best results are underline.

表8：在METR - LA数据集上，输入长度为12、预测长度为$\tau  \in  \{ 3,6,9,{12}\}$的多步短期预测结果比较。我们用粗体突出显示最佳结果，用下划线标注次佳结果。

<table><tr><td rowspan="2">Length Metrics</td><td colspan="2">3</td><td colspan="2">6</td><td colspan="2">9</td><td colspan="2">12</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>Reformer</td><td>0.086</td><td>0.154</td><td>0.097</td><td>0.176</td><td>0.107</td><td>0.193</td><td>0.118</td><td>0.206</td></tr><tr><td>Informer</td><td>0.082</td><td>0.156</td><td>0.094</td><td>0.176</td><td>0.108</td><td>0.193</td><td>0.125</td><td>0.214</td></tr><tr><td>Autoformer</td><td>0.087</td><td>0.149</td><td>0.091</td><td>0.162</td><td>0.106</td><td>0.178</td><td>0.099</td><td>0.184</td></tr><tr><td>FEDformer</td><td>0.064</td><td>0.127</td><td>0.073</td><td>0.145</td><td>0.079</td><td>0.160</td><td>0.086</td><td>0.175</td></tr><tr><td>DCRNN</td><td>0.160</td><td>0.204</td><td>0.191</td><td>0.243</td><td>0.216</td><td>0.269</td><td>0.241</td><td>0.291</td></tr><tr><td>STGCN</td><td>0.058</td><td>0.133</td><td>0.080</td><td>0.177</td><td>0.102</td><td>0.209</td><td>0.128</td><td>0.238</td></tr><tr><td>GraphWaveNet</td><td>0.180</td><td>0.366</td><td>0.184</td><td>0.375</td><td>0.196</td><td>0.382</td><td>0.202</td><td>0.386</td></tr><tr><td>MTGNN</td><td>0.135</td><td>0.294</td><td>0.144</td><td>0.307</td><td>0.149</td><td>0.328</td><td>0.153</td><td>0.316</td></tr><tr><td>StemGNN</td><td>0.052</td><td>0.115</td><td>0.069</td><td>0.141</td><td>0.080</td><td>0.162</td><td>0.093</td><td>0.175</td></tr><tr><td>AGCRN</td><td>0.062</td><td>0.131</td><td>0.086</td><td>0.165</td><td>0.099</td><td>0.188</td><td>0.109</td><td>0.204</td></tr><tr><td>$\mathbf{{FreTS}}$</td><td>0.050</td><td>0.113</td><td>0.066</td><td>0.140</td><td>0.076</td><td>0.158</td><td>0.080</td><td>0.166</td></tr></table>

<table><tbody><tr><td rowspan="2">长度指标</td><td colspan="2">3</td><td colspan="2">6</td><td colspan="2">9</td><td colspan="2">12</td></tr><tr><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td></tr><tr><td>改革者模型（Reformer）</td><td>0.086</td><td>0.154</td><td>0.097</td><td>0.176</td><td>0.107</td><td>0.193</td><td>0.118</td><td>0.206</td></tr><tr><td>告密者模型（Informer）</td><td>0.082</td><td>0.156</td><td>0.094</td><td>0.176</td><td>0.108</td><td>0.193</td><td>0.125</td><td>0.214</td></tr><tr><td>自动转换器模型（Autoformer）</td><td>0.087</td><td>0.149</td><td>0.091</td><td>0.162</td><td>0.106</td><td>0.178</td><td>0.099</td><td>0.184</td></tr><tr><td>联邦前馈变换器（FEDformer）</td><td>0.064</td><td>0.127</td><td>0.073</td><td>0.145</td><td>0.079</td><td>0.160</td><td>0.086</td><td>0.175</td></tr><tr><td>扩散卷积循环神经网络（DCRNN）</td><td>0.160</td><td>0.204</td><td>0.191</td><td>0.243</td><td>0.216</td><td>0.269</td><td>0.241</td><td>0.291</td></tr><tr><td>时空图卷积网络（STGCN）</td><td>0.058</td><td>0.133</td><td>0.080</td><td>0.177</td><td>0.102</td><td>0.209</td><td>0.128</td><td>0.238</td></tr><tr><td>图波网络（GraphWaveNet）</td><td>0.180</td><td>0.366</td><td>0.184</td><td>0.375</td><td>0.196</td><td>0.382</td><td>0.202</td><td>0.386</td></tr><tr><td>多任务图神经网络（MTGNN）</td><td>0.135</td><td>0.294</td><td>0.144</td><td>0.307</td><td>0.149</td><td>0.328</td><td>0.153</td><td>0.316</td></tr><tr><td>茎干图神经网络（StemGNN）</td><td>0.052</td><td>0.115</td><td>0.069</td><td>0.141</td><td>0.080</td><td>0.162</td><td>0.093</td><td>0.175</td></tr><tr><td>农业基因组学研究网络（Agricultural Genomics Research Network，假设，需结合实际确定准确含义）</td><td>0.062</td><td>0.131</td><td>0.086</td><td>0.165</td><td>0.099</td><td>0.188</td><td>0.109</td><td>0.204</td></tr><tr><td>$\mathbf{{FreTS}}$</td><td>0.050</td><td>0.113</td><td>0.066</td><td>0.140</td><td>0.076</td><td>0.158</td><td>0.080</td><td>0.166</td></tr></tbody></table>

Table 9: Multi-step short-term forecasting results comparison on the COVID-19 dataset with the input length of 12 and the prediction length of $\tau  \in  \{ 3,6,9,{12}\}$ . We highlight the best results in bold and the second best results are underline.

表9：在输入长度为12、预测长度为$\tau  \in  \{ 3,6,9,{12}\}$的COVID - 19数据集上的多步短期预测结果比较。我们将最佳结果用粗体突出显示，次佳结果用下划线标注。

<table><tr><td rowspan="2">Length Metrics</td><td colspan="2">3</td><td colspan="2">6</td><td colspan="2">9</td><td colspan="2">12</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>Reformer</td><td>0.212</td><td>0.282</td><td>0.139</td><td>0.186</td><td>0.148</td><td>0.197</td><td>0.152</td><td>0.209</td></tr><tr><td>Informer</td><td>0.234</td><td>0.312</td><td>0.190</td><td>0.245</td><td>0.184</td><td>0.242</td><td>0.200</td><td>0.259</td></tr><tr><td>Autoformer</td><td>0.212</td><td>0.280</td><td>0.144</td><td>0.191</td><td>0.152</td><td>0.201</td><td>0.159</td><td>0.211</td></tr><tr><td>FEDformer</td><td>0.246</td><td>0.328</td><td>0.169</td><td>0.242</td><td>0.175</td><td>0.247</td><td>0.160</td><td>0.219</td></tr><tr><td>GraphWaveNet</td><td>0.092</td><td>0.129</td><td>0.133</td><td>0.179</td><td>0.171</td><td>0.225</td><td>0.201</td><td>0.255</td></tr><tr><td>StemGNN</td><td>0.247</td><td>0.318</td><td>0.344</td><td>0.429</td><td>0.359</td><td>0.442</td><td>0.421</td><td>0.508</td></tr><tr><td>AGCRN</td><td>0.130</td><td>0.172</td><td>0.171</td><td>0.218</td><td>0.224</td><td>0.277</td><td>0.254</td><td>0.309</td></tr><tr><td>MTGNN</td><td>0.276</td><td>0.379</td><td>0.446</td><td>0.513</td><td>0.484</td><td>0.548</td><td>0.394</td><td>0.488</td></tr><tr><td>TAMP-S2GCNets</td><td>0.140</td><td>0.190</td><td>0.150</td><td>0.200</td><td>0.170</td><td>0.230</td><td>0.180</td><td>0.230</td></tr><tr><td>$\mathbf{{FreTS}}$</td><td>0.071</td><td>0.103</td><td>0.093</td><td>0.131</td><td>0.109</td><td>0.148</td><td>0.124</td><td>0.164</td></tr></table>

<table><tbody><tr><td rowspan="2">长度指标</td><td colspan="2">3</td><td colspan="2">6</td><td colspan="2">9</td><td colspan="2">12</td></tr><tr><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td></tr><tr><td>改革者模型（Reformer）</td><td>0.212</td><td>0.282</td><td>0.139</td><td>0.186</td><td>0.148</td><td>0.197</td><td>0.152</td><td>0.209</td></tr><tr><td>告密者模型（Informer）</td><td>0.234</td><td>0.312</td><td>0.190</td><td>0.245</td><td>0.184</td><td>0.242</td><td>0.200</td><td>0.259</td></tr><tr><td>自动转换器模型（Autoformer）</td><td>0.212</td><td>0.280</td><td>0.144</td><td>0.191</td><td>0.152</td><td>0.201</td><td>0.159</td><td>0.211</td></tr><tr><td>FEDformer（联邦前馈变换器）</td><td>0.246</td><td>0.328</td><td>0.169</td><td>0.242</td><td>0.175</td><td>0.247</td><td>0.160</td><td>0.219</td></tr><tr><td>图波网络（GraphWaveNet）</td><td>0.092</td><td>0.129</td><td>0.133</td><td>0.179</td><td>0.171</td><td>0.225</td><td>0.201</td><td>0.255</td></tr><tr><td>茎图神经网络（StemGNN）</td><td>0.247</td><td>0.318</td><td>0.344</td><td>0.429</td><td>0.359</td><td>0.442</td><td>0.421</td><td>0.508</td></tr><tr><td>自适应图卷积循环网络（AGCRN）</td><td>0.130</td><td>0.172</td><td>0.171</td><td>0.218</td><td>0.224</td><td>0.277</td><td>0.254</td><td>0.309</td></tr><tr><td>多任务图神经网络（MTGNN）</td><td>0.276</td><td>0.379</td><td>0.446</td><td>0.513</td><td>0.484</td><td>0.548</td><td>0.394</td><td>0.488</td></tr><tr><td>时间感知多路径时空图卷积网络（TAMP - S2GCNets）</td><td>0.140</td><td>0.190</td><td>0.150</td><td>0.200</td><td>0.170</td><td>0.230</td><td>0.180</td><td>0.230</td></tr><tr><td>$\mathbf{{FreTS}}$</td><td>0.071</td><td>0.103</td><td>0.093</td><td>0.131</td><td>0.109</td><td>0.148</td><td>0.124</td><td>0.164</td></tr></tbody></table>

<!-- Media -->

### F.2 Long-Term Forecasting under Varying Lookback Window

### F.2 不同回溯窗口下的长期预测

In Table 10, we present the long-term forecasting results of our FreTS and other baselines (PatchTST [39], LTSF-linear [35], FEDformer [30], Autoformer [14], Informer [13], and Reformer [18]) under different lookback window lengths $L \in  \{ {96},{192},{336}\}$ on the Exchange dataset. The prediction lengths are $\{ {96},{192},{336},{720}\}$ . From the table,we can observe that our FreTS outperforms all baselines in all settings and achieves significant improvements than FEDformer [30], Autoformer [14], Informer [13], and Reformer [18]. It verifies the effectiveness of our FreTS in learning informative representation under different lookback window.

在表10中，我们展示了我们的FreTS模型和其他基线模型（PatchTST [39]、LTSF-linear [35]、FEDformer [30]、Autoformer [14]、Informer [13]和Reformer [18]）在Exchange数据集上不同回溯窗口长度$L \in  \{ {96},{192},{336}\}$下的长期预测结果。预测长度为$\{ {96},{192},{336},{720}\}$。从表中我们可以观察到，在所有设置下，我们的FreTS模型都优于所有基线模型，并且与FEDformer [30]、Autoformer [14]、Informer [13]和Reformer [18]相比有显著改进。这验证了我们的FreTS模型在不同回溯窗口下学习信息表示的有效性。

## G Visualizations

## G 可视化

### G.1 Weight Visualizations for Energy Compaction

### G.1 能量压缩的权重可视化

We further visualize the weights $\mathcal{W} = {\mathcal{W}}_{r} + j{\mathcal{W}}_{i}$ in the frequency temporal learner under different settings, including different lookback window sizes and prediction lengths, on the Traffic and Electricity datasets. The results are illustrated in Figures 7 and 8 . These figures demonstrate that the weight coefficients of the real or imaginary part exhibit energy aggregation characteristics (clear diagonal patterns) which can facilitate frequency-domain MLPs in learning the significant features.

我们进一步可视化了在不同设置（包括不同的回溯窗口大小和预测长度）下，交通和电力数据集上频率时间学习器中的权重 $\mathcal{W} = {\mathcal{W}}_{r} + j{\mathcal{W}}_{i}$。结果如图 7 和图 8 所示。这些图表明，实部或虚部的权重系数呈现出能量聚集特征（清晰的对角模式），这有助于频域多层感知器（MLPs）学习重要特征。

<!-- Media -->

Table 10: Long-term forecasting results comparison with different lookback window lengths $L \in$ $\{ {96},{192},{336}\}$ . The prediction lengths are as $\tau  \in  \{ {96},{192},{336},{720}\}$ . The best results are in bold and the second best results are underlined.

表 10：不同回溯窗口长度 $L \in$ $\{ {96},{192},{336}\}$ 的长期预测结果比较。预测长度为 $\tau  \in  \{ {96},{192},{336},{720}\}$。最佳结果用粗体表示，次佳结果用下划线表示。

<table><tr><td colspan="2" rowspan="2">Models Metrics</td><td colspan="2">FreTS</td><td colspan="2">PatchTST</td><td colspan="2">LTSF-Linear</td><td colspan="2">FEDformer</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td><td colspan="2">Reformer</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="4">公</td><td>96</td><td>0.037</td><td>0.051</td><td>0.039</td><td>0.052</td><td>0.038</td><td>0.052</td><td>0.050</td><td>0.067</td><td>0.050</td><td>0.066</td><td>0.066</td><td>0.084</td><td>0.126</td><td>0.146</td></tr><tr><td>192</td><td>0.050</td><td>0.067</td><td>0.055</td><td>0.074</td><td>0.053</td><td>0.069</td><td>0.064</td><td>0.082</td><td>0.063</td><td>0.083</td><td>0.068</td><td>0.088</td><td>0.147</td><td>0.169</td></tr><tr><td>336</td><td>0.062</td><td>0.082</td><td>0.071</td><td>0.093</td><td>0.064</td><td>0.085</td><td>0.080</td><td>0.105</td><td>0.075</td><td>0.101</td><td>0.093</td><td>0.127</td><td>0.157</td><td>0.189</td></tr><tr><td>720</td><td>0.088</td><td>0.110</td><td>0.132</td><td>0.166</td><td>0.092</td><td>$\underline{0.116}$</td><td>0.151</td><td>0.183</td><td>0.150</td><td>0.181</td><td>0.117</td><td>0.170</td><td>0.166</td><td>0.201</td></tr><tr><td rowspan="4">192</td><td>96</td><td>0.036</td><td>0.050</td><td>0.037</td><td>0.051</td><td>0.038</td><td>0.051</td><td>0.067</td><td>0.086</td><td>0.066</td><td>0.085</td><td>0.109</td><td>0.131</td><td>0.123</td><td>0.143</td></tr><tr><td>192</td><td>0.051</td><td>0.068</td><td>0.052</td><td>0.070</td><td>0.053</td><td>0.070</td><td>0.080</td><td>0.101</td><td>0.080</td><td>0.102</td><td>0.144</td><td>0.172</td><td>0.139</td><td>0.161</td></tr><tr><td>336</td><td>0.066</td><td>0.087</td><td>0.072</td><td>0.097</td><td>0.073</td><td>0.096</td><td>0.093</td><td>0.122</td><td>0.099</td><td>0.129</td><td>0.141</td><td>0.177</td><td>0.155</td><td>0.181</td></tr><tr><td>720</td><td>0.088</td><td>0.110</td><td>0.099</td><td>0.128</td><td>0.098</td><td>0.122</td><td>0.190</td><td>0.222</td><td>0.191</td><td>0.224</td><td>0.173</td><td>0.210</td><td>0.159</td><td>0.193</td></tr><tr><td rowspan="4">336</td><td>96</td><td>0.038</td><td>0.052</td><td>0.039</td><td>0.053</td><td>0.040</td><td>0.055</td><td>0.088</td><td>0.113</td><td>0.088</td><td>0.110</td><td>0.137</td><td>0.169</td><td>0.128</td><td>0.148</td></tr><tr><td>192</td><td>0.053</td><td>0.070</td><td>0.055</td><td>0.071</td><td>0.055</td><td>0.072</td><td>0.103</td><td>0.133</td><td>0.104</td><td>0.133</td><td>0.161</td><td>0.195</td><td>0.138</td><td>0.159</td></tr><tr><td>336</td><td>0.071</td><td>0.092</td><td>0.074</td><td>0.099</td><td>0.077</td><td>0.100</td><td>0.123</td><td>0.155</td><td>0.127</td><td>0.159</td><td>0.156</td><td>0.193</td><td>0.156</td><td>0.179</td></tr><tr><td>720</td><td>0.082</td><td>0.108</td><td>0.100</td><td>0.129</td><td>0.087</td><td>0.110</td><td>0.210</td><td>0.242</td><td>0.211</td><td>0.244</td><td>0.173</td><td>0.210</td><td>0.168</td><td>0.205</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">模型指标</td><td colspan="2">FreTS（原文未明确含义，保留英文）</td><td colspan="2">PatchTST（原文未明确含义，保留英文）</td><td colspan="2">LTSF线性模型（LTSF-Linear）</td><td colspan="2">FEDformer（原文未明确含义，保留英文）</td><td colspan="2">自动转换器（Autoformer）</td><td colspan="2">告密者</td><td colspan="2">改革者</td></tr><tr><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td><td>平均绝对误差（MAE）</td><td>均方根误差（RMSE）</td></tr><tr><td rowspan="4">公</td><td>96</td><td>0.037</td><td>0.051</td><td>0.039</td><td>0.052</td><td>0.038</td><td>0.052</td><td>0.050</td><td>0.067</td><td>0.050</td><td>0.066</td><td>0.066</td><td>0.084</td><td>0.126</td><td>0.146</td></tr><tr><td>192</td><td>0.050</td><td>0.067</td><td>0.055</td><td>0.074</td><td>0.053</td><td>0.069</td><td>0.064</td><td>0.082</td><td>0.063</td><td>0.083</td><td>0.068</td><td>0.088</td><td>0.147</td><td>0.169</td></tr><tr><td>336</td><td>0.062</td><td>0.082</td><td>0.071</td><td>0.093</td><td>0.064</td><td>0.085</td><td>0.080</td><td>0.105</td><td>0.075</td><td>0.101</td><td>0.093</td><td>0.127</td><td>0.157</td><td>0.189</td></tr><tr><td>720</td><td>0.088</td><td>0.110</td><td>0.132</td><td>0.166</td><td>0.092</td><td>$\underline{0.116}$</td><td>0.151</td><td>0.183</td><td>0.150</td><td>0.181</td><td>0.117</td><td>0.170</td><td>0.166</td><td>0.201</td></tr><tr><td rowspan="4">192</td><td>96</td><td>0.036</td><td>0.050</td><td>0.037</td><td>0.051</td><td>0.038</td><td>0.051</td><td>0.067</td><td>0.086</td><td>0.066</td><td>0.085</td><td>0.109</td><td>0.131</td><td>0.123</td><td>0.143</td></tr><tr><td>192</td><td>0.051</td><td>0.068</td><td>0.052</td><td>0.070</td><td>0.053</td><td>0.070</td><td>0.080</td><td>0.101</td><td>0.080</td><td>0.102</td><td>0.144</td><td>0.172</td><td>0.139</td><td>0.161</td></tr><tr><td>336</td><td>0.066</td><td>0.087</td><td>0.072</td><td>0.097</td><td>0.073</td><td>0.096</td><td>0.093</td><td>0.122</td><td>0.099</td><td>0.129</td><td>0.141</td><td>0.177</td><td>0.155</td><td>0.181</td></tr><tr><td>720</td><td>0.088</td><td>0.110</td><td>0.099</td><td>0.128</td><td>0.098</td><td>0.122</td><td>0.190</td><td>0.222</td><td>0.191</td><td>0.224</td><td>0.173</td><td>0.210</td><td>0.159</td><td>0.193</td></tr><tr><td rowspan="4">336</td><td>96</td><td>0.038</td><td>0.052</td><td>0.039</td><td>0.053</td><td>0.040</td><td>0.055</td><td>0.088</td><td>0.113</td><td>0.088</td><td>0.110</td><td>0.137</td><td>0.169</td><td>0.128</td><td>0.148</td></tr><tr><td>192</td><td>0.053</td><td>0.070</td><td>0.055</td><td>0.071</td><td>0.055</td><td>0.072</td><td>0.103</td><td>0.133</td><td>0.104</td><td>0.133</td><td>0.161</td><td>0.195</td><td>0.138</td><td>0.159</td></tr><tr><td>336</td><td>0.071</td><td>0.092</td><td>0.074</td><td>0.099</td><td>0.077</td><td>0.100</td><td>0.123</td><td>0.155</td><td>0.127</td><td>0.159</td><td>0.156</td><td>0.193</td><td>0.156</td><td>0.179</td></tr><tr><td>720</td><td>0.082</td><td>0.108</td><td>0.100</td><td>0.129</td><td>0.087</td><td>0.110</td><td>0.210</td><td>0.242</td><td>0.211</td><td>0.244</td><td>0.173</td><td>0.210</td><td>0.168</td><td>0.205</td></tr></tbody></table>

<!-- figureText: (a) ${\mathcal{W}}_{r}$ under $\mathrm{I}/\mathrm{O} = {48}/{192}$ (d) ${\mathcal{W}}_{i}$ under $\mathrm{I}/\mathrm{O} = {48}/{192}$ -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_20.jpg?x=333&y=891&w=1132&h=723&r=0"/>

Figure 7: The visualizations of the weights $\mathcal{W}$ in the frequency temporal learner on the Traffic dataset. ’I/O’ denotes lookback window sizes/prediction lengths. ${\mathcal{W}}_{r}$ and ${\mathcal{W}}_{i}$ are the real and imaginary parts of $\mathcal{W}$ ,respectively.

图7：交通数据集上频率时间学习器中权重$\mathcal{W}$的可视化结果。“输入/输出（I/O）”表示回溯窗口大小/预测长度。${\mathcal{W}}_{r}$和${\mathcal{W}}_{i}$分别是$\mathcal{W}$的实部和虚部。

<!-- Media -->

### G.2 Weight Visualizations for Global View

### G.2 全局视角的权重可视化

To verify the characteristics of a global view of learning in the frequency domain, we perform additional experiments on the Traffic and Electricity datasets and compare the weights learned on the input in the time domain with those learned on the input frequency spectrum. The results are presented in Figures 9 and 10 The left side of the figures displays the weights learned on the input in the time domain, while the right side shows those learned on the real part of the input frequency spectrum. From the figures, we can observe that the patterns learned on the input frequency spectrum exhibit more obvious periodic patterns compared to the time domain. This is attributed to the global view characteristics of the frequency domain. Furthermore, we visualize the predictions of FreTS on the Traffic and Electricity datasets, as depicted in Figures 11 and 12, which show that FreTS exhibit a good ability to fit cyclic patterns. In summary, these results demonstrate that FreTS has a strong capability to capture the global periodic patterns, which benefits from the global view characteristics of the frequency domain.

为了验证频域学习全局视角的特性，我们在交通（Traffic）和电力（Electricity）数据集上进行了额外的实验，并将在时域输入上学习到的权重与在输入频谱上学习到的权重进行了比较。结果如图9和图10所示。图的左侧展示了在时域输入上学习到的权重，而右侧展示了在输入频谱实部上学习到的权重。从图中我们可以观察到，与在时域上学习到的模式相比，在输入频谱上学习到的模式呈现出更明显的周期性模式。这归因于频域的全局视角特性。此外，我们将FreTS在交通（Traffic）和电力（Electricity）数据集上的预测结果进行了可视化，如图11和图12所示，这些图表明FreTS具有良好的拟合循环模式的能力。总之，这些结果表明FreTS具有强大的捕捉全局周期性模式的能力，这得益于频域的全局视角特性。

<!-- Media -->

<!-- figureText: -0.8 - 0.15 - 0.10 - 0.05 - 0.00 -0.15 (b) Learned on the frequency spectrum 0.6 -0.4 -0.2 -0.0 (a) Learned on the input -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_21.jpg?x=318&y=1311&w=1143&h=474&r=0"/>

Figure 9: Visualization of the weights $\left( {L \times  \tau }\right)$ on the Traffic dataset with lookback window size of 72 and prediction length of 336.

图9：回望窗口大小为72、预测长度为336时，交通（Traffic）数据集上权重$\left( {L \times  \tau }\right)$的可视化结果。

<!-- figureText: 0.08 - 0.100 - 0.075 - 0.050 - 0.025 L0.000 -0.025 -0.050 -0.075 -0.100 80 (b) Learned on the frequency spectrum F 0.06 -0.04 -0.00 (a) Learned on the input -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_22.jpg?x=353&y=324&w=1112&h=481&r=0"/>

Figure 10: Visualization of the weights $\left( {L \times  \tau }\right)$ on the Electricity dataset with lookback window size of 96 and prediction length of 96 .

图10：在电力数据集（Electricity dataset）上，回溯窗口大小为96且预测长度为96时，权重 $\left( {L \times  \tau }\right)$ 的可视化结果。

<!-- figureText: 0.02 GroundTruth 0.08 0.06 0.04 0.02 0.00 0.02 0.00 (d) $\mathrm{I}/\mathrm{O} = {48}/{336}$ 0.08 Prediction 0.06 0.04 0.02 0.00 (c) $\mathrm{I}/\mathrm{O} = {48}/{192}$ -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_22.jpg?x=393&y=1125&w=1002&h=807&r=0"/>

Figure 11: Visualizations of predictions (forecast vs. actual) on the Traffic dataset. 'I/O' denotes lookback window sizes/prediction lengths.

图11：交通数据集（Traffic dataset）上的预测可视化结果（预测值与实际值对比）。“输入/输出（I/O）”表示回溯窗口大小/预测长度。

<!-- figureText: (a) $\mathrm{I}/\mathrm{O} = {96}/{96}$ 0.60 0.55 0.50 0.45 (b) $\mathrm{I}/\mathrm{O} = {96}/{192}$ (d) $\mathrm{I}/\mathrm{O} = {96}/{720}$ (c) $\mathrm{I}/\mathrm{O} = {96}/{336}$ -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_23.jpg?x=392&y=720&w=1003&h=809&r=0"/>

Figure 12: Visualizations of predictions (forecast vs. actual) on the Electricity dataset. 'I/O' denotes lookback window sizes/prediction lengths.

图12：电力数据集（Electricity dataset）上的预测可视化结果（预测值与实际值对比）。“输入/输出（I/O）”表示回溯窗口大小/预测长度。

<!-- figureText: (a) ${\mathcal{W}}_{r}$ under $\mathrm{I}/\mathrm{O} = {96}/{96}$ (b) ${\mathcal{W}}_{r}$ under $\mathrm{I}/\mathrm{O} = {96}/{336}$ (c) ${\mathcal{W}}_{r}$ under $\mathrm{I}/\mathrm{O} = {96}/{720}$ (f) ${\mathcal{W}}_{i}$ under $\mathrm{I}/\mathrm{O} = {96}/{720}$ (d) ${\mathcal{W}}_{i}$ under $\mathrm{I}/\mathrm{O} = {96}/{96}$ -->

<img src="https://cdn.noedgeai.com/01957f64-a00a-72df-a5d2-5472df2c2d4d_21.jpg?x=333&y=217&w=1139&h=733&r=0"/>

Figure 8: The visualizations of the weights $\mathcal{W}$ in the frequency temporal learner on the Electricity dataset. ’I/O’ denotes lookback window sizes/prediction lengths. ${\mathcal{W}}_{r}$ and ${\mathcal{W}}_{i}$ are the real and imaginary parts of $\mathcal{W}$ ,respectively.

图8：电力数据集（Electricity dataset）上，频率时间学习器中权重 $\mathcal{W}$ 的可视化结果。“输入/输出（I/O）”表示回溯窗口大小/预测长度。${\mathcal{W}}_{r}$ 和 ${\mathcal{W}}_{i}$ 分别是 $\mathcal{W}$ 的实部和虚部。

<!-- Media -->