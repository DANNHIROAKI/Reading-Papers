# FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting

# FEDformer：用于长期序列预测的频率增强分解变压器

Tian Zhou ${}^{ * }{}^{1}$ Ziqing Ma ${}^{ * }{}^{1}$ Qingsong Wen ${}^{1}$ Xue Wang ${}^{1}$ Liang Sun ${}^{1}$ Rong Jin ${}^{1}$

周天（Tian Zhou） ${}^{ * }{}^{1}$ 马子清（Ziqing Ma） ${}^{ * }{}^{1}$ 文青松（Qingsong Wen） ${}^{1}$ 王雪（Xue Wang） ${}^{1}$ 孙亮（Liang Sun） ${}^{1}$ 金荣（Rong Jin） ${}^{1}$

## Abstract

## 摘要

Although Transformer-based methods have significantly improved state-of-the-art results for long-term series forecasting, they are not only computationally expensive but more importantly, are unable to capture the global view of time series (e.g. overall trend). To address these problems, we propose to combine Transformer with the seasonal-trend decomposition method, in which the decomposition method captures the global profile of time series while Transformers capture more detailed structures. To further enhance the performance of Transformer for long-term prediction, we exploit the fact that most time series tend to have a sparse representation in well-known basis such as Fourier transform, and develop a frequency enhanced Transformer. Besides being more effective, the proposed method, termed as Frequency Enhanced Decomposed Transformer (FEDformer), is more efficient than standard Transformer with a linear complexity to the sequence length. Our empirical studies with six benchmark datasets show that compared with state-of-the-art methods, FED-former can reduce prediction error by ${14.8}\%$ and ${22.6}\%$ for multivariate and univariate time series, respectively. Code is publicly available at https://github.com/MAZiqing/FEDformer.

尽管基于Transformer的方法显著提升了长期序列预测的现有最优结果，但它们不仅计算成本高昂，更重要的是，无法捕捉时间序列的全局视图（例如总体趋势）。为解决这些问题，我们提议将Transformer与季节性趋势分解方法相结合，其中分解方法用于捕捉时间序列的全局特征，而Transformer则用于捕捉更详细的结构。为进一步提升Transformer在长期预测中的性能，我们利用了大多数时间序列在诸如傅里叶变换（Fourier transform）等知名基上倾向于具有稀疏表示这一事实，开发了一种频率增强的Transformer。除了更有效之外，所提出的方法，即频率增强分解Transformer（Frequency Enhanced Decomposed Transformer，FEDformer），比标准Transformer更高效，其复杂度与序列长度呈线性关系。我们使用六个基准数据集进行的实证研究表明，与现有最优方法相比，FEDformer在多变量和单变量时间序列上分别可将预测误差降低${14.8}\%$和${22.6}\%$。代码可在https://github.com/MAZiqing/FEDformer 上公开获取。

## 1. Introduction

## 1. 引言

Long-term time series forecasting is a long-standing challenge in various applications (e.g., energy, weather, traffic, economics). Despite the impressive results achieved by RNN-type methods (Rangapuram et al., 2018; Flunkert et al., 2017), they often suffer from the problem of gradient vanishing or exploding (Pascanu et al., 2013), significantly limiting their performance. Following the recent success in NLP and CV community (Vaswani et al., 2017; Devlin et al., 2019; Dosovitskiy et al., 2021; Rao et al., 2021), Transformer (Vaswani et al., 2017) has been introduced to capture long-term dependencies in time series forecasting and shows promising results (Zhou et al., 2021; Wu et al., 2021). Since high computational complexity and memory requirement make it difficult for Transformer to be applied to long sequence modeling, numerous studies are devoted to reduce the computational cost of Transformer (Li et al., 2019; Kitaev et al., 2020; Zhou et al., 2021; Wang et al., 2020; Xiong et al., 2021; Ma et al., 2021). A through overview of this line of works can be found in Appendix A.

长期时间序列预测在各种应用（如能源、气象、交通、经济）中一直是一项具有挑战性的任务。尽管循环神经网络（RNN）类方法取得了令人瞩目的成果（兰加普拉姆等人，2018年；弗伦克特等人，2017年），但它们常常受到梯度消失或梯度爆炸问题的困扰（帕斯卡努等人，2013年），这极大地限制了其性能。随着自然语言处理（NLP）和计算机视觉（CV）领域近期取得的成功（瓦斯瓦尼等人，2017年；德夫林等人，2019年；多索维茨基等人，2021年；饶等人，2021年），Transformer（瓦斯瓦尼等人，2017年）被引入到时间序列预测中以捕捉长期依赖关系，并展现出了良好的效果（周等人，2021年；吴等人，2021年）。由于Transformer的计算复杂度高且内存需求大，难以应用于长序列建模，因此众多研究致力于降低Transformer的计算成本（李等人，2019年；基塔耶夫等人，2020年；周等人，2021年；王等人，2020年；熊等人，2021年；马等人，2021年）。关于这一系列工作的全面概述可在附录A中找到。

Despite the progress made by Transformer-based methods for time series forecasting, they tend to fail in capturing the overall characteristics/distribution of time series in some cases. In Figure 1, we compare the time series of ground truth with that predicted by the vanilla Transformer method (Vaswani et al., 2017) in a real-world ETTm1 dataset (Zhou et al., 2021). It is clear that the predicted time series shared a different distribution from that of ground truth. The discrepancy between ground truth and prediction could be explained by the point-wise attention and prediction in Transformer. Since prediction for each timestep is made individually and independently, it is likely that the model fails to maintain the global property and statistics of time series as a whole. To address this problem, we exploit two ideas in this work. The first idea is to incorporate a seasonal-trend decomposition approach (Cleveland et al., 1990; Wen et al., 2019), which is widely used in time series analysis, into the Transformer-based method. Although this idea has been exploited before (Oreshkin et al., 2019; Wu et al., 2021), we present a special design of network that is effective in bringing the distribution of prediction close to that of ground truth, according to Kologrov-Smirnov distribution test. Our second idea is to combine Fourier analysis with the Transformer-based method. Instead of applying Transformer to the time domain, we apply it to the frequency domain which helps Transformer better capture global properties of time series. Combining both ideas, we propose a Frequency Enhanced Decomposition Transformer, or, FEDformer for short, for long-term time series forecasting.

尽管基于Transformer的时间序列预测方法取得了进展，但在某些情况下，它们往往无法捕捉到时间序列的整体特征/分布。在图1中，我们将真实时间序列与普通Transformer方法（Vaswani等人，2017年）在真实世界ETTm1数据集（Zhou等人，2021年）上的预测结果进行了比较。显然，预测的时间序列与真实时间序列的分布不同。真实值与预测值之间的差异可以用Transformer中的逐点注意力和预测来解释。由于每个时间步的预测是单独且独立进行的，因此模型很可能无法维持时间序列作为一个整体的全局属性和统计特征。为了解决这个问题，我们在这项工作中采用了两个思路。第一个思路是将一种季节趋势分解方法（Cleveland等人，1990年；Wen等人，2019年）（该方法在时间序列分析中被广泛使用）融入基于Transformer的方法中。尽管之前已经有人采用过这个思路（Oreshkin等人，2019年；Wu等人，2021年），但根据柯尔莫哥洛夫 - 斯米尔诺夫分布检验，我们提出了一种特殊的网络设计，该设计能有效地使预测分布接近真实分布。我们的第二个思路是将傅里叶分析与基于Transformer的方法相结合。我们不是将Transformer应用于时域，而是将其应用于频域，这有助于Transformer更好地捕捉时间序列的全局属性。结合这两个思路，我们提出了一种频率增强分解Transformer（Frequency Enhanced Decomposition Transformer），简称FEDformer，用于长期时间序列预测。

---

<!-- Footnote -->

*Equal contribution ${}^{1}$ Machine Intelligence Technology,Al-ibaba Group.. Correspondence to: Tian Zhou <tian.zt@alibaba-inc.com>, Rong Jin <jinrong.jr@alibaba-inc.com>.

*同等贡献 ${}^{1}$ 阿里巴巴集团机器智能技术部。通信作者：周田（Tian Zhou）<tian.zt@alibaba-inc.com>，金榕（Rong Jin）<jinrong.jr@alibaba-inc.com>。

Proceedings of the ${39}^{\text{th }}$ International Conference on Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022. Copyright 2022 by the author(s).

${39}^{\text{th }}$ 国际机器学习会议论文集，美国马里兰州巴尔的摩，《机器学习研究会议录》第162卷，2022年。版权所有©2022 作者。

<!-- Footnote -->

---

One critical question with FEDformer is which subset of frequency components should be used by Fourier analysis to represent time series. A common wisdom is to keep low-frequency components and throw away the high-frequency ones. This may not be appropriate for time series forecasting as some of trend changes in time series are related to important events, and this piece of information could be lost if we simply remove all high-frequency components. We address this problem by effectively exploiting the fact that time series tend to have (unknown) sparse representations on a basis like Fourier basis. According to our theoretical analysis, a randomly selected subset of frequency components, including both low and high ones, will give a better representation for time series, which is further verified by extensive empirical studies. Besides being more effective for long term forecasting, combining Transformer with frequency analysis allows us to reduce the computational cost of Transformer from quadratic to linear complexity. We note that this is different from previous efforts on speeding up Transformer, which often leads to a performance drop.

FEDformer的一个关键问题是，傅里叶分析应使用哪些频率分量子集来表示时间序列。一种常见的做法是保留低频分量而舍弃高频分量。这对于时间序列预测可能并不合适，因为时间序列中的一些趋势变化与重要事件相关，如果我们简单地去除所有高频分量，这些信息可能会丢失。我们通过有效利用时间序列在傅里叶基等基上往往具有（未知的）稀疏表示这一事实来解决这个问题。根据我们的理论分析，一个随机选择的频率分量子集，包括低频和高频分量，将能更好地表示时间序列，这一点也得到了大量实证研究的进一步验证。除了在长期预测中更有效之外，将Transformer与频率分析相结合还能将Transformer的计算成本从二次复杂度降低到线性复杂度。我们注意到，这与之前加速Transformer的努力不同，那些努力往往会导致性能下降。

In short, we summarize the key contributions of this work as follows:

简而言之，我们将这项工作的主要贡献总结如下：

1. We propose a frequency enhanced decomposed Transformer architecture with mixture of experts for seasonal-trend decomposition in order to better capture global properties of time series.

1. 我们提出了一种具有专家混合机制的频率增强分解Transformer架构，用于季节性 - 趋势分解，以便更好地捕捉时间序列的全局特性。

2. We propose Fourier enhanced blocks and Wavelet enhanced blocks in the Transformer structure that allows us to capture important structures in time series through frequency domain mapping. They serve as substitutions for both self-attention and cross-attention blocks.

2. 我们在Transformer结构中提出了傅里叶增强块（Fourier enhanced blocks）和小波增强块（Wavelet enhanced blocks），这使我们能够通过频域映射捕捉时间序列中的重要结构。它们可替代自注意力块和交叉注意力块。

3. By randomly selecting a fixed number of Fourier components, the proposed model achieves linear computational complexity and memory cost. The effectiveness of this selection method is verified both theoretically and empirically.

3. 通过随机选择固定数量的傅里叶分量（Fourier components），所提出的模型实现了线性计算复杂度和内存成本。这种选择方法的有效性在理论和实证上都得到了验证。

4. We conduct extensive experiments over 6 benchmark datasets across multiple domains (energy, traffic, economics, weather and disease). Our empirical studies show that the proposed model improves the performance of state-of-the-art methods by ${14.8}\%$ and ${22.6}\%$ for multivariate and univariate forecasting,respectively.

4. 我们在跨多个领域（能源、交通、经济、天气和疾病）的6个基准数据集上进行了广泛的实验。我们的实证研究表明，所提出的模型在多变量和单变量预测方面分别将最先进方法的性能提高了${14.8}\%$和${22.6}\%$。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957f69-db4d-7f34-af19-bfc9e93e263b_1.jpg?x=904&y=215&w=662&h=230&r=0"/>

Figure 1. Different distribution between ground truth and forecasting output from vanilla Transformer in a real-world ETTm1 dataset. Left: frequency mode and trend shift. Right: trend shift.

图1. 在真实世界的ETTm1数据集中，原始Transformer的真实值与预测输出之间的不同分布。左：频率模式和趋势变化。右：趋势变化。

<!-- Media -->

## 2. Compact Representation of Time Series in Frequency Domain

## 2. 时间序列在频域的紧凑表示

It is well-known that time series data can be modeled from the time domain and frequency domain. One key contribution of our work which separates from other long-term forecasting algorithms is the frequency-domain operation with a neural network. As Fourier analysis is a common tool to dive into the frequency domain, while how to appropriately represent the information in time series using Fourier analysis is critical. Simply keeping all the frequency components may result in inferior representations since many high-frequency changes in time series are due to noisy inputs. On the other hand, only keeping the low-frequency components may also be inappropriate for series forecasting as some trend changes in time series represent important events. Instead, keeping a compact representation of time series using a small number of selected Fourier components will lead to efficient computation of transformer, which is crucial for modelling long sequences. We propose to represent time series by randomly selecting a constant number of Fourier components, including both high-frequency and low-frequency. Below, an analysis that justifies the random selection is presented theoretically. Empirical verification can be found in the experimental session.

众所周知，可以从时域和频域对时间序列数据进行建模。我们的工作与其他长期预测算法的一个关键区别在于，我们使用神经网络进行频域操作。傅里叶分析是深入研究频域的常用工具，而如何使用傅里叶分析恰当地表示时间序列中的信息至关重要。简单地保留所有频率分量可能会导致表示效果不佳，因为时间序列中的许多高频变化是由噪声输入引起的。另一方面，仅保留低频分量对于序列预测可能也不合适，因为时间序列中的一些趋势变化代表着重要事件。相反，使用少量选定的傅里叶分量对时间序列进行紧凑表示，将有助于变压器（transformer）的高效计算，这对于对长序列进行建模至关重要。我们建议通过随机选择固定数量的傅里叶分量（包括高频和低频分量）来表示时间序列。下面从理论上对随机选择的合理性进行分析。实证验证可在实验部分找到。

Consider we have $m$ time series,denoted as ${X}_{1}\left( t\right) ,\ldots ,{X}_{m}\left( t\right)$ . By applying Fourier transform to each time series,we turn each ${X}_{i}\left( t\right)$ into a vector ${a}_{i} = {\left( {a}_{i,1},\ldots ,{a}_{i,d}\right) }^{\top } \in  {\mathbb{R}}^{d}$ . By putting all the Fourier transform vectors into a matrix, we have $A = {\left( {a}_{1},{a}_{2},\ldots ,{a}_{m}\right) }^{\top } \in  {\mathbb{R}}^{m \times  d}$ ,with each row corresponding to a different time series and each column corresponding to a different Fourier component. Although using all the Fourier components allows us to best preserve the history information in the time series, it may potentially lead to overfitting of the history data and consequentially a poor prediction of future signals. Hence, we need to select a subset of Fourier components, that on the one hand should be small enough to avoid the overfitting problem and on the other hand, should be able to preserve most of the history information. Here, we propose to select $s$ components from the $d$ Fourier components $\left( {s < d}\right)$ uniformly at random. More specifically, we denote by ${i}_{1} < {i}_{2} < \ldots  < {i}_{s}$ the randomly selected components. We construct matrix $S \in  \{ 0,1{\} }^{s \times  d}$ ,with ${S}_{i,k} = 1$ if $i = {i}_{k}$ and ${S}_{i,k} = 0$ otherwise. Then,our representation of multivariate time series becomes ${A}^{\prime } = A{S}^{\top } \in  {\mathbb{R}}^{m \times  s}$ . Below, we will show that, although the Fourier basis are randomly selected,under a mild condition, ${A}^{\prime }$ is able to preserve most of the information from $A$ .

考虑我们有$m$个时间序列，记为${X}_{1}\left( t\right) ,\ldots ,{X}_{m}\left( t\right)$。对每个时间序列应用傅里叶变换（Fourier transform），我们将每个${X}_{i}\left( t\right)$转换为一个向量${a}_{i} = {\left( {a}_{i,1},\ldots ,{a}_{i,d}\right) }^{\top } \in  {\mathbb{R}}^{d}$。将所有的傅里叶变换向量放入一个矩阵中，我们得到$A = {\left( {a}_{1},{a}_{2},\ldots ,{a}_{m}\right) }^{\top } \in  {\mathbb{R}}^{m \times  d}$，其中每一行对应一个不同的时间序列，每一列对应一个不同的傅里叶分量（Fourier component）。虽然使用所有的傅里叶分量可以让我们最好地保留时间序列中的历史信息，但这可能会导致对历史数据的过拟合（overfitting），从而对未来信号的预测效果不佳。因此，我们需要选择傅里叶分量的一个子集，一方面这个子集要足够小以避免过拟合问题，另一方面，它应该能够保留大部分的历史信息。在这里，我们建议从$d$个傅里叶分量$\left( {s < d}\right)$中均匀随机地选择$s$个分量。更具体地说，我们用${i}_{1} < {i}_{2} < \ldots  < {i}_{s}$表示随机选择的分量。我们构造矩阵$S \in  \{ 0,1{\} }^{s \times  d}$，如果$i = {i}_{k}$，则${S}_{i,k} = 1$；否则${S}_{i,k} = 0$。那么，我们对多变量时间序列的表示就变成了${A}^{\prime } = A{S}^{\top } \in  {\mathbb{R}}^{m \times  s}$。下面，我们将证明，尽管傅里叶基是随机选择的，但在一个温和的条件下，${A}^{\prime }$能够保留$A$中的大部分信息。

<!-- Media -->

<!-- figureText: Frequency MOE Feed MOE Decomp MOE Feed MOE Decomp Forward Decomp Enhanced Decomp Forward FEDformer Encoder Frequency MOE Frequency Enhanced Enhanced Decomp FEDformer Decoder -->

<img src="https://cdn.noedgeai.com/01957f69-db4d-7f34-af19-bfc9e93e263b_2.jpg?x=197&y=182&w=1372&h=451&r=0"/>

Figure 2. FEDformer Structure. The FEDformer consists of $N$ encoders and $M$ decoders. The Frequency Enhanced Block (FEB,green blocks) and Frequency Enhanced Attention (FEA, red blocks) are used to perform representation learning in frequency domain. Either FEB or FEA has two subversions (FEB-f & FEB-w or FEA-f & FEA-w), where '-f' means using Fourier basis and '-w' means using Wavelet basis. The Mixture Of Expert Decomposition Blocks (MOEDecomp, you books) are used to extract seasonal-trend patterns from the input data.

图2. FEDformer架构。FEDformer由$N$个编码器和$M$个解码器组成。频率增强模块（Frequency Enhanced Block，FEB，绿色模块）和频率增强注意力机制（Frequency Enhanced Attention，FEA，红色模块）用于在频域中进行表征学习。FEB和FEA都有两个子版本（FEB - f和FEB - w或FEA - f和FEA - w），其中“- f”表示使用傅里叶基，“- w”表示使用小波基。专家混合分解模块（Mixture Of Expert Decomposition Blocks，MOEDecomp）用于从输入数据中提取季节性 - 趋势模式。

<!-- Media -->

In order to measure how well ${A}^{\prime }$ is able to preserve information from $A$ ,we project each column vector of $A$ into the subspace spanned by the column vectors in ${A}^{\prime }$ . We denote by ${P}_{{A}^{\prime }}\left( A\right)$ the resulting matrix after the projection,where ${P}_{{A}^{\prime }}\left( \cdot \right)$ represents the projection operator. If ${A}^{\prime }$ preserves a large portion of information from $A$ ,we would expect a small error between $A$ and ${P}_{{A}^{\prime }}\left( A\right)$ ,i.e. $\left| {A - {P}_{{A}^{\prime }}\left( A\right) }\right|$ . Let ${A}_{k}$ represent the approximation of $A$ by its first $k$ largest single value decomposition. The theorem below shows that $\left| {A - {P}_{{A}^{\prime }}\left( A\right) }\right|$ is close to $\left| {A - {A}_{k}}\right|$ if the number of randomly sampled Fourier components $s$ is on the order of ${k}^{2}$ .

为了衡量${A}^{\prime }$能够在多大程度上保留来自$A$的信息，我们将$A$的每个列向量投影到由${A}^{\prime }$的列向量所张成的子空间中。我们用${P}_{{A}^{\prime }}\left( A\right)$表示投影后得到的矩阵，其中${P}_{{A}^{\prime }}\left( \cdot \right)$表示投影算子。如果${A}^{\prime }$保留了$A$的大部分信息，我们预计$A$和${P}_{{A}^{\prime }}\left( A\right)$之间的误差会很小，即$\left| {A - {P}_{{A}^{\prime }}\left( A\right) }\right|$。设${A}_{k}$表示$A$通过其前$k$个最大奇异值分解得到的近似。下面的定理表明，如果随机采样的傅里叶分量的数量$s$的数量级为${k}^{2}$，则$\left| {A - {P}_{{A}^{\prime }}\left( A\right) }\right|$接近$\left| {A - {A}_{k}}\right|$。

Theorem 1. Assume that $\mu \left( A\right)$ ,the coherence measure of matrix $A$ ,is $\Omega \left( {k/n}\right)$ . Then,with a high probability,we have

定理1. 假设矩阵$A$的相干性度量$\mu \left( A\right)$为$\Omega \left( {k/n}\right)$。那么，在大概率情况下，我们有

$$
\left| {A - {P}_{{A}^{\prime }}\left( A\right) }\right|  \leq  \left( {1 + \epsilon }\right) \left| {A - {A}_{k}}\right| 
$$

$$
\text{if}s = O\left( {{k}^{2}/{\epsilon }^{2}}\right) \text{.}
$$

The detailed analysis can be found in Appendix C.

详细分析见附录C。

For real-world multivariate times series, the corresponding matrix $A$ from Fourier transform often exhibit low rank property, since those univaraite variables in multivariate times series depend not only on its past values but also has dependency on each other, as well as share similar frequency components. Therefore, as indicated by the Theorem 1, randomly selecting a subset of Fourier components allows us to appropriately represent the information in Fourier matrix $A$ .

对于现实世界中的多变量时间序列，由傅里叶变换得到的相应矩阵$A$通常具有低秩特性，因为多变量时间序列中的那些单变量不仅依赖于其过去的值，而且彼此之间也存在依赖关系，并且共享相似的频率分量。因此，正如定理1所示，随机选择傅里叶分量的一个子集可以让我们恰当地表示傅里叶矩阵$A$中的信息。

Similarly, wavelet orthogonal polynomials, such as Legendre Polynomials, obey restricted isometry property (RIP) and can be used for capture information in time series as well. Compared to Fourier basis, wavelet based representation is more effective in capturing local structures in time series and thus can be more effective for some forecasting tasks. We defer the discussion of wavelet based representation in Appendix B. In the next section, we will present the design of frequency enhanced decomposed Transformer architecture that incorporate the Fourier transform into transformer.

同样地，小波正交多项式，如勒让德多项式（Legendre Polynomials），服从受限等距性质（RIP），也可用于捕捉时间序列中的信息。与傅里叶基相比，基于小波的表示在捕捉时间序列的局部结构方面更有效，因此在一些预测任务中可能更有效。我们将基于小波的表示的讨论推迟到附录B中。在下一节中，我们将介绍频率增强分解Transformer架构的设计，该架构将傅里叶变换融入到Transformer中。

## 3. Model Structure

## 3. 模型结构

In this section, we will introduce (1) the overall structure of FEDformer, as shown in Figure 2, (2) two subversion structures for signal process: one uses Fourier basis and the other uses Wavelet basis, (3) the mixture of experts mechanism for seasonal-trend decomposition, and (4) the complexity analysis of the proposed model.

在本节中，我们将介绍（1）FEDformer的整体结构，如图2所示；（2）用于信号处理的两种子结构：一种使用傅里叶基，另一种使用小波基；（3）用于季节性 - 趋势分解的专家混合机制；（4）所提出模型的复杂度分析。

### 3.1. FEDformer Framework

### 3.1. FEDformer框架

Preliminary Long-term time series forecasting is a sequence to sequence problem. We denote the input length as $I$ and output length as $O$ . We denote $D$ as the hidden states of the series. The input of the encoder is a $I \times  D$ matrix and the decoder has $\left( {I/2 + O}\right)  \times  D$ input.

初步的长期时间序列预测是一个序列到序列的问题。我们将输入长度记为$I$，输出长度记为$O$。我们将$D$记为序列的隐藏状态。编码器的输入是一个$I \times  D$矩阵，解码器有$\left( {I/2 + O}\right)  \times  D$个输入。

FEDformer Structure Inspired by the seasonal-trend decomposition and distribution analysis as discussed in Section 1, we renovate Transformer as a deep decomposition architecture as shown in Figure 2, including Frequency Enhanced Block (FEB), Frequency Enhanced Attention (FEA) connecting encoder and decoder, and the Mixture Of Experts Decomposition block (MOEDecomp). The detailed description of FEB, FEA, and MOEDecomp blocks will be given in the following Section 3.2, 3.3, and 3.4 respectively.

FEDformer架构 受第1节中讨论的季节性 - 趋势分解和分布分析的启发，我们将Transformer改进为如图2所示的深度分解架构，包括频率增强块（Frequency Enhanced Block，FEB）、连接编码器和解码器的频率增强注意力机制（Frequency Enhanced Attention，FEA）以及专家混合分解块（Mixture Of Experts Decomposition block，MOEDecomp）。FEB、FEA和MOEDecomp块的详细描述将分别在接下来的3.2节、3.3节和3.4节中给出。

The encoder adopts a multilayer structure as: ${\mathcal{X}}_{\text{en }}^{l} =$ Encoder $\left( {\mathcal{X}}_{\text{en }}^{l - 1}\right)$ ,where $l \in  \{ 1,\cdots ,N\}$ denotes the output of $l$ -th encoder layer and ${\mathcal{X}}_{\text{en }}^{0} \in  {\mathbb{R}}^{I \times  D}$ is the embedded historical series. The Encoder $\left( \cdot \right)$ is formalized as

编码器采用多层结构，如下所示：${\mathcal{X}}_{\text{en }}^{l} =$ 编码器 $\left( {\mathcal{X}}_{\text{en }}^{l - 1}\right)$ ，其中 $l \in  \{ 1,\cdots ,N\}$ 表示第 $l$ 个编码器层的输出，${\mathcal{X}}_{\text{en }}^{0} \in  {\mathbb{R}}^{I \times  D}$ 是嵌入的历史序列。编码器 $\left( \cdot \right)$ 形式化表示为

$$
{\mathcal{S}}_{\mathrm{{en}}}^{l,1}, -  = \operatorname{MOEDecomp}\left( {\mathrm{{FEB}}\left( {\mathcal{X}}_{\mathrm{{en}}}^{l - 1}\right)  + {\mathcal{X}}_{\mathrm{{en}}}^{l - 1}}\right) ,
$$

$$
{\mathcal{S}}_{\text{en }}^{l,2}, -  = \text{MOEDecomp(FeedForward}\left( {\mathcal{S}}_{\text{en }}^{l,1}\right)  + {\mathcal{S}}_{\text{en }}^{l,1}\text{),} \tag{1}
$$

$$
{\mathcal{X}}_{\text{en }}^{l} = {\mathcal{S}}_{\text{en }}^{l,2}
$$

where ${\mathcal{S}}_{\text{en }}^{l,i},i \in  \{ 1,2\}$ represents the seasonal component after the $i$ -th decomposition block in the $l$ -th layer respectively. For FEB module, it has two different versions (FEB-f & FEB-w) which are implemented through Discrete Fourier transform (DFT) and Discrete Wavelet transform (DWT) mechanism respectively and can seamlessly replace the self-attention block.

其中 ${\mathcal{S}}_{\text{en }}^{l,i},i \in  \{ 1,2\}$ 分别表示第 $l$ 层中第 $i$ 个分解块之后的季节性分量。对于FEB模块（频率增强块模块，Frequency Enhancement Block），它有两个不同的版本（FEB - f和FEB - w），分别通过离散傅里叶变换（DFT）和离散小波变换（DWT）机制实现，并且可以无缝替换自注意力块。

The decoder also adopts a multilayer structure as: ${\mathcal{X}}_{\mathrm{{de}}}^{l},{\mathcal{T}}_{\mathrm{{de}}}^{l} = \operatorname{Decoder}\left( {{\mathcal{X}}_{\mathrm{{de}}}^{l - 1},{\mathcal{T}}_{\mathrm{{de}}}^{l - 1}}\right)$ ,where $l \in  \{ 1,\cdots ,M\}$ denotes the output of $l$ -th decoder layer. The Decoder $\left( \cdot \right)$ is formalized as

解码器同样采用多层结构，如下所示：${\mathcal{X}}_{\mathrm{{de}}}^{l},{\mathcal{T}}_{\mathrm{{de}}}^{l} = \operatorname{Decoder}\left( {{\mathcal{X}}_{\mathrm{{de}}}^{l - 1},{\mathcal{T}}_{\mathrm{{de}}}^{l - 1}}\right)$ ，其中 $l \in  \{ 1,\cdots ,M\}$ 表示第 $l$ 层解码器的输出。解码器 $\left( \cdot \right)$ 可形式化表示为

$$
{\mathcal{S}}_{\mathrm{{de}}}^{l,1},{\mathcal{T}}_{\mathrm{{de}}}^{l,1} = \operatorname{MOEDecomp}\left( {\operatorname{FEB}\left( {\mathcal{X}}_{\mathrm{{de}}}^{l - 1}\right)  + {\mathcal{X}}_{\mathrm{{de}}}^{l - 1}}\right) ,
$$

$$
{\mathcal{S}}_{\mathrm{{de}}}^{l,2},{\mathcal{T}}_{\mathrm{{de}}}^{l,2} = \operatorname{MOEDecomp}\left( {\operatorname{FEA}\left( {{\mathcal{S}}_{\mathrm{{de}}}^{l,1},{\mathcal{X}}_{\mathrm{{en}}}^{N}}\right)  + {\mathcal{S}}_{\mathrm{{de}}}^{l,1}}\right) ,
$$

$$
{\mathcal{S}}_{\mathrm{{de}}}^{l,3},{\mathcal{T}}_{\mathrm{{de}}}^{l,3} = \operatorname{MOEDecomp}\left( {\operatorname{FeedForward}\left( {\mathcal{S}}_{\mathrm{{de}}}^{l,2}\right)  + {\mathcal{S}}_{\mathrm{{de}}}^{l,2}}\right) ,
$$

$$
{\mathcal{X}}_{\mathrm{{de}}}^{l} = {\mathcal{S}}_{\mathrm{{de}}}^{l,3}
$$

$$
{\mathcal{T}}_{\mathrm{{de}}}^{l} = {\mathcal{T}}_{\mathrm{{de}}}^{l - 1} + {\mathcal{W}}_{l,1} \cdot  {\mathcal{T}}_{\mathrm{{de}}}^{l,1} + {\mathcal{W}}_{l,2} \cdot  {\mathcal{T}}_{\mathrm{{de}}}^{l,2} + {\mathcal{W}}_{l,3} \cdot  {\mathcal{T}}_{\mathrm{{de}}}^{l,3},
$$

(2)

where ${\mathcal{S}}_{\mathrm{{de}}}^{l,i},{\mathcal{T}}_{\mathrm{{de}}}^{l,i},i \in  \{ 1,2,3\}$ represent the seasonal and trend component after the $i$ -th decomposition block in the $l$ - th layer respectively. ${\mathcal{W}}_{l,i},i \in  \{ 1,2,3\}$ represents the projector for the $i$ -th extracted trend ${\mathcal{T}}_{\mathrm{{de}}}^{l,i}$ . Similar to FEB,FEA has two different versions (FEA-f & FEA-w) which are implemented through DFT and DWT projection respectively with attention design, and can replace the cross-attention block. The detailed description of FEA(.) will be given in the following Section 3.3.

其中 ${\mathcal{S}}_{\mathrm{{de}}}^{l,i},{\mathcal{T}}_{\mathrm{{de}}}^{l,i},i \in  \{ 1,2,3\}$ 分别表示第 $l$ 层中第 $i$ 个分解块之后的季节性和趋势分量。${\mathcal{W}}_{l,i},i \in  \{ 1,2,3\}$ 表示第 $i$ 个提取的趋势 ${\mathcal{T}}_{\mathrm{{de}}}^{l,i}$ 的投影算子。与FEB（快速经验分解，Fast Empirical Block）类似，FEA（快速经验分析，Fast Empirical Analysis）有两个不同的版本（FEA - f和FEA - w），它们分别通过离散傅里叶变换（DFT，Discrete Fourier Transform）和离散小波变换（DWT，Discrete Wavelet Transform）投影并结合注意力设计来实现，并且可以替代交叉注意力块。FEA(.) 的详细描述将在接下来的3.3节中给出。

The final prediction is the sum of the two refined decomposed components as ${\mathcal{W}}_{\mathcal{S}} \cdot  {\mathcal{X}}_{\mathrm{{de}}}^{M} + {\mathcal{T}}_{\mathrm{{de}}}^{M}$ ,where ${\mathcal{W}}_{\mathcal{S}}$ is to project the deep transformed seasonal component ${\mathcal{X}}_{\mathrm{{de}}}^{M}$ to the target dimension.

最终预测结果是两个精炼分解分量之和，如 ${\mathcal{W}}_{\mathcal{S}} \cdot  {\mathcal{X}}_{\mathrm{{de}}}^{M} + {\mathcal{T}}_{\mathrm{{de}}}^{M}$ 所示，其中 ${\mathcal{W}}_{\mathcal{S}}$ 用于将深度变换后的季节性分量 ${\mathcal{X}}_{\mathrm{{de}}}^{M}$ 投影到目标维度。

### 3.2. Fourier Enhanced Structure

### 3.2. 傅里叶增强结构

Discrete Fourier Transform (DFT) The proposed Fourier Enhanced Structures use discrete Fourier transform (DFT). Let $\mathcal{F}$ denotes the Fourier transform and ${\mathcal{F}}^{-1}$ de-

离散傅里叶变换（Discrete Fourier Transform，DFT） 所提出的傅里叶增强结构采用离散傅里叶变换（DFT）。设 $\mathcal{F}$ 表示傅里叶变换，${\mathcal{F}}^{-1}$ 为

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957f69-db4d-7f34-af19-bfc9e93e263b_3.jpg?x=895&y=183&w=704&h=183&r=0"/>

Figure 3. Frequency Enhanced Block with Fourier transform (FEB-f) structure.

图 3. 具有傅里叶变换的频率增强块（Frequency Enhanced Block with Fourier transform，FEB - f）结构。

<img src="https://cdn.noedgeai.com/01957f69-db4d-7f34-af19-bfc9e93e263b_3.jpg?x=896&y=474&w=701&h=175&r=0"/>

Figure 4. Frequency Enhanced Attention with Fourier transform (FEA-f) structure, $\sigma \left( \cdot \right)$ is the activation function.

图 4. 具有傅里叶变换的频率增强注意力（Frequency Enhanced Attention with Fourier transform，FEA - f）结构，$\sigma \left( \cdot \right)$ 为激活函数。

<!-- Media -->

notes the inverse Fourier transform. Given a sequence of real numbers ${x}_{n}$ in time domain,where $n = 1,2\ldots N$ . DFT is defined as ${X}_{l} = \mathop{\sum }\limits_{{n = 0}}^{{N - 1}}{x}_{n}{e}^{-{i\omega ln}}$ ,where $i$ is the imaginary unit and ${X}_{l},l = 1,2\ldots L$ is a sequence of complex numbers in the frequency domain. Similarly, the inverse DFT is defined as ${x}_{n} = \mathop{\sum }\limits_{{l = 0}}^{{L - 1}}{X}_{l}{e}^{i\omega ln}$ . The complexity of DFT is $O\left( {N}^{2}\right)$ . With fast Fourier transform (FFT), the computation complexity can be reduced to $O\left( {N\log N}\right)$ . Here a random subset of the Fourier basis is used and the scale of the subset is bounded by a scalar. When we choose the mode index before DFT and reverse DFT operations, the computation complexity can be further reduced to $O\left( N\right)$ .

注意到傅里叶逆变换。给定一个时域中的实数序列${x}_{n}$，其中$n = 1,2\ldots N$。离散傅里叶变换（DFT）定义为${X}_{l} = \mathop{\sum }\limits_{{n = 0}}^{{N - 1}}{x}_{n}{e}^{-{i\omega ln}}$，其中$i$是虚数单位，${X}_{l},l = 1,2\ldots L$是频域中的一个复数序列。类似地，离散傅里叶逆变换（IDFT）定义为${x}_{n} = \mathop{\sum }\limits_{{l = 0}}^{{L - 1}}{X}_{l}{e}^{i\omega ln}$。离散傅里叶变换的复杂度为$O\left( {N}^{2}\right)$。使用快速傅里叶变换（FFT），计算复杂度可以降低到$O\left( {N\log N}\right)$。这里使用了傅里叶基的一个随机子集，并且该子集的规模由一个标量界定。当我们在离散傅里叶变换和离散傅里叶逆变换操作之前选择模式索引时，计算复杂度可以进一步降低到$O\left( N\right)$。

Frequency Enhanced Block with Fourier Transform (FEB-f) The FEB-f is used in both encoder and decoder as shown in Figure 2. The input $\left( {\mathbf{x} \in  {\mathbb{R}}^{N \times  D}}\right)$ of the FEB-f block is first linearly projected with $\mathbf{w} \in  {\mathbb{R}}^{D \times  D}$ ,so $\mathbf{q} = \mathbf{x} \cdot  \mathbf{w}$ . Then $\mathbf{q}$ is converted from the time domain to the frequency domain. The Fourier transform of $\mathbf{q}$ is denoted as $\mathbf{Q} \in  {\mathbb{C}}^{N \times  D}$ . In frequency domain,only the randomly selected $M$ modes are kept so we use a select operator as

基于傅里叶变换的频率增强模块（Frequency Enhanced Block with Fourier Transform，FEB - f）如图2所示，FEB - f模块在编码器和解码器中均有使用。FEB - f模块的输入$\left( {\mathbf{x} \in  {\mathbb{R}}^{N \times  D}}\right)$首先与$\mathbf{w} \in  {\mathbb{R}}^{D \times  D}$进行线性投影，因此得到$\mathbf{q} = \mathbf{x} \cdot  \mathbf{w}$。然后将$\mathbf{q}$从时域转换到频域。$\mathbf{q}$的傅里叶变换记为$\mathbf{Q} \in  {\mathbb{C}}^{N \times  D}$。在频域中，仅保留随机选择的$M$个模式，因此我们使用一个选择算子

$$
\widetilde{\mathbf{Q}} = \operatorname{Select}\left( \mathbf{Q}\right)  = \operatorname{Select}\left( {\mathcal{F}\left( \mathbf{q}\right) }\right) , \tag{3}
$$

where $\widetilde{\mathbf{Q}} \in  {\mathbb{C}}^{M \times  D}$ and $M <  < N$ . Then,the FEB-f is defined as

其中 $\widetilde{\mathbf{Q}} \in  {\mathbb{C}}^{M \times  D}$ 和 $M <  < N$ 。然后，FEB - f（有限元边界 - f）定义为

$$
\text{ FEB-f }\left( \mathbf{q}\right)  = {\mathcal{F}}^{-1}\left( {\operatorname{Padding}\left( {\widetilde{\mathbf{Q}} \odot  \mathbf{R}}\right) }\right) , \tag{4}
$$

where $\mathbf{R} \in  {\mathbb{C}}^{D \times  D \times  M}$ is a parameterized kernel initialized randomly. Let $\mathbf{Y} = \mathbf{Q} \odot  \mathbf{C}$ ,with $\mathbf{Y} \in  {\mathbb{C}}^{M \times  D}$ . The production operator $\odot$ is defined as: ${Y}_{m,{d}_{o}} = \mathop{\sum }\limits_{{{d}_{i} = 0}}^{D}{Q}_{m,{d}_{i}}$ . ${R}_{{d}_{i},{d}_{o},m}$ ,where ${d}_{i} = 1,2\ldots D$ is the input channel and ${d}_{o} = 1,2\ldots D$ is the output channel. The result of $\mathbf{Q} \odot  \mathbf{R}$ is then zero-padded to ${\mathbb{C}}^{N \times  D}$ before performing inverse Fourier transform back to the time domain. The structure is shown in Figure 3.

其中 $\mathbf{R} \in  {\mathbb{C}}^{D \times  D \times  M}$ 是一个随机初始化的参数化核。设 $\mathbf{Y} = \mathbf{Q} \odot  \mathbf{C}$，其中 $\mathbf{Y} \in  {\mathbb{C}}^{M \times  D}$。卷积算子 $\odot$ 定义为：${Y}_{m,{d}_{o}} = \mathop{\sum }\limits_{{{d}_{i} = 0}}^{D}{Q}_{m,{d}_{i}}$。${R}_{{d}_{i},{d}_{o},m}$，其中 ${d}_{i} = 1,2\ldots D$ 是输入通道，${d}_{o} = 1,2\ldots D$ 是输出通道。然后，在执行逆傅里叶变换回到时域之前，将 $\mathbf{Q} \odot  \mathbf{R}$ 的结果零填充到 ${\mathbb{C}}^{N \times  D}$。该结构如图 3 所示。

Frequency Enhanced Attention with Fourier Transform (FEA-f) We use the expression of the canonical transformer. The input: queries, keys, values are denoted as $\mathbf{q} \in  {\mathbb{R}}^{L \times  D},\mathbf{k} \in  {\mathbb{R}}^{L \times  D},\mathbf{v} \in  {\mathbb{R}}^{L \times  D}$ . In cross-attention, the queries come from the decoder and can be obtained by $\mathbf{q} = {\mathbf{x}}_{en} \cdot  {\mathbf{w}}_{q}$ ,where ${\mathbf{w}}_{q} \in  {\mathbb{R}}^{D \times  D}$ . The keys and values are from the encoder and can be obtained by $\mathbf{k} = {\mathbf{x}}_{de} \cdot  {\mathbf{w}}_{k}$ and $\mathbf{v} = {\mathbf{x}}_{de} \cdot  {\mathbf{w}}_{v}$ ,where ${\mathbf{w}}_{k},{\mathbf{w}}_{v} \in  {\mathbb{R}}^{D \times  D}$ . Formally,the canonical attention can be written as

基于傅里叶变换的频率增强注意力机制（Frequency Enhanced Attention with Fourier Transform，FEA - f）我们采用经典Transformer的表达式。输入：查询（queries）、键（keys）和值（values）表示为$\mathbf{q} \in  {\mathbb{R}}^{L \times  D},\mathbf{k} \in  {\mathbb{R}}^{L \times  D},\mathbf{v} \in  {\mathbb{R}}^{L \times  D}$。在交叉注意力机制中，查询来自解码器，可通过$\mathbf{q} = {\mathbf{x}}_{en} \cdot  {\mathbf{w}}_{q}$获得，其中${\mathbf{w}}_{q} \in  {\mathbb{R}}^{D \times  D}$。键和值来自编码器，可分别通过$\mathbf{k} = {\mathbf{x}}_{de} \cdot  {\mathbf{w}}_{k}$和$\mathbf{v} = {\mathbf{x}}_{de} \cdot  {\mathbf{w}}_{v}$获得，其中${\mathbf{w}}_{k},{\mathbf{w}}_{v} \in  {\mathbb{R}}^{D \times  D}$。形式上，经典注意力机制可写为

$$
\operatorname{Atten}\left( {\mathbf{q},\mathbf{k},\mathbf{v}}\right)  = \operatorname{Softmax}\left( \frac{\mathbf{q}{\mathbf{k}}^{\top }}{\sqrt{{d}_{q}}}\right) \mathbf{v}. \tag{5}
$$

In FEA-f, we convert the queries, keys, and values with Fourier Transform and perform a similar attention mechanism in the frequency domain,by randomly selecting $\mathrm{M}$ modes. We denote the selected version after Fourier Transform as $\widetilde{\mathbf{Q}} \in  {\mathbb{C}}^{M \times  D},\widetilde{\mathbf{K}} \in  {\mathbb{C}}^{M \times  D},\widetilde{\mathbf{V}} \in  {\mathbb{C}}^{M \times  D}$ . The FEA-f is defined as

在FEA - f（频域高效注意力机制 - f）中，我们使用傅里叶变换（Fourier Transform）对查询、键和值进行转换，并通过随机选择$\mathrm{M}$个模式在频域中执行类似的注意力机制。我们将傅里叶变换后选择的版本表示为$\widetilde{\mathbf{Q}} \in  {\mathbb{C}}^{M \times  D},\widetilde{\mathbf{K}} \in  {\mathbb{C}}^{M \times  D},\widetilde{\mathbf{V}} \in  {\mathbb{C}}^{M \times  D}$。FEA - f定义如下

$$
\widetilde{\mathbf{Q}} = \operatorname{Select}\left( {\mathcal{F}\left( \mathbf{q}\right) }\right) 
$$

$$
\widetilde{\mathbf{K}} = \operatorname{Select}\left( {\mathcal{F}\left( \mathbf{k}\right) }\right)  \tag{6}
$$

$$
\widetilde{\mathbf{V}} = \operatorname{Select}\left( {\mathcal{F}\left( \mathbf{v}\right) }\right) 
$$

$$
\operatorname{FEA-f}\left( {\mathbf{q},\mathbf{k},\mathbf{v}}\right)  = {\mathcal{F}}^{-1}\left( {\operatorname{Padding}\left( {\sigma \left( {\widetilde{\mathbf{Q}} \cdot  {\widetilde{\mathbf{K}}}^{\top }}\right)  \cdot  \widetilde{\mathbf{V}}}\right) }\right) , \tag{7}
$$

where $\sigma$ is the activation function. We use softmax or tanh for activation, since their converging performance differs in different data sets. Let $\mathbf{Y} = \sigma \left( {\widetilde{\mathbf{Q}} \cdot  {\widetilde{\mathbf{K}}}^{\top }}\right)  \cdot  \widetilde{\mathbf{V}}$ ,and $\mathbf{Y} \in  {\mathbb{C}}^{M \times  D}$ needs to be zero-padded to ${\mathbb{C}}^{L \times  D}$ before performing inverse Fourier transform. The FEA-f structure is shown in Figure 4.

其中 $\sigma$ 是激活函数。我们使用 softmax（软最大值函数）或 tanh（双曲正切函数）进行激活，因为它们在不同数据集中的收敛性能有所不同。设 $\mathbf{Y} = \sigma \left( {\widetilde{\mathbf{Q}} \cdot  {\widetilde{\mathbf{K}}}^{\top }}\right)  \cdot  \widetilde{\mathbf{V}}$，并且 $\mathbf{Y} \in  {\mathbb{C}}^{M \times  D}$ 在进行傅里叶逆变换之前需要零填充至 ${\mathbb{C}}^{L \times  D}$。FEA - f 结构如图 4 所示。

### 3.3. Wavelet Enhanced Structure

### 3.3. 小波增强结构

Discrete Wavelet Transform (DWT) While the Fourier transform creates a representation of the signal in the frequency domain, the Wavelet transform creates a representation in both the frequency and time domain, allowing efficient access of localized information of the signal. The mul-tiwavelet transform synergizes the advantages of orthogonal polynomials as well as wavelets. For a given $f\left( x\right)$ , the multiwavelet coefficients at the scale $n$ can be defined as ${\mathbf{s}}_{l}^{n} = {\left\lbrack  {\left\langle  f,{\phi }_{il}^{n}\right\rangle  }_{{\mu }_{n}}\right\rbrack  }_{i = 0}^{k - 1},{\mathbf{d}}_{l}^{n} = {\left\lbrack  {\left\langle  f,{\psi }_{il}^{n}\right\rangle  }_{{\mu }_{n}}\right\rbrack  }_{i = 0}^{k - 1}$ ,respectively,w.r.t. measure ${\mu }_{n}$ with ${\mathbf{s}}_{l}^{n},{\mathbf{d}}_{l}^{n} \in  {\mathbb{R}}^{k \times  {2}^{n}}.{\phi }_{il}^{n}$ are wavelet orthonormal basis of piecewise polynomials. The decomposition/reconstruction across scales is defined as

离散小波变换（DWT） 傅里叶变换在频域中创建信号的表示，而小波变换则在频域和时域中都创建表示，从而能够有效地获取信号的局部信息。多小波变换结合了正交多项式和小波的优点。对于给定的 $f\left( x\right)$，在尺度 $n$ 下的多小波系数可以分别定义为 ${\mathbf{s}}_{l}^{n} = {\left\lbrack  {\left\langle  f,{\phi }_{il}^{n}\right\rangle  }_{{\mu }_{n}}\right\rbrack  }_{i = 0}^{k - 1},{\mathbf{d}}_{l}^{n} = {\left\lbrack  {\left\langle  f,{\psi }_{il}^{n}\right\rangle  }_{{\mu }_{n}}\right\rbrack  }_{i = 0}^{k - 1}$，相对于测度 ${\mu }_{n}$，其中 ${\mathbf{s}}_{l}^{n},{\mathbf{d}}_{l}^{n} \in  {\mathbb{R}}^{k \times  {2}^{n}}.{\phi }_{il}^{n}$ 是分段多项式的小波标准正交基。跨尺度的分解/重构定义为

$$
{\mathbf{s}}_{l}^{n} = {H}^{\left( 0\right) }{\mathbf{s}}_{2l}^{n + 1} + {H}^{\left( 1\right) }{\mathbf{s}}_{{2l} + 1}^{n + 1},
$$

$$
{\mathbf{s}}_{2l}^{n + 1} = {\sum }^{\left( 0\right) }\left( {{H}^{\left( 0\right) T}{\mathbf{s}}_{l}^{n} + {G}^{\left( 0\right) T}{\mathbf{d}}_{l}^{n}}\right) , \tag{8}
$$

$$
{\mathbf{d}}_{l}^{n} = {G}^{\left( 0\right) }{\mathbf{s}}_{2l}^{n + 1} + {H}^{\left( 1\right) }{\mathbf{s}}_{{2l} + 1}^{n + 1},
$$

$$
{\mathbf{s}}_{{2l} + 1}^{n + 1} = {\sum }^{\left( 1\right) }\left( {{H}^{\left( 1\right) T}{\mathbf{s}}_{l}^{n} + {G}^{\left( 1\right) T}{\mathbf{d}}_{l}^{n}}\right) ,
$$

where $\left( {{H}^{\left( 0\right) },{H}^{\left( 1\right) },{G}^{\left( 0\right) },{G}^{\left( 1\right) }}\right)$ are linear coefficients for multiwavelet decomposition filters. They are fixed matrices

其中 $\left( {{H}^{\left( 0\right) },{H}^{\left( 1\right) },{G}^{\left( 0\right) },{G}^{\left( 1\right) }}\right)$ 是多小波分解滤波器的线性系数。它们是固定矩阵

<!-- Media -->

<!-- figureText: Reconstruction -->

<img src="https://cdn.noedgeai.com/01957f69-db4d-7f34-af19-bfc9e93e263b_4.jpg?x=912&y=185&w=680&h=695&r=0"/>

Figure 5. Top Left: Wavelet frequency enhanced block decomposition stage. Top Right: Wavelet block reconstruction stage shared by FEB-w and FEA-w. Bottom: Wavelet frequency enhanced cross attention decomposition stage.

图 5。左上：小波频率增强块分解阶段。右上：FEB - w 和 FEA - w 共享的小波块重建阶段。底部：小波频率增强交叉注意力分解阶段。

<!-- Media -->

used for wavelet decomposition. The multiwavelet representation of a signal can be obtained by the tensor product of multiscale and multiwavelet basis. Note that the basis at various scales are coupled by the tensor product, so we need to untangle it. Inspired by (Gupta et al., 2021), we adapt a non-standard wavelet representation to reduce the model complexity. For a map function $F\left( x\right)  = {x}^{\prime }$ ,the map under multiwavelet domain can be written as

用于小波分解。信号的多小波表示可以通过多尺度和多小波基的张量积得到。注意，不同尺度的基通过张量积耦合，因此我们需要将其解开。受（古普塔等人，2021 年）的启发，我们采用一种非标准的小波表示来降低模型复杂度。对于一个映射函数 $F\left( x\right)  = {x}^{\prime }$，多小波域下的映射可以写成

$$
{U}_{dl}^{n} = {A}_{n}{d}_{l}^{n} + {B}_{n}{s}_{l}^{n},\;{U}_{\widehat{s}l}^{n} = {C}_{n}{d}_{l}^{n},\;{U}_{sl}^{L} = \bar{F}{s}_{l}^{L}, \tag{9}
$$

where $\left( {{U}_{sl}^{n},{U}_{dl}^{n},{s}_{l}^{n},{d}_{l}^{n}}\right)$ are the multiscale,multiwavelet coefficients, $L$ is the coarsest scale under recursive decomposition,and ${A}_{n},{B}_{n},{C}_{n}$ are three independent FEB-f blocks modules used for processing different signal during decomposition and reconstruction. Here $\bar{F}$ is a single-layer of perceptrons which processes the remaining coarsest signal after $L$ decomposed steps. More designed detail is described in Appendix D.

其中$\left( {{U}_{sl}^{n},{U}_{dl}^{n},{s}_{l}^{n},{d}_{l}^{n}}\right)$是多尺度、多小波系数，$L$是递归分解下的最粗尺度，${A}_{n},{B}_{n},{C}_{n}$是三个独立的FEB - f块模块，用于在分解和重构过程中处理不同的信号。这里$\bar{F}$是一个单层感知器，用于处理经过$L$次分解步骤后剩余的最粗信号。更多设计细节见附录D。

Frequency Enhanced Block with Wavelet Transform (FEB-w) The overall FEB-w architecture is shown in Figure 5. It differs from FEB-f in the recursive mechanism: the input is decomposed into 3 parts recursively and operates individually. For the wavelet decomposition part, we implement the fixed Legendre wavelets basis decomposition matrix. Three FEB-f modules are used to process the resulting high-frequency part, low-frequency part, and remaining part from wavelet decomposition respectively. For each cycle $L$ ,it produces a processed high-frequency tensor ${Ud}\left( L\right)$ ,a processed low-frequency frequency tensor ${Us}\left( L\right)$ ,and the raw low-frequency tensor $X\left( {L + 1}\right)$ . This is a ladder-down approach, and the decomposition stage performs the decimation of the signal by a factor of $1/2$ ,running for a maximum of $L$ cycles,where $L < {\log }_{2}\left( M\right)$ for a given input sequence of size $M$ . In practice, $L$ is set as a fixed argument parameter. The three sets of FEB-f blocks are shared during different decomposition cycles $L$ . For the wavelet reconstruction part, we recursively build up our output tensor as well. For each cycle $L$ ,we combine $X\left( {L + 1}\right)$ , ${Us}\left( L\right)$ ,and ${Ud}\left( L\right)$ produced from the decomposition part and produce $X\left( L\right)$ for the next reconstruction cycle. For each cycle, the length dimension of the signal tensor is increased by 2 times.

基于小波变换的频率增强模块（FEB - w） 整体的FEB - w架构如图5所示。它与FEB - f的递归机制不同：输入被递归地分解为3部分并分别进行处理。对于小波分解部分，我们实现了固定的勒让德小波基分解矩阵。三个FEB - f模块分别用于处理小波分解得到的高频部分、低频部分和剩余部分。对于每个循环$L$，它会产生一个处理后的高频张量${Ud}\left( L\right)$、一个处理后的低频张量${Us}\left( L\right)$和原始低频张量$X\left( {L + 1}\right)$。这是一种逐级递减的方法，分解阶段将信号以$1/2$为因子进行抽取，最多运行$L$个循环，其中对于给定大小为$M$的输入序列，有$L < {\log }_{2}\left( M\right)$。在实践中，$L$被设置为一个固定的参数。三组FEB - f模块在不同的分解循环$L$中共享。对于小波重构部分，我们也递归地构建输出张量。对于每个循环$L$，我们将分解部分产生的$X\left( {L + 1}\right)$、${Us}\left( L\right)$和${Ud}\left( L\right)$组合起来，为下一个重构循环产生$X\left( L\right)$。对于每个循环，信号张量的长度维度增加2倍。

Frequency Enhanced Attention with Wavelet Transform (FEA-w) FEA-w contains the decomposition stage and reconstruction stage like FEB-w. Here we keep the reconstruction stage unchanged. The only difference lies in the decomposition stage. The same decomposed matrix is used to decompose $\mathbf{q},\mathbf{k},\mathbf{v}$ signal separately,and $\mathbf{q},\mathbf{k},\mathbf{v}$ share the same sets of module to process them as well. As shown above, a frequency enhanced block with wavelet decomposition block (FEB-w) contains three FEB-f blocks for the signal process. We can view the FEB-f as a substitution of self-attention mechanism. We use a straightforward way to build the frequency enhanced cross attention with wavelet decomposition,substituting each FEB-f with a FEA-f module. Besides, another FEA-f module is added to process the coarsest remaining $q\left( L\right) ,k\left( L\right) ,v\left( L\right)$ signal.

基于小波变换的频率增强注意力机制（FEA - w） 与FEB - w类似，FEA - w包含分解阶段和重构阶段。在此，我们保持重构阶段不变。唯一的区别在于分解阶段。使用相同的分解矩阵分别对$\mathbf{q},\mathbf{k},\mathbf{v}$信号进行分解，并且$\mathbf{q},\mathbf{k},\mathbf{v}$也共享同一组模块来处理它们。如上所示，带有小波分解块的频率增强块（FEB - w）包含三个用于信号处理的FEB - f块。我们可以将FEB - f视为自注意力机制的替代方案。我们采用一种直接的方法来构建基于小波分解的频率增强交叉注意力机制，用一个FEA - f模块替代每个FEB - f。此外，还添加了另一个FEA - f模块来处理最粗糙的剩余$q\left( L\right) ,k\left( L\right) ,v\left( L\right)$信号。

### 3.4. Mixture of Experts for Seasonal-Trend Decomposition

### 3.4. 用于季节性 - 趋势分解的专家混合模型

Because of the commonly observed complex periodic pattern coupled with the trend component on real-world data, extracting the trend can be hard with fixed window average pooling. To overcome such a problem, we design a Mixture Of Experts Decomposition block (MOEDecomp). It contains a set of average filters with different sizes to extract multiple trend components from the input signal and a set of data-dependent weights for combining them as the final trend. Formally, we have

由于在现实世界数据中普遍观察到复杂的周期性模式与趋势成分相互交织，使用固定窗口平均池化来提取趋势可能会很困难。为了克服这一问题，我们设计了一个专家混合分解模块（Mixture Of Experts Decomposition，MOEDecomp）。它包含一组不同大小的平均滤波器，用于从输入信号中提取多个趋势成分，以及一组依赖于数据的权重，用于将这些成分组合成最终的趋势。形式上，我们有

$$
{\mathbf{X}}_{\text{trend }} = \operatorname{Softmax}\left( {L\left( x\right) }\right)  * \left( {F\left( x\right) }\right) , \tag{10}
$$

where $F\left( \cdot \right)$ is a set of average pooling filters and $\operatorname{Softmax}\left( {L\left( x\right) }\right)$ is the weights for mixing these extracted trends.

其中 $F\left( \cdot \right)$ 是一组平均池化滤波器，$\operatorname{Softmax}\left( {L\left( x\right) }\right)$ 是混合这些提取的趋势的权重。

### 3.5. Complexity Analysis

### 3.5. 复杂度分析

For FEDformer-f, the computational complexity for time and memory is $O\left( L\right)$ with a fixed number of randomly selected modes in FEB & FEA blocks. We set modes number $M = {64}$ as default value. Though the complexity of full DFT transformation by FFT is $(O\left( {L\log \left( L\right) }\right)$ ,our model only needs $O\left( L\right)$ cost and memory complexity with the pre-selected set of Fourier basis for quick implementation. For FEDformer-w, when we set the recursive decompose step to a fixed number $L$ and use a fixed number of randomly selected modes the same as FEDformer-f, the time complexity and memory usage are $O\left( L\right)$ as well. In practice,we choose $L = 3$ and modes number $M = {64}$ as default value. The comparisons of the time complexity and memory usage in training and the inference steps in testing are summarized in Table 1. It can be seen that the proposed FEDformer achieves the best overall complexity among Transformer-based forecasting models.

对于FEDformer - f，在FEB和FEA模块中随机选择固定数量的模式时，时间和内存的计算复杂度为$O\left( L\right)$。我们将模式数量$M = {64}$设为默认值。尽管通过快速傅里叶变换（FFT）进行完整离散傅里叶变换（DFT）的复杂度为$(O\left( {L\log \left( L\right) }\right)$，但我们的模型仅需$O\left( L\right)$的计算成本和内存复杂度，通过预先选择的傅里叶基集可实现快速计算。对于FEDformer - w，当我们将递归分解步骤设置为固定数量$L$，并使用与FEDformer - f相同的固定数量的随机选择模式时，时间复杂度和内存使用量同样为$O\left( L\right)$。在实践中，我们选择$L = 3$和模式数量$M = {64}$作为默认值。表1总结了训练阶段和测试推理阶段的时间复杂度和内存使用情况的比较。可以看出，所提出的FEDformer在基于Transformer的预测模型中实现了最佳的整体复杂度。

<!-- Media -->

Table 1. Complexity analysis of different forecasting models.

表1. 不同预测模型的复杂度分析。

<table><tr><td rowspan="2">Methods</td><td colspan="2">Training</td><td>Testing</td></tr><tr><td>Time</td><td>Memory</td><td>Steps</td></tr><tr><td>FEDformer</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td><td>1</td></tr><tr><td>Autoformer</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>1</td></tr><tr><td>Informer</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>1</td></tr><tr><td>Transformer</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$L$</td></tr><tr><td>LogTrans</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>1</td></tr><tr><td>Reformer</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$L$</td></tr><tr><td>LSTM</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td><td>$L$</td></tr></table>

<table><tbody><tr><td rowspan="2">方法</td><td colspan="2">训练</td><td>测试</td></tr><tr><td>时间</td><td>内存</td><td>步骤</td></tr><tr><td>FEDformer（联邦前馈变换器）</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td><td>1</td></tr><tr><td>Autoformer（自动变换器）</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>1</td></tr><tr><td>Informer（信息者变换器）</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>1</td></tr><tr><td>Transformer（变换器）</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$L$</td></tr><tr><td>LogTrans（对数变换器）</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>1</td></tr><tr><td>Reformer（革新者变换器）</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$L$</td></tr><tr><td>长短期记忆网络（LSTM）</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td><td>$L$</td></tr></tbody></table>

<!-- Media -->

## 4. Experiments

## 4. 实验

To evaluate the proposed FEDformer, we conduct extensive experiments on six popular real-world datasets, including energy, economics, traffic, weather, and disease. Since classic models like ARIMA and basic RNN/CNN models perform relatively inferior as shown in (Zhou et al., 2021) and (Wu et al., 2021), we mainly include four state-of-the-art transformer-based models for comparison, i.e., Auto-former (Wu et al., 2021), Informer (Zhou et al., 2021), Log-Trans (Li et al., 2019) and Reformer (Kitaev et al., 2020) as baseline models. Note that since Autoformer holds the best performance in all the six benchmarks, it is used as the main baseline model for comparison. More details about baseline models, datasets, and implementation are described in Appendix A.2, F.1, and F.2, respectively.

为了评估所提出的FEDformer（联邦变换器），我们在六个流行的真实世界数据集上进行了广泛的实验，这些数据集涵盖能源、经济、交通、天气和疾病领域。由于如（Zhou等人，2021年）和（Wu等人，2021年）所示，像ARIMA（自回归积分滑动平均模型）这样的经典模型以及基本的RNN（循环神经网络）/CNN（卷积神经网络）模型表现相对较差，我们主要纳入了四个最先进的基于变换器（Transformer）的模型进行比较，即Autoformer（自动变换器，Wu等人，2021年）、Informer（信息变换器，Zhou等人，2021年）、Log - Trans（对数变换模型，Li等人，2019年）和Reformer（改革者模型，Kitaev等人，2020年）作为基线模型。请注意，由于Autoformer在所有六个基准测试中表现最佳，因此将其用作主要的基线模型进行比较。关于基线模型、数据集和实现的更多详细信息分别在附录A.2、F.1和F.2中描述。

### 4.1. Main Results

### 4.1. 主要结果

For better comparison, we follow the experiment settings of Autoformer in (Wu et al., 2021) where the input length is fixed to 96 , and the prediction lengths for both training and evaluation are fixed to be 96, 192, 336, and 720, respectively.

为了更好地进行比较，我们遵循了《Autoformer》（吴等人，2021年）中的实验设置，其中输入长度固定为96，训练和评估的预测长度分别固定为96、192、336和720。

Multivariate Results For the multivariate forecasting, FEDformer achieves the best performance on all six benchmark datasets at all horizons as shown in Table 2. Compared with Autoformer, the proposed FEDformer yields an overall 14.8% relative MSE reduction. It is worth noting that for some of the datasets, such as Exchange and ILI, the improvement is even more significant (over 20%). Note that the Exchange dataset does not exhibit clear periodicity in its time series, but FEDformer can still achieve superior performance. Overall, the improvement made by FEDformer is consistent with varying horizons, implying its strength in long term forecasting. More detailed results on ETT full benchmark are provided in Appendix F.3.

多变量结果 对于多变量预测，如表2所示，FEDformer在所有六个基准数据集的所有预测范围内均取得了最佳性能。与Autoformer相比，所提出的FEDformer的均方误差（MSE）总体相对降低了14.8%。值得注意的是，对于某些数据集，如Exchange和ILI，改进更为显著（超过20%）。请注意，Exchange数据集的时间序列没有明显的周期性，但FEDformer仍然可以实现卓越的性能。总体而言，FEDformer的改进在不同的预测范围内是一致的，这意味着它在长期预测方面具有优势。附录F.3提供了ETT完整基准的更详细结果。

<!-- Media -->

Table 2. Multivariate long-term series forecasting results on six datasets with input length $I = {96}$ and prediction length $O \in$ $\{ {96},{192},{336},{720}\}$ (For ILI dataset,we use input length $I = {36}$ and prediction length $O \in  \{ {24},{36},{48},{60}\}$ ). A lower MSE indicates better performance, and the best results are highlighted in bold.

表2. 在六个数据集上输入长度为$I = {96}$、预测长度为$O \in$ $\{ {96},{192},{336},{720}\}$的多变量长期序列预测结果（对于流感样病例（ILI）数据集，我们使用输入长度$I = {36}$和预测长度$O \in  \{ {24},{36},{48},{60}\}$）。均方误差（MSE）越低表示性能越好，最佳结果以粗体突出显示。

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Metric</td><td colspan="4">ETTm2</td><td colspan="4">Electricity</td><td colspan="4">Exchange</td><td colspan="4">Traffic</td><td colspan="3">Weather</td><td colspan="4">ILI</td></tr><tr><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>72096</td><td>192</td><td>336</td><td>720</td><td>24</td><td>36</td><td>48</td><td>60</td></tr><tr><td rowspan="2">FEDformer-f</td><td>MSE</td><td>0.203</td><td>0.269</td><td>0.325</td><td>0.421</td><td>0.193</td><td>0.201</td><td>0.214</td><td>0.246</td><td>0.148</td><td>0.271</td><td>0.460</td><td>1.195</td><td>0.587</td><td>0.604</td><td>0.621</td><td>0.6260.217</td><td>0.276</td><td>0.339</td><td>0.403</td><td>3.228</td><td>2.679</td><td>2.622</td><td>2.857</td></tr><tr><td>MAE</td><td>0.287</td><td>0.328</td><td>0.366</td><td>0.415</td><td>0.308</td><td>0.315</td><td>0.329</td><td>0.355</td><td>0.278</td><td>0.380</td><td>0.500</td><td>0.841</td><td>0.366</td><td>0.373</td><td>0.383</td><td>0.3820.296</td><td>0.336</td><td>0.380</td><td>0.428</td><td>1.260</td><td>1.080</td><td>1.078</td><td>1.157</td></tr><tr><td rowspan="2">FEDformer-w</td><td>MSE</td><td>0.204</td><td>0.316</td><td>0.359</td><td>0.433</td><td>0.183</td><td>0.195</td><td>0.212</td><td>0.231</td><td>0.139</td><td>0.256</td><td>0.426</td><td>1.090</td><td>0.562</td><td>0.562</td><td>0.570</td><td>0.5960.227</td><td>0.295</td><td>0.381</td><td>0.424</td><td>2.203</td><td>2.272</td><td>2.209</td><td>2.545</td></tr><tr><td>MAE</td><td>0.288</td><td>0.363</td><td>0.387</td><td>0.432</td><td>0.297</td><td>0.308</td><td>0.313</td><td>0.343</td><td>0.276</td><td>0.369</td><td>0.464</td><td>0.800</td><td>0.349</td><td>0.346</td><td>0.323</td><td>0.3680.304</td><td>0.363</td><td>0.416</td><td>0.434</td><td>0.963</td><td>0.976</td><td>0.981</td><td>1.061</td></tr><tr><td rowspan="2">Autoformer</td><td>MSE</td><td>0.255</td><td>0.281</td><td>0.339</td><td>0.422</td><td>0.201</td><td>0.222</td><td>0.231</td><td>0.254</td><td>0.197</td><td>0.300</td><td>0.509</td><td>1.447</td><td>0.613</td><td>0.616</td><td>0.622</td><td>0.6600.266</td><td>0.307</td><td>0.359</td><td>0.419</td><td>3.483</td><td>3.103</td><td>2.669</td><td>2,770</td></tr><tr><td>MAE</td><td>0.339</td><td>0.340</td><td>0.372</td><td>0.419</td><td>0.317</td><td>0.334</td><td>0.338</td><td>0.361</td><td>0.323</td><td>0.369</td><td>0.524</td><td>0.941</td><td>0.388</td><td>0.382</td><td>0.337</td><td>0.4080.336</td><td>0.367</td><td>0.395</td><td>0.428</td><td>1.287</td><td>1.148</td><td>1.085</td><td>1.125</td></tr><tr><td rowspan="2">Informer</td><td>MSE</td><td>0.365</td><td>0.533</td><td>1.363</td><td>3.379</td><td>0.274</td><td>0.296</td><td>0.300</td><td>0.373</td><td>0.847</td><td>1.204</td><td>1.672</td><td>2.478</td><td>0.719</td><td>0.696</td><td>0.777</td><td>0.8640.300</td><td>0.598</td><td>0.578</td><td>1.059</td><td>5.764</td><td>4.755</td><td>4.763</td><td>5.264</td></tr><tr><td>MAE</td><td>0.453</td><td>0.563</td><td>0.887</td><td>1.338</td><td>0.368</td><td>0.386</td><td>0.394</td><td>0.439</td><td>0.752</td><td>0.895</td><td>1.036</td><td>1.310</td><td>0.391</td><td>0.379</td><td>0.420</td><td>0.4720.384</td><td>0.544</td><td>0.523</td><td>0.741</td><td>1.677</td><td>1.467</td><td>1.469</td><td>1.564</td></tr><tr><td rowspan="2">LogTrans</td><td>MSE</td><td>0.768</td><td>0.989</td><td>1.334</td><td>3.048</td><td>0.258</td><td>0.266</td><td>0.280</td><td>0.283</td><td>0.968</td><td>1.040</td><td>1.659</td><td>1.941</td><td>0.684</td><td>0.685</td><td>0.7337</td><td>0.7170.458</td><td>0.658</td><td>0.797</td><td>0.869</td><td>4.480</td><td>4.799</td><td>4.800</td><td>5.278</td></tr><tr><td>MAE</td><td>0.642</td><td>0.757</td><td>0.872</td><td>1.328</td><td>0.357</td><td>0.368</td><td>0.380</td><td>0.376</td><td>0.812</td><td>0.851</td><td>1.081</td><td>1.127</td><td>0.384</td><td>0.390</td><td>0.408</td><td>0.3960.490</td><td>0.589</td><td>0.652</td><td>0.675</td><td>1.444</td><td>1.467</td><td>1.468</td><td>1.560</td></tr><tr><td rowspan="2">Reformer</td><td>MSE</td><td>0.658</td><td>1.078</td><td>1.549</td><td>2.631</td><td>0.312</td><td>0.348</td><td>0.350</td><td>0.340</td><td>1.065</td><td>1.188</td><td>1.357</td><td>1.510</td><td>0.732</td><td>0.733</td><td>0.742</td><td>0.7550.689</td><td>0.752</td><td>0.639</td><td>1.130</td><td>4.400</td><td>4.783</td><td>4.832</td><td>4.882</td></tr><tr><td>MAE</td><td>0.619</td><td>0.827</td><td>0.972</td><td>1.242</td><td>0.402</td><td>0.433</td><td>0.433</td><td>0.420</td><td>0.829</td><td>0.906</td><td>0.976</td><td>1.016</td><td>0.423</td><td>0.420</td><td>0.420</td><td>4230.596</td><td>0.638</td><td>0.596</td><td>0.792</td><td>1.382</td><td>1.448</td><td>1.465</td><td>1.483</td></tr></table>

<table><tbody><tr><td rowspan="2">方法</td><td rowspan="2">指标</td><td colspan="4">ETTm2（ETTm2）</td><td colspan="4">电力</td><td colspan="4">交换；交易</td><td colspan="4">交通</td><td colspan="3">天气</td><td colspan="4">流感样疾病（ILI）</td></tr><tr><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>72096</td><td>192</td><td>336</td><td>720</td><td>24</td><td>36</td><td>48</td><td>60</td></tr><tr><td rowspan="2">联邦变压器 - f（FEDformer - f）</td><td>均方误差（MSE）</td><td>0.203</td><td>0.269</td><td>0.325</td><td>0.421</td><td>0.193</td><td>0.201</td><td>0.214</td><td>0.246</td><td>0.148</td><td>0.271</td><td>0.460</td><td>1.195</td><td>0.587</td><td>0.604</td><td>0.621</td><td>0.6260.217</td><td>0.276</td><td>0.339</td><td>0.403</td><td>3.228</td><td>2.679</td><td>2.622</td><td>2.857</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.287</td><td>0.328</td><td>0.366</td><td>0.415</td><td>0.308</td><td>0.315</td><td>0.329</td><td>0.355</td><td>0.278</td><td>0.380</td><td>0.500</td><td>0.841</td><td>0.366</td><td>0.373</td><td>0.383</td><td>0.3820.296</td><td>0.336</td><td>0.380</td><td>0.428</td><td>1.260</td><td>1.080</td><td>1.078</td><td>1.157</td></tr><tr><td rowspan="2">联邦变压器 - w（FEDformer - w）</td><td>均方误差（MSE）</td><td>0.204</td><td>0.316</td><td>0.359</td><td>0.433</td><td>0.183</td><td>0.195</td><td>0.212</td><td>0.231</td><td>0.139</td><td>0.256</td><td>0.426</td><td>1.090</td><td>0.562</td><td>0.562</td><td>0.570</td><td>0.5960.227</td><td>0.295</td><td>0.381</td><td>0.424</td><td>2.203</td><td>2.272</td><td>2.209</td><td>2.545</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.288</td><td>0.363</td><td>0.387</td><td>0.432</td><td>0.297</td><td>0.308</td><td>0.313</td><td>0.343</td><td>0.276</td><td>0.369</td><td>0.464</td><td>0.800</td><td>0.349</td><td>0.346</td><td>0.323</td><td>0.3680.304</td><td>0.363</td><td>0.416</td><td>0.434</td><td>0.963</td><td>0.976</td><td>0.981</td><td>1.061</td></tr><tr><td rowspan="2">自动转换器（Autoformer）</td><td>均方误差（MSE）</td><td>0.255</td><td>0.281</td><td>0.339</td><td>0.422</td><td>0.201</td><td>0.222</td><td>0.231</td><td>0.254</td><td>0.197</td><td>0.300</td><td>0.509</td><td>1.447</td><td>0.613</td><td>0.616</td><td>0.622</td><td>0.6600.266</td><td>0.307</td><td>0.359</td><td>0.419</td><td>3.483</td><td>3.103</td><td>2.669</td><td>2,770</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.339</td><td>0.340</td><td>0.372</td><td>0.419</td><td>0.317</td><td>0.334</td><td>0.338</td><td>0.361</td><td>0.323</td><td>0.369</td><td>0.524</td><td>0.941</td><td>0.388</td><td>0.382</td><td>0.337</td><td>0.4080.336</td><td>0.367</td><td>0.395</td><td>0.428</td><td>1.287</td><td>1.148</td><td>1.085</td><td>1.125</td></tr><tr><td rowspan="2">信息器（Informer）</td><td>均方误差（MSE）</td><td>0.365</td><td>0.533</td><td>1.363</td><td>3.379</td><td>0.274</td><td>0.296</td><td>0.300</td><td>0.373</td><td>0.847</td><td>1.204</td><td>1.672</td><td>2.478</td><td>0.719</td><td>0.696</td><td>0.777</td><td>0.8640.300</td><td>0.598</td><td>0.578</td><td>1.059</td><td>5.764</td><td>4.755</td><td>4.763</td><td>5.264</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.453</td><td>0.563</td><td>0.887</td><td>1.338</td><td>0.368</td><td>0.386</td><td>0.394</td><td>0.439</td><td>0.752</td><td>0.895</td><td>1.036</td><td>1.310</td><td>0.391</td><td>0.379</td><td>0.420</td><td>0.4720.384</td><td>0.544</td><td>0.523</td><td>0.741</td><td>1.677</td><td>1.467</td><td>1.469</td><td>1.564</td></tr><tr><td rowspan="2">对数转换器（LogTrans）</td><td>均方误差（MSE）</td><td>0.768</td><td>0.989</td><td>1.334</td><td>3.048</td><td>0.258</td><td>0.266</td><td>0.280</td><td>0.283</td><td>0.968</td><td>1.040</td><td>1.659</td><td>1.941</td><td>0.684</td><td>0.685</td><td>0.7337</td><td>0.7170.458</td><td>0.658</td><td>0.797</td><td>0.869</td><td>4.480</td><td>4.799</td><td>4.800</td><td>5.278</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.642</td><td>0.757</td><td>0.872</td><td>1.328</td><td>0.357</td><td>0.368</td><td>0.380</td><td>0.376</td><td>0.812</td><td>0.851</td><td>1.081</td><td>1.127</td><td>0.384</td><td>0.390</td><td>0.408</td><td>0.3960.490</td><td>0.589</td><td>0.652</td><td>0.675</td><td>1.444</td><td>1.467</td><td>1.468</td><td>1.560</td></tr><tr><td rowspan="2">改革器（Reformer）</td><td>均方误差（MSE）</td><td>0.658</td><td>1.078</td><td>1.549</td><td>2.631</td><td>0.312</td><td>0.348</td><td>0.350</td><td>0.340</td><td>1.065</td><td>1.188</td><td>1.357</td><td>1.510</td><td>0.732</td><td>0.733</td><td>0.742</td><td>0.7550.689</td><td>0.752</td><td>0.639</td><td>1.130</td><td>4.400</td><td>4.783</td><td>4.832</td><td>4.882</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.619</td><td>0.827</td><td>0.972</td><td>1.242</td><td>0.402</td><td>0.433</td><td>0.433</td><td>0.420</td><td>0.829</td><td>0.906</td><td>0.976</td><td>1.016</td><td>0.423</td><td>0.420</td><td>0.420</td><td>4230.596</td><td>0.638</td><td>0.596</td><td>0.792</td><td>1.382</td><td>1.448</td><td>1.465</td><td>1.483</td></tr></tbody></table>

Table 3. Univariate long-term series forecasting results on six datasets with input length $I = {96}$ and prediction length $O \in$ $\{ {96},{192},{336},{720}\}$ (For ILI dataset,we use input length $I = {36}$ and prediction length $O \in  \{ {24},{36},{48},{60}\}$ ). A lower MSE indicates better performance, and the best results are highlighted in bold.

表3. 在六个数据集上，输入长度为$I = {96}$、预测长度为$O \in$ $\{ {96},{192},{336},{720}\}$的单变量长期序列预测结果（对于流感样病例（ILI）数据集，我们使用的输入长度为$I = {36}$，预测长度为$O \in  \{ {24},{36},{48},{60}\}$）。均方误差（MSE）越低表示性能越好，最佳结果以粗体突出显示。

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Metric</td><td colspan="4">ETTm2</td><td colspan="4">Electricity</td><td colspan="4">Exchange</td><td colspan="4">Traffic</td><td colspan="4">Weather</td><td colspan="4">ILI</td></tr><tr><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>24</td><td>36</td><td>48</td><td>60</td></tr><tr><td rowspan="2">FEDformer-f</td><td>MSE</td><td>0.072</td><td>0.102</td><td>0.130</td><td>0.178</td><td>0.253</td><td>0.282</td><td>0.346</td><td>0.422</td><td>0.154</td><td>0.286</td><td>0.511</td><td>1.301</td><td>0.207</td><td>0.205</td><td>0.219</td><td>0.244</td><td>0.0062</td><td>0.0060</td><td>0.0041</td><td>0.0055</td><td>0.708</td><td>0.584</td><td>0.717</td><td>0.855</td></tr><tr><td>MAE</td><td>0.206</td><td>0.245</td><td>0.279</td><td>0.325</td><td>0.370</td><td>0.386</td><td>0.431</td><td>0.484</td><td>0.304</td><td>0.420</td><td>0.555</td><td>0.879</td><td>0.312</td><td>0.312</td><td>0.323</td><td>0.344</td><td>0.062</td><td>0.062</td><td>0.050</td><td>0.059</td><td>0.627</td><td>0.617</td><td>0.697</td><td>0.774</td></tr><tr><td rowspan="2">FEDformer-w</td><td>MSE</td><td>0.063</td><td>0.110</td><td>0.147</td><td>0.219</td><td>0.262</td><td>0.316</td><td>0.361</td><td>0.448</td><td>0.131</td><td>0.277</td><td>0.426</td><td>1.162</td><td>0.170</td><td>0.173</td><td>0.178</td><td>0.187</td><td>0.0035</td><td>0.0054</td><td>0.008</td><td>0.015</td><td>0.693</td><td>0.554</td><td>0.699</td><td>0.828</td></tr><tr><td>MAE</td><td>0.189</td><td>0.252</td><td>0.301</td><td>0.368</td><td>0.378</td><td>0.410</td><td>0.445</td><td>0.501</td><td>0.284</td><td>0.420</td><td>0.511</td><td>0.832</td><td>0.263</td><td>0.265</td><td>0.266</td><td>0.286</td><td>0.046</td><td>0.059</td><td>0.072</td><td>0.091</td><td>0.629</td><td>0.604</td><td>0.696</td><td>0.770</td></tr><tr><td rowspan="2">Autoformer</td><td>MSE</td><td>0.065</td><td>0.118</td><td>0.154</td><td>0.182</td><td>0.341</td><td>0.345</td><td>0.406</td><td>0.565</td><td>0.241</td><td>0.300</td><td>0.509</td><td>1.260</td><td>0.246</td><td>0.266</td><td>0.263</td><td>0.269</td><td>0.011</td><td>0.0075</td><td>0.0063</td><td>0.0085</td><td>0.948</td><td>0.634</td><td>0.791</td><td>0.874</td></tr><tr><td>MAE</td><td>0.189</td><td>0.256</td><td>0.305</td><td>0.335</td><td>0.438</td><td>0.428</td><td>0.470</td><td>0.581</td><td>0.387</td><td>0.369</td><td>0.524</td><td>0.867</td><td>0.346</td><td>0.370</td><td>0.371</td><td>0.372</td><td>0.081</td><td>0.067</td><td>0.062</td><td>0.070</td><td>0.732</td><td>0.650</td><td>0.752</td><td>0.797</td></tr><tr><td rowspan="2">Informer</td><td>MSE</td><td>0.080</td><td>0.112</td><td>0.166</td><td>0.228</td><td>0.258</td><td>0.285</td><td>0.336</td><td>0.607</td><td>1.327</td><td>1.258</td><td>2.179</td><td>1.280</td><td>0.257</td><td>0.299</td><td>0.312</td><td>0.366</td><td>0.004</td><td>0.002</td><td>0.004</td><td>0.003</td><td>5.282</td><td>4.554</td><td>4.273</td><td>5.214</td></tr><tr><td>MAE</td><td>0.217</td><td>0.259</td><td>0.314</td><td>0.380</td><td>0.367</td><td>0.388</td><td>0.423</td><td>0.599</td><td>0.944</td><td>0.924</td><td>1.296</td><td>0.953</td><td>0.353</td><td>0.376</td><td>0.387</td><td>0.436</td><td>0.044</td><td>0.040</td><td>0.049</td><td>0.042</td><td>2.050</td><td>1.916</td><td>1.846</td><td>2.057</td></tr><tr><td rowspan="2">LogTrans</td><td>MSE</td><td>0.075</td><td>0.129</td><td>0.154</td><td>0.160</td><td>0.288</td><td>0.432</td><td>0.430</td><td>0.491</td><td>0.237</td><td>0.738</td><td>2.018</td><td>2.405</td><td>0.226</td><td>0.314</td><td>0.387</td><td>0.437</td><td>0.0046</td><td>0.0060</td><td>0.0060</td><td>0.007</td><td>3.607</td><td>2.407</td><td>3.106</td><td>3.698</td></tr><tr><td>MAE</td><td>0.208</td><td>0.275</td><td>0.302</td><td>0.322</td><td>0.393</td><td>0.483</td><td>0.483</td><td>0.531</td><td>0.377</td><td>0.619</td><td>1.070</td><td>1.175</td><td>0.317</td><td>0.408</td><td>0.453</td><td>0.491</td><td>0.052</td><td>0.060</td><td>0.054</td><td>0.059</td><td>1.662</td><td>1.363</td><td>1.575</td><td>1.733</td></tr><tr><td rowspan="2">Reformer</td><td>MSE</td><td>0.077</td><td>0.138</td><td>0.160</td><td>0.168</td><td>0.275</td><td>0.304</td><td>0.370</td><td>0.460</td><td>0.298</td><td>0.777</td><td>1.833</td><td>1.203</td><td>0.313</td><td>0.386</td><td>0.423</td><td>0.378</td><td>0.012</td><td>0.0098</td><td>0.013</td><td>0.011</td><td>3.838</td><td>2.934</td><td>3.755</td><td>4.162</td></tr><tr><td>MAE</td><td>0.214</td><td>0.290</td><td>0.313</td><td>0.334</td><td>0.379</td><td>0.402</td><td>0.448</td><td>0.511</td><td>0.444</td><td>0.719</td><td>1.128</td><td>0.956</td><td>0.383</td><td>0.453</td><td>0.468</td><td>0.433</td><td>0.087</td><td>0.044</td><td>0.100</td><td>0.083</td><td>1.720</td><td>1.520</td><td>1.749</td><td>1.847</td></tr></table>

<table><tbody><tr><td rowspan="2">方法</td><td rowspan="2">指标</td><td colspan="4">ETTm2</td><td colspan="4">电力</td><td colspan="4">交换；交易</td><td colspan="4">交通</td><td colspan="4">天气</td><td colspan="4">流感样疾病（ILI）</td></tr><tr><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>24</td><td>36</td><td>48</td><td>60</td></tr><tr><td rowspan="2">联邦变换器 - f（FEDformer - f）</td><td>均方误差（MSE）</td><td>0.072</td><td>0.102</td><td>0.130</td><td>0.178</td><td>0.253</td><td>0.282</td><td>0.346</td><td>0.422</td><td>0.154</td><td>0.286</td><td>0.511</td><td>1.301</td><td>0.207</td><td>0.205</td><td>0.219</td><td>0.244</td><td>0.0062</td><td>0.0060</td><td>0.0041</td><td>0.0055</td><td>0.708</td><td>0.584</td><td>0.717</td><td>0.855</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.206</td><td>0.245</td><td>0.279</td><td>0.325</td><td>0.370</td><td>0.386</td><td>0.431</td><td>0.484</td><td>0.304</td><td>0.420</td><td>0.555</td><td>0.879</td><td>0.312</td><td>0.312</td><td>0.323</td><td>0.344</td><td>0.062</td><td>0.062</td><td>0.050</td><td>0.059</td><td>0.627</td><td>0.617</td><td>0.697</td><td>0.774</td></tr><tr><td rowspan="2">联邦变换器 - w（FEDformer - w）</td><td>均方误差（MSE）</td><td>0.063</td><td>0.110</td><td>0.147</td><td>0.219</td><td>0.262</td><td>0.316</td><td>0.361</td><td>0.448</td><td>0.131</td><td>0.277</td><td>0.426</td><td>1.162</td><td>0.170</td><td>0.173</td><td>0.178</td><td>0.187</td><td>0.0035</td><td>0.0054</td><td>0.008</td><td>0.015</td><td>0.693</td><td>0.554</td><td>0.699</td><td>0.828</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.189</td><td>0.252</td><td>0.301</td><td>0.368</td><td>0.378</td><td>0.410</td><td>0.445</td><td>0.501</td><td>0.284</td><td>0.420</td><td>0.511</td><td>0.832</td><td>0.263</td><td>0.265</td><td>0.266</td><td>0.286</td><td>0.046</td><td>0.059</td><td>0.072</td><td>0.091</td><td>0.629</td><td>0.604</td><td>0.696</td><td>0.770</td></tr><tr><td rowspan="2">自动变换器（Autoformer）</td><td>均方误差（MSE）</td><td>0.065</td><td>0.118</td><td>0.154</td><td>0.182</td><td>0.341</td><td>0.345</td><td>0.406</td><td>0.565</td><td>0.241</td><td>0.300</td><td>0.509</td><td>1.260</td><td>0.246</td><td>0.266</td><td>0.263</td><td>0.269</td><td>0.011</td><td>0.0075</td><td>0.0063</td><td>0.0085</td><td>0.948</td><td>0.634</td><td>0.791</td><td>0.874</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.189</td><td>0.256</td><td>0.305</td><td>0.335</td><td>0.438</td><td>0.428</td><td>0.470</td><td>0.581</td><td>0.387</td><td>0.369</td><td>0.524</td><td>0.867</td><td>0.346</td><td>0.370</td><td>0.371</td><td>0.372</td><td>0.081</td><td>0.067</td><td>0.062</td><td>0.070</td><td>0.732</td><td>0.650</td><td>0.752</td><td>0.797</td></tr><tr><td rowspan="2">信息器（Informer）</td><td>均方误差（MSE）</td><td>0.080</td><td>0.112</td><td>0.166</td><td>0.228</td><td>0.258</td><td>0.285</td><td>0.336</td><td>0.607</td><td>1.327</td><td>1.258</td><td>2.179</td><td>1.280</td><td>0.257</td><td>0.299</td><td>0.312</td><td>0.366</td><td>0.004</td><td>0.002</td><td>0.004</td><td>0.003</td><td>5.282</td><td>4.554</td><td>4.273</td><td>5.214</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.217</td><td>0.259</td><td>0.314</td><td>0.380</td><td>0.367</td><td>0.388</td><td>0.423</td><td>0.599</td><td>0.944</td><td>0.924</td><td>1.296</td><td>0.953</td><td>0.353</td><td>0.376</td><td>0.387</td><td>0.436</td><td>0.044</td><td>0.040</td><td>0.049</td><td>0.042</td><td>2.050</td><td>1.916</td><td>1.846</td><td>2.057</td></tr><tr><td rowspan="2">对数变换器（LogTrans）</td><td>均方误差（MSE）</td><td>0.075</td><td>0.129</td><td>0.154</td><td>0.160</td><td>0.288</td><td>0.432</td><td>0.430</td><td>0.491</td><td>0.237</td><td>0.738</td><td>2.018</td><td>2.405</td><td>0.226</td><td>0.314</td><td>0.387</td><td>0.437</td><td>0.0046</td><td>0.0060</td><td>0.0060</td><td>0.007</td><td>3.607</td><td>2.407</td><td>3.106</td><td>3.698</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.208</td><td>0.275</td><td>0.302</td><td>0.322</td><td>0.393</td><td>0.483</td><td>0.483</td><td>0.531</td><td>0.377</td><td>0.619</td><td>1.070</td><td>1.175</td><td>0.317</td><td>0.408</td><td>0.453</td><td>0.491</td><td>0.052</td><td>0.060</td><td>0.054</td><td>0.059</td><td>1.662</td><td>1.363</td><td>1.575</td><td>1.733</td></tr><tr><td rowspan="2">改革者变换器（Reformer）</td><td>均方误差（MSE）</td><td>0.077</td><td>0.138</td><td>0.160</td><td>0.168</td><td>0.275</td><td>0.304</td><td>0.370</td><td>0.460</td><td>0.298</td><td>0.777</td><td>1.833</td><td>1.203</td><td>0.313</td><td>0.386</td><td>0.423</td><td>0.378</td><td>0.012</td><td>0.0098</td><td>0.013</td><td>0.011</td><td>3.838</td><td>2.934</td><td>3.755</td><td>4.162</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.214</td><td>0.290</td><td>0.313</td><td>0.334</td><td>0.379</td><td>0.402</td><td>0.448</td><td>0.511</td><td>0.444</td><td>0.719</td><td>1.128</td><td>0.956</td><td>0.383</td><td>0.453</td><td>0.468</td><td>0.433</td><td>0.087</td><td>0.044</td><td>0.100</td><td>0.083</td><td>1.720</td><td>1.520</td><td>1.749</td><td>1.847</td></tr></tbody></table>

<!-- Media -->

Univariate Results The results for univariate time series forecasting are summarized in Table 3. Compared with Autoformer, FEDformer yields an overall 22.6% relative MSE reduction, and on some datasets, such as traffic and weather,the improvement can be more than ${30}\%$ . It again verifies that FEDformer is more effective in long-term forecasting. Note that due to the difference between Fourier and wavelet basis, FEDformer-f and FEDformer-w perform well on different datasets, making them complementary choice for long term forecasting. More detailed results on ETT full benchmark are provided in Appendix F.3.

单变量结果 单变量时间序列预测的结果总结在表3中。与Autoformer相比，FEDformer的均方误差（MSE）总体相对降低了22.6%，在一些数据集上，如交通和天气数据集，改进幅度可能超过${30}\%$。这再次验证了FEDformer在长期预测中更有效。值得注意的是，由于傅里叶（Fourier）基和小波（wavelet）基的差异，FEDformer-f和FEDformer-w在不同数据集上表现良好，这使得它们成为长期预测的互补选择。附录F.3提供了ETT完整基准测试的更详细结果。

### 4.2. Ablation Studies

### 4.2. 消融实验

In this section, the ablation experiments are conducted, aiming at comparing the performance of frequency enhanced block and its alternatives. The current SOTA results of Aut-oformer which uses the autocorrelation mechanism serve as the baseline. Three ablation variants of FEDformer are tested: 1) FEDformer V1: we use FEB to substitute self-attention only; 2) FEDformer V2: we use FEA to substitute cross attention only; 3) FEDFormer V3: we use FEA to substitute both self and cross attention. The ablated versions of FEDformer-f as well as the SOTA models are compared in Table 4, and we use a bold number if the ablated version brings improvements compared with Auto-former. We omit the similar results in FEDformer-w due to space limit. It can be seen in Table 4 that FEDformer V1 brings improvement in 10/16 cases, while FEDformer V2 improves in 12/16 cases. The best performance is achieved in our FEDformer with FEB and FEA blocks which improves performance in all ${16}/{16}$ cases. This verifies the effectiveness of the designed FEB, FEA for substituting self and cross attention. Furthermore, experiments on ETT and Weather datasets show that the adopted MOEDecomp (mixture of experts decomposition) scheme can bring an average of ${2.96}\%$ improvement compared with the single decomposition scheme. More details are provided in Appendix F.5. bold.

在本节中，进行了消融实验，旨在比较频率增强模块（Frequency Enhanced Block）及其替代方案的性能。使用自相关机制的当前最先进（SOTA）的Aut-oformer模型的结果作为基线。对FEDformer的三个消融变体进行了测试：1）FEDformer V1：仅使用频率增强块（FEB）替代自注意力机制；2）FEDformer V2：仅使用频率增强注意力（FEA）替代交叉注意力机制；3）FEDFormer V3：使用频率增强注意力（FEA）同时替代自注意力和交叉注意力机制。表4中比较了FEDformer-f的消融版本以及最先进的模型，如果消融版本与Auto-former相比有改进，我们使用粗体数字表示。由于篇幅限制，我们省略了FEDformer-w中类似的结果。从表4中可以看出，FEDformer V1在16种情况中的10种情况下带来了改进，而FEDformer V2在16种情况中的12种情况下有所改进。我们采用频率增强块（FEB）和频率增强注意力（FEA）模块的FEDformer在所有${16}/{16}$种情况下都实现了性能提升，达到了最佳性能。这验证了所设计的用于替代自注意力和交叉注意力的频率增强块（FEB）和频率增强注意力（FEA）的有效性。此外，在ETT和天气数据集上的实验表明，与单一分解方案相比，所采用的专家混合分解（MOEDecomp，Mixture of Experts Decomposition）方案平均可带来${2.96}\%$的性能提升。更多细节见附录F.5。粗体。

<!-- Media -->

Table 4. Ablation studies: multivariate long-term series forecasting results on ETTm1 and ETTm2 with input length $I = {96}$ and prediction length $O \in  \{ {96},{192},{336},{720}\}$ . Three variants of FEDformer-f are compared with baselines. The best results are highlighted in

表4. 消融实验：在ETTm1和ETTm2数据集上，输入长度为$I = {96}$、预测长度为$O \in  \{ {96},{192},{336},{720}\}$的多变量长期序列预测结果。将FEDformer - f的三种变体与基线模型进行了比较。最佳结果以

<table><tr><td colspan="2">Methods</td><td colspan="2">Transformer</td><td colspan="2">Informer</td><td colspan="2">Autoformer</td><td colspan="2">FEDformer V1</td><td colspan="2">FEDformer V2</td><td colspan="2">FEDformer V3</td><td colspan="2">FEDformer-f</td></tr><tr><td colspan="2">Self-att Cross-att</td><td colspan="2">FullAtt FullAtt</td><td colspan="2">ProbAtt ProbAtt</td><td colspan="2">AutoCorr AutoCorr</td><td colspan="2">FEB-f(Eq. 4) AutoCorr</td><td colspan="2">AutoCorr FEA-f(Eq. 7)</td><td colspan="2">FEA-f(Eq. 7) FEA-f(Eq. 7)</td><td colspan="2">FEB-f(Eq. 4) FEA-f(Eq. 7)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.525</td><td>0.486</td><td>0.458</td><td>0.465</td><td>0.481</td><td>0.463</td><td>0.378</td><td>0.419</td><td>0.539</td><td>0.490</td><td>0.534</td><td>0.482</td><td>0.379</td><td>0.419</td></tr><tr><td>192</td><td>0.526</td><td>0.502</td><td>0.564</td><td>0.521</td><td>0.628</td><td>0.526</td><td>0.417</td><td>0.442</td><td>0.556</td><td>0.499</td><td>0.552</td><td>0.493</td><td>0.426</td><td>0.441</td></tr><tr><td>336</td><td>0.514</td><td>0.502</td><td>0.672</td><td>0.559</td><td>0.728</td><td>0.567</td><td>0.480</td><td>0.477</td><td>0.541</td><td>0.498</td><td>0.565</td><td>0.503</td><td>0.445</td><td>0.459</td></tr><tr><td>720</td><td>0.564</td><td>0.529</td><td>0.714</td><td>0.596</td><td>0.658</td><td>0.548</td><td>0.543</td><td>0.517</td><td>0.558</td><td>0.507</td><td>0.585</td><td>0.515</td><td>0.543</td><td>0.490</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.268</td><td>0.346</td><td>0.227</td><td>0.305</td><td>0.255</td><td>0.339</td><td>0.259</td><td>0.337</td><td>0.216</td><td>0.297</td><td>0.211</td><td>0.292</td><td>0.203</td><td>0.287</td></tr><tr><td>192</td><td>0.304</td><td>0.355</td><td>0.300</td><td>0.360</td><td>0.281</td><td>0.340</td><td>0.285</td><td>0.344</td><td>0.274</td><td>0.331</td><td>0.272</td><td>0.329</td><td>0.269</td><td>0.328</td></tr><tr><td>336</td><td>0.365</td><td>0.400</td><td>0.382</td><td>0.410</td><td>0.339</td><td>0.372</td><td>0.320</td><td>0.373</td><td>0.334</td><td>0.369</td><td>0.327</td><td>0.363</td><td>0.325</td><td>0.366</td></tr><tr><td>720</td><td>0.475</td><td>0.466</td><td>1.637</td><td>0.794</td><td>0.422</td><td>0.419</td><td>0.761</td><td>0.628</td><td>0.427</td><td>0.420</td><td>0.418</td><td>0.415</td><td>0.421</td><td>0.415</td></tr><tr><td colspan="2">Count</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>5</td><td>5</td><td>6</td><td>6</td><td>7</td><td>7</td><td>8</td><td>8</td></tr></table>

<table><tbody><tr><td colspan="2">方法</td><td colspan="2">变换器（Transformer）</td><td colspan="2">信息者（Informer）</td><td colspan="2">自动变换器（Autoformer）</td><td colspan="2">联邦变换器V1（FEDformer V1）</td><td colspan="2">联邦变换器V2（FEDformer V2）</td><td colspan="2">FEDformer V3（FEDformer第三代）</td><td colspan="2">FEDformer-f</td></tr><tr><td colspan="2">自注意力 交叉注意力</td><td colspan="2">全注意力 全注意力</td><td colspan="2">概率注意力 概率注意力</td><td colspan="2">自相关 自相关</td><td colspan="2">FEB-f（公式4）自相关</td><td colspan="2">自相关 FEA-f（公式7）</td><td colspan="2">FEA-f（公式7） FEA-f（公式7）</td><td colspan="2">FEB-f（公式4） FEA-f（公式7）</td></tr><tr><td colspan="2">度量指标</td><td>均方误差</td><td>平均绝对误差（MAE）</td><td>均方误差</td><td>平均绝对误差（MAE）</td><td>均方误差</td><td>平均绝对误差（MAE）</td><td>均方误差</td><td>平均绝对误差（MAE）</td><td>均方误差</td><td>平均绝对误差（MAE）</td><td>均方误差</td><td>平均绝对误差（MAE）</td><td>均方误差</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.525</td><td>0.486</td><td>0.458</td><td>0.465</td><td>0.481</td><td>0.463</td><td>0.378</td><td>0.419</td><td>0.539</td><td>0.490</td><td>0.534</td><td>0.482</td><td>0.379</td><td>0.419</td></tr><tr><td>192</td><td>0.526</td><td>0.502</td><td>0.564</td><td>0.521</td><td>0.628</td><td>0.526</td><td>0.417</td><td>0.442</td><td>0.556</td><td>0.499</td><td>0.552</td><td>0.493</td><td>0.426</td><td>0.441</td></tr><tr><td>336</td><td>0.514</td><td>0.502</td><td>0.672</td><td>0.559</td><td>0.728</td><td>0.567</td><td>0.480</td><td>0.477</td><td>0.541</td><td>0.498</td><td>0.565</td><td>0.503</td><td>0.445</td><td>0.459</td></tr><tr><td>720</td><td>0.564</td><td>0.529</td><td>0.714</td><td>0.596</td><td>0.658</td><td>0.548</td><td>0.543</td><td>0.517</td><td>0.558</td><td>0.507</td><td>0.585</td><td>0.515</td><td>0.543</td><td>0.490</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.268</td><td>0.346</td><td>0.227</td><td>0.305</td><td>0.255</td><td>0.339</td><td>0.259</td><td>0.337</td><td>0.216</td><td>0.297</td><td>0.211</td><td>0.292</td><td>0.203</td><td>0.287</td></tr><tr><td>192</td><td>0.304</td><td>0.355</td><td>0.300</td><td>0.360</td><td>0.281</td><td>0.340</td><td>0.285</td><td>0.344</td><td>0.274</td><td>0.331</td><td>0.272</td><td>0.329</td><td>0.269</td><td>0.328</td></tr><tr><td>336</td><td>0.365</td><td>0.400</td><td>0.382</td><td>0.410</td><td>0.339</td><td>0.372</td><td>0.320</td><td>0.373</td><td>0.334</td><td>0.369</td><td>0.327</td><td>0.363</td><td>0.325</td><td>0.366</td></tr><tr><td>720</td><td>0.475</td><td>0.466</td><td>1.637</td><td>0.794</td><td>0.422</td><td>0.419</td><td>0.761</td><td>0.628</td><td>0.427</td><td>0.420</td><td>0.418</td><td>0.415</td><td>0.421</td><td>0.415</td></tr><tr><td colspan="2">数量</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>5</td><td>5</td><td>6</td><td>6</td><td>7</td><td>7</td><td>8</td><td>8</td></tr></tbody></table>

<!-- figureText: 0.60 0.435 III 0.430 0.425 0.420 Mode number 0.50 Mode number -->

<img src="https://cdn.noedgeai.com/01957f69-db4d-7f34-af19-bfc9e93e263b_7.jpg?x=164&y=758&w=681&h=252&r=0"/>

Figure 6. Comparison of two base-modes selection method (Fix&Rand). Rand policy means randomly selecting a subset of modes, Fix policy means selecting the lowest frequency modes. Two policies are compared on a variety of base-modes number $M \in  \{ 2,4,8\ldots {256}\}$ on ETT full-benchmark (h1, m1, h2, m2).

图6. 两种基模式选择方法（固定选择和随机选择）的比较。随机（Rand）策略是指随机选择一部分模式，固定（Fix）策略是指选择频率最低的模式。在ETT全基准测试（h1、m1、h2、m2）中，针对多种基模式数量$M \in  \{ 2,4,8\ldots {256}\}$对这两种策略进行了比较。

<!-- Media -->

### 4.3. Mode Selection Policy

### 4.3. 模式选择策略

The selection of discrete Fourier basis is the key to effectively representing the signal and maintaining the model's linear complexity. As we discussed in Section 2, random Fourier mode selection is a better policy in forecasting tasks. more importantly, random policy requires no prior knowledge of the input and generalizes easily in new tasks. Here we empirically compare the random selection policy with fixed selection policy, and summarize the experimental results in Figure 6. It can be observed that the adopted random policy achieves better performance than the common fixed policy which only keeps the low frequency modes. Meanwhile, the random policy exhibits some mode saturation effect, indicating an appropriate random number of modes instead of all modes would bring better performance, which is also consistent with the theoretical analysis in Section 2.

离散傅里叶基的选择是有效表示信号并保持模型线性复杂度的关键。正如我们在第2节中所讨论的，随机傅里叶模式选择在预测任务中是一种更好的策略。更重要的是，随机策略不需要输入的先验知识，并且在新任务中易于泛化。在这里，我们通过实验将随机选择策略与固定选择策略进行了比较，并将实验结果总结在图6中。可以观察到，所采用的随机策略比仅保留低频模式的常见固定策略取得了更好的性能。同时，随机策略表现出一定的模式饱和效应，这表明选择适当数量的随机模式而非所有模式会带来更好的性能，这也与第2节中的理论分析一致。

### 4.4. Distribution Analysis of Forecasting Output

### 4.4. 预测输出的分布分析

In this section, we evaluate the distribution similarity between the input sequence and forecasting output of different transformer models quantitatively. In Table 5, we applied the Kolmogrov-Smirnov test to check if the forecasting results of different models made on ETTm1 and ETTm2 are consistent with the input sequences. In particular, we test if the input sequence of fixed 96-time steps come from the same distribution as the predicted sequence, with the null hypothesis that both sequences come from the same distribution. On both datasets, by setting the common P-value as 0.01, various existing Transformer baseline models have much less values than 0.01 except Aut-oformer, which indicates their forecasting output have a higher probability to be sampled from the different distributions compared to the input sequence. In contrast, Auto-former and FEDformer have much larger P-value compared to others, which mainly contributes to their seasonal-trend decomposition mechanism. Though we get close results from ETTm2 by both models, the proposed FEDformer has much larger P-value in ETTm1. And it's the only model whose null hypothesis can not be rejected with P-value larger than 0.01 in all cases of the two datasets, implying that the output sequence generated by FEDformer shares a more similar distribution as the input sequence than others and thus justifies the our design motivation of FEDformer as discussed in Section 1. More detailed analysis are provided in Appendix E.

在本节中，我们定量评估不同Transformer模型的输入序列与预测输出之间的分布相似度。在表5中，我们应用柯尔莫哥洛夫 - 斯米尔诺夫检验（Kolmogrov - Smirnov test）来检查不同模型在ETTm1和ETTm2上的预测结果是否与输入序列一致。具体而言，我们检验固定96个时间步长的输入序列是否与预测序列来自同一分布，原假设为两个序列来自同一分布。在两个数据集上，将共同的P值设为0.01时，除Auto - former外，各种现有的Transformer基线模型的值远小于0.01，这表明与输入序列相比，它们的预测输出更有可能是从不同分布中采样得到的。相比之下，Auto - former和FEDformer的P值比其他模型大得多，这主要归功于它们的季节性 - 趋势分解机制。虽然这两个模型在ETTm2上得到的结果相近，但所提出的FEDformer在ETTm1上的P值要大得多。并且它是唯一在两个数据集的所有情况下P值都大于0.01而不能拒绝原假设的模型，这意味着FEDformer生成的输出序列与输入序列的分布比其他模型更相似，从而证明了我们在第1节中讨论的FEDformer的设计动机是合理的。附录E中提供了更详细的分析。

<!-- Media -->

Table 5. P-values of Kolmogrov-Smirnov test of different transformer models for long-term forecasting output on ETTm1 and ETTm2 dataset. Larger value indicates the hypothesis (the input sequence and forecasting output come from the same distribution) is less likely to be rejected. The best results are highlighted.

表5. 不同Transformer模型在ETTm1和ETTm2数据集上进行长期预测输出的科尔莫戈罗夫 - 斯米尔诺夫检验（Kolmogrov - Smirnov test）的P值。数值越大表明原假设（输入序列和预测输出来自同一分布）越不可能被拒绝。最佳结果已高亮显示。

<table><tr><td colspan="2">Methods</td><td>Transformer</td><td>Informer</td><td>Autoformer</td><td>FEDformer</td><td>True</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.0090</td><td>0.0055</td><td>0.020</td><td>0.048</td><td>0.023</td></tr><tr><td>192</td><td>0.0052</td><td>0.0029</td><td>0.015</td><td>0.028</td><td>0.013</td></tr><tr><td>336</td><td>0.0022</td><td>0.0019</td><td>0.012</td><td>0.015</td><td>0.010</td></tr><tr><td>720</td><td>0.0023</td><td>0.0016</td><td>0.008</td><td>0.014</td><td>0.004</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.0012</td><td>0.0008</td><td>0.079</td><td>0.071</td><td>0.087</td></tr><tr><td>192</td><td>0.0011</td><td>0.0006</td><td>0.047</td><td>0.045</td><td>0.060</td></tr><tr><td>336</td><td>0.0005</td><td>0.00009</td><td>0.027</td><td>0.028</td><td>0.042</td></tr><tr><td>720</td><td>0.0008</td><td>0.0002</td><td>0.023</td><td>0.021</td><td>0.023</td></tr><tr><td colspan="2">Count</td><td>0</td><td>0</td><td>3</td><td>5</td><td>NA</td></tr></table>

<table><tbody><tr><td colspan="2">方法</td><td>变换器（Transformer）</td><td>先知模型（Informer）</td><td>自动变换器（Autoformer）</td><td>频域增强变换器（FEDformer）</td><td>真</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.0090</td><td>0.0055</td><td>0.020</td><td>0.048</td><td>0.023</td></tr><tr><td>192</td><td>0.0052</td><td>0.0029</td><td>0.015</td><td>0.028</td><td>0.013</td></tr><tr><td>336</td><td>0.0022</td><td>0.0019</td><td>0.012</td><td>0.015</td><td>0.010</td></tr><tr><td>720</td><td>0.0023</td><td>0.0016</td><td>0.008</td><td>0.014</td><td>0.004</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.0012</td><td>0.0008</td><td>0.079</td><td>0.071</td><td>0.087</td></tr><tr><td>192</td><td>0.0011</td><td>0.0006</td><td>0.047</td><td>0.045</td><td>0.060</td></tr><tr><td>336</td><td>0.0005</td><td>0.00009</td><td>0.027</td><td>0.028</td><td>0.042</td></tr><tr><td>720</td><td>0.0008</td><td>0.0002</td><td>0.023</td><td>0.021</td><td>0.023</td></tr><tr><td colspan="2">数量</td><td>0</td><td>0</td><td>3</td><td>5</td><td>不适用</td></tr></tbody></table>

<!-- Media -->

### 4.5. Differences Compared to Autoformer baseline

### 4.5. 与Autoformer基线的差异

Since we use the decomposed encoder-decoder overall architecture as Autoformer, we think it is critical to emphasize the differences. In Autoformer, the authors consider a nice idea to use the top-k sub-sequence correlation (autocorrelation) module instead of point-wise attention, and the Fourier method is applied to improve the efficiency for subsequence level similarity computation. In general, Auto-former can be considered as decomposing the sequence into multiple time domain sub-sequences for feature exaction. In contrast, We use frequency transform to decompose the sequence into multiple frequency domain modes to extract the feature. In particular, we do not use a selective approach in sub-sequence selection. Instead, all frequency features are computed from the whole sequence, and this global property makes our model engage better performance for long sequence.

由于我们和Autoformer一样采用了分解式编解码器的整体架构，因此我们认为强调两者的差异至关重要。在Autoformer中，作者提出了一个很好的想法，即使用前k个子序列相关性（自相关性）模块来替代逐点注意力机制，并应用傅里叶方法提高子序列级相似度计算的效率。总体而言，Autoformer可以被视为将序列分解为多个时域子序列以进行特征提取。相比之下，我们使用频率变换将序列分解为多个频域模式来提取特征。具体来说，我们在子序列选择中不采用选择性方法。相反，所有频率特征都是从整个序列中计算得出的，这种全局性使得我们的模型在处理长序列时表现更优。

## 5. Conclusions

## 5. 结论

This paper proposes a frequency enhanced transformer model for long-term series forecasting which achieves state-of-the-art performance and enjoys linear computational complexity and memory cost. We propose an attention mechanism with low-rank approximation in frequency and a mixture of experts decomposition to control the distribution shifting. The proposed frequency enhanced structure decouples the input sequence length and the attention matrix dimension, leading to the linear complexity. Moreover, we theoretically and empirically prove the effectiveness of the adopted random mode selection policy in frequency. Lastly, extensive experiments show that the proposed model achieves the best forecasting performance on six benchmark datasets in comparison with four state-of-the-art algorithms.

本文提出了一种用于长期序列预测的频率增强变压器（Transformer）模型，该模型达到了当前最优性能，并且具有线性计算复杂度和内存成本。我们提出了一种在频率上进行低秩近似的注意力机制和一种专家混合分解方法来控制分布偏移。所提出的频率增强结构将输入序列长度和注意力矩阵维度解耦，从而实现了线性复杂度。此外，我们从理论和实证两方面证明了所采用的频率随机模式选择策略的有效性。最后，大量实验表明，与四种当前最优算法相比，所提出的模型在六个基准数据集上取得了最佳的预测性能。

## References

## 参考文献

Arriaga, R. I. and Vempala, S. S. An algorithmic theory of learning: Robust concepts and random projection. Mach. Learn., 63(2):161-182, 2006.

阿里亚加（Arriaga），R. I. 和温帕拉（Vempala），S. S. 学习的算法理论：鲁棒概念与随机投影。《机器学习》（Mach. Learn.），63(2):161 - 182，2006年。

Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. CoRR, abs/2004.05150, 2020.

贝尔塔吉（Beltagy），I.、彼得斯（Peters），M. E. 和科汉（Cohan），A. 《长文档变压器：Longformer》。《计算机研究报告》（CoRR），abs/2004.05150，2020年。

Box, G. E. P. and Jenkins, G. M. Some recent advances in forecasting and control. Journal of the Royal Statistical Society. Series C (Applied Statistics), 17(2):91-109, 1968.

博克斯（Box, G. E. P.）和詹金斯（Jenkins, G. M.）。预测与控制的一些近期进展。《皇家统计学会杂志》。C辑（应用统计学），17(2):91 - 109，1968年。

Box, G. E. P. and Pierce, D. A. Distribution of residual autocorrelations in autoregressive-integrated moving average time series models. volume 65, pp. 1509-1526. Taylor & Francis, 1970.

博克斯（Box, G. E. P.）和皮尔斯（Pierce, D. A.）。自回归积分滑动平均时间序列模型中残差自相关的分布。第65卷，第1509 - 1526页。泰勒与弗朗西斯出版社（Taylor & Francis），1970年。

Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlós, T., Hawkins, P., Davis, J. Q., Mohi-uddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In 9th International Conference on Learning Representations (ICLR), Virtual Event, Austria, May 3-7, 2021, 2021.

乔罗曼斯基（Choromanski, K. M.）、利霍舍斯托夫（Likhosherstov, V.）、多汉（Dohan, D.）、宋（Song, X.）、加内（Gane, A.）、萨尔洛斯（Sarlós, T.）、霍金斯（Hawkins, P.）、戴维斯（Davis, J. Q.）、莫希 - 乌丁（Mohi - uddin, A.）、凯泽（Kaiser, L.）、贝朗热（Belanger, D. B.）、科尔韦尔（Colwell, L. J.）和韦勒（Weller, A.）。用执行者重新思考注意力机制。见第9届国际学习表征会议（ICLR），线上会议，奥地利，2021年5月3 - 7日，2021年。

Chung, J., Gülçehre, Ç., Cho, K., and Bengio, Y. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

钟（Chung）、居勒切雷（Gülçehre）、赵（Cho）和本吉奥（Bengio）。门控循环神经网络在序列建模上的实证评估。计算机研究报告（CoRR），编号abs/1412.3555，2014年。

Cleveland, R. B., Cleveland, W. S., McRae, J. E., and Ter-penning, I. Stl: A seasonal-trend decomposition. Journal of official statistics, 6(1):3-73, 1990.

克利夫兰（Cleveland）、克利夫兰（Cleveland）、麦克雷（McRae）和特尔彭宁（Terpenning）。STL：一种季节性 - 趋势分解方法。《官方统计杂志》，6(1):3 - 73，1990年。

Devlin, J., Chang, M., Lee, K., and Toutanova, K. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Minneapolis, MN, USA, June 2- 7, 2019, pp. 4171-4186, 2019.

德夫林（Devlin）、张（Chang）、李（Lee）和图托纳娃（Toutanova）。BERT：用于语言理解的深度双向变换器预训练。收录于《2019年北美计算语言学协会人类语言技术会议（NAACL - HLT）论文集》，美国明尼苏达州明尼阿波利斯市，2019年6月2 - 7日，第4171 - 4186页，2019年。

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

多索维茨基（Dosovitskiy, A.）、拜尔（Beyer, L.）、科列斯尼科夫（Kolesnikov, A.）、魏森伯恩（Weissenborn, D.）、翟（Zhai, X.）、翁特尔蒂纳（Unterthiner, T.）、德赫加尼（Dehghani, M.）、明德勒（Minderer, M.）、海戈尔德（Heigold, G.）、盖利（Gelly, S.）、乌斯佐赖特（Uszkoreit, J.）和霍尔斯比（Houlsby, N.）。一张图像价值16×16个单词：大规模图像识别的Transformer。见第9届国际学习表征会议（9th International Conference on Learning Representations, ICLR 2021），虚拟会议，奥地利，2021年5月3 - 7日。OpenReview.net，2021年。

Drineas, P., Mahoney, M. W., and Muthukrishnan, S. Relative-error CUR matrix decompositions. CoRR, abs/0708.3696, 2007.

德里尼亚斯（Drineas, P.）、马奥尼（Mahoney, M. W.）和穆图克里什南（Muthukrishnan, S.）。相对误差CUR矩阵分解。预印本库（CoRR），编号abs/0708.3696，2007年。

Flunkert, V., Salinas, D., and Gasthaus, J. Deepar: Probabilistic forecasting with autoregressive recurrent networks. CoRR, abs/1704.04110, 2017.

弗伦克特（Flunkert, V.）、萨利纳斯（Salinas, D.）和加施豪斯（Gasthaus, J.）。Deepar：基于自回归循环网络的概率预测。预印本库（CoRR），编号abs/1704.04110，2017年。

Gupta, G., Xiao, X., and Bogdan, P. Multiwavelet-based operator learning for differential equations, 2021.

古普塔（Gupta, G.）、肖（Xiao, X.）和博格丹（Bogdan, P.）。基于多小波的微分方程算子学习，2021年。

Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory. Neural Computation, 9(8):1735-1780, November 1997. ISSN 0899-7667, 1530-888X.

霍赫赖特（Hochreiter, S.）和施密德胡伯（Schmidhuber, J.）。长短期记忆网络。《神经计算》，1997年11月，9(8):1735 - 1780。国际标准连续出版物编号：0899 - 7667，1530 - 888X。

Homayouni, H., Ghosh, S., Ray, I., Gondalia, S., Duggan, J., and Kahn, M. G. An autocorrelation-based lstm-autoencoder for anomaly detection on time-series data. In 2020 IEEE International Conference on Big Data (Big Data), pp. 5068-5077, 2020.

霍马尤尼（Homayouni, H.）、戈什（Ghosh, S.）、雷（Ray, I.）、贡达利亚（Gondalia, S.）、达根（Duggan, J.）和卡恩（Kahn, M. G.）。一种基于自相关的长短期记忆自动编码器用于时间序列数据的异常检测。见《2020年电气与电子工程师协会国际大数据会议（Big Data）》，第5068 - 5077页，2020年。

Johnson, W. B. Extensions of lipschitz mappings into hilbert space. Contemporary mathematics, 26:189-206, 1984.

约翰逊（Johnson, W. B.）。利普希茨映射到希尔伯特空间的扩展。《当代数学》，1984年，26:189 - 206。

Kingma, D. P. and Ba, J. Adam: A Method for Stochastic Optimization. arXiv:1412.6980 [cs], January 2017. arXiv: 1412.6980.

金马（Kingma, D. P.）和巴（Ba, J.）。Adam：一种随机优化方法。预印本arXiv:1412.6980 [计算机科学]，2017年1月。预印本编号：arXiv: 1412.6980。

Kitaev, N., Kaiser, L., and Levskaya, A. Reformer: The efficient transformer. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020, 2020.

基塔耶夫（Kitaev, N.）、凯泽（Kaiser, L.）和列夫斯卡娅（Levskaya, A.）。《改革者：高效的Transformer》。发表于第8届国际学习表征会议（8th International Conference on Learning Representations, ICLR 2020），2020年4月26 - 30日，埃塞俄比亚亚的斯亚贝巴，2020年。

Lai, G., Chang, W.-C., Yang, Y., and Liu, H. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pp. 95-104, 2018.

赖（Lai, G.）、张（Chang, W.-C.）、杨（Yang, Y.）和刘（Liu, H.）。《使用深度神经网络对长短期时间模式进行建模》。发表于第41届国际计算机协会信息检索研究与发展会议（The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval），第95 - 104页，2018年。

Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y.- X., and Yan, X. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. In Advances in Neural Information Processing Systems, volume 32, 2019.

李（Li, S.）、金（Jin, X.）、宣（Xuan, Y.）、周（Zhou, X.）、陈（Chen, W.）、王（Wang, Y.- X.）和严（Yan, X.）。《增强Transformer在时间序列预测中的局部性并打破内存瓶颈》。发表于《神经信息处理系统进展》（Advances in Neural Information Processing Systems），第32卷，2019年。

Li, Z., Kovachki, N. B., Azizzadenesheli, K., Liu, B., Bhat-tacharya, K., Stuart, A. M., and Anandkumar, A. Fourier neural operator for parametric partial differential equations. CoRR, abs/2010.08895, 2020.

李（Li）、科瓦奇基（Kovachki）、阿齐扎德内舍利（Azizzadenesheli）、刘（Liu）、巴塔查里亚（Bhat - tacharya）、斯图尔特（Stuart）和阿南德库马尔（Anandkumar）。用于参数偏微分方程的傅里叶神经算子。预印本库（CoRR），编号abs/2010.08895，2020年。

Ma, X., Kong, X., Wang, S., Zhou, C., May, J., Ma, H., and Zettlemoyer, L. Luna: Linear unified nested attention. CoRR, abs/2106.01540, 2021.

马（Ma）、孔（Kong）、王（Wang）、周（Zhou）、梅（May）、马（Ma）和泽特勒莫耶（Zettlemoyer）。卢娜（Luna）：线性统一嵌套注意力。预印本库（CoRR），编号abs/2106.01540，2021年。

Mathieu, M., Henaff, M., and LeCun, Y. Fast training of convolutional networks through ffts. In 2nd International Conference on Learning Representations (ICLR), Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014.

马蒂厄（Mathieu）、埃纳夫（Henaff）和勒昆（LeCun）。通过快速傅里叶变换（FFTs）快速训练卷积网络。收录于2014年4月14 - 16日在加拿大阿尔伯塔省班夫市举行的第二届学习表征国际会议（ICLR）会议论文集，2014年。

Oreshkin, B. N., Carpov, D., Chapados, N., and Bengio, Y. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In Proceedings of the International Conference on Learning Representations (ICLR), 2019.

奥列什金（Oreshkin）、卡尔波夫（Carpov）、查帕多斯（Chapados）和本吉奥（Bengio）。N - BEATS：用于可解释时间序列预测的神经基扩展分析。收录于学习表征国际会议（ICLR）论文集，2019年。

Pascanu, R., Mikolov, T., and Bengio, Y. On the difficulty of training recurrent neural networks. In Proceedings of the 30th International Conference on Machine Learning, ICML 2013, Atlanta, GA, USA, 16-21 June 2013, volume 28, pp. 1310-1318, 2013.

帕斯卡努（Pascanu, R.）、米科洛夫（Mikolov, T.）和本吉奥（Bengio, Y.）。关于训练循环神经网络的难度。见《第30届国际机器学习会议论文集》（Proceedings of the 30th International Conference on Machine Learning），ICML 2013，美国佐治亚州亚特兰大（Atlanta, GA, USA），2013年6月16 - 21日，第28卷，第1310 - 1318页，2013年。

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, pp. 8024-8035. 2019.

帕兹克（Paszke, A.）、格罗斯（Gross, S.）、马萨（Massa, F.）、勒雷尔（Lerer, A.）、布拉德伯里（Bradbury, J.）、查南（Chanan, G.）、基林（Killeen, T.）、林（Lin, Z.）、吉梅尔申（Gimelshein, N.）、安蒂加（Antiga, L.）、德斯迈松（Desmaison, A.）、科普夫（Kopf, A.）、杨（Yang, E.）、德维托（DeVito, Z.）、赖森（Raison, M.）、特贾尼（Tejani, A.）、奇拉姆库尔蒂（Chilamkurthy, S.）、施泰纳（Steiner, B.）、方（Fang, L.）、白（Bai, J.）和钦塔拉（Chintala, S.）。PyTorch：一种命令式风格的高性能深度学习库。见《神经信息处理系统进展》（Advances in Neural Information Processing Systems），第8024 - 8035页，2019年。

Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., and Cottrell, G. W. A dual-stage attention-based recurrent neural network for time series prediction. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI), Melbourne, Australia, August 19-25, 2017, pp. 2627-2633. ijcai.org, 2017.

秦（Qin）、宋（Song）、陈（Chen）、程（Cheng）、蒋（Jiang）和科特雷尔（Cottrell, G. W.）。基于双阶段注意力的循环神经网络用于时间序列预测。见《第26届国际人工智能联合会议（IJCAI）论文集》，澳大利亚墨尔本，2017年8月19 - 25日，第2627 - 2633页。ijcai.org，2017年。

Qiu, J., Ma, H., Levy, O., Yih, W., Wang, S., and Tang, J. Blockwise self-attention for long document understanding. In Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, volume EMNLP 2020 of Findings of ACL, pp. 2555-2565. Association for Computational Linguistics, 2020.

邱（Qiu）、马（Ma）、利维（Levy, O.）、易（Yih, W.）、王（Wang）和唐（Tang）。用于长文档理解的分块自注意力机制。见《计算语言学协会成果：EMNLP 2020》，线上活动，2020年11月16 - 20日，《计算语言学协会成果》EMNLP 2020卷，第2555 - 2565页。计算语言学协会，2020年。

Rahimi, A. and Recht, B. Random features for large-scale kernel machines. In Platt, J., Koller, D., Singer, Y., and Roweis, S. (eds.), Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc., 2008.

拉希米（Rahimi, A.）和雷克特（Recht, B.）。大规模核机器的随机特征。见普拉特（Platt, J.）、科勒（Koller, D.）、辛格（Singer, Y.）和罗威斯（Roweis, S.）（编），《神经信息处理系统进展》，第20卷。柯伦联合公司，2008年。

Rangapuram, S. S., Seeger, M. W., Gasthaus, J., Stella, L., Wang, Y., and Januschowski, T. Deep state space models for time series forecasting. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

兰加普拉姆（Rangapuram, S. S.）、西格（Seeger, M. W.）、加斯豪斯（Gasthaus, J.）、斯特拉（Stella, L.）、王（Wang, Y.）和亚努绍夫斯基（Januschowski, T.）。用于时间序列预测的深度状态空间模型。《神经信息处理系统进展》，第31卷。柯伦联合公司（Curran Associates, Inc.），2018年。

Rao, Y., Zhao, W., Zhu, Z., Lu, J., and Zhou, J. Global filter networks for image classification. CoRR, abs/2107.00645, 2021.

饶（Rao, Y.）、赵（Zhao, W.）、朱（Zhu, Z.）、陆（Lu, J.）和周（Zhou, J.）。用于图像分类的全局滤波器网络。预印本库（CoRR），编号abs/2107.00645，2021年。

Rawat, A. S., Chen, J., Yu, F. X., Suresh, A. T., and Kumar, S. Sampled softmax with random fourier features. In Advances in Neural Information Processing Systems (NeurIPS), December 8-14, 2019, Vancouver, BC, Canada, pp. 13834-13844, 2019.

拉瓦特（Rawat, A. S.）、陈（Chen, J.）、余（Yu, F. X.）、苏雷什（Suresh, A. T.）和库马尔（Kumar, S.）。基于随机傅里叶特征的采样softmax。《神经信息处理系统进展》（NeurIPS），2019年12月8 - 14日，加拿大不列颠哥伦比亚省温哥华，第13834 - 13844页，2019年。

Sen, R., Yu, H., and Dhillon, I. S. Think globally, act locally: A deep neural network approach to high-dimensional time series forecasting. In Advances in Neural Information Processing Systems (NeurIPS), December 8-14, 2019, Vancouver, BC, Canada, pp. 4838-4847, 2019.

森（Sen, R.）、于（Yu, H.）和狄龙（Dhillon, I. S.）。放眼全球，立足本地：一种用于高维时间序列预测的深度神经网络方法。收录于《神经信息处理系统进展》（Advances in Neural Information Processing Systems，NeurIPS），2019年12月8 - 14日，加拿大不列颠哥伦比亚省温哥华，第4838 - 4847页，2019年。

Tay, Y., Bahri, D., Yang, L., Metzler, D., and Juan, D. Sparse sinkhorn attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pp. 9438-9447. PMLR, 2020.

泰（Tay, Y.）、巴希里（Bahri, D.）、杨（Yang, L.）、梅茨勒（Metzler, D.）和胡安（Juan, D.）。稀疏辛克霍恩注意力机制。收录于《第37届国际机器学习会议论文集》（Proceedings of the 37th International Conference on Machine Learning，ICML 2020），2020年7月13 - 18日，线上会议，《机器学习研究会议录》（Proceedings of Machine Learning Research）第119卷，第9438 - 9447页。机器学习研究会议录出版（PMLR），2020年。

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. CoRR, abs/1706.03762, 2017.

瓦斯瓦尼（Vaswani, A.）、沙泽尔（Shazeer, N.）、帕尔马（Parmar, N.）、乌兹科雷特（Uszkoreit, J.）、琼斯（Jones, L.）、戈麦斯（Gomez, A. N.）、凯泽（Kaiser, L.）和波洛苏金（Polosukhin, I.）。注意力就是你所需要的一切。计算机研究报告库（CoRR），编号abs/1706.03762，2017年。

Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. CoRR, abs/2006.04768, 2020.

王（Wang）、李（Li）、哈布萨（Khabsa）、方（Fang）和马（Ma）。《线性变换器：具有线性复杂度的自注意力机制》。计算机研究报告（CoRR），编号abs/2006.04768，2020年。

Wen, Q., Gao, J., Song, X., Sun, L., Xu, H., and Zhu, S. RobustSTL: A robust seasonal-trend decomposition algorithm for long time series. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 5409-5416, 2019.

温（Wen）、高（Gao）、宋（Song）、孙（Sun）、徐（Xu）和朱（Zhu）。《鲁棒季节性趋势分解算法（RobustSTL）：一种用于长时间序列的鲁棒季节性 - 趋势分解算法》。收录于《人工智能协会会议论文集》（Proceedings of the AAAI Conference on Artificial Intelligence），第33卷，第5409 - 5416页，2019年。

Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), pp. 101-112, 2021.

吴（Wu）、徐（Xu）、王（Wang）和龙（Long）。《自动变换器：用于长期序列预测的具有自相关的分解变换器》。收录于《神经信息处理系统进展会议论文集》（Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)），第101 - 112页，2021年。

Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., and Singh, V. Nyströmformer: A nyström-based algorithm for approximating self-attention. In Thirty-Fifth AAAI Conference on Artificial Intelligence, pp. 14138- 14148, 2021.

熊（Xiong）、曾（Zeng）、查克拉博蒂（Chakraborty）、谭（Tan）、冯（Fung）、李（Li）和辛格（Singh）。《Nyströmformer：一种基于Nyström方法的自注意力近似算法》。发表于《第三十五届美国人工智能协会会议》，第14138 - 14148页，2021年。

Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontañón, S., Pham, P., Ravula, A., Wang, Q., Yang, L., and Ahmed, A. Big bird: Transformers for longer sequences. In Annual Conference on Neural Information Processing Systems (NeurIPS), December 6- 12, 2020, virtual, 2020.

扎希尔（Zaheer）、古鲁加内什（Guruganesh）、杜贝（Dubey）、安斯利（Ainslie）、阿尔贝蒂（Alberti）、翁塔尼翁（Ontañón）、范（Pham）、拉武拉（Ravula）、王（Wang）、杨（Yang）和艾哈迈德（Ahmed）。《大飞鸟：用于长序列的Transformer模型》。发表于《神经信息处理系统年度会议（NeurIPS）》，2020年12月6 - 12日，线上会议，2020年。

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. Informer: Beyond efficient transformer for long sequence time-series forecasting. In The Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Virtual Conference, volume 35, pp. 11106-11115. AAAI Press, 2021.

周（Zhou）、张（Zhang）、彭（Peng）、张（Zhang）、李（Li）、熊（Xiong）和张（Zhang）。《Informer：超越高效Transformer的长序列时间序列预测》。收录于第三十五届人工智能协会会议（The Thirty-Fifth AAAI Conference on Artificial Intelligence，AAAI 2021），线上会议，第35卷，第11106 - 11115页。人工智能协会出版社（AAAI Press），2021年。

Zhu, Z. and Soricut, R. H-transformer-1d: Fast one-dimensional hierarchical attention for sequences. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL) 2021, Virtual Event, August 1-6, 2021, pp. 3801-3815, 2021.

朱（Zhu）和索里库特（Soricut）。《H-transformer-1d：用于序列的快速一维分层注意力机制》。收录于2021年计算语言学协会第59届年会（Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics，ACL 2021），线上活动，2021年8月1 - 6日，第3801 - 3815页，2021年。

## A. Related Work

## A. 相关工作

In this section, an overview of the literature for time series forecasting will be given. The relevant works include traditional times series models (A.1), deep learning models (A.1), Transformer-based models (A.2), and the Fourier Transform in neural networks (A.3).

本节将对时间序列预测的相关文献进行概述。相关工作包括传统时间序列模型（A.1）、深度学习模型（A.1）、基于Transformer的模型（A.2）以及神经网络中的傅里叶变换（A.3）。

### A.1. Traditional Time Series Models

### A.1. 传统时间序列模型

Data-driven time series forecasting helps researchers understand the evolution of the systems without architecting the exact physics law behind them. After decades of renovation, time series models have been well developed and served as the backbone of various projects in numerous application fields. The first generation of data-driven methods can date back to 1970. ARIMA (Box & Jenkins, 1968; Box & Pierce, 1970) follows the Markov process and builds an auto-regressive model for recursively sequential forecasting. However, an autoregressive process is not enough to deal with nonlinear and non-stationary sequences. With the bloom of deep neural networks in the new century, recurrent neural networks (RNN) was designed especially for tasks involving sequential data. Among the family of RNNs, LSTM (Hochreiter & Schmidhuber, 1997) and GRU (Chung et al., 2014) employ gated structure to control the information flow to deal with the gradient vanishing or exploration problem. DeepAR (Flunkert et al., 2017) uses a sequential architecture for probabilistic forecasting by incorporating binomial likelihood. Attention based RNN (Qin et al., 2017) uses temporal attention to capture long-range dependencies. However, the recurrent model is not parallelizable and unable to handle long dependencies. The temporal convolutional network (Sen et al., 2019) is another family efficient in sequential tasks. However, limited to the reception field of the kernel, the features extracted still stay local and long-term dependencies are hard to grasp.

数据驱动的时间序列预测有助于研究人员在不构建系统背后精确物理定律的情况下理解系统的演变。经过数十年的革新，时间序列模型得到了充分发展，并成为众多应用领域中各种项目的支柱。第一代数据驱动方法可以追溯到1970年。自回归积分滑动平均模型（ARIMA）（Box & Jenkins，1968年；Box & Pierce，1970年）遵循马尔可夫过程，并构建自回归模型进行递归顺序预测。然而，自回归过程不足以处理非线性和非平稳序列。随着新世纪深度神经网络的蓬勃发展，循环神经网络（RNN）专门为涉及顺序数据的任务而设计。在循环神经网络家族中，长短期记忆网络（LSTM）（Hochreiter & Schmidhuber，1997年）和门控循环单元（GRU）（Chung等人，2014年）采用门控结构来控制信息流，以解决梯度消失或梯度爆炸问题。深度自回归模型（DeepAR）（Flunkert等人，2017年）通过结合二项似然使用顺序架构进行概率预测。基于注意力机制的循环神经网络（Qin等人，2017年）使用时间注意力来捕捉长距离依赖关系。然而，循环模型无法并行计算，并且难以处理长依赖关系。时间卷积网络（Sen等人，2019年）是另一类在顺序任务中高效的网络。然而，由于受限于卷积核的感受野，提取的特征仍然是局部的，难以把握长期依赖关系。

### A.2. Transformers for Time Series Forecasting

### A.2. 用于时间序列预测的Transformer模型

With the innovation of transformers in natural language processing (Vaswani et al., 2017; Devlin et al., 2019) and computer vision tasks (Dosovitskiy et al., 2021; Rao et al., 2021), transformer-based models are also discussed, renovated, and applied in time series forecasting (Zhou et al., 2021; Wu et al., 2021). In sequence to sequence time series forecasting tasks an encoder-decoder architecture is popularly employed. The self-attention and cross-attention mechanisms are used as the core layers in transformers. However, when employing a point-wise connected matrix, the transformers suffer from quadratic computation complexity.

随着Transformer模型在自然语言处理（瓦斯瓦尼等人，2017年；德夫林等人，2019年）和计算机视觉任务（多索维茨基等人，2021年；饶等人，2021年）中的创新，基于Transformer的模型也在时间序列预测中得到了讨论、改进和应用（周等人，2021年；吴等人，2021年）。在序列到序列的时间序列预测任务中，编码器 - 解码器架构被广泛采用。自注意力和交叉注意力机制被用作Transformer模型的核心层。然而，当采用逐点连接矩阵时，Transformer模型会面临二次计算复杂度的问题。

To get efficient computation without sacrificing too much on performance, the earliest modifications specify the attention matrix with predefined patterns. Examples include: (Qiu et al., 2020) uses block-wise attention which reduces the complexity to the square of block size. Long-former (Beltagy et al., 2020) employs a stride window with fixed intervals. LogTrans (Li et al., 2019) uses log-sparse attention and achieves $N{\log }^{2}N$ complexity. H-transformer (Zhu & Soricut, 2021) uses a hierarchical pattern for sparse approximation of attention matrix with $O\left( n\right)$ complexity. Some work uses a combination of patterns (BIGBIRD (Zaheer et al., 2020)) mentioned above. Another strategy is to use dynamic patterns: Reformer (Kitaev et al., 2020) introduces a local-sensitive hashing which reduces the complexity to $N\log N$ . (Zhu &Soricut,2021) introduces a hierarchical pattern. Sinkhorn (Tay et al., 2020) employs a block sorting method to achieve quasi-global attention with only local windows.

为了在不过多牺牲性能的前提下实现高效计算，早期的改进方法采用预定义模式来指定注意力矩阵。示例包括：（邱等人，2020年）使用分块注意力机制，将复杂度降低至块大小的平方。长文档变换器（Long-former，贝尔塔吉等人，2020年）采用具有固定间隔的跨步窗口。对数变换器（LogTrans，李等人，2019年）使用对数稀疏注意力机制，实现了$N{\log }^{2}N$复杂度。H变换器（H-transformer，朱和索里库特，2021年）使用分层模式对注意力矩阵进行稀疏近似，复杂度为$O\left( n\right)$。一些工作结合使用了上述模式（大飞鸟模型（BIGBIRD，扎希尔等人，2020年））。另一种策略是使用动态模式：改革者模型（Reformer，基塔耶夫等人，2020年）引入了局部敏感哈希，将复杂度降低至$N\log N$。（朱和索里库特，2021年）引入了一种分层模式。辛克霍恩模型（Sinkhorn，泰等人，2020年）采用块排序方法，仅通过局部窗口实现准全局注意力。

Similarly,some work employs a top-k truncating to accelerate computing: Informer (Zhou et al., 2021) uses a KL-divergence based method to select top-k in attention matrix. This sparser matrix costs only $N\log N$ in complexity. Aut-oformer (Wu et al., 2021) introduces an auto-correlation block in place of canonical attention to get the sub-series level attention,which achieves $N\log N$ complexity with the help of Fast Fourier transform and top-k selection in an auto-correlation matrix.

同样，一些工作采用了前 k 截断法来加速计算：Informer（周等人，2021 年）使用基于 KL 散度的方法在注意力矩阵中选择前 k 个元素。这种更稀疏的矩阵的复杂度仅为 $N\log N$。Autoformer（吴等人，2021 年）引入了自相关模块来替代传统的注意力机制，以获得子序列级别的注意力，借助快速傅里叶变换和自相关矩阵中的前 k 选择，实现了 $N\log N$ 的复杂度。

Another emerging strategy is to employ a low-rank approximation of the attention matrix. Linformer (Wang et al., 2020) uses trainable linear projection to compress the sequence length and achieves $O\left( n\right)$ complexity and theoretically proves the boundary of approximation error based on JL lemma. Luna (Ma et al., 2021) develops a nested linear structure with $O\left( n\right)$ complexity. Nyströformer (Xiong et al., 2021) leverages the idea of Nyström approximation in the attention mechanism and achieves an $O\left( n\right)$ complexity. Performer (Choromanski et al., 2021) adopts an orthogonal random features approach to efficiently model kernel-izable attention mechanisms.

另一种新兴策略是对注意力矩阵进行低秩近似。Linformer（王等人，2020年）使用可训练的线性投影来压缩序列长度，实现了$O\left( n\right)$复杂度，并基于约翰逊 - 林登斯特劳斯引理（JL lemma）从理论上证明了近似误差的边界。Luna（马等人，2021年）开发了一种具有$O\left( n\right)$复杂度的嵌套线性结构。Nyströformer（熊等人，2021年）在注意力机制中运用了尼斯特伦近似（Nyström approximation）的思想，实现了$O\left( n\right)$复杂度。Performer（乔罗曼斯基等人，2021年）采用正交随机特征方法来高效建模可核化的注意力机制。

### A.3. Fourier Transform in Transformers

### A.3. Transformer中的傅里叶变换

Thanks to the algorithm of fast Fourier transform (FFT), the computation complexity of Fourier transform is compressed from ${N}^{2}$ to $N\log N$ . The Fourier transform has the property that convolution in the time domain is equivalent to multiplication in the frequency domain. Thus the FFT can be used in the acceleration of convolutional networks (Mathieu et al., 2014). FFT can also be used in efficient computing of auto-correlation function, which can be used as a building neural networks block (Wu et al., 2021) and also useful in numerous anomaly detection tasks (Homayouni et al., 2020). (Li et al., 2020; Gupta et al., 2021) first introduced Fourier Neural Operator in solving partial differential equations (PDEs). FNO is used as an inner block of networks to perform efficient representation learning in the low-frequency domain. FNO is also proved efficient in computer vision tasks (Rao et al., 2021). It also serves as a working horse to build the Wavelet Neural Operator (WNO), which is recently introduced in solving PEDs (Gupta et al., 2021). While FNO keeps the spectrum modes in low frequency, random Fourier method use randomly selected modes. (Rahimi & Recht, 2008) proposes to map the input data to a randomized low-dimensional feature space to accelerate the training of kernel machines. (Rawat et al., 2019) proposes the Random Fourier softmax (RF-softmax) method that utilizes the powerful Random Fourier Features to enable more efficient and accurate sampling from an approximate softmax distribution.

由于快速傅里叶变换（FFT）算法，傅里叶变换的计算复杂度从${N}^{2}$压缩到了$N\log N$。傅里叶变换具有时域卷积等价于频域乘法的性质。因此，FFT可用于加速卷积网络（马蒂厄等人，2014年）。FFT还可用于高效计算自相关函数，自相关函数可用作神经网络模块（吴等人，2021年），并且在众多异常检测任务中也很有用（霍马尤尼等人，2020年）。（李等人，2020年；古普塔等人，2021年）首次将傅里叶神经算子（Fourier Neural Operator，FNO）引入偏微分方程（PDEs）的求解中。FNO用作网络的内部模块，以在低频域中进行高效的表征学习。FNO在计算机视觉任务中也被证明是有效的（拉奥等人，2021年）。它还作为构建小波神经算子（Wavelet Neural Operator，WNO）的基础，小波神经算子最近被引入用于求解偏微分方程（古普塔等人，2021年）。虽然FNO保留低频的频谱模式，但随机傅里叶方法使用随机选择的模式。（拉希米和雷克特，2008年）提出将输入数据映射到随机低维特征空间，以加速核机器的训练。（拉瓦特等人，2019年）提出了随机傅里叶softmax（RF - softmax）方法，该方法利用强大的随机傅里叶特征，从近似softmax分布中实现更高效、准确的采样。

To the best of our knowledge, our proposed method is the first work to achieve fast attention mechanism through low rank approximated transformation in frequency domain for time series forecasting.

据我们所知，我们提出的方法是首个通过频域中的低秩近似变换实现快速注意力机制以进行时间序列预测的工作。

## B. Low-rank Approximation of Attention

## B. 注意力的低秩近似

In this section, we discuss the low-rank approximation of the attention mechanism. First, we present the Restricted Isometry Property (RIP) matrices whose approximate error bound could be theoretically given in B.1. Then in B.2, we follow prior work and present how to leverage RIP matrices and attention mechanisms.

在本节中，我们讨论注意力机制的低秩近似。首先，我们在B.1中介绍受限等距性质（Restricted Isometry Property，RIP）矩阵，其近似误差界可以从理论上给出。然后在B.2中，我们参考前人的工作，介绍如何利用RIP矩阵和注意力机制。

If the signal of interest is sparse or compressible on a fixed basis, then it is possible to recover the signal from fewer measurements. (Wang et al., 2020; Xiong et al., 2021) suggest that the attention matrix is low-rank, so the attention matrix can be well approximated if being projected into a subspace where the attention matrix is sparse. For the efficient computation of the attention matrix, how to properly select the basis of the projection yet remains to be an open question. The basis which follows the RIP is a potential candidate.

如果感兴趣的信号在固定基上是稀疏的或可压缩的，那么就有可能从较少的测量值中恢复该信号。（Wang等人，2020年；Xiong等人，2021年）指出注意力矩阵是低秩的，因此如果将注意力矩阵投影到一个使其稀疏的子空间中，就可以很好地对其进行近似。对于注意力矩阵的高效计算而言，如何恰当地选择投影基仍是一个悬而未决的问题。满足受限等距性质（RIP）的基是一个潜在的选择。

#### B.1.RIP Matrices

#### B.1.受限等距性质（RIP）矩阵

The definition of the RIP matrices is:

受限等距性质（RIP）矩阵的定义如下：

Definition B.1. RIP matrices. Let $m < n$ be positive integers, $\Phi$ be a $m \times  n$ matrix with real entries, $\delta  > 0$ ,and $K < m$ be an integer. We say that $\Phi$ is $\left( {K,\delta }\right)  - {RIP}$ ,if for every $K$ -sparse vector $x \in  {\mathbb{R}}^{n}$ we have $\left( {1 - \delta }\right) \parallel x\parallel  \leq$ $\parallel {\Phi x}\parallel  \leq  \left( {1 + \delta }\right) \parallel x\parallel .$

定义B.1. RIP矩阵。设$m < n$为正整数，$\Phi$为一个具有实元素的$m \times  n$矩阵，$\delta  > 0$，且$K < m$为整数。我们称$\Phi$是$\left( {K,\delta }\right)  - {RIP}$，如果对于每一个$K$ - 稀疏向量$x \in  {\mathbb{R}}^{n}$，我们有$\left( {1 - \delta }\right) \parallel x\parallel  \leq$ $\parallel {\Phi x}\parallel  \leq  \left( {1 + \delta }\right) \parallel x\parallel .$

RIP matrices are the matrices that satisfy the restricted isometry property, discovered by D. Donoho, E. Candès and T. Tao in the field of compressed sensing. RIP matrices might be good choices for low-rank approximation because of their good properties. A random matrix has a negligible probability of not satisfying the RIP and many kinds of matrices have proven to be RIP, for example, Gaussian basis, Bernoulli basis, and Fourier basis.

RIP矩阵是满足受限等距性质（Restricted Isometry Property）的矩阵，由D. 多诺霍（D. Donoho）、E. 坎德斯（E. Candès）和T. 陶哲轩（T. Tao）在压缩感知领域发现。由于RIP矩阵具有良好的性质，它们可能是低秩逼近的不错选择。随机矩阵不满足RIP的概率可以忽略不计，而且许多类型的矩阵已被证明是RIP矩阵，例如高斯基（Gaussian basis）、伯努利基（Bernoulli basis）和傅里叶基（Fourier basis）。

Theorem 2. Let $m < n$ be positive integers, $\delta  > 0$ ,and $K = O\left( \frac{m}{{\log }^{4}n}\right)$ . Let $\Phi$ be the random matrix defined by one of the following methods:

定理2。设$m < n$为正整数，$\delta  > 0$，且$K = O\left( \frac{m}{{\log }^{4}n}\right)$。设$\Phi$为通过以下方法之一定义的随机矩阵：

(Gaussian basis) Let the entries of $\Phi$ be i.i.d. with a normal distribution $N\left( {0,\frac{1}{m}}\right)$ .

（高斯基）设$\Phi$的元素是独立同分布的，且服从正态分布$N\left( {0,\frac{1}{m}}\right)$。

(Bernoulli basis) Let the entries of $\Phi$ be i.i.d. with a Bernoulli distribution taking the values $\pm  \frac{1}{\sqrt{m}}\mathrm{\;m}$ ,each with 50% probability.

（伯努利基（Bernoulli basis））设 $\Phi$ 的元素是独立同分布的，服从伯努利分布，取值为 $\pm  \frac{1}{\sqrt{m}}\mathrm{\;m}$ ，每个值的概率均为 50%。

(Random selected Discrete Fourier basis) Let $A \subset$ $\{ 0,\ldots ,n - 1\}$ be a random subset of size $m$ . Let $\Phi$ be the matrix obtained from the Discrete Fourier transform matrix (i.e. the matrix $F$ with entries $F\left\lbrack  {l,j}\right\rbrack   = {\exp }^{-{2\pi ilj}/n}/\sqrt{n}$ ) for $l,j \in  \{ 0,..,n - 1\}$ by selecting the rows indexed by $A$ .

（随机选择的离散傅里叶基（Random selected Discrete Fourier basis））设 $A \subset$ $\{ 0,\ldots ,n - 1\}$ 是一个大小为 $m$ 的随机子集。设 $\Phi$ 是通过选择离散傅里叶变换矩阵（即元素为 $F\left\lbrack  {l,j}\right\rbrack   = {\exp }^{-{2\pi ilj}/n}/\sqrt{n}$ 的矩阵 $F$ ）中由 $A$ 索引的行，为 $l,j \in  \{ 0,..,n - 1\}$ 得到的矩阵。

Then $\Phi$ is $\left( {K,\sigma }\right)  - {RIP}$ with probability $p \approx  1 - {e}^{-n}$ .

那么 $\Phi$ 以概率 $p \approx  1 - {e}^{-n}$ 等于 $\left( {K,\sigma }\right)  - {RIP}$ 。

Theorem 2 states that Gaussian basis, Bernoulli basis and Fourier basis follow RIP. In the following section, the Fourier basis is used as an example and show how to use RIP basis in low-rank approximation in the attention mechanism.

定理2表明，高斯基（Gaussian basis）、伯努利基（Bernoulli basis）和傅里叶基（Fourier basis）满足受限等距性质（RIP）。在接下来的部分，将以傅里叶基为例，展示如何在注意力机制的低秩近似中使用满足受限等距性质的基。

### B.2. Low-rank Approximation with Fourier Basis/Legendre Polynomials

### B.2. 使用傅里叶基/勒让德多项式的低秩近似

Linformer (Wang et al., 2020) demonstrates that the attention mechanism can be approximated by a low-rank matrix. Linformer uses a trainable kernel initialized with Gaussian distribution for the low-rank approximation, While our proposed FEDformer uses Fourier basis/Legendre Polynomials, Gaussian basis, Fourier basis, and Legendre Polynomials all obey RIP, so similar conclusions could be drawn.

Linformer（Wang等人，2020）证明了注意力机制可以用一个低秩矩阵来近似。Linformer使用一个初始化为高斯分布的可训练核进行低秩近似，而我们提出的FEDformer使用傅里叶基/勒让德多项式。高斯基、傅里叶基和勒让德多项式都满足受限等距性质，因此可以得出类似的结论。

Starting from Johnson-Lindenstrauss lemma (Johnson, 1984) and using the version from (Arriaga & Vempala, 2006), Linformer proves that a low-rank approximation of the attention matrix could be made.

从约翰逊 - 林登斯特劳斯引理（Johnson，1984）出发，并采用（Arriaga & Vempala，2006）中的版本，Linformer证明了可以对注意力矩阵进行低秩近似。

Let $\Phi  \in  {\mathbb{R}}^{N \times  M}$ be the random selected Fourier basis/Legendre Polynomials. $\Phi$ is RIP matrix. Referring to Theorem 2,with a probability $p \approx  1 - {e}^{-n}$ ,for any $x \in  {\mathbb{R}}^{N}$ , we have

设 $\Phi  \in  {\mathbb{R}}^{N \times  M}$ 为随机选取的傅里叶基/勒让德多项式（Fourier basis/Legendre Polynomials）。$\Phi$ 是受限等距性质（RIP）矩阵。参考定理 2，以概率 $p \approx  1 - {e}^{-n}$，对于任意 $x \in  {\mathbb{R}}^{N}$，我们有

$$
\left( {1 - \delta }\right) \parallel x\parallel  \leq  \parallel {\Phi x}\parallel  \leq  \left( {1 + \delta }\right) \parallel x\parallel . \tag{11}
$$

Referring to (Arriaga & Vempala, 2006), with a probability $p \approx  1 - 4{e}^{-n}$ ,for any ${x}_{1},{x}_{2} \in  {\mathbb{R}}^{N}$ ,we have

参考（阿瑞亚加（Arriaga）和温帕拉（Vempala），2006 年），以概率 $p \approx  1 - 4{e}^{-n}$，对于任意 ${x}_{1},{x}_{2} \in  {\mathbb{R}}^{N}$，我们有

$$
\left( {1 - \delta }\right) \begin{Vmatrix}{{x}_{1}{x}_{2}^{\top }}\end{Vmatrix} \leq  \begin{Vmatrix}{{x}_{1}{\Phi }^{\top }\Phi {x}_{2}^{\top }}\end{Vmatrix} \leq  \left( {1 + \delta }\right) \begin{Vmatrix}{{x}_{1}{x}_{2}^{\top }}\end{Vmatrix}. \tag{12}
$$

With the above inequation function, we now discuss the case in attention mechanism. Let the attention matrix

有了上述不等式函数，我们现在讨论注意力机制中的情况。设注意力矩阵

$B = \operatorname{softmax}\left( \frac{Q{K}^{\top }}{\sqrt{d}}\right)  = \exp \left( A\right)  \cdot  {D}_{A}^{-1}$ ,where ${\left( {D}_{A}\right) }_{ii} =$ $\mathop{\sum }\limits_{{n = 1}}^{N}\exp \left( {A}_{ni}\right)$ . Following Linformer,we can conclude a theorem as (please refer to (Wang et al., 2020) for the detailed proof)

$B = \operatorname{softmax}\left( \frac{Q{K}^{\top }}{\sqrt{d}}\right)  = \exp \left( A\right)  \cdot  {D}_{A}^{-1}$ ，其中 ${\left( {D}_{A}\right) }_{ii} =$ $\mathop{\sum }\limits_{{n = 1}}^{N}\exp \left( {A}_{ni}\right)$ 。遵循线性变换器（Linformer），我们可以得出一个定理如下（详细证明请参考（王等人，2020）(Wang et al., 2020)）

Theorem 3. For any row vector $p \in  {\mathbb{R}}^{N}$ of matrix $B$ and any column vector $v \in  {\mathbb{R}}^{N}$ of matrix $V$ ,with a probability $p = 1 - o\left( 1\right)$ ,we have

定理3。对于矩阵 $B$ 的任意行向量 $p \in  {\mathbb{R}}^{N}$ 和矩阵 $V$ 的任意列向量 $v \in  {\mathbb{R}}^{N}$ ，以概率 $p = 1 - o\left( 1\right)$ ，我们有

$$
\begin{Vmatrix}{b{\Phi }^{\top }\Phi {v}^{\top } - b{v}^{\top }}\end{Vmatrix} \leq  \delta \begin{Vmatrix}{b{v}^{\top }}\end{Vmatrix}. \tag{13}
$$

Theorem 3 points out the fact that, using Fourier basis/Legendre Polynomials $\Phi$ between the multiplication of attention matrix(P)and values(V),the computation complexity can be reduced from $O\left( {{N}^{2}d}\right)$ to $O\left( {NMd}\right)$ ,where $d$ is the hidden dimension of the matrix. In the meantime, the error of the low-rank approximation is bounded. However, Theorem 3 only discussed the case which is without the activation function.

定理3指出了这样一个事实：在注意力矩阵（P）和值矩阵（V）的乘法运算中，使用傅里叶基/勒让德多项式$\Phi$，计算复杂度可以从$O\left( {{N}^{2}d}\right)$降低到$O\left( {NMd}\right)$，其中$d$是矩阵的隐藏维度。同时，低秩近似的误差是有界的。然而，定理3仅讨论了没有激活函数的情况。

Furthermore, with the Cauchy inequality and the fact that the exponential function is Lipchitz continuous in a compact region (please refer to (Wang et al., 2020) for the proof), we can draw the following theorem:

此外，利用柯西不等式以及指数函数在紧致区域内是利普希茨连续的这一事实（证明请参考（Wang等人，2020）），我们可以得出以下定理：

Theorem 4. For any row vector ${A}_{i} \in  {\mathbb{R}}^{N}$ in matrix $A$ $\left( {A = \frac{Q{K}^{\top }}{\sqrt{d}}}\right)$ ,with a probability of $p = 1 - o\left( 1\right)$ ,we have

定理4. 对于矩阵$A$ $\left( {A = \frac{Q{K}^{\top }}{\sqrt{d}}}\right)$中的任意行向量${A}_{i} \in  {\mathbb{R}}^{N}$，以$p = 1 - o\left( 1\right)$的概率，我们有

$$
\begin{Vmatrix}{\exp \left( {{A}_{i}{\Phi }^{\top }}\right) \Phi {v}^{\top } - \exp \left( {A}_{i}\right) {v}^{\top }}\end{Vmatrix} \leq  \delta \begin{Vmatrix}{\exp \left( {A}_{i}\right) {v}^{\top }}\end{Vmatrix}. \tag{14}
$$

Theorem 4 states that with the activation function (soft-max), the above discussed bound still holds.

定理4表明，使用激活函数（soft-max）时，上述讨论的边界仍然成立。

In summary, we can leverage RIP matrices for low-rank approximation of attention. Moreover, there exists theoretical error bound when using a randomly selected Fourier basis for low-rank approximation in the attention mechanism.

综上所述，我们可以利用受限等距性质（RIP）矩阵对注意力机制进行低秩近似。此外，在注意力机制中使用随机选择的傅里叶基进行低秩近似时，存在理论误差边界。

## C. Fourier Component Selection

## C. 傅里叶分量选择

Let ${X}_{1}\left( t\right) ,\ldots ,{X}_{m}\left( t\right)$ be $m$ time series. By applying Fourier transform to each time series,we turn each ${X}_{i}\left( t\right)$ into a vector ${a}_{i} = {\left( {a}_{i,1},\ldots ,{a}_{i,d}\right) }^{\top } \in  {\mathbb{R}}^{d}$ . By putting all the Fourier transform vectors into a matrix, we have $A = {\left( {a}_{1},{a}_{2},\ldots ,{a}_{m}\right) }^{\top } \in  {\mathbb{R}}^{m \times  d}$ ,with each row corresponding to a different time series and each column corresponding to a different Fourier component. Here, we propose to select $s$ components from the $d$ Fourier components $\left( {s < d}\right)$ uniformly at random. More specifically,we denote by ${i}_{1} < {i}_{2} < \ldots  < {i}_{s}$ the randomly selected components. We construct matrix $S \in  \{ 0,1{\} }^{s \times  d}$ ,with ${S}_{i,k} = 1$ if $i = {i}_{k}$ and ${S}_{i,k} = 0$ otherwise. Then,our representation of multivariate time series becomes ${A}^{\prime } = A{S}^{\top } \in  {\mathbb{R}}^{m \times  s}$ . The following theorem shows that, although the Fourier basis is randomly selected,under a mild condition, ${A}^{\prime }$ can preserve most of the information from $A$ .

设${X}_{1}\left( t\right) ,\ldots ,{X}_{m}\left( t\right)$为$m$时间序列。通过对每个时间序列应用傅里叶变换（Fourier transform），我们将每个${X}_{i}\left( t\right)$转换为向量${a}_{i} = {\left( {a}_{i,1},\ldots ,{a}_{i,d}\right) }^{\top } \in  {\mathbb{R}}^{d}$。将所有傅里叶变换向量放入一个矩阵中，我们得到$A = {\left( {a}_{1},{a}_{2},\ldots ,{a}_{m}\right) }^{\top } \in  {\mathbb{R}}^{m \times  d}$，其中每一行对应一个不同的时间序列，每一列对应一个不同的傅里叶分量（Fourier component）。在此，我们提议从$d$个傅里叶分量$\left( {s < d}\right)$中均匀随机地选择$s$个分量。更具体地说，我们用${i}_{1} < {i}_{2} < \ldots  < {i}_{s}$表示随机选择的分量。我们构建矩阵$S \in  \{ 0,1{\} }^{s \times  d}$，若$i = {i}_{k}$，则${S}_{i,k} = 1$；否则${S}_{i,k} = 0$。然后，我们对多元时间序列的表示变为${A}^{\prime } = A{S}^{\top } \in  {\mathbb{R}}^{m \times  s}$。以下定理表明，尽管傅里叶基是随机选择的，但在一个温和的条件下，${A}^{\prime }$可以保留$A$的大部分信息。

Theorem 5. Assume that $\mu \left( A\right)$ ,the coherence measure of matrix $A$ ,is $\Omega \left( {k/n}\right)$ . Then,with a high probability,we have

定理5. 假设矩阵$A$的相干性度量$\mu \left( A\right)$为$\Omega \left( {k/n}\right)$。那么，在大概率情况下，我们有

$$
\left| {A - {P}_{{A}^{\prime }}\left( A\right) }\right|  \leq  \left( {1 + \epsilon }\right) \left| {A - {A}_{k}}\right| 
$$

if $s = O\left( {{k}^{2}/{\epsilon }^{2}}\right)$ .

如果$s = O\left( {{k}^{2}/{\epsilon }^{2}}\right)$。

Proof. Following the analysis in Theorem 3 from (Drineas et al., 2007), we have

证明. 按照（德里尼亚斯等人（Drineas et al.），2007年）中定理3的分析，我们有

$$
\left| {A - {P}_{{A}^{\prime }}\left( A\right) }\right|  \leq  \left| {A - {A}^{\prime }{\left( {A}^{\prime }\right) }^{ \dagger  }{A}_{k}}\right| 
$$

$$
 = \left| {A - \left( {A{S}^{\top }}\right) {\left( A{S}^{\top }\right) }^{ \dagger  }{A}_{k}}\right| 
$$

$$
 = \left| {A - \left( {A{S}^{\top }}\right) {\left( {A}_{k}{S}^{\top }\right) }^{ \dagger  }{A}_{k}}\right| .
$$

Using Theorem 5 from (Drineas et al., 2007), we have, with a probability at least 0.7 ,

使用（德里尼亚斯等人（Drineas et al.），2007年）中的定理5，我们可知，至少有0.7的概率

$$
\left| {A - \left( {A{S}^{\top }}\right) {\left( {A}_{k}{S}^{\top }\right) }^{ \dagger  }{A}_{k}}\right|  \leq  \left( {1 + \epsilon }\right) \left| {A - {A}_{k}}\right| 
$$

if $s = O\left( {{k}^{2}/{\epsilon }^{2} \times  \mu \left( A\right) n/k}\right)$ . The theorem follows because $\mu \left( A\right)  = O\left( {k/n}\right)$ .

如果 $s = O\left( {{k}^{2}/{\epsilon }^{2} \times  \mu \left( A\right) n/k}\right)$ 。该定理成立是因为 $\mu \left( A\right)  = O\left( {k/n}\right)$ 。

## D. Wavelets

## D. 小波

In this section, we present some technical background about Wavelet transform which is used in our proposed framework.

在本节中，我们介绍一些关于小波变换（Wavelet transform）的技术背景，该变换将用于我们提出的框架中。

### D.1. Continuous Wavelet Transform

### D.1. 连续小波变换

First,let’s see how a function $f\left( t\right)$ is decomposed into a set of basis functions ${\psi }_{\mathrm{s},\tau }\left( t\right)$ ,called the wavelets. It is known as the continuous wavelet transform or ${CWT}$ . More formally it is written as

首先，让我们看看函数 $f\left( t\right)$ 是如何分解为一组被称为小波（wavelets）的基函数 ${\psi }_{\mathrm{s},\tau }\left( t\right)$ 的。这被称为连续小波变换（continuous wavelet transform）或 ${CWT}$ 。更正式地，它可以写成

$$
\gamma \left( {s,\tau }\right)  = \int f\left( t\right) {\Psi }_{s,\tau }^{ * }\left( t\right) {dt}
$$

where * denotes complex conjugation. This equation shows the variables $\gamma \left( {s,\tau }\right) ,s$ and $\tau$ are the new dimensions,scale, and translation after the wavelet transform, respectively.

其中 * 表示复共轭。该方程表明变量 $\gamma \left( {s,\tau }\right) ,s$ 和 $\tau$ 分别是小波变换后的新维度、尺度和位移。

The wavelets are generated from a single basic wavelet $\Psi \left( t\right)$ ,the so-called mother wavelet,by scaling and translation as

小波是由单个基本小波 $\Psi \left( t\right)$（即所谓的母小波）通过缩放和平移生成的，如下所示

$$
{\psi }_{s,\tau }\left( t\right)  = \frac{1}{\sqrt{s}}\psi \left( \frac{t - \tau }{s}\right) ,
$$

where $s$ is the scale factor, $\tau$ is the translation factor,and $\sqrt{s}$ is used for energy normalization across the different scales.

其中 $s$ 是尺度因子，$\tau$ 是平移因子，$\sqrt{s}$ 用于不同尺度间的能量归一化。

### D.2. Discrete Wavelet Transform

### D.2. 离散小波变换

Continues wavelet transform maps a one-dimensional signal to a two-dimensional time-scale joint representation which is highly redundant. To overcome this problem, people introduce discrete wavelet transformation (DWT) with mother wavelet as

连续小波变换将一维信号映射到二维时 - 频联合表示，这种表示具有高度冗余性。为克服这一问题，人们引入了以母小波为基础的离散小波变换（DWT）

$$
{\psi }_{j,k}\left( t\right)  = \frac{1}{\sqrt{{s}_{0}^{j}}}\psi \left( \frac{t - k{\tau }_{0}{s}_{0}^{j}}{{s}_{0}^{j}}\right) 
$$

DWT is not continuously scalable and translatable but can be scaled and translated in discrete steps. Here $j$ and $k$ are integers and ${s}_{0} > 1$ is a fixed dilation step. The translation factor ${\tau }_{0}$ depends on the dilation step. The effect of discretizing the wavelet is that the time-scale space is now sampled at discrete intervals. We usually choose ${s}_{0} = 2$ so that the sampling of the frequency axis corresponds to dyadic sampling. For the translation factor, we usually choose ${\tau }_{0} = 1$ so that we also have a dyadic sampling of the time axis.

离散小波变换（DWT）并非连续可缩放和平移的，但可以进行离散的缩放和平移操作。这里$j$和$k$为整数，${s}_{0} > 1$是固定的伸缩步长。平移因子${\tau }_{0}$取决于伸缩步长。对小波进行离散化处理的结果是，时频空间现在以离散间隔进行采样。我们通常选择${s}_{0} = 2$，使得频率轴的采样对应于二进采样。对于平移因子，我们通常选择${\tau }_{0} = 1$，这样时间轴也能进行二进采样。

When discrete wavelets are used to transform a continuous signal, the result will be a series of wavelet coefficients and it is referred to as the wavelet decomposition.

当使用离散小波对连续信号进行变换时，结果将是一系列小波系数，这一过程被称为小波分解。

### D.3. Orthogonal Polynomials

### D.3. 正交多项式

The next thing we need to focus on is orthogonal polynomials (OPs), which will serve as the mother wavelet function we introduce before. A lot of properties have to be maintained to be a mother wavelet, like admissibility condition, regularity conditions, and vanishing moments. In short, we are interested in the OPs that are non-zero over a finite domain and are zero almost everywhere else. Legendre is a popular set of OPs used it in our work here. Some other popular OPs can also be used here like Chebyshev without much modification.

我们接下来需要关注的是正交多项式（OPs），它将作为我们之前介绍过的母小波函数。要成为母小波，需要满足许多性质，如可容许性条件、正则性条件和消失矩。简而言之，我们关注的是在有限区间内非零，而在其他几乎所有地方都为零的正交多项式。勒让德多项式（Legendre）是一组常用的正交多项式，我们在本文中使用了它。其他一些常用的正交多项式，如切比雪夫多项式（Chebyshev），也可以稍作修改后使用。

### D.4. Legendre Polynomails

### D.4. 勒让德多项式（Legendre Polynomails）

The Legendre polynomials are defined with respect to (w.r.t.) a uniform weight function ${w}_{L}\left( x\right)  = 1$ for $- 1 \leq$ $x \leq  1$ or ${w}_{L}\left( x\right)  = {\mathbf{1}}_{\left\lbrack  -1,1\right\rbrack  }\left( x\right)$ such that

勒让德多项式是相对于（w.r.t.）一个均匀权函数 ${w}_{L}\left( x\right)  = 1$ 来定义的，其中 $- 1 \leq$ $x \leq  1$ 或 ${w}_{L}\left( x\right)  = {\mathbf{1}}_{\left\lbrack  -1,1\right\rbrack  }\left( x\right)$ 满足以下条件

$$
{\int }_{-1}^{1}{P}_{i}\left( x\right) {P}_{j}\left( x\right) {dx} = \left\{  \begin{array}{ll} \frac{2}{{2i} + 1} & i = j, \\  0 & i \neq  j. \end{array}\right. 
$$

Here the function is defined over $\left\lbrack  {-1,1}\right\rbrack$ ,but it can be extended to any interval $\left\lbrack  {a,b}\right\rbrack$ by performing different shift and scale operations.

在此，该函数定义在$\left\lbrack  {-1,1}\right\rbrack$上，但通过执行不同的平移和缩放操作，它可以扩展到任何区间$\left\lbrack  {a,b}\right\rbrack$。

### D.5. Multiwavelets

### D.5. 多小波

The multiwavelets which we use in this work combine advantages of the wavelet and OPs we introduce before. Other than projecting a given function onto a single wavelet function, multiwavelet projects it onto a subspace of degree-restricted polynomials. In this work, we restricted our exploration to one family of OPs: Legendre Polynomials.

我们在这项工作中使用的多小波结合了我们之前介绍的小波和正交多项式（OPs）的优点。与将给定函数投影到单个小波函数上不同，多小波将其投影到一个次数受限的多项式子空间上。在这项工作中，我们将研究范围限制在一类正交多项式上：勒让德多项式（Legendre Polynomials）。

First, the basis is defined as: A set of orthonormal basis w.r.t. measure $\mu$ ,are ${\phi }_{0},\ldots ,{\phi }_{k - 1}$ such that ${\left\langle  {\phi }_{i},{\phi }_{j}\right\rangle  }_{\mu } = {\delta }_{ij}$ . With a specific measure (weighting function $w\left( x\right)$ ),the orthonormality condition can be written as $\int {\phi }_{i}\left( x\right) {\phi }_{j}\left( x\right) w\left( x\right) {dx} = {\delta }_{ij}.$

首先，基的定义如下：关于测度 $\mu$ 的一组标准正交基为 ${\phi }_{0},\ldots ,{\phi }_{k - 1}$，使得 ${\left\langle  {\phi }_{i},{\phi }_{j}\right\rangle  }_{\mu } = {\delta }_{ij}$ 成立。对于特定的测度（加权函数 $w\left( x\right)$），标准正交性条件可以写成 $\int {\phi }_{i}\left( x\right) {\phi }_{j}\left( x\right) w\left( x\right) {dx} = {\delta }_{ij}.$

Follow the derivation in (Gupta et al., 2021), through using the tools of Gaussian Quadrature and Gram-Schmidt Orthogonalizaition, the filter coefficients of multiwavelets using Legendre polynomials can be written as

遵循（古普塔等人（Gupta et al.），2021 年）的推导过程，通过使用高斯求积法和格拉姆 - 施密特正交化（Gram - Schmidt Orthogonalizaition）工具，使用勒让德多项式（Legendre polynomials）的多小波滤波器系数可以表示为

$$
{H}_{ij}^{\left( 0\right) } = \sqrt{2}{\int }_{0}^{1/2}{\phi }_{i}\left( x\right) {\phi }_{j}\left( {2x}\right) {w}_{L}\left( {{2x} - 1}\right) {dx}
$$

$$
 = \frac{1}{\sqrt{2}}{\int }_{0}^{1}{\phi }_{i}\left( {x/2}\right) {\phi }_{j}\left( x\right) {dx}
$$

$$
 = \frac{1}{\sqrt{2}}\mathop{\sum }\limits_{{i = 1}}^{k}{\omega }_{i}{\phi }_{i}\left( \frac{{x}_{i}}{2}\right) {\phi }_{j}\left( {x}_{i}\right) .
$$

For example,if $k = 3$ ,following the formula,the filter coefficients are derived as follows

例如，如果 $k = 3$，根据该公式，滤波器系数推导如下

$$
{H}^{0} = \left\lbrack  \begin{matrix} \frac{1}{\sqrt{2}} & 0 & 0 & \frac{1}{\sqrt{2}} & 0 & 0 \\   - \frac{\sqrt{3}}{2\sqrt{2}} & \frac{1}{2\sqrt{2}} & 0 & 0 & & \\  0 &  - \frac{\sqrt{15}}{4\sqrt{2}} & \frac{1}{4\sqrt{2}} & 0 & \frac{1}{4\sqrt{2}} &  \end{matrix}\right\rbrack  ,{H}^{1} = \left\lbrack  \begin{matrix} \frac{1}{\sqrt{2}} & 0 & 0 \\  \frac{\sqrt{3}}{2\sqrt{2}} & \frac{1}{2\sqrt{2}} & 0 \\  0 & \frac{\sqrt{15}}{4\sqrt{2}} & \frac{1}{4\sqrt{2}} \end{matrix}\right\rbrack  ,
$$

$$
{G}^{0} = \left\lbrack  \begin{matrix} \frac{1}{2\sqrt{2}} & \frac{\sqrt{3}}{2\sqrt{2}} & 0 &  - \frac{1}{2\sqrt{2}} & \frac{\sqrt{3}}{2\sqrt{2}} & 0 \\  0 & \frac{1}{4\sqrt{2}} & \frac{\sqrt{15}}{4\sqrt{2}} & 0 &  - \left\lbrack  \begin{matrix} 0 &  - \frac{1}{2\sqrt{2}} & \frac{\sqrt{15}}{4\sqrt{2}} \\  0 & 0 &  - \frac{1}{\sqrt{2}} \end{matrix}\right\rbrack  &  \end{matrix}\right\rbrack  
$$

## E. Output Distribution Analysis

## E. 输出分布分析

#### E.1.Bad Case Analysis

#### E.1. 不良情况分析

Using vanilla Transformer as baseline model, we demonstrate two bad long-term series forecasting cases in ETTm1 dataset as shown in the following Figure 7. The forecasting shifts in Figure 7 is particularly related to the point-wise generation mechanism adapted by the vanilla Transformer model. To the contrary of classic models like Autoregressive integrated moving average (ARIMA) which has a predefined data bias structure for output distribution, Transformer-based models forecast each point independently and solely based on the overall MSE loss learning. This would result in different distribution between ground truth and forecasting output in some cases, leading to performance degradation.

以原始Transformer作为基准模型，我们在ETTm1数据集中展示了两个糟糕的长期序列预测案例，如下图7所示。图7中的预测偏移尤其与原始Transformer模型采用的逐点生成机制有关。与经典模型（如自回归积分滑动平均模型（Autoregressive integrated moving average，ARIMA））不同，ARIMA对输出分布有预定义的数据偏差结构，而基于Transformer的模型独立地预测每个点，并且仅基于整体均方误差（MSE）损失学习。这在某些情况下会导致真实值和预测输出之间的分布不同，从而导致性能下降。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957f69-db4d-7f34-af19-bfc9e93e263b_14.jpg?x=152&y=1744&w=677&h=241&r=0"/>

Figure 7. Different distribution between ground truth and forecasting output from vanilla Transformer in a real-world ETTm1 dataset. Left: frequency mode and trend shift. Right: trend shift.

图7. 在真实世界的ETTm1数据集中，原始Transformer的真实值和预测输出之间的不同分布。左：频率模式和趋势偏移。右：趋势偏移。

<!-- Media -->

### E.2. Kolmogorov-Smirnov Test

### E.2. 柯尔莫哥洛夫 - 斯米尔诺夫检验（Kolmogorov - Smirnov Test）

We adopt Kolmogorov-Smirnov (KS) test to check whether the two data samples come from the same distribution. KS test is a nonparametric test of the equality of continuous or discontinuous, two-dimensional probability distributions. In essence, the test answers the question "what is the probability that these two sets of samples were drawn from the same (but unknown) probability distribution". It quantifies a distance between the empirical distribution function of two samples. The Kolmogorov-Smirnov statistic is

我们采用柯尔莫哥洛夫 - 斯米尔诺夫（Kolmogorov - Smirnov，KS）检验来检查两个数据样本是否来自同一分布。KS检验是一种用于检验连续或不连续二维概率分布是否相等的非参数检验。本质上，该检验回答了这样一个问题：“这两组样本是从同一个（但未知的）概率分布中抽取出来的概率是多少”。它量化了两个样本的经验分布函数之间的距离。柯尔莫哥洛夫 - 斯米尔诺夫统计量为

$$
{D}_{n,m} = \mathop{\sup }\limits_{x}\left| {{F}_{1,n}\left( x\right)  - {F}_{2,m}\left( x\right) }\right| 
$$

where ${F}_{1,n}$ and ${F}_{2,m}$ are the empirical distribution functions of the first and the second sample respectively, and sup is the supremum function. For large samples, the null hypothesis is rejected at level $\alpha$ if

其中${F}_{1,n}$和${F}_{2,m}$分别是第一个和第二个样本的经验分布函数，sup是上确界函数。对于大样本，如果满足以下条件，则在显著性水平$\alpha$下拒绝原假设

$$
{D}_{n,m} > \sqrt{-\frac{1}{2}\ln \left( \frac{\alpha }{2}\right) } \cdot  \sqrt{\frac{n + m}{n \cdot  m}},
$$

where $n$ and $m$ are the sizes of the first and second samples respectively.

其中$n$和$m$分别是第一个和第二个样本的大小。

### E.3. Distribution Experiments and Analysis

### E.3. 分布实验与分析

Though the KS test omits the temporal information from the input and output sequence, it can be used as a tool to measure the global property of the foretasting output sequence compared to the input sequence. The null hypothesis is that the two samples come from the same distribution. We can tell that if the P-value of the KS test is large and then the null hypothesis is less likely to be rejected for true output distribution.

尽管柯尔莫哥洛夫 - 斯米尔诺夫检验（KS检验）忽略了输入和输出序列中的时间信息，但它可以作为一种工具，用于衡量预尝输出序列相对于输入序列的全局特性。原假设是这两个样本来自同一分布。我们可以判断，如果KS检验的P值较大，那么对于真实的输出分布，原假设不太可能被拒绝。

We applied KS test on the output sequence of 96-720 prediction tasks for various models on the ETTm1 and ETTm2 datasets, and the results are summarized in Table 6. In the test, we compare the fixed 96-time step input sequence distribution with the output sequence distribution of different lengths. Using a 0.01 P-value as statistics, various existing Transformer baseline models have much less P-value than 0.01 except Autoformer, which indicates they have a higher probability to be sampled from the different distributions. Autoformer and FEDformer have much larger P value compared to other models, which mainly contributes to their seasonal trend decomposition mechanism. Though we get close results from ETTm1 by both models, the proposed FEDformer has much larger P-values in ETTm1. And it is the only model whose null hypothesis can not be rejected with P-value larger than 0.01 in all cases of the two datasets, implying that the output sequence generated by FEDformer shares a more similar distribution as the input sequence than others and thus justifies the our design motivation of FEDformer as discussed in Section 1.

我们对ETTm1和ETTm2数据集上各种模型的96 - 720预测任务的输出序列进行了KS检验（Kolmogorov - Smirnov检验），结果总结在表6中。在检验中，我们将固定的96个时间步长的输入序列分布与不同长度的输出序列分布进行比较。以0.01的P值作为统计量，除Autoformer外，各种现有的Transformer基线模型的P值远小于0.01，这表明它们更有可能是从不同的分布中抽样得到的。与其他模型相比，Autoformer和FEDformer的P值要大得多，这主要归功于它们的季节性趋势分解机制。虽然这两个模型在ETTm1上得到的结果相近，但所提出的FEDformer在ETTm1中的P值要大得多。并且它是唯一在两个数据集的所有情况下P值都大于0.01而不能拒绝原假设的模型，这意味着FEDformer生成的输出序列与输入序列的分布比其他模型更相似，从而证明了我们在第1节中讨论的FEDformer的设计动机是合理的。

<!-- Media -->

Table 6. Kolmogrov-Smirnov test P value for long sequence time-series forecasting output on ETT dataset (full experiment)

表6. ETT数据集上长序列时间序列预测输出的科尔莫戈罗夫 - 斯米尔诺夫检验（Kolmogrov-Smirnov test）P值（完整实验）

<table><tr><td colspan="2">Methods</td><td>Transformer</td><td>LogTrans</td><td>Informer</td><td>Reformer</td><td>Autoformer</td><td>FEDformer</td><td>True</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.0090</td><td>0.0073</td><td>0.0055</td><td>0.0055</td><td>0.020</td><td>0.048</td><td>0.023</td></tr><tr><td>192</td><td>0.0052</td><td>0.0043</td><td>0.0029</td><td>0.0013</td><td>0.015</td><td>0.028</td><td>0.013</td></tr><tr><td>336</td><td>0.0022</td><td>0.0026</td><td>0.0019</td><td>0.0006</td><td>0.012</td><td>0.015</td><td>0.010</td></tr><tr><td>720</td><td>0.0023</td><td>0.0064</td><td>0.0016</td><td>0.0011</td><td>0.008</td><td>0.014</td><td>0.004</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.0012</td><td>0.0025</td><td>0.0008</td><td>0.0028</td><td>0.078</td><td>0.071</td><td>0.087</td></tr><tr><td>192</td><td>0.0011</td><td>0.0011</td><td>0.0006</td><td>0.0015</td><td>0.047</td><td>0.045</td><td>0.060</td></tr><tr><td>336</td><td>0.0005</td><td>0.0011</td><td>0.00009</td><td>0.0007</td><td>0.027</td><td>0.028</td><td>0.042</td></tr><tr><td>720</td><td>0.0008</td><td>0.0005</td><td>0.0002</td><td>0.0005</td><td>0.023</td><td>0.021</td><td>0.023</td></tr><tr><td colspan="2">Count</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>5</td><td>NA</td></tr></table>

<table><tbody><tr><td colspan="2">方法</td><td>变换器（Transformer）</td><td>对数变换（LogTrans）</td><td>信息者（Informer）</td><td>改革者（Reformer）</td><td>自动变换器（Autoformer）</td><td>FEDformer（联邦前馈变换器）</td><td>真</td></tr><tr><td rowspan="4">ETTm1（电力变压器时间序列数据集1）</td><td>96</td><td>0.0090</td><td>0.0073</td><td>0.0055</td><td>0.0055</td><td>0.020</td><td>0.048</td><td>0.023</td></tr><tr><td>192</td><td>0.0052</td><td>0.0043</td><td>0.0029</td><td>0.0013</td><td>0.015</td><td>0.028</td><td>0.013</td></tr><tr><td>336</td><td>0.0022</td><td>0.0026</td><td>0.0019</td><td>0.0006</td><td>0.012</td><td>0.015</td><td>0.010</td></tr><tr><td>720</td><td>0.0023</td><td>0.0064</td><td>0.0016</td><td>0.0011</td><td>0.008</td><td>0.014</td><td>0.004</td></tr><tr><td rowspan="4">ETTm2（电力变压器时间序列数据集2）</td><td>96</td><td>0.0012</td><td>0.0025</td><td>0.0008</td><td>0.0028</td><td>0.078</td><td>0.071</td><td>0.087</td></tr><tr><td>192</td><td>0.0011</td><td>0.0011</td><td>0.0006</td><td>0.0015</td><td>0.047</td><td>0.045</td><td>0.060</td></tr><tr><td>336</td><td>0.0005</td><td>0.0011</td><td>0.00009</td><td>0.0007</td><td>0.027</td><td>0.028</td><td>0.042</td></tr><tr><td>720</td><td>0.0008</td><td>0.0005</td><td>0.0002</td><td>0.0005</td><td>0.023</td><td>0.021</td><td>0.023</td></tr><tr><td colspan="2">计数</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>5</td><td>不适用</td></tr></tbody></table>

Table 7. Summarized feature details of six datasets.

表7. 六个数据集的特征详情总结。

<table><tr><td>DATASET</td><td>LEN</td><td>DIM</td><td>FREQ</td></tr><tr><td>ETTM2</td><td>69680</td><td>8</td><td>15 MIN</td></tr><tr><td>ELECTRICITY</td><td>26304</td><td>322</td><td>1H</td></tr><tr><td>EXCHANGE</td><td>7588</td><td>9</td><td>1 DAY</td></tr><tr><td>TRAFFIC</td><td>17544</td><td>863</td><td>1H</td></tr><tr><td>WEATHER</td><td>52696</td><td>22</td><td>10 MIN</td></tr><tr><td>ILI</td><td>966</td><td>8</td><td>7 DAYS</td></tr></table>

<table><tbody><tr><td>数据集</td><td>长度</td><td>维度</td><td>频率</td></tr><tr><td>ETTM2</td><td>69680</td><td>8</td><td>15分钟</td></tr><tr><td>电力</td><td>26304</td><td>322</td><td>1H</td></tr><tr><td>交换；交易</td><td>7588</td><td>9</td><td>1天</td></tr><tr><td>交通</td><td>17544</td><td>863</td><td>1H</td></tr><tr><td>天气</td><td>52696</td><td>22</td><td>10分钟</td></tr><tr><td>流感样疾病（ILI）</td><td>966</td><td>8</td><td>7天</td></tr></tbody></table>

<!-- Media -->

Note that in the ETTm1 dataset, the True output sequence has a smaller P-value compared to our FEDformer's predicted output, it shows that the model's close output distribution is achieved through model's control other than merely more accurate prediction. This analysis shed some light on why the seasonal-trend decomposition architecture can give us better performance in long-term forecasting. The design is used to constrain the trend (mean) of the output distribution. Inspired by such observation, we design frequency enhanced block to constrain the seasonality (frequency mode) of the output distribution.

请注意，在ETTm1数据集（ETTm1数据集）中，真实输出序列与我们的FEDformer（频率增强分解变换器）预测输出相比，具有更小的P值，这表明模型接近的输出分布是通过模型控制实现的，而不仅仅是更准确的预测。这一分析揭示了为什么季节性趋势分解架构能在长期预测中为我们带来更好的性能。该设计用于约束输出分布的趋势（均值）。受此观察的启发，我们设计了频率增强模块来约束输出分布的季节性（频率模式）。

## F. Supplemental Experiments

## F. 补充实验

### F.1. Dataset Details

### F.1. 数据集详情

In this paragraph, the details of the experiment datasets are summarized as follows: 1) ETT (Zhou et al., 2021) dataset contains two sub-dataset: ETT1 and ETT2, collected from two electricity transformers at two stations. Each of them has two versions in different resolutions ( ${15}\mathrm{\;{min}}\& 1\mathrm{\;h}$ ). ETT dataset contains multiple series of loads and one series of oil temperatures. 2) Electricity ${}^{1}$ dataset contains the electricity consumption of clients with each column corresponding to one client. 3) Exchange (Lai et al., 2018) contains the current exchange of 8 countries. 4) Traffic ${}^{2}$ dataset contains the occupation rate of freeway system across the State of California. 5) Weather ${}^{3}$ dataset contains 21 meteorological indicators for a range of 1 year in Germany. 6) Ill- ${\text{ness}}^{4}$ dataset contains the influenza-like illness patients in the United States. Table 7 summarizes feature details (Sequence Length: Len, Dimension: Dim, Frequency: Freq) of the six datasets. All datasets are split into the training set, validation set and test set by the ratio of 7:1:2.

在这段中，实验数据集的详细信息总结如下：1) ETT（周等人，2021）数据集包含两个子数据集：ETT1和ETT2，它们是从两个站点的两台电力变压器收集而来。每个子数据集都有两种不同分辨率的版本（${15}\mathrm{\;{min}}\& 1\mathrm{\;h}$）。ETT数据集包含多个负荷序列和一个油温序列。2) 电力${}^{1}$数据集包含客户的用电量，每一列对应一个客户。3) 汇率（赖等人，2018）数据集包含8个国家的当前汇率。4) 交通${}^{2}$数据集包含加利福尼亚州高速公路系统的占有率。5) 气象${}^{3}$数据集包含德国一年的21个气象指标。6) 流感${\text{ness}}^{4}$数据集包含美国的流感样疾病患者数据。表7总结了这六个数据集的特征细节（序列长度：Len，维度：Dim，频率：Freq）。所有数据集均按7:1:2的比例划分为训练集、验证集和测试集。

### F.2. Implementation Details

### F.2. 实现细节

Our model is trained using ADAM (Kingma & Ba, 2017) optimizer with a learning rate of $1{e}^{-4}$ . The batch size is set to 32. An early stopping counter is employed to stop the training process after three epochs if no loss degradation on the valid set is observed. The mean square error (MSE) and mean absolute error (MAE) are used as metrics. All experiments are repeated 5 times and the mean of the metrics is used in the final results. All the deep learning networks are implemented in PyTorch (Paszke et al., 2019) and trained on NVIDIA V100 32GB GPUs.

我们的模型使用ADAM优化器（金玛和巴，2017年）进行训练，学习率为$1{e}^{-4}$。批量大小设置为32。如果在验证集上观察到三个周期内损失没有下降，则使用提前停止计数器来终止训练过程。均方误差（MSE）和平均绝对误差（MAE）被用作评估指标。所有实验重复5次，最终结果采用指标的平均值。所有深度学习网络均在PyTorch（帕兹克等人，2019年）中实现，并在NVIDIA V100 32GB GPU上进行训练。

#### F.3.ETT Full Benchmark

#### F.3. ETT完整基准测试

We present the full-benchmark on the four ETT datasets (Zhou et al., 2021) in Table 8 (multivariate forecasting) and Table 9 (univariate forecasting). The ETTh1 and ETTh2 are recorded hourly while ETTm1 and ETTm2 are recorded every 15 minutes. The time series in ETTh1 and ETTm1 follow the same pattern, and the only difference is the sampling rate,similarly for ETTh2 and ETTm2. On average, our FEDformer yields a 11.5% relative MSE reduction for multivariate forecasting,and a ${9.4}\%$ reduction for univariate forecasting over the SOTA results from Autoformer.

我们在表8（多变量预测）和表9（单变量预测）中展示了四个ETT数据集（周等人，2021）的完整基准测试结果。ETTh1和ETTh2按小时记录，而ETTm1和ETTm2每15分钟记录一次。ETTh1和ETTm1中的时间序列遵循相同的模式，唯一的区别是采样率，ETTh2和ETTm2也是如此。平均而言，与Autoformer的最优（SOTA）结果相比，我们的FEDformer在多变量预测中相对均方误差（MSE）降低了11.5%，在单变量预测中降低了${9.4}\%$。

---

<!-- Footnote -->

${}^{1}$ https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams 20112014

${}^{1}$ https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams 20112014

${}^{2}$ http://pems.dot.ca.gov

${}^{2}$ http://pems.dot.ca.gov

${}^{3}$ https://www.bgc-jena.mpg.de/wetter/

${}^{3}$ https://www.bgc-jena.mpg.de/wetter/

${}^{4}$ https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html

${}^{4}$ https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html

<!-- Footnote -->

---

<!-- Media -->

Table 8. Multivariate long sequence time-series forecasting results on ETT full benchmark. The best results are highlighted in bold.

表8. ETT完整基准测试上的多变量长序列时间序列预测结果。最佳结果以粗体突出显示。

<table><tr><td colspan="2">Methods</td><td colspan="2">FEDformer-f</td><td colspan="2">FEDformer-w</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td><td colspan="2">LogTrans</td><td colspan="2">Reformer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.376</td><td>0.419</td><td>0.395</td><td>0.424</td><td>0.449</td><td>0.459</td><td>0.865</td><td>0.713</td><td>0.878</td><td>0.740</td><td>0.837</td><td>0.728</td></tr><tr><td>192</td><td>0.420</td><td>0.448</td><td>0.469</td><td>0.470</td><td>0.500</td><td>0.482</td><td>1.008</td><td>0.792</td><td>1.037</td><td>0.824</td><td>0.923</td><td>0.766</td></tr><tr><td>336</td><td>0.459</td><td>0.465</td><td>0.530</td><td>0.499</td><td>0.521</td><td>0.496</td><td>1.107</td><td>0.809</td><td>1.238</td><td>0.932</td><td>1.097</td><td>0.835</td></tr><tr><td>720</td><td>0.506</td><td>0.507</td><td>0.598</td><td>0.544</td><td>0.514</td><td>0.512</td><td>1.181</td><td>0.865</td><td>1.135</td><td>0.852</td><td>1.257</td><td>0.889</td></tr><tr><td rowspan="4">${ETTh2}$</td><td>96</td><td>0.346</td><td>0.388</td><td>0.394</td><td>0.414</td><td>0.358</td><td>0.397</td><td>3.755</td><td>1.525</td><td>2.116</td><td>1.197</td><td>2.626</td><td>1.317</td></tr><tr><td>192</td><td>0.429</td><td>0.439</td><td>0.439</td><td>0.445</td><td>0.456</td><td>0.452</td><td>5.602</td><td>1.931</td><td>4.315</td><td>1.635</td><td>11.12</td><td>2.979</td></tr><tr><td>336</td><td>0.496</td><td>0.487</td><td>0.482</td><td>0.480</td><td>0.482</td><td>0.486</td><td>4.721</td><td>1.835</td><td>1.124</td><td>1.604</td><td>9.323</td><td>2.769</td></tr><tr><td>720</td><td>0.463</td><td>0474</td><td>0.500</td><td>0.509</td><td>0.515</td><td>0.511</td><td>3.647</td><td>1.625</td><td>3.188</td><td>1.540</td><td>3.874</td><td>1.697</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.379</td><td>0.419</td><td>0.378</td><td>0.418</td><td>0.505</td><td>0.475</td><td>0.672</td><td>0.571</td><td>0.600</td><td>0.546</td><td>0.538</td><td>0.528</td></tr><tr><td>192</td><td>0.426</td><td>0.441</td><td>0.464</td><td>0.463</td><td>0.553</td><td>0.496</td><td>0.795</td><td>0.669</td><td>0.837</td><td>0.700</td><td>0.658</td><td>0.592</td></tr><tr><td>336</td><td>0.445</td><td>0.459</td><td>0.508</td><td>0.487</td><td>0.621</td><td>0.537</td><td>1.212</td><td>0.871</td><td>1.124</td><td>0.832</td><td>0.898</td><td>0.721</td></tr><tr><td>720</td><td>0.543</td><td>0.490</td><td>0.561</td><td>0.515</td><td>0.671</td><td>0.561</td><td>1.166</td><td>0.823</td><td>1.153</td><td>0.820</td><td>1.102</td><td>0.841</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.203</td><td>0.287</td><td>0.204</td><td>0.288</td><td>0.255</td><td>0.339</td><td>0.365</td><td>0.453</td><td>0.768</td><td>0.642</td><td>0.658</td><td>0.619</td></tr><tr><td>192</td><td>0.269</td><td>0.328</td><td>0.316</td><td>0.363</td><td>0.281</td><td>0.340</td><td>0.533</td><td>0.563</td><td>0.989</td><td>0.757</td><td>1.078</td><td>0.827</td></tr><tr><td>336</td><td>0.325</td><td>0.366</td><td>0.359</td><td>0.387</td><td>0.339</td><td>0.372</td><td>1.363</td><td>0.887</td><td>1.334</td><td>0.872</td><td>1.549</td><td>0.972</td></tr><tr><td>720</td><td>0.421</td><td>0.415</td><td>0.433</td><td>0.432</td><td>0.422</td><td>0.419</td><td>3.379</td><td>1.338</td><td>3.048</td><td>1.328</td><td>2.631</td><td>1.242</td></tr></table>

<table><tbody><tr><td colspan="2">方法</td><td colspan="2">FEDformer - f</td><td colspan="2">FEDformer - w</td><td colspan="2">自动转换器（Autoformer）</td><td colspan="2">信息器（Informer）</td><td colspan="2">对数转换器（LogTrans）</td><td colspan="2">改革器（Reformer）</td></tr><tr><td colspan="2">指标（Metric）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.376</td><td>0.419</td><td>0.395</td><td>0.424</td><td>0.449</td><td>0.459</td><td>0.865</td><td>0.713</td><td>0.878</td><td>0.740</td><td>0.837</td><td>0.728</td></tr><tr><td>192</td><td>0.420</td><td>0.448</td><td>0.469</td><td>0.470</td><td>0.500</td><td>0.482</td><td>1.008</td><td>0.792</td><td>1.037</td><td>0.824</td><td>0.923</td><td>0.766</td></tr><tr><td>336</td><td>0.459</td><td>0.465</td><td>0.530</td><td>0.499</td><td>0.521</td><td>0.496</td><td>1.107</td><td>0.809</td><td>1.238</td><td>0.932</td><td>1.097</td><td>0.835</td></tr><tr><td>720</td><td>0.506</td><td>0.507</td><td>0.598</td><td>0.544</td><td>0.514</td><td>0.512</td><td>1.181</td><td>0.865</td><td>1.135</td><td>0.852</td><td>1.257</td><td>0.889</td></tr><tr><td rowspan="4">${ETTh2}$</td><td>96</td><td>0.346</td><td>0.388</td><td>0.394</td><td>0.414</td><td>0.358</td><td>0.397</td><td>3.755</td><td>1.525</td><td>2.116</td><td>1.197</td><td>2.626</td><td>1.317</td></tr><tr><td>192</td><td>0.429</td><td>0.439</td><td>0.439</td><td>0.445</td><td>0.456</td><td>0.452</td><td>5.602</td><td>1.931</td><td>4.315</td><td>1.635</td><td>11.12</td><td>2.979</td></tr><tr><td>336</td><td>0.496</td><td>0.487</td><td>0.482</td><td>0.480</td><td>0.482</td><td>0.486</td><td>4.721</td><td>1.835</td><td>1.124</td><td>1.604</td><td>9.323</td><td>2.769</td></tr><tr><td>720</td><td>0.463</td><td>0474</td><td>0.500</td><td>0.509</td><td>0.515</td><td>0.511</td><td>3.647</td><td>1.625</td><td>3.188</td><td>1.540</td><td>3.874</td><td>1.697</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.379</td><td>0.419</td><td>0.378</td><td>0.418</td><td>0.505</td><td>0.475</td><td>0.672</td><td>0.571</td><td>0.600</td><td>0.546</td><td>0.538</td><td>0.528</td></tr><tr><td>192</td><td>0.426</td><td>0.441</td><td>0.464</td><td>0.463</td><td>0.553</td><td>0.496</td><td>0.795</td><td>0.669</td><td>0.837</td><td>0.700</td><td>0.658</td><td>0.592</td></tr><tr><td>336</td><td>0.445</td><td>0.459</td><td>0.508</td><td>0.487</td><td>0.621</td><td>0.537</td><td>1.212</td><td>0.871</td><td>1.124</td><td>0.832</td><td>0.898</td><td>0.721</td></tr><tr><td>720</td><td>0.543</td><td>0.490</td><td>0.561</td><td>0.515</td><td>0.671</td><td>0.561</td><td>1.166</td><td>0.823</td><td>1.153</td><td>0.820</td><td>1.102</td><td>0.841</td></tr><tr><td rowspan="4">ETTm2（原文未变，因可能为特定专业术语暂无通用中文译法）</td><td>96</td><td>0.203</td><td>0.287</td><td>0.204</td><td>0.288</td><td>0.255</td><td>0.339</td><td>0.365</td><td>0.453</td><td>0.768</td><td>0.642</td><td>0.658</td><td>0.619</td></tr><tr><td>192</td><td>0.269</td><td>0.328</td><td>0.316</td><td>0.363</td><td>0.281</td><td>0.340</td><td>0.533</td><td>0.563</td><td>0.989</td><td>0.757</td><td>1.078</td><td>0.827</td></tr><tr><td>336</td><td>0.325</td><td>0.366</td><td>0.359</td><td>0.387</td><td>0.339</td><td>0.372</td><td>1.363</td><td>0.887</td><td>1.334</td><td>0.872</td><td>1.549</td><td>0.972</td></tr><tr><td>720</td><td>0.421</td><td>0.415</td><td>0.433</td><td>0.432</td><td>0.422</td><td>0.419</td><td>3.379</td><td>1.338</td><td>3.048</td><td>1.328</td><td>2.631</td><td>1.242</td></tr></tbody></table>

<!-- Media -->

### F.4. Cross Attention Visualization

### F.4. 交叉注意力可视化

The $\sigma \left( {\widetilde{\mathbf{Q}} \cdot  {\widetilde{\mathbf{K}}}^{\top }}\right)$ can be viewed as the cross attention weight for our proposed frequency enhanced cross attention block. Several different activation functions can be used for attention matrix activation. Tanh and softmax are tested in this work with various performances on different datasets. We use tanh as the default one. Different attention patterns are visualized in Figure 8. Here two samples of cross attention maps are shown for FEDformer-f training on the ETTm2 dataset using tanh and softmax respectively. It can be seen that attention with Softmax as activation function seems to be more sparse than using tanh. Overall we can see attention in the frequency domain is much sparser compared to the normal attention graph in the time domain, which indicates our proposed attention can represent the signal more compactly. Also this compact representation supports our random mode selection mechanism to achieve linear complexity.

$\sigma \left( {\widetilde{\mathbf{Q}} \cdot  {\widetilde{\mathbf{K}}}^{\top }}\right)$ 可被视为我们提出的频率增强交叉注意力模块的交叉注意力权重。可以使用几种不同的激活函数来进行注意力矩阵激活。在这项工作中测试了双曲正切函数（tanh）和 Softmax 函数，它们在不同数据集上表现各异。我们默认使用双曲正切函数。不同的注意力模式如图 8 所示。这里分别展示了 FEDformer - f 在 ETTm2 数据集上训练时使用双曲正切函数和 Softmax 函数的两个交叉注意力图样本。可以看出，以 Softmax 函数作为激活函数的注意力似乎比使用双曲正切函数时更稀疏。总体而言，与时间域中的普通注意力图相比，频域中的注意力要稀疏得多，这表明我们提出的注意力机制可以更紧凑地表示信号。此外，这种紧凑的表示支持我们的随机模式选择机制以实现线性复杂度。

### F.5. Improvements of Mixture of Experts Decomposition

### F.5. 专家混合分解的改进

We design a mixture of experts decomposition mechanism which adopts a set of average pooling layers to extract the trend and a set of data-dependent weights to combine them.

我们设计了一种专家混合分解机制，该机制采用一组平均池化层来提取趋势，并使用一组依赖于数据的权重将它们组合起来。

<!-- Media -->

<!-- figureText: 0.15 - 0.05 - 0.75 - 0.75 -->

<img src="https://cdn.noedgeai.com/01957f69-db4d-7f34-af19-bfc9e93e263b_16.jpg?x=942&y=955&w=638&h=866&r=0"/>

Figure 8. Multihead attention map with 8 heads using tanh (top) and softmax (bottom) as activation map for the FEDformer-f training on ETTm2 dataset.

图8. 在ETTm2数据集上对FEDformer - f进行训练时，使用tanh（上）和softmax（下）作为激活映射的8头多头注意力图。

<!-- Media -->

The default average pooling layers contain filters with kernel size7,12,14,24and 48 respectively. For comparison, we use single expert decomposition mechanism which employs a single average pooling layer with a fixed kernel size of 24 as the baseline. In Table 10, a comparison study of multivariate forecasting is shown using FEDformer-f model on two typical datasets. It is observed that the designed mixture of experts decomposition brings better performance than the single decomposition scheme.

默认的平均池化层分别包含核大小为7、12、14、24和48的滤波器。为了进行比较，我们使用单一专家分解机制作为基线，该机制采用核大小固定为24的单个平均池化层。在表10中，展示了使用FEDformer - f模型在两个典型数据集上进行多变量预测的对比研究。可以观察到，所设计的专家混合分解比单一分解方案带来了更好的性能。

<!-- Media -->

Table 9. Univariate long sequence time-series forecasting results on ETT full benchmark. The best results are highlighted in bold.

表9. ETT完整基准上的单变量长序列时间序列预测结果。最佳结果以粗体突出显示。

<table><tr><td colspan="2">Methods</td><td colspan="2">FEDformer-f</td><td colspan="2">FEDformer-w</td><td colspan="2">Autoformer</td><td colspan="2">Informer</td><td colspan="2">LogTrans</td><td colspan="2">Reformer</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.079</td><td>0.215</td><td>0.080</td><td>0.214</td><td>0.071</td><td>0.206</td><td>0.193</td><td>0.377</td><td>0.283</td><td>0.468</td><td>0.532</td><td>0.569</td></tr><tr><td>192</td><td>0.104</td><td>0.245</td><td>0.105</td><td>0.256</td><td>0.114</td><td>0.262</td><td>0.217</td><td>0.395</td><td>0.234</td><td>0.409</td><td>0.568</td><td>0.575</td></tr><tr><td>336</td><td>0.119</td><td>0.270</td><td>0.120</td><td>0.269</td><td>0.107</td><td>0.258</td><td>0.202</td><td>0.381</td><td>0.386</td><td>0.546</td><td>0.635</td><td>0.589</td></tr><tr><td>720</td><td>0.142</td><td>0.299</td><td>0.127</td><td>0.280</td><td>0.126</td><td>0.283</td><td>0.183</td><td>0.355</td><td>0.475</td><td>0.628</td><td>0.762</td><td>0.666</td></tr><tr><td rowspan="4">${ETTh2}$</td><td>96</td><td>0.128</td><td>0.271</td><td>0.156</td><td>0.306</td><td>0.153</td><td>0.306</td><td>0.213</td><td>0.373</td><td>0.217</td><td>0.379</td><td>1.411</td><td>0.838</td></tr><tr><td>192</td><td>0.185</td><td>0.330</td><td>0.238</td><td>0.380</td><td>0.204</td><td>0.351</td><td>0.227</td><td>0.387</td><td>0.281</td><td>0.429</td><td>5.658</td><td>1.671</td></tr><tr><td>336</td><td>0.231</td><td>0.378</td><td>0.271</td><td>0.412</td><td>0.246</td><td>0.389</td><td>0.242</td><td>0.401</td><td>0.293</td><td>0.437</td><td>4.777</td><td>1.582</td></tr><tr><td>720</td><td>0.278</td><td>0.420</td><td>0.288</td><td>0.438</td><td>0.268</td><td>0.409</td><td>0.291</td><td>0.439</td><td>0.218</td><td>0.387</td><td>2.042</td><td>1.039</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.033</td><td>0.140</td><td>0.036</td><td>0.149</td><td>0.056</td><td>0.183</td><td>0.109</td><td>0.277</td><td>0.049</td><td>0.171</td><td>0.296</td><td>0.355</td></tr><tr><td>192</td><td>0.058</td><td>0.186</td><td>0.069</td><td>0.206</td><td>0.081</td><td>0.216</td><td>0.151</td><td>0.310</td><td>0.157</td><td>0.317</td><td>0.429</td><td>0.474</td></tr><tr><td>336</td><td>0.084</td><td>0.231</td><td>0.071</td><td>0.209</td><td>0.076</td><td>0.218</td><td>0.427</td><td>0.591</td><td>0.289</td><td>0.459</td><td>0.585</td><td>0.583</td></tr><tr><td>720</td><td>0.102</td><td>0.250</td><td>0.105</td><td>0.248</td><td>0.110</td><td>0.267</td><td>0.438</td><td>0.586</td><td>0.430</td><td>0.579</td><td>0.782</td><td>0.730</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.067</td><td>0.198</td><td>0.063</td><td>0.189</td><td>0.065</td><td>0.189</td><td>0.088</td><td>0.225</td><td>0.075</td><td>0.208</td><td>0.076</td><td>0.214</td></tr><tr><td>192</td><td>0.102</td><td>0.245</td><td>0.110</td><td>0.252</td><td>0.118</td><td>0.256</td><td>0.132</td><td>0.283</td><td>0.129</td><td>0.275</td><td>0.132</td><td>0.290</td></tr><tr><td>336</td><td>0.130</td><td>0.279</td><td>0.147</td><td>0.301</td><td>0.154</td><td>0.305</td><td>0.180</td><td>0.336</td><td>0.154</td><td>0.302</td><td>0.160</td><td>0.312</td></tr><tr><td>720</td><td>0.178</td><td>0.325</td><td>0.219</td><td>0.368</td><td>0.182</td><td>0.335</td><td>0.300</td><td>0.435</td><td>0.160</td><td>0.321</td><td>0.168</td><td>0.335</td></tr></table>

<table><tbody><tr><td colspan="2">方法</td><td colspan="2">FEDformer - f</td><td colspan="2">FEDformer - w</td><td colspan="2">自动转换器（Autoformer）</td><td colspan="2">信息器（Informer）</td><td colspan="2">对数转换器（LogTrans）</td><td colspan="2">改革器（Reformer）</td></tr><tr><td colspan="2">指标（Metric）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="4">ETTh1</td><td>96</td><td>0.079</td><td>0.215</td><td>0.080</td><td>0.214</td><td>0.071</td><td>0.206</td><td>0.193</td><td>0.377</td><td>0.283</td><td>0.468</td><td>0.532</td><td>0.569</td></tr><tr><td>192</td><td>0.104</td><td>0.245</td><td>0.105</td><td>0.256</td><td>0.114</td><td>0.262</td><td>0.217</td><td>0.395</td><td>0.234</td><td>0.409</td><td>0.568</td><td>0.575</td></tr><tr><td>336</td><td>0.119</td><td>0.270</td><td>0.120</td><td>0.269</td><td>0.107</td><td>0.258</td><td>0.202</td><td>0.381</td><td>0.386</td><td>0.546</td><td>0.635</td><td>0.589</td></tr><tr><td>720</td><td>0.142</td><td>0.299</td><td>0.127</td><td>0.280</td><td>0.126</td><td>0.283</td><td>0.183</td><td>0.355</td><td>0.475</td><td>0.628</td><td>0.762</td><td>0.666</td></tr><tr><td rowspan="4">${ETTh2}$</td><td>96</td><td>0.128</td><td>0.271</td><td>0.156</td><td>0.306</td><td>0.153</td><td>0.306</td><td>0.213</td><td>0.373</td><td>0.217</td><td>0.379</td><td>1.411</td><td>0.838</td></tr><tr><td>192</td><td>0.185</td><td>0.330</td><td>0.238</td><td>0.380</td><td>0.204</td><td>0.351</td><td>0.227</td><td>0.387</td><td>0.281</td><td>0.429</td><td>5.658</td><td>1.671</td></tr><tr><td>336</td><td>0.231</td><td>0.378</td><td>0.271</td><td>0.412</td><td>0.246</td><td>0.389</td><td>0.242</td><td>0.401</td><td>0.293</td><td>0.437</td><td>4.777</td><td>1.582</td></tr><tr><td>720</td><td>0.278</td><td>0.420</td><td>0.288</td><td>0.438</td><td>0.268</td><td>0.409</td><td>0.291</td><td>0.439</td><td>0.218</td><td>0.387</td><td>2.042</td><td>1.039</td></tr><tr><td rowspan="4">ETTm1</td><td>96</td><td>0.033</td><td>0.140</td><td>0.036</td><td>0.149</td><td>0.056</td><td>0.183</td><td>0.109</td><td>0.277</td><td>0.049</td><td>0.171</td><td>0.296</td><td>0.355</td></tr><tr><td>192</td><td>0.058</td><td>0.186</td><td>0.069</td><td>0.206</td><td>0.081</td><td>0.216</td><td>0.151</td><td>0.310</td><td>0.157</td><td>0.317</td><td>0.429</td><td>0.474</td></tr><tr><td>336</td><td>0.084</td><td>0.231</td><td>0.071</td><td>0.209</td><td>0.076</td><td>0.218</td><td>0.427</td><td>0.591</td><td>0.289</td><td>0.459</td><td>0.585</td><td>0.583</td></tr><tr><td>720</td><td>0.102</td><td>0.250</td><td>0.105</td><td>0.248</td><td>0.110</td><td>0.267</td><td>0.438</td><td>0.586</td><td>0.430</td><td>0.579</td><td>0.782</td><td>0.730</td></tr><tr><td rowspan="4">ETTm2（原文未变，因可能为特定专业术语暂无通用中文译法）</td><td>96</td><td>0.067</td><td>0.198</td><td>0.063</td><td>0.189</td><td>0.065</td><td>0.189</td><td>0.088</td><td>0.225</td><td>0.075</td><td>0.208</td><td>0.076</td><td>0.214</td></tr><tr><td>192</td><td>0.102</td><td>0.245</td><td>0.110</td><td>0.252</td><td>0.118</td><td>0.256</td><td>0.132</td><td>0.283</td><td>0.129</td><td>0.275</td><td>0.132</td><td>0.290</td></tr><tr><td>336</td><td>0.130</td><td>0.279</td><td>0.147</td><td>0.301</td><td>0.154</td><td>0.305</td><td>0.180</td><td>0.336</td><td>0.154</td><td>0.302</td><td>0.160</td><td>0.312</td></tr><tr><td>720</td><td>0.178</td><td>0.325</td><td>0.219</td><td>0.368</td><td>0.182</td><td>0.335</td><td>0.300</td><td>0.435</td><td>0.160</td><td>0.321</td><td>0.168</td><td>0.335</td></tr></tbody></table>

Table 10. Performance improvement of the designed mixture of experts decomposition scheme.

表10. 所设计的专家混合分解方案的性能提升情况。

<table><tr><td>Methods</td><td colspan="2">FEDformer-f</td><td colspan="2">FEDformer-f</td></tr><tr><td>Dataset</td><td colspan="2">ETTh1</td><td colspan="2">Weather</td></tr><tr><td>Mechanism</td><td>MOE</td><td>Single</td><td>MOE</td><td>Single</td></tr><tr><td>96</td><td>0.217</td><td>0.238</td><td>0.376</td><td>0.375</td></tr><tr><td>192</td><td>0.276</td><td>0.291</td><td>0.420</td><td>0.412</td></tr><tr><td>336</td><td>0.339</td><td>0.352</td><td>0.450</td><td>0.455</td></tr><tr><td>720</td><td>0.403</td><td>0.413</td><td>0.496</td><td>0.502</td></tr><tr><td>Improvement</td><td colspan="2">5.35%</td><td colspan="2">0.57%</td></tr></table>

<table><tbody><tr><td>方法</td><td colspan="2">FEDformer - f</td><td colspan="2">FEDformer - f</td></tr><tr><td>数据集</td><td colspan="2">ETTh1</td><td colspan="2">天气</td></tr><tr><td>机制</td><td>教育部（Ministry of Education）</td><td>单身；单个</td><td>教育部（Ministry of Education）</td><td>单身；单个</td></tr><tr><td>96</td><td>0.217</td><td>0.238</td><td>0.376</td><td>0.375</td></tr><tr><td>192</td><td>0.276</td><td>0.291</td><td>0.420</td><td>0.412</td></tr><tr><td>336</td><td>0.339</td><td>0.352</td><td>0.450</td><td>0.455</td></tr><tr><td>720</td><td>0.403</td><td>0.413</td><td>0.496</td><td>0.502</td></tr><tr><td>改进；改善</td><td colspan="2">5.35%</td><td colspan="2">0.57%</td></tr></tbody></table>

<!-- Media -->

### F.6. Multiple random runs

### F.6. 多次随机运行

Table 11 lists both mean and standard deviation (STD) for FEDformer-f and Autoformer with 5 runs. We observe a small variance in the performance of FEDformer-f, despite the randomness in frequency selection.

表11列出了FEDformer - f和Autoformer在5次运行中的均值和标准差（STD）。我们发现，尽管在频率选择上存在随机性，但FEDformer - f的性能波动较小。

### F.7. Sensitivity to the number of modes: ETTx1 vs ETTx2

### F.7. 对模态数量的敏感性：ETTx1与ETTx2

The choice of modes number depends on data complexity. The time series that exhibits the higher complex patterns requires the larger the number of modes. To verify this claim, we summarize the complexity of ETT datasets, measured by permutation entropy and SVD entropy, in Table 12. It is observed that ETTx1 has a significantly higher complexity (corresponding to a higher entropy value) than ETTx2, thus requiring a larger number of modes.

模态数量的选择取决于数据的复杂程度。呈现出更复杂模式的时间序列需要更多的模态数量。为了验证这一说法，我们在表12中总结了通过排列熵和奇异值分解（SVD）熵衡量的ETT数据集的复杂度。可以观察到，ETTx1的复杂度（对应于更高的熵值）明显高于ETTx2，因此需要更多的模态数量。

<!-- Media -->

Table 11. A subset of the benchmark showing both Mean and STD.

表11. 展示均值和标准差的基准测试子集。

<table><tr><td colspan="2">MSE</td><td>ETTm2</td><td>Electricity</td><td>Exchange</td><td>Traffic</td></tr><tr><td rowspan="4">FED-f</td><td>96</td><td>${0.203} \pm  {0.0042}$</td><td>${0.194} \pm  {0.0008}$</td><td>${0.148} \pm  {0.002}$</td><td>${0.217} \pm  {0.008}$</td></tr><tr><td>192</td><td>${0.269} \pm  {0.0023}$</td><td>${0.201} \pm  {0.0015}$</td><td>${0.270} \pm  {0.008}$</td><td>${0.604} \pm  {0.004}$</td></tr><tr><td>336</td><td>${0.325} \pm  {0.0015}$</td><td>${0.215} \pm  {0.0018}$</td><td>${0.460} \pm  {0.016}$</td><td>${0.621} \pm  {0.006}$</td></tr><tr><td>720</td><td>${0.421} \pm  {0.0038}$</td><td>${0.246} \pm  {0.0020}$</td><td>${1.195} \pm  {0.026}$</td><td>${0.626} \pm  {0.003}$</td></tr><tr><td rowspan="4">Autoformer</td><td>96</td><td>${0.255} \pm  {0.020}$</td><td>${0.201} \pm  {0.003}$</td><td>${0.197} \pm  {0.019}$</td><td>${0.613} \pm  {0.028}$</td></tr><tr><td>192</td><td>${0.281} \pm  {0.027}$</td><td>${0.222} \pm  {0.003}$</td><td>${0.300} \pm  {0.020}$</td><td>${0.616} \pm  {0.042}$</td></tr><tr><td>336</td><td>${0.339} \pm  {0.018}$</td><td>${0.231} \pm  {0.006}$</td><td>${0.509} \pm  {0.041}$</td><td>${0.622} \pm  {0.016}$</td></tr><tr><td>720</td><td>${0.422} \pm  {0.015}$</td><td>${0.254} \pm  {0.007}$</td><td>${1.447} \pm  {0.084}$</td><td>${0.419} \pm  {0.017}$</td></tr></table>

<table><tbody><tr><td colspan="2">均方误差（MSE）</td><td>增强型变压器时间序列模型2（ETTm2）</td><td>电力</td><td>交换；交易</td><td>交通</td></tr><tr><td rowspan="4">联邦学习框架（FED - f）</td><td>96</td><td>${0.203} \pm  {0.0042}$</td><td>${0.194} \pm  {0.0008}$</td><td>${0.148} \pm  {0.002}$</td><td>${0.217} \pm  {0.008}$</td></tr><tr><td>192</td><td>${0.269} \pm  {0.0023}$</td><td>${0.201} \pm  {0.0015}$</td><td>${0.270} \pm  {0.008}$</td><td>${0.604} \pm  {0.004}$</td></tr><tr><td>336</td><td>${0.325} \pm  {0.0015}$</td><td>${0.215} \pm  {0.0018}$</td><td>${0.460} \pm  {0.016}$</td><td>${0.621} \pm  {0.006}$</td></tr><tr><td>720</td><td>${0.421} \pm  {0.0038}$</td><td>${0.246} \pm  {0.0020}$</td><td>${1.195} \pm  {0.026}$</td><td>${0.626} \pm  {0.003}$</td></tr><tr><td rowspan="4">自动变换器（Autoformer）</td><td>96</td><td>${0.255} \pm  {0.020}$</td><td>${0.201} \pm  {0.003}$</td><td>${0.197} \pm  {0.019}$</td><td>${0.613} \pm  {0.028}$</td></tr><tr><td>192</td><td>${0.281} \pm  {0.027}$</td><td>${0.222} \pm  {0.003}$</td><td>${0.300} \pm  {0.020}$</td><td>${0.616} \pm  {0.042}$</td></tr><tr><td>336</td><td>${0.339} \pm  {0.018}$</td><td>${0.231} \pm  {0.006}$</td><td>${0.509} \pm  {0.041}$</td><td>${0.622} \pm  {0.016}$</td></tr><tr><td>720</td><td>${0.422} \pm  {0.015}$</td><td>${0.254} \pm  {0.007}$</td><td>${1.447} \pm  {0.084}$</td><td>${0.419} \pm  {0.017}$</td></tr></tbody></table>

Table 12. Complexity experiments for datasets

表12. 数据集的复杂度实验

<table><tr><td>Methods</td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td></tr><tr><td>Permutation Entropy</td><td>0.954</td><td>0.866</td><td>0.959</td><td>0.788</td></tr><tr><td>SVD Entropy</td><td>0.807</td><td>0.495</td><td>0.589</td><td>0.361</td></tr></table>

<table><tbody><tr><td>方法</td><td>ETTh1</td><td>ETTh2</td><td>ETTm1</td><td>ETTm2</td></tr><tr><td>排列熵</td><td>0.954</td><td>0.866</td><td>0.959</td><td>0.788</td></tr><tr><td>奇异值分解熵（SVD Entropy）</td><td>0.807</td><td>0.495</td><td>0.589</td><td>0.361</td></tr></tbody></table>

<!-- Media -->

### F.8. When Fourier/Wavelet model performs better

### F.8. 傅里叶/小波模型何时表现更佳

Our high level principle of model deployment is that Fourier-based model is usually better for less complex time series, while wavelet is normally more suitable for complex ones. Specifically, we found that wavelet-based model is more effective on multivariate time series, while Fourier-based one normally achieves better results on univariate time series. As indicated in Table 13, complexity measures on multivariate time series are higher than those on univariate ones.

我们进行模型部署的高级原则是，基于傅里叶变换（Fourier）的模型通常更适用于不太复杂的时间序列，而小波（Wavelet）模型通常更适合复杂的时间序列。具体而言，我们发现基于小波的模型在多变量时间序列上更有效，而基于傅里叶变换的模型通常在单变量时间序列上能取得更好的效果。如表13所示，多变量时间序列的复杂度度量值高于单变量时间序列。

<!-- Media -->

Table 13. Perm Entropy Complexity comparison for multi vs uni

表13. 多变量与单变量的排列熵复杂度比较

<table><tr><td>Permutation Entropy</td><td>Electricity</td><td>Traffic</td><td>Exchange</td><td>Illness</td></tr><tr><td>Multivariate</td><td>0.910</td><td>0.792</td><td>0.961</td><td>0.960</td></tr><tr><td>Univariate</td><td>0.902</td><td>0.790</td><td>0.949</td><td>0.867</td></tr></table>

<table><tbody><tr><td>排列熵（Permutation Entropy）</td><td>电力；电</td><td>交通</td><td>交换；交易</td><td>疾病</td></tr><tr><td>多变量的</td><td>0.910</td><td>0.792</td><td>0.961</td><td>0.960</td></tr><tr><td>单变量（Univariate）</td><td>0.902</td><td>0.790</td><td>0.949</td><td>0.867</td></tr></tbody></table>

<!-- Media -->