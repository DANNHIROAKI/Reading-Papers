# Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

# Autoformer：用于长期序列预测的自相关分解Transformer

Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long (   )

吴海旭（Haixu Wu）、徐杰辉（Jiehui Xu）、王建民（Jianmin Wang）、龙明盛（Mingsheng Long）

School of Software, BNRist, Tsinghua University, China

中国清华大学软件学院，北京信息科学与技术国家研究中心（BNRist）

\{whx20,xjh20\}@mails.tsinghua.edu.cn, \{jimwang,mingsheng\}@tsinghua.edu.cn

\{whx20,xjh20\}@mails.tsinghua.edu.cn, \{jimwang,mingsheng\}@tsinghua.edu.cn

## Abstract

## 摘要

Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning. This paper studies the long-term forecasting problem of time series. Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck. Going beyond Transformers, we design Auto-former as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of deep models. This design empowers Autoformer with progressive decomposition capacities for complex time series. Further, inspired by the stochastic process theory, we design the Auto-Correlation mechanism based on the series periodicity, which conducts the dependencies discovery and representation aggregation at the sub-series level. Auto-Correlation outperforms self-attention in both efficiency and accuracy. In long-term forecasting, Autoformer yields state-of-the-art accuracy,with a ${38}\%$ relative improvement on six benchmarks,covering five practical applications: energy, traffic, economics, weather and disease. Code is available at this repository: https://github.com/thuml/Autoformer

延长预测时间是实际应用中的关键需求，例如极端天气预警和长期能源消耗规划。本文研究时间序列的长期预测问题。先前基于Transformer的模型采用各种自注意力机制来发现长距离依赖关系。然而，长期未来复杂的时间模式阻碍了模型找到可靠的依赖关系。此外，为了提高长序列处理效率，Transformer模型不得不采用逐点自注意力的稀疏版本，这导致了信息利用瓶颈。超越Transformer模型，我们设计了Autoformer，它是一种具有自相关机制的新型分解架构。我们打破了序列分解的预处理常规，将其改造为深度模型的基本内部模块。这种设计使Autoformer具备对复杂时间序列进行渐进式分解的能力。此外，受随机过程理论的启发，我们基于序列周期性设计了自相关机制，该机制在子序列层面进行依赖关系发现和表示聚合。自相关机制在效率和准确性上均优于自注意力机制。在长期预测中，Autoformer取得了最先进的准确性，在六个基准测试中相对提升了${38}\%$，涵盖了能源、交通、经济、天气和疾病这五个实际应用领域。代码可在以下仓库获取：https://github.com/thuml/Autoformer

## 1 Introduction

## 1 引言

Time series forecasting has been widely used in energy consumption, traffic and economics planning, weather and disease propagation forecasting. In these real-world applications, one pressing demand is to extend the forecast time into the far future, which is quite meaningful for the long-term planning and early warning. Thus, in this paper, we study the long-term forecasting problem of time series, characterizing itself by the large length of predicted time series. Recent deep forecasting models 48, 23, 26, 34, 29, 35, 25, 41] have achieved great progress, especially the Transformer-based models. Benefiting from the self-attention mechanism, Transformers obtain great advantage in modeling long-term dependencies for sequential data, which enables more powerful big models [8, 13].

时间序列预测已广泛应用于能源消耗、交通与经济规划、天气和疾病传播预测等领域。在这些实际应用中，一个紧迫的需求是将预测时间延伸到遥远的未来，这对于长期规划和早期预警具有重要意义。因此，在本文中，我们研究时间序列的长期预测问题，其特点是预测的时间序列长度较长。最近的深度预测模型[48, 23, 26, 34, 29, 35, 25, 41]取得了巨大进展，尤其是基于Transformer（变换器）的模型。得益于自注意力机制，Transformer在对序列数据的长期依赖关系进行建模方面具有很大优势，这使得更强大的大模型得以实现[8, 13]。

However, the forecasting task is extremely challenging under the long-term setting. First, it is unreliable to discover the temporal dependencies directly from the long-term time series because the dependencies can be obscured by entangled temporal patterns. Second, canonical Transformers with self-attention mechanisms are computationally prohibitive for long-term forecasting because of the quadratic complexity of sequence length. Previous Transformer-based forecasting models [48, 23, 26] mainly focus on improving self-attention to a sparse version. While performance is significantly improved, these models still utilize the point-wise representation aggregation. Thus, in the process of efficiency improvement, they will sacrifice the information utilization because of the sparse point-wise connections, resulting in a bottleneck for long-term forecasting of time series. To reason about the intricate temporal patterns, we try to take the idea of decomposition, which is a standard method in time series analysis [1, 33]. It can be used to process the complex time series and extract more predictable components. However, under the forecasting context, it can only be used as the pre-processing of past series because the future is unknown [20]. This common usage limits the capabilities of decomposition and overlooks the potential future interactions among decomposed components. Thus, we attempt to go beyond pre-processing usage of decomposition and propose a generic architecture to empower the deep forecasting models with immanent capacity of progressive decomposition. Further, decomposition can ravel out the entangled temporal patterns and highlight the inherent properties of time series [20]. Benefiting from this, we try to take advantage of the series periodicity to renovate the point-wise connection in self-attention. We observe that the sub-series at the same phase position among periods often present similar temporal processes. Thus, we try to construct a series-level connection based on the process similarity derived by series periodicity.

然而，在长期设置下，预测任务极具挑战性。首先，直接从长期时间序列中发现时间依赖关系是不可靠的，因为这些依赖关系可能会被复杂的时间模式所掩盖。其次，具有自注意力机制的经典Transformer模型由于序列长度的二次复杂度，在长期预测中计算成本过高。以往基于Transformer的预测模型[48, 23, 26]主要致力于将自注意力机制改进为稀疏版本。虽然性能有了显著提升，但这些模型仍然采用逐点表示聚合。因此，在提高效率的过程中，由于稀疏的逐点连接，它们会牺牲信息利用率，从而导致时间序列长期预测出现瓶颈。为了推断复杂的时间模式，我们尝试采用分解的思想，这是时间序列分析中的一种标准方法[1, 33]。它可用于处理复杂的时间序列并提取更具可预测性的成分。然而，在预测场景下，由于未来情况未知，它只能用作过去序列的预处理[20]。这种常见的用法限制了分解的能力，并且忽略了分解成分之间潜在的未来交互。因此，我们试图超越分解的预处理用途，提出一种通用架构，使深度预测模型具备渐进分解的内在能力。此外，分解可以梳理复杂的时间模式，突出时间序列的内在特性[20]。受益于此，我们尝试利用序列周期性来革新自注意力中的逐点连接。我们观察到，周期中相同相位位置的子序列通常呈现出相似的时间过程。因此，我们尝试基于序列周期性得出的过程相似性构建序列级连接。

Based on the above motivations, we propose an original Autoformer in place of the Transformers for long-term time series forecasting. Autoformer still follows residual and encoder-decoder structure but renovates Transformer into a decomposition forecasting architecture. By embedding our proposed decomposition blocks as the inner operators, Autoformer can progressively separate the long-term trend information from predicted hidden variables. This design allows our model to alternately decompose and refine the intermediate results during the forecasting procedure. Inspired by the stochastic process theory [9, 30], Autoformer introduces an Auto-Correlation mechanism in place of self-attention, which discovers the sub-series similarity based on the series periodicity and aggregates similar sub-series from underlying periods. This series-wise mechanism achieves $\mathcal{O}\left( {L\log L}\right)$ complexity for length- $L$ series and breaks the information utilization bottleneck by expanding the point-wise representation aggregation to sub-series level. Autoformer achieves the state-of-the-art accuracy on six benchmarks. The contributions are summarized as follows:

基于上述动机，我们提出了一种原创的自动变换器（Autoformer），用于替代变换器（Transformers）进行长期时间序列预测。自动变换器仍然遵循残差和编码器 - 解码器结构，但将变换器改进为一种分解预测架构。通过将我们提出的分解模块嵌入作为内部算子，自动变换器可以逐步从预测的隐藏变量中分离出长期趋势信息。这种设计使我们的模型能够在预测过程中交替分解和细化中间结果。受随机过程理论[9, 30]的启发，自动变换器引入了自相关（Auto - Correlation）机制来替代自注意力机制，该机制基于序列周期性发现子序列的相似性，并从潜在周期中聚合相似的子序列。这种基于序列的机制对于长度为$L$的序列实现了$\mathcal{O}\left( {L\log L}\right)$的复杂度，并通过将逐点表示聚合扩展到子序列级别打破了信息利用瓶颈。自动变换器在六个基准测试中达到了最先进的准确率。主要贡献总结如下：

- To tackle the intricate temporal patterns of the long-term future, we present Autoformer as a decomposition architecture and design the inner decomposition block to empower the deep forecasting model with immanent progressive decomposition capacity.

- 为了处理长期未来复杂的时间模式，我们将自动变换器作为一种分解架构提出，并设计了内部分解模块，以使深度预测模型具备内在的渐进分解能力。

- We propose an Auto-Correlation mechanism with dependencies discovery and information aggregation at the series level. Our mechanism is beyond previous self-attention family and can simultaneously benefit the computation efficiency and information utilization.

- 我们提出了一种自相关机制，该机制可在序列层面进行依赖关系发现和信息聚合。我们的机制超越了以往的自注意力机制家族，能够同时提高计算效率和信息利用率。

- Autoformer achieves a 38% relative improvement under the long-term setting on six benchmarks, covering five real-world applications: energy, traffic, economics, weather and disease.

- Autoformer（自回归变换器）在六个基准测试的长期设置下实现了38%的相对提升，涵盖了五个现实世界的应用领域：能源、交通、经济、气象和疾病。

## 2 Related Work

## 2 相关工作

### 2.1 Models for Time Series Forecasting

### 2.1 时间序列预测模型

Due to the immense importance of time series forecasting, various models have been well developed. Many time series forecasting methods start from the classic tools [38, 10]. ARIMA [7, 6] tackles the forecasting problem by transforming the non-stationary process to stationary through differencing. The filtering method is also introduced for series forecasting [24, 12]. Besides, recurrent neural networks (RNNs) models are used to model the temporal dependencies for time series [42, 32, 47, 28]. DeepAR [34] combines autoregressive methods and RNNs to model the probabilistic distribution of future series. LSTNet [25] introduces convolutional neural networks (CNNs) with recurrent-skip connections to capture the short-term and long-term temporal patterns. Attention-based RNNs [46, 36, 37] introduce the temporal attention to explore the long-range dependencies for prediction. Also, many works based on temporal convolution networks (TCN) [40, 5, 4, 35] attempt to model the temporal causality with the causal convolution. These deep forecasting models mainly focus on the temporal relation modeling by recurrent connections, temporal attention or causal convolution.

由于时间序列预测具有极其重要的意义，各种模型已得到了很好的发展。许多时间序列预测方法都始于经典工具[38, 10]。自回归积分滑动平均模型（ARIMA）[7, 6]通过差分将非平稳过程转换为平稳过程来解决预测问题。滤波方法也被引入到序列预测中[24, 12]。此外，循环神经网络（RNNs）模型被用于对时间序列的时间依赖关系进行建模[42, 32, 47, 28]。深度自回归模型（DeepAR）[34]将自回归方法和循环神经网络相结合，对未来序列的概率分布进行建模。长短期时间序列网络（LSTNet）[25]引入了具有循环跳跃连接的卷积神经网络（CNNs），以捕捉短期和长期的时间模式。基于注意力机制的循环神经网络[46, 36, 37]引入了时间注意力机制，以探索用于预测的长距离依赖关系。此外，许多基于时间卷积网络（TCN）[40, 5, 4, 35]的研究尝试通过因果卷积对时间因果关系进行建模。这些深度预测模型主要通过循环连接、时间注意力机制或因果卷积来关注时间关系建模。

Recently, Transformers [41, 45] based on the self-attention mechanism shows great power in sequential data, such as natural language processing [13, 8], audio processing [19] and even computer vision [16, 27]. However, applying self-attention to long-term time series forecasting is computationally prohibitive because of the quadratic complexity of sequence length $L$ in both memory and time. LogTrans [26] introduces the local convolution to Transformer and proposes the LogSparse attention to select time steps following the exponentially increasing intervals, which reduces the complexity to $\mathcal{O}\left( {L{\left( \log L\right) }^{2}}\right)$ . Reformer [23] presents the local-sensitive hashing (LSH) attention and reduces the complexity to $\mathcal{O}\left( {L\log L}\right)$ . Informer [48] extends Transformer with KL-divergence based ProbSparse attention and also achieves $\mathcal{O}\left( {L\log L}\right)$ complexity. Note that these methods are based on the vanilla Transformer and try to improve the self-attention mechanism to a sparse version, which still follows the point-wise dependency and aggregation. In this paper, our proposed Auto-Correlation mechanism is based on the inherent periodicity of time series and can provide series-wise connections.

最近，基于自注意力机制的Transformer模型 [41, 45] 在序列数据处理方面展现出强大的能力，例如自然语言处理 [13, 8]、音频处理 [19]，甚至计算机视觉 [16, 27]。然而，由于自注意力机制在内存和时间上的复杂度与序列长度 $L$ 呈二次方关系，将其应用于长期时间序列预测时计算成本过高。LogTrans [26] 在Transformer中引入局部卷积，并提出LogSparse注意力机制，按照指数增长的间隔选择时间步，将复杂度降低至 $\mathcal{O}\left( {L{\left( \log L\right) }^{2}}\right)$。Reformer [23] 提出局部敏感哈希（LSH，Local-Sensitive Hashing）注意力机制，将复杂度降低至 $\mathcal{O}\left( {L\log L}\right)$。Informer [48] 通过基于KL散度的ProbSparse注意力机制扩展了Transformer，同样实现了 $\mathcal{O}\left( {L\log L}\right)$ 的复杂度。值得注意的是，这些方法均基于原始的Transformer模型，试图将自注意力机制改进为稀疏版本，但仍然遵循逐点依赖和聚合的方式。在本文中，我们提出的自相关机制基于时间序列的内在周期性，能够提供序列级别的连接。

### 2.2 Decomposition of Time Series

### 2.2 时间序列分解

As a standard method in time series analysis, time series decomposition [1, 33] deconstructs a time series into several components, each representing one of the underlying categories of patterns that are more predictable. It is primarily useful for exploring historical changes over time. For the forecasting tasks, decomposition is always used as the pre-processing of historical series before predicting future series [20, 2], such as Prophet [39] with trend-seasonality decomposition and N-BEATS [29] with basis expansion and DeepGLO [35] with matrix decomposition. However, such pre-processing is limited by the plain decomposition effect of historical series and overlooks the hierarchical interaction between the underlying patterns of series in the long-term future. This paper takes the decomposition idea from a new progressive dimension. Our Autoformer harnesses the decomposition as an inner block of deep models, which can progressively decompose the hidden series throughout the whole forecasting process, including both the past series and the predicted intermediate results.

作为时间序列分析中的一种标准方法，时间序列分解[1, 33]将一个时间序列分解为几个组成部分，每个部分代表一种更具可预测性的潜在模式类别。它主要用于探究随时间的历史变化。对于预测任务，分解通常作为预测未来序列之前对历史序列的预处理步骤[20, 2]，例如采用趋势 - 季节性分解的Prophet（先知模型）[39]、采用基扩展的N - BEATS（神经基扩展分析时间序列）[29]以及采用矩阵分解的DeepGLO（深度全局预测）[35]。然而，这种预处理受限于历史序列的简单分解效果，并且忽略了长期未来中序列潜在模式之间的层次交互。本文从一个新的渐进维度采用分解思想。我们的Autoformer（自动转换器）将分解作为深度模型的一个内部模块，它可以在整个预测过程中逐步分解隐藏序列，包括过去的序列和预测的中间结果。

## 3 Autoformer

## 3 自动转换器

The time series forecasting problem is to predict the most probable length- $O$ series in the future given the past length- $I$ series,denoting as input- $I$ -predict- $O$ . The long-term forecasting setting is to predict the long-term future,i.e. larger $O$ . As aforementioned,we have highlighted the difficulties of long-term series forecasting: handling intricate temporal patterns and breaking the bottleneck of computation efficiency and information utilization. To tackle these two challenges, we introduce the decomposition as a builtin block to the deep forecasting model and propose Autoformer as a decomposition architecture. Besides, we design the Auto-Correlation mechanism to discover the period-based dependencies and aggregate similar sub-series from underlying periods.

时间序列预测问题是在给定过去长度为 $I$ 的序列的情况下，预测未来最可能出现的长度为 $O$ 的序列，记为输入 $I$ 预测 $O$。长期预测设置是对远期未来进行预测，即 $O$ 取值更大。如前所述，我们已经强调了长期序列预测的难点：处理复杂的时间模式，以及突破计算效率和信息利用的瓶颈。为应对这两个挑战，我们将分解模块作为内置组件引入深度预测模型，并提出了Autoformer（自相关分解器）作为一种分解架构。此外，我们设计了自相关机制来发现基于周期的依赖关系，并从潜在周期中聚合相似的子序列。

### 3.1 Decomposition Architecture

### 3.1 分解架构

We renovate Transformer [41] to a deep decomposition architecture (Figure 1), including the inner series decomposition block, Auto-Correlation mechanism, and corresponding Encoder and Decoder.

我们将Transformer（变换器）[41]改进为一种深度分解架构（图1），其中包括内部序列分解模块、自相关机制以及相应的编码器和解码器。

Series decomposition block To learn with the complex temporal patterns in long-term forecasting context, we take the idea of decomposition [1, 33], which can separate the series into trend-cyclical and seasonal parts. These two parts reflect the long-term progression and the seasonality of the series respectively. However, directly decomposing is unrealizable for future series because the future is just unknown. To tackle this dilemma, we present a series decomposition block as an inner operation of Autoformer (Figure 1), which can extract the long-term stationary trend from predicted intermediate hidden variables progressively. Concretely, we adapt the moving average to smooth out periodic fluctuations and highlight the long-term trends. For length- $L$ input series $\mathcal{X} \in  {\mathbb{R}}^{L \times  d}$ ,the process is:

序列分解模块 为了学习长期预测环境中的复杂时间模式，我们采用分解的思想[1, 33]，该思想可以将序列分解为趋势 - 周期部分和季节性部分。这两部分分别反映了序列的长期变化和季节性特征。然而，由于未来情况未知，直接对未来序列进行分解是不可行的。为了解决这一难题，我们提出了一个序列分解模块作为自动变压器（Autoformer）的内部操作（图1），该模块可以逐步从预测的中间隐藏变量中提取长期平稳趋势。具体而言，我们采用移动平均法来消除周期性波动并突出长期趋势。对于长度为 $L$ 的输入序列 $\mathcal{X} \in  {\mathbb{R}}^{L \times  d}$，其过程如下：

$$
{\mathcal{X}}_{\mathrm{t}} = \operatorname{AvgPool}\left( {\operatorname{Padding}\left( \mathcal{X}\right) }\right)  \tag{1}
$$

$$
{\mathcal{X}}_{\mathrm{s}} = \mathcal{X} - {\mathcal{X}}_{\mathrm{t}}
$$

where ${\mathcal{X}}_{\mathrm{s}},{\mathcal{X}}_{\mathrm{t}} \in  {\mathbb{R}}^{L \times  d}$ denote the seasonal and the extracted trend-cyclical part respectively. We adopt the AvgPool(-) for moving average with the padding operation to keep the series length unchanged. We use ${\mathcal{X}}_{\mathrm{s}},{\mathcal{X}}_{\mathrm{t}} = \operatorname{SeriesDecomp}\left( \mathcal{X}\right)$ to summarize above equations,which is a model inner block.

其中 ${\mathcal{X}}_{\mathrm{s}},{\mathcal{X}}_{\mathrm{t}} \in  {\mathbb{R}}^{L \times  d}$ 分别表示季节性部分和提取的趋势 - 周期部分。我们采用平均池化（AvgPool(-)）进行移动平均，并使用填充操作以保持序列长度不变。我们使用 ${\mathcal{X}}_{\mathrm{s}},{\mathcal{X}}_{\mathrm{t}} = \operatorname{SeriesDecomp}\left( \mathcal{X}\right)$ 来总结上述方程，这是一个模型内部模块。

Model inputs The inputs of encoder part are the past $I$ time steps ${\mathcal{X}}_{\text{en }} \in  {\mathbb{R}}^{I \times  d}$ . As a decomposition architecture (Figure 1),the input of Autoformer decoder contains both the seasonal part ${\mathcal{X}}_{\text{des }} \in$ ${\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  d}$ and trend-cyclical part ${\mathcal{X}}_{\text{det }} \in  {\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  d}$ to be refined. Each initialization consists of two parts: the component decomposed from the latter half of encoder’s input ${\mathcal{X}}_{\text{en }}$ with length $\frac{I}{2}$ to provide recent information,placeholders with length $O$ filled by scalars. It’s formulized as follows:

模型输入 编码器部分的输入是过去的 $I$ 个时间步 ${\mathcal{X}}_{\text{en }} \in  {\mathbb{R}}^{I \times  d}$。作为一种分解架构（图 1），Autoformer 解码器的输入包含待细化的季节性部分 ${\mathcal{X}}_{\text{des }} \in$ ${\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  d}$ 和趋势 - 周期性部分 ${\mathcal{X}}_{\text{det }} \in  {\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  d}$。每次初始化都包含两部分：从编码器输入后半部分分解得到的分量 ${\mathcal{X}}_{\text{en }}$，长度为 $\frac{I}{2}$，用于提供近期信息；长度为 $O$ 的占位符，由标量填充。其公式化表示如下：

$$
{\mathcal{X}}_{\text{ens }},{\mathcal{X}}_{\text{ent }} = \operatorname{SeriesDecomp}\left( {\mathcal{X}}_{\text{en }\frac{I}{2} : I}\right) 
$$

$$
{\mathcal{X}}_{\text{des }} = \operatorname{Concat}\left( {{\mathcal{X}}_{\text{ens }},{\mathcal{X}}_{0}}\right)  \tag{2}
$$

$$
{\mathcal{X}}_{\text{det }} = \operatorname{Concat}\left( {{\mathcal{X}}_{\text{ent }},{\mathcal{X}}_{\text{Mean }}}\right) ,
$$

<!-- Media -->

<!-- figureText: Autoformer Encoder Forward Decomp Decomp Forward Decomp M x Encoder Input To Predict Correlation Decomp Seasonal Init Correlation Decomp Trend-cyclical Init Input Data Mean Autoformer Decoder -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_3.jpg?x=308&y=201&w=1183&h=474&r=0"/>

Figure 1: Autoformer architecture. The encoder eliminates the long-term trend-cyclical part by series decomposition blocks (blue blocks) and focuses on seasonal patterns modeling. The decoder accumulates the trend part extracted from hidden variables progressively. The past seasonal information from encoder is utilized by the encoder-decoder Auto-Correlation (center green block in decoder).

图1：Autoformer架构。编码器通过序列分解模块（蓝色模块）消除长期趋势-周期性部分，并专注于季节性模式建模。解码器逐步累积从隐藏变量中提取的趋势部分。编码器 - 解码器自相关（解码器中心的绿色模块）利用了来自编码器的过去季节性信息。

<!-- Media -->

where ${\mathcal{X}}_{\text{ens }},{\mathcal{X}}_{\text{ent }} \in  {\mathbb{R}}^{\frac{I}{2} \times  d}$ denote the seasonal and trend-cyclical parts of ${\mathcal{X}}_{\text{en }}$ respectively,and ${\mathcal{X}}_{0},{\mathcal{X}}_{\text{Mean }} \in  {\mathbb{R}}^{O \times  d}$ denote the placeholders filled with zero and the mean of ${\mathcal{X}}_{\text{en }}$ respectively.

其中${\mathcal{X}}_{\text{ens }},{\mathcal{X}}_{\text{ent }} \in  {\mathbb{R}}^{\frac{I}{2} \times  d}$分别表示${\mathcal{X}}_{\text{en }}$的季节性部分和趋势 - 周期性部分，${\mathcal{X}}_{0},{\mathcal{X}}_{\text{Mean }} \in  {\mathbb{R}}^{O \times  d}$分别表示用零填充的占位符和${\mathcal{X}}_{\text{en }}$的均值。

Encoder As shown in Figure 1, the encoder focuses on the seasonal part modeling. The output of the encoder contains the past seasonal information and will be used as the cross information to help the decoder refine prediction results. Suppose we have $N$ encoder layers. The overall equations for $l$ -th encoder layer are summarized as ${\mathcal{X}}_{\text{en }}^{l} = \operatorname{Encoder}\left( {\mathcal{X}}_{\text{en }}^{l - 1}\right)$ . Details are shown as follows:

编码器 如图1所示，编码器专注于季节性部分的建模。编码器的输出包含过去的季节性信息，并将作为交叉信息，帮助解码器优化预测结果。假设我们有$N$个编码器层。第$l$个编码器层的总体方程总结为${\mathcal{X}}_{\text{en }}^{l} = \operatorname{Encoder}\left( {\mathcal{X}}_{\text{en }}^{l - 1}\right)$。详细信息如下：

$$
{\mathcal{S}}_{\text{en }}^{l,1}, = \text{SeriesDecomp}\left( {\text{Auto-Correlation}\left( {\mathcal{X}}_{\text{en }}^{l - 1}\right)  + {\mathcal{X}}_{\text{en }}^{l - 1}}\right)  \tag{3}
$$

$$
{\mathcal{S}}_{\text{en }}^{l,2}, = \text{ SeriesDecomp }\left( {\operatorname{FeedForward}\left( {\mathcal{S}}_{\text{en }}^{l,1}\right)  + {\mathcal{S}}_{\text{en }}^{l,1}}\right) ,
$$

where "_" is the eliminated trend part. ${\mathcal{X}}_{\text{en }}^{l} = {\mathcal{S}}_{\text{en }}^{l,2},l \in  \{ 1,\cdots ,N\}$ denotes the output of $l$ -th encoder layer and ${\mathcal{X}}_{\mathrm{{en}}}^{0}$ is the embedded ${\mathcal{X}}_{\mathrm{{en}}}.{\mathcal{S}}_{\mathrm{{en}}}^{l,i},i \in  \{ 1,2\}$ represents the seasonal component after the $i$ -th series decomposition block in the $l$ -th layer respectively. We will give detailed description of Auto-Correlation $\left( \cdot \right)$ in the next section,which can seamlessly replace the self-attention.

其中“_”是被消除的趋势部分。${\mathcal{X}}_{\text{en }}^{l} = {\mathcal{S}}_{\text{en }}^{l,2},l \in  \{ 1,\cdots ,N\}$表示第$l$个编码器层的输出，${\mathcal{X}}_{\mathrm{{en}}}^{0}$是嵌入的，${\mathcal{X}}_{\mathrm{{en}}}.{\mathcal{S}}_{\mathrm{{en}}}^{l,i},i \in  \{ 1,2\}$分别表示第$l$层中第$i$个序列分解块之后的季节性分量。我们将在下一节详细描述自相关$\left( \cdot \right)$，它可以无缝替代自注意力机制。

Decoder The decoder contains two parts: the accumulation structure for trend-cyclical components and the stacked Auto-Correlation mechanism for seasonal components (Figure 1). Each decoder layer contains the inner Auto-Correlation and encoder-decoder Auto-Correlation, which can refine the prediction and utilize the past seasonal information respectively. Note that the model extracts the potential trend from the intermediate hidden variables during the decoder, allowing Autoformer to progressively refine the trend prediction and eliminate interference information for period-based dependencies discovery in Auto-Correlation. Suppose there are $M$ decoder layers. With the latent variable ${\mathcal{X}}_{\mathrm{{en}}}^{N}$ from the encoder,the equations of $l$ -th decoder layer can be summarized as ${\mathcal{X}}_{\mathrm{{de}}}^{l} =$ Decoder $\left( {{\mathcal{X}}_{\mathrm{{de}}}^{l - 1},{\mathcal{X}}_{\mathrm{{en}}}^{N}}\right)$ . The decoder can be formalized as follows:

解码器 解码器包含两部分：用于趋势 - 周期性成分的累积结构和用于季节性成分的堆叠自相关机制（图1）。每个解码器层包含内部自相关和编码器 - 解码器自相关，它们分别可以细化预测并利用过去的季节性信息。请注意，模型在解码过程中从中间隐藏变量中提取潜在趋势，使Autoformer能够逐步细化趋势预测，并消除自相关中基于周期的依赖关系发现的干扰信息。假设存在$M$个解码器层。利用来自编码器的潜在变量${\mathcal{X}}_{\mathrm{{en}}}^{N}$，第$l$个解码器层的方程可以总结为${\mathcal{X}}_{\mathrm{{de}}}^{l} =$ 解码器$\left( {{\mathcal{X}}_{\mathrm{{de}}}^{l - 1},{\mathcal{X}}_{\mathrm{{en}}}^{N}}\right)$。解码器可以形式化表示如下：

$$
{\mathcal{S}}_{\mathrm{{de}}}^{l,1},{\mathcal{T}}_{\mathrm{{de}}}^{l,1} = \operatorname{SeriesDecomp}\left( {\operatorname{Auto} - \operatorname{Correlation}\left( {\mathcal{X}}_{\mathrm{{de}}}^{l - 1}\right)  + {\mathcal{X}}_{\mathrm{{de}}}^{l - 1}}\right) 
$$

$$
{\mathcal{S}}_{\mathrm{{de}}}^{l,2},{\mathcal{T}}_{\mathrm{{de}}}^{l,2} = \operatorname{SeriesDecomp}\left( {\operatorname{Auto-Correlation}\left( {{\mathcal{S}}_{\mathrm{{de}}}^{l,1},{\mathcal{X}}_{\mathrm{{en}}}^{N}}\right)  + {\mathcal{S}}_{\mathrm{{de}}}^{l,1}}\right)  \tag{4}
$$

$$
{\mathcal{S}}_{\mathrm{{de}}}^{l,3},{\mathcal{T}}_{\mathrm{{de}}}^{l,3} = \operatorname{SeriesDecomp}\left( {\operatorname{FeedForward}\left( {\mathcal{S}}_{\mathrm{{de}}}^{l,2}\right)  + {\mathcal{S}}_{\mathrm{{de}}}^{l,2}}\right) 
$$

$$
{\mathcal{T}}_{\mathrm{{de}}}^{l} = {\mathcal{T}}_{\mathrm{{de}}}^{l - 1} + {\mathcal{W}}_{l,1} * {\mathcal{T}}_{\mathrm{{de}}}^{l,1} + {\mathcal{W}}_{l,2} * {\mathcal{T}}_{\mathrm{{de}}}^{l,2} + {\mathcal{W}}_{l,3} * {\mathcal{T}}_{\mathrm{{de}}}^{l,3},
$$

where ${\mathcal{X}}_{\mathrm{{de}}}^{l} = {\mathcal{S}}_{\mathrm{{de}}}^{l,3},l \in  \{ 1,\cdots ,M\}$ denotes the output of $l$ -th decoder layer. ${\mathcal{X}}_{\mathrm{{de}}}^{0}$ is embedded from ${\mathcal{X}}_{\text{des }}$ for deep transform and ${\mathcal{T}}_{\text{de }}^{0} = {\mathcal{X}}_{\text{det }}$ is for accumulation. ${\mathcal{S}}_{\text{de }}^{l,i},{\mathcal{T}}_{\text{de }}^{l,i},i \in  \{ 1,2,3\}$ represent the seasonal component and trend-cyclical component after the $i$ -th series decomposition block in the $l$ -th layer respectively. ${\mathcal{W}}_{l,i},i \in  \{ 1,2,3\}$ represents the projector for the $i$ -th extracted trend ${\mathcal{T}}_{\mathrm{{de}}}^{l,i}$ .

其中 ${\mathcal{X}}_{\mathrm{{de}}}^{l} = {\mathcal{S}}_{\mathrm{{de}}}^{l,3},l \in  \{ 1,\cdots ,M\}$ 表示第 $l$ 个解码器层的输出。${\mathcal{X}}_{\mathrm{{de}}}^{0}$ 是从 ${\mathcal{X}}_{\text{des }}$ 嵌入而来用于深度变换，${\mathcal{T}}_{\text{de }}^{0} = {\mathcal{X}}_{\text{det }}$ 用于累加。${\mathcal{S}}_{\text{de }}^{l,i},{\mathcal{T}}_{\text{de }}^{l,i},i \in  \{ 1,2,3\}$ 分别表示第 $l$ 层中第 $i$ 个序列分解块之后的季节性分量和趋势 - 周期性分量。${\mathcal{W}}_{l,i},i \in  \{ 1,2,3\}$ 表示第 $i$ 个提取的趋势 ${\mathcal{T}}_{\mathrm{{de}}}^{l,i}$ 的投影器。

<!-- Media -->

<!-- figureText: $\operatorname{Roll}\left( {\tau }_{\mathrm{k}}\right)$ -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_4.jpg?x=306&y=200&w=1185&h=443&r=0"/>

Figure 2: Auto-Correlation (left) and Time Delay Aggregation (right). We utilize the Fast Fourier Transform to calculate the autocorrelation $\mathcal{R}\left( \tau \right)$ ,which reflects the time-delay similarities. Then the similar sub-processes are rolled to the same index based on selected delay $\tau$ and aggregated by $\mathcal{R}\left( \tau \right)$ .

图2：自相关（左）和时间延迟聚合（右）。我们利用快速傅里叶变换（Fast Fourier Transform）来计算自相关 $\mathcal{R}\left( \tau \right)$，它反映了时间延迟的相似性。然后，基于选定的延迟 $\tau$，将相似的子过程滚动到相同的索引，并通过 $\mathcal{R}\left( \tau \right)$ 进行聚合。

<!-- Media -->

The final prediction is the sum of the two refined decomposed components,as ${\mathcal{W}}_{\mathcal{S}} * {\mathcal{X}}_{\mathrm{{de}}}^{M} + {\mathcal{T}}_{\mathrm{{de}}}^{M}$ , where ${\mathcal{W}}_{\mathcal{S}}$ is to project the deep transformed seasonal component ${\mathcal{X}}_{\mathrm{{de}}}^{M}$ to the target dimension.

最终预测结果是两个精炼分解分量的总和，如 ${\mathcal{W}}_{\mathcal{S}} * {\mathcal{X}}_{\mathrm{{de}}}^{M} + {\mathcal{T}}_{\mathrm{{de}}}^{M}$ 所示，其中 ${\mathcal{W}}_{\mathcal{S}}$ 用于将深度变换后的季节性分量 ${\mathcal{X}}_{\mathrm{{de}}}^{M}$ 投影到目标维度。

### 3.2 Auto-Correlation Mechanism

### 3.2 自相关机制

As shown in Figure 2, we propose the Auto-Correlation mechanism with series-wise connections to expand the information utilization. Auto-Correlation discovers the period-based dependencies by calculating the series autocorrelation and aggregates similar sub-series by time delay aggregation.

如图2所示，我们提出了具有序列连接的自相关机制，以扩大信息利用率。自相关通过计算序列自相关来发现基于周期的依赖关系，并通过时间延迟聚合来聚合相似的子序列。

Period-based dependencies It is observed that the same phase position among periods naturally provides similar sub-processes.

基于周期的依赖关系 可以观察到，周期之间相同的相位位置自然会提供相似的子过程。

Inspired by the stochastic process theory [9,30],for a real discrete-time process $\left\{  {\mathcal{X}}_{t}\right\}$ ,we can obtain the autocorrelation ${\mathcal{R}}_{\mathcal{X}\mathcal{X}}\left( \tau \right)$ by the following equations:

受随机过程理论[9,30]的启发，对于一个实际的离散时间过程$\left\{  {\mathcal{X}}_{t}\right\}$，我们可以通过以下方程得到自相关${\mathcal{R}}_{\mathcal{X}\mathcal{X}}\left( \tau \right)$：

$$
{\mathcal{R}}_{\mathcal{X}\mathcal{X}}\left( \tau \right)  = \mathop{\lim }\limits_{{L \rightarrow  \infty }}\frac{1}{L}\mathop{\sum }\limits_{{t = 1}}^{L}{\mathcal{X}}_{t}{\mathcal{X}}_{t - \tau } \tag{5}
$$

${\mathcal{R}}_{\mathcal{X}\mathcal{X}}\left( \tau \right)$ reflects the time-delay similarity between $\left\{  {\mathcal{X}}_{t}\right\}$ and its $\tau$ lag series $\left\{  {\mathcal{X}}_{t - \tau }\right\}$ . As shown in Figure 2,we use the autocorrelation $\mathcal{R}\left( \tau \right)$ as the unnormalized confidence of estimated period length $\tau$ . Then,we choose the most possible $k$ period lengths ${\tau }_{1},\cdots ,{\tau }_{k}$ . The period-based dependencies are derived by the above estimated periods and can be weighted by the corresponding autocorrelation.

${\mathcal{R}}_{\mathcal{X}\mathcal{X}}\left( \tau \right)$反映了$\left\{  {\mathcal{X}}_{t}\right\}$与其$\tau$阶滞后序列$\left\{  {\mathcal{X}}_{t - \tau }\right\}$之间的时延相似性。如图2所示，我们使用自相关系数$\mathcal{R}\left( \tau \right)$作为估计周期长度$\tau$的未归一化置信度。然后，我们选择最可能的$k$个周期长度${\tau }_{1},\cdots ,{\tau }_{k}$。基于周期的依赖关系由上述估计的周期得出，并可以通过相应的自相关系数进行加权。

Time delay aggregation The period-based dependencies connect the sub-series among estimated periods. Thus, we present the time delay aggregation block (Figure 2), which can roll the series based on selected time delay ${\tau }_{1},\cdots ,{\tau }_{k}$ . This operation can align similar sub-series that are at the same phase position of estimated periods, which is different from the point-wise dot-product aggregation in self-attention family. Finally, we aggregate the sub-series by softmax normalized confidences.

时间延迟聚合 基于周期的依赖关系连接了估计周期之间的子序列。因此，我们提出了时间延迟聚合模块（图2），它可以根据选定的时间延迟 ${\tau }_{1},\cdots ,{\tau }_{k}$ 滚动序列。此操作可以对齐处于估计周期相同相位位置的相似子序列，这与自注意力家族中的逐点点积聚合不同。最后，我们通过softmax归一化置信度来聚合子序列。

For the single head situation and time series $\mathcal{X}$ with length- $L$ ,after the projector,we get query $\mathcal{Q}$ ,key $\mathcal{K}$ and value $\mathcal{V}$ . Thus,it can replace self-attention seamlessly. The Auto-Correlation mechanism is:

对于单头情况和长度为 $L$ 的时间序列 $\mathcal{X}$，经过投影器后，我们得到查询向量 $\mathcal{Q}$、键向量 $\mathcal{K}$ 和值向量 $\mathcal{V}$。因此，它可以无缝替代自注意力机制。自相关机制如下：

$$
{\tau }_{1},\cdots ,{\tau }_{k} = \underset{\tau  \in  \{ 1,\cdots ,L\} }{\arg \operatorname{Topk}}\left( {{\mathcal{R}}_{\mathcal{Q},\mathcal{K}}\left( \tau \right) }\right) 
$$

$$
{\widehat{\mathcal{R}}}_{\mathcal{Q},\mathcal{K}}\left( {\tau }_{1}\right) ,\cdots ,{\widehat{\mathcal{R}}}_{\mathcal{Q},\mathcal{K}}\left( {\tau }_{k}\right)  = \operatorname{SoftMax}\left( {{\mathcal{R}}_{\mathcal{Q},\mathcal{K}}\left( {\tau }_{1}\right) ,\cdots ,{\mathcal{R}}_{\mathcal{Q},\mathcal{K}}\left( {\tau }_{k}\right) }\right)  \tag{6}
$$

$$
\text{Auto-Correlation}\left( {\mathcal{Q},\mathcal{K},\mathcal{V}}\right)  = \mathop{\sum }\limits_{{i = 1}}^{k}\operatorname{Roll}\left( {\mathcal{V},{\tau }_{i}}\right) {\widehat{\mathcal{R}}}_{\mathcal{Q},\mathcal{K}}\left( {\tau }_{i}\right) \text{,}
$$

where arg $\operatorname{Topk}\left( \cdot \right)$ is to get the arguments of the Topk autocorrelations and let $k = \lfloor c \times  \log L\rfloor ,c$ is a hyper-parameter. ${\mathcal{R}}_{\mathcal{Q},\mathcal{K}}$ is autocorrelation between series $\mathcal{Q}$ and $\mathcal{K}$ . Roll $\left( {\mathcal{X},\tau }\right)$ represents the operation to $\mathcal{X}$ with time delay $\tau$ ,during which elements that are shifted beyond the first position are re-introduced at the last position. For the encoder-decoder Auto-Correlation (Figure 1), $\mathcal{K},\mathcal{V}$ are from the encoder ${\mathcal{X}}_{\text{en }}^{N}$ and will be resized to length- $O,\mathcal{Q}$ is from the previous block of the decoder.

其中参数 $\operatorname{Topk}\left( \cdot \right)$ 用于获取前 k 个自相关系数的参数，设 $k = \lfloor c \times  \log L\rfloor ,c$ 为一个超参数。${\mathcal{R}}_{\mathcal{Q},\mathcal{K}}$ 是序列 $\mathcal{Q}$ 和 $\mathcal{K}$ 之间的自相关系数。循环移位 $\left( {\mathcal{X},\tau }\right)$ 表示对 $\mathcal{X}$ 进行时间延迟为 $\tau$ 的操作，在此期间，移出首位的元素会重新插入到末位。对于编码器 - 解码器自相关（图 1），$\mathcal{K},\mathcal{V}$ 来自编码器 ${\mathcal{X}}_{\text{en }}^{N}$ 并将调整为长度为 $O,\mathcal{Q}$，$O,\mathcal{Q}$ 来自解码器的前一个模块。

<!-- Media -->

<!-- figureText: (a) LogSparse Attention -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_5.jpg?x=309&y=201&w=1180&h=494&r=0"/>

Figure 3: Auto-Correlation vs. self-attention family. Full Attention [41] (a) adapts the fully connection among all time points. Sparse Attention [23, 48] (b) selects points based on the proposed similarity metrics. LogSparse Attention [26] (c) (c) chooses points following the exponentially increasing intervals. Auto-Correlation (d) focuses on the connections of sub-series among underlying periods.

图3：自相关与自注意力家族。全注意力机制[41]（a）采用所有时间点之间的全连接方式。稀疏注意力机制[23, 48]（b）根据提出的相似度指标选择时间点。对数稀疏注意力机制[26]（c）按照指数增长的间隔选择时间点。自相关（d）关注潜在周期内子序列之间的连接。

<!-- Media -->

For the multi-head version used in Autoformer,with hidden variables of ${d}_{\text{model }}$ channels, $h$ heads, the query,key and value for $i$ -th head are ${\mathcal{Q}}_{i},{\mathcal{K}}_{i},{\mathcal{V}}_{i} \in  {\mathbb{R}}^{L \times  \frac{{d}_{\text{model }}}{h}},i \in  \{ 1,\cdots ,h\}$ . The process is:

对于Autoformer中使用的多头版本，其隐藏变量有${d}_{\text{model }}$个通道、$h$个头，第$i$个头的查询、键和值为${\mathcal{Q}}_{i},{\mathcal{K}}_{i},{\mathcal{V}}_{i} \in  {\mathbb{R}}^{L \times  \frac{{d}_{\text{model }}}{h}},i \in  \{ 1,\cdots ,h\}$。过程如下：

$$
\operatorname{MultiHead}\left( {\mathcal{Q},\mathcal{K},\mathcal{V}}\right)  = {\mathcal{W}}_{\text{output }} * \operatorname{Concat}\left( {{\operatorname{head}}_{1},\cdots ,{\operatorname{head}}_{h}}\right)  \tag{7}
$$

$$
\text{where}{\operatorname{head}}_{i} = \text{Auto-Correlation}\left( {{\mathcal{Q}}_{i},{\mathcal{K}}_{i},{\mathcal{V}}_{i}}\right) \text{.}
$$

Efficient computation For period-based dependencies, these dependencies point to sub-processes at the same phase position of underlying periods and are inherently sparse. Here, we select the most possible delays to avoid picking the opposite phases. Because we aggregate $\mathcal{O}\left( {\log L}\right)$ series whose length is $L$ ,the complexity of Equations 6 and 7 is $\mathcal{O}\left( {L\log L}\right)$ . For the autocorrelation computation (Equation 5),given time series $\left\{  {\mathcal{X}}_{t}\right\}  ,{\mathcal{R}}_{\mathcal{X}\mathcal{X}}\left( \tau \right)$ can be calculated by Fast Fourier Transforms (FFT) based on the Wiener-Khinchin theorem [43]:

高效计算 对于基于周期的依赖关系，这些依赖关系指向底层周期相同相位位置的子过程，并且本质上是稀疏的。在这里，我们选择最可能的延迟以避免选择相反的相位。由于我们聚合长度为 $L$ 的 $\mathcal{O}\left( {\log L}\right)$ 序列，因此方程 6 和 7 的复杂度为 $\mathcal{O}\left( {L\log L}\right)$。对于自相关计算（方程 5），给定的时间序列 $\left\{  {\mathcal{X}}_{t}\right\}  ,{\mathcal{R}}_{\mathcal{X}\mathcal{X}}\left( \tau \right)$ 可以根据维纳 - 辛钦定理 [43] 通过快速傅里叶变换（FFT）计算得出：

$$
{\mathcal{S}}_{\mathcal{X}\mathcal{X}}\left( f\right)  = \mathcal{F}\left( {\mathcal{X}}_{t}\right) {\mathcal{F}}^{ * }\left( {\mathcal{X}}_{t}\right)  = {\int }_{-\infty }^{\infty }{\mathcal{X}}_{t}{e}^{-{i2\pi tf}}\mathrm{\;d}t{\int }_{-\infty }^{\infty }{\mathcal{X}}_{t}{e}^{-{i2\pi tf}}\mathrm{\;d}t \tag{8}
$$

$$
{\mathcal{R}}_{\mathcal{X}\mathcal{X}}\left( \tau \right)  = {\mathcal{F}}^{-1}\left( {{\mathcal{S}}_{\mathcal{X}\mathcal{X}}\left( f\right) }\right)  = {\int }_{-\infty }^{\infty }{\mathcal{S}}_{\mathcal{X}\mathcal{X}}\left( f\right) {e}^{i2\pi f\tau }\mathrm{d}f,
$$

where $\tau  \in  \{ 1,\cdots ,L\} ,\mathcal{F}$ denotes the FFT and ${\mathcal{F}}^{-1}$ is its inverse. * denotes the conjugate operation and ${\mathcal{S}}_{\mathcal{{XX}}}\left( f\right)$ is in the frequency domain. Note that the series autocorrelation of all lags in $\{ 1,\cdots ,L\}$ can be calculated at once by FFT. Thus,Auto-Correlation achieves the $\mathcal{O}\left( {L\log L}\right)$ complexity.

其中 $\tau  \in  \{ 1,\cdots ,L\} ,\mathcal{F}$ 表示快速傅里叶变换（FFT），${\mathcal{F}}^{-1}$ 是其逆变换。* 表示共轭运算，${\mathcal{S}}_{\mathcal{{XX}}}\left( f\right)$ 在频域中。请注意，$\{ 1,\cdots ,L\}$ 中所有滞后的序列自相关可以通过快速傅里叶变换一次性计算得出。因此，自相关的复杂度为 $\mathcal{O}\left( {L\log L}\right)$。

Auto-Correlation vs. self-attention family Different from the point-wise self-attention family, Auto-Correlation presents the series-wise connections (Figure 3). Concretely, for the temporal dependencies, we find the dependencies among sub-series based on the periodicity. In contrast, the self-attention family only calculates the relation between scattered points. Though some self-attentions [26, 48] consider the local information, they only utilize this to help point-wise dependencies discovery. For the information aggregation, we adopt the time delay block to aggregate the similar sub-series from underlying periods. In contrast, self-attentions aggregate the selected points by dot-product. Benefiting from the inherent sparsity and sub-series-level representation aggregation, Auto-Correlation can simultaneously benefit the computation efficiency and information utilization.

自相关与自注意力家族 与逐点自注意力家族不同，自相关呈现出序列级别的联系（图3）。具体而言，对于时间依赖关系，我们基于周期性发现子序列之间的依赖关系。相比之下，自注意力家族仅计算离散点之间的关系。尽管一些自注意力方法[26, 48]考虑了局部信息，但它们仅利用这些信息来辅助逐点依赖关系的发现。在信息聚合方面，我们采用时间延迟块来聚合来自底层周期的相似子序列。相比之下，自注意力方法通过点积来聚合所选的点。得益于固有的稀疏性和子序列级别的表示聚合，自相关可以同时提高计算效率和信息利用率。

## 4 Experiments

## 4 实验

We extensively evaluate the proposed Autoformer on six real-world benchmarks, covering five mainstream time series forecasting applications: energy, traffic, economics, weather and disease.

我们在六个真实世界的基准测试中对提出的Autoformer进行了广泛评估，涵盖了五个主流的时间序列预测应用：能源、交通、经济、天气和疾病。

Datasets Here is a description of the six experiment datasets: (1) ETT [48] dataset contains the data collected from electricity transformers, including load and oil temperature that are recorded every

数据集 以下是对六个实验数据集的描述：（1）ETT [48]数据集包含从电力变压器收集的数据，包括每[此处原文未完整]记录的负载和油温。

<!-- Media -->

Table 1: Multivariate results with different prediction lengths $O \in  \{ {96},{192},{336},{720}\}$ . We set the input length $I$ as 36 for ILI and 96 for the others. A lower MSE or MAE indicates a better prediction.

表1：不同预测时长的多元分析结果 $O \in  \{ {96},{192},{336},{720}\}$。我们将输入时长 $I$ 设定为：流感样病例（ILI）为36，其他为96。均方误差（MSE）或平均绝对误差（MAE）越低，表明预测效果越好。

<table><tr><td colspan="2" rowspan="2">Models Metric</td><td colspan="2">Autoformer</td><td colspan="2">Informer[48]</td><td colspan="2">LogTrans[26</td><td colspan="2">Reformer[23]</td><td colspan="2">LSTNet 25</td><td colspan="2">LSTM[17]</td><td colspan="2">TCN[4]</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETT*</td><td>96</td><td>0.255</td><td>0.339</td><td>0.365</td><td>0.453</td><td>0.768</td><td>0.642</td><td>0.658</td><td>0.619</td><td>3.142</td><td>1.365</td><td>2.041</td><td>1.073</td><td>3.041</td><td>1.330</td></tr><tr><td>192</td><td>0.281</td><td>0.340</td><td>0.533</td><td>0.563</td><td>0.989</td><td>0.757</td><td>1.078</td><td>0.827</td><td>3.154</td><td>1.369</td><td>2.249</td><td>1.112</td><td>3.072</td><td>1.339</td></tr><tr><td>336</td><td>0.339</td><td>0.372</td><td>1.363</td><td>0.887</td><td>1.334</td><td>0.872</td><td>1.549</td><td>0.972</td><td>3.160</td><td>1.369</td><td>2.568</td><td>1.238</td><td>3.105</td><td>1.348</td></tr><tr><td>720</td><td>0.422</td><td>0.419</td><td>3.379</td><td>1.388</td><td>3.048</td><td>1.328</td><td>2.631</td><td>1.242</td><td>3.171</td><td>1.368</td><td>2.720</td><td>1.287</td><td>3.135</td><td>1.354</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.201</td><td>0.317</td><td>0.274</td><td>0.368</td><td>0.258</td><td>0.357</td><td>0.312</td><td>0.402</td><td>0.680</td><td>0.645</td><td>0.375</td><td>0.437</td><td>0.985</td><td>0.813</td></tr><tr><td>192</td><td>0.222</td><td>0.334</td><td>0.296</td><td>0.386</td><td>0.266</td><td>0.368</td><td>0.348</td><td>0.433</td><td>0.725</td><td>0.676</td><td>0.442</td><td>0.473</td><td>0.996</td><td>0.821</td></tr><tr><td>336</td><td>0.231</td><td>0.338</td><td>0.300</td><td>0.394</td><td>0.280</td><td>0.380</td><td>0.350</td><td>0.433</td><td>0.828</td><td>0.727</td><td>0.439</td><td>0.473</td><td>1.000</td><td>0.824</td></tr><tr><td>720</td><td>0.254</td><td>0.361</td><td>0.373</td><td>0.439</td><td>0.283</td><td>0.376</td><td>0.340</td><td>0.420</td><td>0.957</td><td>0.811</td><td>0.980</td><td>0.814</td><td>1.438</td><td>0.784</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.197</td><td>0.323</td><td>0.847</td><td>0.752</td><td>0.968</td><td>0.812</td><td>1.065</td><td>0.829</td><td>1.551</td><td>1.058</td><td>1.453</td><td>1.049</td><td>3.004</td><td>1.432</td></tr><tr><td>192</td><td>0.300</td><td>0.369</td><td>1.204</td><td>0.895</td><td>1.040</td><td>0.851</td><td>1.188</td><td>0.906</td><td>1.477</td><td>1.028</td><td>1.846</td><td>1.179</td><td>3.048</td><td>1.444</td></tr><tr><td>336</td><td>0.509</td><td>0.524</td><td>1.672</td><td>1.036</td><td>1.659</td><td>1.081</td><td>1.357</td><td>0.976</td><td>1.507</td><td>1.031</td><td>2.136</td><td>1.231</td><td>3.113</td><td>1.459</td></tr><tr><td>720</td><td>1.447</td><td>0.941</td><td>2.478</td><td>1.310</td><td>1.941</td><td>1.127</td><td>1.510</td><td>1.016</td><td>2.285</td><td>1.243</td><td>2.984</td><td>1.427</td><td>3.150</td><td>1.458</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.613</td><td>0.388</td><td>0.719</td><td>0.391</td><td>0.684</td><td>0.384</td><td>0.732</td><td>0.423</td><td>1.107</td><td>0.685</td><td>0.843</td><td>0.453</td><td>1.438</td><td>0.784</td></tr><tr><td>192</td><td>0.616</td><td>0.382</td><td>0.696</td><td>0.379</td><td>0.685</td><td>0.390</td><td>0.733</td><td>0.420</td><td>1.157</td><td>0.706</td><td>0.847</td><td>0.453</td><td>1.463</td><td>0.794</td></tr><tr><td>336</td><td>0.622</td><td>0.337</td><td>0.777</td><td>0.420</td><td>0.733</td><td>0.408</td><td>0.742</td><td>0.420</td><td>1.216</td><td>0.730</td><td>0.853</td><td>0.455</td><td>1.479</td><td>0.799</td></tr><tr><td>720</td><td>0.660</td><td>0.408</td><td>0.864</td><td>0.472</td><td>0.717</td><td>0.396</td><td>0.755</td><td>0.423</td><td>1.481</td><td>0.805</td><td>1.500</td><td>0.805</td><td>1.499</td><td>0.804</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.266</td><td>0.336</td><td>0.300</td><td>0.384</td><td>0.458</td><td>0.490</td><td>0.689</td><td>0.596</td><td>0.594</td><td>0.587</td><td>0.369</td><td>0.406</td><td>0.615</td><td>0.589</td></tr><tr><td>192</td><td>0.307</td><td>0.367</td><td>0.598</td><td>0.544</td><td>0.658</td><td>0.589</td><td>0.752</td><td>0.638</td><td>0.560</td><td>0.565</td><td>0.416</td><td>0.435</td><td>0.629</td><td>0.600</td></tr><tr><td>336</td><td>0.359</td><td>0.395</td><td>0.578</td><td>0.523</td><td>0.797</td><td>0.652</td><td>0.639</td><td>0.596</td><td>0.597</td><td>0.587</td><td>0.455</td><td>0.454</td><td>0.639</td><td>0.608</td></tr><tr><td>720</td><td>0.419</td><td>0.428</td><td>1.059</td><td>0.741</td><td>0.869</td><td>0.675</td><td>1.130</td><td>0.792</td><td>0.618</td><td>0.599</td><td>0.535</td><td>0.520</td><td>0.639</td><td>0.610</td></tr><tr><td rowspan="4">日</td><td>24</td><td>3.483</td><td>1.287</td><td>5.764</td><td>1.677</td><td>4.480</td><td>1.444</td><td>4.400</td><td>1.382</td><td>6.026</td><td>1.770</td><td>5.914</td><td>1.734</td><td>6.624</td><td>1.830</td></tr><tr><td>36</td><td>3.103</td><td>1.148</td><td>4.755</td><td>1.467</td><td>4.799</td><td>1.467</td><td>4.783</td><td>1.448</td><td>5.340</td><td>1.668</td><td>6.631</td><td>1.845</td><td>6.858</td><td>1.879</td></tr><tr><td>48</td><td>2.669</td><td>1.085</td><td>4.763</td><td>1.469</td><td>4.800</td><td>1.468</td><td>4.832</td><td>1.465</td><td>6.080</td><td>1.787</td><td>6.736</td><td>1.857</td><td>6.968</td><td>1.892</td></tr><tr><td>60</td><td>2.770</td><td>1.125</td><td>5.264</td><td>1.564</td><td>5.278</td><td>1.560</td><td>4.882</td><td>1.483</td><td>5.548</td><td>1.720</td><td>6.870</td><td>1.879</td><td>7.127</td><td>1.918</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">模型指标</td><td colspan="2">自动转换器（Autoformer）</td><td colspan="2">信息转换器（Informer）[48]</td><td colspan="2">对数转换器（LogTrans）[26</td><td colspan="2">改革者（Reformer）[23]</td><td colspan="2">长短期记忆网络（LSTNet）25</td><td colspan="2">长短期记忆网络（LSTM）[17]</td><td colspan="2">时间卷积网络（TCN）[4]</td></tr><tr><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="4">电力变压器测试（ETT）*</td><td>96</td><td>0.255</td><td>0.339</td><td>0.365</td><td>0.453</td><td>0.768</td><td>0.642</td><td>0.658</td><td>0.619</td><td>3.142</td><td>1.365</td><td>2.041</td><td>1.073</td><td>3.041</td><td>1.330</td></tr><tr><td>192</td><td>0.281</td><td>0.340</td><td>0.533</td><td>0.563</td><td>0.989</td><td>0.757</td><td>1.078</td><td>0.827</td><td>3.154</td><td>1.369</td><td>2.249</td><td>1.112</td><td>3.072</td><td>1.339</td></tr><tr><td>336</td><td>0.339</td><td>0.372</td><td>1.363</td><td>0.887</td><td>1.334</td><td>0.872</td><td>1.549</td><td>0.972</td><td>3.160</td><td>1.369</td><td>2.568</td><td>1.238</td><td>3.105</td><td>1.348</td></tr><tr><td>720</td><td>0.422</td><td>0.419</td><td>3.379</td><td>1.388</td><td>3.048</td><td>1.328</td><td>2.631</td><td>1.242</td><td>3.171</td><td>1.368</td><td>2.720</td><td>1.287</td><td>3.135</td><td>1.354</td></tr><tr><td rowspan="4">电力</td><td>96</td><td>0.201</td><td>0.317</td><td>0.274</td><td>0.368</td><td>0.258</td><td>0.357</td><td>0.312</td><td>0.402</td><td>0.680</td><td>0.645</td><td>0.375</td><td>0.437</td><td>0.985</td><td>0.813</td></tr><tr><td>192</td><td>0.222</td><td>0.334</td><td>0.296</td><td>0.386</td><td>0.266</td><td>0.368</td><td>0.348</td><td>0.433</td><td>0.725</td><td>0.676</td><td>0.442</td><td>0.473</td><td>0.996</td><td>0.821</td></tr><tr><td>336</td><td>0.231</td><td>0.338</td><td>0.300</td><td>0.394</td><td>0.280</td><td>0.380</td><td>0.350</td><td>0.433</td><td>0.828</td><td>0.727</td><td>0.439</td><td>0.473</td><td>1.000</td><td>0.824</td></tr><tr><td>720</td><td>0.254</td><td>0.361</td><td>0.373</td><td>0.439</td><td>0.283</td><td>0.376</td><td>0.340</td><td>0.420</td><td>0.957</td><td>0.811</td><td>0.980</td><td>0.814</td><td>1.438</td><td>0.784</td></tr><tr><td rowspan="4">交换；交易；交流</td><td>96</td><td>0.197</td><td>0.323</td><td>0.847</td><td>0.752</td><td>0.968</td><td>0.812</td><td>1.065</td><td>0.829</td><td>1.551</td><td>1.058</td><td>1.453</td><td>1.049</td><td>3.004</td><td>1.432</td></tr><tr><td>192</td><td>0.300</td><td>0.369</td><td>1.204</td><td>0.895</td><td>1.040</td><td>0.851</td><td>1.188</td><td>0.906</td><td>1.477</td><td>1.028</td><td>1.846</td><td>1.179</td><td>3.048</td><td>1.444</td></tr><tr><td>336</td><td>0.509</td><td>0.524</td><td>1.672</td><td>1.036</td><td>1.659</td><td>1.081</td><td>1.357</td><td>0.976</td><td>1.507</td><td>1.031</td><td>2.136</td><td>1.231</td><td>3.113</td><td>1.459</td></tr><tr><td>720</td><td>1.447</td><td>0.941</td><td>2.478</td><td>1.310</td><td>1.941</td><td>1.127</td><td>1.510</td><td>1.016</td><td>2.285</td><td>1.243</td><td>2.984</td><td>1.427</td><td>3.150</td><td>1.458</td></tr><tr><td rowspan="4">交通；流量</td><td>96</td><td>0.613</td><td>0.388</td><td>0.719</td><td>0.391</td><td>0.684</td><td>0.384</td><td>0.732</td><td>0.423</td><td>1.107</td><td>0.685</td><td>0.843</td><td>0.453</td><td>1.438</td><td>0.784</td></tr><tr><td>192</td><td>0.616</td><td>0.382</td><td>0.696</td><td>0.379</td><td>0.685</td><td>0.390</td><td>0.733</td><td>0.420</td><td>1.157</td><td>0.706</td><td>0.847</td><td>0.453</td><td>1.463</td><td>0.794</td></tr><tr><td>336</td><td>0.622</td><td>0.337</td><td>0.777</td><td>0.420</td><td>0.733</td><td>0.408</td><td>0.742</td><td>0.420</td><td>1.216</td><td>0.730</td><td>0.853</td><td>0.455</td><td>1.479</td><td>0.799</td></tr><tr><td>720</td><td>0.660</td><td>0.408</td><td>0.864</td><td>0.472</td><td>0.717</td><td>0.396</td><td>0.755</td><td>0.423</td><td>1.481</td><td>0.805</td><td>1.500</td><td>0.805</td><td>1.499</td><td>0.804</td></tr><tr><td rowspan="4">天气</td><td>96</td><td>0.266</td><td>0.336</td><td>0.300</td><td>0.384</td><td>0.458</td><td>0.490</td><td>0.689</td><td>0.596</td><td>0.594</td><td>0.587</td><td>0.369</td><td>0.406</td><td>0.615</td><td>0.589</td></tr><tr><td>192</td><td>0.307</td><td>0.367</td><td>0.598</td><td>0.544</td><td>0.658</td><td>0.589</td><td>0.752</td><td>0.638</td><td>0.560</td><td>0.565</td><td>0.416</td><td>0.435</td><td>0.629</td><td>0.600</td></tr><tr><td>336</td><td>0.359</td><td>0.395</td><td>0.578</td><td>0.523</td><td>0.797</td><td>0.652</td><td>0.639</td><td>0.596</td><td>0.597</td><td>0.587</td><td>0.455</td><td>0.454</td><td>0.639</td><td>0.608</td></tr><tr><td>720</td><td>0.419</td><td>0.428</td><td>1.059</td><td>0.741</td><td>0.869</td><td>0.675</td><td>1.130</td><td>0.792</td><td>0.618</td><td>0.599</td><td>0.535</td><td>0.520</td><td>0.639</td><td>0.610</td></tr><tr><td rowspan="4">日</td><td>24</td><td>3.483</td><td>1.287</td><td>5.764</td><td>1.677</td><td>4.480</td><td>1.444</td><td>4.400</td><td>1.382</td><td>6.026</td><td>1.770</td><td>5.914</td><td>1.734</td><td>6.624</td><td>1.830</td></tr><tr><td>36</td><td>3.103</td><td>1.148</td><td>4.755</td><td>1.467</td><td>4.799</td><td>1.467</td><td>4.783</td><td>1.448</td><td>5.340</td><td>1.668</td><td>6.631</td><td>1.845</td><td>6.858</td><td>1.879</td></tr><tr><td>48</td><td>2.669</td><td>1.085</td><td>4.763</td><td>1.469</td><td>4.800</td><td>1.468</td><td>4.832</td><td>1.465</td><td>6.080</td><td>1.787</td><td>6.736</td><td>1.857</td><td>6.968</td><td>1.892</td></tr><tr><td>60</td><td>2.770</td><td>1.125</td><td>5.264</td><td>1.564</td><td>5.278</td><td>1.560</td><td>4.882</td><td>1.483</td><td>5.548</td><td>1.720</td><td>6.870</td><td>1.879</td><td>7.127</td><td>1.918</td></tr></tbody></table>

* ETT means the ETTm2. See Appendix A for the full benchmark of ETTh1, ETTh2, ETTm1.

* ETT指的是ETTm2。有关ETTh1、ETTh2、ETTm1的完整基准测试，请参阅附录A。

<!-- Media -->

15 minutes between July 2016 and July 2018. (2) Electricity 1 dataset contains the hourly electricity consumption of 321 customers from 2012 to 2014. (3) Exchange [25] records the daily exchange rates of eight different countries ranging from 1990 to 2016. (4) Traffic ${}^{2}$ is a collection of hourly data from California Department of Transportation, which describes the road occupancy rates measured by different sensors on San Francisco Bay area freeways. (5) Weather ${}^{3}$ is recorded every 10 minutes for 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc. (6) $I{L}^{4}$ includes the weekly recorded influenza-like illness (ILI) patients data from Centers for Disease Control and Prevention of the United States between 2002 and 2021, which describes the ratio of patients seen with ILI and the total number of the patients. We follow standard protocol and split all datasets into training, validation and test set in chronological order by the ratio of 6:2:2 for the ETT dataset and 7:1:2 for the other datasets.

2016年7月至2018年7月期间，每15分钟的数据。（2）电力1数据集包含2012年至2014年321个客户的每小时用电量。（3）汇率数据集[25]记录了1990年至2016年八个不同国家的每日汇率。（4）交通数据集${}^{2}$是加利福尼亚州交通运输部（California Department of Transportation）的每小时数据集合，描述了旧金山湾区高速公路上不同传感器测量的道路占有率。（5）天气数据集${}^{3}$记录了2020年全年每10分钟的数据，包含21个气象指标，如气温、湿度等。（6）数据集$I{L}^{4}$包含美国疾病控制与预防中心（Centers for Disease Control and Prevention）2002年至2021年每周记录的流感样疾病（ILI）患者数据，描述了流感样疾病患者与患者总数的比例。我们遵循标准协议，按时间顺序将所有数据集划分为训练集、验证集和测试集，ETT数据集的划分比例为6:2:2，其他数据集的划分比例为7:1:2。

Implementation details Our method is trained with the L2 loss, using the ADAM [22] optimizer with an initial learning rate of ${10}^{-4}$ . Batch size is set to 32 . The training process is early stopped within 10 epochs. All experiments are repeated three times, implemented in PyTorch [31] and conducted on a single NVIDIA TITAN RTX 24GB GPUs. The hyper-parameter $c$ of Auto-Correlation is in the range of 1 to 3 to trade off performance and efficiency. See Appendix E and B for standard deviations and sensitivity analysis. Autoformer contains 2 encoder layers and 1 decoder layer.

实现细节 我们的方法使用L2损失进行训练，采用ADAM [22]优化器，初始学习率为${10}^{-4}$。批量大小设置为32。训练过程在10个训练周期内提前停止。所有实验均重复三次，在PyTorch [31]中实现，并在单块NVIDIA TITAN RTX 24GB GPU上进行。自相关（Auto - Correlation）的超参数$c$范围为1到3，以权衡性能和效率。有关标准差和敏感性分析，请参阅附录E和B。Autoformer包含2个编码器层和1个解码器层。

Baselines We include 10 baseline methods. For the multivariate setting, we select three latest state-of-the-art transformer-based models: Informer [48], Reformer [23], LogTrans [26], two RNN-based models: LSTNet [25], LSTM [17] and CNN-based TCN [4] as baselines. For the univariate setting, we include more competitive baselines: N-BEATS [29], DeepAR [34], Prophet [39] and ARMIA [1].

基线模型 我们纳入了10种基线方法。对于多变量设置，我们选择了三种最新的基于Transformer的最先进模型：Informer [48]、Reformer [23]、LogTrans [26]，两种基于循环神经网络（RNN）的模型：LSTNet [25]、长短期记忆网络（LSTM）[17]，以及基于卷积神经网络（CNN）的时间卷积网络（TCN）[4]作为基线。对于单变量设置，我们纳入了更多具有竞争力的基线模型：N - BEATS [29]、深度自回归模型（DeepAR）[34]、Prophet [39]和自回归移动平均积分滑动平均模型（ARMIA）[1]。

---

<!-- Footnote -->

https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

http://pems.dot.ca.gov

https://www.bgc-jena.mpg.de/wetter/

https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html

<!-- Footnote -->

---

<!-- Media -->

Table 2: Univariate results with different prediction lengths $O \in  \{ {96},{192},{336},{720}\}$ on typical datasets. We set the input length $I$ as 96 . A lower MSE or MAE indicates a better prediction.

表2：典型数据集上不同预测长度$O \in  \{ {96},{192},{336},{720}\}$的单变量结果。我们将输入长度$I$设置为96。均方误差（MSE）或平均绝对误差（MAE）越低，表示预测效果越好。

<table><tr><td colspan="2" rowspan="2">Models Metric</td><td colspan="2">Autoformer</td><td colspan="2">N-BEATS 29</td><td colspan="2">Informer 48</td><td colspan="2">LogTrans 26</td><td colspan="2">Reformer 23</td><td colspan="2">DeepAR[34]</td><td colspan="2">Prophet[39]</td><td colspan="2">ARIMA[1]</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">加盟</td><td>96</td><td>0.065</td><td>0.189</td><td>0.082</td><td>0.219</td><td>0.088</td><td>0.225</td><td>0.082</td><td>0.217</td><td>0.131</td><td>0.288</td><td>0.099</td><td>0.237</td><td>0.287</td><td>0.456</td><td>0.211</td><td>0.362</td></tr><tr><td>192</td><td>0.118</td><td>0.256</td><td>0.120</td><td>0.268</td><td>0.132</td><td>0.283</td><td>0.133</td><td>0.284</td><td>0.186</td><td>0.354</td><td>0.154</td><td>0.310</td><td>0.312</td><td>0.483</td><td>0.261</td><td>0.406</td></tr><tr><td>336</td><td>0.154</td><td>0.305</td><td>0.226</td><td>0.370</td><td>0.180</td><td>0.336</td><td>0.201</td><td>0.361</td><td>0.220</td><td>0.381</td><td>0.277</td><td>0.428</td><td>0.331</td><td>0.474</td><td>0.317</td><td>0.448</td></tr><tr><td>720</td><td>0.182</td><td>0.335</td><td>0.188</td><td>0.338</td><td>0.300</td><td>0.435</td><td>0.268</td><td>0.407</td><td>0.267</td><td>0.430</td><td>0.332</td><td>0.468</td><td>0.534</td><td>0.593</td><td>0.366</td><td>0.487</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>0.241</td><td>0.387</td><td>0.156</td><td>0.299</td><td>0.591</td><td>0.615</td><td>0.279</td><td>0.441</td><td>1.327</td><td>0.944</td><td>0.417</td><td>0.515</td><td>0.828</td><td>0.762</td><td>0.112</td><td>0.245</td></tr><tr><td>192</td><td>0.273</td><td>0.403</td><td>0.669</td><td>0.665</td><td>1.183</td><td>0.912</td><td>1.950</td><td>1.048</td><td>1.258</td><td>0.924</td><td>0.813</td><td>0.735</td><td>0.909</td><td>0.974</td><td>0.304</td><td>0.404</td></tr><tr><td>336</td><td>0.508</td><td>0.539</td><td>0.611</td><td>0.605</td><td>1.367</td><td>0.984</td><td>2.438</td><td>1.262</td><td>2.179</td><td>1.296</td><td>1.331</td><td>0.962</td><td>1.304</td><td>0.988</td><td>0.736</td><td>0.598</td></tr><tr><td>720</td><td>0.991</td><td>0.768</td><td>1.111</td><td>0.860</td><td>1.872</td><td>1.072</td><td>2.010</td><td>1.247</td><td>1.280</td><td>0.953</td><td>1.894</td><td>1.181</td><td>3.238</td><td>1.566</td><td>1.871</td><td>0.935</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">模型指标</td><td colspan="2">自动转换器（Autoformer）</td><td colspan="2">N-BEATS 29</td><td colspan="2">信息器 48（Informer 48）</td><td colspan="2">对数转换器 26（LogTrans 26）</td><td colspan="2">改革者 23（Reformer 23）</td><td colspan="2">深度自回归（DeepAR）[34]</td><td colspan="2">先知（Prophet）[39]</td><td colspan="2">自回归积分滑动平均模型（ARIMA）[1]</td></tr><tr><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="4">加盟</td><td>96</td><td>0.065</td><td>0.189</td><td>0.082</td><td>0.219</td><td>0.088</td><td>0.225</td><td>0.082</td><td>0.217</td><td>0.131</td><td>0.288</td><td>0.099</td><td>0.237</td><td>0.287</td><td>0.456</td><td>0.211</td><td>0.362</td></tr><tr><td>192</td><td>0.118</td><td>0.256</td><td>0.120</td><td>0.268</td><td>0.132</td><td>0.283</td><td>0.133</td><td>0.284</td><td>0.186</td><td>0.354</td><td>0.154</td><td>0.310</td><td>0.312</td><td>0.483</td><td>0.261</td><td>0.406</td></tr><tr><td>336</td><td>0.154</td><td>0.305</td><td>0.226</td><td>0.370</td><td>0.180</td><td>0.336</td><td>0.201</td><td>0.361</td><td>0.220</td><td>0.381</td><td>0.277</td><td>0.428</td><td>0.331</td><td>0.474</td><td>0.317</td><td>0.448</td></tr><tr><td>720</td><td>0.182</td><td>0.335</td><td>0.188</td><td>0.338</td><td>0.300</td><td>0.435</td><td>0.268</td><td>0.407</td><td>0.267</td><td>0.430</td><td>0.332</td><td>0.468</td><td>0.534</td><td>0.593</td><td>0.366</td><td>0.487</td></tr><tr><td rowspan="4">交换；交易；兑换</td><td>96</td><td>0.241</td><td>0.387</td><td>0.156</td><td>0.299</td><td>0.591</td><td>0.615</td><td>0.279</td><td>0.441</td><td>1.327</td><td>0.944</td><td>0.417</td><td>0.515</td><td>0.828</td><td>0.762</td><td>0.112</td><td>0.245</td></tr><tr><td>192</td><td>0.273</td><td>0.403</td><td>0.669</td><td>0.665</td><td>1.183</td><td>0.912</td><td>1.950</td><td>1.048</td><td>1.258</td><td>0.924</td><td>0.813</td><td>0.735</td><td>0.909</td><td>0.974</td><td>0.304</td><td>0.404</td></tr><tr><td>336</td><td>0.508</td><td>0.539</td><td>0.611</td><td>0.605</td><td>1.367</td><td>0.984</td><td>2.438</td><td>1.262</td><td>2.179</td><td>1.296</td><td>1.331</td><td>0.962</td><td>1.304</td><td>0.988</td><td>0.736</td><td>0.598</td></tr><tr><td>720</td><td>0.991</td><td>0.768</td><td>1.111</td><td>0.860</td><td>1.872</td><td>1.072</td><td>2.010</td><td>1.247</td><td>1.280</td><td>0.953</td><td>1.894</td><td>1.181</td><td>3.238</td><td>1.566</td><td>1.871</td><td>0.935</td></tr></tbody></table>

<!-- Media -->

### 4.1 Main Results

### 4.1 主要结果

To compare performances under different future horizons, we fix the input length and evaluate models with a wide range of prediction lengths:96,192,336,720. This setting precisely meets the definition of long-term forecasting. Here are results on both the multivariate and univariate settings.

为了比较不同未来时间范围下的性能，我们固定输入长度，并使用多种预测长度（96、192、336、720）对模型进行评估。这一设置恰好符合长期预测的定义。以下是多变量和单变量设置下的结果。

Multivariate results As for the multivariate setting, Autoformer achieves the consistent state-of-the-art performance in all benchmarks and all prediction length settings (Table 10). Especially, under the input-96-predict-336 setting, compared to previous state-of-the-art results, Autoformer gives 74% (1.334 $\rightarrow$ 0.339) MSE reduction in ETT, ${18}\% \left( {{0.280} \rightarrow  {0.231}}\right)$ in Electricity, ${61}\% \left( {{1.357} \rightarrow  {0.509}}\right)$ in Exchange, ${15}\% \left( {{0.733} \rightarrow  {0.622}}\right)$ in Traffic and ${21}\% \left( {{0.455} \rightarrow  {0.359}}\right)$ in Weather. For the input- 36-predict-60 setting of ILI,Autoformer makes ${43}\% \left( {{4.882} \rightarrow  {2.770}}\right)$ MSE reduction. Overall, Autoformer yields a $\mathbf{{38}\% }$ averaged MSE reduction among above settings. Note that Autoformer still provides remarkable improvements in the Exchange dataset that is without obvious periodicity. See Appendix E for detailed showcases. Besides, we can also find that the performance of Autoformer changes quite steadily as the prediction length $O$ increases. It means that Autoformer retains better long-term robustness, which is meaningful for real-world practical applications, such as weather early warning and long-term energy consumption planning.

多变量结果 对于多变量设置，自动变换器（Autoformer）在所有基准测试和所有预测长度设置下均实现了一致的最先进性能（表10）。特别是在输入96步预测336步的设置下，与之前的最先进结果相比，自动变换器在电力负荷时间序列数据集（ETT）中均方误差（MSE）降低了74%（1.334 $\rightarrow$ 0.339），在电力数据集（Electricity）中${18}\% \left( {{0.280} \rightarrow  {0.231}}\right)$，在汇率数据集（Exchange）中${61}\% \left( {{1.357} \rightarrow  {0.509}}\right)$，在交通数据集（Traffic）中${15}\% \left( {{0.733} \rightarrow  {0.622}}\right)$，在气象数据集（Weather）中${21}\% \left( {{0.455} \rightarrow  {0.359}}\right)$。对于流感样疾病数据集（ILI）输入36步预测60步的设置，自动变换器使均方误差降低了${43}\% \left( {{4.882} \rightarrow  {2.770}}\right)$。总体而言，自动变换器在上述设置下平均使均方误差降低了$\mathbf{{38}\% }$。值得注意的是，自动变换器在没有明显周期性的汇率数据集中仍有显著改进。详细展示请见附录E。此外，我们还可以发现，随着预测长度$O$的增加，自动变换器的性能变化相当稳定。这意味着自动变换器具有更好的长期鲁棒性，这对于现实世界的实际应用（如气象预警和长期能源消耗规划）具有重要意义。

Univariate results We list the univariate results of two typical datasets in Table 2 Under the comparison with extensive baselines, our Autoformer still achieves state-of-the-art performance for the long-term forecasting tasks. In particular, for the input-96-predict-336 setting, our model achieves ${14}\% \left( {{0.180} \rightarrow  {0.145}}\right)$ MSE reduction on the ETT dataset with obvious periodicity. For the Exchange dataset without obvious periodicity,Autoformer surpasses other baselines by ${17}\% \left( {{0.611} \rightarrow  {0.508}}\right)$ and shows greater long-term forecasting capacity. Also, we find that ARIMA [1] performs best in the input-96-predict-96 setting of the Exchange dataset but fails in the long-term setting. This situation of ARIMA can be benefited from its inherent capacity for non-stationary economic data but is limited by the intricate temporal patterns of real-world series.

单变量结果 我们在表2中列出了两个典型数据集的单变量结果。与大量基线模型相比，我们的Autoformer在长期预测任务中仍能达到最先进的性能。特别是在输入96步预测336步的设置下，我们的模型在具有明显周期性的ETT数据集上实现了${14}\% \left( {{0.180} \rightarrow  {0.145}}\right)$的均方误差（MSE）降低。对于没有明显周期性的Exchange数据集，Autoformer比其他基线模型高出${17}\% \left( {{0.611} \rightarrow  {0.508}}\right)$，并显示出更强的长期预测能力。此外，我们发现自回归积分滑动平均模型（ARIMA）[1]在Exchange数据集的输入96步预测96步设置中表现最佳，但在长期设置中表现不佳。ARIMA的这种情况得益于其对非平稳经济数据的固有处理能力，但受到现实世界序列复杂时间模式的限制。

### 4.2 Ablation studies

### 4.2 消融研究

<!-- Media -->

Table 3: Ablation of decomposition in multivariate ETT with MSE metric. Ours adopts our progressive architecture into other models. Sep employs two models to forecast pre-decomposed seasonal and trend-cyclical components separately. Promotion is the MSE reduction compared to Origin.

表3：使用均方误差（MSE）指标对多变量电力负荷时间序列预测（multivariate ETT）中的分解进行消融实验。“我们的方法”将我们的渐进式架构应用于其他模型。“分离法”采用两个模型分别预测预分解的季节性和趋势 - 周期性成分。“提升效果”是与“原始方法”相比均方误差的降低值。

<table><tr><td>Input-96</td><td colspan="3">Transformer [41]</td><td colspan="3">Informer [48]</td><td colspan="3">LogTrans[23]</td><td colspan="3">Reformer[26]</td><td colspan="2">Promotion</td></tr><tr><td>Predict-O</td><td>Origin</td><td>Sep</td><td>Ours</td><td>Origin</td><td>Sep</td><td>Ours</td><td>Origin</td><td>Sep</td><td>Ours</td><td>Origin</td><td>Sep</td><td>Ours</td><td>Sep</td><td>Ours</td></tr><tr><td>96</td><td>0.604</td><td>0.311</td><td>0.204</td><td>0.365</td><td>0.490</td><td>0.354</td><td>0.768</td><td>0.862</td><td>0.231</td><td>0.658</td><td>0.445</td><td>0.218</td><td>0.069</td><td>0.347</td></tr><tr><td>192</td><td>1.060</td><td>0.760</td><td>0.266</td><td>0.533</td><td>0.658</td><td>0.432</td><td>0.989</td><td>0.533</td><td>0.378</td><td>1.078</td><td>0.510</td><td>0.336</td><td>0.300</td><td>0.562</td></tr><tr><td>336</td><td>1.413</td><td>0.665</td><td>0.375</td><td>1.363</td><td>1.469</td><td>0.481</td><td>1.334</td><td>0.762</td><td>0.362</td><td>1.549</td><td>1.028</td><td>0.366</td><td>0.434</td><td>1.019</td></tr><tr><td>720</td><td>2.672</td><td>3.200</td><td>0.537</td><td>3.379</td><td>2.766</td><td>0.822</td><td>3.048</td><td>2.601</td><td>0.539</td><td>2.631</td><td>2.845</td><td>0.502</td><td>0.079</td><td>2.332</td></tr></table>

<table><tbody><tr><td>输入 - 96</td><td colspan="3">Transformer模型 [41]</td><td colspan="3">Informer模型 [48]</td><td colspan="3">LogTrans模型[23]</td><td colspan="3">Reformer模型[26]</td><td colspan="2">推广</td></tr><tr><td>预测-O（Predict-O）</td><td>起源（Origin）</td><td>九月（Sep）</td><td>我们的（Ours）</td><td>起源（Origin）</td><td>九月（Sep）</td><td>我们的（Ours）</td><td>起源（Origin）</td><td>九月（Sep）</td><td>我们的（Ours）</td><td>起源（Origin）</td><td>九月（Sep）</td><td>我们的（Ours）</td><td>九月（Sep）</td><td>我们的（Ours）</td></tr><tr><td>96</td><td>0.604</td><td>0.311</td><td>0.204</td><td>0.365</td><td>0.490</td><td>0.354</td><td>0.768</td><td>0.862</td><td>0.231</td><td>0.658</td><td>0.445</td><td>0.218</td><td>0.069</td><td>0.347</td></tr><tr><td>192</td><td>1.060</td><td>0.760</td><td>0.266</td><td>0.533</td><td>0.658</td><td>0.432</td><td>0.989</td><td>0.533</td><td>0.378</td><td>1.078</td><td>0.510</td><td>0.336</td><td>0.300</td><td>0.562</td></tr><tr><td>336</td><td>1.413</td><td>0.665</td><td>0.375</td><td>1.363</td><td>1.469</td><td>0.481</td><td>1.334</td><td>0.762</td><td>0.362</td><td>1.549</td><td>1.028</td><td>0.366</td><td>0.434</td><td>1.019</td></tr><tr><td>720</td><td>2.672</td><td>3.200</td><td>0.537</td><td>3.379</td><td>2.766</td><td>0.822</td><td>3.048</td><td>2.601</td><td>0.539</td><td>2.631</td><td>2.845</td><td>0.502</td><td>0.079</td><td>2.332</td></tr></tbody></table>

<!-- Media -->

Decomposition architecture With our proposed progressive decomposition architecture, other models can gain consistent promotion,especially as the prediction length $O$ increases (Table 3). This verifies that our method can generalize to other models and release the capacity of other dependencies learning mechanisms, alleviate the distraction caused by intricate patterns. Besides, our architecture outperforms the pre-processing, although the latter employs a bigger model and more parameters. Especially, pre-decomposing may even bring negative effect because it neglects the interaction of components during long-term future, such as Transformer [41] predict-720, Informer [48] predict-336.

分解架构 通过我们提出的渐进式分解架构，其他模型可以获得持续的提升，尤其是当预测长度 $O$ 增加时（表3）。这验证了我们的方法可以推广到其他模型，并释放其他依赖学习机制的能力，减轻复杂模式造成的干扰。此外，我们的架构优于预处理方法，尽管后者采用了更大的模型和更多的参数。特别是，预分解甚至可能带来负面影响，因为它忽略了长期未来中各组件之间的相互作用，例如Transformer [41] 预测720、Informer [48] 预测336。

Auto-Correlation vs. self-attention family As shown in Table 4, our proposed Auto-Correlation achieves the best performance under various input- $I$ -predict- $O$ settings,which verifies the effectiveness of series-wise connections comparing to point-wise self-attentions (Figure 3). Furthermore, we can also observe that Auto-Correlation is memory efficiency from the last column of Table 4, which can be used in long sequence forecasting, such as input-336-predict-1440.

自相关与自注意力家族 如表4所示，我们提出的自相关方法在各种输入 $I$ -预测 $O$ 设置下都取得了最佳性能，这验证了与逐点自注意力相比，序列级连接的有效性（图3）。此外，从表4的最后一列我们还可以观察到，自相关方法具有内存效率，可用于长序列预测，例如输入336 - 预测1440。

<!-- Media -->

Table 4: Comparison of Auto-Correlation and self-attention in the multivariate ETT. We replace the Auto-Correlation in Autoformer with different self-attentions. The "-" indicates the out-of-memory.

表4：多元电力变压器负荷预测（ETT）中自相关与自注意力机制的比较。我们用不同的自注意力机制替换了自former模型中的自相关机制。“-”表示内存不足。

<table><tr><td colspan="2">Input Length $I$</td><td colspan="3">96</td><td colspan="3">192</td><td colspan="3">336</td></tr><tr><td>Prediction Length $O$</td><td/><td>336</td><td>720</td><td>1440</td><td>336</td><td>720</td><td>1440</td><td>336</td><td>720</td><td>1440</td></tr><tr><td>Auto-</td><td>MSE</td><td>0.339</td><td>0.422</td><td>0.555</td><td>0.355</td><td>0.429</td><td>0.503</td><td>0.361</td><td>0.425</td><td>0.574</td></tr><tr><td>Correlation</td><td>MAE</td><td>0.372</td><td>0.419</td><td>0.496</td><td>0.392</td><td>0.430</td><td>0.484</td><td>0.406</td><td>0.440</td><td>0.534</td></tr><tr><td>Full</td><td>MSE</td><td>0.375</td><td>0.537</td><td>0.667</td><td>0.450</td><td>0.554</td><td>-</td><td>0.501</td><td>0.647</td><td>-</td></tr><tr><td>Attention [41]</td><td>MAE</td><td>0.425</td><td>0.502</td><td>0.589</td><td>0.470</td><td>0.533</td><td>-</td><td>0.485</td><td>0.491</td><td>-</td></tr><tr><td>LogSparse</td><td>MSE</td><td>0.362</td><td>0.539</td><td>0.582</td><td>0.420</td><td>0.552</td><td>0.958</td><td>0.474</td><td>0.601</td><td>-</td></tr><tr><td>Attention 26</td><td>MAE</td><td>0.413</td><td>0.522</td><td>0.529</td><td>0.450</td><td>0.513</td><td>0.736</td><td>0.474</td><td>0.524</td><td>-</td></tr><tr><td>LSH</td><td>MSE</td><td>0.366</td><td>0.502</td><td>0.663</td><td>0.407</td><td>0.636</td><td>1.069</td><td>0.442</td><td>0.615</td><td>-</td></tr><tr><td>Attention 23</td><td>MAE</td><td>0.404</td><td>0.475</td><td>0.567</td><td>0.421</td><td>0.571</td><td>0.756</td><td>0.476</td><td>0.532</td><td>-</td></tr><tr><td>ProbSparse</td><td>MSE</td><td>0.481</td><td>0.822</td><td>0.715</td><td>0.404</td><td>1.148</td><td>0.732</td><td>0.417</td><td>0.631</td><td>1.133</td></tr><tr><td>Attention 48</td><td>MAE</td><td>0.472</td><td>0.559</td><td>0.586</td><td>0.425</td><td>0.654</td><td>0.602</td><td>0.434</td><td>0.528</td><td>0.691</td></tr></table>

<table><tbody><tr><td colspan="2">输入长度 $I$</td><td colspan="3">96</td><td colspan="3">192</td><td colspan="3">336</td></tr><tr><td>预测长度 $O$</td><td></td><td>336</td><td>720</td><td>1440</td><td>336</td><td>720</td><td>1440</td><td>336</td><td>720</td><td>1440</td></tr><tr><td>自动-</td><td>均方误差（Mean Squared Error，MSE）</td><td>0.339</td><td>0.422</td><td>0.555</td><td>0.355</td><td>0.429</td><td>0.503</td><td>0.361</td><td>0.425</td><td>0.574</td></tr><tr><td>相关性</td><td>平均绝对误差（Mean Absolute Error，MAE）</td><td>0.372</td><td>0.419</td><td>0.496</td><td>0.392</td><td>0.430</td><td>0.484</td><td>0.406</td><td>0.440</td><td>0.534</td></tr><tr><td>完整</td><td>均方误差（Mean Squared Error，MSE）</td><td>0.375</td><td>0.537</td><td>0.667</td><td>0.450</td><td>0.554</td><td>-</td><td>0.501</td><td>0.647</td><td>-</td></tr><tr><td>全注意力机制 [41]</td><td>平均绝对误差（Mean Absolute Error，MAE）</td><td>0.425</td><td>0.502</td><td>0.589</td><td>0.470</td><td>0.533</td><td>-</td><td>0.485</td><td>0.491</td><td>-</td></tr><tr><td>对数稀疏</td><td>均方误差（Mean Squared Error，MSE）</td><td>0.362</td><td>0.539</td><td>0.582</td><td>0.420</td><td>0.552</td><td>0.958</td><td>0.474</td><td>0.601</td><td>-</td></tr><tr><td>注意力机制26</td><td>平均绝对误差（Mean Absolute Error，MAE）</td><td>0.413</td><td>0.522</td><td>0.529</td><td>0.450</td><td>0.513</td><td>0.736</td><td>0.474</td><td>0.524</td><td>-</td></tr><tr><td>局部敏感哈希（LSH）</td><td>均方误差（Mean Squared Error，MSE）</td><td>0.366</td><td>0.502</td><td>0.663</td><td>0.407</td><td>0.636</td><td>1.069</td><td>0.442</td><td>0.615</td><td>-</td></tr><tr><td>注意力机制23</td><td>平均绝对误差（Mean Absolute Error，MAE）</td><td>0.404</td><td>0.475</td><td>0.567</td><td>0.421</td><td>0.571</td><td>0.756</td><td>0.476</td><td>0.532</td><td>-</td></tr><tr><td>概率稀疏（ProbSparse）</td><td>均方误差（Mean Squared Error，MSE）</td><td>0.481</td><td>0.822</td><td>0.715</td><td>0.404</td><td>1.148</td><td>0.732</td><td>0.417</td><td>0.631</td><td>1.133</td></tr><tr><td>注意力机制48</td><td>平均绝对误差（Mean Absolute Error，MAE）</td><td>0.472</td><td>0.559</td><td>0.586</td><td>0.425</td><td>0.654</td><td>0.602</td><td>0.434</td><td>0.528</td><td>0.691</td></tr></tbody></table>

<!-- Media -->

### 4.3 Model Analysis

### 4.3 模型分析

Time series decomposition As shown in Figure 4, without our series decomposition block, the forecasting model cannot capture the increasing trend and peaks of the seasonal part. By adding the series decomposition blocks, Autoformer can aggregate and refine the trend-cyclical part from series progressively. This design also facilitates the learning of the seasonal part, especially the peaks and troughs. This verifies the necessity of our proposed progressive decomposition architecture.

时间序列分解 如图4所示，如果没有我们的序列分解模块，预测模型无法捕捉到季节性部分的增长趋势和峰值。通过添加序列分解模块，Autoformer可以逐步从序列中聚合和提炼出趋势 - 周期部分。这种设计也有助于对季节性部分的学习，尤其是峰值和谷值。这验证了我们提出的渐进式分解架构的必要性。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_8.jpg?x=312&y=1421&w=1174&h=297&r=0"/>

Figure 4: Visualization of learned seasonal ${\mathcal{X}}_{\mathrm{{de}}}^{M}$ and trend-cyclical ${\mathcal{T}}_{\mathrm{{de}}}^{M}$ of the last decoder layer. We gradually add the decomposition blocks in decoder from left to right. This case is from ETT dataset under input-96-predict-720 setting. For clearness, we add the linear growth to raw data additionally.

图4：最后一个解码器层学习到的季节性 ${\mathcal{X}}_{\mathrm{{de}}}^{M}$ 和趋势 - 周期 ${\mathcal{T}}_{\mathrm{{de}}}^{M}$ 的可视化。我们从左到右逐步在解码器中添加分解模块。此案例来自ETT（电力变压器温度，Electric Transformer Temperature）数据集，输入为96个数据点、预测720个数据点的设置。为清晰起见，我们额外在原始数据中添加了线性增长。

<!-- Media -->

Dependencies learning The marked time delay sizes in Figure 5(a) indicate the most likely periods. Our learned periodicity can guide the model to aggregate the sub-series from the same or neighbor phase of periods by $\operatorname{Roll}\left( {\mathcal{X},{\tau }_{i}}\right) ,i \in  \{ 1,\cdots ,6\}$ . For the last time step (declining stage),AutoCorrelation fully utilizes all similar sub-series without omissions or errors compared to self-attentions. This verifies that Autoformer can discover the relevant information more sufficiently and precisely.

依赖关系学习 图5(a)中标记的时间延迟大小表示最可能的周期。我们学习到的周期性可以引导模型通过$\operatorname{Roll}\left( {\mathcal{X},{\tau }_{i}}\right) ,i \in  \{ 1,\cdots ,6\}$聚合来自相同或相邻周期相位的子序列。对于最后一个时间步（下降阶段），与自注意力机制相比，自相关机制能够充分利用所有相似的子序列，且不会出现遗漏或错误。这验证了Autoformer（自变压器）能够更充分、更精确地发现相关信息。

Complex seasonality modeling As shown in Figure 6, the lags that Autoformer learns from deep representations can indicate the real seasonality of raw series. For example, the learned lags of the

复杂季节性建模 如图6所示，Autoformer（自变压器）从深度表示中学习到的滞后可以指示原始序列的真实季节性。例如，所学习到的

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_9.jpg?x=307&y=202&w=1181&h=277&r=0"/>

Figure 5: Visualization of learned dependencies. For clearness, we select the top-6 time delay sizes ${\tau }_{1},\cdots ,{\tau }_{6}$ of Auto-Correlation and mark them in raw series (red lines). For self-attentions,top-6 similar points with respect to the last time step (red stars) are also marked by orange points.

图5：学习到的依赖关系的可视化。为清晰起见，我们选择自相关机制的前6个时间延迟大小${\tau }_{1},\cdots ,{\tau }_{6}$，并将它们标记在原始序列中（红色线条）。对于自注意力机制，相对于最后一个时间步的前6个相似点（红色星号）也用橙色点标记。

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_9.jpg?x=308&y=606&w=1177&h=260&r=0"/>

Figure 6: Statistics of learned lags. For each time series in the test set, we count the top 10 lags learned by decoder for the input-96-predict-336 task. Figure (a)-(d) are the density histograms.

图6：学习到的滞后值统计。对于测试集中的每个时间序列，我们统计了解码器在输入96步预测336步任务中学习到的前10个滞后值。图(a)-(d)是密度直方图。

<!-- Media -->

daily recorded Exchange dataset present the monthly, quarterly and yearly periods (Figure 6(b)). For the hourly recorded Traffic dataset (Figure 6(c)), the learned lags show the intervals as 24-hours and 168-hours, which match the daily and weekly periods of real-world scenarios. These results show that Autoformer can capture the complex seasonalities of real-world series from deep representations and further provide a human-interpretable prediction.

每日记录的汇率数据集呈现出月度、季度和年度周期（图6(b)）。对于每小时记录的交通数据集（图6(c)），学习到的滞后值显示间隔为24小时和168小时，这与现实场景中的每日和每周周期相匹配。这些结果表明，Autoformer可以从深度表示中捕捉现实世界序列的复杂季节性，并进一步提供可由人类解释的预测。

Efficiency analysis We compare the running memory and time among Auto-Correlation-based and self-attention-based models (Figure 7) during the training phase. The proposed Autoformer shows $\mathcal{O}\left( {L\log L}\right)$ complexity in both memory and time and achieves better long-term sequences efficiency.

效率分析 我们在训练阶段比较了基于自相关和基于自注意力的模型之间的运行内存和时间（图7）。所提出的Autoformer在内存和时间上均显示出$\mathcal{O}\left( {L\log L}\right)$复杂度，并实现了更好的长期序列效率。

<!-- Media -->

<!-- figureText: 15 $\rightarrow$ Full Attention From Transformer $\rightarrow$ LSH Attention From Reformer 1024 4096 (b) Running Time Efficiency Analysis - Auto-Correlation From Autoformer $\leftarrow$ Full Attention From Transformer - ProbSparse Attention From Informer 384 768 1536 Output Length (a) Memory Efficiency Analysis -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_9.jpg?x=307&y=1299&w=1181&h=351&r=0"/>

Figure 7: Efficiency Analysis. For memory, we replace Auto-Correlation with self-attention family in Autoformer and record the memory with input 96. For running time, we run the Auto-Correlation or self-attentions ${10}^{3}$ times to get the execution time per step. The output length increases exponentially.

图7：效率分析。对于内存，我们在自相关变换器（Autoformer）中将自相关（Auto-Correlation）替换为自注意力（self-attention）族，并记录输入为96时的内存使用情况。对于运行时间，我们将自相关或自注意力机制运行${10}^{3}$次以获得每一步的执行时间。输出长度呈指数级增长。

<!-- Media -->

## 5 Conclusions

## 5 结论

This paper studies the long-term forecasting problem of time series, which is a pressing demand for real-world applications. However, the intricate temporal patterns prevent the model from learning reliable dependencies. We propose the Autoformer as a decomposition architecture by embedding the series decomposition block as an inner operator, which can progressively aggregate the long-term trend part from intermediate prediction. Besides, we design an efficient Auto-Correlation mechanism to conduct dependencies discovery and information aggregation at the series level, which contrasts clearly from the previous self-attention family. Autoformer can naturally achieve $\mathcal{O}\left( {L\log L}\right)$ complexity and yield consistent state-of-the-art performance in extensive real-world datasets.

本文研究了时间序列的长期预测问题，这是现实世界应用中的迫切需求。然而，复杂的时间模式阻碍了模型学习可靠的依赖关系。我们提出了自相关变换器（Autoformer）作为一种分解架构，将序列分解模块嵌入为内部算子，该架构可以从中间预测中逐步聚合长期趋势部分。此外，我们设计了一种高效的自相关机制，用于在序列层面进行依赖关系发现和信息聚合，这与以往的自注意力族有明显区别。自相关变换器（Autoformer）自然可以达到$\mathcal{O}\left( {L\log L}\right)$复杂度，并在大量现实世界数据集上取得一致的最先进性能。

## Acknowledgments and Disclosure of Funding

## 致谢与资金披露

This work was supported by the National Natural Science Foundation of China under Grants 62022050 and 62021002, Beijing Nova Program under Grant Z201100006820041, China's Ministry of Industry and Information Technology, the MOE Innovation Plan and the BNRist Innovation Fund.

本工作得到了国家自然科学基金（项目编号：62022050、62021002）、北京市科技新星计划（项目编号：Z201100006820041）、中国工业和信息化部、教育部创新计划以及北京智源人工智能研究院（BNRist）创新基金的资助。

## References

## 参考文献

[1] O. Anderson and M. Kendall. Time-series. 2nd edn. J. R. Stat. Soc. (Series D), 1976.

[1] O. 安德森（O. Anderson）和 M. 肯德尔（M. Kendall）。《时间序列》（Time-series），第 2 版。《皇家统计学会杂志》（Journal of the Royal Statistical Society）（D 辑），1976 年。

[2] Reza Asadi and Amelia C Regan. A spatio-temporal decomposition based deep neural network for time series forecasting. Appl. Soft Comput., 2020.

[2] 礼萨·阿萨迪（Reza Asadi）和阿米莉亚·C·里根（Amelia C Regan）。基于时空分解的深度神经网络用于时间序列预测。《应用软计算》（Applied Soft Computing），2020 年。

[3] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. ICLR, 2015.

[4] 德米特里·巴达纳乌（Dzmitry Bahdanau）、郑京勋（Kyunghyun Cho）和约书亚·本吉奥（Yoshua Bengio）。通过联合学习对齐和翻译进行神经机器翻译。国际学习表征会议（ICLR），2015 年。

[4] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018.

[5] 白少杰（Shaojie Bai）、J·齐科·科尔特（J Zico Kolter）和弗拉德连·科尔图恩（Vladlen Koltun）。用于序列建模的通用卷积和循环网络的实证评估。预印本 arXiv:1803.01271，2018 年。

[5] Anastasia Borovykh, Sander Bohte, and Cornelis W Oosterlee. Conditional time series forecasting with convolutional neural networks. arXiv preprint arXiv:1703.04691, 2017.

[5] 阿纳斯塔西娅·博罗维赫（Anastasia Borovykh）、桑德·博赫特（Sander Bohte）和科内利斯·W·奥斯特利（Cornelis W Oosterlee）。基于卷积神经网络的条件时间序列预测。预印本arXiv:1703.04691，2017年。

[6] G. E. P. Box and Gwilym M. Jenkins. Time series analysis, forecasting and control. 1970.

[6] G. E. P. 博克斯（G. E. P. Box）和格威利姆·M·詹金斯（Gwilym M. Jenkins）。时间序列分析、预测与控制。1970年。

[7] George EP Box and Gwilym M Jenkins. Some recent advances in forecasting and control. J. R. Stat. Soc. (Series-C), 1968.

[7] 乔治·E·P·博克斯（George EP Box）和格威利姆·M·詹金斯（Gwilym M Jenkins）。预测与控制的一些最新进展。《皇家统计学会杂志》（C辑），1968年。

[8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS, 2020.

[8] 汤姆·布朗（Tom Brown）、本杰明·曼（Benjamin Mann）、尼克·赖德（Nick Ryder）、梅兰妮·苏比亚（Melanie Subbiah）、贾里德·D·卡普兰（Jared D Kaplan）、普拉富拉·达里瓦尔（Prafulla Dhariwal）、阿温德·尼尔坎坦（Arvind Neelakantan）、普拉纳夫·夏姆（Pranav Shyam）、吉里什·萨斯特里（Girish Sastry）、阿曼达·阿斯凯尔（Amanda Askell）、桑迪尼·阿加瓦尔（Sandhini Agarwal）、阿里尔·赫伯特 - 沃斯（Ariel Herbert-Voss）、格雷琴·克鲁格（Gretchen Krueger）、汤姆·海宁汉（Tom Henighan）、雷翁·蔡尔德（Rewon Child）、阿迪亚·拉梅什（Aditya Ramesh）、丹尼尔·齐格勒（Daniel Ziegler）、杰弗里·吴（Jeffrey Wu）、克莱门斯·温特（Clemens Winter）、克里斯·赫斯（Chris Hesse）、马克·陈（Mark Chen）、埃里克·西格勒（Eric Sigler）、马泰乌什·利特温（Mateusz Litwin）、斯科特·格雷（Scott Gray）、本杰明·切斯（Benjamin Chess）、杰克·克拉克（Jack Clark）、克里斯托弗·伯纳尔（Christopher Berner）、山姆·麦坎德利什（Sam McCandlish）、亚历克·拉德福德（Alec Radford）、伊利亚·苏茨克维（Ilya Sutskever）和达里奥·阿莫迪（Dario Amodei）。语言模型是少样本学习者。发表于《神经信息处理系统大会论文集》（NeurIPS），2020年。

[9] Chris Chatfield. The analysis of time series: an introduction. 1981.

[9] 克里斯·查特菲尔德（Chris Chatfield）。《时间序列分析导论》。1981年。

[10] Renyi Chen and Molei Tao. Data-driven prediction of general hamiltonian dynamics via learning exactly-symplectic maps. ICML, 2021.

[10] 陈仁毅（Renyi Chen）和陶茉莉（Molei Tao）。通过学习精确辛映射实现通用哈密顿动力学的数据驱动预测。发表于《国际机器学习会议论文集》（ICML），2021年。

[11] Lawrence J Christiano and Terry J Fitzgerald. The band pass filter. Int. Econ. Rev., 2003.

[11] 劳伦斯·J·克里斯蒂亚诺（Lawrence J Christiano）和特里·J·菲茨杰拉德（Terry J Fitzgerald）。带通滤波器。发表于《国际经济评论》（Int. Econ. Rev.），2003年。

[12] Emmanuel de Bézenac, Syama Sundar Rangapuram, Konstantinos Benidis, Michael Bohlke-Schneider, Richard Kurle, Lorenzo Stella, Hilaf Hasson, Patrick Gallinari, and Tim Januschowski. Normalizing kalman filters for multivariate time series analysis. In NeurIPS, 2020.

[12] 伊曼纽尔·德·贝泽纳克（Emmanuel de Bézenac）、西亚马·桑达尔·兰加普拉姆（Syama Sundar Rangapuram）、康斯坦丁诺斯·贝尼迪迪斯（Konstantinos Benidis）、迈克尔·博尔克 - 施奈德（Michael Bohlke-Schneider）、理查德·库尔勒（Richard Kurle）、洛伦佐·斯特拉（Lorenzo Stella）、希拉夫·哈森（Hilaf Hasson）、帕特里克·加利纳里（Patrick Gallinari）和蒂姆·亚努绍夫斯基（Tim Januschowski）。用于多变量时间序列分析的归一化卡尔曼滤波器。发表于《神经信息处理系统大会》（NeurIPS），2020年。

[13] J. Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, 2019.

[13] J. 德夫林（J. Devlin）、张明伟（Ming-Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。BERT：用于语言理解的深度双向变换器的预训练。发表于《北美计算语言学协会 - 人机语言技术会议》（NAACL-HLT），2019年。

[14] Francis X Diebold and Lutz Kilian. Measuring predictability: theory and macroeconomic applications. J. Appl. Econom., 2001.

[14] 弗朗西斯·X·迪博尔德（Francis X Diebold）和卢茨·基利安（Lutz Kilian）。衡量可预测性：理论与宏观经济应用。《应用计量经济学杂志》（J. Appl. Econom.），2001年。

[15] E. Dong, H. Du, and L. Gardner. An interactive web-based dashboard to track covid-19 in real time. Lancet Infect. Dis., 2020.

[15] 董恩（E. Dong）、杜航（H. Du）和利·加德纳（L. Gardner）。一个基于网络的交互式仪表盘，用于实时跟踪新冠疫情。《柳叶刀·传染病》（Lancet Infect. Dis.），2020年。

[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth ${16} \times  {16}$ words: Transformers for image recognition at scale. In ${ICLR}$ , 2021.

[16] 阿列克谢·多索维茨基（Alexey Dosovitskiy）、卢卡斯·拜尔（Lucas Beyer）、亚历山大·科列斯尼科夫（Alexander Kolesnikov）、德克·魏森伯恩（Dirk Weissenborn）、翟晓华（Xiaohua Zhai）、托马斯·翁特希纳（Thomas Unterthiner）、穆斯塔法·德赫加尼（Mostafa Dehghani）、马蒂亚斯·明德勒（Matthias Minderer）、格奥尔格·海戈尔（Georg Heigold）、西尔万·热利（Sylvain Gelly）、雅各布·乌斯考雷特（Jakob Uszkoreit）和尼尔·霍尔斯比（Neil Houlsby）。一张图像胜过${16} \times  {16}$ 个字：大规模图像识别的Transformer模型。发表于${ICLR}$ ，2021年。

[17] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Comput., 1997.

[17] S. 霍赫赖特（S. Hochreiter）和J. 施密德胡伯（J. Schmidhuber）。长短期记忆网络。《神经计算》，1997年。

[18] Robert J Hodrick and Edward C Prescott. Postwar us business cycles: an empirical investigation. J. Money Credit Bank., 1997.

[18] 罗伯特·J·霍德里克（Robert J Hodrick）和爱德华·C·普雷斯科特（Edward C Prescott）。战后美国商业周期：一项实证研究。《货币、信贷与银行期刊》，1997年。

[19] Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Ian Simon, Curtis Hawthorne, Noam Shazeer, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, and Douglas Eck. Music transformer. In ICLR, 2019.

[19] 黄程志（Cheng-Zhi Anna Huang）、阿什什·瓦斯瓦尼（Ashish Vaswani）、雅各布·乌斯考雷特（Jakob Uszkoreit）、伊恩·西蒙（Ian Simon）、柯蒂斯·霍索恩（Curtis Hawthorne）、诺姆·沙泽尔（Noam Shazeer）、安德鲁·M·戴（Andrew M. Dai）、马修·D·霍夫曼（Matthew D. Hoffman）、莫妮卡·丁库列斯库（Monica Dinculescu）和道格拉斯·埃克（Douglas Eck）。音乐Transformer模型。发表于国际学习表征会议（ICLR），2019年。

[20] Rob J Hyndman and George Athanasopoulos. Forecasting: principles and practice. 2018.

[20] 罗布·J·海恩德曼（Rob J Hyndman）和乔治·阿萨纳西奥普洛斯（George Athanasopoulos）。《预测：原理与实践》。2018年。

[21] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.

[21] 谢尔盖·约菲（Sergey Ioffe）和克里斯蒂安·塞格迪（Christian Szegedy）。《批量归一化：通过减少内部协变量偏移加速深度网络训练》。发表于2015年国际机器学习会议（ICML）。

[22] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

[22] 迪德里克·P·金马（Diederik P. Kingma）和吉米·巴（Jimmy Ba）。《Adam：一种随机优化方法》。发表于2015年国际学习表征会议（ICLR）。

[23] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In ICLR, 2020.

[23] 尼基塔·基塔耶夫（Nikita Kitaev）、卢卡斯·凯泽（Lukasz Kaiser）和安塞尔姆·列夫斯卡亚（Anselm Levskaya）。《改革者：高效的Transformer》。发表于2020年国际学习表征会议（ICLR）。

[24] Richard Kurle, Syama Sundar Rangapuram, Emmanuel de Bézenac, Stephan Günnemann, and Jan Gasthaus. Deep rao-blackwellised particle filters for time series forecasting. In NeurIPS, 2020.

[24] 理查德·库尔勒（Richard Kurle）、西亚玛·桑达尔·兰加普拉姆（Syama Sundar Rangapuram）、伊曼纽尔·德·贝泽纳克（Emmanuel de Bézenac）、斯蒂芬·居内曼（Stephan Günnemann）和扬·加施豪斯（Jan Gasthaus）。《用于时间序列预测的深度 Rao - Blackwell 化粒子滤波器》。发表于2020年神经信息处理系统大会（NeurIPS）。

[25] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In SIGIR, 2018.

[25] 赖国坤（Guokun Lai）、张维正（Wei - Cheng Chang）、杨一鸣（Yiming Yang）和刘瀚霄（Hanxiao Liu）。《使用深度神经网络对长期和短期时间模式进行建模》。发表于2018年国际信息检索研究与发展会议（SIGIR）。

[26] Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. In NeurIPS, 2019.

[26] 李诗阳（Shiyang Li）、金晓勇（Xiaoyong Jin）、宣姚（Yao Xuan）、周西有（Xiyou Zhou）、陈文虎（Wenhu Chen）、王宇翔（Yu-Xiang Wang）和闫锡峰（Xifeng Yan）。增强Transformer在时间序列预测中的局部性并突破内存瓶颈。发表于《神经信息处理系统大会》（NeurIPS），2019年。

[27] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ${ICCV},{2021}$ .

[27] 刘泽（Ze Liu）、林雨桐（Yutong Lin）、曹越（Yue Cao）、胡瀚（Han Hu）、魏一轩（Yixuan Wei）、张政（Zheng Zhang）、林Stephen（Stephen Lin）和郭百宁（Baining Guo）。Swin Transformer：使用移位窗口的分层视觉Transformer。发表于${ICCV},{2021}$。

[28] Danielle C Maddix, Yuyang Wang, and Alex Smola. Deep factors with gaussian processes for forecasting. arXiv preprint arXiv:1812.00098, 2018.

[28] 丹妮尔·C·麦迪克斯（Danielle C Maddix）、王宇阳（Yuyang Wang）和亚历克斯·斯莫拉（Alex Smola）。用于预测的带高斯过程的深度因子。预印本arXiv:1812.00098，2018年。

[29] Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. ICLR, 2019.

[29] 鲍里斯·N·奥列什金（Boris N Oreshkin）、德米特里·卡尔波夫（Dmitri Carpov）、尼古拉斯·查帕多斯（Nicolas Chapados）和约书亚·本吉奥（Yoshua Bengio）。N - BEATS：用于可解释时间序列预测的神经基扩展分析。发表于《国际学习表征会议》（ICLR），2019年。

[30] Athanasios Papoulis and H Saunders. Probability, random variables and stochastic processes. 1989.

[30] 阿萨纳西奥斯·帕普利斯（Athanasios Papoulis）和H·桑德斯（H Saunders）。《概率、随机变量和随机过程》。1989年。

[31] Adam Paszke, S. Gross, Francisco Massa, A. Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Z. Lin, N. Gimelshein, L. Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019.

[31] 亚当·帕斯凯（Adam Paszke）、S. 格罗斯（S. Gross）、弗朗西斯科·马萨（Francisco Massa）、A. 勒雷尔（A. Lerer）、詹姆斯·布拉德伯里（James Bradbury）、格雷戈里·查南（Gregory Chanan）、特雷弗·基林（Trevor Killeen）、Z. 林（Z. Lin）、N. 吉梅尔申（N. Gimelshein）、L. 安蒂加（L. Antiga）、阿尔班·德斯梅森（Alban Desmaison）、安德里亚斯·科普夫（Andreas Köpf）、爱德华·杨（Edward Yang）、扎克·德维托（Zach DeVito）、马丁·赖森（Martin Raison）、阿利汗·特贾尼（Alykhan Tejani）、萨桑克·奇拉姆库尔蒂（Sasank Chilamkurthy）、贝努瓦·施泰纳（Benoit Steiner）、方璐（Lu Fang）、白俊杰（Junjie Bai）和苏米思·钦塔拉（Soumith Chintala）。PyTorch：一种命令式风格的高性能深度学习库。发表于《神经信息处理系统大会论文集》（NeurIPS），2019 年。

[32] Syama Sundar Rangapuram, Matthias W Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. In NeurIPS, 2018.

[32] 西亚马·桑达尔·兰加普拉姆（Syama Sundar Rangapuram）、马蒂亚斯·W·西格（Matthias W Seeger）、扬·加施豪斯（Jan Gasthaus）、洛伦佐·斯特拉（Lorenzo Stella）、王宇阳（Yuyang Wang）和蒂姆·亚努绍夫斯基（Tim Januschowski）。用于时间序列预测的深度状态空间模型。发表于《神经信息处理系统大会论文集》（NeurIPS），2018 年。

[33] Cleveland Robert, C William, and Terpenning Irma. STL: A seasonal-trend decomposition procedure based on loess. J. Off. Stat, 1990.

[33] 克利夫兰·罗伯特（Cleveland Robert）、C·威廉（C William）和特尔彭宁·厄玛（Terpenning Irma）。STL：一种基于局部加权回归散点平滑法（loess）的季节性 - 趋势分解程序。《官方统计杂志》（J. Off. Stat），1990 年。

[34] Davide Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. Int. J. Forecast., 2020.

[34] 达维德·萨利纳斯（Davide Salinas）、瓦伦丁·弗伦克特（Valentin Flunkert）、扬·加施豪斯（Jan Gasthaus）和蒂姆·亚努绍夫斯基（Tim Januschowski）。《DeepAR：基于自回归循环网络的概率预测》，《国际预测期刊》（Int. J. Forecast.），2020年。

[35] Rajat Sen, Hsiang-Fu Yu, and Inderjit S. Dhillon. Think globally, act locally: A deep neural network approach to high-dimensional time series forecasting. In NeurIPS, 2019.

[35] 拉贾特·森（Rajat Sen）、余翔富（Hsiang-Fu Yu）和英德吉特·S·狄龙（Inderjit S. Dhillon）。《全局思考，局部行动：一种用于高维时间序列预测的深度神经网络方法》，发表于神经信息处理系统大会（NeurIPS），2019年。

[36] Shun-Yao Shih, Fan-Keng Sun, and Hung-yi Lee. Temporal pattern attention for multivariate time series forecasting. Mach. Learn., 2019.

[36] 施顺耀（Shun-Yao Shih）、孙凡耕（Fan-Keng Sun）和李宏毅（Hung-yi Lee）。《用于多变量时间序列预测的时间模式注意力机制》，《机器学习》（Mach. Learn.），2019年。

[37] Huan Song, Deepta Rajan, Jayaraman Thiagarajan, and Andreas Spanias. Attend and diagnose: Clinical time series analysis using attention models. In ${AAAI},{2018}$ .

[37] 宋欢（Huan Song）、迪普塔·拉詹（Deepta Rajan）、贾亚拉曼·蒂亚加拉扬（Jayaraman Thiagarajan）和安德里亚斯·斯帕尼亚斯（Andreas Spanias）。《关注与诊断：使用注意力模型进行临床时间序列分析》，发表于${AAAI},{2018}$ 。

[38] Antti Sorjamaa, Jin Hao, Nima Reyhani, Yongnan Ji, and Amaury Lendasse. Methodology for long-term prediction of time series. Neurocomputing, 2007.

[38] 安蒂·索尔贾马（Antti Sorjamaa）、郝进（Jin Hao）、尼玛·雷哈尼（Nima Reyhani）、纪永楠（Yongnan Ji）和阿莫里·伦达塞（Amaury Lendasse）。《时间序列长期预测方法》，《神经计算》（Neurocomputing），2007年。

[39] Sean J Taylor and Benjamin Letham. Forecasting at scale. Am. Stat., 2018.

[39] 肖恩·J·泰勒（Sean J Taylor）和本杰明·莱瑟姆（Benjamin Letham）。大规模预测。《美国统计学家》（Am. Stat.），2018年。

[40] Aäron van den Oord, S. Dieleman, H. Zen, K. Simonyan, Oriol Vinyals, A. Graves, Nal Kalchbrenner, A. Senior, and K. Kavukcuoglu. Wavenet: A generative model for raw audio. In SSW, 2016.

[40] 阿隆·范登·奥尔德（Aäron van den Oord）、S·迪莱曼（S. Dieleman）、H·曾（H. Zen）、K·西蒙扬（K. Simonyan）、奥里奥尔·维尼亚尔斯（Oriol Vinyals）、A·格雷夫斯（A. Graves）、纳尔·卡尔赫布伦纳（Nal Kalchbrenner）、A·西尼尔（A. Senior）和K·卡武库奥卢（K. Kavukcuoglu）。波形网络：原始音频的生成模型。发表于2016年的SSW会议。

[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

[41] 阿什什·瓦斯瓦尼（Ashish Vaswani）、诺姆·沙泽尔（Noam Shazeer）、尼基·帕尔马（Niki Parmar）、雅各布·乌斯库雷特（Jakob Uszkoreit）、利昂·琼斯（Llion Jones）、艾丹·N·戈麦斯（Aidan N Gomez）、卢卡斯·凯泽（Łukasz Kaiser）和伊利亚·波洛苏金（Illia Polosukhin）。注意力就是你所需要的一切。发表于2017年的神经信息处理系统大会（NeurIPS）。

[42] Ruofeng Wen, Kari Torkkola, Balakrishnan Narayanaswamy, and Dhruv Madeka. A multi-horizon quantile recurrent forecaster. NeurIPS, 2017.

[42] 文若峰（Ruofeng Wen）、卡里·托尔科拉（Kari Torkkola）、巴拉卡什南·纳拉亚纳斯瓦米（Balakrishnan Narayanaswamy）和德鲁夫·马德卡（Dhruv Madeka）。多步分位数循环预测器。发表于2017年的神经信息处理系统大会（NeurIPS）。

[43] Norbert Wiener. Generalized harmonic analysis. Acta Math, 1930.

[43] 诺伯特·维纳（Norbert Wiener）。广义调和分析。《数学学报》（Acta Math），1930年。

[44] Ulrich Woitek. A note on the baxter-king filter. 1998.

[44] 乌尔里希·沃泰克（Ulrich Woitek）。关于巴克斯特 - 金滤波器（baxter - king filter）的一则笔记。1998年。

[45] Sifan Wu, Xi Xiao, Qianggang Ding, Peilin Zhao, Ying Wei, and Junzhou Huang. Adversarial sparse transformer for time series forecasting. In NeurIPS, 2020.

[45] 吴思凡（Sifan Wu）、肖曦（Xi Xiao）、丁强刚（Qianggang Ding）、赵佩琳（Peilin Zhao）、魏莹（Ying Wei）和黄军洲（Junzhou Huang）。用于时间序列预测的对抗性稀疏变压器。发表于神经信息处理系统大会（NeurIPS），2020年。

[46] Q. Yao, D. Song, H. Chen, C. Wei, and G. W. Cottrell. A dual-stage attention-based recurrent neural network for time series prediction. In IJCAI, 2017.

[46] 姚Q.（Q. Yao）、宋D.（D. Song）、陈H.（H. Chen）、魏C.（C. Wei）和科特雷尔G. W.（G. W. Cottrell）。一种基于双阶段注意力的循环神经网络用于时间序列预测。发表于《国际人工智能联合会议》（IJCAI），2017年。

[47] Rose Yu, Stephan Zheng, Anima Anandkumar, and Yisong Yue. Long-term forecasting using tensor-train rnns. arXiv preprint arXiv:1711.00073, 2017.

[47] 于罗斯（Rose Yu）、郑斯蒂芬（Stephan Zheng）、阿南德库马尔阿尼马（Anima Anandkumar）和岳亦松（Yisong Yue）。使用张量列车循环神经网络进行长期预测。预印本arXiv:1711.00073，2017年。

[48] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In AAAI, 2021.

[48] 周浩毅（Haoyi Zhou）、张上航（Shanghang Zhang）、彭杰奇（Jieqi Peng）、张帅（Shuai Zhang）、李建新（Jianxin Li）、熊辉（Hui Xiong）和张万财（Wancai Zhang）。Informer：超越高效Transformer的长序列时间序列预测方法。发表于《美国人工智能协会会议》（AAAI），2021年。

## A Full Benchmark on the ETT Datasets

## 电力变压器温度（ETT）数据集的完整基准测试

As shown in Table 5 we build the benchmark on the four ETT datasets [48], which includes the hourly recorded ETTh1 and ETTh2, 15-minutely recorded ETTm1 and ETTm2.

如表5所示，我们在四个ETT数据集[48]上构建了基准测试，其中包括每小时记录的ETTh1和ETTh2，以及每15分钟记录的ETTm1和ETTm2。

Autoformer achieves sharp improvement over the state-of-the-art on various forecasting horizons. For the input-96-predict-336 long-term setting,Autoformer surpasses previous best results by ${55}\% \left( {{1.128} \rightarrow  {0.505}}\right)$ in ETTh1,80% (2.544 $\rightarrow  {0.471}$ ) in ETTh2. For the input-96-predict-288 long-term setting,Autoformer achieves ${40}\% \left( {{1.056} \rightarrow  {0.634}}\right)$ MSE reduction in ETTm1 and ${66}\% \left( {{0.969} \rightarrow  {0.342}}\right)$ in ETTm2. These results show a ${60}\%$ average MSE reduction over previous state-of-the-art.

Autoformer（自动变换器）在各种预测时程上相较于现有最优方法取得了显著改进。在输入96个时间步长并预测336个时间步长的长期设置下，Autoformer在ETTh1数据集上超过了之前的最佳结果${55}\% \left( {{1.128} \rightarrow  {0.505}}\right)$，在ETTh2数据集上超过了80%（2.544 $\rightarrow  {0.471}$ ）。在输入96个时间步长并预测288个时间步长的长期设置下，Autoformer在ETTm1数据集上实现了${40}\% \left( {{1.056} \rightarrow  {0.634}}\right)$的均方误差（MSE）降低，在ETTm2数据集上实现了${66}\% \left( {{0.969} \rightarrow  {0.342}}\right)$的均方误差降低。这些结果表明，相较于之前的最优方法，平均均方误差降低了${60}\%$。

<!-- Media -->

Table 5: Multivariate results on the four ETT datasets with predicted length as $\{ {24},{48},{168},{288},{336}$ , ${672},{720}\}$ . We fix the input length of Autoformer as 96 . The experiments of the main text are on the ETTm2 dataset.

表5：四个ETT数据集上以$\{ {24},{48},{168},{288},{336}$、${672},{720}\}$为预测长度的多变量结果。我们将Autoformer的输入长度固定为96。正文的实验是在ETTm2数据集上进行的。

<table><tr><td colspan="2">Models</td><td colspan="2">Autoformer</td><td colspan="2">Informer [48]</td><td colspan="2">LogTrans [26]</td><td colspan="2">Reformer [23]</td><td colspan="2">LSTNet [25]</td><td colspan="2">LSTMa [3]</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">ETTh1</td><td>24</td><td>0.384</td><td>0.425</td><td>0.577</td><td>0.549</td><td>0.686</td><td>0.604</td><td>0.991</td><td>0.754</td><td>1.293</td><td>0.901</td><td>0.650</td><td>0.624</td></tr><tr><td>48</td><td>0.392</td><td>0.419</td><td>0.685</td><td>0.625</td><td>0.766</td><td>0.757</td><td>1.313</td><td>0.906</td><td>1.456</td><td>0.960</td><td>0.702</td><td>0.675</td></tr><tr><td>168</td><td>0.490</td><td>0.481</td><td>0.931</td><td>0.752</td><td>1.002</td><td>0.846</td><td>1.824</td><td>1.138</td><td>1.997</td><td>1.214</td><td>1.212</td><td>0.867</td></tr><tr><td>336</td><td>0.505</td><td>0.484</td><td>1.128</td><td>0.873</td><td>1.362</td><td>0.952</td><td>2.117</td><td>1.280</td><td>2.655</td><td>1.369</td><td>1.424</td><td>0.994</td></tr><tr><td>720</td><td>0.498</td><td>0.500</td><td>1.215</td><td>0.896</td><td>1.397</td><td>1.291</td><td>2.415</td><td>1.520</td><td>2.143</td><td>1.380</td><td>1.960</td><td>1.322</td></tr><tr><td rowspan="5">ETTh2</td><td>24</td><td>0.261</td><td>0.341</td><td>0.720</td><td>0.665</td><td>0.828</td><td>0.750</td><td>1.531</td><td>1.613</td><td>2.742</td><td>1.457</td><td>1.143</td><td>0.813</td></tr><tr><td>48</td><td>0.312</td><td>0.373</td><td>1.457</td><td>1.001</td><td>1.806</td><td>1.034</td><td>1.871</td><td>1.735</td><td>3.567</td><td>1.687</td><td>1.671</td><td>1.221</td></tr><tr><td>168</td><td>0.457</td><td>0.455</td><td>3.489</td><td>1.515</td><td>4.070</td><td>1.681</td><td>4.660</td><td>1.846</td><td>3.242</td><td>2.513</td><td>4.117</td><td>1.674</td></tr><tr><td>336</td><td>0.471</td><td>0.475</td><td>2.723</td><td>1.340</td><td>3.875</td><td>1.763</td><td>4.028</td><td>1.688</td><td>2.544</td><td>2.591</td><td>3.434</td><td>1.549</td></tr><tr><td>720</td><td>0.474</td><td>0.484</td><td>3.467</td><td>1.473</td><td>3.913</td><td>1.552</td><td>5.381</td><td>2.015</td><td>4.625</td><td>3.709</td><td>3.963</td><td>1.788</td></tr><tr><td rowspan="5">ETTm1</td><td>24</td><td>0.383</td><td>0.403</td><td>0.323</td><td>0.369</td><td>0.419</td><td>0.412</td><td>0.724</td><td>0.607</td><td>1.968</td><td>1.170</td><td>0.621</td><td>0.629</td></tr><tr><td>48</td><td>0.454</td><td>0.453</td><td>0.494</td><td>0.503</td><td>0.507</td><td>0.583</td><td>1.098</td><td>0.777</td><td>1.999</td><td>1.215</td><td>1.392</td><td>0.939</td></tr><tr><td>96</td><td>0.481</td><td>0.463</td><td>0.678</td><td>0.614</td><td>0.768</td><td>0.792</td><td>1.433</td><td>0.945</td><td>2.762</td><td>1.542</td><td>1.339</td><td>0.913</td></tr><tr><td>288</td><td>0.634</td><td>0.528</td><td>1.056</td><td>0.786</td><td>1.462</td><td>1.320</td><td>1.820</td><td>1.094</td><td>1.257</td><td>2.076</td><td>1.740</td><td>1.124</td></tr><tr><td>672</td><td>0.606</td><td>0.542</td><td>1.192</td><td>0.926</td><td>1.669</td><td>1.461</td><td>2.187</td><td>1.232</td><td>1.917</td><td>2.941</td><td>2.736</td><td>1.555</td></tr><tr><td rowspan="5">ETTm2</td><td>24</td><td>0.153</td><td>0.261</td><td>0.173</td><td>0.301</td><td>0.211</td><td>0.332</td><td>0.333</td><td>0.429</td><td>1.101</td><td>0.831</td><td>0.580</td><td>0.572</td></tr><tr><td>48</td><td>0.178</td><td>0.280</td><td>0.303</td><td>0.409</td><td>0.427</td><td>0.487</td><td>0.558</td><td>0.571</td><td>2.619</td><td>1.393</td><td>0.747</td><td>0.630</td></tr><tr><td>96</td><td>0.255</td><td>0.339</td><td>0.365</td><td>0.453</td><td>0.768</td><td>0.642</td><td>0.658</td><td>0.619</td><td>3.142</td><td>1.365</td><td>2.041</td><td>1.073</td></tr><tr><td>288</td><td>0.342</td><td>0.378</td><td>1.047</td><td>0.804</td><td>1.090</td><td>0.806</td><td>2.441</td><td>1.190</td><td>2.856</td><td>1.329</td><td>0.969</td><td>0.742</td></tr><tr><td>672</td><td>0.434</td><td>0.430</td><td>3.126</td><td>1.302</td><td>2.397</td><td>1.214</td><td>3.090</td><td>1.328</td><td>3.409</td><td>1.420</td><td>2.541</td><td>1.239</td></tr></table>

<table><tbody><tr><td colspan="2">模型</td><td colspan="2">自动变换器（Autoformer）</td><td colspan="2">信息器（Informer） [48]</td><td colspan="2">对数变换器（LogTrans） [26]</td><td colspan="2">改革者（Reformer） [23]</td><td colspan="2">长短期记忆网络（LSTNet） [25]</td><td colspan="2">长短期记忆网络a（LSTMa） [3]</td></tr><tr><td colspan="2">指标</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="5">ETTh1</td><td>24</td><td>0.384</td><td>0.425</td><td>0.577</td><td>0.549</td><td>0.686</td><td>0.604</td><td>0.991</td><td>0.754</td><td>1.293</td><td>0.901</td><td>0.650</td><td>0.624</td></tr><tr><td>48</td><td>0.392</td><td>0.419</td><td>0.685</td><td>0.625</td><td>0.766</td><td>0.757</td><td>1.313</td><td>0.906</td><td>1.456</td><td>0.960</td><td>0.702</td><td>0.675</td></tr><tr><td>168</td><td>0.490</td><td>0.481</td><td>0.931</td><td>0.752</td><td>1.002</td><td>0.846</td><td>1.824</td><td>1.138</td><td>1.997</td><td>1.214</td><td>1.212</td><td>0.867</td></tr><tr><td>336</td><td>0.505</td><td>0.484</td><td>1.128</td><td>0.873</td><td>1.362</td><td>0.952</td><td>2.117</td><td>1.280</td><td>2.655</td><td>1.369</td><td>1.424</td><td>0.994</td></tr><tr><td>720</td><td>0.498</td><td>0.500</td><td>1.215</td><td>0.896</td><td>1.397</td><td>1.291</td><td>2.415</td><td>1.520</td><td>2.143</td><td>1.380</td><td>1.960</td><td>1.322</td></tr><tr><td rowspan="5">ETTh2</td><td>24</td><td>0.261</td><td>0.341</td><td>0.720</td><td>0.665</td><td>0.828</td><td>0.750</td><td>1.531</td><td>1.613</td><td>2.742</td><td>1.457</td><td>1.143</td><td>0.813</td></tr><tr><td>48</td><td>0.312</td><td>0.373</td><td>1.457</td><td>1.001</td><td>1.806</td><td>1.034</td><td>1.871</td><td>1.735</td><td>3.567</td><td>1.687</td><td>1.671</td><td>1.221</td></tr><tr><td>168</td><td>0.457</td><td>0.455</td><td>3.489</td><td>1.515</td><td>4.070</td><td>1.681</td><td>4.660</td><td>1.846</td><td>3.242</td><td>2.513</td><td>4.117</td><td>1.674</td></tr><tr><td>336</td><td>0.471</td><td>0.475</td><td>2.723</td><td>1.340</td><td>3.875</td><td>1.763</td><td>4.028</td><td>1.688</td><td>2.544</td><td>2.591</td><td>3.434</td><td>1.549</td></tr><tr><td>720</td><td>0.474</td><td>0.484</td><td>3.467</td><td>1.473</td><td>3.913</td><td>1.552</td><td>5.381</td><td>2.015</td><td>4.625</td><td>3.709</td><td>3.963</td><td>1.788</td></tr><tr><td rowspan="5">ETTm1（ETTm1）</td><td>24</td><td>0.383</td><td>0.403</td><td>0.323</td><td>0.369</td><td>0.419</td><td>0.412</td><td>0.724</td><td>0.607</td><td>1.968</td><td>1.170</td><td>0.621</td><td>0.629</td></tr><tr><td>48</td><td>0.454</td><td>0.453</td><td>0.494</td><td>0.503</td><td>0.507</td><td>0.583</td><td>1.098</td><td>0.777</td><td>1.999</td><td>1.215</td><td>1.392</td><td>0.939</td></tr><tr><td>96</td><td>0.481</td><td>0.463</td><td>0.678</td><td>0.614</td><td>0.768</td><td>0.792</td><td>1.433</td><td>0.945</td><td>2.762</td><td>1.542</td><td>1.339</td><td>0.913</td></tr><tr><td>288</td><td>0.634</td><td>0.528</td><td>1.056</td><td>0.786</td><td>1.462</td><td>1.320</td><td>1.820</td><td>1.094</td><td>1.257</td><td>2.076</td><td>1.740</td><td>1.124</td></tr><tr><td>672</td><td>0.606</td><td>0.542</td><td>1.192</td><td>0.926</td><td>1.669</td><td>1.461</td><td>2.187</td><td>1.232</td><td>1.917</td><td>2.941</td><td>2.736</td><td>1.555</td></tr><tr><td rowspan="5">ETTm2（ETTm2）</td><td>24</td><td>0.153</td><td>0.261</td><td>0.173</td><td>0.301</td><td>0.211</td><td>0.332</td><td>0.333</td><td>0.429</td><td>1.101</td><td>0.831</td><td>0.580</td><td>0.572</td></tr><tr><td>48</td><td>0.178</td><td>0.280</td><td>0.303</td><td>0.409</td><td>0.427</td><td>0.487</td><td>0.558</td><td>0.571</td><td>2.619</td><td>1.393</td><td>0.747</td><td>0.630</td></tr><tr><td>96</td><td>0.255</td><td>0.339</td><td>0.365</td><td>0.453</td><td>0.768</td><td>0.642</td><td>0.658</td><td>0.619</td><td>3.142</td><td>1.365</td><td>2.041</td><td>1.073</td></tr><tr><td>288</td><td>0.342</td><td>0.378</td><td>1.047</td><td>0.804</td><td>1.090</td><td>0.806</td><td>2.441</td><td>1.190</td><td>2.856</td><td>1.329</td><td>0.969</td><td>0.742</td></tr><tr><td>672</td><td>0.434</td><td>0.430</td><td>3.126</td><td>1.302</td><td>2.397</td><td>1.214</td><td>3.090</td><td>1.328</td><td>3.409</td><td>1.420</td><td>2.541</td><td>1.239</td></tr></tbody></table>

<!-- Media -->

## B Hyper-Parameter Sensitivity

## B 超参数敏感性

As shown in Table 6 we can verify the model robustness with respect to hyper-parameter $c$ (Equation 6 in the main text). To trade-off performance and efficiency,we set $c$ to the range of 1 to 3 . It is also observed that datasets with obvious periodicity tend to have a large factor $c$ ,such as the ETT and Traffic datasets. For the ILI dataset without obvious periodicity, the larger factor may bring noises.

如表6所示，我们可以验证模型相对于超参数 $c$（正文公式6）的鲁棒性。为了在性能和效率之间进行权衡，我们将 $c$ 的范围设置为1到3。还可以观察到，具有明显周期性的数据集往往具有较大的因子 $c$，例如ETT（原文未明确含义，保留）和交通数据集。对于没有明显周期性的ILI（原文未明确含义，保留）数据集，较大的因子可能会引入噪声。

<!-- Media -->

Table 6: Autoformer performance under different choices of hyper-parameter $c$ in the AutoCorrelation mechanism. We adopt the forecasting setting as input-36-predict-48 for the ILI dataset and input-96-predict-336 for the other datasets.

表6：自相关机制中不同超参数 $c$ 选择下Autoformer（原文未明确含义，保留）的性能。对于ILI数据集，我们采用输入36步预测48步的预测设置；对于其他数据集，采用输入96步预测336步的预测设置。

<table><tr><td>Dataset</td><td colspan="2">ETT</td><td colspan="2">Electricity</td><td colspan="2">Exchange</td><td colspan="2">Traffic</td><td colspan="2">Weather</td><td colspan="2">ILI</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>$c = 1$</td><td>0.339</td><td>0.372</td><td>0.252</td><td>0.356</td><td>0.511</td><td>0.528</td><td>0.706</td><td>0.488</td><td>0.348</td><td>0.388</td><td>2.754</td><td>1.088</td></tr><tr><td>$c = 2$</td><td>0.363</td><td>0.389</td><td>0.224</td><td>0.332</td><td>0.511</td><td>0.528</td><td>0.673</td><td>0.418</td><td>0.358</td><td>0.390</td><td>2.641</td><td>1.072</td></tr><tr><td>$c = 3$</td><td>0.339</td><td>0.372</td><td>0.231</td><td>0.338</td><td>0.509</td><td>0.524</td><td>0.619</td><td>0.385</td><td>0.359</td><td>0.395</td><td>2.669</td><td>1.085</td></tr><tr><td>$c = 4$</td><td>0.336</td><td>0.369</td><td>0.232</td><td>0.341</td><td>0.513</td><td>0.527</td><td>0.607</td><td>0.378</td><td>0.349</td><td>0.388</td><td>3.041</td><td>1.178</td></tr><tr><td>$c = 5$</td><td>0.410</td><td>0.415</td><td>0.273</td><td>0.371</td><td>0.517</td><td>0.527</td><td>0.618</td><td>0.379</td><td>0.366</td><td>0.399</td><td>3.076</td><td>1.172</td></tr></table>

<table><tbody><tr><td>数据集</td><td colspan="2">ETT（原文未明确含义，保留英文）</td><td colspan="2">电力</td><td colspan="2">交换；交易</td><td colspan="2">交通</td><td colspan="2">天气</td><td colspan="2">流感样疾病（ILI）</td></tr><tr><td>指标</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td>$c = 1$</td><td>0.339</td><td>0.372</td><td>0.252</td><td>0.356</td><td>0.511</td><td>0.528</td><td>0.706</td><td>0.488</td><td>0.348</td><td>0.388</td><td>2.754</td><td>1.088</td></tr><tr><td>$c = 2$</td><td>0.363</td><td>0.389</td><td>0.224</td><td>0.332</td><td>0.511</td><td>0.528</td><td>0.673</td><td>0.418</td><td>0.358</td><td>0.390</td><td>2.641</td><td>1.072</td></tr><tr><td>$c = 3$</td><td>0.339</td><td>0.372</td><td>0.231</td><td>0.338</td><td>0.509</td><td>0.524</td><td>0.619</td><td>0.385</td><td>0.359</td><td>0.395</td><td>2.669</td><td>1.085</td></tr><tr><td>$c = 4$</td><td>0.336</td><td>0.369</td><td>0.232</td><td>0.341</td><td>0.513</td><td>0.527</td><td>0.607</td><td>0.378</td><td>0.349</td><td>0.388</td><td>3.041</td><td>1.178</td></tr><tr><td>$c = 5$</td><td>0.410</td><td>0.415</td><td>0.273</td><td>0.371</td><td>0.517</td><td>0.527</td><td>0.618</td><td>0.379</td><td>0.366</td><td>0.399</td><td>3.076</td><td>1.172</td></tr></tbody></table>

<!-- Media -->

## C Model Input Selection

## C 模型输入选择

### C.1 Input Length Selection

### C.1 输入长度选择

Because the forecasting horizon is always fixed upon the application's demand, we need to tune the input length in real-world applications. Our study shows that the relationship between input length and model performance is dataset-specific, so we need to select the model input based on the data characteristics. For example, for the ETT dataset with obvious periodicity, an input with length-96 is enough to provide enough information. But for the ILI dataset without obvious periodicity, the model needs longer inputs to discover more informative temporal dependencies. Thus, a longer input will provide a better performance in the ILI dataset.

由于预测范围通常取决于应用需求，因此在实际应用中我们需要调整输入长度。我们的研究表明，输入长度与模型性能之间的关系因数据集而异，所以我们需要根据数据特征来选择模型输入。例如，对于具有明显周期性的 ETT 数据集，长度为 96 的输入足以提供足够的信息。但对于没有明显周期性的 ILI 数据集，模型需要更长的输入来发现更具信息量的时间依赖关系。因此，在 ILI 数据集中，更长的输入将带来更好的性能。

<!-- Media -->

Table 7: Autoformer performance under different input lengths. We fix the forecasting horizon as 48 for ILI and 336 for the others. The input lengths $I$ of the ILI dataset are in the $\{ {24},{36},{48},{60}\}$ . And for the ETT and Exchange datasets,the input lengths $I$ are in the $\{ {96},{192},{336},{720}\}$ .

表 7：Autoformer 在不同输入长度下的性能。我们将 ILI 的预测范围固定为 48，其他数据集的预测范围固定为 336。ILI 数据集的输入长度 $I$ 在 $\{ {24},{36},{48},{60}\}$ 范围内。对于 ETT 和 Exchange 数据集，输入长度 $I$ 在 $\{ {96},{192},{336},{720}\}$ 范围内。

<table><tr><td>Dataset</td><td colspan="2">ETT</td><td colspan="2">Electricity</td><td>Dataset</td><td colspan="2">ILI</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>Metric</td><td>MSE</td><td>MAE</td></tr><tr><td>$I = {96}$</td><td>0.339</td><td>0.372</td><td>0.231</td><td>0.338</td><td>$I = {24}$</td><td>3.406</td><td>1.247</td></tr><tr><td>$I = {192}$</td><td>0.355</td><td>0.392</td><td>0.200</td><td>0.316</td><td>$I = {36}$</td><td>2.669</td><td>1.085</td></tr><tr><td>$I = {336}$</td><td>0.361</td><td>0.406</td><td>0.225</td><td>0.335</td><td>$I = {48}$</td><td>2.656</td><td>1.075</td></tr><tr><td>$I = {720}$</td><td>0.419</td><td>0.430</td><td>0.226</td><td>0.346</td><td>$I = {60}$</td><td>2.779</td><td>1.091</td></tr></table>

<table><tbody><tr><td>数据集</td><td colspan="2">ETT（原文未明确含义，保留英文）</td><td colspan="2">电力</td><td>数据集</td><td colspan="2">ILI（原文未明确含义，保留英文）</td></tr><tr><td>指标</td><td>均方误差（Mean Squared Error）</td><td>平均绝对误差（Mean Absolute Error）</td><td>均方误差（Mean Squared Error）</td><td>平均绝对误差（Mean Absolute Error）</td><td>指标</td><td>均方误差（Mean Squared Error）</td><td>平均绝对误差（Mean Absolute Error）</td></tr><tr><td>$I = {96}$</td><td>0.339</td><td>0.372</td><td>0.231</td><td>0.338</td><td>$I = {24}$</td><td>3.406</td><td>1.247</td></tr><tr><td>$I = {192}$</td><td>0.355</td><td>0.392</td><td>0.200</td><td>0.316</td><td>$I = {36}$</td><td>2.669</td><td>1.085</td></tr><tr><td>$I = {336}$</td><td>0.361</td><td>0.406</td><td>0.225</td><td>0.335</td><td>$I = {48}$</td><td>2.656</td><td>1.075</td></tr><tr><td>$I = {720}$</td><td>0.419</td><td>0.430</td><td>0.226</td><td>0.346</td><td>$I = {60}$</td><td>2.779</td><td>1.091</td></tr></tbody></table>

<!-- Media -->

### C.2 Past Information Utilization

### C.2 过往信息利用

For the decoder input of Autoformer,we attach the length- $\frac{I}{2}$ past information to the placeholder. This design is to provide recent past information to the decoder. As shown in Table 8, the model with more past information will obtain a better performance, but it also causes a larger memory cost. Thus, we set the decoder input as $\frac{I}{2} + O$ to trade off both the performance and efficiency.

对于Autoformer（自变压器）的解码器输入，我们将长度为$\frac{I}{2}$的过往信息附加到占位符上。这种设计是为了解码器提供近期的过往信息。如表8所示，拥有更多过往信息的模型会获得更好的性能，但这也会导致更大的内存开销。因此，我们将解码器输入设置为$\frac{I}{2} + O$，以在性能和效率之间进行权衡。

<!-- Media -->

Table 8: Autoformer performance under different lengths of input of the decoder. $O,\frac{I}{2} + O,I + O$ corresponds to the decoder input without past information, with half past information, with full past information respectively. We fix the forecasting setting as input-96-predict-336 on the ETT dataset.

表8：解码器不同输入长度下Autoformer（自变压器）的性能。$O,\frac{I}{2} + O,I + O$分别对应无过往信息、有一半过往信息、有全部过往信息的解码器输入。我们将ETT数据集的预测设置固定为输入96个数据点、预测336个数据点。

<table><tr><td>Decoder input length</td><td>$O$ (without past)</td><td>$\frac{I}{2} + O$ (with half past)</td><td>$I + O$ (with full past)</td></tr><tr><td>MSE</td><td>0.360</td><td>0.339</td><td>0.333</td></tr><tr><td>MAE</td><td>0.383</td><td>0.372</td><td>0.369</td></tr><tr><td>Memory Cost</td><td>3029 MB</td><td>3271 MB</td><td>3599 MB</td></tr></table>

<table><tbody><tr><td>解码器输入长度</td><td>$O$（无历史信息）</td><td>$\frac{I}{2} + O$（有半历史信息）</td><td>$I + O$（有全历史信息）</td></tr><tr><td>均方误差（MSE）</td><td>0.360</td><td>0.339</td><td>0.333</td></tr><tr><td>平均绝对误差（MAE）</td><td>0.383</td><td>0.372</td><td>0.369</td></tr><tr><td>内存成本</td><td>3029兆字节</td><td>3271兆字节</td><td>3599兆字节</td></tr></tbody></table>

<!-- Media -->

## D Ablation of Decomposition Architecture

## D 分解架构的消融实验

In this section, we attempt to further verify the effectiveness of our proposed progressive decomposition architecture. We adopt more well-established decomposition algorithms as the pre-processing for separate prediction settings. As shown in Table 9, our proposed progressive decomposition architecture consistently outperforms the separate prediction (especially the long-term forecasting setting), despite the latter being with mature decomposition algorithms and twice bigger model.

在本节中，我们试图进一步验证所提出的渐进式分解架构的有效性。我们采用更成熟的分解算法作为单独预测设置的预处理。如表9所示，尽管单独预测采用了成熟的分解算法且模型规模是我们模型的两倍，但我们提出的渐进式分解架构在性能上始终优于单独预测（尤其是在长期预测设置下）。

<!-- Media -->

Table 9: Ablation of decomposition architecture in ETT dataset under the input-96-predict-O setting, where $O \in  \{ {96},{192},{336},{720}\}$ . The backbone of separate prediction is canonical Transformer [41]. We adopt various decomposition algorithms as the pre-processing and use two Transformers to separately forecast the seasonal and trend-cyclical parts. The result is the sum of two parts prediction.

表9：在输入96步预测O设置下，ETT数据集上分解架构的消融实验，其中$O \in  \{ {96},{192},{336},{720}\}$ 。单独预测的骨干网络是经典Transformer [41]。我们采用各种分解算法进行预处理，并使用两个Transformer分别预测季节性和趋势循环部分。结果是两部分预测的总和。

<table><tr><td rowspan="2">Decomposition</td><td>Predict O</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">Separately</td><td>STL [33]</td><td>0.523</td><td>0.516</td><td>0.638</td><td>0.605</td><td>1.004</td><td>0.794</td><td>3.678</td><td>1.462</td></tr><tr><td>Hodrick-Prescott Filter [18]</td><td>0.464</td><td>0.495</td><td>0.816</td><td>0.733</td><td>0.814</td><td>0.722</td><td>2.181</td><td>1.173</td></tr><tr><td>Christiano-Fitzgerald Filter [11]</td><td>0.373</td><td>0.458</td><td>0.819</td><td>0.668</td><td>1.083</td><td>0.835</td><td>2.462</td><td>1.189</td></tr><tr><td>Baxter-King Filter [44]</td><td>0.440</td><td>0.514</td><td>0.623</td><td>0.626</td><td>0.861</td><td>0.741</td><td>2.150</td><td>1.175</td></tr><tr><td>Progressively</td><td>Autoformer</td><td>0.255</td><td>0.339</td><td>0.281</td><td>0.340</td><td>0.339</td><td>0.372</td><td>0.422</td><td>0.419</td></tr></table>

<table><tbody><tr><td rowspan="2">分解</td><td>预测O</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td></tr><tr><td>指标</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="4">分别地</td><td>标准模板库（STL） [33]</td><td>0.523</td><td>0.516</td><td>0.638</td><td>0.605</td><td>1.004</td><td>0.794</td><td>3.678</td><td>1.462</td></tr><tr><td>霍德里克 - 普雷斯科特滤波器（Hodrick-Prescott Filter） [18]</td><td>0.464</td><td>0.495</td><td>0.816</td><td>0.733</td><td>0.814</td><td>0.722</td><td>2.181</td><td>1.173</td></tr><tr><td>克里斯蒂亚诺 - 菲茨杰拉德滤波器（Christiano-Fitzgerald Filter） [11]</td><td>0.373</td><td>0.458</td><td>0.819</td><td>0.668</td><td>1.083</td><td>0.835</td><td>2.462</td><td>1.189</td></tr><tr><td>巴克斯特 - 金滤波器（Baxter-King Filter） [44]</td><td>0.440</td><td>0.514</td><td>0.623</td><td>0.626</td><td>0.861</td><td>0.741</td><td>2.150</td><td>1.175</td></tr><tr><td>逐步地</td><td>自动变压器（Autoformer）</td><td>0.255</td><td>0.339</td><td>0.281</td><td>0.340</td><td>0.339</td><td>0.372</td><td>0.422</td><td>0.419</td></tr></tbody></table>

<!-- Media -->

## E Supplementary of Main Results

## 主要结果的补充材料

### E.1 Multivariate Showcases

### E.1 多变量示例

To evaluate the prediction of different models, we plot the last dimension of forecasting results that are from the test set of ETT dataset for qualitative comparison (Figures 8, 9, 10, and 11). Our model gives the best performance among different models. Moreover, we observe that Autoformer can accurately predict the periodicity and long-term variation.

为了评估不同模型的预测效果，我们绘制了来自ETT数据集测试集的预测结果的最后一个维度，用于定性比较（图8、图9、图10和图11）。我们的模型在不同模型中表现最佳。此外，我们观察到自动转换器（Autoformer）能够准确预测周期性和长期变化。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_14.jpg?x=309&y=989&w=1179&h=251&r=0"/>

Figure 8: Prediction cases from the ETT dataset under the input-96-predict-96 setting. Blue lines are the ground truth and orange lines are the model prediction. The first part with length 96 is the input.

图8：在输入96步预测96步设置下，ETT数据集的预测案例。蓝色线为真实值，橙色线为模型预测值。长度为96的第一部分是输入。

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_14.jpg?x=307&y=1391&w=1181&h=255&r=0"/>

Figure 9: Prediction cases from the ETT dataset under the input-96-predict-192 setting.

图9：在输入96步预测192步设置下，ETT数据集的预测案例。

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_14.jpg?x=308&y=1765&w=1181&h=258&r=0"/>

Figure 10: Prediction cases from the ETT dataset under the input-96-predict-336 setting.

图10：在输入96步预测336步设置下，ETT数据集的预测案例。

<!-- figureText: Autoformer -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_15.jpg?x=306&y=204&w=1182&h=258&r=0"/>

Figure 11: Prediction cases from the ETT dataset under the input-96-predict-720 setting.

图11：在输入96步预测720步的设置下，ETT数据集的预测案例。

<!-- Media -->

### E.2 Performance on Data without Obvious Periodicity

### E.2 对无明显周期性数据的性能表现

Autoformer yields the best performance among six datasets, even in the Exchange dataset that does not have obvious periodicity. This section will give some showcases from the test set of multivariate Exchange dataset for qualitative evaluation. We observed that the series in the Exchange dataset show rapid fluctuations. And because of the inherent properties of economic data, the series does not present obvious periodicity. This aperiodicity causes extreme difficulties for prediction. As shown in Figure 12, compared to other models, Autoformer can still predict the exact long-term variations. It is verified the robustness of our model performance among various data characteristics.

Autoformer在六个数据集中表现最佳，即使是在没有明显周期性的Exchange数据集上也是如此。本节将从多变量Exchange数据集的测试集中给出一些示例进行定性评估。我们观察到，Exchange数据集中的序列呈现出快速波动。并且由于经济数据的固有特性，该序列没有呈现出明显的周期性。这种非周期性给预测带来了极大的困难。如图12所示，与其他模型相比，Autoformer仍然能够准确预测长期变化。这验证了我们的模型在各种数据特征下的性能鲁棒性。

<!-- Media -->

<!-- figureText: Autoformer Informer -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_15.jpg?x=308&y=846&w=1180&h=261&r=0"/>

Figure 12: Prediction cases from the Exchange dataset under the input-96-predict-192 setting.

图12：在输入96步预测192步的设置下，Exchange数据集的预测案例。

<!-- Media -->

### E.3 Univariate Forecasting Showcases

### E.3 单变量预测示例

As shown in Figure 13 Autoformer gives the most accurate prediction. Compared to Informer [48], Autoformer can precisely capture the periods of the future horizon. Besides, our model provides better prediction in the center area than LogTrans [26]. Compared with Reformer [23], our prediction series is smooth and closer to ground truth. Also, the fluctuation of DeepAR [34] prediction is getting smaller as prediction length increases and suffers from the over-smoothing problem, which does not happen in our Autoformer.

如图13所示，自动变压器模型（Autoformer）给出了最准确的预测。与告密者模型（Informer） [48] 相比，自动变压器模型能够精确捕捉未来时段的周期。此外，与对数变换模型（LogTrans） [26] 相比，我们的模型在中心区域提供了更好的预测。与改革者模型（Reformer） [23] 相比，我们的预测序列更平滑，更接近真实值。此外，深度自回归模型（DeepAR） [34] 的预测波动随着预测长度的增加而变小，并且存在过度平滑的问题，而我们的自动变压器模型不会出现这种情况。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_15.jpg?x=311&y=1432&w=1175&h=210&r=0"/>

Figure 13: Prediction cases from the ETT dataset under the input-96-predict-720 univariate setting.

图13：在输入96个数据点、预测720个数据点的单变量设置下，电力变压器温度（ETT）数据集的预测案例。

<!-- Media -->

### E.4 Main Results with Standard Deviations

### E.4 带有标准差的主要结果

To get more robust experimental results, we repeat each experiment three times. The results are shown without standard deviations in the main text due to the limited pages. Table 10 shows the standard deviations.

为了获得更可靠的实验结果，我们将每个实验重复三次。由于篇幅有限，正文中展示的结果未包含标准差。表10展示了标准差。

## F COVID-19: Case Study

## F 新冠疫情（COVID - 19）：案例研究

We also apply our model to the COVID-19 real-world data [15]. This dataset contains the data collected from countries, including the number of confirmed deaths and recovered patients of COVID-19 recorded daily from January 22, 2020, to May 20, 2021. We select two anonymous countries in Europe for the experiments. The data is split into training, validation and test set in chronological order following the ratio of 7:1:2 and normalized. Note that this problem is quite challenging because the training data is limited.

我们还将我们的模型应用于新冠疫情（COVID - 19）的真实世界数据[15]。该数据集包含从各个国家收集的数据，包括从2020年1月22日至2021年5月20日每日记录的新冠确诊死亡人数和康复患者人数。我们选择了欧洲的两个匿名国家进行实验。数据按照7:1:2的比例按时间顺序划分为训练集、验证集和测试集，并进行了归一化处理。请注意，由于训练数据有限，这个问题颇具挑战性。

<!-- Media -->

Table 10: Quantitative results with fluctuations under different prediction lengths $O$ for multivariate forecasting. We set the input length $I$ as 36 for ILI and 96 for the other datasets. A lower MSE or MAE indicates a better performance.

表10：多元预测在不同预测长度$O$下带有波动的定量结果。我们将输入长度$I$设置为：流感样疾病（ILI）数据集为36，其他数据集为96。均方误差（MSE）或平均绝对误差（MAE）越低，表示性能越好。

<table><tr><td colspan="2" rowspan="2">Models Metric</td><td colspan="2">Autoformer</td><td colspan="2">Informer[48]</td><td colspan="2">LogTrans [26]</td><td colspan="2">Reformer[23]</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">江</td><td>96</td><td>${0.255} \pm  {0.020}$</td><td>0.339±0.020</td><td>${0.365} \pm  {0.062}$</td><td>${0.453} \pm  {0.047}$</td><td>${0.768} \pm  {0.071}$</td><td>${0.642} \pm  {0.020}$</td><td>${0.658} \pm  {0.121}$</td><td>${0.619} \pm  {0.021}$</td></tr><tr><td>192</td><td>0.281 $\pm  {0.027}$</td><td>0.340±0.025</td><td>${0.533} \pm  {0.109}$</td><td>${0.563} \pm  {0.050}$</td><td>${0.989} \pm  {0.124}$</td><td>${0.757} \pm  {0.049}$</td><td>${1.078} \pm  {0.106}$</td><td>${0.827} \pm  {0.012}$</td></tr><tr><td>336</td><td>0.339±0.018</td><td>0.372±0.015</td><td>${1.363} \pm  {0.173}$</td><td>${0.887} \pm  {0.056}$</td><td>1.334±0.168</td><td>${0.872} \pm  {0.054}$</td><td>${1.549} \pm  {0.146}$</td><td>${0.972} \pm  {0.015}$</td></tr><tr><td>720</td><td>0.422 ± 0.015</td><td>${0.419} \pm  {0.010}$</td><td>${3.379} \pm  {0.143}$</td><td>${1.388} \pm  {0.037}$</td><td>${3.048} \pm  {0.140}$</td><td>${1.328} \pm  {0.023}$</td><td>${2.631} \pm  {0.126}$</td><td>${1.242} \pm  {0.014}$</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.201 $\pm  {0.003}$</td><td>0.317±0.004</td><td>${0.274} \pm  {0.004}$</td><td>${0.368} \pm  {0.003}$</td><td>${0.258} \pm  {0.002}$</td><td>${0.357} \pm  {0.002}$</td><td>${0.312} \pm  {0.003}$</td><td>${0.402} \pm  {0.004}$</td></tr><tr><td>192</td><td>0.222±0.003</td><td>0.334±0.004</td><td>${0.296} \pm  {0.009}$</td><td>${0.386} \pm  {0.007}$</td><td>${0.266} \pm  {0.005}$</td><td>${0.368} \pm  {0.004}$</td><td>${0.348} \pm  {0.004}$</td><td>${0.433} \pm  {0.005}$</td></tr><tr><td>336</td><td>0.231 $\pm  {0.006}$</td><td>0.338±0.004</td><td>${0.300} \pm  {0.007}$</td><td>${0.394} \pm  {0.004}$</td><td>${0.280} \pm  {0.006}$</td><td>${0.380} \pm  {0.001}$</td><td>${0.350} \pm  {0.004}$</td><td>${0.433} \pm  {0.003}$</td></tr><tr><td>720</td><td>0.254±0.007</td><td>0.361±0.008</td><td>${0.373} \pm  {0.034}$</td><td>${0.439} \pm  {0.024}$</td><td>${0.283} \pm  {0.003}$</td><td>0.376±0.002</td><td>${0.340} \pm  {0.002}$</td><td>${0.420} \pm  {0.002}$</td></tr><tr><td rowspan="4">Exchange</td><td>96</td><td>${0.197} \pm  {0.019}$</td><td>${0.323} \pm  {0.012}$</td><td>${0.847} \pm  {0.150}$</td><td>${0.752} \pm  {0.060}$</td><td>${0.968} \pm  {0.177}$</td><td>${0.812} \pm  {0.027}$</td><td>${1.065} \pm  {0.070}$</td><td>${0.829} \pm  {0.013}$</td></tr><tr><td>192</td><td>0.300±0.020</td><td>$\mathbf{{0.369}} \pm  {0.016}$</td><td>${1.204} \pm  {0.149}$</td><td>${0.895} \pm  {0.061}$</td><td>${1.040} \pm  {0.232}$</td><td>${0.851} \pm  {0.029}$</td><td>${1.188} \pm  {0.041}$</td><td>${0.906} \pm  {0.008}$</td></tr><tr><td>336</td><td>0.509±0.041</td><td>0.524±0.016</td><td>${1.672} \pm  {0.036}$</td><td>${1.036} \pm  {0.014}$</td><td>${1.659} \pm  {0.122}$</td><td>${1.081} \pm  {0.015}$</td><td>${1.357} \pm  {0.027}$</td><td>0.976±0.010</td></tr><tr><td>720</td><td>1.447 ± 0.084</td><td>$\mathbf{{0.941}} \pm  {0.028}$</td><td>${2.478} \pm  {0.198}$</td><td>${1.310} \pm  {0.070}$</td><td>${1.941} \pm  {0.327}$</td><td>${1.127} \pm  {0.030}$</td><td>${1.510} \pm  {0.071}$</td><td>${1.016} \pm  {0.008}$</td></tr><tr><td rowspan="4">千港元</td><td>96</td><td>$\mathbf{{0.613}} \pm  {0.028}$</td><td>$\mathbf{{0.388}} \pm  {0.012}$</td><td>${0.719} \pm  {0.015}$</td><td>${0.391} \pm  {0.004}$</td><td>${0.684} \pm  {0.041}$</td><td>${0.384} \pm  {0.008}$</td><td>${0.732} \pm  {0.027}$</td><td>${0.423} \pm  {0.025}$</td></tr><tr><td>192</td><td>0.616±0.042</td><td>0.382±0.020</td><td>${0.696} \pm  {0.050}$</td><td>${0.379} \pm  {0.023}$</td><td>${0.685} \pm  {0.055}$</td><td>${0.390} \pm  {0.021}$</td><td>${0.733} \pm  {0.013}$</td><td>${0.420} \pm  {0.011}$</td></tr><tr><td>336</td><td>0.622±0.016</td><td>0.337±0.011</td><td>${0.777} \pm  {0.009}$</td><td>${0.420} \pm  {0.003}$</td><td>${0.733} \pm  {0.069}$</td><td>${0.408} \pm  {0.026}$</td><td>${0.742} \pm  {0.012}$</td><td>${0.420} \pm  {0.008}$</td></tr><tr><td>720</td><td>$\mathbf{{0.660}} \pm  {0.025}$</td><td>$\mathbf{{0.408}} \pm  {0.015}$</td><td>${0.864} \pm  {0.026}$</td><td>${0.472} \pm  {0.015}$</td><td>${0.717} \pm  {0.030}$</td><td>${0.396} \pm  {0.010}$</td><td>${0.755} \pm  {0.023}$</td><td>${0.423} \pm  {0.014}$</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.266±0.007</td><td>$\mathbf{{0.336}} \pm  {0.006}$</td><td>${0.300} \pm  {0.013}$</td><td>${0.384} \pm  {0.013}$</td><td>${0.458} \pm  {0.143}$</td><td>${0.490} \pm  {0.038}$</td><td>${0.689} \pm  {0.042}$</td><td>${0.596} \pm  {0.019}$</td></tr><tr><td>192</td><td>0.307 ± 0.024</td><td>0.367 ± 0.022</td><td>${0.598} \pm  {0.045}$</td><td>${0.544} \pm  {0.028}$</td><td>${0.658} \pm  {0.151}$</td><td>${0.589} \pm  {0.032}$</td><td>${0.752} \pm  {0.048}$</td><td>${0.638} \pm  {0.029}$</td></tr><tr><td>336</td><td>0.359±0.035</td><td>0.395±0.031</td><td>${0.578} \pm  {0.024}$</td><td>${0.523} \pm  {0.016}$</td><td>${0.797} \pm  {0.034}$</td><td>${0.652} \pm  {0.019}$</td><td>${0.639} \pm  {0.030}$</td><td>${0.596} \pm  {0.021}$</td></tr><tr><td>720</td><td>0.419 ± 0.017</td><td>0.428 ± 0.014</td><td>${1.059} \pm  {0.096}$</td><td>${0.741} \pm  {0.042}$</td><td>${0.869} \pm  {0.045}$</td><td>${0.675} \pm  {0.093}$</td><td>${1.130} \pm  {0.084}$</td><td>${0.792} \pm  {0.055}$</td></tr><tr><td rowspan="4">日</td><td>24</td><td>3.483 $\pm  {0.107}$</td><td>${1.287} \pm  {0.018}$</td><td>${5.764} \pm  {0.354}$</td><td>${1.677} \pm  {0.080}$</td><td>${4.480} \pm  {0.313}$</td><td>${1.444} \pm  {0.033}$</td><td>${4.400} \pm  {0.117}$</td><td>${1.382} \pm  {0.021}$</td></tr><tr><td>36</td><td>3.103 $\pm  {0.139}$</td><td>1.148±0.025</td><td>${4.755} \pm  {0.248}$</td><td>${1.467} \pm  {0.067}$</td><td>${4.799} \pm  {0.251}$</td><td>${1.467} \pm  {0.023}$</td><td>${4.783} \pm  {0.138}$</td><td>${1.448} \pm  {0.023}$</td></tr><tr><td>48</td><td>${2.669} \pm  {0.151}$</td><td>1.085±0.037</td><td>${4.763} \pm  {0.295}$</td><td>${1.469} \pm  {0.059}$</td><td>${4.800} \pm  {0.233}$</td><td>${1.468} \pm  {0.021}$</td><td>${4.832} \pm  {0.122}$</td><td>${1.465} \pm  {0.016}$</td></tr><tr><td>60</td><td>$\mathbf{{2.770}} \pm  {0.085}$</td><td>1.125 $\pm  {0.019}$</td><td>${5.264} \pm  {0.237}$</td><td>${1.564} \pm  {0.044}$</td><td>${5.278} \pm  {0.231}$</td><td>${1.560} \pm  {0.014}$</td><td>${4.882} \pm  {0.123}$</td><td>${1.483} \pm  {0.016}$</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">模型指标</td><td colspan="2">自动变换器（Autoformer）</td><td colspan="2">信息变换器（Informer）[48]</td><td colspan="2">对数变换器（LogTrans） [26]</td><td colspan="2">重构变换器（Reformer）[23]</td></tr><tr><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="4">江</td><td>96</td><td>${0.255} \pm  {0.020}$</td><td>0.339±0.020</td><td>${0.365} \pm  {0.062}$</td><td>${0.453} \pm  {0.047}$</td><td>${0.768} \pm  {0.071}$</td><td>${0.642} \pm  {0.020}$</td><td>${0.658} \pm  {0.121}$</td><td>${0.619} \pm  {0.021}$</td></tr><tr><td>192</td><td>0.281 $\pm  {0.027}$</td><td>0.340±0.025</td><td>${0.533} \pm  {0.109}$</td><td>${0.563} \pm  {0.050}$</td><td>${0.989} \pm  {0.124}$</td><td>${0.757} \pm  {0.049}$</td><td>${1.078} \pm  {0.106}$</td><td>${0.827} \pm  {0.012}$</td></tr><tr><td>336</td><td>0.339±0.018</td><td>0.372±0.015</td><td>${1.363} \pm  {0.173}$</td><td>${0.887} \pm  {0.056}$</td><td>1.334±0.168</td><td>${0.872} \pm  {0.054}$</td><td>${1.549} \pm  {0.146}$</td><td>${0.972} \pm  {0.015}$</td></tr><tr><td>720</td><td>0.422 ± 0.015</td><td>${0.419} \pm  {0.010}$</td><td>${3.379} \pm  {0.143}$</td><td>${1.388} \pm  {0.037}$</td><td>${3.048} \pm  {0.140}$</td><td>${1.328} \pm  {0.023}$</td><td>${2.631} \pm  {0.126}$</td><td>${1.242} \pm  {0.014}$</td></tr><tr><td rowspan="4">电力</td><td>96</td><td>0.201 $\pm  {0.003}$</td><td>0.317±0.004</td><td>${0.274} \pm  {0.004}$</td><td>${0.368} \pm  {0.003}$</td><td>${0.258} \pm  {0.002}$</td><td>${0.357} \pm  {0.002}$</td><td>${0.312} \pm  {0.003}$</td><td>${0.402} \pm  {0.004}$</td></tr><tr><td>192</td><td>0.222±0.003</td><td>0.334±0.004</td><td>${0.296} \pm  {0.009}$</td><td>${0.386} \pm  {0.007}$</td><td>${0.266} \pm  {0.005}$</td><td>${0.368} \pm  {0.004}$</td><td>${0.348} \pm  {0.004}$</td><td>${0.433} \pm  {0.005}$</td></tr><tr><td>336</td><td>0.231 $\pm  {0.006}$</td><td>0.338±0.004</td><td>${0.300} \pm  {0.007}$</td><td>${0.394} \pm  {0.004}$</td><td>${0.280} \pm  {0.006}$</td><td>${0.380} \pm  {0.001}$</td><td>${0.350} \pm  {0.004}$</td><td>${0.433} \pm  {0.003}$</td></tr><tr><td>720</td><td>0.254±0.007</td><td>0.361±0.008</td><td>${0.373} \pm  {0.034}$</td><td>${0.439} \pm  {0.024}$</td><td>${0.283} \pm  {0.003}$</td><td>0.376±0.002</td><td>${0.340} \pm  {0.002}$</td><td>${0.420} \pm  {0.002}$</td></tr><tr><td rowspan="4">交换</td><td>96</td><td>${0.197} \pm  {0.019}$</td><td>${0.323} \pm  {0.012}$</td><td>${0.847} \pm  {0.150}$</td><td>${0.752} \pm  {0.060}$</td><td>${0.968} \pm  {0.177}$</td><td>${0.812} \pm  {0.027}$</td><td>${1.065} \pm  {0.070}$</td><td>${0.829} \pm  {0.013}$</td></tr><tr><td>192</td><td>0.300±0.020</td><td>$\mathbf{{0.369}} \pm  {0.016}$</td><td>${1.204} \pm  {0.149}$</td><td>${0.895} \pm  {0.061}$</td><td>${1.040} \pm  {0.232}$</td><td>${0.851} \pm  {0.029}$</td><td>${1.188} \pm  {0.041}$</td><td>${0.906} \pm  {0.008}$</td></tr><tr><td>336</td><td>0.509±0.041</td><td>0.524±0.016</td><td>${1.672} \pm  {0.036}$</td><td>${1.036} \pm  {0.014}$</td><td>${1.659} \pm  {0.122}$</td><td>${1.081} \pm  {0.015}$</td><td>${1.357} \pm  {0.027}$</td><td>0.976±0.010</td></tr><tr><td>720</td><td>1.447 ± 0.084</td><td>$\mathbf{{0.941}} \pm  {0.028}$</td><td>${2.478} \pm  {0.198}$</td><td>${1.310} \pm  {0.070}$</td><td>${1.941} \pm  {0.327}$</td><td>${1.127} \pm  {0.030}$</td><td>${1.510} \pm  {0.071}$</td><td>${1.016} \pm  {0.008}$</td></tr><tr><td rowspan="4">千港元</td><td>96</td><td>$\mathbf{{0.613}} \pm  {0.028}$</td><td>$\mathbf{{0.388}} \pm  {0.012}$</td><td>${0.719} \pm  {0.015}$</td><td>${0.391} \pm  {0.004}$</td><td>${0.684} \pm  {0.041}$</td><td>${0.384} \pm  {0.008}$</td><td>${0.732} \pm  {0.027}$</td><td>${0.423} \pm  {0.025}$</td></tr><tr><td>192</td><td>0.616±0.042</td><td>0.382±0.020</td><td>${0.696} \pm  {0.050}$</td><td>${0.379} \pm  {0.023}$</td><td>${0.685} \pm  {0.055}$</td><td>${0.390} \pm  {0.021}$</td><td>${0.733} \pm  {0.013}$</td><td>${0.420} \pm  {0.011}$</td></tr><tr><td>336</td><td>0.622±0.016</td><td>0.337±0.011</td><td>${0.777} \pm  {0.009}$</td><td>${0.420} \pm  {0.003}$</td><td>${0.733} \pm  {0.069}$</td><td>${0.408} \pm  {0.026}$</td><td>${0.742} \pm  {0.012}$</td><td>${0.420} \pm  {0.008}$</td></tr><tr><td>720</td><td>$\mathbf{{0.660}} \pm  {0.025}$</td><td>$\mathbf{{0.408}} \pm  {0.015}$</td><td>${0.864} \pm  {0.026}$</td><td>${0.472} \pm  {0.015}$</td><td>${0.717} \pm  {0.030}$</td><td>${0.396} \pm  {0.010}$</td><td>${0.755} \pm  {0.023}$</td><td>${0.423} \pm  {0.014}$</td></tr><tr><td rowspan="4">天气</td><td>96</td><td>0.266±0.007</td><td>$\mathbf{{0.336}} \pm  {0.006}$</td><td>${0.300} \pm  {0.013}$</td><td>${0.384} \pm  {0.013}$</td><td>${0.458} \pm  {0.143}$</td><td>${0.490} \pm  {0.038}$</td><td>${0.689} \pm  {0.042}$</td><td>${0.596} \pm  {0.019}$</td></tr><tr><td>192</td><td>0.307 ± 0.024</td><td>0.367 ± 0.022</td><td>${0.598} \pm  {0.045}$</td><td>${0.544} \pm  {0.028}$</td><td>${0.658} \pm  {0.151}$</td><td>${0.589} \pm  {0.032}$</td><td>${0.752} \pm  {0.048}$</td><td>${0.638} \pm  {0.029}$</td></tr><tr><td>336</td><td>0.359±0.035</td><td>0.395±0.031</td><td>${0.578} \pm  {0.024}$</td><td>${0.523} \pm  {0.016}$</td><td>${0.797} \pm  {0.034}$</td><td>${0.652} \pm  {0.019}$</td><td>${0.639} \pm  {0.030}$</td><td>${0.596} \pm  {0.021}$</td></tr><tr><td>720</td><td>0.419 ± 0.017</td><td>0.428 ± 0.014</td><td>${1.059} \pm  {0.096}$</td><td>${0.741} \pm  {0.042}$</td><td>${0.869} \pm  {0.045}$</td><td>${0.675} \pm  {0.093}$</td><td>${1.130} \pm  {0.084}$</td><td>${0.792} \pm  {0.055}$</td></tr><tr><td rowspan="4">日</td><td>24</td><td>3.483 $\pm  {0.107}$</td><td>${1.287} \pm  {0.018}$</td><td>${5.764} \pm  {0.354}$</td><td>${1.677} \pm  {0.080}$</td><td>${4.480} \pm  {0.313}$</td><td>${1.444} \pm  {0.033}$</td><td>${4.400} \pm  {0.117}$</td><td>${1.382} \pm  {0.021}$</td></tr><tr><td>36</td><td>3.103 $\pm  {0.139}$</td><td>1.148±0.025</td><td>${4.755} \pm  {0.248}$</td><td>${1.467} \pm  {0.067}$</td><td>${4.799} \pm  {0.251}$</td><td>${1.467} \pm  {0.023}$</td><td>${4.783} \pm  {0.138}$</td><td>${1.448} \pm  {0.023}$</td></tr><tr><td>48</td><td>${2.669} \pm  {0.151}$</td><td>1.085±0.037</td><td>${4.763} \pm  {0.295}$</td><td>${1.469} \pm  {0.059}$</td><td>${4.800} \pm  {0.233}$</td><td>${1.468} \pm  {0.021}$</td><td>${4.832} \pm  {0.122}$</td><td>${1.465} \pm  {0.016}$</td></tr><tr><td>60</td><td>$\mathbf{{2.770}} \pm  {0.085}$</td><td>1.125 $\pm  {0.019}$</td><td>${5.264} \pm  {0.237}$</td><td>${1.564} \pm  {0.044}$</td><td>${5.278} \pm  {0.231}$</td><td>${1.560} \pm  {0.014}$</td><td>${4.882} \pm  {0.123}$</td><td>${1.483} \pm  {0.016}$</td></tr></tbody></table>

<!-- Media -->

### F.1 Quantitative Results

### F.1 定量结果

We still follow the long-term forecasting task and let the model predict the next week, half month, full month respectively. The prediction lengths are 1, 2.1, 4.3 times the input length. As shown in Table 11 Autoformer still keeps the state-of-the-art accuracy under the limited data and short input situation.

我们仍然遵循长期预测任务，让模型分别预测下周、半月和整月的情况。预测长度分别是输入长度的1倍、2.1倍和4.3倍。如表11所示，在数据有限和输入时长较短的情况下，Autoformer（自注意力变压器）仍保持了最先进的准确率。

<!-- Media -->

Table 11: Quantitative results for COVID-19 data. We set the input length $I$ as 7,which means that the data in one week. The prediction length $O$ is in $\{ 7,{15},{30}\}$ ,which represents a week,half a month, a month respectively. A lower MSE or MAE indicates a better prediction.

表11：新冠疫情数据的定量结果。我们将输入长度$I$设为7，这意味着一周的数据。预测长度$O$取值于$\{ 7,{15},{30}\}$，分别代表一周、半月和一个月。均方误差（MSE）或平均绝对误差（MAE）越低，表示预测效果越好。

<table><tr><td colspan="2" rowspan="2">Models Metric</td><td colspan="2">Autoformer</td><td colspan="2">Informer[48]</td><td colspan="2">LogTrans [26]</td><td colspan="2">Reformer 23</td><td colspan="2">Transformer 41</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="3">Country 1</td><td>7</td><td>0.110</td><td>0.213</td><td>0.168</td><td>0.323</td><td>0.190</td><td>0.311</td><td>0.219</td><td>0.312</td><td>0.156</td><td>0.254</td></tr><tr><td>15</td><td>0.168</td><td>0.264</td><td>0.443</td><td>0.482</td><td>0.229</td><td>0.361</td><td>0.276</td><td>0.403</td><td>0.289</td><td>0.382</td></tr><tr><td>30</td><td>0.261</td><td>0.319</td><td>0.443</td><td>0.482</td><td>0.311</td><td>0.356</td><td>0.276</td><td>0.403</td><td>0.362</td><td>0.444</td></tr><tr><td rowspan="3">Country 2</td><td>7</td><td>1.747</td><td>0.891</td><td>1.806</td><td>0.969</td><td>1.834</td><td>1.013</td><td>2.403</td><td>1.071</td><td>1.798</td><td>0.955</td></tr><tr><td>15</td><td>1.749</td><td>0.905</td><td>1.842</td><td>0.969</td><td>1.829</td><td>1.004</td><td>2.627</td><td>1.111</td><td>1.830</td><td>0.999</td></tr><tr><td>30</td><td>1.749</td><td>0.903</td><td>2.087</td><td>1.116</td><td>2.147</td><td>1.106</td><td>3.316</td><td>1.267</td><td>2.190</td><td>1.172</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">模型指标</td><td colspan="2">自动变换器（Autoformer）</td><td colspan="2">信息变换器（Informer）[48]</td><td colspan="2">对数变换器（LogTrans） [26]</td><td colspan="2">重构变换器（Reformer） 23</td><td colspan="2">变换器（Transformer） 41</td></tr><tr><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="3">国家1</td><td>7</td><td>0.110</td><td>0.213</td><td>0.168</td><td>0.323</td><td>0.190</td><td>0.311</td><td>0.219</td><td>0.312</td><td>0.156</td><td>0.254</td></tr><tr><td>15</td><td>0.168</td><td>0.264</td><td>0.443</td><td>0.482</td><td>0.229</td><td>0.361</td><td>0.276</td><td>0.403</td><td>0.289</td><td>0.382</td></tr><tr><td>30</td><td>0.261</td><td>0.319</td><td>0.443</td><td>0.482</td><td>0.311</td><td>0.356</td><td>0.276</td><td>0.403</td><td>0.362</td><td>0.444</td></tr><tr><td rowspan="3">国家2</td><td>7</td><td>1.747</td><td>0.891</td><td>1.806</td><td>0.969</td><td>1.834</td><td>1.013</td><td>2.403</td><td>1.071</td><td>1.798</td><td>0.955</td></tr><tr><td>15</td><td>1.749</td><td>0.905</td><td>1.842</td><td>0.969</td><td>1.829</td><td>1.004</td><td>2.627</td><td>1.111</td><td>1.830</td><td>0.999</td></tr><tr><td>30</td><td>1.749</td><td>0.903</td><td>2.087</td><td>1.116</td><td>2.147</td><td>1.106</td><td>3.316</td><td>1.267</td><td>2.190</td><td>1.172</td></tr></tbody></table>

<!-- Media -->

### F.2 Showcases

### F.2 展示案例

As shown in Figure 14, compared to other models, our Autoformer can accurately predict the peaks and troughs at the beginning and can almost predict the exact value in the long-term future. The forecasting of extreme values and long-term trends are essential to epidemic prevention and control.

如图14所示，与其他模型相比，我们的自动变压器模型（Autoformer）能够在开始时准确预测峰值和谷值，并且几乎可以预测长期未来的精确值。极值和长期趋势的预测对于疫情防控至关重要。

<!-- Media -->

<!-- figureText: Autoformer -->

<img src="https://cdn.noedgeai.com/01957bd5-4059-7d5b-b0ed-e735eb0a09eb_17.jpg?x=308&y=208&w=1177&h=215&r=0"/>

Figure 14: Showcases from the second country of COVID-19 under the input-7-predict-15 setting.

图14：在输入7步预测15步的设置下，第二个国家新冠肺炎（COVID - 19）的展示案例。

<!-- Media -->

## G Autoformer: Implementation Details

## G 自动变压器模型（Autoformer）：实现细节

### G.1 Model Design

### G.1 模型设计

We provide the pseudo-code of Autoformer and Auto-Correlation mechanism in Algorithms 1 and 2 respectively. The tensor shapes and hyper-parameter settings are also included. Besides the above standard version, we speed up the Auto-Correlation to a batch-normalization-style block for efficiency, namely speedup version. All the experiment results of this paper are from the speedup version. Here are the implementation details.

我们分别在算法1和算法2中提供了自动变压器模型（Autoformer）和自相关机制（Auto - Correlation）的伪代码。还包括了张量形状和超参数设置。除了上述标准版本，为了提高效率，我们将自相关机制加速为一种批量归一化风格的模块，即加速版本。本文的所有实验结果均来自加速版本。以下是实现细节。

Speedup version Note that the gather operation in Algorithm 2 is not memory-access friendly. We borrow the design of batch normalization [21] to speedup the Auto-Correlation mechanism. We separate the whole procedure as the training phase and the inference phase. Because of the property of the linear layer, the channels of deep representations are equivalent. Thus, we reduce the channel and head dimension for both the training and inference phases. Especially for the training phase, we average the autocorrelation within a batch to simplify the learned lags. This design speeds up Auto-Correlation and performs as normalization to obtain a global judgment of the learned lags because the series within a batch are samples from the same time-series dataset. The pseudo-code for the training phase is presented in Algorithm 3 . For the testing phase, we still use the gather operation with respect to the simplified lags, which is more memory-access friendly than the standard version. The pseudo-code for the inference phase is presented in Algorithm 4

加速版本 请注意，算法2中的收集操作对内存访问不友好。我们借鉴批量归一化[21]的设计来加速自相关机制。我们将整个过程分为训练阶段和推理阶段。由于线性层的特性，深度表示的通道是等价的。因此，我们在训练和推理阶段都降低了通道和头的维度。特别是在训练阶段，我们对一个批次内的自相关进行平均，以简化学习到的滞后。这种设计加速了自相关过程，并起到归一化的作用，从而对学习到的滞后进行全局判断，因为一个批次内的序列是来自同一时间序列数据集的样本。训练阶段的伪代码在算法3中给出。在测试阶段，我们仍然针对简化后的滞后使用收集操作，这比标准版本对内存访问更友好。推理阶段的伪代码在算法4中给出

Complexity analysis Our model provides the series-wise aggregation for $\lfloor c \times  \log L\rfloor$ delayed length- $L$ series. Thus,the complexity is $\mathcal{O}\left( {L\log L}\right)$ for both the standard version and the speedup version. However,the latter is faster because it is more memory-access friendly.

复杂度分析 我们的模型为$\lfloor c \times  \log L\rfloor$个延迟的长度为$L$的序列提供了逐序列聚合。因此，标准版本和加速版本的复杂度均为$\mathcal{O}\left( {L\log L}\right)$。然而，加速版本更快，因为它对内存访问更友好。

### G.2 Experiment Details

### G.2 实验细节

All these transformer-based models are built with two encoder layers and one decoder layer for the sake of the fair comparison in performance and efficiency, including Informer [48], Reformer [23], LogTrans [26] and canonical Transformer [41]. Besides, all these models adopt the embedding method and the one-step generation strategy as Informer [48]. Note that our proposed series-wise aggregation can provide enough sequential information. Thus, we do not employ the position embedding as other baselines but keep the value embedding and time stamp embedding.

为了在性能和效率方面进行公平比较，所有这些基于Transformer的模型都由两个编码器层和一个解码器层构建而成，包括Informer [48]、Reformer [23]、LogTrans [26]和经典Transformer [41]。此外，所有这些模型都采用了与Informer [48]相同的嵌入方法和单步生成策略。请注意，我们提出的逐序列聚合可以提供足够的序列信息。因此，我们不像其他基线模型那样使用位置嵌入，而是保留值嵌入和时间戳嵌入。

## H Broader Impact

## H 更广泛的影响

Real-world applications Our proposed Autoformer focuses on the long-term time series forecasting problem, which is a valuable and urgent demand in extensive applications. Our method achieves consistent state-of-the-art performance in five real-world applications: energy, traffic, economics, weather and disease. In addition, we provide the case study of the COVID-19 dataset. Thus, people who work in these areas may benefit greatly from our work. We believe that better time series forecasting can help our society make better decisions and prevent risks in advance for various fields.

实际应用 我们提出的自动变换器（Autoformer）专注于长期时间序列预测问题，这在广泛的应用中是一项有价值且迫切的需求。我们的方法在五个实际应用领域中均达到了一致的最先进水平：能源、交通、经济、气象和疾病。此外，我们还提供了新冠疫情（COVID - 19）数据集的案例研究。因此，在这些领域工作的人员可能会从我们的工作中受益匪浅。我们相信，更出色的时间序列预测能够帮助我们的社会做出更优决策，并为各个领域提前防范风险。

Academic research In this paper, we take the ideas from classic time series analysis and stochastic process theory. We innovate a general deep decomposition architecture with a novel Auto-Correlation mechanism, which is a worthwhile addition to time series forecasting models. Code is available at this repository: https: //github.com/thuml/Autoformer

学术研究 在本文中，我们借鉴了经典时间序列分析和随机过程理论的思想。我们创新了一种具有新型自相关机制的通用深度分解架构，这是对时间序列预测模型的一项有价值的补充。代码可在以下仓库获取：https: //github.com/thuml/Autoformer

Model Robustness Based on the extensive experiments, we do not find exceptional failure cases. Autoformer even provides good performance and long-term robustness in the Exchange dataset that does not present obvious periodicity. Autoformer can progressively get purer series components by the inner decomposition block and make it easy to discover the deeply hidden periodicity. But if the data is random or with extremely weak temporal coherence, Autoformer and any other models may degenerate because the series is with poor predictability [14].

模型鲁棒性 基于大量实验，我们未发现异常的失败案例。自动分解变换器（Autoformer）甚至在未呈现明显周期性的外汇数据集上也能提供良好的性能和长期鲁棒性。自动分解变换器可以通过内部分解模块逐步提取更纯净的序列分量，从而易于发现深层次隐藏的周期性。但如果数据是随机的或时间连贯性极弱，自动分解变换器和其他任何模型都可能退化，因为该序列的可预测性较差 [14]。

Our work only focuses on the scientific problem, so there is no potential ethical risk.

我们的工作仅聚焦于科学问题，因此不存在潜在的伦理风险。

<!-- Media -->

Algorithm 1 Overall Autoformer Procedure

算法1 自动分解变换器（Autoformer）总体流程

Input: Input past time series $\mathcal{X}$ ; Input Length $I$ ; Predict length $O$ ; Data dimension $d$ ; Hidden state

输入：过去的时间序列输入 $\mathcal{X}$；输入长度 $I$；预测长度 $O$；数据维度 $d$；隐藏状态

---

	channel ${d}_{\text{model }}$ ; Encoder layers number $N$ ; Decoder layers number $M$ ; Moving average window

	size $k$ . Technically,we set ${d}_{\text{model }}$ as ${512},N$ as $2,M$ as $1,k$ as25.

		${\mathcal{X}}_{\text{ens }},{\mathcal{X}}_{\text{ent }} =$ SeriesDecomp $\left( {\mathcal{X}}_{\frac{I}{2} : I}\right)$ $\vartriangleright  \mathcal{X} \in  {\mathbb{R}}^{I \times  d},{\mathcal{X}}_{\text{ens }},{\mathcal{X}}_{\text{ent }} \in  {\mathbb{R}}^{\frac{I}{2} \times  d}$

2: ${\mathcal{X}}_{0},{\mathcal{X}}_{\text{mean }} = \operatorname{Zeros}\left( \left\lbrack  {O,d}\right\rbrack  \right)$ ,Repeat $\left( {\operatorname{Mean}\left( {{\mathcal{X}}_{\frac{I}{2} : I},\dim  = 0}\right) ,\dim  = 0}\right) \; \vartriangleright  {\mathcal{X}}_{0},{\mathcal{X}}_{\text{mean }} \in  {\mathbb{R}}^{O \times  }$

3: ${\mathcal{X}}_{\text{des }},{\mathcal{X}}_{\text{det }} = \operatorname{Concat}\left( {{\mathcal{X}}_{\text{ens }},{\mathcal{X}}_{0}}\right) ,\operatorname{Concat}\left( {{\mathcal{X}}_{\text{ent }},{\mathcal{X}}_{\text{mean }}}\right) \; \vartriangleright  {\mathcal{X}}_{\text{des }},{\mathcal{X}}_{\text{det }} \in  {\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  d}$

	${\mathcal{X}}_{\text{en }}^{0} = \operatorname{Embed}\left( \mathcal{X}\right)$ $\vartriangleright  {\mathcal{X}}_{\text{en }}^{0} \in  {\mathbb{R}}^{I \times  {d}_{\text{model }}}$

	for $l$ in $\{ 1,\cdots ,N\}$ : $\vartriangleright$ Autoformer Encoder

		${\mathcal{S}}_{\text{en }}^{l,1}, =$ SeriesDecomp $\left( {\text{ Auto-Correlation }\left( {\mathcal{X}}_{\text{en }}^{l - 1}\right)  + {\mathcal{X}}_{\text{en }}^{l - 1}}\right)$ $\vartriangleright  {\mathcal{S}}_{\text{en }}^{l,1} \in  {\mathbb{R}}^{I \times  {d}_{\text{model }}}$

			${\mathcal{S}}_{\text{en }}^{l,2}, =$ SeriesDecomp $\left( {\operatorname{FeedForward}\left( {\mathcal{S}}_{\text{en }}^{l,1}\right)  + {\mathcal{S}}_{\text{en }}^{l,1}}\right)$ $\vartriangleright  {\mathcal{S}}_{\text{en }}^{l,2} \in  {\mathbb{R}}^{I \times  {d}_{\text{model }}}$

			${\mathcal{X}}_{\text{en }}^{l} = {\mathcal{S}}_{\text{en }}^{l,2}$ $\vartriangleright  {\mathcal{X}}_{\text{en }}^{l} \in  {\mathbb{R}}^{I \times  {d}_{\text{model }}}$

	End for

	${\mathcal{X}}_{\mathrm{{de}}}^{0} = \operatorname{Embed}\left( {\mathcal{X}}_{\mathrm{{des}}}\right) ,{\mathcal{T}}_{\mathrm{{de}}}^{0} = {\mathcal{X}}_{\mathrm{{det}}},$ $\vartriangleright  {\mathcal{X}}_{\mathrm{{de}}}^{0} \in  {\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  {d}_{\text{model }}},{\mathcal{T}}_{\mathrm{{de}}}^{0} \in  {\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  d}$

	for $l$ in $\{ 1,\cdots ,M\}$ : $\vartriangleright$ Autoformer Decoder

			${\mathcal{S}}_{\mathrm{{de}}}^{l,1},{\mathcal{T}}_{\mathrm{{de}}}^{l,1} =$ SeriesDecomp $\left( {\text{Auto-Correlation}\left( {\mathcal{X}}_{\mathrm{{de}}}^{l - 1}\right)  + {\mathcal{X}}_{\mathrm{{de}}}^{l - 1}}\right)$

			${\mathcal{S}}_{\mathrm{{de}}}^{l,2},{\mathcal{T}}_{\mathrm{{de}}}^{l,2} = \operatorname{SeriesDecomp}\left( {\operatorname{Auto} - \operatorname{Correlation}\left( {{\mathcal{S}}_{\mathrm{{de}}}^{l,1},{\mathcal{X}}_{\mathrm{{en}}}^{N}}\right)  + {\mathcal{S}}_{\mathrm{{de}}}^{l,1}}\right)$

			${\mathcal{S}}_{\mathrm{{de}}}^{l,3},{\mathcal{T}}_{\mathrm{{de}}}^{l,3} = \operatorname{SeriesDecomp}\left( {\operatorname{FeedForward}\left( {\mathcal{S}}_{\mathrm{{de}}}^{l,2}\right)  + {\mathcal{S}}_{\mathrm{{de}}}^{l,2}}\right) \; \vartriangleright  {\mathcal{S}}_{\mathrm{{de}}}^{l, \cdot  },{\mathcal{T}}_{\mathrm{{de}}}^{l, \cdot  } \in  {\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  {d}_{\text{model }}}$

			${\mathcal{T}}_{\mathrm{{de}}}^{l} = {\mathcal{T}}_{\mathrm{{de}}}^{l - 1} + \operatorname{MLP}\left( {\mathcal{T}}_{\mathrm{{de}}}^{l,1}\right)  + \operatorname{MLP}\left( {\mathcal{T}}_{\mathrm{{de}}}^{l,2}\right)  + \operatorname{MLP}\left( {\mathcal{T}}_{\mathrm{{de}}}^{l,3}\right)$ $\vartriangleright  {\mathcal{T}}_{\mathrm{{de}}}^{l} \in  {\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  d}$

			${\mathcal{X}}_{\mathrm{{de}}}^{l} = {\mathcal{S}}_{\mathrm{{de}}}^{l,3}$ $\vartriangleright  {\mathcal{X}}_{\mathrm{{de}}}^{l} \in  {\mathbb{R}}^{\left( {\frac{I}{2} + O}\right)  \times  {d}_{\text{model }}}$

	End for

	${\mathcal{X}}_{\text{pred }} = \operatorname{MLP}\left( {\mathcal{X}}_{\text{de }}^{M}\right)  + {\mathcal{T}}_{\text{de }}^{M}$ $\vartriangleright  {\mathcal{X}}_{\text{pred }} \in  \mathbb{R}\left( {\frac{I}{2} + O}\right)  \times  {d}_{\text{model }}$

	Return ${\mathcal{X}}_{\text{pred }\frac{I}{2} : \frac{I}{2} + O}$ $\vartriangleright$ Return the prediction results

---

Algorithm 2 Auto-Correlation (multi-head standard version for a batch of data)

算法2 自相关算法（一批数据的多头标准版本）

---

Input: Queries $\mathcal{Q} \in  {\mathbb{R}}^{B \times  L \times  {d}_{\text{model }}}$ ; Keys $\mathcal{K} \in  {\mathbb{R}}^{B \times  S \times  {d}_{\text{model }}}$ ; Values $\mathcal{V} \in  {\mathbb{R}}^{B \times  S \times  {d}_{\text{model }}}$ ; Number of

																																													heads $h$ ; Hidden state channel ${d}_{\text{model }}$ ; Hyper-parameter $c$ . We set ${d}_{\text{model }}$ as ${512},h$ as $8,1 \leq  c \leq  3$ .

																						1: $\mathcal{K},\mathcal{V} = \operatorname{Resize}\left( \mathcal{K}\right)$ ,Resize $\left( \mathcal{V}\right) \; \vartriangleright$ Resize is truncation or zero filling. $\mathcal{K},\mathcal{V} \in  {\mathbb{R}}^{B \times  L \times  {d}_{\text{model }}}$

																										$\mathcal{2} : \mathcal{Q},\mathcal{K},\mathcal{V} = \operatorname{Reshape}\left( \mathcal{Q}\right)$ ,Reshape(K),Reshape(V) $\vartriangleright  \mathcal{Q},\mathcal{K},\mathcal{V} \in  {\mathcal{R}}^{L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

																																						$\mathcal{Q} = \operatorname{FFT}\left( {\mathcal{Q},\dim  = 0}\right) ,\mathcal{K} = \operatorname{FFT}\left( {\mathcal{K},\dim  = 0}\right) ,$ $\vartriangleright  \mathcal{Q},\mathcal{K} \in  {\mathbb{C}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

																																										$\operatorname{Corr} = \operatorname{IFFT}\left( {\mathcal{Q} \times  \operatorname{Conj}\left( \mathcal{K}\right) ,\dim  = 0}\right)$ $\vartriangleright$ Autocorrelation Corr $\in  {\mathbb{R}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

											5: ${\mathrm{W}}_{\text{topk }},{\mathrm{I}}_{\text{topk }} = \operatorname{Topk}\left( {\text{ Corr,}\lfloor c \times  \log L\rfloor ,\text{dim=0 }}\right)  \vartriangleright$ Largest weights ${\mathrm{W}}_{\text{topk }}$ and their indices ${\mathrm{I}}_{\text{topk }}$

																																								${\mathrm{W}}_{\text{topk }} = \operatorname{Softmax}\left( {{\mathrm{W}}_{\text{topk }},\dim  = 0}\right)$ $\vartriangleright  {\mathbf{W}}_{\text{topk }},{\mathbf{I}}_{\text{topk }} \in  {\mathbb{R}}^{B \times  \left( \left\lfloor  {c \times  \log L}\right\rfloor  \right)  \times  h \times  \frac{{d}_{\text{model }}}{h}}$

																					7: Index = Repeat $\left( {\operatorname{arange}\left( L\right) }\right)$ $\vartriangleright$ Initialize series indices. Index $\in  {\mathbb{R}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

																																		: $\mathcal{V} = \operatorname{Repeat}\left( \mathcal{V}\right)$ $\vartriangleright  \mathcal{V} \in  \mathbb{R}B \times  \left( {2L}\right)  \times  h \times  \frac{{d}_{\text{model }}}{h}$

									9: $\mathcal{R} = \left\lbrack  {{\mathbf{W}}_{\text{topk }}{}_{i, : , : } \times  \operatorname{gather}\left( {\mathcal{V},\left( {{\operatorname{I}}_{{\text{topk }}_{i, : , : }} + \operatorname{Index}}\right) }\right) \text{ for }i\text{ in range }\left( {\lfloor c \times  \log L\rfloor }\right) }\right\rbrack   \vartriangleright$ Aggregation

																																												$\mathcal{R} = \operatorname{Sum}\left( {\operatorname{Stack}\left( {\mathcal{R},\dim  = 0}\right) ,\dim  = 0}\right)$ $\vartriangleright  \mathcal{R} \in  {\mathbb{R}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

																																																	Return $\mathcal{R}$ $\vartriangleright$ Return transformed results

---

Algorithm 3 Auto-Correlation (multi-head speedup version for the training phase)

算法3 自相关算法（训练阶段的多头加速版本）

---

Input: Queries $\mathcal{Q} \in  {\mathbb{R}}^{B \times  L \times  {d}_{\text{model }}}$ ; Keys $\mathcal{K} \in  {\mathbb{R}}^{B \times  S \times  {d}_{\text{model }}}$ ; Values $\mathcal{V} \in  {\mathbb{R}}^{B \times  S \times  {d}_{\text{model }}}$ ; Number of

		heads $h$ ; Hidden state channel ${d}_{\text{model }}$ ; Hyper-parameter $c$ . We set ${d}_{\text{model }}$ as ${512},h$ as $8,1 \leq  c \leq  3$ .

							size(K),Resize $\left( \mathcal{V}\right) \; \vartriangleright$ Resize is truncation or zero filling. $\mathcal{K},\mathcal{V} \in  {\mathbb{R}}^{B \times  L \times  {d}_{\text{model }}}$

	2: $\mathcal{Q},\mathcal{K},\mathcal{V} = \operatorname{Reshape}\left( \mathcal{Q}\right)$ ,Reshape(K),Reshape(V) $\vartriangleright  \mathcal{Q},\mathcal{K},\mathcal{V} \in  {\mathcal{R}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

	3: $\mathcal{Q} = \operatorname{FFT}\left( {\mathcal{Q},\dim  = 0}\right) ,\mathcal{K} = \operatorname{FFT}\left( {\mathcal{K},\dim  = 0}\right)$ , $\vartriangleright  \mathcal{Q},\mathcal{K} \in  {\mathbb{C}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

	4: $\operatorname{Corr} = \operatorname{IFFT}\left( {\mathcal{Q} \times  \operatorname{Conj}\left( \mathcal{K}\right) ,\dim  = 0}\right)$ $\vartriangleright$ Autocorrelation Corr $\in  {\mathbb{R}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

	5: $\operatorname{Corr} = \operatorname{Mean}\left( {\text{ Corr,}\dim  = 0,2,3}\right)$ $\vartriangleright$ Simplify lags. Corr $\in  {\mathbb{R}}^{L}$

	6: ${\mathrm{W}}_{\text{topk }},{\mathrm{I}}_{\text{topk }} = \operatorname{Topk}\left( {\operatorname{Corr},\lfloor c \times  \log L\rfloor ,\dim  = 0}\right) \; \vartriangleright$ Largest weights ${\mathrm{W}}_{\text{topk }}$ and their indices ${\mathrm{I}}_{\text{topk }}$

	7: ${W}_{\text{topk }} = \operatorname{Softmax}\left( {{W}_{\text{topk }},\dim  = 0}\right)$ $\vartriangleright  {\mathbf{W}}_{\text{topk }},{\mathbf{I}}_{\text{topk }} \in  \mathbb{R}\left( \left\lfloor  {c \times  \log L}\right\rfloor  \right)$

	8: $\mathcal{R} = \left\lbrack  {{\mathbf{W}}_{\text{topk }}{}_{i, : , : } \times  \operatorname{Roll}\left( {\mathcal{V},{\mathbf{I}}_{\text{topk }}{}_{i, : , : },\text{ dim } = 1}\right) \text{ for }i\text{ in range }\left( \left\lfloor  {c \times  \log L}\right\rfloor  \right) }\right\rbrack   \vartriangleright$ Aggregation

	9: $\mathcal{R} = \operatorname{Sum}\left( {\operatorname{Stack}\left( {\mathcal{R},\dim  = 0}\right) ,\dim  = 0}\right)$ $\vartriangleright  \mathcal{R} \in  {\mathbb{R}}^{L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

		Return $\mathcal{R}$ $\vartriangleright$ Return transformed results

---

Algorithm 4 Auto-Correlation (multi-head speedup version for the inference phase)

算法4 自相关算法（推理阶段的多头加速版本）

---

Input: Queries $\mathcal{Q} \in  {\mathbb{R}}^{B \times  L \times  {d}_{\text{model }}}$ ; Keys $\mathcal{K} \in  {\mathbb{R}}^{B \times  S \times  {d}_{\text{model }}}$ ; Values $\mathcal{V} \in  {\mathbb{R}}^{B \times  S \times  {d}_{\text{model }}}$ ; Number of

		heads $h$ ; Hidden state channel ${d}_{\text{model }}$ ; Hyper-parameter $c$ . We set ${d}_{\text{model }}$ as 512, $h$ as $8,1 \leq  c \leq  3$ .

1: $\mathcal{K},\mathcal{V} = \operatorname{Resize}\left( \mathcal{K}\right)$ ,Resize $\left( \mathcal{V}\right) \; \vartriangleright$ Resize is truncation or zero filling. $\mathcal{K},\mathcal{V} \in  {\mathbb{R}}^{B \times  L \times  {d}_{\text{model }}}$

	2: $\mathcal{Q},\mathcal{K},\mathcal{V} = \operatorname{Reshape}\left( \mathcal{Q}\right)$ ,Reshape(K),Reshape(V) $\vartriangleright  \mathcal{Q},\mathcal{K},\mathcal{V} \in  {\mathcal{R}}^{L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

	3: $\mathcal{Q} = \operatorname{FFT}\left( {\mathcal{Q},\dim  = 0}\right) ,\mathcal{K} = \operatorname{FFT}\left( {\mathcal{K},\dim  = 0}\right)$ , $\vartriangleright  \mathcal{Q},\mathcal{K} \in  {\mathbb{C}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

	4: $\operatorname{Corr} = \operatorname{IFFT}\left( {\mathcal{Q} \times  \operatorname{Conj}\left( \mathcal{K}\right) ,\dim  = 0}\right)$ $\vartriangleright$ Autocorrelation Corr $\in  {\mathbb{R}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

	5: $\operatorname{Corr} = \operatorname{Mean}\left( {\text{ Corr,}\dim  = 0,2,3}\right)$ $\vartriangleright$ Simplify lags. Corr $\in  {\mathbb{R}}^{L}$

	6: ${\mathrm{W}}_{\text{topk }},{\mathrm{I}}_{\text{topk }} = \operatorname{Topk}\left( {\operatorname{Corr},\lfloor c \times  \log L\rfloor ,\dim  = 0}\right) \; \vartriangleright$ Largest weights ${\mathrm{W}}_{\text{topk }}$ and their indices ${\mathrm{I}}_{\text{topk }}$

	7: ${W}_{\text{topk }} = \operatorname{Softmax}\left( {{W}_{\text{topk }},\dim  = 0}\right)$ $\vartriangleright  {\mathbf{W}}_{\text{topk }},{\mathbf{I}}_{\text{topk }} \in  {\mathbb{R}}^{\left( \lfloor c \times  \log L\rfloor \right) }$

	8: Index = Repeat $\left( {\operatorname{arange}\left( L\right) }\right)$ $\vartriangleright$ Initialize series indices. Index $\in  {\mathbb{R}}^{B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}}$

	9: $\mathcal{V} =$ Repeat(V) $\vartriangleright  \mathcal{V} \in  {\mathbb{R}}^{B \times  \left( {2L}\right)  \times  h \times  \frac{{d}_{\text{model }}}{h}}$

10: $\mathcal{R} = \left\lbrack  {{\mathrm{W}}_{{\text{topk }}_{i, : , : }} \times  \text{gather}\left( {\mathcal{V},\left( {{\operatorname{I}}_{{\text{topk }}_{i, : , : }} + \text{Index}}\right) }\right) \text{for }i\text{ in range }\left( {\lfloor c \times  \log L\rfloor }\right) }\right\rbrack   \vartriangleright$ Aggregation

		$\mathcal{R} = \operatorname{Sum}\left( {\operatorname{Stack}\left( {\mathcal{R},\dim  = 0}\right) ,\dim  = 0}\right)$ $\vartriangleright  \mathcal{R} \in  \mathbb{R}B \times  L \times  h \times  \frac{{d}_{\text{model }}}{h}$

12: Return $\mathcal{R}$ $\vartriangleright$ Return transformed results

---

<!-- Media -->