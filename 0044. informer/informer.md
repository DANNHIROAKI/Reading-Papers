# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

# 信息器（Informer）：超越高效Transformer的长序列时间序列预测方法

Haoyi Zhou, ${}^{1}$ Shanghang Zhang, ${}^{2}$ Jieqi Peng, ${}^{1}$ Shuai Zhang, ${}^{1}$ Jianxin Li, ${}^{1}$ Hui Xiong, ${}^{3}$ Wancai Zhang ${}^{4}$

周浩毅，${}^{1}$ 张上航，${}^{2}$ 彭杰琦，${}^{1}$ 张帅，${}^{1}$ 李建新，${}^{1}$ 熊辉，${}^{3}$ 张万财 ${}^{4}$

${}^{1}$ Beihang University ${}^{2}$ UC Berkeley ${}^{3}$ Rutgers University ${}^{4}$ SEDD Company

${}^{1}$ 北京航空航天大学（Beihang University） ${}^{2}$ 加州大学伯克利分校（UC Berkeley） ${}^{3}$ 罗格斯大学（Rutgers University） ${}^{4}$ 赛迪公司（SEDD Company）

\{zhouhy, pengjq, zhangs, lijx\}@act.buaa.edu.cn, shz@eecs.berkeley.edu, \{xionghui,zhangwancaibuaa\}@gmail.com

\{zhouhy, pengjq, zhangs, lijx\}@act.buaa.edu.cn, shz@eecs.berkeley.edu, \{xionghui,zhangwancaibuaa\}@gmail.com

## Abstract

## 摘要

Many real-world applications require the prediction of long sequence time-series, such as electricity consumption planning. Long sequence time-series forecasting (LSTF) demands a high prediction capacity of the model, which is the ability to capture precise long-range dependency coupling between output and input efficiently. Recent studies have shown the potential of Transformer to increase the prediction capacity. However, there are several severe issues with Transformer that prevent it from being directly applicable to LSTF, including quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. To address these issues, we design an efficient transformer-based model for LSTF, named Informer, with three distinctive characteristics: (i) a ProbSparse self-attention mechanism, which achieves $\mathcal{O}\left( {L\log L}\right)$ in time complexity and memory usage, and has comparable performance on sequences' dependency alignment. (ii) the self-attention distilling highlights dominating attention by halving cascading layer input, and efficiently handles extreme long input sequences. (iii) the generative style decoder, while conceptually simple, predicts the long time-series sequences at one forward operation rather than a step-by-step way, which drastically improves the inference speed of long-sequence predictions. Extensive experiments on four large-scale datasets demonstrate that Informer significantly outperforms existing methods and provides a new solution to the LSTF problem.

许多实际应用需要对长序列时间序列进行预测，例如电力消耗规划。长序列时间序列预测（Long sequence time-series forecasting，LSTF）要求模型具备较高的预测能力，即能够高效捕捉输出与输入之间精确的长距离依赖耦合关系。近期研究表明，Transformer模型在提升预测能力方面具有潜力。然而，Transformer存在一些严重问题，使其无法直接应用于LSTF，包括二次时间复杂度、高内存使用以及编解码器架构的固有局限性。为解决这些问题，我们设计了一种基于Transformer的高效LSTF模型，名为Informer，它具有三个显著特点：（i）一种概率稀疏自注意力机制（ProbSparse self-attention mechanism），该机制在时间复杂度和内存使用方面达到了$\mathcal{O}\left( {L\log L}\right)$，并且在序列依赖对齐方面具有相当的性能。（ii）自注意力蒸馏（self-attention distilling）通过将级联层输入减半来突出主导注意力，并能有效处理极长的输入序列。（iii）生成式解码器（generative style decoder）虽然概念简单，但它可以通过一次前向运算预测长时序列，而不是采用逐步预测的方式，这极大地提高了长序列预测的推理速度。在四个大规模数据集上进行的大量实验表明，Informer显著优于现有方法，为LSTF问题提供了一种新的解决方案。

## 1 Introduction

## 1 引言

Time-series forecasting is a critical ingredient across many domains, such as sensor network monitoring (Papadimitriou and Yu 2006), energy and smart grid management, economics and finance (Zhu and Shasha 2002), and disease propagation analysis (Matsubara et al. 2014). In these scenarios, we can leverage a substantial amount of time-series data on past behavior to make a forecast in the long run, namely long sequence time-series forecasting (LSTF). However, existing methods are mostly designed under short-term problem setting, like predicting 48 points or less (Hochreiter and Schmidhuber 1997; Li et al. 2018; Yu et al. 2017; Liu et al. 2019; Qin et al. 2017; Wen et al. 2017). The increasingly long sequences strain the models' prediction capacity to the point where this trend is holding the research on LSTF. As an empirical example, Fig. 1) shows the forecasting results on a real dataset, where the LSTM network predicts the hourly temperature of an electrical transformer station from the short-term period (12 points, 0.5 days) to the long-term period (480 points, 20 days). The overall performance gap is substantial when the prediction length is greater than 48 points (the solid star in Fig. 1b)), where the MSE rises to unsatisfactory performance, the inference speed gets sharp drop, and the LSTM model starts to fail.

时间序列预测是许多领域的关键要素，例如传感器网络监测（帕帕迪米特里乌和于，2006年）、能源与智能电网管理、经济与金融（朱和沙沙，2002年）以及疾病传播分析（松原等人，2014年）。在这些场景中，我们可以利用大量关于过去行为的时间序列数据进行长期预测，即长序列时间序列预测（LSTF）。然而，现有方法大多是在短期问题设定下设计的，比如预测48个点或更少（霍赫赖特和施密德胡伯，1997年；李等人，2018年；余等人，2017年；刘等人，2019年；秦等人，2017年；文等人，2017年）。日益增长的长序列对模型的预测能力造成了巨大压力，这种趋势正制约着长序列时间序列预测的研究。作为一个实证例子，图1展示了在一个真实数据集上的预测结果，其中长短期记忆网络（LSTM）从短期（12个点，0.5天）到长期（480个点，20天）预测一个变电站的每小时温度。当预测长度大于48个点时（图1b中的实心星），整体性能差距显著，均方误差（MSE）上升到不理想的水平，推理速度急剧下降，并且LSTM模型开始失效。

<!-- Media -->

<!-- figureText: (a) Sequence Forecasting. The predict sequence length (b) Run LSTM on sequences. -->

<img src="https://cdn.noedgeai.com/01957ae6-5cfc-72b0-ad59-2c1c4e1a17b2_0.jpg?x=940&y=638&w=701&h=237&r=0"/>

Figure 1: (a) LSTF can cover an extended period than the short sequence predictions, making vital distinction in policy-planning and investment-protecting. (b) The prediction capacity of existing methods limits LSTF's performance. E.g., starting from length=48, MSE rises unacceptably high, and the inference speed drops rapidly.

图1：(a) 长序列时间序列预测（LSTF）相比短序列预测能够覆盖更长的时间段，这在政策规划和投资保护方面有着至关重要的区别。(b) 现有方法的预测能力限制了长序列时间序列预测（LSTF）的性能。例如，从长度为48开始，均方误差（MSE）高得令人难以接受，并且推理速度迅速下降。

<!-- Media -->

The major challenge for LSTF is to enhance the prediction capacity to meet the increasingly long sequence demand, which requires (a) extraordinary long-range alignment ability and (b) efficient operations on long sequence inputs and outputs. Recently, Transformer models have shown superior performance in capturing long-range dependency than RNN models. The self-attention mechanism can reduce the maximum length of network signals traveling paths into the theoretical shortest $\mathcal{O}\left( 1\right)$ and avoid the recurrent structure, whereby Transformer shows great potential for the LSTF problem. Nevertheless, the self-attention mechanism violates requirement (b) due to its $L$ -quadratic computation and memory consumption on $L$ -length inputs/outputs. Some large-scale Transformer models pour resources and yield impressive results on NLP tasks (Brown et al. 2020), but the training on dozens of GPUs and expensive deploying cost make theses models unaffordable on real-world LSTF problem. The efficiency of the self-attention mechanism and Transformer architecture becomes the bottleneck of applying them to LSTF problems. Thus, in this paper, we seek to answer the question: can we improve Transformer models to be computation, memory, and architecture efficient, as well as maintaining higher prediction capacity?

长序列时间序列预测（LSTF）面临的主要挑战是增强预测能力，以满足日益增长的长序列需求，这需要（a）出色的长距离对齐能力，以及（b）对长序列输入和输出进行高效操作。最近，Transformer模型在捕捉长距离依赖关系方面表现出比循环神经网络（RNN）模型更优越的性能。自注意力机制可以将网络信号传播路径的最大长度缩短到理论最短$\mathcal{O}\left( 1\right)$，并避免循环结构，因此Transformer在长序列时间序列预测问题上显示出巨大潜力。然而，自注意力机制由于其在长度为$L$的输入/输出上具有$L$的二次方计算量和内存消耗，不满足要求（b）。一些大规模Transformer模型投入大量资源，在自然语言处理（NLP）任务中取得了令人瞩目的成果（Brown等人，2020），但在数十个图形处理器（GPU）上进行训练以及高昂的部署成本，使得这些模型在实际的长序列时间序列预测问题中难以承受。自注意力机制和Transformer架构的效率成为将它们应用于长序列时间序列预测问题的瓶颈。因此，在本文中，我们试图回答这样一个问题：我们能否改进Transformer模型，使其在计算、内存和架构方面更高效，同时保持较高的预测能力？

Vanilla Transformer (Vaswani et al. 2017) has three significant limitations when solving the LSTF problem:

香草Transformer（Vanilla Transformer，瓦斯瓦尼等人，2017年）在解决长序列时间序列预测（LSTF）问题时有三个显著的局限性：

1. The quadratic computation of self-attention. The atom operation of self-attention mechanism, namely canonical dot-product, causes the time complexity and memory usage per layer to be $\mathcal{O}\left( {L}^{2}\right)$ .

1. 自注意力机制的二次计算。自注意力机制的原子操作，即经典的点积运算，导致每层的时间复杂度和内存使用量为$\mathcal{O}\left( {L}^{2}\right)$。

2. The memory bottleneck in stacking layers for long inputs. The stack of $J$ encoder/decoder layers makes total memory usage to be $\mathcal{O}\left( {J \cdot  {L}^{2}}\right)$ ,which limits the model scalability in receiving long sequence inputs.

2. 处理长输入时堆叠层的内存瓶颈。$J$个编码器/解码器层的堆叠使得总内存使用量达到$\mathcal{O}\left( {J \cdot  {L}^{2}}\right)$，这限制了模型接收长序列输入时的可扩展性。

3. The speed plunge in predicting long outputs. Dynamic decoding of vanilla Transformer makes the step-by-step inference as slow as RNN-based model (Fig. 1b)).

3. 预测长输出时速度骤降。香草Transformer的动态解码使得逐步推理的速度和基于循环神经网络（RNN）的模型一样慢（图1b）。

There are some prior works on improving the efficiency of self-attention. The Sparse Transformer (Child et al. 2019), LogSparse Transformer (Li et al. 2019), and Longformer (Beltagy, Peters, and Cohan 2020) all use a heuristic method to tackle limitation 1 and reduce the complexity of self-attention mechanism to $\mathcal{O}\left( {L\log L}\right)$ ,where their efficiency gain is limited (Qiu et al. 2019). Reformer (Kitaev, Kaiser, and Levskaya 2019) also achieves $\mathcal{O}\left( {L\log L}\right)$ with locally-sensitive hashing self-attention, but it only works on extremely long sequences. More recently, Linformer (Wang et al. 2020) claims a linear complexity $\mathcal{O}\left( L\right)$ ,but the project matrix can not be fixed for real-world long sequence input,which may have the risk of degradation to $\mathcal{O}\left( {L}^{2}\right)$ . Transformer-XL (Dai et al. 2019) and Compressive Transformer (Rae et al. 2019) use auxiliary hidden states to capture long-range dependency, which could amplify limitation 1 and be adverse to break the efficiency bottleneck. All these works mainly focus on limitation 1 , and the limitation 2&3 remains unsolved in the LSTF problem. To enhance the prediction capacity, we tackle all these limitations and achieve improvement beyond efficiency in the proposed Informer.

有一些关于提高自注意力机制效率的前期工作。稀疏Transformer（Sparse Transformer，Child等人，2019年）、对数稀疏Transformer（LogSparse Transformer，Li等人，2019年）和长序列Transformer（Longformer，Beltagy、Peters和Cohan，2020年）都采用启发式方法来解决限制1，并将自注意力机制的复杂度降低到$\mathcal{O}\left( {L\log L}\right)$，不过它们的效率提升有限（Qiu等人，2019年）。改革者Transformer（Reformer，Kitaev、Kaiser和Levskaya，2019年）也通过局部敏感哈希自注意力机制实现了$\mathcal{O}\left( {L\log L}\right)$，但它仅适用于极长序列。最近，线性Transformer（Linformer，Wang等人，2020年）宣称具有线性复杂度$\mathcal{O}\left( L\right)$，但对于现实世界中的长序列输入，投影矩阵无法固定，这可能存在退化为$\mathcal{O}\left( {L}^{2}\right)$的风险。Transformer - XL（Dai等人，2019年）和压缩Transformer（Compressive Transformer，Rae等人，2019年）使用辅助隐藏状态来捕捉长距离依赖关系，这可能会加剧限制1，不利于突破效率瓶颈。所有这些工作主要关注限制1，而在长序列时间序列预测（LSTF）问题中，限制2和3仍未得到解决。为了增强预测能力，我们解决了所有这些限制，并在提出的Informer中实现了超越效率的改进。

To this end, our work delves explicitly into these three issues. We investigate the sparsity in the self-attention mechanism, make improvements of network components, and conduct extensive experiments. The contributions of this paper are summarized as follows:

为此，我们的工作明确地深入研究了这三个问题。我们研究了自注意力机制中的稀疏性，对网络组件进行了改进，并进行了广泛的实验。本文的贡献总结如下：

- We propose Informer to successfully enhance the prediction capacity in the LSTF problem, which validates the Transformer-like model's potential value to capture individual long-range dependency between long sequence time-series outputs and inputs.

- 我们提出了Informer（信息者）模型，成功增强了长序列时间序列预测（LSTF）问题中的预测能力，这验证了类Transformer模型在捕捉长序列时间序列输出与输入之间个体长程依赖关系方面的潜在价值。

- We propose ProbSparse self-attention mechanism to efficiently replace the canonical self-attention. It achieves the $\mathcal{O}\left( {L\log L}\right)$ time complexity and $\mathcal{O}\left( {L\log L}\right)$ memory usage on dependency alignments.

- 我们提出了ProbSparse（概率稀疏）自注意力机制，以高效地替代传统的自注意力机制。它在依赖对齐方面实现了$\mathcal{O}\left( {L\log L}\right)$的时间复杂度和$\mathcal{O}\left( {L\log L}\right)$的内存使用。

- We propose self-attention distilling operation to privilege dominating attention scores in $J$ -stacking layers and sharply reduce the total space complexity to be $\mathcal{O}((2 -$ $\epsilon )L\log L$ ),which helps receiving long sequence input.

- 我们提出了自注意力蒸馏操作，以在$J$堆叠层中突出主要的注意力得分，并将总空间复杂度大幅降低至$\mathcal{O}((2 -$ $\epsilon )L\log L$ ），这有助于接收长序列输入。

- We propose generative style decoder to acquire long sequence output with only one forward step needed, simultaneously avoiding cumulative error spreading during the inference phase.

- 我们提出了生成式风格解码器，仅需一步前向传播即可获得长序列输出，同时避免了推理阶段累积误差的传播。

<!-- Media -->

<!-- figureText: Concatenated Feature Map Outputs WIMPLECTION Fully Connected Layer Decoder Multi-head Attention Masked Multi-head\\ ProbSparse Self-attentior Inputs: ${\mathrm{X}}_{\mathrm{{de}}} = \left\{  {{\mathrm{X}}_{\text{token }},{\mathrm{X}}_{0}}\right\}$ Encoder Multi-head ProbSparse Self-attention Multi-head ProbSparse Self-attention Inputs: ${\mathrm{X}}_{\text{en }}$ -->

<img src="https://cdn.noedgeai.com/01957ae6-5cfc-72b0-ad59-2c1c4e1a17b2_1.jpg?x=946&y=154&w=679&h=390&r=0"/>

Figure 2: Informer model overview. Left: The encoder receives massive long sequence inputs (green series). We replace canonical self-attention with the proposed ProbSparse self-attention. The blue trapezoid is the self-attention distilling operation to extract dominating attention, reducing the network size sharply. The layer stacking replicas increase robustness. Right: The decoder receives long sequence inputs, pads the target elements into zero, measures the weighted attention composition of the feature map, and instantly predicts output elements (orange series) in a generative style.

图2：Informer模型概述。左图：编码器接收大量长序列输入（绿色序列）。我们用提出的ProbSparse自注意力机制取代了传统的自注意力机制。蓝色梯形表示自注意力蒸馏操作，用于提取主要注意力，大幅减小网络规模。层堆叠副本增强了鲁棒性。右图：解码器接收长序列输入，将目标元素填充为零，衡量特征图的加权注意力组合，并以生成式风格即时预测输出元素（橙色序列）。

<!-- Media -->

## 2 Preliminary

## 2 预备知识

We first provide the LSTF problem definition. Under the rolling forecasting setting with a fixed size window, we have the input ${\mathcal{X}}^{t} = \left\{  {{\mathbf{x}}_{1}^{t},\ldots ,{\mathbf{x}}_{{L}_{x}}^{t} \mid  {\mathbf{x}}_{i}^{t} \in  {\mathbb{R}}^{{d}_{x}}}\right\}$ at time $t$ , and the output is to predict corresponding sequence ${\mathcal{Y}}^{t} =$ $\left\{  {{\mathbf{y}}_{1}^{t},\ldots ,{\mathbf{y}}_{{L}_{y}}^{t} \mid  {\mathbf{y}}_{i}^{t} \in  {\mathbb{R}}^{{d}_{y}}}\right\}$ . The LSTF problem encourages a longer output’s length ${L}_{y}$ than previous works (Cho et al. 2014; Sutskever, Vinyals, and Le 2014) and the feature dimension is not limited to univariate case $\left( {{d}_{y} \geq  1}\right)$ .

我们首先给出长序列时间预测（LSTF）问题的定义。在固定大小窗口的滚动预测设置下，我们在时间$t$有输入${\mathcal{X}}^{t} = \left\{  {{\mathbf{x}}_{1}^{t},\ldots ,{\mathbf{x}}_{{L}_{x}}^{t} \mid  {\mathbf{x}}_{i}^{t} \in  {\mathbb{R}}^{{d}_{x}}}\right\}$，输出是预测相应的序列${\mathcal{Y}}^{t} =$$\left\{  {{\mathbf{y}}_{1}^{t},\ldots ,{\mathbf{y}}_{{L}_{y}}^{t} \mid  {\mathbf{y}}_{i}^{t} \in  {\mathbb{R}}^{{d}_{y}}}\right\}$。与以往的工作（赵等人，2014年；苏斯克维、维尼亚尔斯和勒，2014年）相比，长序列时间预测问题要求输出序列的长度${L}_{y}$更长，并且特征维度不限于单变量情况$\left( {{d}_{y} \geq  1}\right)$。

Encoder-decoder architecture Many popular models are devised to "encode" the input representations ${\mathcal{X}}^{t}$ into a hidden state representations ${\mathcal{H}}^{t}$ and "decode" an output representations ${\mathcal{Y}}^{t}$ from ${\mathcal{H}}^{t} = \left\{  {{\mathbf{h}}_{1}^{t},\ldots ,{\mathbf{h}}_{{L}_{h}}^{t}}\right\}$ . The inference involves a step-by-step process named "dynamic decoding", where the decoder computes a new hidden state ${\mathbf{h}}_{k + 1}^{t}$ from the previous state ${\mathbf{h}}_{k}^{t}$ and other necessary outputs from $k$ -th step then predict the $\left( {k + 1}\right)$ -th sequence ${\mathbf{y}}_{k + 1}^{t}$ .

编码器 - 解码器架构 许多流行的模型旨在将输入表示 ${\mathcal{X}}^{t}$ “编码”为隐藏状态表示 ${\mathcal{H}}^{t}$，并从 ${\mathcal{H}}^{t} = \left\{  {{\mathbf{h}}_{1}^{t},\ldots ,{\mathbf{h}}_{{L}_{h}}^{t}}\right\}$ “解码”出输出表示 ${\mathcal{Y}}^{t}$。推理过程涉及一个名为“动态解码”的逐步过程，在该过程中，解码器根据前一个状态 ${\mathbf{h}}_{k}^{t}$ 和 $k$ 步的其他必要输出计算出新的隐藏状态 ${\mathbf{h}}_{k + 1}^{t}$，然后预测第 $\left( {k + 1}\right)$ 个序列 ${\mathbf{y}}_{k + 1}^{t}$。

Input Representation A uniform input representation is given to enhance the global positional context and local temporal context of the time-series inputs. To avoid trivializing description, we put the details in Appendix B.

输入表示 为增强时间序列输入的全局位置上下文和局部时间上下文，采用统一的输入表示。为避免描述过于琐碎，我们将细节放在附录B中。

## 3 Methodology

## 3 方法

Existing methods for time-series forecasting can be roughly grouped into two categories ${}^{1}$ . Classical time-series models serve as a reliable workhorse for time-series forecasting (Box et al. 2015; Ray 1990; Seeger et al. 2017; Seeger, Salinas, and Flunkert 2016), and deep learning techniques mainly develop an encoder-decoder prediction paradigm by using RNN and their variants (Hochreiter and Schmidhuber 1997; Li et al. 2018; Yu et al. 2017). Our proposed Informer holds the encoder-decoder architecture while targeting the LSTF problem. Please refer to Fig. 2 for an overview and the following sections for details.

现有的时间序列预测方法大致可分为两类 ${}^{1}$。经典的时间序列模型是时间序列预测的可靠工具（博克斯（Box）等人，2015年；雷（Ray），1990年；西格（Seeger）等人，2017年；西格（Seeger）、萨利纳斯（Salinas）和弗伦克特（Flunkert），2016年），而深度学习技术主要通过使用循环神经网络（RNN）及其变体来构建编码器 - 解码器预测范式（霍赫赖特（Hochreiter）和施密德胡伯（Schmidhuber），1997年；李（Li）等人，2018年；余（Yu）等人，2017年）。我们提出的Informer模型采用编码器 - 解码器架构，旨在解决长序列时间序列预测（LSTF）问题。整体概述请参考图2，详细内容请参阅后续章节。

---

<!-- Footnote -->

${}^{1}$ Related work is in Appendix A due to space limitation.

${}^{1}$ 由于篇幅限制，相关工作见附录A。

<!-- Footnote -->

---

## Efficient Self-attention Mechanism

## 高效自注意力机制

The canonical self-attention in (Vaswani et al. 2017) is defined based on the tuple inputs, i.e, query, key and value, which performs the scaled dot-product as $\mathcal{A}\left( {\mathbf{Q},\mathbf{K},\mathbf{V}}\right)  =$ Softmax $\left( {{\mathbf{{QK}}}^{\top }/\sqrt{d}}\right) \mathbf{V}$ ,where $\mathbf{Q} \in  {\mathbb{R}}^{{L}_{Q} \times  d},\mathbf{K} \in  {\mathbb{R}}^{{L}_{K} \times  d}$ , $\mathbf{V} \in  {\mathbb{R}}^{{L}_{V} \times  d}$ and $d$ is the input dimension. To further discuss the self-attention mechanism,let ${\mathbf{q}}_{i},{\mathbf{k}}_{i},{\mathbf{v}}_{i}$ stand for the $i$ -th row in $\mathbf{Q},\mathbf{K},\mathbf{V}$ respectively. Following the formulation in (Tsai et al. 2019),the $i$ -th query’s attention is defined as a kernel smoother in a probability form:

（Vaswani等人，2017年）中规范的自注意力机制是基于元组输入（即查询（query）、键（key）和值（value））定义的，它执行缩放点积操作，如$\mathcal{A}\left( {\mathbf{Q},\mathbf{K},\mathbf{V}}\right)  =$ Softmax $\left( {{\mathbf{{QK}}}^{\top }/\sqrt{d}}\right) \mathbf{V}$ ，其中$\mathbf{Q} \in  {\mathbb{R}}^{{L}_{Q} \times  d},\mathbf{K} \in  {\mathbb{R}}^{{L}_{K} \times  d}$ 、$\mathbf{V} \in  {\mathbb{R}}^{{L}_{V} \times  d}$ 和$d$ 是输入维度。为了进一步讨论自注意力机制，让${\mathbf{q}}_{i},{\mathbf{k}}_{i},{\mathbf{v}}_{i}$ 分别表示$\mathbf{Q},\mathbf{K},\mathbf{V}$ 中的第$i$ 行。根据（Tsai等人，2019年）中的公式，第$i$ 个查询的注意力被定义为概率形式的核平滑器：

$$
\mathcal{A}\left( {{\mathbf{q}}_{i},\mathbf{K},\mathbf{V}}\right)  = \mathop{\sum }\limits_{j}\frac{k\left( {{\mathbf{q}}_{i},{\mathbf{k}}_{j}}\right) }{\mathop{\sum }\limits_{l}k\left( {{\mathbf{q}}_{i},{\mathbf{k}}_{l}}\right) }{\mathbf{v}}_{j} = {\mathbb{E}}_{p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right) }\left\lbrack  {\mathbf{v}}_{j}\right\rbrack  , \tag{1}
$$

where $p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)  = k\left( {{\mathbf{q}}_{i},{\mathbf{k}}_{j}}\right) /\mathop{\sum }\limits_{l}k\left( {{\mathbf{q}}_{i},{\mathbf{k}}_{l}}\right)$ and $k\left( {{\mathbf{q}}_{i},{\mathbf{k}}_{j}}\right)$ selects the asymmetric exponential kernel $\exp \left( {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right)$ . The self-attention combines the values and acquires outputs based on computing the probability $p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)$ . It requires the quadratic times dot-product computation and $\mathcal{O}\left( {{L}_{Q}{L}_{K}}\right)$ memory usage, which is the major drawback when enhancing prediction capacity.

其中 $p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)  = k\left( {{\mathbf{q}}_{i},{\mathbf{k}}_{j}}\right) /\mathop{\sum }\limits_{l}k\left( {{\mathbf{q}}_{i},{\mathbf{k}}_{l}}\right)$ 和 $k\left( {{\mathbf{q}}_{i},{\mathbf{k}}_{j}}\right)$ 选择了非对称指数核函数 $\exp \left( {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right)$。自注意力机制结合这些值，并基于计算概率 $p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)$ 来获取输出。它需要进行二次点积计算，并且内存使用量为 $\mathcal{O}\left( {{L}_{Q}{L}_{K}}\right)$，这在提升预测能力时是主要的缺点。

Some previous attempts have revealed that the distribution of self-attention probability has potential sparsity, and they have designed "selective" counting strategies on all $p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)$ without significantly affecting the performance. The Sparse Transformer (Child et al. 2019) incorporates both the row outputs and column inputs, in which the sparsity arises from the separated spatial correlation. The LogSparse Transformer (Li et al. 2019) notices the cyclical pattern in self-attention and forces each cell to attend to its previous one by an exponential step size. The Longformer (Beltagy, Peters, and Cohan 2020) extends previous two works to more complicated sparse configuration. However, they are limited to theoretical analysis from following heuristic methods and tackle each multi-head self-attention with the same strategy, which narrows their further improvement.

此前的一些尝试表明，自注意力概率的分布具有潜在的稀疏性，并且它们在所有$p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)$上设计了“选择性”计数策略，而不会显著影响性能。稀疏Transformer（Child等人，2019年）同时结合了行输出和列输入，其中稀疏性源于分离的空间相关性。对数稀疏Transformer（Li等人，2019年）注意到自注意力中的周期性模式，并通过指数步长迫使每个单元关注其前一个单元。长序列Transformer（Beltagy、Peters和Cohan，2020年）将前两项工作扩展到更复杂的稀疏配置。然而，它们仅限于通过以下启发式方法进行理论分析，并且用相同的策略处理每个多头自注意力，这限制了它们的进一步改进。

To motivate our approach, we first perform a qualitative assessment on the learned attention patterns of the canonical self-attention. The "sparsity" self-attention score forms a long tail distribution (see Appendix C for details), i.e., a few dot-product pairs contribute to the major attention, and others generate trivial attention. Then, the next question is how to distinguish them?

为了说明我们方法的动机，我们首先对经典自注意力机制所学习到的注意力模式进行了定性评估。“稀疏性”自注意力得分呈长尾分布（详情见附录C），即少数点积对贡献了主要的注意力，而其他点积对产生的注意力微不足道。那么，接下来的问题是如何区分它们呢？

Query Sparsity Measurement From Eq. 1,the $i$ -th query's attention on all the keys are defined as a probability $p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)$ and the output is its composition with values $\mathbf{v}$ . The dominant dot-product pairs encourage the corresponding query's attention probability distribution away from the uniform distribution. If $p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)$ is close to a uniform distribution $q\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)  = 1/{L}_{K}$ ,the self-attention becomes a trivial sum of values $\mathbf{V}$ and is redundant to the residential input. Naturally,the "likeness" between distribution $p$ and $q$ can be used to distinguish the "important" queries. We measure the "likeness" through Kullback-Leibler divergence ${KL}\left( {q\parallel p}\right)  = \ln \mathop{\sum }\limits_{{l = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{l}^{\top }/\sqrt{d}} - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d} -$ $\ln {L}_{K}$ . Dropping the constant,we define the $i$ -th query’s sparsity measurement as

查询稀疏性度量 从公式1可知，第 $i$ 个查询对所有键的注意力被定义为一个概率 $p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)$，其输出是它与值 $\mathbf{v}$ 的组合。占主导地位的点积对会使相应查询的注意力概率分布偏离均匀分布。如果 $p\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)$ 接近均匀分布 $q\left( {{\mathbf{k}}_{j} \mid  {\mathbf{q}}_{i}}\right)  = 1/{L}_{K}$，则自注意力就变成了值 $\mathbf{V}$ 的简单求和，并且对于剩余输入而言是多余的。自然地，分布 $p$ 和 $q$ 之间的“相似度”可用于区分“重要”的查询。我们通过KL散度（Kullback-Leibler divergence）${KL}\left( {q\parallel p}\right)  = \ln \mathop{\sum }\limits_{{l = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{l}^{\top }/\sqrt{d}} - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d} -$ $\ln {L}_{K}$ 来度量这种“相似度”。去掉常数项后，我们将第 $i$ 个查询的稀疏性度量定义为

$$
M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  = \ln \mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{\frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}} - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}\;, \tag{2}
$$

where the first term is the Log-Sum-Exp (LSE) of ${\mathbf{q}}_{i}$ on all the keys, and the second term is the arithmetic mean on them. If the $i$ -th query gains a larger $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ ,its attention probability $p$ is more "diverse" and has a high chance to contain the dominate dot-product pairs in the header field of the long tail self-attention distribution.

其中第一项是 ${\mathbf{q}}_{i}$ 在所有键上的对数求和指数（Log - Sum - Exp，LSE），第二项是它们的算术平均值。如果第 $i$ 个查询获得了更大的 $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$，其注意力概率 $p$ 会更 “多样化”，并且很有可能包含长尾自注意力分布头部字段中的主导点积对。

ProbSparse Self-attention Based on the proposed measurement, we have the ProbSparse self-attention by allowing each key to only attend to the $u$ dominant queries:

基于所提出的度量方法，我们通过允许每个键仅关注 $u$ 个主导查询，得到了概率稀疏（ProbSparse）自注意力机制：

$$
\mathcal{A}\left( {\mathbf{Q},\mathbf{K},\mathbf{V}}\right)  = \operatorname{Softmax}\left( \frac{\overline{\mathbf{Q}}{\mathbf{K}}^{\top }}{\sqrt{d}}\right) \mathbf{V} \tag{3}
$$

where $\overline{\mathbf{Q}}$ is a sparse matrix of the same size of $\mathbf{q}$ and it only contains the Top- $u$ queries under the sparsity measurement $M\left( {\mathbf{q},\mathbf{K}}\right)$ . Controlled by a constant sampling factor $c$ , we set $u = c \cdot  \ln {L}_{Q}$ ,which makes the ProbSparse self-attention only need to calculate $\mathcal{O}\left( {\ln {L}_{Q}}\right)$ dot-product for each query-key lookup and the layer memory usage maintains $\mathcal{O}\left( {{L}_{K}\ln {L}_{Q}}\right)$ . Under the multi-head perspective,this attention generates different sparse query-key pairs for each head, which avoids severe information loss in return.

其中 $\overline{\mathbf{Q}}$ 是一个与 $\mathbf{q}$ 大小相同的稀疏矩阵，它仅包含在稀疏性度量 $M\left( {\mathbf{q},\mathbf{K}}\right)$ 下的前 $u$ 个查询。在常数采样因子 $c$ 的控制下，我们设置 $u = c \cdot  \ln {L}_{Q}$，这使得概率稀疏自注意力机制（ProbSparse self-attention）每次查询 - 键查找仅需计算 $\mathcal{O}\left( {\ln {L}_{Q}}\right)$ 次点积，并且该层的内存使用量保持为 $\mathcal{O}\left( {{L}_{K}\ln {L}_{Q}}\right)$。在多头视角下，这种注意力机制为每个头生成不同的稀疏查询 - 键对，从而避免了严重的信息损失。

However, the traversing of all the queries for the measurement $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ requires calculating each dot-product pairs, i.e.,quadratically $\mathcal{O}\left( {{L}_{Q}{L}_{K}}\right)$ ,besides the LSE operation has the potential numerical stability issue. Motivated by this, we propose an empirical approximation for the efficient acquisition of the query sparsity measurement.

然而，对测量 $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ 的所有查询进行遍历需要计算每一对点积，即复杂度为二次方 $\mathcal{O}\left( {{L}_{Q}{L}_{K}}\right)$，此外，对数和指数（LSE）运算存在潜在的数值稳定性问题。受此启发，我们提出一种经验近似方法，以高效获取查询稀疏性测量值。

Lemma 1. For each query ${\mathbf{q}}_{i} \in  {\mathbb{R}}^{d}$ and ${\mathbf{k}}_{j} \in  {\mathbb{R}}^{d}$ in the keys set $\mathbf{K}$ ,we have the bound as $\ln {L}_{K} \leq  M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  \leq$ $\mathop{\max }\limits_{j}\left\{  {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right\}   - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left\{  {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right\}   + \ln {L}_{K}$ . When ${\mathbf{q}}_{i} \in  \mathbf{K}$ ,it also holds.

引理1. 对于键集$\mathbf{K}$中的每个查询${\mathbf{q}}_{i} \in  {\mathbb{R}}^{d}$和${\mathbf{k}}_{j} \in  {\mathbb{R}}^{d}$，我们有如下界：$\ln {L}_{K} \leq  M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  \leq$ $\mathop{\max }\limits_{j}\left\{  {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right\}   - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left\{  {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right\}   + \ln {L}_{K}$。当${\mathbf{q}}_{i} \in  \mathbf{K}$时，该结论同样成立。

From the Lemma 1 (proof is given in Appendix D.1), we propose the max-mean measurement as

根据引理1（证明见附录D.1），我们提出最大均值测量方法如下

$$
\bar{M}\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  = \mathop{\max }\limits_{j}\left\{  \frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}\right\}   - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}. \tag{4}
$$

The range of Top- $u$ approximately holds in the boundary relaxation with Proposition 1 (refers in Appendix D.2). Under the long tail distribution, we only need to randomly sample $U = {L}_{K}\ln {L}_{Q}$ dot-product pairs to calculate the $\bar{M}\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ ,i.e.,filling other pairs with zero. Then,we select sparse Top- $u$ from them as $\overline{\mathbf{Q}}$ . The max-operator in $\bar{M}\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ is less sensitive to zero values and is numerical stable. In practice, the input length of queries and keys are typically equivalent in the self-attention computation, i.e ${L}_{Q} = {L}_{K} = L$ such that the total ProbSparse self-attention time complexity and space complexity are $\mathcal{O}\left( {L\ln L}\right)$ .

在命题1（详见附录D.2）的边界松弛条件下，Top - $u$的范围大致成立。在长尾分布下，我们只需随机采样$U = {L}_{K}\ln {L}_{Q}$个点积对来计算$\bar{M}\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$，即把其他对填充为零。然后，我们从中选择稀疏的Top - $u$作为$\overline{\mathbf{Q}}$。$\bar{M}\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$中的最大运算符对零值不太敏感，并且在数值上是稳定的。实际上，在自注意力计算中，查询和键的输入长度通常是相等的，即${L}_{Q} = {L}_{K} = L$，因此ProbSparse自注意力的总时间复杂度和空间复杂度为$\mathcal{O}\left( {L\ln L}\right)$。

<!-- Media -->

<!-- figureText: Attention Block 1 -->

<img src="https://cdn.noedgeai.com/01957ae6-5cfc-72b0-ad59-2c1c4e1a17b2_3.jpg?x=276&y=162&w=1247&h=491&r=0"/>

Figure 3: The single stack in Informer's encoder. (1) The horizontal stack stands for an individual one of the encoder replicas in Fig. 2). (2) The presented one is the main stack receiving the whole input sequence. Then the second stack takes half slices of the input, and the subsequent stacks repeat. (3) The red layers are dot-product matrixes, and they get cascade decrease by applying self-attention distilling on each layer. (4) Concatenate all stacks' feature maps as the encoder's output.

图3：Informer编码器中的单个堆栈。（1）水平堆栈代表图2中编码器副本中的一个独立堆栈。（2）图中展示的是接收整个输入序列的主堆栈。然后，第二个堆栈接收输入的一半切片，后续堆栈依此类推。（3）红色层是点积矩阵，通过在每一层应用自注意力蒸馏，它们的规模逐级减小。（4）将所有堆栈的特征图拼接起来作为编码器的输出。

<!-- Media -->

## Encoder: Allowing for Processing Longer Sequential Inputs under the Memory Usage Limitation

## 编码器：在内存使用限制下处理更长的序列输入

The encoder is designed to extract the robust long-range dependency of the long sequential inputs. After the input representation,the $t$ -th sequence input ${\mathcal{X}}^{t}$ has been shaped into a matrix ${\mathbf{X}}_{\text{en }}^{t} \in  {\mathbb{R}}^{{L}_{x} \times  {d}_{\text{model }}}$ . We give a sketch of the encoder in Fig. 3) for clarity.

编码器旨在提取长序列输入的稳健长程依赖关系。在完成输入表示后，第$t$个序列输入${\mathcal{X}}^{t}$已被转换为矩阵${\mathbf{X}}_{\text{en }}^{t} \in  {\mathbb{R}}^{{L}_{x} \times  {d}_{\text{model }}}$。为清晰起见，我们在图3中给出了编码器的示意图。

Self-attention Distilling As the natural consequence of the ProbSparse self-attention mechanism, the encoder's feature map has redundant combinations of value $\mathbf{V}$ . We use the distilling operation to privilege the superior ones with dominating features and make a focused self-attention feature map in the next layer. It trims the input's time dimension sharply,seeing the $n$ -heads weights matrix (overlapping red squares) of Attention blocks in Fig. 3). Inspired by the dilated convolution (Yu, Koltun, and Funkhouser 2017; Gupta and Rush 2017), our "distilling" procedure forwards from $j$ -th layer into $\left( {j + 1}\right)$ -th layer as:

自注意力蒸馏 作为ProbSparse自注意力机制的自然结果，编码器的特征图存在值$\mathbf{V}$的冗余组合。我们使用蒸馏操作来突出具有主导特征的优势特征，并在下一层中生成聚焦的自注意力特征图。它会大幅修剪输入的时间维度（见图3中注意力模块的$n$头权重矩阵（重叠的红色方块））。受扩张卷积（于（Yu）、科尔图恩（Koltun）和芬克豪泽（Funkhouser），2017年；古普塔（Gupta）和拉什（Rush），2017年）的启发，我们的“蒸馏”过程从第$j$层推进到第$\left( {j + 1}\right)$层，具体如下：

$$
{\mathbf{X}}_{j + 1}^{t} = \operatorname{MaxPool}\left( {\operatorname{ELU}\left( {\operatorname{Conv1d}\left( {\left\lbrack  {\mathbf{X}}_{j}^{t}\right\rbrack  }_{\mathrm{{AB}}}\right) }\right) }\right)  \tag{5}
$$

where ${\left\lbrack  \cdot \right\rbrack  }_{\mathrm{{AB}}}$ represents the attention block. It contains the Multi-head ProbSparse self-attention and the essential operations,where $\operatorname{Conv1d}\left( \cdot \right)$ performs an $1 - \mathrm{D}$ convolutional filters (kernel width=3) on time dimension with the ELU(.) activation function (Clevert, Unterthiner, and Hochreiter 2016). We add a max-pooling layer with stride 2 and down-sample ${\mathbf{X}}^{t}$ into its half slice after stacking a layer,which reduces the whole memory usage to be $\mathcal{O}\left( {\left( {2 - \epsilon }\right) L\log L}\right)$ , where $\epsilon$ is a small number. To enhance the robustness of the distilling operation, we build replicas of the main stack with halving inputs, and progressively decrease the number of self-attention distilling layers by dropping one layer at a time, like a pyramid in Fig. 2), such that their output dimension is aligned. Thus, we concatenate all the stacks' outputs and have the final hidden representation of encoder.

其中 ${\left\lbrack  \cdot \right\rbrack  }_{\mathrm{{AB}}}$ 表示注意力模块。它包含多头概率稀疏自注意力和必要的操作，其中 $\operatorname{Conv1d}\left( \cdot \right)$ 在时间维度上执行 $1 - \mathrm{D}$ 个卷积滤波器（核宽度 = 3），并使用 ELU(.) 激活函数（克莱弗特（Clevert）、安特蒂纳（Unterthiner）和霍赫赖特（Hochreiter），2016 年）。我们添加一个步长为 2 的最大池化层，在堆叠一层后将 ${\mathbf{X}}^{t}$ 下采样为其一半切片，这将整个内存使用量减少到 $\mathcal{O}\left( {\left( {2 - \epsilon }\right) L\log L}\right)$，其中 $\epsilon$ 是一个小数字。为了增强蒸馏操作的鲁棒性，我们构建主堆叠的副本并将输入减半，并通过每次删除一层逐步减少自注意力蒸馏层的数量，如图 2 所示呈金字塔状，以使它们的输出维度对齐。因此，我们将所有堆叠的输出连接起来，得到编码器的最终隐藏表示。

## Decoder: Generating Long Sequential Outputs Through One Forward Procedure

## 解码器：通过一次前向过程生成长序列输出

We use a standard decoder structure (Vaswani et al. 2017) in Fig. 2), and it is composed of a stack of two identical multihead attention layers. However, the generative inference is employed to alleviate the speed plunge in long prediction. We feed the decoder with the following vectors as

我们采用图2所示的标准解码器结构（瓦斯瓦尼等人，2017年），它由两个相同的多头注意力层堆叠而成。然而，采用生成式推理来缓解长预测中的速度骤降问题。我们将以下向量输入解码器作为

$$
{\mathbf{X}}_{\mathrm{{de}}}^{t} = \operatorname{Concat}\left( {{\mathbf{X}}_{\text{token }}^{t},{\mathbf{X}}_{\mathbf{0}}^{t}}\right)  \in  {\mathbb{R}}^{\left( {{L}_{\text{token }} + {L}_{y}}\right)  \times  {d}_{\text{model }}} \tag{6}
$$

where ${\mathbf{X}}_{\text{token }}^{t} \in  {\mathbb{R}}^{{L}_{\text{token }} \times  {d}_{\text{model }}}$ is the start token, ${\mathbf{X}}_{\mathbf{0}}^{t} \in$ ${\mathbb{R}}^{{L}_{y} \times  {d}_{\text{model }}}$ is a placeholder for the target sequence (set scalar as 0 ). Masked multi-head attention is applied in the ProbSparse self-attention computing by setting masked dot-products to $- \infty$ . It prevents each position from attending to coming positions, which avoids auto-regressive. A fully connected layer acquires the final output,and its outsize ${d}_{y}$ depends on whether we are performing a univariate forecasting or a multivariate one.

其中 ${\mathbf{X}}_{\text{token }}^{t} \in  {\mathbb{R}}^{{L}_{\text{token }} \times  {d}_{\text{model }}}$ 是起始标记，${\mathbf{X}}_{\mathbf{0}}^{t} \in$ ${\mathbb{R}}^{{L}_{y} \times  {d}_{\text{model }}}$ 是目标序列的占位符（将标量设为 0）。在 ProbSparse 自注意力计算中，通过将掩码点积设置为 $- \infty$ 来应用掩码多头注意力。它防止每个位置关注后续位置，从而避免自回归。全连接层获取最终输出，其输出维度 ${d}_{y}$ 取决于我们是进行单变量预测还是多变量预测。

Generative Inference Start token is efficiently applied in NLP's "dynamic decoding" (Devlin et al. 2018), and we extend it into a generative way. Instead of choosing specific flags as the token,we sample a ${L}_{\text{token }}$ long sequence in the input sequence, such as an earlier slice before the output sequence. Take predicting 168 points as an example (7-day temperature prediction in the experiment section), we will take the known 5 days before the target sequence as "start-token", and feed the generative-style inference decoder with ${\mathbf{X}}_{\mathrm{{de}}} = \left\{  {{\mathbf{X}}_{5d},{\mathbf{X}}_{\mathbf{0}}}\right\}$ . The ${\mathbf{X}}_{\mathbf{0}}$ contains target sequence’s time stamp, i.e., the context at the target week. Then our proposed decoder predicts outputs by one forward procedure rather than the time consuming "dynamic decoding" in the conventional encoder-decoder architecture. A detailed performance comparison is given in the computation efficiency section.

生成推理起始标记在自然语言处理（NLP）的“动态解码”（德夫林等人，2018年）中得到了有效应用，我们将其扩展为一种生成式方法。我们不是选择特定的标记作为起始标记，而是在输入序列中采样一个${L}_{\text{token }}$长的序列，例如输出序列之前的较早片段。以预测168个点为例（实验部分中的7天温度预测），我们将目标序列之前已知的5天作为“起始标记”，并将${\mathbf{X}}_{\mathrm{{de}}} = \left\{  {{\mathbf{X}}_{5d},{\mathbf{X}}_{\mathbf{0}}}\right\}$输入到生成式推理解码器中。${\mathbf{X}}_{\mathbf{0}}$包含目标序列的时间戳，即目标周的上下文信息。然后，我们提出的解码器通过一次前向过程来预测输出，而不是采用传统编码器 - 解码器架构中耗时的“动态解码”方法。计算效率部分给出了详细的性能比较。

Loss function We choose the MSE loss function on prediction w.r.t the target sequences, and the loss is propagated back from the decoder's outputs across the entire model.

损失函数 我们选择针对目标序列的预测采用均方误差（MSE）损失函数，并且损失从解码器的输出反向传播至整个模型。

<!-- Media -->

<table><tr><td>0.247 0.319 0.346 0.387 0.435 0.240 0.314 0.389 0.417 0.431 0.137 0.203 0.372 0.554 0.644 0.251 0.318 0.398 0.416 0.466 0.359 0.503 0.528 0.571 0.608</td><td>MethodsInformer</td><td>Informer ${}^{ \dagger  }$</td><td/><td>LogTrans</td><td/><td>Reformer</td><td>LSTMa</td><td/><td>DeepAR</td><td/><td>ARIMA</td><td>Prophet</td></tr><tr><td/><td>MetricMSEMAE</td><td>MSE</td><td>MAE</td><td>MSEE MAE</td><td>MSE</td><td>E MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSEMAE</td><td>MSEMAE</td></tr><tr><td rowspan="5">${\mathrm{{ETTh}}}_{1}$</td><td>240.098</td><td>0.092</td><td>0.246</td><td>0.1030.259</td><td>0.222</td><td>0.389</td><td>0.114</td><td>0.272</td><td>0.107</td><td>0.280</td><td>0.1080.284</td><td>0.1150.275</td></tr><tr><td>480.158</td><td>0.161</td><td>0.322</td><td>0.1670.328</td><td>0.284</td><td>0.445</td><td>0.193</td><td>0.358</td><td>0.162</td><td>0.327</td><td>0.1750.424</td><td>0.1680.330</td></tr><tr><td>1680.183</td><td>0.187</td><td>0.355</td><td>0.2070.375</td><td>1.522</td><td>1.191</td><td>0.236</td><td>0.392</td><td>0.239</td><td>0.422</td><td>0.3960.504</td><td>1.2240.763</td></tr><tr><td>3360.222</td><td>0.215</td><td>0.369</td><td>0.2300.398</td><td>1.860</td><td>1.124</td><td>0.590</td><td>0.698</td><td>0.445</td><td>0.552</td><td>0.4680.593</td><td>1.5491.820</td></tr><tr><td>7200.269</td><td>0.257</td><td>0.421</td><td>0.2730.463</td><td>2.112</td><td>1.436</td><td>0.683</td><td>0.768</td><td>0.658</td><td>0.707</td><td>0.6590.766</td><td>2.7353.253</td></tr><tr><td rowspan="5">${\mathrm{{ETTh}}}_{2}$</td><td>240.093</td><td>0.099</td><td>0.241</td><td>0.1020.255</td><td>0.263</td><td>0.437</td><td>0.155</td><td>0.307</td><td>0.098</td><td>0.263</td><td>3.5540.445</td><td>0.1990.381</td></tr><tr><td>480.155</td><td>0.159</td><td>0.317</td><td>0.1690.348</td><td>0.458</td><td>0.545</td><td>0.190</td><td>0.348</td><td>0.163</td><td>0.341</td><td>3.1900.474</td><td>0.3040.462</td></tr><tr><td>1680.232</td><td>0.235</td><td>0.390</td><td>0.2460.422</td><td>1.029</td><td>0.879</td><td>0.385</td><td>0.514</td><td>0.255</td><td>0.414</td><td>2.8000.595</td><td>2.1451.068</td></tr><tr><td>3360.263</td><td>0.258</td><td>0.423</td><td>0.2670.437</td><td>1.668</td><td>1.228</td><td>0.558</td><td>0.606</td><td>0.604</td><td>0.607</td><td>2.7530.738</td><td>2.0962.543</td></tr><tr><td>7200.277</td><td>0.285</td><td>0.442</td><td>0.3030.493</td><td>2.030</td><td>1.721</td><td>0.640</td><td>0.681</td><td>0.429</td><td>0.580</td><td>2.8781.044</td><td>3.3554.664</td></tr><tr><td rowspan="5">${\mathrm{{ETTm}}}_{1}$</td><td>240.030</td><td>0.034</td><td>0.160</td><td>0.0650.202</td><td>0.095</td><td>0.228</td><td>0.121</td><td>0.233</td><td>0.091</td><td>0.243</td><td>0.0900.206</td><td>0.1200.290</td></tr><tr><td>480.069</td><td>0.066</td><td>0.194</td><td>0.0780.220</td><td>0.249</td><td>0.390</td><td>0.305</td><td>0.411</td><td>0.219</td><td>0.362</td><td>0.1790.306</td><td>0.1330.305</td></tr><tr><td>960.194</td><td>0.187</td><td>0.384</td><td>0.1990.386</td><td>0.920</td><td>0.767</td><td>0.287</td><td>0.420</td><td>0.364</td><td>0.496</td><td>0.2720.399</td><td>0.1940.396</td></tr><tr><td>2880.401</td><td>0.409</td><td>0.548</td><td>0.4110.572</td><td>1.108</td><td>1.245</td><td>0.524</td><td>0.584</td><td>0.948</td><td>0.795</td><td>0.4620.558</td><td>0.4520.574</td></tr><tr><td>6720.512</td><td>0.519</td><td>0.665</td><td>0.5980.702</td><td>1.793</td><td>1.528</td><td>1.064</td><td>0.873</td><td>2.437</td><td>1.352</td><td>0.6390.697</td><td>2.7471.174</td></tr><tr><td rowspan="5">Weather</td><td>240.117</td><td>0.119</td><td>0.256</td><td>0.1360.279</td><td>0.231</td><td>0.401</td><td>0.131</td><td>0.254</td><td>0.128</td><td>0.274</td><td>0.2190.355</td><td>0.3020.433</td></tr><tr><td>480.178</td><td>0.185</td><td>0.316</td><td>0.2060.356</td><td>0.328</td><td>0.423</td><td>0.190</td><td>0.334</td><td>0.203</td><td>0.353</td><td>0.2730.409</td><td>0.4450.536</td></tr><tr><td>1680.266</td><td>0.269</td><td>0.404</td><td>0.3090.439</td><td>0.654</td><td>0.634</td><td>0.341</td><td>0.448</td><td>0.293</td><td>0.451</td><td>0.5030.599</td><td>2.4411.142</td></tr><tr><td>3360.297</td><td>0.310</td><td>0.422</td><td>0.3590.484</td><td>1.792</td><td>1.093</td><td>0.456</td><td>0.554</td><td>0.585</td><td>0.644</td><td>0.7280.730</td><td>1.9872.468</td></tr><tr><td>7200.359</td><td>0.361</td><td>0.471</td><td>0.3880.499</td><td>2.087</td><td>1.534</td><td>0.866</td><td>0.809</td><td>0.499</td><td>0.596</td><td>1.0620.943</td><td>3.8591.144</td></tr><tr><td rowspan="5">ECL</td><td>480.239</td><td>0.238</td><td>0.368</td><td>0.2800.429</td><td>0.971</td><td>0.884</td><td>0.493</td><td>0.539</td><td>0.204</td><td>0.357</td><td>0.8790.764</td><td>0.5240.595</td></tr><tr><td>1680.447</td><td>0.442</td><td>0.514</td><td>0.4540.529</td><td>1.671</td><td>1.587</td><td>0.723</td><td>0.655</td><td>0.315</td><td>0.436</td><td>1.0320.833</td><td>2.7251.273</td></tr><tr><td>3360.489</td><td>0.501</td><td>0.552</td><td>0.5140.563</td><td>3.528</td><td>2.196</td><td>1.212</td><td>0.898</td><td>0.414</td><td>0.519</td><td>1.1360.876</td><td>2.2463.077</td></tr><tr><td>7200.540</td><td>0.543</td><td>0.578</td><td>0.5580.609</td><td>4.891</td><td>4.047</td><td>1.511</td><td>0.966</td><td>0.563</td><td>0.595</td><td>1.2510.933</td><td>4.2431.415</td></tr><tr><td>9600.582</td><td>0.594</td><td>0.638</td><td>0.6240.645</td><td>7.019</td><td>5.105</td><td>1.545</td><td>1.006</td><td>0.657</td><td>0.683</td><td>1.3700.982</td><td>6.9014.264</td></tr><tr><td/><td>Count32</td><td/><td>12</td><td>│0</td><td/><td>0</td><td>│</td><td>0│</td><td/><td>6</td><td>0</td><td>0</td></tr></table>

<table><tbody><tr><td>0.247 0.319 0.346 0.387 0.435 0.240 0.314 0.389 0.417 0.431 0.137 0.203 0.372 0.554 0.644 0.251 0.318 0.398 0.416 0.466 0.359 0.503 0.528 0.571 0.608</td><td>方法信息器（MethodsInformer）</td><td>信息器（Informer） ${}^{ \dagger  }$</td><td></td><td>对数变换（LogTrans）</td><td></td><td>改革器（Reformer）</td><td>长短期记忆网络变体（LSTMa）</td><td></td><td>深度自回归模型（DeepAR）</td><td></td><td>自回归积分滑动平均模型（ARIMA）</td><td>先知模型（Prophet）</td></tr><tr><td></td><td>指标均方误差平均绝对误差（MetricMSEMAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差 平均绝对误差（MSEE MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（E MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差与平均绝对误差（MSEMAE）</td><td>均方误差与平均绝对误差（MSEMAE）</td></tr><tr><td rowspan="5">${\mathrm{{ETTh}}}_{1}$</td><td>240.098</td><td>0.092</td><td>0.246</td><td>0.1030.259</td><td>0.222</td><td>0.389</td><td>0.114</td><td>0.272</td><td>0.107</td><td>0.280</td><td>0.1080.284</td><td>0.1150.275</td></tr><tr><td>480.158</td><td>0.161</td><td>0.322</td><td>0.1670.328</td><td>0.284</td><td>0.445</td><td>0.193</td><td>0.358</td><td>0.162</td><td>0.327</td><td>0.1750.424</td><td>0.1680.330</td></tr><tr><td>1680.183</td><td>0.187</td><td>0.355</td><td>0.2070.375</td><td>1.522</td><td>1.191</td><td>0.236</td><td>0.392</td><td>0.239</td><td>0.422</td><td>0.3960.504</td><td>1.2240.763</td></tr><tr><td>3360.222</td><td>0.215</td><td>0.369</td><td>0.2300.398</td><td>1.860</td><td>1.124</td><td>0.590</td><td>0.698</td><td>0.445</td><td>0.552</td><td>0.4680.593</td><td>1.5491.820</td></tr><tr><td>7200.269</td><td>0.257</td><td>0.421</td><td>0.2730.463</td><td>2.112</td><td>1.436</td><td>0.683</td><td>0.768</td><td>0.658</td><td>0.707</td><td>0.6590.766</td><td>2.7353.253</td></tr><tr><td rowspan="5">${\mathrm{{ETTh}}}_{2}$</td><td>240.093</td><td>0.099</td><td>0.241</td><td>0.1020.255</td><td>0.263</td><td>0.437</td><td>0.155</td><td>0.307</td><td>0.098</td><td>0.263</td><td>3.5540.445</td><td>0.1990.381</td></tr><tr><td>480.155</td><td>0.159</td><td>0.317</td><td>0.1690.348</td><td>0.458</td><td>0.545</td><td>0.190</td><td>0.348</td><td>0.163</td><td>0.341</td><td>3.1900.474</td><td>0.3040.462</td></tr><tr><td>1680.232</td><td>0.235</td><td>0.390</td><td>0.2460.422</td><td>1.029</td><td>0.879</td><td>0.385</td><td>0.514</td><td>0.255</td><td>0.414</td><td>2.8000.595</td><td>2.1451.068</td></tr><tr><td>3360.263</td><td>0.258</td><td>0.423</td><td>0.2670.437</td><td>1.668</td><td>1.228</td><td>0.558</td><td>0.606</td><td>0.604</td><td>0.607</td><td>2.7530.738</td><td>2.0962.543</td></tr><tr><td>7200.277</td><td>0.285</td><td>0.442</td><td>0.3030.493</td><td>2.030</td><td>1.721</td><td>0.640</td><td>0.681</td><td>0.429</td><td>0.580</td><td>2.8781.044</td><td>3.3554.664</td></tr><tr><td rowspan="5">${\mathrm{{ETTm}}}_{1}$</td><td>240.030</td><td>0.034</td><td>0.160</td><td>0.0650.202</td><td>0.095</td><td>0.228</td><td>0.121</td><td>0.233</td><td>0.091</td><td>0.243</td><td>0.0900.206</td><td>0.1200.290</td></tr><tr><td>480.069</td><td>0.066</td><td>0.194</td><td>0.0780.220</td><td>0.249</td><td>0.390</td><td>0.305</td><td>0.411</td><td>0.219</td><td>0.362</td><td>0.1790.306</td><td>0.1330.305</td></tr><tr><td>960.194</td><td>0.187</td><td>0.384</td><td>0.1990.386</td><td>0.920</td><td>0.767</td><td>0.287</td><td>0.420</td><td>0.364</td><td>0.496</td><td>0.2720.399</td><td>0.1940.396</td></tr><tr><td>2880.401</td><td>0.409</td><td>0.548</td><td>0.4110.572</td><td>1.108</td><td>1.245</td><td>0.524</td><td>0.584</td><td>0.948</td><td>0.795</td><td>0.4620.558</td><td>0.4520.574</td></tr><tr><td>6720.512</td><td>0.519</td><td>0.665</td><td>0.5980.702</td><td>1.793</td><td>1.528</td><td>1.064</td><td>0.873</td><td>2.437</td><td>1.352</td><td>0.6390.697</td><td>2.7471.174</td></tr><tr><td rowspan="5">天气</td><td>240.117</td><td>0.119</td><td>0.256</td><td>0.1360.279</td><td>0.231</td><td>0.401</td><td>0.131</td><td>0.254</td><td>0.128</td><td>0.274</td><td>0.2190.355</td><td>0.3020.433</td></tr><tr><td>480.178</td><td>0.185</td><td>0.316</td><td>0.2060.356</td><td>0.328</td><td>0.423</td><td>0.190</td><td>0.334</td><td>0.203</td><td>0.353</td><td>0.2730.409</td><td>0.4450.536</td></tr><tr><td>1680.266</td><td>0.269</td><td>0.404</td><td>0.3090.439</td><td>0.654</td><td>0.634</td><td>0.341</td><td>0.448</td><td>0.293</td><td>0.451</td><td>0.5030.599</td><td>2.4411.142</td></tr><tr><td>3360.297</td><td>0.310</td><td>0.422</td><td>0.3590.484</td><td>1.792</td><td>1.093</td><td>0.456</td><td>0.554</td><td>0.585</td><td>0.644</td><td>0.7280.730</td><td>1.9872.468</td></tr><tr><td>7200.359</td><td>0.361</td><td>0.471</td><td>0.3880.499</td><td>2.087</td><td>1.534</td><td>0.866</td><td>0.809</td><td>0.499</td><td>0.596</td><td>1.0620.943</td><td>3.8591.144</td></tr><tr><td rowspan="5">欧洲气候评估与数据集（ECL）</td><td>480.239</td><td>0.238</td><td>0.368</td><td>0.2800.429</td><td>0.971</td><td>0.884</td><td>0.493</td><td>0.539</td><td>0.204</td><td>0.357</td><td>0.8790.764</td><td>0.5240.595</td></tr><tr><td>1680.447</td><td>0.442</td><td>0.514</td><td>0.4540.529</td><td>1.671</td><td>1.587</td><td>0.723</td><td>0.655</td><td>0.315</td><td>0.436</td><td>1.0320.833</td><td>2.7251.273</td></tr><tr><td>3360.489</td><td>0.501</td><td>0.552</td><td>0.5140.563</td><td>3.528</td><td>2.196</td><td>1.212</td><td>0.898</td><td>0.414</td><td>0.519</td><td>1.1360.876</td><td>2.2463.077</td></tr><tr><td>7200.540</td><td>0.543</td><td>0.578</td><td>0.5580.609</td><td>4.891</td><td>4.047</td><td>1.511</td><td>0.966</td><td>0.563</td><td>0.595</td><td>1.2510.933</td><td>4.2431.415</td></tr><tr><td>9600.582</td><td>0.594</td><td>0.638</td><td>0.6240.645</td><td>7.019</td><td>5.105</td><td>1.545</td><td>1.006</td><td>0.657</td><td>0.683</td><td>1.3700.982</td><td>6.9014.264</td></tr><tr><td></td><td>计数32（Count32）</td><td></td><td>12</td><td>│0</td><td></td><td>0</td><td>│</td><td>0│</td><td></td><td>6</td><td>0</td><td>0</td></tr></tbody></table>

Table 1: Univariate long sequence time-series forecasting results on four datasets (five cases).

表1：四个数据集（五种情况）的单变量长序列时间序列预测结果。

<!-- Media -->

## 4 Experiment

## 4 实验

## Datasets

## 数据集

We extensively perform experiments on four datasets, including 2 collected real-world datasets for LSTF and 2 public benchmark datasets.

我们在四个数据集上进行了广泛的实验，其中包括2个为长序列时间序列预测（LSTF）收集的真实世界数据集和2个公开的基准数据集。

ETT (Electricity Transformer Temperature) ${}^{2}$ : The ETT is a crucial indicator in the electric power long-term deployment. We collected 2-year data from two separated counties in China. To explore the granularity on the LSTF problem, we create separate datasets as $\left\{  {{\mathrm{{ETTh}}}_{1},{\mathrm{{ETTh}}}_{2}}\right\}$ for 1-hour-level and ${\mathrm{{ETTm}}}_{1}$ for 15-minute-level. Each data point consists of the target value "oil temperature" and 6 power load features. The train/val/test is ${12}/4/4$ months.

ETT（电力变压器温度，Electricity Transformer Temperature） ${}^{2}$ ：ETT是电力长期部署中的一个关键指标。我们从中国两个不同的县收集了两年的数据。为了探究长序列时间序列预测问题的粒度，我们分别创建了数据集 $\left\{  {{\mathrm{{ETTh}}}_{1},{\mathrm{{ETTh}}}_{2}}\right\}$ （1小时级）和 ${\mathrm{{ETTm}}}_{1}$ （15分钟级）。每个数据点由目标值“油温”和6个电力负荷特征组成。训练/验证/测试集的时间跨度为 ${12}/4/4$ 个月。

ECL (Electricity Consuming Load) ${}^{3}$ : It collects the electricity consumption (Kwh) of 321 clients. Due to the missing data (Li et al. 2019), we convert the dataset into hourly consumption of 2 years and set ’ $\mathrm{{MT}} - {320}^{ \circ  }$ as the target value. The train/val/test is 15/3/4 months.

ECL（电力消耗负荷） ${}^{3}$ ：它收集了321个客户的用电量（千瓦时）。由于存在缺失数据（Li等人，2019年），我们将该数据集转换为两年的每小时用电量，并将 ’ $\mathrm{{MT}} - {320}^{ \circ  }$ 设为目标值。训练集/验证集/测试集的时间分别为15个月/3个月/4个月。

Weather ${}^{4}$ : This dataset contains local climatological data for nearly 1,600 U.S. locations, 4 years from 2010 to 2013, where data points are collected every 1 hour. Each data point consists of the target value "wet bulb" and 11 climate features. The train/val/test is 28/10/10 months.

天气 ${}^{4}$ ：该数据集包含近1600个美国地点的当地气候数据，时间跨度为2010年至2013年的4年，数据点每1小时收集一次。每个数据点由目标值“湿球温度”和11个气候特征组成。训练集/验证集/测试集的时间分别为28个月/10个月/10个月。

## Experimental Details

## 实验细节

We briefly summarize basics, and more information on network components and setups are given in Appendix E.

我们简要总结了基础知识，有关网络组件和设置的更多信息见附录E。

Baselines: We have selected five time-series forecasting methods as comparison, including ARIMA (Ariyo, Adewumi, and Ayo 2014), Prophet (Taylor and Letham 2018), LSTMa (Bahdanau, Cho, and Bengio 2015), LST-net (Lai et al. 2018) and DeepAR (Flunkert, Salinas, and Gasthaus 2017). To better explore the ProbSparse self-attention's performance in our proposed Informer, we incorporate the canonical self-attention variant (Informer ${}^{ \dagger  }$ ), the efficient variant Reformer (Kitaev, Kaiser, and Levskaya 2019) and the most related work LogSparse self-attention (Li et al. 2019) in the experiments. The details of network components are given in Appendix E.1.

基线：我们选择了五种时间序列预测方法进行比较，包括自回归积分滑动平均模型（ARIMA）（Ariyo、Adewumi和Ayo，2014年）、先知模型（Prophet）（Taylor和Letham，2018年）、带注意力机制的长短期记忆网络（LSTMa）（Bahdanau、Cho和Bengio，2015年）、长短期记忆网络（LST - net）（Lai等人，2018年）和深度自回归模型（DeepAR）（Flunkert、Salinas和Gasthaus，2017年）。为了更好地探究概率稀疏自注意力机制（ProbSparse self - attention）在我们提出的信息者模型（Informer）中的性能，我们在实验中纳入了经典自注意力变体（信息者模型${}^{ \dagger  }$）、高效变体改革者模型（Reformer）（Kitaev、Kaiser和Levskaya，2019年）以及最相关的工作对数稀疏自注意力机制（LogSparse self - attention）（Li等人，2019年）。网络组件的详细信息见附录E.1。

Hyper-parameter tuning: We conduct grid search over the hyper-parameters, and detailed ranges are given in Appendix E.3. Informer contains a 3-layer stack and a 1- layer stack (1/4 input) in the encoder, and a 2-layer decoder. Our proposed methods are optimized with Adam optimizer,and its learning rate starts from $1{e}^{-4}$ ,decaying two times smaller every epoch. The total number of epochs is 8 with proper early stopping. We set the comparison methods as recommended, and the batch size is 32 . Setup: The input of each dataset is zero-mean normalized. Under the LSTF settings, we prolong the prediction windows size ${L}_{y}$ progressively,i.e., $\{ 1\mathrm{\;d},2\mathrm{\;d},7\mathrm{\;d},{14}\mathrm{\;d},{30}\mathrm{\;d},{40}\mathrm{\;d}\}$ in $\{$ ETTh,ECL,Weather $\} ,\{ 6\mathrm{\;h},{12}\mathrm{\;h},{24}\mathrm{\;h},{72}\mathrm{\;h},{168}\mathrm{\;h}\}$ in ETTm. Metrics: We use two evaluation metrics, including MSE $= \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}{\left( \mathbf{y} - \widehat{\mathbf{y}}\right) }^{2}$ and MAE $= \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}\left| {\mathbf{y} - \widehat{\mathbf{y}}}\right|$ on each prediction window (averaging for multivariate prediction),and roll the whole set with stride $= 1$ . Platform: All the models were trained/tested on a single Nvidia V100 32GB GPU. The source code is available at https://github.com/zhouhaoyi/Informer2020.

超参数调优：我们对超参数进行网格搜索，详细范围见附录E.3。Informer（信息者模型）的编码器包含一个3层堆栈和一个1层堆栈（1/4输入），解码器为2层。我们提出的方法使用Adam优化器进行优化，其学习率从$1{e}^{-4}$开始，每轮训练衰减为原来的一半。总训练轮数为8轮，并采用适当的提前停止策略。我们按照推荐设置对比方法，批量大小为32。设置：每个数据集的输入进行零均值归一化处理。在长序列时间序列预测（LSTF）设置下，我们逐步延长预测窗口大小${L}_{y}$，即在$\{$的ETTh、ECL、Weather数据集中为$\{ 1\mathrm{\;d},2\mathrm{\;d},7\mathrm{\;d},{14}\mathrm{\;d},{30}\mathrm{\;d},{40}\mathrm{\;d}\}$，在ETTm数据集中为$\} ,\{ 6\mathrm{\;h},{12}\mathrm{\;h},{24}\mathrm{\;h},{72}\mathrm{\;h},{168}\mathrm{\;h}\}$。指标：我们使用两种评估指标，包括均方误差（MSE，$= \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}{\left( \mathbf{y} - \widehat{\mathbf{y}}\right) }^{2}$）和平均绝对误差（MAE，$= \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}\left| {\mathbf{y} - \widehat{\mathbf{y}}}\right|$），对每个预测窗口进行评估（多变量预测取平均值），并以步长$= 1$滚动整个数据集。平台：所有模型均在单块Nvidia V100 32GB GPU上进行训练和测试。源代码可在https://github.com/zhouhaoyi/Informer2020获取。

---

<!-- Footnote -->

${}^{2}$ We collected the ETT dataset and published it at https:// github.com/zhouhaoyi/ETDataset

${}^{2}$ 我们收集了ETT数据集（Electricity Transformer Temperature dataset），并将其发布在https://github.com/zhouhaoyi/ETDataset

${}^{3}$ ECL dataset was acquired at https://archive.ics.uci.edu/ml/ datasets/ElectricityLoadDiagrams20112014

${}^{3}$ ECL数据集（Electricity Consumption Load dataset）是从https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014获取的

${}^{4}$ Weather dataset was acquired at https://www.ncei.noaa.gov/ data/local-climatological-data/

${}^{4}$ 气象数据集是从https://www.ncei.noaa.gov/data/local-climatological-data/获取的

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td colspan="2">Methods</td><td colspan="2">Informer</td><td colspan="2">Informer ${}^{ \dagger  }$</td><td colspan="2">LogTrans</td><td colspan="2">Reformer</td><td colspan="2">LSTMa</td><td colspan="2">LSTnet</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="5">${\mathrm{{ETTh}}}_{1}$</td><td>24</td><td>0.577</td><td>0.549</td><td>0.620</td><td>0.577</td><td>0.686</td><td>0.604</td><td>0.991</td><td>0.754</td><td>0.650</td><td>0.624</td><td>1.293</td><td>0.901</td></tr><tr><td>48</td><td>0.685</td><td>0.625</td><td>0.692</td><td>0.671</td><td>0.766</td><td>0.757</td><td>1.313</td><td>0.906</td><td>0.702</td><td>0.675</td><td>1.456</td><td>0.960</td></tr><tr><td>168</td><td>0.931</td><td>0.752</td><td>0.947</td><td>0.797</td><td>1.002</td><td>0.846</td><td>1.824</td><td>1.138</td><td>1.212</td><td>0.867</td><td>1.997</td><td>1.214</td></tr><tr><td>336</td><td>1.128</td><td>0.873</td><td>1.094</td><td>0.813</td><td>1.362</td><td>0.952</td><td>2.117</td><td>1.280</td><td>1.424</td><td>0.994</td><td>2.655</td><td>1.369</td></tr><tr><td>720</td><td>1.215</td><td>0.896</td><td>1.241</td><td>0.917</td><td>1.397</td><td>1.291</td><td>2.415</td><td>1.520</td><td>1.960</td><td>1.322</td><td>2.143</td><td>1.380</td></tr><tr><td rowspan="5">${\mathrm{{ETTh}}}_{2}$</td><td>24</td><td>0.720</td><td>0.665</td><td>0.753</td><td>0.727</td><td>0.828</td><td>0.750</td><td>1.531</td><td>1.613</td><td>1.143</td><td>0.813</td><td>2.742</td><td>1.457</td></tr><tr><td>48</td><td>1.457</td><td>1.001</td><td>1.461</td><td>1.077</td><td>1.806</td><td>1.034</td><td>1.871</td><td>1.735</td><td>1.671</td><td>1.221</td><td>3.567</td><td>1.687</td></tr><tr><td>168</td><td>3.489</td><td>1.515</td><td>3.485</td><td>1.612</td><td>4.070</td><td>1.681</td><td>4.660</td><td>1.846</td><td>4.117</td><td>1.674</td><td>3.242</td><td>2.513</td></tr><tr><td>336</td><td>2.723</td><td>1.340</td><td>2.626</td><td>1.285</td><td>3.875</td><td>1.763</td><td>4.028</td><td>1.688</td><td>3.434</td><td>1.549</td><td>2.544</td><td>2.591</td></tr><tr><td>720</td><td>3.467</td><td>1.473</td><td>3.548</td><td>1.495</td><td>3.913</td><td>1.552</td><td>5.381</td><td>2.015</td><td>3.963</td><td>1.788</td><td>4.625</td><td>3.709</td></tr><tr><td rowspan="5">ETTm1</td><td>24</td><td>0.323</td><td>0.369</td><td>0.306</td><td>0.371</td><td>0.419</td><td>0.412</td><td>0.724</td><td>0.607</td><td>0.621</td><td>0.629</td><td>1.968</td><td>1.170</td></tr><tr><td>48</td><td>0.494</td><td>0.503</td><td>0.465</td><td>0.470</td><td>0.507</td><td>0.583</td><td>1.098</td><td>0.777</td><td>1.392</td><td>0.939</td><td>1.999</td><td>1.215</td></tr><tr><td>96</td><td>0.678</td><td>0.614</td><td>0.681</td><td>0.612</td><td>0.768</td><td>0.792</td><td>1.433</td><td>0.945</td><td>1.339</td><td>0.913</td><td>2.762</td><td>1.542</td></tr><tr><td>288</td><td>1.056</td><td>0.786</td><td>1.162</td><td>0.879</td><td>1.462</td><td>1.320</td><td>1.820</td><td>1.094</td><td>1.740</td><td>1.124</td><td>1.257</td><td>2.076</td></tr><tr><td>672</td><td>1.192</td><td>0.926</td><td>1.231</td><td>1.103</td><td>1.669</td><td>1.461</td><td>2.187</td><td>1.232</td><td>2.736</td><td>1.555</td><td>1.917</td><td>2.941</td></tr><tr><td rowspan="5">Weather</td><td>24</td><td>0.335</td><td>0.381</td><td>0.349</td><td>0.397</td><td>0.435</td><td>0.477</td><td>0.655</td><td>0.583</td><td>0.546</td><td>0.570</td><td>0.615</td><td>0.545</td></tr><tr><td>48</td><td>0.395</td><td>0.459</td><td>0.386</td><td>0.433</td><td>0.426</td><td>0.495</td><td>0.729</td><td>0.666</td><td>0.829</td><td>0.677</td><td>0.660</td><td>0.589</td></tr><tr><td>168</td><td>0.608</td><td>0.567</td><td>0.613</td><td>0.582</td><td>0.727</td><td>0.671</td><td>1.318</td><td>0.855</td><td>1.038</td><td>0.835</td><td>0.748</td><td>0.647</td></tr><tr><td>336</td><td>0.702</td><td>0.620</td><td>0.707</td><td>0.634</td><td>0.754</td><td>0.670</td><td>1.930</td><td>1.167</td><td>1.657</td><td>1.059</td><td>0.782</td><td>0.683</td></tr><tr><td>720</td><td>0.831</td><td>0.731</td><td>0.834</td><td>0.741</td><td>0.885</td><td>0.773</td><td>2.726</td><td>1.575</td><td>1.536</td><td>1.109</td><td>0.851</td><td>0.757</td></tr><tr><td rowspan="5">ECL</td><td>48</td><td>0.344</td><td>0.393</td><td>0.334</td><td>0.399</td><td>0.355</td><td>0.418</td><td>1.404</td><td>0.999</td><td>0.486</td><td>0.572</td><td>0.369</td><td>0.445</td></tr><tr><td>168</td><td>0.368</td><td>0.424</td><td>0.353</td><td>0.420</td><td>0.368</td><td>0.432</td><td>1.515</td><td>1.069</td><td>0.574</td><td>0.602</td><td>0.394</td><td>0.476</td></tr><tr><td>336</td><td>0.381</td><td>0.431</td><td>0.381</td><td>0.439</td><td>0.373</td><td>0.439</td><td>1.601</td><td>1.104</td><td>0.886</td><td>0.795</td><td>0.419</td><td>0.477</td></tr><tr><td>720</td><td>0.406</td><td>0.443</td><td>0.391</td><td>0.438</td><td>0.409</td><td>0.454</td><td>2.009</td><td>1.170</td><td>1.676</td><td>1.095</td><td>0.556</td><td>0.565</td></tr><tr><td>960</td><td>0.460</td><td>0.548</td><td>0.492</td><td>0.550</td><td>0.477</td><td>0.589</td><td>2.141</td><td>1.387</td><td>1.591</td><td>1.128</td><td>0.605</td><td>0.599</td></tr><tr><td colspan="2">Count</td><td colspan="2">33</td><td colspan="2">14</td><td colspan="2">1</td><td colspan="2">0</td><td colspan="2">0</td><td colspan="2">2</td></tr></table>

<table><tbody><tr><td colspan="2">方法</td><td colspan="2">信息者（Informer）</td><td colspan="2">信息者（Informer） ${}^{ \dagger  }$</td><td colspan="2">对数变换（LogTrans）</td><td colspan="2">改革者（Reformer）</td><td colspan="2">长短期记忆网络变体（LSTMa）</td><td colspan="2">长短期时间序列网络（LSTnet）</td></tr><tr><td colspan="2">指标</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td><td>均方误差（MSE）</td><td>平均绝对误差（MAE）</td></tr><tr><td rowspan="5">${\mathrm{{ETTh}}}_{1}$</td><td>24</td><td>0.577</td><td>0.549</td><td>0.620</td><td>0.577</td><td>0.686</td><td>0.604</td><td>0.991</td><td>0.754</td><td>0.650</td><td>0.624</td><td>1.293</td><td>0.901</td></tr><tr><td>48</td><td>0.685</td><td>0.625</td><td>0.692</td><td>0.671</td><td>0.766</td><td>0.757</td><td>1.313</td><td>0.906</td><td>0.702</td><td>0.675</td><td>1.456</td><td>0.960</td></tr><tr><td>168</td><td>0.931</td><td>0.752</td><td>0.947</td><td>0.797</td><td>1.002</td><td>0.846</td><td>1.824</td><td>1.138</td><td>1.212</td><td>0.867</td><td>1.997</td><td>1.214</td></tr><tr><td>336</td><td>1.128</td><td>0.873</td><td>1.094</td><td>0.813</td><td>1.362</td><td>0.952</td><td>2.117</td><td>1.280</td><td>1.424</td><td>0.994</td><td>2.655</td><td>1.369</td></tr><tr><td>720</td><td>1.215</td><td>0.896</td><td>1.241</td><td>0.917</td><td>1.397</td><td>1.291</td><td>2.415</td><td>1.520</td><td>1.960</td><td>1.322</td><td>2.143</td><td>1.380</td></tr><tr><td rowspan="5">${\mathrm{{ETTh}}}_{2}$</td><td>24</td><td>0.720</td><td>0.665</td><td>0.753</td><td>0.727</td><td>0.828</td><td>0.750</td><td>1.531</td><td>1.613</td><td>1.143</td><td>0.813</td><td>2.742</td><td>1.457</td></tr><tr><td>48</td><td>1.457</td><td>1.001</td><td>1.461</td><td>1.077</td><td>1.806</td><td>1.034</td><td>1.871</td><td>1.735</td><td>1.671</td><td>1.221</td><td>3.567</td><td>1.687</td></tr><tr><td>168</td><td>3.489</td><td>1.515</td><td>3.485</td><td>1.612</td><td>4.070</td><td>1.681</td><td>4.660</td><td>1.846</td><td>4.117</td><td>1.674</td><td>3.242</td><td>2.513</td></tr><tr><td>336</td><td>2.723</td><td>1.340</td><td>2.626</td><td>1.285</td><td>3.875</td><td>1.763</td><td>4.028</td><td>1.688</td><td>3.434</td><td>1.549</td><td>2.544</td><td>2.591</td></tr><tr><td>720</td><td>3.467</td><td>1.473</td><td>3.548</td><td>1.495</td><td>3.913</td><td>1.552</td><td>5.381</td><td>2.015</td><td>3.963</td><td>1.788</td><td>4.625</td><td>3.709</td></tr><tr><td rowspan="5">电力变压器时间序列数据集1（ETTm1）</td><td>24</td><td>0.323</td><td>0.369</td><td>0.306</td><td>0.371</td><td>0.419</td><td>0.412</td><td>0.724</td><td>0.607</td><td>0.621</td><td>0.629</td><td>1.968</td><td>1.170</td></tr><tr><td>48</td><td>0.494</td><td>0.503</td><td>0.465</td><td>0.470</td><td>0.507</td><td>0.583</td><td>1.098</td><td>0.777</td><td>1.392</td><td>0.939</td><td>1.999</td><td>1.215</td></tr><tr><td>96</td><td>0.678</td><td>0.614</td><td>0.681</td><td>0.612</td><td>0.768</td><td>0.792</td><td>1.433</td><td>0.945</td><td>1.339</td><td>0.913</td><td>2.762</td><td>1.542</td></tr><tr><td>288</td><td>1.056</td><td>0.786</td><td>1.162</td><td>0.879</td><td>1.462</td><td>1.320</td><td>1.820</td><td>1.094</td><td>1.740</td><td>1.124</td><td>1.257</td><td>2.076</td></tr><tr><td>672</td><td>1.192</td><td>0.926</td><td>1.231</td><td>1.103</td><td>1.669</td><td>1.461</td><td>2.187</td><td>1.232</td><td>2.736</td><td>1.555</td><td>1.917</td><td>2.941</td></tr><tr><td rowspan="5">天气</td><td>24</td><td>0.335</td><td>0.381</td><td>0.349</td><td>0.397</td><td>0.435</td><td>0.477</td><td>0.655</td><td>0.583</td><td>0.546</td><td>0.570</td><td>0.615</td><td>0.545</td></tr><tr><td>48</td><td>0.395</td><td>0.459</td><td>0.386</td><td>0.433</td><td>0.426</td><td>0.495</td><td>0.729</td><td>0.666</td><td>0.829</td><td>0.677</td><td>0.660</td><td>0.589</td></tr><tr><td>168</td><td>0.608</td><td>0.567</td><td>0.613</td><td>0.582</td><td>0.727</td><td>0.671</td><td>1.318</td><td>0.855</td><td>1.038</td><td>0.835</td><td>0.748</td><td>0.647</td></tr><tr><td>336</td><td>0.702</td><td>0.620</td><td>0.707</td><td>0.634</td><td>0.754</td><td>0.670</td><td>1.930</td><td>1.167</td><td>1.657</td><td>1.059</td><td>0.782</td><td>0.683</td></tr><tr><td>720</td><td>0.831</td><td>0.731</td><td>0.834</td><td>0.741</td><td>0.885</td><td>0.773</td><td>2.726</td><td>1.575</td><td>1.536</td><td>1.109</td><td>0.851</td><td>0.757</td></tr><tr><td rowspan="5">增强化学发光法（ECL）</td><td>48</td><td>0.344</td><td>0.393</td><td>0.334</td><td>0.399</td><td>0.355</td><td>0.418</td><td>1.404</td><td>0.999</td><td>0.486</td><td>0.572</td><td>0.369</td><td>0.445</td></tr><tr><td>168</td><td>0.368</td><td>0.424</td><td>0.353</td><td>0.420</td><td>0.368</td><td>0.432</td><td>1.515</td><td>1.069</td><td>0.574</td><td>0.602</td><td>0.394</td><td>0.476</td></tr><tr><td>336</td><td>0.381</td><td>0.431</td><td>0.381</td><td>0.439</td><td>0.373</td><td>0.439</td><td>1.601</td><td>1.104</td><td>0.886</td><td>0.795</td><td>0.419</td><td>0.477</td></tr><tr><td>720</td><td>0.406</td><td>0.443</td><td>0.391</td><td>0.438</td><td>0.409</td><td>0.454</td><td>2.009</td><td>1.170</td><td>1.676</td><td>1.095</td><td>0.556</td><td>0.565</td></tr><tr><td>960</td><td>0.460</td><td>0.548</td><td>0.492</td><td>0.550</td><td>0.477</td><td>0.589</td><td>2.141</td><td>1.387</td><td>1.591</td><td>1.128</td><td>0.605</td><td>0.599</td></tr><tr><td colspan="2">计数</td><td colspan="2">33</td><td colspan="2">14</td><td colspan="2">1</td><td colspan="2">0</td><td colspan="2">0</td><td colspan="2">2</td></tr></tbody></table>

Table 2: Multivariate long sequence time-series forecasting results on four datasets (five cases).

表2：四个数据集（五种情况）的多变量长序列时间序列预测结果。

<!-- Media -->

## Results and Analysis

## 结果与分析

Table 1 and Table 2 summarize the univariate/multivariate evaluation results of all the methods on 4 datasets. We gradually prolong the prediction horizon as a higher requirement of prediction capacity, where the LSTF problem setting is precisely controlled to be tractable on one single GPU for each method. The best results are highlighted in boldface.

表1和表2总结了所有方法在4个数据集上的单变量/多变量评估结果。我们逐步延长预测范围，以提高对预测能力的要求，其中长序列时间序列预测（LSTF）问题的设置经过精确控制，使每种方法都能在单个GPU上易于处理。最佳结果以粗体突出显示。

Univariate Time-series Forecasting Under this setting, each method attains predictions as a single variable over time series. From Table 1, we can observe that: (1) The proposed model Informer significantly improves the inference performance (wining-counts in the last column) across all datasets, and their predict error rises smoothly and slowly within the growing prediction horizon, which demonstrates the success of Informer in enhancing the prediction capacity in the LSTF problem. (2) The Informer beats its canonical degradation Informer ${}^{ \dagger  }$ mostly in wining-counts,i.e., ${32} > {12}$ , which supports the query sparsity assumption in providing a comparable attention feature map. Our proposed method also out-performs the most related work LogTrans and Reformer. We note that the Reformer keeps dynamic decoding and performs poorly in LSTF, while other methods benefit from the generative style decoder as nonautoregressive predictors. (3) The Informer model shows significantly better results than recurrent neural networks LSTMa. Our method has a MSE decrease of 26.8% (at 168), 52.4% (at 336) and ${60.1}\%$ (at 720). This reveals a shorter network path in the self-attention mechanism acquires better prediction capacity than the RNN-based models. (4) The proposed method outperforms DeepAR, ARIMA and Prophet on MSE by decreasing 49.3% (at 168), 61.1% (at 336), and 65.1% (at 720) in average. On the ECL dataset, DeepAR performs better on shorter horizons $\left( { \leq  {336}}\right)$ ,and our method surpasses on longer horizons. We attribute this to a specific example, in which the effectiveness of prediction capacity is reflected with the problem scalability.

单变量时间序列预测 在这种设置下，每种方法都将预测结果作为时间序列上的单个变量。从表1中，我们可以观察到：（1）所提出的Informer模型在所有数据集上显著提高了推理性能（最后一列的获胜次数），并且其预测误差在不断增长的预测范围内平稳且缓慢地上升，这表明Informer在增强长序列时间序列预测（LSTF）问题的预测能力方面取得了成功。（2）Informer在获胜次数上大多击败了其标准退化版本Informer ${}^{ \dagger  }$，即 ${32} > {12}$，这支持了查询稀疏性假设在提供可比注意力特征图方面的作用。我们提出的方法也优于最相关的工作LogTrans和Reformer。我们注意到，Reformer采用动态解码，在LSTF中表现不佳，而其他方法作为非自回归预测器受益于生成式解码器。（3）Informer模型的结果明显优于循环神经网络LSTMa。我们的方法在均方误差（MSE）上分别降低了26.8%（预测步长为168时）、52.4%（预测步长为336时）和 ${60.1}\%$（预测步长为720时）。这表明自注意力机制中较短的网络路径比基于循环神经网络（RNN）的模型具有更好的预测能力。（4）所提出的方法在均方误差上平均比DeepAR、ARIMA和Prophet分别降低了49.3%（预测步长为168时）、61.1%（预测步长为336时）和65.1%（预测步长为720时）。在ECL数据集上，DeepAR在较短预测步长 $\left( { \leq  {336}}\right)$ 上表现更好，而我们的方法在较长预测步长上表现更优。我们将此归因于一个具体示例，其中预测能力的有效性随着问题的可扩展性得到体现。

Multivariate Time-series Forecasting Within this setting, some univariate methods are inappropriate, and LSTnet is the state-of-art baseline. On the contrary, our proposed Informer is easy to change from univariate prediction to multivariate one by adjusting the final FCN layer. From Table 2, we observe that: (1) The proposed model Informer greatly outperforms other methods and the findings $1\& 2$ in the univariate settings still hold for the multivariate time-series. (2) The Informer model shows better results than RNN-based LSTMa and CNN-based LSTnet, and the MSE decreases 26.6% (at 168), 28.2% (at 336), 34.3% (at 720) in average. Compared with the univariate results, the overwhelming performance is reduced, and such phenomena can be caused by the anisotropy of feature dimensions' prediction capacity. It is beyond the scope of this paper, and we will explore it in the future work.

多变量时间序列预测 在这种情况下，一些单变量方法并不适用，而LSTnet是当前最先进的基线模型。相反，我们提出的Informer模型通过调整最后的全连接网络（FCN）层，很容易从单变量预测转换为多变量预测。从表2中我们可以观察到：（1）所提出的Informer模型大大优于其他方法，并且在单变量设置下的研究结果$1\& 2$在多变量时间序列中仍然成立。（2）Informer模型比基于循环神经网络（RNN）的LSTMa和基于卷积神经网络（CNN）的LSTnet表现更好，均方误差（MSE）平均降低了26.6%（预测步长为168时）、28.2%（预测步长为336时）、34.3%（预测步长为720时）。与单变量结果相比，其压倒性的性能有所降低，这种现象可能是由特征维度预测能力的各向异性导致的。这超出了本文的研究范围，我们将在未来的工作中进行探索。

<!-- Media -->

<!-- figureText: 0.4 0.20 -+- Informer, factor c=3 --+-- L-scale Dependency - - Informer, factor c=5 - - - L/2-scale Dependency $- \infty$ Informer,factor $\mathrm{c} = 8$ - $\star   \star   \star$ L/4-scale Dependency - - Informer, factor c=10 0.05 $\begin{array}{lllllll} {48} & {96} & {168} & {240} & {480} & {624} & {720} \end{array}$ Encoder Input Length $\left( {L}_{x}\right)$ Encoder Input Length $\left( {L}_{x}\right)$ (b) Sampling Factor. (c) Stacking Combination. 0.3 0.3 0.2 --w-- Encoder Input (horizon=48) --★-- Decoder Token (horizon=48) 0.0 - Encoder Input (horizon=168) $\rightarrow$ Decoder Token (horizon=168) -0.1 $\begin{array}{llllllll} {48} & {96} & {168} & {240} & {336} & {480} & {624} & {720} \end{array}$ Prolong Input Length $\left( {{L}_{x},{L}_{\text{token }}}\right)$ (a) Input length. -->

<img src="https://cdn.noedgeai.com/01957ae6-5cfc-72b0-ad59-2c1c4e1a17b2_6.jpg?x=161&y=168&w=1480&h=399&r=0"/>

Figure 4: The parameter sensitivity of three components in Informer.

图4：Informer中三个组件的参数敏感性。

<table><tr><td colspan="2">Prediction length</td><td colspan="3">336</td><td colspan="3">720</td></tr><tr><td colspan="2">Encoder's input</td><td>336</td><td>720</td><td>1440</td><td>720</td><td>1440</td><td>2880</td></tr><tr><td rowspan="2">Informer</td><td>MSE</td><td>0.249</td><td>0.225</td><td>0.216</td><td>0.271</td><td>0.261</td><td>0.257</td></tr><tr><td>MAE</td><td>0.393</td><td>0.384</td><td>0.376</td><td>0.435</td><td>0.431</td><td>0.422</td></tr><tr><td rowspan="2">Informer ${}^{ \dagger  }$</td><td>MSE</td><td>0.241</td><td>0.214</td><td>-</td><td>0.259</td><td>-</td><td>-</td></tr><tr><td>MAE</td><td>0.383</td><td>0.371</td><td>-</td><td>0.423</td><td>-</td><td>-</td></tr><tr><td rowspan="2">LogTrans</td><td>MSE</td><td>0.263</td><td>0.231</td><td>-</td><td>0.273</td><td>-</td><td>-</td></tr><tr><td>MAE</td><td>0.418</td><td>0.398</td><td>-</td><td>0.463</td><td>-</td><td>-</td></tr><tr><td rowspan="2">Reformer</td><td>MSE</td><td>1.875</td><td>1.865</td><td>1.861</td><td>2.243</td><td>2.174</td><td>2.113</td></tr><tr><td>MAE</td><td>1.144</td><td>1.129</td><td>1.125</td><td>1.536</td><td>1.497</td><td>1.434</td></tr></table>

<table><tbody><tr><td colspan="2">预测长度</td><td colspan="3">336</td><td colspan="3">720</td></tr><tr><td colspan="2">编码器的输入</td><td>336</td><td>720</td><td>1440</td><td>720</td><td>1440</td><td>2880</td></tr><tr><td rowspan="2">Informer（信息者模型）</td><td>均方误差（MSE，Mean Squared Error）</td><td>0.249</td><td>0.225</td><td>0.216</td><td>0.271</td><td>0.261</td><td>0.257</td></tr><tr><td>平均绝对误差（MAE，Mean Absolute Error）</td><td>0.393</td><td>0.384</td><td>0.376</td><td>0.435</td><td>0.431</td><td>0.422</td></tr><tr><td rowspan="2">Informer ${}^{ \dagger  }$（信息者模型 ${}^{ \dagger  }$）</td><td>均方误差（MSE，Mean Squared Error）</td><td>0.241</td><td>0.214</td><td>-</td><td>0.259</td><td>-</td><td>-</td></tr><tr><td>平均绝对误差（MAE，Mean Absolute Error）</td><td>0.383</td><td>0.371</td><td>-</td><td>0.423</td><td>-</td><td>-</td></tr><tr><td rowspan="2">日志转换（LogTrans）</td><td>均方误差（MSE，Mean Squared Error）</td><td>0.263</td><td>0.231</td><td>-</td><td>0.273</td><td>-</td><td>-</td></tr><tr><td>平均绝对误差（MAE，Mean Absolute Error）</td><td>0.418</td><td>0.398</td><td>-</td><td>0.463</td><td>-</td><td>-</td></tr><tr><td rowspan="2">改革者（Reformer）</td><td>均方误差（MSE，Mean Squared Error）</td><td>1.875</td><td>1.865</td><td>1.861</td><td>2.243</td><td>2.174</td><td>2.113</td></tr><tr><td>平均绝对误差（MAE，Mean Absolute Error）</td><td>1.144</td><td>1.129</td><td>1.125</td><td>1.536</td><td>1.497</td><td>1.434</td></tr></tbody></table>

${}^{1}$ Informer ${}^{ \dagger  }$ uses the canonical self-attention mechanism.

${}^{1}$ 告密者（Informer）${}^{ \dagger  }$ 使用了标准的自注意力机制。

${}^{2}$ The ‘-’ indicates failure for the out-of-memory.

${}^{2}$ “-” 表示出现内存不足错误。

Table 3: Ablation study of the ProbSparse self-attention mechanism.

表 3：ProbSparse 自注意力机制的消融研究。

<table><tr><td rowspan="2">Methods</td><td colspan="2">Training</td><td>Testing</td></tr><tr><td>Time</td><td>Memory</td><td>Steps</td></tr><tr><td>Informer</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>1</td></tr><tr><td>Transformer</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$L$</td></tr><tr><td>LogTrans</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>1*</td></tr><tr><td>Reformer</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$L$</td></tr><tr><td>LSTM</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td><td>$L$</td></tr></table>

<table><tbody><tr><td rowspan="2">方法</td><td colspan="2">训练</td><td>测试</td></tr><tr><td>时间</td><td>内存</td><td>步骤</td></tr><tr><td>告密者</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>1</td></tr><tr><td>变换器；变压器；转换器</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>$L$</td></tr><tr><td>日志变换器（LogTrans）</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L}^{2}\right)$</td><td>1*</td></tr><tr><td>改革者</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$\mathcal{O}\left( {L\log L}\right)$</td><td>$L$</td></tr><tr><td>长短期记忆网络（LSTM）</td><td>$\mathcal{O}\left( L\right)$</td><td>$\mathcal{O}\left( L\right)$</td><td>$L$</td></tr></tbody></table>

${}^{1}$ The LSTnet is hard to present in a closed form.

${}^{1}$ LSTnet（长短期时间序列网络）很难用封闭形式表示。

${}^{2}$ The $\star$ denotes applying our proposed decoder.

${}^{2}$ $\star$ 表示应用我们提出的解码器。

Table 4: $L$ -related computation statics of each layer.

表 4：$L$ 各层的相关计算统计。

<!-- Media -->

LSTF with Granularity Consideration We perform an additional comparison to explore the performance with various granularities. The sequences $\{ {96},{288},{672}\}$ of ETTm ${}_{1}$ (minutes-level) are aligned with $\{ {24},{48},{168}\}$ of ${\mathrm{{ETTh}}}_{1}$ (hour-level). The Informer outperforms other baselines even if the sequences are at different granularity levels.

考虑粒度的长短期时间序列预测（LSTF） 我们进行了额外的比较，以探索不同粒度下的性能。ETTm ${}_{1}$（分钟级）的序列 $\{ {96},{288},{672}\}$ 与 ${\mathrm{{ETTh}}}_{1}$（小时级）的 $\{ {24},{48},{168}\}$ 对齐。即使序列处于不同的粒度级别，Informer（信息者模型）的表现也优于其他基线模型。

## Parameter Sensitivity

## 参数敏感性

We perform the sensitivity analysis of the proposed Informer model on ETTh1 under the univariate setting. Input Length: In Fig. 4a), when predicting short sequences (like 48), initially increasing input length of encoder/decoder degrades performance, but further increasing causes the MSE to drop because it brings repeat short-term patterns. However, the MSE gets lower with longer inputs in predicting long sequences (like 168). Because the longer encoder input may contain more dependencies, and the longer decoder token has rich local information. Sampling Factor: The sampling factor controls the information bandwidth of ProbSparse self-attention in Eq. (3). We start from the small factor $\left( { = 3}\right)$ to large ones,and the general performance increases a little and stabilizes at last in Fig. 4b). It verifies our query sparsity assumption that there are redundant dot-product pairs in the self-attention mechanism. We set the sample factor $c = 5$ (the red line) in practice. The Combination of Layer Stacking: The replica of Layers is complementary for the self-attention distilling, and we investigate each stack $\{ L,L/2,L/4\}$ ’s behavior in Fig. 4c). The longer stack is more sensitive to the inputs, partly due to receiving more long-term information. Our method's selection (the red line), i.e., joining L and L/4, is the most robust strategy.

我们在单变量设置下对ETTh1数据集上提出的Informer模型进行敏感性分析。输入长度：在图4a)中，当预测短序列（如48）时，最初增加编码器/解码器的输入长度会降低性能，但进一步增加会使均方误差（MSE）下降，因为这会引入重复的短期模式。然而，在预测长序列（如168）时，输入长度越长，均方误差越低。因为更长的编码器输入可能包含更多的依赖关系，更长的解码器令牌具有丰富的局部信息。采样因子：采样因子控制着公式(3)中ProbSparse自注意力机制的信息带宽。我们从较小的因子$\left( { = 3}\right)$开始逐渐增大，如图4b)所示，总体性能略有提升，最终趋于稳定。这验证了我们的查询稀疏性假设，即自注意力机制中存在冗余的点积对。在实践中，我们将采样因子设置为$c = 5$（红线）。层堆叠组合：层的复制对自注意力蒸馏起到补充作用，我们在图4c)中研究了每个堆叠$\{ L,L/2,L/4\}$的表现。更长的堆叠对输入更敏感，部分原因是它能接收到更多的长期信息。我们方法的选择（红线），即结合L和L/4，是最稳健的策略。

## Ablation Study: How well Informer works?

## 消融研究：Informer的效果如何？

We also conducted additional experiments on ${\mathrm{{ETTh}}}_{1}$ with ablation consideration.

我们还对${\mathrm{{ETTh}}}_{1}$进行了考虑消融的额外实验。

The performance of ProbSparse self-attention mechanism In the overall results Table 1 & 2 we limited the problem setting to make the memory usage feasible for the canonical self-attention. In this study, we compare our methods with LogTrans and Reformer, and thoroughly explore their extreme performance. To isolate the memory efficient problem,we first reduce settings as $\{$ batch size=8,heads=8, dim=64\}, and maintain other setups in the univariate case. In Table 3, the ProbSparse self-attention shows better performance than the counterparts. The LogTrans gets OOM in extreme cases because its public implementation is the mask of the full-attention,which still has $\mathcal{O}\left( {L}^{2}\right)$ memory usage. Our proposed ProbSparse self-attention avoids this from the simplicity brought by the query sparsity assumption in Eq. 4), referring to the pseudo-code in Appendix E.2, and reaches smaller memory usage.

ProbSparse自注意力机制的性能 在整体结果表1和表2中，我们限制了问题设置，以使标准自注意力的内存使用可行。在本研究中，我们将我们的方法与LogTrans和Reformer进行比较，并全面探索它们的极限性能。为了分离内存效率问题，我们首先将设置减少为$\{$（批量大小 = 8，头数 = 8，维度 = 64），并在单变量情况下保持其他设置。在表3中，ProbSparse自注意力的性能优于其他方法。LogTrans在极端情况下会出现内存不足（OOM）问题，因为其公开实现是全注意力的掩码，仍然有$\mathcal{O}\left( {L}^{2}\right)$的内存使用。我们提出的ProbSparse自注意力通过公式（4）中的查询稀疏性假设带来的简单性避免了这个问题（参见附录E.2中的伪代码），并实现了更小的内存使用。

<!-- Media -->

<table><tr><td colspan="2">Prediction length</td><td colspan="5">336</td><td colspan="5">480</td></tr><tr><td colspan="2">Encoder's input</td><td>336</td><td>480</td><td>720</td><td>960</td><td>1200</td><td>336</td><td>480</td><td>720</td><td>960</td><td>1200</td></tr><tr><td rowspan="2">Informer ${}^{ \dagger  }$</td><td>MSE</td><td>0.249</td><td>0.208</td><td>0.225</td><td>0.199</td><td>0.186</td><td>0.197</td><td>0.243</td><td>0.213</td><td>0.192</td><td>0.174</td></tr><tr><td>MAE</td><td>0.393</td><td>0.385</td><td>0.384</td><td>0.371</td><td>0.365</td><td>0.388</td><td>0.392</td><td>0.383</td><td>0.377</td><td>0.362</td></tr><tr><td rowspan="2">Informer ${}^{ \ddagger  }$</td><td>MSE</td><td>0.229</td><td>0.215</td><td>0.204</td><td>-</td><td>-</td><td>0.224</td><td>0.208</td><td>0.197</td><td>-</td><td>-</td></tr><tr><td>MAE</td><td>0.391</td><td>0.387</td><td>0.377</td><td>-</td><td>-</td><td>0.381</td><td>0.376</td><td>0.370</td><td>-</td><td>-</td></tr></table>

<table><tbody><tr><td colspan="2">预测长度</td><td colspan="5">336</td><td colspan="5">480</td></tr><tr><td colspan="2">编码器的输入</td><td>336</td><td>480</td><td>720</td><td>960</td><td>1200</td><td>336</td><td>480</td><td>720</td><td>960</td><td>1200</td></tr><tr><td rowspan="2">Informer ${}^{ \dagger  }$（Informer ${}^{ \dagger  }$）</td><td>均方误差（MSE，Mean Squared Error）</td><td>0.249</td><td>0.208</td><td>0.225</td><td>0.199</td><td>0.186</td><td>0.197</td><td>0.243</td><td>0.213</td><td>0.192</td><td>0.174</td></tr><tr><td>平均绝对误差（MAE，Mean Absolute Error）</td><td>0.393</td><td>0.385</td><td>0.384</td><td>0.371</td><td>0.365</td><td>0.388</td><td>0.392</td><td>0.383</td><td>0.377</td><td>0.362</td></tr><tr><td rowspan="2">Informer ${}^{ \ddagger  }$（Informer ${}^{ \ddagger  }$）</td><td>均方误差（MSE，Mean Squared Error）</td><td>0.229</td><td>0.215</td><td>0.204</td><td>-</td><td>-</td><td>0.224</td><td>0.208</td><td>0.197</td><td>-</td><td>-</td></tr><tr><td>平均绝对误差（MAE，Mean Absolute Error）</td><td>0.391</td><td>0.387</td><td>0.377</td><td>-</td><td>-</td><td>0.381</td><td>0.376</td><td>0.370</td><td>-</td><td>-</td></tr></tbody></table>

${}^{1}$ Informer ${}^{ \dagger  }$ removes the self-attention distilling from Informer ${}^{ \dagger  }$ .

${}^{1}$信息者 ${}^{ \dagger  }$（Informer ${}^{ \dagger  }$）去除了信息者 ${}^{ \dagger  }$ 中的自注意力蒸馏。

${}^{2}$ The ‘-’ indicates failure for the out-of-memory.

${}^{2}$“-”表示出现内存不足错误。

Table 5: Ablation study of the self-attention distilling.

表 5：自注意力蒸馏的消融研究。

<table><tr><td colspan="2">Prediction length</td><td colspan="5">336</td><td colspan="5">480</td></tr><tr><td colspan="2">Prediction offset</td><td>+0</td><td>+12</td><td>+24</td><td>+48</td><td>+72</td><td>+0</td><td>+48</td><td>+96</td><td>+144</td><td>+168</td></tr><tr><td rowspan="2">Informer ${}^{ \ddagger  }$</td><td>MSE</td><td>0.207</td><td>0.209</td><td>0.211</td><td>0.211</td><td>0.216</td><td>0.198</td><td>0.203</td><td>0.203</td><td>0.208</td><td>0.208</td></tr><tr><td>MAE</td><td>0.385</td><td>0.387</td><td>0.391</td><td>0.393</td><td>0.397</td><td>0.390</td><td>0.392</td><td>0.393</td><td>0.401</td><td>0.403</td></tr><tr><td rowspan="2">Informer ${}^{3}$</td><td>MSE</td><td>0.201</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.392</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MAE</td><td>0.393</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.484</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

<table><tbody><tr><td colspan="2">预测长度</td><td colspan="5">336</td><td colspan="5">480</td></tr><tr><td colspan="2">预测偏移量</td><td>+0</td><td>+12</td><td>+24</td><td>+48</td><td>+72</td><td>+0</td><td>+48</td><td>+96</td><td>+144</td><td>+168</td></tr><tr><td rowspan="2">Informer（Informer） ${}^{ \ddagger  }$</td><td>均方误差（MSE，Mean Squared Error）</td><td>0.207</td><td>0.209</td><td>0.211</td><td>0.211</td><td>0.216</td><td>0.198</td><td>0.203</td><td>0.203</td><td>0.208</td><td>0.208</td></tr><tr><td>平均绝对误差（MAE，Mean Absolute Error）</td><td>0.385</td><td>0.387</td><td>0.391</td><td>0.393</td><td>0.397</td><td>0.390</td><td>0.392</td><td>0.393</td><td>0.401</td><td>0.403</td></tr><tr><td rowspan="2">Informer（Informer） ${}^{3}$</td><td>均方误差（MSE，Mean Squared Error）</td><td>0.201</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.392</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>平均绝对误差（MAE，Mean Absolute Error）</td><td>0.393</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.484</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></tbody></table>

${}^{1}$ Informer ${}^{§}$ replaces our decoder with dynamic decoding one in Informer ${}^{ \ddagger  }$ .

${}^{1}$Informer ${}^{§}$在Informer ${}^{ \ddagger  }$中用动态解码模块替换了我们的解码器。

${}^{2}$ The ’-’ indicates failure for the unacceptable metric results.

${}^{2}$“ - ”表示指标结果不可接受，即失败。

Table 6: Ablation study of the generative style decoder.

表6：生成式风格解码器的消融实验。

<!-- figureText: --♦-- LSTnet --♦-- LSTnet Decoder predict length $\left( {L}_{y}\right)$ --- Informer ${}^{ \dagger  }$ - $\leftarrow$ LogTrans - $\rightarrow   \leftarrow$ Reformer Encoder Input length $\left( {L}_{x}\right)$ -->

<img src="https://cdn.noedgeai.com/01957ae6-5cfc-72b0-ad59-2c1c4e1a17b2_7.jpg?x=161&y=940&w=684&h=339&r=0"/>

Figure 5: The total runtime of training/testing phase.

图5：训练/测试阶段的总运行时间。

<!-- Media -->

The performance of self-attention distilling In this study,we use Informer ${}^{ \dagger  }$ as the benchmark to eliminate additional effects of ProbSparse self-attention. The other experimental setup is aligned with the settings of univariate Time-series. From Table 5,Informer ${}^{ \dagger  }$ has fulfilled all the experiments and achieves better performance after taking advantage of long sequence inputs. The comparison method Informer ${}^{ \ddagger  }$ removes the distilling operation and reaches OOM with longer inputs $\left( { > {720}}\right)$ . Regarding the benefits of long sequence inputs in the LSTF problem, we conclude that the self-attention distilling is worth adopting, especially when a longer prediction is required.

自注意力蒸馏的性能 在本研究中，我们使用Informer ${}^{ \dagger  }$作为基准，以消除ProbSparse自注意力的额外影响。其他实验设置与单变量时间序列的设置一致。从表5可以看出，Informer ${}^{ \dagger  }$完成了所有实验，并在利用长序列输入后取得了更好的性能。对比方法Informer ${}^{ \ddagger  }$去除了蒸馏操作，在输入序列$\left( { > {720}}\right)$更长时出现了内存溢出（OOM）。关于长序列输入在长序列时间序列预测（LSTF）问题中的好处，我们得出结论：自注意力蒸馏值得采用，特别是在需要更长预测长度的情况下。

The performance of generative style decoder In this study, we testify the potential value of our decoder in acquiring a "generative" results. Unlike the existing methods, the labels and outputs are forced to be aligned in the training and inference, our proposed decoder's predicting relies solely on the time stamp, which can predict with offsets. From Table 6 , we can see that the general prediction performance of Informer ${}^{ \ddagger  }$ resists with the offset increasing,while the counterpart fails for the dynamic decoding. It proves the decoder's ability to capture individual long-range dependency between arbitrary outputs and avoid error accumulation.

生成式风格解码器的性能 在本研究中，我们验证了我们的解码器在获取“生成式”结果方面的潜在价值。与现有方法不同，现有方法在训练和推理过程中强制标签和输出对齐，而我们提出的解码器的预测仅依赖于时间戳，能够进行有偏移的预测。从表6中可以看出，Informer ${}^{ \ddagger  }$的总体预测性能会随着偏移量的增加而下降，而我们提出的解码器在动态解码方面表现良好。这证明了解码器能够捕捉任意输出之间的个体长距离依赖关系，并避免误差累积。

## Computation Efficiency

## 计算效率

With the multivariate setting and all the methods' current finest implement, we perform a rigorous runtime comparison in Fig. 5). During the training phase, the Informer (red line) achieves the best training efficiency among Transformer-based methods. During the testing phase, our methods are much faster than others with the generative style decoding. The comparisons of theoretical time complexity and memory usage are summarized in Table 4. The performance of Informer is aligned with the runtime experiments. Note that the LogTrans focus on improving the self-attention mechanism, and we apply our proposed decoder in LogTrans for a fair comparison (the $\star$ in Table 4).

在多变量设置和所有方法当前的最优实现下，我们在图5中进行了严格的运行时间比较。在训练阶段，Informer（红线）在基于Transformer的方法中实现了最佳的训练效率。在测试阶段，采用生成式解码的我们的方法比其他方法快得多。理论时间复杂度和内存使用情况的比较总结在表4中。Informer的性能与运行时间实验结果一致。请注意，LogTrans专注于改进自注意力机制，为了进行公平比较，我们在LogTrans中应用了我们提出的解码器（表4中的$\star$）。

## 5 Conclusion

## 5 结论

In this paper, we studied the long-sequence time-series forecasting problem and proposed Informer to predict long sequences. Specifically, we designed the ProbSparse self-attention mechanism and distilling operation to handle the challenges of quadratic time complexity and quadratic memory usage in vanilla Transformer. Also, the carefully designed generative decoder alleviates the limitation of traditional encoder-decoder architecture. The experiments on real-world data demonstrated the effectiveness of Informer for enhancing the prediction capacity in LSTF problem.

在本文中，我们研究了长序列时间序列预测问题，并提出了Informer来预测长序列。具体来说，我们设计了ProbSparse自注意力机制和蒸馏操作，以应对原始Transformer中二次时间复杂度和二次内存使用的挑战。此外，精心设计的生成式解码器缓解了传统编码器 - 解码器架构的局限性。在真实世界数据上的实验证明了Informer在增强长序列时间序列预测（LSTF）问题的预测能力方面的有效性。

## Appendices

## 附录

## Appendix A Related Work

## 附录A 相关工作

We provide a literature review of the long sequence time-series forecasting (LSTF) problem below.

下面我们对长序列时间序列预测（LSTF）问题进行文献综述。

Time-series Forecasting Existing methods for time-series forecasting can be roughly grouped into two categories: classical models and deep learning based methods. Classical time-series models serve as a reliable workhorse for time-series forecasting, with appealing properties such as interpretability and theoretical guarantees (Box et al. 2015; Ray 1990). Modern extensions include the support for missing data (Seeger et al. 2017) and multiple data types (Seeger, Salinas, and Flunkert 2016). Deep learning based methods mainly develop sequence to sequence prediction paradigm by using RNN and their variants, achieving ground-breaking performance (Hochreiter and Schmidhuber 1997; Li et al. 2018; Yu et al. 2017). Despite the substantial progress, existing algorithms still fail to predict long sequence time series with satisfying accuracy. Typical state-of-the-art approaches (Seeger et al. 2017; Seeger, Salinas, and Flunkert 2016), especially deep-learning methods (Yu et al. 2017; Qin et al. 2017; Flunkert, Salinas, and Gasthaus 2017; Mukher-jee et al. 2018; Wen et al. 2017), remain as a sequence to sequence prediction paradigm with step-by-step process, which have the following limitations: (i) Even though they may achieve accurate prediction for one step forward, they often suffer from accumulated error from the dynamic decoding, resulting in the large errors for LSTF problem (Liu et al. 2019; Qin et al. 2017). The prediction accuracy decays along with the increase of the predicted sequence length. (ii) Due to the problem of vanishing gradient and memory constraint (Sutskever, Vinyals, and Le 2014), most existing methods cannot learn from the past behavior of the whole history of the time-series. In our work, the Informer is designed to address this two limitations.

时间序列预测 现有的时间序列预测方法大致可分为两类：经典模型和基于深度学习的方法。经典时间序列模型是时间序列预测中可靠的常用工具，具有可解释性和理论保证等吸引人的特性（博克斯（Box）等人，2015年；雷（Ray），1990年）。现代扩展包括对缺失数据的支持（西格（Seeger）等人，2017年）和对多种数据类型的支持（西格（Seeger）、萨利纳斯（Salinas）和弗伦克特（Flunkert），2016年）。基于深度学习的方法主要通过使用循环神经网络（RNN）及其变体来发展序列到序列的预测范式，取得了突破性的性能（霍赫赖特（Hochreiter）和施密德胡贝尔（Schmidhuber），1997年；李（Li）等人，2018年；余（Yu）等人，2017年）。尽管取得了重大进展，但现有算法仍然无法以令人满意的精度预测长序列时间序列。典型的最先进方法（西格（Seeger）等人，2017年；西格（Seeger）、萨利纳斯（Salinas）和弗伦克特（Flunkert），2016年），尤其是深度学习方法（余（Yu）等人，2017年；秦（Qin）等人，2017年；弗伦克特（Flunkert）、萨利纳斯（Salinas）和加施豪斯（Gasthaus），2017年；穆克吉（Mukherjee）等人，2018年；文（Wen）等人，2017年），仍然是逐步进行的序列到序列预测范式，存在以下局限性：（i）尽管它们可能在一步预测中实现准确预测，但它们往往会受到动态解码累积误差的影响，导致在长序列时间序列预测（LSTF）问题中出现较大误差（刘（Liu）等人，2019年；秦（Qin）等人，2017年）。预测精度会随着预测序列长度的增加而下降。（ii）由于梯度消失问题和内存限制（苏茨克维（Sutskever）、维尼亚尔斯（Vinyals）和乐（Le），2014年），大多数现有方法无法从时间序列的整个历史过去行为中学习。在我们的工作中，Informer被设计用于解决这两个局限性。

Long sequence input problem From the above discussion, we refer to the second limitation as to the long sequence time-series input (LSTI) problem. We will explore related works and draw a comparison between our LSTF problem. The researchers truncate / summarize / sample the input sequence to handle a very long sequence in practice, but valuable data may be lost in making accurate predictions. Instead of modifying inputs, Truncated BPTT (Aicher, Foti, and Fox 2019) only uses last time steps to estimate the gradients in weight updates, and Auxiliary Losses (Trinh et al. 2018) enhance the gradients flow by adding auxiliary gradients. Other attempts includes Recurrent Highway Networks (Zilly et al. 2017) and Bootstrapping Regularizer (Cao and Xu 2019). Theses methods try to improve the gradient flows in the recurrent network's long path, but the performance is limited with the sequence length growing in the LSTI problem. CNN-based methods (Stoller et al. 2019; Bai, Kolter, and Koltun 2018) use the convolutional filter to capture the long term dependency, and their receptive fields grow exponentially with the stacking of layers, which hurts the sequence alignment. In the LSTI problem, the main task is to enhance the model's capacity of receiving long sequence inputs and extract the long-range dependency from these inputs. But the LSTF problem seeks to enhance the model's prediction capacity of forecasting long sequence outputs, which requires establishing the long-range dependency between outputs and inputs. Thus, the above methods are not feasible for LSTF directly.

长序列输入问题 从上述讨论中，我们将第二个限制称为长序列时间序列输入（LSTI）问题。我们将探讨相关工作，并将其与我们的长序列时间序列预测（LSTF）问题进行比较。在实践中，研究人员会对输入序列进行截断/总结/采样来处理非常长的序列，但在进行准确预测时可能会丢失有价值的数据。与修改输入不同，截断的反向传播时间算法（Truncated BPTT，艾切尔（Aicher）、福蒂（Foti）和福克斯（Fox），2019年）仅使用最后几个时间步来估计权重更新中的梯度，而辅助损失法（Auxiliary Losses，特里恩（Trinh）等人，2018年）通过添加辅助梯度来增强梯度流。其他尝试包括循环高速公路网络（Recurrent Highway Networks，齐利（Zilly）等人，2017年）和自举正则化器（Bootstrapping Regularizer，曹（Cao）和徐（Xu），2019年）。这些方法试图改善循环网络长路径中的梯度流，但在LSTI问题中，随着序列长度的增加，性能会受到限制。基于卷积神经网络（CNN）的方法（斯托勒（Stoller）等人，2019年；白（Bai）、科尔特（Kolter）和科尔图恩（Koltun），2018年）使用卷积滤波器来捕捉长期依赖关系，并且随着层数的堆叠，其感受野呈指数增长，这会影响序列对齐。在LSTI问题中，主要任务是增强模型接收长序列输入的能力，并从这些输入中提取长距离依赖关系。但LSTF问题旨在增强模型预测长序列输出的能力，这需要建立输出和输入之间的长距离依赖关系。因此，上述方法直接用于LSTF是不可行的。

Attention model Bahdanau et al. firstly proposed the addictive attention (Bahdanau, Cho, and Bengio 2015) to improve the word alignment of the encoder-decoder architecture in the translation task. Then, its variant (Luong, Pham, and Manning 2015) has proposed the widely used location, general, and dot-product attention. The popular self-attention based Transformer (Vaswani et al. 2017) has recently been proposed as new thinking of sequence modeling and has achieved great success, especially in the NLP field. The ability of better sequence alignment has been validated by applying it to translation, speech, music, and image generation. In our work, the Informer takes advantage of its sequence alignment ability and makes it amenable to the LSTF problem.

注意力模型方面，巴赫达诺夫（Bahdanau）等人首先提出了加法注意力（Bahdanau、Cho和Bengio，2015年），以改进翻译任务中编码器 - 解码器架构的词对齐。随后，其变体（Luong、Pham和Manning，2015年）提出了广泛使用的位置注意力、通用注意力和点积注意力。近期，基于自注意力机制的流行的Transformer模型（Vaswani等人，2017年）作为序列建模的新思路被提出，并取得了巨大成功，尤其在自然语言处理（NLP）领域。通过将其应用于翻译、语音、音乐和图像生成等任务，其更好的序列对齐能力得到了验证。在我们的工作中，Informer模型利用了其序列对齐能力，并使其适用于长序列时间序列预测（LSTF）问题。

Transformer-based time-series model The most related works (Song et al. 2018; Ma et al. 2019; Li et al. 2019) all start from a trail on applying Transformer in time-series data and fail in LSTF forecasting as they use the vanilla Transformer. And some other works (Child et al. 2019; Li et al. 2019) noticed the sparsity in self-attention mechanism and we have discussed them in the main context.

基于Transformer的时间序列模型 最相关的研究（宋等人，2018年；马等人，2019年；李等人，2019年）均始于将Transformer应用于时间序列数据的尝试，但由于使用了原始的Transformer，它们在长序列时间序列预测（LSTF）中均告失败。其他一些研究（蔡尔德等人，2019年；李等人，2019年）注意到了自注意力机制中的稀疏性，我们已在正文部分对这些研究进行了讨论。

## Appendix B The Uniform Input Representation

## 附录B 统一输入表示

The RNN models (Schuster and Paliwal 1997; Hochre-iter and Schmidhuber 1997; Chung et al. 2014; Sutskever, Vinyals, and Le 2014; Qin et al. 2017; Chang et al. 2018) capture the time-series pattern by the recurrent structure itself and barely relies on time stamps. The vanilla transformer (Vaswani et al. 2017; Devlin et al. 2018) uses pointwise self-attention mechanism and the time stamps serve as local positional context. However, in the LSTF problem, the ability to capture long-range independence requires global information like hierarchical time stamps (week, month and year) and agnostic time stamps (holidays, events). These are hardly leveraged in canonical self-attention and consequent query-key mismatches between the encoder and decoder bring underlying degradation on the forecasting performance. We propose a uniform input representation to mitigate the issue, the Fig. 6 gives an intuitive overview.

循环神经网络（RNN）模型（舒斯特（Schuster）和帕利瓦尔（Paliwal），1997年；霍赫赖特（Hochreiter）和施密德胡贝尔（Schmidhuber），1997年；钟（Chung）等人，2014年；苏茨克维（Sutskever）、维尼亚尔斯（Vinyals）和乐（Le），2014年；秦（Qin）等人，2017年；张（Chang）等人，2018年）通过循环结构本身捕捉时间序列模式，几乎不依赖时间戳。原始的Transformer模型（瓦斯瓦尼（Vaswani）等人，2017年；德夫林（Devlin）等人，2018年）使用逐点自注意力机制，时间戳作为局部位置上下文。然而，在长序列时间序列预测（LSTF）问题中，捕捉长距离独立性的能力需要全局信息，如分层时间戳（周、月和年）和无关时间戳（节假日、事件）。在经典的自注意力机制中，这些信息很难被利用，并且编码器和解码器之间随之而来的查询 - 键不匹配会导致预测性能的潜在下降。我们提出了一种统一的输入表示方法来缓解这个问题，图6给出了直观的概述。

Assuming we have $t$ -th sequence input ${\mathcal{X}}^{t}$ and $p$ types of global time stamps and the feature dimension after input representation is ${d}_{\text{model }}$ . We firstly preserve the local context by using a fixed position embedding:

假设我们有第 $t$ 个序列输入 ${\mathcal{X}}^{t}$ 以及 $p$ 种全局时间戳，并且输入表示后的特征维度为 ${d}_{\text{model }}$。我们首先通过使用固定位置嵌入来保留局部上下文：

$$
{\mathrm{{PE}}}_{\left( \text{pos },2j\right) } = \sin \left( {\text{ pos }/{\left( 2{L}_{x}\right) }^{{2j}/{d}_{\text{model }}}}\right)  \tag{7}
$$

$$
{\mathrm{{PE}}}_{\left( \text{pos },2j + 1\right) } = \cos \left( {\mathrm{{pos}}/{\left( 2{L}_{x}\right) }^{{2j}/{d}_{\text{model }}}}\right) 
$$

where $j \in  \left\{  {1,\ldots ,\left| {{d}_{\text{model }}/2}\right| }\right\}$ . Each global time stamp is employed by a learnable stamp embeddings ${\mathrm{{SE}}}_{\left( \text{pos }\right) }$ with limited vocab size (up to 60, namely taking minutes as the finest granularity). That is, the self-attention's similarity computation can have access to global context and the computation consuming is affordable on long inputs. To align the dimension,we project the scalar context ${\mathbf{x}}_{i}^{t}$ into ${d}_{\text{model }}$ -dim vector ${\mathbf{u}}_{i}^{t}$ with 1-D convolutional filters (kernel width=3, stride=1). Thus, we have the feeding vector

其中 $j \in  \left\{  {1,\ldots ,\left| {{d}_{\text{model }}/2}\right| }\right\}$ 。每个全局时间戳由一个可学习的时间戳嵌入 ${\mathrm{{SE}}}_{\left( \text{pos }\right) }$ 使用，其词汇量有限（最多60个，即以分钟为最细粒度）。也就是说，自注意力的相似度计算可以访问全局上下文，并且对于长输入而言，计算消耗是可以承受的。为了对齐维度，我们使用一维卷积滤波器（核宽度 = 3，步长 = 1）将标量上下文 ${\mathbf{x}}_{i}^{t}$ 投影到 ${d}_{\text{model }}$ 维向量 ${\mathbf{u}}_{i}^{t}$ 中。因此，我们得到了输入向量

$$
{\mathcal{X}}_{\text{feed }\left\lbrack  i\right\rbrack  }^{t} = \alpha {\mathbf{u}}_{i}^{t} + {\mathrm{{PE}}}_{\left( {L}_{x} \times  \left( t - 1\right)  + i,\right) } + \mathop{\sum }\limits_{p}\left\lbrack  {\mathrm{{SE}}}_{\left( {L}_{x} \times  \left( t - 1\right)  + i\right) }\right\rbrack  p \tag{8}
$$

where $i \in  \left\{  {1,\ldots ,{L}_{x}}\right\}$ ,and $\alpha$ is the factor balancing the magnitude between the scalar projection and local/global embeddings. We recommend $\alpha  = 1$ if the sequence input has been normalized.

其中 $i \in  \left\{  {1,\ldots ,{L}_{x}}\right\}$，并且 $\alpha$ 是平衡标量投影与局部/全局嵌入之间大小的因子。如果序列输入已被归一化，我们建议使用 $\alpha  = 1$。

<!-- Media -->

<!-- figureText: Projection ${\mathrm{u}}_{0}$ Week Week Week E1 E2 E3 E0 H2 Local Time Stamp Position Embeddings Global Time Stamp Week Week Embeddings E0 Month Embeddings Holiday Embeddings E0 -->

<img src="https://cdn.noedgeai.com/01957ae6-5cfc-72b0-ad59-2c1c4e1a17b2_9.jpg?x=216&y=622&w=595&h=471&r=0"/>

Figure 6: The input representation of Informer. The inputs's embedding consists of three separate parts, a scalar projection, the local time stamp (Position) and global time stamp embeddings (Minutes, Hours, Week, Month, Holiday etc.).

图 6：Informer 的输入表示。输入的嵌入由三个独立部分组成，即标量投影、局部时间戳（位置）和全局时间戳嵌入（分钟、小时、周、月、节假日等）。

<!-- Media -->

## Appendix C The long tail distribution in self-attention feature map

## 附录 C 自注意力特征图中的长尾分布

We have performed the vanilla Transformer on the ETTh ${}_{1}$ dataset to investigate the distribution of self-attention feature map. We select the attention score of $\{$ Head1,Head7 $\}$ @ Layer1. The blue line in Fig. 7 forms a long tail distribution, i.e. a few dot-product pairs contribute to the major attention and others can be ignored.

我们在 ETTh ${}_{1}$ 数据集上执行了原始的 Transformer 模型，以研究自注意力特征图的分布。我们选择了 $\{$ 第 1 头、第 7 头 $\}$ @ 第 1 层的注意力分数。图 7 中的蓝色线形成了长尾分布，即少数点积对贡献了主要的注意力，而其他的可以忽略不计。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957ae6-5cfc-72b0-ad59-2c1c4e1a17b2_9.jpg?x=171&y=1684&w=655&h=247&r=0"/>

Figure 7: The Softmax scores in the self-attention from a 4-layer canonical Transformer trained on ${\mathbf{{ETTh}}}_{1}$ dataset.

图 7：在 ${\mathbf{{ETTh}}}_{1}$ 数据集上训练的 4 层标准 Transformer 的自注意力中的 Softmax 分数。

<!-- Media -->

## Appendix D Details of the proof

## 附录D 证明细节

## Proof of Lemma 1

## 引理1的证明

Proof. For the individual ${\mathbf{q}}_{i}$ ,we can relax the discrete keys into the continuous $d$ -dimensional variable,i.e. vector ${\mathbf{k}}_{j}$ . The query sparsity measurement is defined as the $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  = \ln \mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}} - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left( {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right) .$

证明。对于个体 ${\mathbf{q}}_{i}$，我们可以将离散密钥放宽为连续的 $d$ 维变量，即向量 ${\mathbf{k}}_{j}$。查询稀疏性度量定义为 $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  = \ln \mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}} - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left( {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right) .$

Firstly, we look into the left part of the inequality. For each query ${\mathbf{q}}_{i}$ ,the first term of the $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ becomes the $\log$ -sum-exp of the inner-product of a fixed query ${\mathbf{q}}_{i}$ and all the keys ,and we can define ${f}_{i}\left( \mathbf{K}\right)  = \ln \mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}$ . From the Eq.(2) in the Log-sum-exp network(Calafiore, Gaubert, and Possieri 2018) and the further analysis, the function ${f}_{i}\left( \mathbf{K}\right)$ is convex. Moreover, ${f}_{i}\left( \mathbf{K}\right)$ add a linear combination of ${\mathbf{k}}_{j}$ makes the $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ to be the convex function for a fixed query. Then we can take the derivation of the measurement with respect to the individual vector ${\mathbf{k}}_{j}$ as $\frac{\partial M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right) }{\partial {\mathbf{k}}_{j}} = \frac{{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}}{\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}} \cdot  \frac{{\mathbf{q}}_{i}}{\sqrt{d}} - \frac{1}{{L}_{K}} \cdot  \frac{{\mathbf{q}}_{i}}{\sqrt{d}}$ . To reach the minimum value,we let $\overrightarrow{\nabla }M\left( {\mathbf{q}}_{i}\right)  = \overrightarrow{0}$ and the following condition is acquired as ${\mathbf{q}}_{i}{\mathbf{k}}_{1}^{\top } + \ln {L}_{K} = \cdots  =$ ${\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top } + \ln {L}_{K} = \cdots  = \ln \mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}$ . Naturally,it requires ${\mathbf{k}}_{1} = {\mathbf{k}}_{2} = \cdots  = {\mathbf{k}}_{{L}_{K}}$ ,and we have the measurement’s minimum as $\ln {L}_{K}$ ,i.e.

首先，我们研究不等式的左半部分。对于每个查询 ${\mathbf{q}}_{i}$，$M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ 的第一项变为固定查询 ${\mathbf{q}}_{i}$ 与所有键的内积的对数和指数（Log-sum-exp），我们可以定义 ${f}_{i}\left( \mathbf{K}\right)  = \ln \mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}$。根据对数和指数网络（Log-sum-exp network，卡拉菲奥雷（Calafiore）、戈贝尔（Gaubert）和波西耶里（Possieri），2018 年）中的式 (2) 以及进一步分析，函数 ${f}_{i}\left( \mathbf{K}\right)$ 是凸函数。此外，${f}_{i}\left( \mathbf{K}\right)$ 加上 ${\mathbf{k}}_{j}$ 的线性组合使得对于固定查询而言，$M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ 成为凸函数。然后，我们可以将度量关于单个向量 ${\mathbf{k}}_{j}$ 的导数表示为 $\frac{\partial M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right) }{\partial {\mathbf{k}}_{j}} = \frac{{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}}{\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}} \cdot  \frac{{\mathbf{q}}_{i}}{\sqrt{d}} - \frac{1}{{L}_{K}} \cdot  \frac{{\mathbf{q}}_{i}}{\sqrt{d}}$。为了达到最小值，我们令 $\overrightarrow{\nabla }M\left( {\mathbf{q}}_{i}\right)  = \overrightarrow{0}$，并得到以下条件为 ${\mathbf{q}}_{i}{\mathbf{k}}_{1}^{\top } + \ln {L}_{K} = \cdots  =$ ${\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top } + \ln {L}_{K} = \cdots  = \ln \mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}$。自然地，这要求 ${\mathbf{k}}_{1} = {\mathbf{k}}_{2} = \cdots  = {\mathbf{k}}_{{L}_{K}}$，并且我们得到度量的最小值为 $\ln {L}_{K}$，即

$$
M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  \geq  \ln {L}_{K} \tag{9}
$$

Secondly, we look into the right part of the inequality. If we select the largest inner-product $\mathop{\max }\limits_{j}\left\{  {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right\}$ ,it is easy that

其次，我们研究不等式的右半部分。如果我们选择最大的内积 $\mathop{\max }\limits_{j}\left\{  {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right\}$，很容易得出

$$
M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  = \ln \mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{\frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}} - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left( \frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}\right) 
$$

$$
 \leq  \ln \left( {{L}_{K} \cdot  \mathop{\max }\limits_{j}\left\{  \frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}\right\}  }\right)  - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left( \frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}\right) . \tag{10}
$$

$$
 = \ln {L}_{K} + \mathop{\max }\limits_{j}\left\{  \frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}\right\}   - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left( \frac{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}{\sqrt{d}}\right) 
$$

Combine the Eq.(14) and Eq.(15), we have the results of Lemma 1. When the key set is the same with the query set, the above discussion also holds.

将式(14)和式(15)相结合，我们得到引理1的结果。当键集与查询集相同时，上述讨论同样成立。

Proposition 1. Assuming ${\mathbf{k}}_{j} \sim  \mathcal{N}\left( {\mu ,\sum }\right)$ and we let ${\mathbf{{qk}}}_{i}$ denote set $\left\{  {\left( {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}\right) /\sqrt{d} \mid  j = 1,\ldots ,{L}_{K}}\right\}$ ,then $\forall {M}_{m} =$ $\mathop{\max }\limits_{i}M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ there exist $\kappa  > 0$ such that: in the interval $\forall {\mathbf{q}}_{1},{\mathbf{q}}_{2} \in  \left\{  {\mathbf{q} \mid  M\left( {\mathbf{q},\mathbf{K}}\right)  \in  \left\lbrack  {{M}_{m},{M}_{m} - \kappa }\right) }\right\}$ ,if $\bar{M}\left( {{\mathbf{q}}_{1},\mathbf{K}}\right)  >$ $\bar{M}\left( {{\mathbf{q}}_{2},\mathbf{K}}\right)$ and $\operatorname{Var}\left( {\mathbf{{qk}}}_{1}\right)  > \operatorname{Var}\left( {\mathbf{{qk}}}_{2}\right)$ ,we have high probability that $M\left( {{\mathbf{q}}_{1},\mathbf{K}}\right)  > M\left( {{\mathbf{q}}_{2},\mathbf{K}}\right)$ .

命题1。假设${\mathbf{k}}_{j} \sim  \mathcal{N}\left( {\mu ,\sum }\right)$，并且我们用${\mathbf{{qk}}}_{i}$表示集合$\left\{  {\left( {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }}\right) /\sqrt{d} \mid  j = 1,\ldots ,{L}_{K}}\right\}$，那么$\forall {M}_{m} =$ $\mathop{\max }\limits_{i}M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$存在$\kappa  > 0$，使得：在区间$\forall {\mathbf{q}}_{1},{\mathbf{q}}_{2} \in  \left\{  {\mathbf{q} \mid  M\left( {\mathbf{q},\mathbf{K}}\right)  \in  \left\lbrack  {{M}_{m},{M}_{m} - \kappa }\right) }\right\}$内，如果$\bar{M}\left( {{\mathbf{q}}_{1},\mathbf{K}}\right)  >$ $\bar{M}\left( {{\mathbf{q}}_{2},\mathbf{K}}\right)$且$\operatorname{Var}\left( {\mathbf{{qk}}}_{1}\right)  > \operatorname{Var}\left( {\mathbf{{qk}}}_{2}\right)$，那么我们有很大概率得到$M\left( {{\mathbf{q}}_{1},\mathbf{K}}\right)  > M\left( {{\mathbf{q}}_{2},\mathbf{K}}\right)$。

## Proof of Proposition 1

## 命题1的证明

Proof. To make the further discussion simplify, we can note ${a}_{i,j} = {q}_{i}{k}_{j}^{T}/\sqrt{d}$ ,thus define the array ${A}_{i} = \left\lbrack  {{a}_{i,1},\cdots ,{a}_{i,{L}_{k}}}\right\rbrack$ . Moreover,we denote $\frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left( {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right)  = \operatorname{mean}\left( {A}_{i}\right)$ ,then we can denote $\bar{M}\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  = \max \left( {A}_{i}\right)  - \operatorname{mean}\left( {A}_{i}\right) ,i = 1,2.$

证明。为了简化进一步的讨论，我们可以记${a}_{i,j} = {q}_{i}{k}_{j}^{T}/\sqrt{d}$，因此定义数组${A}_{i} = \left\lbrack  {{a}_{i,1},\cdots ,{a}_{i,{L}_{k}}}\right\rbrack$。此外，我们记$\frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left( {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right)  = \operatorname{mean}\left( {A}_{i}\right)$，然后我们可以记$\bar{M}\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  = \max \left( {A}_{i}\right)  - \operatorname{mean}\left( {A}_{i}\right) ,i = 1,2.$

As for $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ ,we denote each component ${a}_{i,j} =$ $\operatorname{mean}\left( {A}_{i}\right)  + \Delta {a}_{i,j},j = 1,\cdots ,{L}_{k}$ ,then we have the following:

对于 $M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$，我们将每个分量表示为 ${a}_{i,j} =$ $\operatorname{mean}\left( {A}_{i}\right)  + \Delta {a}_{i,j},j = 1,\cdots ,{L}_{k}$，那么我们有以下内容：

$$
M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)  = \ln \mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}{e}^{{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}} - \frac{1}{{L}_{K}}\mathop{\sum }\limits_{{j = 1}}^{{L}_{K}}\left( {{\mathbf{q}}_{i}{\mathbf{k}}_{j}^{\top }/\sqrt{d}}\right) 
$$

$$
 = \ln \left( {{\sum }_{j = 1}^{{L}_{k}}{e}^{\operatorname{mean}\left( {A}_{i}\right) }{e}^{\Delta {a}_{i,j}}}\right)  - \operatorname{mean}\left( {A}_{i}\right) ,
$$

$$
 = \ln \left( {{e}^{\operatorname{mean}\left( {A}_{i}\right) }{\sum }_{j = 1}^{{L}_{k}}{e}^{\Delta {a}_{i,j}}}\right)  - \operatorname{mean}\left( {A}_{i}\right) 
$$

$$
 = \ln \left( {\mathop{\sum }\limits_{{j = 1}}^{{L}_{k}}{e}^{\Delta {a}_{i,j}}}\right) 
$$

and it is easy to find ${\sum }_{j = 1}^{{L}_{k}}\Delta {a}_{i,j} = 0$ .

并且很容易找到 ${\sum }_{j = 1}^{{L}_{k}}\Delta {a}_{i,j} = 0$。

We define the function ${ES}\left( {A}_{i}\right)  = {\sum }_{j = 1}^{{L}_{k}}\exp \left( {\Delta {a}_{i,j}}\right)$ , equivalently defines ${A}_{i} = \left\lbrack  {\Delta {a}_{i,1},\cdots ,\Delta {a}_{i,{L}_{k}}}\right\rbrack$ ,and immediately our proposition can be written as the equivalent form:

我们定义函数 ${ES}\left( {A}_{i}\right)  = {\sum }_{j = 1}^{{L}_{k}}\exp \left( {\Delta {a}_{i,j}}\right)$，等价地定义 ${A}_{i} = \left\lbrack  {\Delta {a}_{i,1},\cdots ,\Delta {a}_{i,{L}_{k}}}\right\rbrack$，并且我们的命题可以立即写成等价形式：

For $\forall {A}_{1},{A}_{2}$ ,if

对于 $\forall {A}_{1},{A}_{2}$ ，如果

1. $\max \left( {A}_{1}\right)  - \operatorname{mean}\left( {A}_{1}\right)  \geq  \max \left( {A}_{2}\right)  - \operatorname{mean}\left( {A}_{2}\right)$

2. $\operatorname{Var}\left( {A}_{1}\right)  > \operatorname{Var}\left( {A}_{2}\right)$

Then we rephrase the original conclusion into more general form that ${ES}\left( {A}_{1}\right)  > {ES}\left( {A}_{2}\right)$ with high probability, and the probability have positive correlation with $\operatorname{Var}\left( {A}_{1}\right)  - \operatorname{Var}\left( {A}_{2}\right)$ .

然后我们将原结论重新表述为更一般的形式，即 ${ES}\left( {A}_{1}\right)  > {ES}\left( {A}_{2}\right)$ 以高概率成立，且该概率与 $\operatorname{Var}\left( {A}_{1}\right)  - \operatorname{Var}\left( {A}_{2}\right)$ 正相关。

Furthermore,we consider a fine case, $\forall {M}_{m} =$ $\mathop{\max }\limits_{i}M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ there exist $\kappa  > 0$ such that in that interval $\forall {\mathbf{q}}_{i},{\mathbf{q}}_{j} \in  \left\{  {\mathbf{q} \mid  M\left( {\mathbf{q},\mathbf{K}}\right)  \in  \left\lbrack  {{M}_{m},{M}_{m} - \kappa }\right) }\right\}$ if $\max \left( {A}_{1}\right)  -$ $\operatorname{mean}\left( {A}_{1}\right)  \geq  \max \left( {A}_{2}\right)  - \operatorname{mean}\left( {A}_{2}\right)$ and $\operatorname{Var}\left( {A}_{1}\right)  >$ $\operatorname{Var}\left( {A}_{2}\right)$ ,we have high probability that $M\left( {{\mathbf{q}}_{1},\mathbf{K}}\right)  >$ $M\left( {{\mathbf{q}}_{2},\mathbf{K}}\right)$ ,which is equivalent to ${ES}\left( {A}_{1}\right)  > {ES}\left( {A}_{2}\right)$ .

此外，我们考虑一个精细的情况，$\forall {M}_{m} =$ $\mathop{\max }\limits_{i}M\left( {{\mathbf{q}}_{i},\mathbf{K}}\right)$ 存在 $\kappa  > 0$ 使得在该区间 $\forall {\mathbf{q}}_{i},{\mathbf{q}}_{j} \in  \left\{  {\mathbf{q} \mid  M\left( {\mathbf{q},\mathbf{K}}\right)  \in  \left\lbrack  {{M}_{m},{M}_{m} - \kappa }\right) }\right\}$ 内，如果 $\max \left( {A}_{1}\right)  -$ $\operatorname{mean}\left( {A}_{1}\right)  \geq  \max \left( {A}_{2}\right)  - \operatorname{mean}\left( {A}_{2}\right)$ 且 $\operatorname{Var}\left( {A}_{1}\right)  >$ $\operatorname{Var}\left( {A}_{2}\right)$ ，那么我们有很大概率得到 $M\left( {{\mathbf{q}}_{1},\mathbf{K}}\right)  >$ $M\left( {{\mathbf{q}}_{2},\mathbf{K}}\right)$ ，这等价于 ${ES}\left( {A}_{1}\right)  > {ES}\left( {A}_{2}\right)$ 。

In the original proposition, ${\mathbf{k}}_{\mathbf{j}} \sim  \mathcal{N}\left( {\mu ,\sum }\right)$ follows multivariate Gaussian distribution,which means that ${k}_{1},\cdots ,{k}_{n}$ are I.I.D Gaussian distribution, thus defined by the Wiener-khinchin law of large Numbers, ${a}_{i,j} = {q}_{i}{k}_{j}^{T}/\sqrt{d}$ is one-dimension Gaussian distribution with the expectation of 0 if $n \rightarrow  \infty$ . So back to our definition, $\Delta {a}_{1,m} \sim$ $N\left( {0,{\sigma }_{1}^{2}}\right) ,\Delta {a}_{2,m} \sim  N\left( {0,{\sigma }_{2}^{2}}\right) ,\forall m \in  1,\cdots ,{L}_{k}$ ,and our proposition is equivalent to a lognormal-distribution sum problem.

在原命题中，${\mathbf{k}}_{\mathbf{j}} \sim  \mathcal{N}\left( {\mu ,\sum }\right)$ 服从多元高斯分布，这意味着 ${k}_{1},\cdots ,{k}_{n}$ 是独立同分布（I.I.D）的高斯分布。因此，根据维纳 - 辛钦大数定律（Wiener - Khinchin law of large Numbers），如果 $n \rightarrow  \infty$ 成立，那么 ${a}_{i,j} = {q}_{i}{k}_{j}^{T}/\sqrt{d}$ 是期望为 0 的一维高斯分布。回到我们的定义，$\Delta {a}_{1,m} \sim$ $N\left( {0,{\sigma }_{1}^{2}}\right) ,\Delta {a}_{2,m} \sim  N\left( {0,{\sigma }_{2}^{2}}\right) ,\forall m \in  1,\cdots ,{L}_{k}$，并且我们的命题等价于一个对数正态分布求和问题。

A lognormal-distribution sum problem is equivalent to approximating the distribution of $\operatorname{ES}\left( {A}_{1}\right)$ accurately, whose history is well-introduced in the articles (Dufresne 2008),(Vargasguzman 2005). Approximating lognormality of sums of lognormals is a well-known rule of thumb, and no general PDF function can be given for the sums of log-normals. However, (Romeo, Da Costa, and Bardou 2003) and (Hcine and Bouallegue 2015) pointed out that in most cases, sums of lognormals is still a lognormal distribution, and by applying central limits theorem in (Beaulieu 2011), we can have a good approximation that ${ES}\left( {A}_{1}\right)$ is a $\log$ - normal distribution,and we have $E\left( {{ES}\left( {A}_{1}\right) }\right)  = n{e}^{\frac{{\sigma }_{1}^{2}}{2}}$ , $\operatorname{Var}\left( {{ES}\left( {A}_{1}\right) }\right)  = n{e}^{{\sigma }_{1}^{2}}\left( {{e}^{{\sigma }_{1}^{2}} - 1}\right)$ . Equally, $E\left( {{ES}\left( {A}_{2}\right) }\right)  =$ $n{e}^{\frac{{\sigma }_{2}^{2}}{2}},\operatorname{Var}\left( {{ES}\left( {A}_{2}\right) }\right)  = n{e}^{{\sigma }_{2}^{2}}\left( {{e}^{{\sigma }_{2}^{2}} - 1}\right) .$

对数正态分布求和问题等价于精确逼近$\operatorname{ES}\left( {A}_{1}\right)$的分布，相关历史在文章（迪弗雷纳（Dufresne），2008年）、（巴尔加斯·古兹曼（Vargasguzman），2005年）中有详细介绍。用对数正态分布来近似对数正态变量之和是一种广为人知的经验法则，而且对于对数正态变量之和，无法给出通用的概率密度函数（PDF）。然而，（罗密欧（Romeo）、达·科斯塔（Da Costa）和巴杜（Bardou），2003年）以及（希内（Hcine）和布阿勒盖（Bouallegue），2015年）指出，在大多数情况下，对数正态变量之和仍然是对数正态分布。通过应用（博略（Beaulieu），2011年）中的中心极限定理，我们可以很好地近似认为${ES}\left( {A}_{1}\right)$是$\log$ - 正态分布，并且有$E\left( {{ES}\left( {A}_{1}\right) }\right)  = n{e}^{\frac{{\sigma }_{1}^{2}}{2}}$ ， $\operatorname{Var}\left( {{ES}\left( {A}_{1}\right) }\right)  = n{e}^{{\sigma }_{1}^{2}}\left( {{e}^{{\sigma }_{1}^{2}} - 1}\right)$ 。同样地，$E\left( {{ES}\left( {A}_{2}\right) }\right)  =$ $n{e}^{\frac{{\sigma }_{2}^{2}}{2}},\operatorname{Var}\left( {{ES}\left( {A}_{2}\right) }\right)  = n{e}^{{\sigma }_{2}^{2}}\left( {{e}^{{\sigma }_{2}^{2}} - 1}\right) .$

We denote ${B}_{1} = {ES}\left( {A}_{1}\right) ,{B}_{2} = {ES}\left( {A}_{2}\right)$ ,and the probability $\Pr \left( {{B}_{1} - {B}_{2} > 0}\right)$ is the final result of our proposition in general conditions,with ${\sigma }_{1}^{2} > {\sigma }_{2}^{2}$ WLOG. The difference of lognormals is still a hard problem to solve.

我们记${B}_{1} = {ES}\left( {A}_{1}\right) ,{B}_{2} = {ES}\left( {A}_{2}\right)$，并且概率$\Pr \left( {{B}_{1} - {B}_{2} > 0}\right)$是我们的命题在一般条件下的最终结果，不失一般性地设${\sigma }_{1}^{2} > {\sigma }_{2}^{2}$。对数正态分布的差值仍然是一个难以解决的问题。

By using the theorem given in(Lo 2012), which gives a general approximation of the probability distribution on the sums and difference for the lognormal distribution. Namely ${S}_{1}$ and ${S}_{2}$ are two lognormal stochastic variables obeying the stochastic differential equations $\frac{d{S}_{i}}{{S}_{i}} = {\sigma }_{i}d{Z}_{i},i = 1,2$ , in which $d{Z}_{1,2}$ presents a standard Weiner process associated with ${S}_{1,2}$ respectively,and ${\sigma }_{i}^{2} = \operatorname{Var}\left( {\ln {S}_{i}}\right) ,{S}^{ \pm  } \equiv$ ${S}_{1} \pm  {S}_{2},{S}_{0}^{ \pm  } \equiv  {S}_{10} \pm  {S}_{20}$ . As for the joint probability distribution function $P\left( {{S}_{1},{S}_{2},t;{S}_{10},{S}_{20},{t}_{0}}\right)$ ,the value of ${S}_{1}$ and ${S}_{2}$ at time $t > {t}_{0}$ are provided by their initial value ${S}_{10}$ and ${S}_{20}$ at initial time ${t}_{0}$ . The Weiner process above is equivalent to the lognormal distribution (Weiner and Solbrig 1984), and the conclusion below is written in general form containing both the sum and difference of lognormal distribution approximation denoting $\pm$ for sum + and difference - respectively.

通过使用（Lo 2012）中给出的定理，该定理给出了对数正态分布的和与差的概率分布的一般近似。即${S}_{1}$和${S}_{2}$是两个服从随机微分方程$\frac{d{S}_{i}}{{S}_{i}} = {\sigma }_{i}d{Z}_{i},i = 1,2$的对数正态随机变量，其中$d{Z}_{1,2}$分别表示与${S}_{1,2}$相关的标准维纳过程（Weiner process），且${\sigma }_{i}^{2} = \operatorname{Var}\left( {\ln {S}_{i}}\right) ,{S}^{ \pm  } \equiv$ ${S}_{1} \pm  {S}_{2},{S}_{0}^{ \pm  } \equiv  {S}_{10} \pm  {S}_{20}$。对于联合概率分布函数$P\left( {{S}_{1},{S}_{2},t;{S}_{10},{S}_{20},{t}_{0}}\right)$，${S}_{1}$和${S}_{2}$在时间$t > {t}_{0}$的值由它们在初始时间${t}_{0}$的初始值${S}_{10}$和${S}_{20}$给出。上述维纳过程等价于对数正态分布（Weiner和Solbrig 1984），下面的结论以一般形式书写，包含对数正态分布近似的和与差，分别用$\pm$表示和（+）与差（-）。

In boundary condition

在边界条件下

$$
{\bar{P}}_{ \pm  }\left( {{S}^{ \pm  },t;{S}_{10},{S}_{20},{t}_{0} \rightarrow  t}\right)  = \delta \left( {{S}_{10} \pm  {S}_{20} - {S}^{ \pm  }}\right) ,
$$

their closed-form probability distribution functions are given by

它们的闭式概率分布函数由以下式子给出

$$
{f}^{\mathrm{{LN}}}\left( {{\widetilde{S}}^{ \pm  },t;{\widetilde{S}}_{0}^{ \pm  },{t}_{0}}\right) 
$$

$$
 = \frac{1}{{\widetilde{S}}^{ \pm  }\sqrt{{2\pi }{\widetilde{\sigma }}_{ \pm  }^{2}\left( {t - {t}_{0}}\right) }}
$$

$$
 \cdot  \exp \left\{  {-\frac{{\left\lbrack  \ln \left( {\widetilde{S}}^{ + }/{\widetilde{S}}_{0}^{ + }\right)  + \left( 1/2\right) {\widetilde{\sigma }}_{ \pm  }^{2}\left( t - {t}_{0}\right) \right\rbrack  }^{2}}{2{\widetilde{\sigma }}_{ \pm  }^{2}\left( {t - {t}_{0}}\right) }}\right\}  
$$

It is an approximately normal distribution,and ${\widetilde{S}}^{ + },{\widetilde{S}}^{ - }$ are lognormal random variables, ${\widetilde{S}}_{0}^{ \pm  }$ are initial condition in ${t}_{0}$ defined by Weiner process above. (Noticed that ${\widetilde{\sigma }}_{ \pm  }^{2}\left( {t - {t}_{0}}\right)$ should be small to make this approximation valid.In our simulation experiment,we set $t - {t}_{0} = 1$ WLOG.) Since

这是一个近似正态分布，且${\widetilde{S}}^{ + },{\widetilde{S}}^{ - }$是对数正态随机变量，${\widetilde{S}}_{0}^{ \pm  }$是由上述维纳过程（Weiner process）在${t}_{0}$中定义的初始条件。（注意，为使该近似有效，${\widetilde{\sigma }}_{ \pm  }^{2}\left( {t - {t}_{0}}\right)$应较小。在我们的模拟实验中，不失一般性地（WLOG），我们设$t - {t}_{0} = 1$。）由于

$$
{\widetilde{S}}_{0}^{ - } = \left( {{S}_{10} - {S}_{20}}\right)  + \left( \frac{{\sigma }_{ - }^{2}}{{\sigma }_{1}^{2} - {\sigma }_{2}^{2}}\right) \left( {{S}_{10} + {S}_{20}}\right) ,
$$

and

以及

$$
\widetilde{{\sigma }_{ - }} = \left( {{\sigma }_{1}^{2} - {\sigma }_{2}^{2}}\right) /\left( {2{\sigma }_{ - }}\right) 
$$

$$
{\sigma }_{ - } = \sqrt{{\sigma }_{1}^{2} + {\sigma }_{2}^{2}}
$$

Noticed that $E\left( {B}_{1}\right)  > E\left( {B}_{2}\right) ,\operatorname{Var}\left( {B}_{1}\right)  > \operatorname{Var}\left( {B}_{2}\right)$ ,the mean value and the variance of the approximate normal distribution shows positive correlation with ${\sigma }_{1}^{2} - {\sigma }_{2}^{2}$ . Besides, the closed-form PDF ${f}^{\mathrm{{LN}}}\left( {{\widetilde{S}}^{ \pm  },t;{\widetilde{S}}_{0}^{ \pm  },{t}_{0}}\right)$ also show positive correlation with ${\sigma }_{1}^{2} - {\sigma }_{2}^{2}$ . Due to the limitation of ${\widetilde{\sigma }}_{ \pm  }^{2}\left( {t - {t}_{0}}\right)$ should be small enough,such positive correlation is not significant in our illustrative numerical experiment.

注意到$E\left( {B}_{1}\right)  > E\left( {B}_{2}\right) ,\operatorname{Var}\left( {B}_{1}\right)  > \operatorname{Var}\left( {B}_{2}\right)$，近似正态分布的均值和方差与${\sigma }_{1}^{2} - {\sigma }_{2}^{2}$呈正相关。此外，封闭形式的概率密度函数（PDF）${f}^{\mathrm{{LN}}}\left( {{\widetilde{S}}^{ \pm  },t;{\widetilde{S}}_{0}^{ \pm  },{t}_{0}}\right)$也与${\sigma }_{1}^{2} - {\sigma }_{2}^{2}$呈正相关。由于${\widetilde{\sigma }}_{ \pm  }^{2}\left( {t - {t}_{0}}\right)$的限制条件是必须足够小，在我们的示例数值实验中，这种正相关性并不显著。

By using Lie-Trotter Operator Splitting Method in (Lo 2012), we can give illustrative numeral examples for the distribution of ${B}_{1} - {B}_{2}$ ,in which the parameters are well chosen to fit for our top-u approximation in actual LLLT experiments. Figure shows that it is of high probability that when ${\sigma }_{1}^{2} > {\sigma }_{2}^{2}$ ,the inequality holds that ${B}_{1} > {B}_{2}$ , ${ES}\left( {A}_{1}\right)  > \bar{E}S\left( {A}_{2}\right)$ .

通过使用（Lo 2012）中的李 - 特罗特算子分裂法（Lie-Trotter Operator Splitting Method），我们可以给出${B}_{1} - {B}_{2}$分布的说明性数值示例，其中参数经过精心选择，以适用于我们在实际局部线性对数变换（LLLT）实验中的前u近似。图显示，当${\sigma }_{1}^{2} > {\sigma }_{2}^{2}$时，不等式${B}_{1} > {B}_{2}$、${ES}\left( {A}_{1}\right)  > \bar{E}S\left( {A}_{2}\right)$成立的概率很高。

<!-- Media -->

<!-- figureText: 0.0175 $=  - {S}_{10} = {110},{S}_{20} = {100},{\sigma }_{1} = {0.3},{\sigma }_{2} = {0.2}$ ${S}_{10} = {170},{S}_{20} = {130},{\sigma }_{1} = {0.25},{\sigma }_{2} = {0.15}$ 0.0150 0.0075 0.0050 -->

<img src="https://cdn.noedgeai.com/01957ae6-5cfc-72b0-ad59-2c1c4e1a17b2_11.jpg?x=262&y=189&w=505&h=368&r=0"/>

Figure 8: Probability Density verses ${S}_{1} - {S}_{2}$ for the approximation of shifted lognormal distribution.

图8：平移对数正态分布近似的概率密度与${S}_{1} - {S}_{2}$的关系。

<!-- Media -->

Finishing prooving our proposition in general conditions, we can consider a more specific condition that if ${\mathbf{q}}_{1},{\mathbf{q}}_{2} \in$ $\left\{  {\mathbf{q} \mid  M\left( {\mathbf{q},\mathbf{K}}\right)  \in  \left\lbrack  {{M}_{m},{M}_{m} - \kappa }\right) }\right\}$ ,the proposition still holds with high probability.

在一般条件下完成对我们命题的证明后，我们可以考虑一个更具体的条件，即如果 ${\mathbf{q}}_{1},{\mathbf{q}}_{2} \in$ $\left\{  {\mathbf{q} \mid  M\left( {\mathbf{q},\mathbf{K}}\right)  \in  \left\lbrack  {{M}_{m},{M}_{m} - \kappa }\right) }\right\}$ ，该命题仍大概率成立。

First,we have $M\left( {{q}_{1},\mathbf{k}}\right)  = \ln \left( {B}_{1}\right)  > \left( {{M}_{m} - \kappa }\right)$ holds for $\forall {q}_{1},{q}_{2}$ in this interval. Since we have proved that $\left. {E\left( {B}_{1}\right) }\right)  = n{e}^{\frac{{\sigma }_{1}^{2}}{2}}$ ,we can conclude that $\forall {q}_{i}$ in the given interval, $\exists \alpha ,{\sigma }_{i}^{2} > \alpha ,i = 1,2$ . Since we have ${\widetilde{S}}_{0}^{ - } =$ $\left( {{S}_{10} - {S}_{20}}\right)  + \left( \frac{{\sigma }_{ - }^{2}}{{\sigma }_{1}^{2} - {\sigma }_{2}^{2}}\right) \left( {{S}_{10} + {S}_{20}}\right)$ ,which also shows positive correlation with ${\sigma }_{1}^{2} + {\sigma }_{2}^{2} > {2\alpha }$ ,and positive correlation with ${\sigma }_{1}^{2} - {\sigma }_{2}^{2}$ . So due to the nature of the approximate normal distribution PDF,if ${\sigma }_{1}^{2} > {\sigma }_{2}^{2}$ WLOG, $\Pr \left( {M\left( {{q}_{1},\mathbf{k}}\right)  > M\left( {{q}_{2},\mathbf{k}}\right) }\right)  \approx  \Phi \left( \frac{{\widetilde{S}}_{0}^{ - }}{{\sigma }_{ - }^{ - }}\right)$ also shows positive correlation with ${\sigma }_{1}^{2} + {\sigma }_{2}^{2} > {2\alpha }$ .

首先，我们有对于该区间内的$\forall {q}_{1},{q}_{2}$，$M\left( {{q}_{1},\mathbf{k}}\right)  = \ln \left( {B}_{1}\right)  > \left( {{M}_{m} - \kappa }\right)$成立。由于我们已经证明了$\left. {E\left( {B}_{1}\right) }\right)  = n{e}^{\frac{{\sigma }_{1}^{2}}{2}}$，我们可以得出在给定区间$\exists \alpha ,{\sigma }_{i}^{2} > \alpha ,i = 1,2$内$\forall {q}_{i}$成立。因为我们有${\widetilde{S}}_{0}^{ - } =$ $\left( {{S}_{10} - {S}_{20}}\right)  + \left( \frac{{\sigma }_{ - }^{2}}{{\sigma }_{1}^{2} - {\sigma }_{2}^{2}}\right) \left( {{S}_{10} + {S}_{20}}\right)$，这也表明其与${\sigma }_{1}^{2} + {\sigma }_{2}^{2} > {2\alpha }$正相关，且与${\sigma }_{1}^{2} - {\sigma }_{2}^{2}$正相关。因此，根据近似正态分布概率密度函数（PDF）的性质，不失一般性地（WLOG），若${\sigma }_{1}^{2} > {\sigma }_{2}^{2}$，则$\Pr \left( {M\left( {{q}_{1},\mathbf{k}}\right)  > M\left( {{q}_{2},\mathbf{k}}\right) }\right)  \approx  \Phi \left( \frac{{\widetilde{S}}_{0}^{ - }}{{\sigma }_{ - }^{ - }}\right)$也与${\sigma }_{1}^{2} + {\sigma }_{2}^{2} > {2\alpha }$正相关。

We give an illustrative numerical examples of the approximation above in Fig. 8). In our actual LTTnet experiment, we choose Top-k of ${\bar{A}}_{1},{A}_{2}$ ,not the whole set.Actually, we can make a naive assumption that in choosing top - $\left\lfloor  {\frac{1}{4}{L}_{k}}\right\rfloor$ variables of ${A}_{1},{A}_{2}$ denoted as ${A}_{1}^{\prime },{A}_{2}^{\prime }$ ,the variation ${\sigma }_{1},{\sigma }_{2}$ don’t change significantly,but the expectation $E\left( {A}_{1}^{\prime }\right) ,E\left( {A}_{2}^{\prime }\right)$ ascends obviously,which leads to initial condition ${S}_{10},{S}_{20}$ ascends significantly,since the initial condition will be sampled from $\operatorname{top} - \left\lfloor  {\frac{1}{4}{L}_{k}}\right\rfloor$ variables,not the whole set.

我们在图8中给出了上述近似的一个示例数值例子。在我们实际的LTTnet实验中，我们选择${\bar{A}}_{1},{A}_{2}$的前k个（Top - k），而非整个集合。实际上，我们可以做一个简单的假设：在选择${A}_{1},{A}_{2}$的前$\left\lfloor  {\frac{1}{4}{L}_{k}}\right\rfloor$个变量（记为${A}_{1}^{\prime },{A}_{2}^{\prime }$）时，方差${\sigma }_{1},{\sigma }_{2}$不会显著变化，但期望$E\left( {A}_{1}^{\prime }\right) ,E\left( {A}_{2}^{\prime }\right)$会明显上升，这导致初始条件${S}_{10},{S}_{20}$显著上升，因为初始条件将从$\operatorname{top} - \left\lfloor  {\frac{1}{4}{L}_{k}}\right\rfloor$个变量中采样，而非整个集合。

In our actual LTTnet experiment,we set $U$ ,namely choosing around $\operatorname{top} - \left\lfloor  {\frac{1}{4}{L}_{k}}\right\rfloor$ of ${A}_{1}$ and ${A}_{2}$ ,it is guaranteed that with over ${99}\%$ probability that in the $\left\lbrack  {{M}_{m},{M}_{m} - \kappa }\right)$ interval, as shown in the black curve of Fig. 8). Typically the condition 2 can be relaxed, and we can believe that if ${q}_{1},{q}_{2}$ fits the condition 1 in our proposition,we have $M\left( {{\mathbf{q}}_{1},\mathbf{K}}\right)  > M\left( {{\mathbf{q}}_{2},\mathbf{K}}\right)$ .

在我们实际的LTTnet实验中，我们设置$U$，即选择${A}_{1}$和${A}_{2}$的约$\operatorname{top} - \left\lfloor  {\frac{1}{4}{L}_{k}}\right\rfloor$，可以保证在$\left\lbrack  {{M}_{m},{M}_{m} - \kappa }\right)$区间内，以超过${99}\%$的概率，如图8中的黑色曲线所示）。通常条件2可以放宽，并且我们可以认为，如果${q}_{1},{q}_{2}$符合我们命题中的条件1，那么我们有$M\left( {{\mathbf{q}}_{1},\mathbf{K}}\right)  > M\left( {{\mathbf{q}}_{2},\mathbf{K}}\right)$。

## Appendix E Reproducibility

## 附录E 可重复性

## Details of the experiments

## 实验细节

The details of proposed Informer model is summarized in Table 7. For the ProbSparse self-attention mechanism, we let $d = {32},n = {16}$ and add residual connections,a position-wise feed-forward network layer (inner-layer dimension is 2048) and a dropout layer $\left( {p = {0.1}}\right)$ likewise. Note that we preserves ${10}\%$ validation data for each dataset,so all the experiments are conducted over 5 random train/val shifting selection along time and the results are averaged over the 5 runs. All the datasets are performed standardization such that the mean of variable is 0 and the standard deviation is 1 .

所提出的Informer模型（Informer模型）的细节总结在表7中。对于ProbSparse自注意力机制，我们令$d = {32},n = {16}$并添加残差连接、一个逐位置前馈网络层（内层维度为2048）和一个丢弃层$\left( {p = {0.1}}\right)$。请注意，我们为每个数据集保留${10}\%$验证数据，因此所有实验都在5次随机的训练/验证时间移位选择上进行，结果是这5次运行的平均值。所有数据集都进行了标准化处理，使得变量的均值为0，标准差为1。

<!-- Media -->

Table 7: The Informer network components in details

表7：Informer网络（Informer网络）组件详情

<table><tr><td colspan="4">Encoder:$\mathrm{N}$</td></tr><tr><td>Inputs</td><td>1x3 Conv1d</td><td>Embedding $\left( {d = {512}}\right)$</td><td rowspan="7">4</td></tr><tr><td rowspan="4">ProbSparse Self-attention Block</td><td colspan="2">Multi-head ProbSparse Attention $\left( {h = {16},d = {32}}\right)$</td></tr><tr><td colspan="2">Add,LayerNorm,Dropout $\left( {p = {0.1}}\right)$</td></tr><tr><td colspan="2">Pos-wise FFN $\left( {{d}_{\text{inner }} = {2048}}\right) ,$ GELU</td></tr><tr><td colspan="2">Add,LayerNorm,Dropout $\left( {p = {0.1}}\right)$</td></tr><tr><td rowspan="2">Distilling</td><td colspan="2">1x3 conv1d, ELU</td></tr><tr><td colspan="2">Max pooling (stride $= 2$ )</td></tr><tr><td colspan="4">Decoder:N</td></tr><tr><td>Inputs</td><td>1x3 Conv1d</td><td>Embedding $\left( {d = {512}}\right)$</td><td rowspan="6">2</td></tr><tr><td>Masked PSB</td><td colspan="2">add Mask on Attention Block</td></tr><tr><td rowspan="4">Self-attention Block</td><td colspan="2">Multi-head Attention $\left( {h = 8,d = {64}}\right)$</td></tr><tr><td colspan="2">Add,LayerNorm,Dropout $\left( {p = {0.1}}\right)$</td></tr><tr><td colspan="2">Pos-wise FFN $\left( {{d}_{\text{inner }} = {2048}}\right)$ ,GELU</td></tr><tr><td/><td>Add,LayerNorm,Dropout $\left( {p = {0.1}}\right)$</td></tr><tr><td colspan="4">Final:</td></tr><tr><td>Outputs</td><td colspan="2">$\mathrm{{FCN}}\left( {d = {d}_{\mathrm{{out}}}}\right)$</td><td/></tr></table>

<table><tbody><tr><td colspan="4">编码器:$\mathrm{N}$</td></tr><tr><td>输入</td><td>1x3一维卷积（1x3 Conv1d）</td><td>嵌入层 $\left( {d = {512}}\right)$</td><td rowspan="7">4</td></tr><tr><td rowspan="4">概率稀疏自注意力模块（ProbSparse Self-attention Block）</td><td colspan="2">多头概率稀疏注意力 $\left( {h = {16},d = {32}}\right)$</td></tr><tr><td colspan="2">加法、层归一化（LayerNorm）、随机失活（Dropout） $\left( {p = {0.1}}\right)$</td></tr><tr><td colspan="2">位置式前馈网络（Pos-wise FFN） $\left( {{d}_{\text{inner }} = {2048}}\right) ,$ 高斯误差线性单元（GELU）</td></tr><tr><td colspan="2">加法、层归一化（LayerNorm）、随机失活（Dropout） $\left( {p = {0.1}}\right)$</td></tr><tr><td rowspan="2">蒸馏</td><td colspan="2">1x3一维卷积（conv1d），指数线性单元（ELU）</td></tr><tr><td colspan="2">最大池化（步长 $= 2$ ）</td></tr><tr><td colspan="4">解码器：N</td></tr><tr><td>输入</td><td>1x3一维卷积（1x3 Conv1d）</td><td>嵌入层 $\left( {d = {512}}\right)$</td><td rowspan="6">2</td></tr><tr><td>掩码化的位置敏感块（Masked PSB）</td><td colspan="2">在注意力模块上添加掩码</td></tr><tr><td rowspan="4">自注意力模块（Self-attention Block）</td><td colspan="2">多头注意力 $\left( {h = 8,d = {64}}\right)$（Multi-head Attention $\left( {h = 8,d = {64}}\right)$）</td></tr><tr><td colspan="2">加法、层归一化（LayerNorm）、随机失活（Dropout） $\left( {p = {0.1}}\right)$</td></tr><tr><td colspan="2">逐位置前馈网络 $\left( {{d}_{\text{inner }} = {2048}}\right)$，高斯误差线性单元（Pos-wise FFN $\left( {{d}_{\text{inner }} = {2048}}\right)$ ,GELU）</td></tr><tr><td></td><td>加法、层归一化（LayerNorm）、随机失活（Dropout） $\left( {p = {0.1}}\right)$</td></tr><tr><td colspan="4">最终结果：</td></tr><tr><td>输出</td><td colspan="2">$\mathrm{{FCN}}\left( {d = {d}_{\mathrm{{out}}}}\right)$</td><td></td></tr></tbody></table>

<!-- Media -->

## Implement of the ProbSparse self-attention

## 概率稀疏自注意力机制的实现

We have implemented the ProbSparse self-attention in Python 3.6 with Pytorch 1.0. The pseudo-code is given in Algo. 1). The source code is available at https://github.com/ zhouhaoyi/Informer2020. All the procedure can be highly efficient vector operation and maintains logarithmic total memory usage. The masked version can be achieved by applying positional mask on step 6 and using cmusum $\left( \cdot \right)$ in mean $\left( \cdot \right)$ of step 7 . In the practice,we can use $\operatorname{sum}\left( \cdot \right)$ as the simpler implement of mean $\left( \cdot \right)$ .

我们使用Python 3.6和PyTorch 1.0实现了概率稀疏自注意力机制。伪代码见算法1。源代码可在https://github.com/zhouhaoyi/Informer2020获取。所有过程都可以通过高效的向量运算实现，并保持对数级的总内存使用量。掩码版本可以通过在步骤6应用位置掩码，并在步骤7的均值计算中使用累积和$\left( \cdot \right)$来实现。在实践中，我们可以使用$\operatorname{sum}\left( \cdot \right)$作为均值$\left( \cdot \right)$的更简单实现。

<!-- Media -->

Algorithm 1 ProbSparse self-attention

算法1 概率稀疏自注意力机制

---

Require: Tensor $\mathbf{Q} \in  {\mathbb{R}}^{m \times  d},\mathbf{K} \in  {\mathbb{R}}^{n \times  d},\mathbf{V} \in  {\mathbb{R}}^{n \times  d}$

	1: print set hyperparameter $c,u = c\ln m$ and $U = m\ln n$

		randomly select $U$ dot-product pairs from $\mathbf{K}$ as $\overline{\mathbf{K}}$

		set the sample score $\overline{\mathbf{S}} = \mathbf{Q}{\overline{\mathbf{K}}}^{\top }$

		compute the measurement $M = \max \left( \overline{\mathbf{S}}\right)  - \operatorname{mean}\left( \overline{\mathbf{S}}\right)$ by row

		set Top- $u$ queries under $M$ as $\overline{\mathbf{Q}}$

		set ${\mathbf{S}}_{1} = \operatorname{softmax}\left( {\overline{\mathbf{Q}}{\mathbf{K}}^{\top }/\sqrt{d}}\right)  \cdot  \mathbf{V}$

		set ${\mathbf{S}}_{0} = \operatorname{mean}\left( \mathbf{V}\right)$

	8: set $\mathbf{S} = \left\{  {{\mathbf{S}}_{1},{\mathbf{S}}_{0}}\right\}$ by their original rows accordingly

Ensure: self-attention feature map $\mathbf{S}$ .

---

<!-- Media -->

## The hyperparameter tuning range

## 超参数调优范围

For all methods, the input length of recurrent component is chosen from $\{ {24},{48},{96},{168},{336},{720}\}$ for the ETTh1, ETTh2, Weather and Electricity dataset, and chosen from $\{ {24},{48},{96},{192},{288},{672}\}$ for the ETTm dataset. For LSTMa and DeepAR, the size of hidden states is chosen from $\{ {32},{64},{128},{256}\}$ . For LSTnet,the hidden dimension of the Recurrent layer and Convolutional layer is chosen from $\{ {64},{128},{256}\}$ and $\{ {32},{64},{128}\}$ for Recurrent-skip layer, and the skip-length of Recurrent-skip layer is set as 24 for the ETTh1, ETTh2, Weather and ECL dataset, and set as 96 for the ETTm dataset. For Informer, the layer of encoder is chosen from $\{ 6,4,3,2\}$ and the layer of decoder is set as 2 . The head number of multi-head attention is chosen from $\{ 8,{16}\}$ ,and the dimension of multi-head attention’s output is set as 512 . The length of encoder's input sequence and decoder’s start token is chosen from $\{ {24},{48},{96},{168}$ , ${336},{480},{720}\}$ for the ETTh1,ETTh2,Weather and ECL dataset,and $\{ {24},{48},{96},{192},{288},{480},{672}\}$ for the ETTm dataset. In the experiment, the decoder's start token is a segment truncated from the encoder's input sequence, so the length of decoder's start token must be less than the length of encoder's input.

对于所有方法，循环组件的输入长度在ETTh1、ETTh2、气象（Weather）和电力（Electricity）数据集上从$\{ {24},{48},{96},{168},{336},{720}\}$中选择，在ETTm数据集上从$\{ {24},{48},{96},{192},{288},{672}\}$中选择。对于LSTMa和DeepAR，隐藏状态的大小从$\{ {32},{64},{128},{256}\}$中选择。对于LSTnet，循环层和卷积层的隐藏维度从$\{ {64},{128},{256}\}$中选择，循环跳跃层（Recurrent - skip layer）的隐藏维度从$\{ {32},{64},{128}\}$中选择，并且循环跳跃层的跳跃长度在ETTh1、ETTh2、气象和电力负荷（ECL）数据集上设置为24，在ETTm数据集上设置为96。对于Informer，编码器的层数从$\{ 6,4,3,2\}$中选择，解码器的层数设置为2。多头注意力的头数从$\{ 8,{16}\}$中选择，多头注意力输出的维度设置为512。编码器输入序列和解码器起始标记的长度在ETTh1、ETTh2、气象和电力负荷数据集上从$\{ {24},{48},{96},{168}$、${336},{480},{720}\}$中选择，在ETTm数据集上从$\{ {24},{48},{96},{192},{288},{480},{672}\}$中选择。在实验中，解码器的起始标记是从编码器输入序列中截取的一段，因此解码器起始标记的长度必须小于编码器输入的长度。

<!-- Media -->

<img src="https://cdn.noedgeai.com/01957ae6-5cfc-72b0-ad59-2c1c4e1a17b2_12.jpg?x=151&y=156&w=1488&h=449&r=0"/>

Figure 9: The predicts (len=336) of Informer,Informer ${}^{ \dagger  }$ ,LogTrans,Reformer,DeepAR,LSTMa,ARIMA and Prophet on the ETTm dataset. The red / blue curves stand for slices of the prediction / ground truth.

图9：Informer、Informer ${}^{ \dagger  }$、LogTrans、Reformer、DeepAR、LSTMa、ARIMA和Prophet在ETTm数据集上的预测结果（长度=336）。红色/蓝色曲线分别代表预测值/真实值的切片。

<!-- Media -->

The RNN-based methods perform a dynamic decoding with left shifting on the prediction windows. Our proposed methods Informer-series and LogTrans (our decoder) perform non-dynamic decoding.

基于循环神经网络（RNN）的方法在预测窗口上进行左移动态解码。我们提出的Informer系列方法和LogTrans（我们的解码器）进行非动态解码。

## Appendix F Extra experimental results

## 附录F 额外实验结果

Fig. 9 presents a slice of the predicts of 8 models. The most realted work LogTrans and Reformer shows acceptable results. The LSTMa model is not amenable for the long sequence prediction task. The ARIMA and DeepAR can capture the long trend of the long sequences. And the Prophet detects the changing point and fits it with a smooth curve better than the ARIMA and DeepAR. Our proposed model Informer and Informer ${}^{ \dagger  }$ show significantly better results than above methods.

图9展示了8个模型预测结果的一个切片。最相关的工作LogTrans和Reformer显示出可接受的结果。LSTMa模型不适合长序列预测任务。ARIMA和DeepAR可以捕捉长序列的长期趋势。与ARIMA和DeepAR相比，Prophet能更好地检测到变化点并以平滑曲线进行拟合。我们提出的模型Informer和Informer ${}^{ \dagger  }$ 的结果明显优于上述方法。

## Appendix G Computing Infrastructure

## 附录G 计算基础设施

All the experiments are conducted on Nvidia Tesla V100 SXM2 GPUs (32GB memory). Other configuration includes 2 * Intel Xeon Gold 6148 CPU, 384GB DDR4 RAM and 2 * 240GB M. 2 SSD, which is sufficient for all the baselines.

所有实验均在英伟达（Nvidia）特斯拉（Tesla）V100 SXM2图形处理器（GPU，显存32GB）上进行。其他配置包括2颗英特尔至强（Intel Xeon）金牌6148中央处理器（CPU）、384GB DDR4随机存取存储器（RAM）和2块240GB M.2固态硬盘（SSD），这些配置足以运行所有基线模型。

## References

## 参考文献

Aicher, C.; Foti, N. J.; and Fox, E. B. 2019. Adaptively Truncating Backpropagation Through Time to Control Gradient Bias. arXiv:1905.07473 .

艾彻（Aicher, C.）；福蒂（Foti, N. J.）；福克斯（Fox, E. B.）。2019年。自适应截断时间反向传播以控制梯度偏差。预印本：arXiv:1905.07473 。

Ariyo, A. A.; Adewumi, A. O.; and Ayo, C. K. 2014. Stock price prediction using the ARIMA model. In The 16th International Conference on Computer Modelling and Simulation, 106-112. IEEE.

阿利约（Ariyo, A. A.）；阿德乌米（Adewumi, A. O.）；阿约（Ayo, C. K.）。2014年。使用自回归积分滑动平均（ARIMA）模型进行股票价格预测。见第16届计算机建模与仿真国际会议论文集，第106 - 112页。电气与电子工程师协会（IEEE）。

Bahdanau, D.; Cho, K.; and Bengio, Y. 2015. Neural Machine Translation by Jointly Learning to Align and Translate. In ICLR 2015.

巴赫达诺乌（Bahdanau, D.）；赵（Cho, K.）；本吉奥（Bengio, Y.）。2015年。通过联合学习对齐与翻译进行神经机器翻译。见2015年国际学习表征会议（ICLR 2015）论文集。

Bai, S.; Kolter, J. Z.; and Koltun, V. 2018. Convolutional sequence modeling revisited. ICLR .

白（Bai, S.）；科尔特（Kolter, J. Z.）；科尔图恩（Koltun, V.）。2018年。卷积序列建模再探。国际学习表征会议（ICLR）。

Beaulieu, N. C. 2011. An extended limit theorem for correlated lognormal sums. IEEE transactions on communications ${60}\left( 1\right)  : {23} - {26}$ .

博略（Beaulieu），N. C. 2011年。相关对数正态和的扩展极限定理。《电气与电子工程师协会通信汇刊》 ${60}\left( 1\right)  : {23} - {26}$ 。

Beltagy, I.; Peters, M. E.; and Cohan, A. 2020. Longformer: The Long-Document Transformer. CoRR abs/2004.05150.

贝尔塔吉（Beltagy），I.；彼得斯（Peters），M. E.；科恩（Cohan），A. 2020年。长文档变换器（Longformer）：长文档Transformer。计算机研究报告库（CoRR）abs/2004.05150。

Box, G. E.; Jenkins, G. M.; Reinsel, G. C.; and Ljung, G. M. 2015. Time series analysis: forecasting and control. John Wiley & Sons.

博克斯（Box），G. E.；詹金斯（Jenkins），G. M.；赖因塞尔（Reinsel），G. C.；永（Ljung），G. M. 2015年。时间序列分析：预测与控制。约翰·威利父子出版公司。

Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan, T.; Child, R.; Ramesh, A.; Ziegler, D. M.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever, I.; and Amodei, D. 2020. Language Models are Few-Shot Learners. CoRR abs/2005.14165.

布朗（Brown, T. B.）；曼（Mann, B.）；赖德（Ryder, N.）；苏比亚（Subbiah, M.）；卡普兰（Kaplan, J.）；达里瓦尔（Dhariwal, P.）；尼尔坎坦（Neelakantan, A.）；希亚姆（Shyam, P.）；萨斯特里（Sastry, G.）；阿斯凯尔（Askell, A.）；阿加瓦尔（Agarwal, S.）；赫伯特 - 沃斯（Herbert-Voss, A.）；克鲁格（Krueger, G.）；亨尼根（Henighan, T.）；蔡尔德（Child, R.）；拉梅什（Ramesh, A.）；齐格勒（Ziegler, D. M.）；吴（Wu, J.）；温特（Winter, C.）；黑塞（Hesse, C.）；陈（Chen, M.）；西格勒（Sigler, E.）；利特温（Litwin, M.）；格雷（Gray, S.）；切斯（Chess, B.）；克拉克（Clark, J.）；伯纳德（Berner, C.）；麦坎德利什（McCandlish, S.）；拉德福德（Radford, A.）；苏茨克维（Sutskever, I.）；阿莫迪（Amodei, D.）。2020年。语言模型是少样本学习者。计算机研究报告库（CoRR）abs/2005.14165。

Calafiore, G. C.; Gaubert, S.; and Possieri, C. 2018. Log-sum-exp neural networks and posynomial models for convex and log-log-convex data. CoRR abs/1806.07850.

卡拉菲奥雷（Calafiore, G. C.）；戈贝尔（Gaubert, S.）；波西耶里（Possieri, C.）。2018年。用于凸数据和对数 - 对数凸数据的对数求和指数神经网络和正项式模型。计算机研究报告库（CoRR）abs/1806.07850。

Cao, Y.; and Xu, P. 2019. Better Long-Range Dependency By Bootstrapping A Mutual Information Regularizer. arXiv:1905.11978 .

曹（Cao），Y.；徐（Xu），P. 2019年。通过引导互信息正则化器实现更好的长距离依赖。预印本：arXiv:1905.11978 。

Chang, Y.-Y.; Sun, F.-Y.; Wu, Y.-H.; and Lin, S.-D. 2018. A Memory-Network Based Solution for Multivariate Time-Series Forecasting. arXiv:1809.02105 .

张（Chang），Y.-Y.；孙（Sun），F.-Y.；吴（Wu），Y.-H.；林（Lin），S.-D. 2018年。一种基于记忆网络的多变量时间序列预测解决方案。预印本：arXiv:1809.02105 。

Child, R.; Gray, S.; Radford, A.; and Sutskever, I. 2019. Generating Long Sequences with Sparse Transformers. arXiv:1904.10509 .

蔡尔德（Child），R.；格雷（Gray），S.；拉德福德（Radford），A.；苏茨克维（Sutskever），I. 2019年。使用稀疏变压器生成长序列。预印本：arXiv:1904.10509 。

Cho, K.; van Merrienboer, B.; Bahdanau, D.; and Bengio, Y. 2014. On the Properties of Neural Machine Translation: Encoder-Decoder Approaches. In Proceedings of SSST@EMNLP 2014, 103-111.

赵（Cho），K.；范·梅里恩博尔（van Merrienboer），B.；巴达诺（Bahdanau），D.；本吉奥（Bengio），Y. 2014年。关于神经机器翻译的特性：编码器 - 解码器方法。收录于《2014年自然语言处理经验方法会议语义句法结构研讨会论文集》，第103 - 111页。

Chung, J.; Gulcehre, C.; Cho, K.; and Bengio, Y. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv:1412.3555 .

钟（Chung），J.；居尔塞赫尔（Gulcehre），C.；赵（Cho），K.；本吉奥（Bengio），Y. 2014年。门控循环神经网络在序列建模上的实证评估。预印本：arXiv:1412.3555 。

Clevert, D.; Unterthiner, T.; and Hochreiter, S. 2016. Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). In ICLR 2016.

克莱弗特（Clevert, D.）；昂特希纳（Unterthiner, T.）；霍赫赖特（Hochreiter, S.）。2016年。通过指数线性单元（ELUs）实现快速准确的深度网络学习。发表于2016年国际学习表征会议（ICLR 2016）。

Dai, Z.; Yang, Z.; Yang, Y.; Carbonell, J.; Le, Q. V.; and Salakhutdinov, R. 2019. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv:1901.02860 .

戴（Dai, Z.）；杨（Yang, Z.）；杨（Yang, Y.）；卡博内尔（Carbonell, J.）；乐（Le, Q. V.）；萨拉胡季诺夫（Salakhutdinov, R.）。2019年。Transformer-xl：超越固定长度上下文的注意力语言模型。预印本编号：arXiv:1901.02860 。

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv:1810.04805 .

德夫林（Devlin, J.）；张（Chang, M.-W.）；李（Lee, K.）；图塔诺娃（Toutanova, K.）。2018年。BERT：用于语言理解的深度双向Transformer预训练。预印本编号：arXiv:1810.04805 。

Dufresne, D. 2008. Sums of lognormals. In Actuarial Research Conference, 1-6.

迪弗雷纳（Dufresne, D.）。2008年。对数正态分布的和。发表于精算研究会议，第1 - 6页。

Flunkert, V.; Salinas, D.; and Gasthaus, J. 2017. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. arXiv:1704.04110 .

弗伦克特（Flunkert, V.）；萨利纳斯（Salinas, D.）；加施豪斯（Gasthaus, J.）。2017年。深度自回归模型（DeepAR）：基于自回归循环网络的概率预测。预印本编号：arXiv:1704.04110 。

Gupta, A.; and Rush, A. M. 2017. Dilated convolutions for modeling long-distance genomic dependencies. arXiv:1710.01278 .

古普塔（Gupta, A.）；拉什（Rush, A. M.）。2017年。用于建模长距离基因组依赖关系的扩张卷积。预印本编号：arXiv:1710.01278 。

Hcine, M. B.; and Bouallegue, R. 2015. On the approximation of the sum of lognormals by a log skew normal distribution. arXiv preprint arXiv:1502.03619 .

赫辛（Hcine, M. B.）；布阿勒盖（Bouallegue, R.）。2015年。用对数偏态正态分布近似对数正态分布之和。预印本编号：arXiv:1502.03619 。

Hochreiter, S.; and Schmidhuber, J. 1997. Long short-term memory. Neural computation 9(8): 1735-1780.

霍赫赖特（Hochreiter, S.）；施密德胡伯（Schmidhuber, J.）。1997年。长短期记忆网络。《神经计算》9(8)：1735 - 1780。

Kitaev, N.; Kaiser, L.; and Levskaya, A. 2019. Reformer: The Efficient Transformer. In ICLR.

基塔耶夫（Kitaev, N.）；凯泽（Kaiser, L.）；列夫斯卡娅（Levskaya, A.）。2019年。改革者模型（Reformer）：高效的Transformer。发表于国际学习表征会议（ICLR）。

Lai, G.; Chang, W.-C.; Yang, Y.; and Liu, H. 2018. Modeling long-and short-term temporal patterns with deep neural networks. In ACM SIGIR 2018, 95-104. ACM.

赖（Lai, G.）；张（Chang, W.-C.）；杨（Yang, Y.）；刘（Liu, H.）。2018年。使用深度神经网络对长短期时间模式进行建模。发表于2018年美国计算机协会信息检索研究与发展会议（ACM SIGIR 2018），第95 - 104页。美国计算机协会（ACM）。

Li, S.; Jin, X.; Xuan, Y.; Zhou, X.; Chen, W.; Wang, Y.-X.; and Yan, X. 2019. Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting. arXiv:1907.00235 .

李（Li）、金（Jin）、宣（Xuan）、周（Zhou）、陈（Chen）、王（Wang）和严（Yan），2019年。增强Transformer在时间序列预测中的局部性并突破内存瓶颈。预印本arXiv:1907.00235。

Li, Y.; Yu, R.; Shahabi, C.; and Liu, Y. 2018. Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. In ICLR 2018.

李（Li）、余（Yu）、沙哈比（Shahabi）和刘（Liu），2018年。扩散卷积循环神经网络：数据驱动的交通流量预测。发表于2018年国际学习表征会议（ICLR 2018）。

Liu, Y.; Gong, C.; Yang, L.; and Chen, Y. 2019. DSTP-RNN: a dual-stage two-phase attention-based recurrent neural networks for long-term and multivariate time series prediction. CoRR abs/1904.07464.

刘（Liu）、龚（Gong）、杨（Yang）和陈（Chen），2019年。DSTP - RNN：一种基于双阶段两阶段注意力的循环神经网络，用于长期多变量时间序列预测。预印本CoRR abs/1904.07464。

Lo, C.-F. 2012. The sum and difference of two lognormal random variables. Journal of Applied Mathematics 2012.

罗（Lo），2012年。两个对数正态随机变量的和与差。《应用数学杂志》，2012年。

Luong, T.; Pham, H.; and Manning, C. D. 2015. Effective Approaches to Attention-based Neural Machine Translation. In Màrquez, L.; Callison-Burch, C.; Su, J.; Pighin, D.; and Marton, Y., eds., EMNLP, 1412-1421. The Association for Computational Linguistics. doi:10.18653/v1/d15-1166. URL https://doi.org/10.18653/v1/d15-1166.

卢昂（Luong），T.；范（Pham），H.；曼宁（Manning），C. D. 2015年。基于注意力机制的神经机器翻译的有效方法。收录于马尔克斯（Màrquez），L.；卡利森 - 伯奇（Callison - Burch），C.；苏（Su），J.；皮金（Pighin），D.；马顿（Marton），Y. 编，《自然语言处理经验方法会议论文集》（EMNLP），第1412 - 1421页。计算语言学协会。doi:10.18653/v1/d15 - 1166。网址：https://doi.org/10.18653/v1/d15 - 1166。

Ma, J.; Shou, Z.; Zareian, A.; Mansour, H.; Vetro, A.; and Chang, S.-F. 2019. CDSA: Cross-Dimensional Self-Attention for Multivariate, Geo-tagged Time Series Imputation. arXiv:1905.09904 .

马（Ma），J.；寿（Shou），Z.；扎雷安（Zareian），A.；曼苏尔（Mansour），H.；维特罗（Vetro），A.；张（Chang），S. - F. 2019年。CDSA：用于多变量地理标记时间序列插补的跨维度自注意力机制。预印本编号：arXiv:1905.09904。

Matsubara, Y.; Sakurai, Y.; van Panhuis, W. G.; and Falout-sos, C. 2014. FUNNEL: automatic mining of spatially coevolving epidemics. In ACM SIGKDD 2014, 105-114.

松原（Matsubara），Y.；樱井（Sakurai），Y.；范潘休斯（van Panhuis），W. G.；法洛托斯（Falout - sos），C. 2014年。漏斗算法（FUNNEL）：空间共演化流行病的自动挖掘。收录于《第20届ACM SIGKDD国际知识发现与数据挖掘会议论文集》（ACM SIGKDD 2014），第105 - 114页。

Mukherjee, S.; Shankar, D.; Ghosh, A.; Tathawadekar, N.; Kompalli, P.; Sarawagi, S.; and Chaudhury, K. 2018. Ar-mdn: Associative and recurrent mixture density networks for eretail demand forecasting. arXiv:1803.03800 .

穆克吉（Mukherjee, S.）；尚卡尔（Shankar, D.）；戈什（Ghosh, A.）；塔塔瓦德卡尔（Tathawadekar, N.）；孔帕利（Kompalli, P.）；萨拉瓦吉（Sarawagi, S.）；乔杜里（Chaudhury, K.）。2018年。Ar - MDN：用于电商零售需求预测的关联和循环混合密度网络。预印本编号：arXiv:1803.03800 。

Papadimitriou, S.; and Yu, P. 2006. Optimal multi-scale patterns in time series streams. In ACM SIGMOD 2006, 647- 658. ACM.

帕帕迪米特里乌（Papadimitriou, S.）；于（Yu, P.）。2006年。时间序列流中的最优多尺度模式。收录于《2006年美国计算机协会管理数据会议论文集》，第647 - 658页。美国计算机协会。

Qin, Y.; Song, D.; Chen, H.; Cheng, W.; Jiang, G.; and Cottrell, G. W. 2017. A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction. In IJCAI 2017, 2627-2633.

秦（Qin, Y.）；宋（Song, D.）；陈（Chen, H.）；程（Cheng, W.）；江（Jiang, G.）；科特雷尔（Cottrell, G. W.）。2017年。基于双阶段注意力机制的循环神经网络用于时间序列预测。收录于《第26届国际人工智能联合会议论文集》，第2627 - 2633页。

Qiu, J.; Ma, H.; Levy, O.; Yih, S. W.-t.; Wang, S.; and Tang, J. 2019. Blockwise Self-Attention for Long Document Understanding. arXiv:1911.02972 .

邱（Qiu, J.）；马（Ma, H.）；利维（Levy, O.）；易（Yih, S. W.-t.）；王（Wang, S.）；唐（Tang, J.）。2019年。用于长文档理解的分块自注意力机制。预印本编号：arXiv:1911.02972 。

Rae, J. W.; Potapenko, A.; Jayakumar, S. M.; and Lillicrap, T. P. 2019. Compressive transformers for long-range sequence modelling. arXiv:1911.05507 .

雷伊（Rae, J. W.）；波塔彭科（Potapenko, A.）；贾亚库马尔（Jayakumar, S. M.）；利利克拉普（Lillicrap, T. P.）。2019年。用于长序列建模的压缩变压器。预印本 arXiv:1911.05507 。

Ray, W. 1990. Time series: theory and methods. Journal of the Royal Statistical Society: Series A (Statistics in Society) 153(3): 400-400.

雷（Ray, W.）。1990年。时间序列：理论与方法。《皇家统计学会杂志：A辑（社会统计学）》153(3): 400 - 400。

Romeo, M.; Da Costa, V.; and Bardou, F. 2003. Broad distribution effects in sums of lognormal random variables. The European Physical Journal B-Condensed Matter and Complex Systems 32(4): 513-525.

罗密欧（Romeo, M.）；达科斯塔（Da Costa, V.）；巴杜（Bardou, F.）。2003年。对数正态随机变量之和的广泛分布效应。《欧洲物理杂志B - 凝聚态物质与复杂系统》32(4): 513 - 525。

Schuster, M.; and Paliwal, K. K. 1997. Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing 45(11): 2673-2681.

舒斯特（Schuster, M.）；帕利瓦尔（Paliwal, K. K.）。1997年。双向循环神经网络。《电气与电子工程师协会信号处理汇刊》45(11): 2673 - 2681。

Seeger, M.; Rangapuram, S.; Wang, Y.; Salinas, D.; Gasthaus, J.; Januschowski, T.; and Flunkert, V. 2017. Approximate bayesian inference in linear state space models for intermittent demand forecasting at scale. arXiv:1709.07638 .

西格（Seeger, M.）；兰加普拉姆（Rangapuram, S.）；王（Wang, Y.）；萨利纳斯（Salinas, D.）；加斯豪斯（Gasthaus, J.）；亚努绍夫斯基（Januschowski, T.）；弗伦克特（Flunkert, V.）。2017年。用于大规模间歇性需求预测的线性状态空间模型中的近似贝叶斯推断。预印本：arXiv:1709.07638 。

Seeger, M. W.; Salinas, D.; and Flunkert, V. 2016. Bayesian intermittent demand forecasting for large inventories. In NIPS, 4646-4654.

西格（Seeger, M. W.）；萨利纳斯（Salinas, D.）；弗伦克特（Flunkert, V.）。2016年。大型库存的贝叶斯间歇性需求预测。载于《神经信息处理系统大会论文集》（NIPS），第4646 - 4654页。

Song, H.; Rajan, D.; Thiagarajan, J. J.; and Spanias, A. 2018. Attend and diagnose: Clinical time series analysis using attention models. In ${AAAI2018}$ .

宋（Song, H.）；拉詹（Rajan, D.）；蒂亚加拉扬（Thiagarajan, J. J.）；斯帕尼亚斯（Spanias, A.）。2018年。关注与诊断：使用注意力模型进行临床时间序列分析。载于${AAAI2018}$ 。

Stoller, D.; Tian, M.; Ewert, S.; and Dixon, S. 2019. Seq-U-Net: A One-Dimensional Causal U-Net for Efficient Sequence Modelling. arXiv:1911.06393 .

斯托勒（Stoller, D.）；田（Tian, M.）；埃沃特（Ewert, S.）；迪克森（Dixon, S.）。2019年。序列U型网络（Seq - U - Net）：用于高效序列建模的一维因果U型网络。预印本：arXiv:1911.06393 。

Sutskever, I.; Vinyals, O.; and Le, Q. V. 2014. Sequence to sequence learning with neural networks. In NIPS, 3104- 3112.

苏茨克维（Sutskever, I.）；维尼亚尔斯（Vinyals, O.）；乐（Le, Q. V.）。2014年。利用神经网络进行序列到序列学习。见《神经信息处理系统大会论文集》（NIPS），第3104 - 3112页。

Taylor, S. J.; and Letham, B. 2018. Forecasting at scale. The American Statistician 72(1): 37-45.

泰勒（Taylor, S. J.）；莱瑟姆（Letham, B.）。2018年。大规模预测。《美国统计学家》（The American Statistician）72(1): 37 - 45。

Trinh, T. H.; Dai, A. M.; Luong, M.-T.; and Le, Q. V. 2018. Learning longer-term dependencies in rnns with auxiliary losses. arXiv preprint arXiv:1803.00144 .

陈（Trinh, T. H.）；戴（Dai, A. M.）；卢昂（Luong, M.-T.）；乐（Le, Q. V.）。2018年。通过辅助损失学习循环神经网络（RNN）中的长期依赖关系。预印本arXiv:1803.00144 。

Tsai, Y.-H. H.; Bai, S.; Yamada, M.; Morency, L.-P.; and Salakhutdinov, R. 2019. Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel. In ${ACL2019},{4335} - {4344}$ .

蔡（Tsai, Y.-H. H.）；白（Bai, S.）；山田（Yamada, M.）；莫伦西（Morency, L.-P.）；萨拉胡季诺夫（Salakhutdinov, R.）。2019年。Transformer剖析：从核函数视角统一理解Transformer的注意力机制。见${ACL2019},{4335} - {4344}$ 。

Vargasguzman, J. A. 2005. Change of Support of Transformations: Conservation of Lognormality Revisited. Mathematical Geosciences 37(6): 551-567.

巴尔加斯·古兹曼（Vargasguzman, J. A.）。2005年。变换支撑集的改变：对数正态性守恒的再探讨。《数学地球科学》（Mathematical Geosciences）37(6): 551 - 567。

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. In NIPS, 5998-6008.

瓦斯瓦尼（Vaswani, A.）；沙泽尔（Shazeer, N.）；帕尔马（Parmar, N.）；乌兹科雷特（Uszkoreit, J.）；琼斯（Jones, L.）；戈麦斯（Gomez, A. N.）；凯泽（Kaiser, Ł.）；波洛苏金（Polosukhin, I.）。2017年。注意力就是你所需要的一切。见《神经信息处理系统大会论文集》（NIPS），第5998 - 6008页。

Wang, S.; Li, B.; Khabsa, M.; Fang, H.; and Ma, H. 2020. Linformer: Self-Attention with Linear Complexity. arXiv:2006.04768 .

王（Wang, S.）；李（Li, B.）；哈布萨（Khabsa, M.）；方（Fang, H.）；马（Ma, H.）。2020年。线性变换器（Linformer）：具有线性复杂度的自注意力机制。预印本arXiv:2006.04768 。

Weiner, J.; and Solbrig, O. T. 1984. The meaning and measurement of size hierarchies in plant populations. Oecologia 61(3): 334-336.

韦纳（Weiner, J.）；索尔布里格（Solbrig, O. T.）。1984年。植物种群大小等级的含义与测量。《生态学》（Oecologia）61(3): 334 - 336。

Wen, R.; Torkkola, K.; Narayanaswamy, B.; and Madeka, D. 2017. A multi-horizon quantile recurrent forecaster. arXiv:1711.11053 .

温（Wen, R.）；托尔科拉（Torkkola, K.）；纳拉亚纳斯瓦米（Narayanaswamy, B.）；马德卡（Madeka, D.）。2017年。多水平分位数循环预测器。预印本arXiv:1711.11053 。

Yu, F.; Koltun, V.; and Funkhouser, T. 2017. Dilated residual networks. In ${CVPR},{472} - {480}$ .

余（Yu, F.）；科尔图恩（Koltun, V.）；芬克豪泽（Funkhouser, T.）。2017年。扩张残差网络。见${CVPR},{472} - {480}$ 。

Yu, R.; Zheng, S.; Anandkumar, A.; and Yue, Y. 2017. Long-term forecasting using tensor-train rnns. arXiv:1711.00073

余（Yu）、郑（Zheng）、阿南德库马尔（Anandkumar）和岳（Yue），2017年。使用张量列车循环神经网络进行长期预测。预印本编号：arXiv:1711.00073

Zhu, Y.; and Shasha, D. E. 2002. StatStream: Statistical Monitoring of Thousands of Data Streams in Real Time. In VLDB 2002, 358-369.

朱（Zhu）和沙莎（Shasha），2002年。StatStream：实时对数千个数据流进行统计监测。收录于《2002年国际超大型数据库会议论文集》，第358 - 369页。

Zilly, J. G.; Srivastava, R. K.; Koutník, J.; and Schmidhuber, J. 2017. Recurrent highway networks. In ICML, 4189-4198.

齐利（Zilly）、斯里瓦斯塔瓦（Srivastava）、库特尼克（Koutník）和施密德胡伯（Schmidhuber），2017年。循环高速公路网络。收录于《国际机器学习会议论文集》，第4189 - 4198页。