# FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting 


#### Abstract

Although Transformer-based methods have significantly improved state-of-the-art results for long-term series forecasting, they are not only computationally expensive but more importantly, are unable to capture the global view of time series (e.g. overall trend). To address these problems, we propose to combine Transformer with the seasonal-trend decomposition method, in which the decomposition method captures the global profile of time series while Transformers capture more detailed structures. To further enhance the performance of Transformer for longterm prediction, we exploit the fact that most time series tend to have a sparse representation in well-known basis such as Fourier transform, and develop a frequency enhanced Transformer. Besides being more effective, the proposed method, termed as Frequency Enhanced Decomposed Transformer (FEDformer), is more efficient than standard Transformer with a linear complexity to the sequence length. Our empirical studies with six benchmark datasets show that compared with state-of-the-art methods, FEDformer can reduce prediction error by $14.8 \%$ and $22.6 \%$ for multivariate and univariate time series, respectively. Code is publicly available at https://github.com/MAZiqing/FEDformer.


## 1. Introduction

Long-term time series forecasting is a long-standing challenge in various applications (e.g., energy, weather, traffic, economics). Despite the impressive results achieved by RNN-type methods (Rangapuram et al., 2018; Flunkert et al., 2017), they often suffer from the problem of gradient vanishing or exploding (Pascanu et al., 2013), significantly limiting their performance. Following the recent success in NLP and CV community (Vaswani et al., 2017; Devlin et al., 2019; Dosovitskiy et al., 2021; Rao et al., 2021), Transformer (Vaswani et al., 2017) has been introduced to capture long-term dependencies in time series forecasting and shows promising results (Zhou et al., 2021; Wu et al., 2021). Since high computational complexity and memory requirement make it difficult for Transformer to be applied to long sequence modeling, numerous studies are devoted to reduce the computational cost of Transformer (Li et al., 2019; Kitaev et al., 2020; Zhou et al., 2021; Wang et al., 2020; Xiong et al., 2021; Ma et al., 2021). A through overview of this line of works can be found in Appendix A.

---

Despite the progress made by Transformer-based methods for time series forecasting, they tend to fail in capturing the overall characteristics/distribution of time series in some cases. In Figure 1, we compare the time series of ground truth with that predicted by the vanilla Transformer method (Vaswani et al., 2017) in a real-world ETTm1 dataset (Zhou et al., 2021). It is clear that the predicted time series shared a different distribution from that of ground truth. The discrepancy between ground truth and prediction could be explained by the point-wise attention and prediction in Transformer. Since prediction for each timestep is made individually and independently, it is likely that the model fails to maintain the global property and statistics of time series as a whole. To address this problem, we exploit two ideas in this work. The first idea is to incorporate a seasonal-trend decomposition approach (Cleveland et al., 1990; Wen et al., 2019), which is widely used in time series analysis, into the Transformerbased method. Although this idea has been exploited before (Oreshkin et al., 2019; Wu et al., 2021), we present a special design of network that is effective in bringing the distribution of prediction close to that of ground truth, according to Kologrov-Smirnov distribution test. Our second idea is to combine Fourier analysis with the Transformerbased method. Instead of applying Transformer to the time domain, we apply it to the frequency domain which helps Transformer better capture global properties of time series. Combining both ideas, we propose a Frequency Enhanced Decomposition Transformer, or, FEDformer for short, for long-term time series forecasting.

---

One critical question with FEDformer is which subset of frequency components should be used by Fourier analysis to represent time series. A common wisdom is to keep lowfrequency components and throw away the high-frequency ones. This may not be appropriate for time series forecasting as some of trend changes in time series are related to important events, and this piece of information could be lost if we simply remove all high-frequency components. We address this problem by effectively exploiting the fact that time series tend to have (unknown) sparse representations on a basis like Fourier basis. According to our theoretical analysis, a randomly selected subset of frequency components, including both low and high ones, will give a better representation for time series, which is further verified by extensive empirical studies. Besides being more effective for long term forecasting, combining Transformer with frequency analysis allows us to reduce the computational cost of Transformer from quadratic to linear complexity. We note that this is different from previous efforts on speeding up Transformer, which often leads to a performance drop.

---

In short, we summarize the key contributions of this work as follows:

1. We propose a frequency enhanced decomposed Transformer architecture with mixture of experts for seasonal-trend decomposition in order to better capture global properties of time series.
2. We propose Fourier enhanced blocks and Wavelet enhanced blocks in the Transformer structure that allows us to capture important structures in time series through frequency domain mapping. They serve as substitutions for both self-attention and crossattention blocks.
3. By randomly selecting a fixed number of Fourier components, the proposed model achieves linear computational complexity and memory cost. The effectiveness of this selection method is verified both theoretically and empirically.
4. We conduct extensive experiments over 6 benchmark datasets across multiple domains (energy, traffic, economics, weather and disease). Our empirical studies show that the proposed model improves the performance of state-of-the-art methods by $14.8 \%$ and $22.6 \%$ for multivariate and univariate forecasting, respectively.

---

Figure 1. Different distribution between ground truth and forecasting output from vanilla Transformer in a real-world ETTm1 dataset. Left: frequency mode and trend shift. Right: trend shift.

## 2. Compact Representation of Time Series in Frequency Domain

It is well-known that time series data can be modeled from the time domain and frequency domain. One key contribution of our work which separates from other long-term forecasting algorithms is the frequency-domain operation with a neural network. As Fourier analysis is a common tool to dive into the frequency domain, while how to appropriately represent the information in time series using Fourier analysis is critical. Simply keeping all the frequency components may result in inferior representations since many high-frequency changes in time series are due to noisy inputs. On the other hand, only keeping the low-frequency components may also be inappropriate for series forecasting as some trend changes in time series represent important events. Instead, keeping a compact representation of time series using a small number of selected Fourier components will lead to efficient computation of transformer, which is crucial for modelling long sequences. We propose to represent time series by randomly selecting a constant number of Fourier components, including both highfrequency and low-frequency. Below, an analysis that justifies the random selection is presented theoretically. Empirical verification can be found in the experimental session.

---

Consider we have $m$ time series, denoted as $X_{1}(t), \ldots, X_{m}(t)$. By applying Fourier transform to each time series, we turn each $X_{i}(t)$ into a vector $a_{i}=\left(a_{i, 1}, \ldots, a_{i, d}\right)^{\top} \in \mathbb{R}^{d}$. By putting all the Fourier transform vectors into a matrix, we have $A=\left(a_{1}, a_{2}, \ldots, a_{m}\right)^{\top} \in \mathbb{R}^{m \times d}$, with each row corresponding to a different time series and each column corresponding to a different Fourier component. Although using all the Fourier components allows us to best preserve the history information in the time series, it may potentially lead to overfitting of the history data and consequentially a poor prediction of future signals. Hence, we need to select a subset of Fourier components, that on the one hand should be small enough to avoid the overfitting problem and on the other hand, should be able to preserve most of the history information. Here, we propose to select $s$ components from the $d$ Fourier components $(s<d)$ uniformly at random. More specifically, we denote by $i_{1}<i_{2}<\ldots<i_{s}$ the randomly selected components. We construct matrix $S \in\{0,1\}^{s \times d}$, with $S_{i, k}=1$ if $i=i_{k}$ and $S_{i, k}=0$ otherwise. Then, our representation of multivariate time series becomes $A^{\prime}=A S^{\top} \in \mathbb{R}^{m \times s}$. Below, we will show that, although the Fourier basis are randomly selected, under a mild condition, $A^{\prime}$ is able to preserve most of the information from $A$.

---

In order to measure how well $A^{\prime}$ is able to preserve information from $A$, we project each column vector of $A$ into the subspace spanned by the column vectors in $A^{\prime}$. We denote by $P_{A^{\prime}}(A)$ the resulting matrix after the projection, where $P_{A^{\prime}}(\cdot)$ represents the projection operator. If $A^{\prime}$ preserves a large portion of information from $A$, we would expect a small error between $A$ and $P_{A^{\prime}}(A)$, i.e. $\left|A-P_{A^{\prime}}(A)\right|$. Let $A_{k}$ represent the approximation of $A$ by its first $k$ largest single value decomposition. The theorem below shows that $\left|A-P_{A^{\prime}}(A)\right|$ is close to $\left|A-A_{k}\right|$ if the number of randomly sampled Fourier components $s$ is on the order of $k^{2}$.

---

Theorem 1. Assume that $\mu(A)$, the coherence measure of matrix $A$, is $\Omega(k / n)$. Then, with a high probability, we have

$$
\left|A-P_{A^{\prime}}(A)\right| \leq(1+\epsilon)\left|A-A_{k}\right|
$$

if $s=O\left(k^{2} / \epsilon^{2}\right)$. The detailed analysis can be found in Appendix C.

---

For real-world multivariate times series, the corresponding matrix $A$ from Fourier transform often exhibit low rank property, since those univaraite variables in multivariate times series depend not only on its past values but also has dependency on each other, as well as share similar frequency components. Therefore, as indicated by the Theorem 1, randomly selecting a subset of Fourier components allows us to appropriately represent the information in Fourier matrix $A$.

---

Similarly, wavelet orthogonal polynomials, such as Legendre Polynomials, obey restricted isometry property (RIP) and can be used for capture information in time series as well. Compared to Fourier basis, wavelet based representation is more effective in capturing local structures in time series and thus can be more effective for some forecasting tasks. We defer the discussion of wavelet based representation in Appendix B. In the next section, we will present the design of frequency enhanced decomposed Transformer architecture that incorporate the Fourier transform into transformer.

## 3. Model Structure

In this section, we will introduce (1) the overall structure of FEDformer, as shown in Figure 2, (2) two subversion structures for signal process: one uses Fourier basis and the other uses Wavelet basis, (3) the mixture of experts mechanism for seasonal-trend decomposition, and (4) the complexity analysis of the proposed model.

- Figure 2. FEDformer Structure. The FEDformer consists of $N$ encoders and $M$ decoders. The Frequency Enhanced Block (FEB, green blocks) and Frequency Enhanced Attention (FEA, red blocks) are used to perform representation learning in frequency domain. Either FEB or FEA has two subversions (FEB-f \& FEB-w or FEA-f \& FEA-w), where '-f' means using Fourier basis and '-w' means using Wavelet basis. The Mixture Of Expert Decomposition Blocks (MOEDecomp, yellow blocks) are used to extract seasonal-trend patterns from the input data.

### 3.1. FEDformer Framework

Preliminary. Long-term time series forecasting is a sequence to sequence problem. We denote the input length as $I$ and output length as $O$. We denote $D$ as the hidden states of the series. The input of the encoder is a $I \times D$ matrix and the decoder has $(I / 2+O) \times D$ input.

---

FEDformer Structure Inspired. by the seasonal-trend decomposition and distribution analysis as discussed in Section 1, we renovate Transformer as a deep decomposition architecture as shown in Figure 2, including Frequency Enhanced Block (FEB), Frequency Enhanced Attention (FEA) connecting encoder and decoder, and the Mixture Of Experts Decomposition block (MOEDecomp). The detailed description of FEB, FEA, and MOEDecomp blocks will be given in the following Section 3.2, 3.3, and 3.4 respectively.

---

The encoder adopts a multilayer structure as: $\mathcal{X}_{\text {en }}^{l}=$ $\operatorname{Encoder}\left(\mathcal{X}_{\text {en }}^{l-1}\right)$, where $l \in\{1, \cdots, N\}$ denotes the output of $l$-th encoder layer and $\mathcal{X}_{\text {en }}^{0} \in \mathbb{R}^{I \times D}$ is the embedded historical series. The Encoder $(\cdot)$ is formalized as

$$
\begin{align*}
\mathcal{S}_{\mathrm{en}}^{l, 1},- & =\operatorname{MOEDecomp}\left(\operatorname{FEB}\left(\mathcal{X}_{\mathrm{en}}^{l-1}\right)+\mathcal{X}_{\mathrm{en}}^{l-1}\right) \\
\mathcal{S}_{\mathrm{en}}^{l, 2},- & =\operatorname{MOEDecomp}\left(\text { FeedForward }\left(\mathcal{S}_{\mathrm{en}}^{l, 1}\right)+\mathcal{S}_{\mathrm{en}}^{l, 1}\right),  \tag{1}\\
\mathcal{X}_{\mathrm{en}}^{l} & =\mathcal{S}_{\mathrm{en}}^{l, 2}
\end{align*}
$$

---

where $\mathcal{S}_{\mathrm{en}}^{l, i}, i \in\{1,2\}$ represents the seasonal component after the $i$-th decomposition block in the $l$-th layer respectively. For FEB module, it has two different versions (FEB-f \& FEB-w) which are implemented through Discrete Fourier transform (DFT) and Discrete Wavelet transform (DWT) mechanism respectively and can seamlessly replace the self-attention block.

---

The decoder also adopts a multilayer structure as: $\mathcal{X}_{\mathrm{de}}^{l}, \mathcal{T}_{\mathrm{de}}^{l}=\operatorname{Decoder}\left(\mathcal{X}_{\mathrm{de}}^{l-1}, \mathcal{T}_{\mathrm{de}}^{l-1}\right)$, where $l \in\{1, \cdots, M\}$ denotes the output of $l$-th decoder layer. The $\operatorname{Decoder}(\cdot)$ is formalized as

$$
\begin{align*}
\mathcal{S}_{\mathrm{de}}^{l, 1}, \mathcal{T}_{\mathrm{de}}^{l, 1} & =\operatorname{MOEDecomp}\left(\operatorname{FEB}\left(\mathcal{X}_{\mathrm{de}}^{l-1}\right)+\mathcal{X}_{\mathrm{de}}^{l-1}\right), \\
\mathcal{S}_{\mathrm{de}}^{l, 2}, \mathcal{T}_{\mathrm{de}}^{l, 2} & =\operatorname{MOEDecomp}\left(\operatorname{FEA}\left(\mathcal{S}_{\mathrm{de}}^{l, 1}, \mathcal{X}_{\mathrm{en}}^{N}\right)+\mathcal{S}_{\mathrm{de}}^{l, 1}\right), \\
\mathcal{S}_{\mathrm{de}}^{l, 3}, \mathcal{T}_{\mathrm{de}}^{l, 3} & =\operatorname{MOEDecomp}\left(\text { FeedForward }\left(\mathcal{S}_{\mathrm{de}}^{l, 2}\right)+\mathcal{S}_{\mathrm{de}}^{l, 2}\right), \\
\mathcal{X}_{\mathrm{de}}^{l} & =\mathcal{S}_{\mathrm{de}}^{l, 3}, \\
\mathcal{T}_{\mathrm{de}}^{l} & =\mathcal{T}_{\mathrm{de}}^{l-1}+\mathcal{W}_{l, 1} \cdot \mathcal{T}_{\mathrm{de}}^{l, 1}+\mathcal{W}_{l, 2} \cdot \mathcal{T}_{\mathrm{de}}^{l, 2}+\mathcal{W}_{l, 3} \cdot \mathcal{T}_{\mathrm{de}}^{l, 3}, \tag{2}
\end{align*}
$$

---

where $\mathcal{S}_{\mathrm{de}}^{l, i}, \mathcal{T}_{\mathrm{de}}^{l, i}, i \in\{1,2,3\}$ represent the seasonal and trend component after the $i$-th decomposition block in the $l$ th layer respectively. $\mathcal{W}_{l, i}, i \in\{1,2,3\}$ represents the projector for the $i$-th extracted trend $\mathcal{T}_{\text {de }}^{l, i}$. Similar to FEB, FEA has two different versions (FEA-f \& FEA-w) which are implemented through DFT and DWT projection respectively with attention design, and can replace the cross-attention block. The detailed description of FEA $(\cdot)$ will be given in the following Section 3.3.

---

The final prediction is the sum of the two refined decomposed components as $\mathcal{W}_{\mathcal{S}} \cdot \mathcal{X}_{\mathrm{de}}^{M}+\mathcal{T}_{\mathrm{de}}^{M}$, where $\mathcal{W}_{\mathcal{S}}$ is to project the deep transformed seasonal component $\mathcal{X}_{\text {de }}^{M}$ to the target dimension.

### 3.2. Fourier Enhanced Structure

Discrete Fourier Transform (DFT) The proposed Fourier Enhanced Structures use discrete Fourier transform (DFT). Let $\mathcal{F}$ denotes the Fourier transform and $\mathcal{F}^{-1}$ denotes the inverse Fourier transform. Given a sequence of real numbers $x_{n}$ in time domain, where $n=1,2 \ldots N$. DFT is defined as $X_{l}=\sum_{n=0}^{N-1} x_{n} e^{-i \omega l n}$, where $i$ is the imaginary unit and $X_{l}, l=1,2 \ldots L$ is a sequence of complex numbers in the frequency domain. Similarly, the inverse DFT is defined as $x_{n}=\sum_{l=0}^{L-1} X_{l} e^{i \omega l n}$. The complexity of DFT is $O\left(N^{2}\right)$. With fast Fourier transform (FFT), the computation complexity can be reduced to $O(N \log N)$. Here a random subset of the Fourier basis is used and the scale of the subset is bounded by a scalar. When we choose the mode index before DFT and reverse DFT operations, the computation complexity can be further reduced to $O(N)$.

---

#### Frequency Enhanced Block with Fourier Transform

 (FEB-f) The FEB-f is used in both encoder and decoder as shown in Figure 2. The input ( $\boldsymbol{x} \in \mathbb{R}^{N \times D}$ ) of the FEB-f block is first linearly projected with $\boldsymbol{w} \in \mathbb{R}^{D \times D}$, so $\boldsymbol{q}=\boldsymbol{x} \cdot \boldsymbol{w}$. Then $\boldsymbol{q}$ is converted from the time domain to the frequency domain. The Fourier transform of $\boldsymbol{q}$ is denoted as $\boldsymbol{Q} \in \mathbb{C}^{N \times D}$. In frequency domain, only the randomly selected $M$ modes are kept so we use a select operator as

$$
\begin{equation*}
\tilde{\boldsymbol{Q}}=\operatorname{Select}(\boldsymbol{Q})=\operatorname{Select}(\mathcal{F}(\boldsymbol{q})), \tag{3}
\end{equation*}
$$

where $\tilde{\boldsymbol{Q}} \in \mathbb{C}^{M \times D}$ and $M \ll N$. Then, the FEB-f is defined as

$$
\begin{equation*}
\operatorname{FEB}-\mathrm{f}(\boldsymbol{q})=\mathcal{F}^{-1}(\operatorname{Padding}(\tilde{\boldsymbol{Q}} \odot \boldsymbol{R})) \tag{4}
\end{equation*}
$$

where $\boldsymbol{R} \in \mathbb{C}^{D \times D \times M}$ is a parameterized kernel initialized randomly. Let $\boldsymbol{Y}=\boldsymbol{Q} \odot \boldsymbol{C}$, with $\boldsymbol{Y} \in \mathbb{C}^{M \times D}$. The production operator $\odot$ is defined as: $Y_{m, d_{o}}=\sum_{d_{i}=0}^{D} Q_{m, d_{i}}$. $R_{d_{i}, d_{o}, m}$, where $d_{i}=1,2 \ldots D$ is the input channel and $d_{o}=1,2 \ldots D$ is the output channel. The result of $\boldsymbol{Q} \odot \boldsymbol{R}$ is then zero-padded to $\mathbb{C}^{N \times D}$ before performing inverse Fourier transform back to the time domain. The structure is shown in Figure 3.

---

Frequency Enhanced Attention with Fourier Transform (FEA-f). We use the expression of the canonical transformer. The input: queries, keys, values are denoted as $\boldsymbol{q} \in \mathbb{R}^{L \times D}, \boldsymbol{k} \in \mathbb{R}^{L \times D}, \boldsymbol{v} \in \mathbb{R}^{L \times D}$. In cross-attention, the queries come from the decoder and can be obtained by $\boldsymbol{q}=\boldsymbol{x}_{e n} \cdot \boldsymbol{w}_{q}$, where $\boldsymbol{w}_{q} \in \mathbb{R}^{D \times D}$. The keys and values are from the encoder and can be obtained by $\boldsymbol{k}=\boldsymbol{x}_{d e} \cdot \boldsymbol{w}_{k}$ and $\boldsymbol{v}=\boldsymbol{x}_{d e} \cdot \boldsymbol{w}_{v}$, where $\boldsymbol{w}_{k}, \boldsymbol{w}_{v} \in \mathbb{R}^{D \times D}$. Formally, the canonical attention can be written as

$$
\begin{equation*}
\operatorname{Atten}(\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v})=\operatorname{Softmax}\left(\frac{\boldsymbol{q} \boldsymbol{k}^{\top}}{\sqrt{d_{q}}}\right) \boldsymbol{v} \tag{5}
\end{equation*}
$$

---

In FEA-f, we convert the queries, keys, and values with Fourier Transform and perform a similar attention mechanism in the frequency domain, by randomly selecting M modes. We denote the selected version after Fourier Transform as $\tilde{\boldsymbol{Q}} \in \mathbb{C}^{M \times D}, \tilde{\boldsymbol{K}} \in \mathbb{C}^{M \times D}, \tilde{\boldsymbol{V}} \in \mathbb{C}^{M \times D}$. The FEA-f is defined as

$$
\begin{align*}
& \tilde{\boldsymbol{Q}}=\operatorname{Select}(\mathcal{F}(\boldsymbol{q})) \\
& \tilde{\boldsymbol{K}}=\operatorname{Select}(\mathcal{F}(\boldsymbol{k}))  \tag{6}\\
& \tilde{\boldsymbol{V}}=\operatorname{Select}(\mathcal{F}(\boldsymbol{v})) \\
& \operatorname{FEA}-\mathrm{f}(\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v})=\mathcal{F}^{-1}\left(\operatorname{Padding}\left(\sigma\left(\tilde{\boldsymbol{Q}} \cdot \tilde{\boldsymbol{K}}^{\top}\right) \cdot \tilde{\boldsymbol{V}}\right)\right), \tag{7}
\end{align*}
$$

---

where $\sigma$ is the activation function. We use softmax or tanh for activation, since their converging performance differs in different data sets. Let $\boldsymbol{Y}=\sigma\left(\tilde{\boldsymbol{Q}} \cdot \tilde{\boldsymbol{K}}^{\top}\right) \cdot \tilde{\boldsymbol{V}}$, and $\boldsymbol{Y} \in \mathbb{C}^{M \times D}$ needs to be zero-padded to $\mathbb{C}^{L \times D}$ before performing inverse Fourier transform. The FEA-f structure is shown in Figure 4.

### 3.3. Wavelet Enhanced Structure

Discrete Wavelet Transform (DWT) While the Fourier transform creates a representation of the signal in the frequency domain, the Wavelet transform creates a representation in both the frequency and time domain, allowing efficient access of localized information of the signal. The multiwavelet transform synergizes the advantages of orthogonal polynomials as well as wavelets. For a given $f(x)$, the multiwavelet coefficients at the scale $n$ can be defined as $\mathbf{s}_{l}^{n}=\left[\left\langle f, \phi_{i l}^{n}\right\rangle_{\mu_{n}}\right]_{i=0}^{k-1}, \mathbf{d}_{l}^{n}=\left[\left\langle f, \psi_{i l}^{n}\right\rangle_{\mu_{n}}\right]_{i=0}^{k-1}$, respectively, w.r.t. measure $\mu_{n}$ with $\mathbf{s}_{l}^{n}, \mathbf{d}_{l}^{n} \in \mathbb{R}^{k \times 2^{n}}$. $\phi_{i l}^{n}$ are wavelet orthonormal basis of piecewise polynomials. The decomposition/reconstruction across scales is defined as

$$
\begin{align*}
\mathbf{s}_{l}^{n} & =H^{(0)} \mathbf{s}_{2 l}^{n+1}+H^{(1)} \mathbf{s}_{2 l+1}^{n+1}, \\
\mathbf{s}_{2 l}^{n+1} & =\Sigma^{(0)}\left(H^{(0) T} \mathbf{s}_{l}^{n}+G^{(0) T} \mathbf{d}_{l}^{n}\right), \\
\mathbf{d}_{l}^{n} & =G^{(0)} \mathbf{s}_{2 l}^{n+1}+H^{(1)} \mathbf{s}_{2 l+1}^{n+1},  \tag{8}\\
\mathbf{s}_{2 l+1}^{n+1} & =\Sigma^{(1)}\left(H^{(1) T} \mathbf{s}_{l}^{n}+G^{(1) T} \mathbf{d}_{l}^{n}\right),
\end{align*}
$$

where $\left(H^{(0)}, H^{(1)}, G^{(0)}, G^{(1)}\right)$ are linear coefficients for multiwavelet decomposition filters. They are fixed matrices

---

used for wavelet decomposition. The multiwavelet representation of a signal can be obtained by the tensor product of multiscale and multiwavelet basis. Note that the basis at various scales are coupled by the tensor product, so we need to untangle it. Inspired by (Gupta et al., 2021), we adapt a non-standard wavelet representation to reduce the model complexity. For a map function $F(x)=x^{\prime}$, the map under multiwavelet domain can be written as

$$
\begin{equation*}
U_{d l}^{n}=A_{n} d_{l}^{n}+B_{n} s_{l}^{n}, \quad U_{s k l}^{n}=C_{n} d_{l}^{n}, \quad U_{s l}^{L}=\bar{F} s_{l}^{L}, \tag{9}
\end{equation*}
$$

where $\left(U_{s l}^{n}, U_{d l}^{n}, s_{l}^{n}, d_{l}^{n}\right)$ are the multiscale, multiwavelet coefficients, $L$ is the coarsest scale under recursive decomposition, and $A_{n}, B_{n}, C_{n}$ are three independent FEB-f blocks modules used for processing different signal during decomposition and reconstruction. Here $\bar{F}$ is a single-layer of perceptrons which processes the remaining coarsest signal after $L$ decomposed steps. More designed detail is described in Appendix D.

---

#### Frequency Enhanced Block with Wavelet Transform

 (FEB-w) The overall FEB-w architecture is shown in Figure 5. It differs from FEB-f in the recursive mechanism: the input is decomposed into 3 parts recursively and operates individually. For the wavelet decomposition part, we implement the fixed Legendre wavelets basis decomposition matrix. Three FEB-f modules are used to process the resulting high-frequency part, low-frequency part, and remaining part from wavelet decomposition respectively. For each cycle $L$, it produces a processed high-frequency tensor $U d(L)$, a processed low-frequency frequency tensor$U s(L)$, and the raw low-frequency tensor $X(L+1)$. This is a ladder-down approach, and the decomposition stage performs the decimation of the signal by a factor of $1 / 2$, running for a maximum of $L$ cycles, where $L<\log _{2}(M)$ for a given input sequence of size $M$. In practice, $L$ is set as a fixed argument parameter. The three sets of FEB-f blocks are shared during different decomposition cycles $L$. For the wavelet reconstruction part, we recursively build up our output tensor as well. For each cycle $L$, we combine $X(L+1)$, $U s(L)$, and $U d(L)$ produced from the decomposition part and produce $X(L)$ for the next reconstruction cycle. For each cycle, the length dimension of the signal tensor is increased by 2 times.

 - Figure 5. Top Left: Wavelet frequency enhanced block decomposition stage. Top Right: Wavelet block reconstruction stage shared by FEB-w and FEA-w. Bottom: Wavelet frequency enhanced cross attention decomposition stage.

---

Frequency Enhanced Attention with Wavelet Transform (FEA-w). FEA-w contains the decomposition stage and reconstruction stage like FEB-w. Here we keep the reconstruction stage unchanged. The only difference lies in the decomposition stage. The same decomposed matrix is used to decompose $\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v}$ signal separately, and $\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v}$ share the same sets of module to process them as well. As shown above, a frequency enhanced block with wavelet decomposition block (FEB-w) contains three FEB-f blocks for the signal process. We can view the FEB-f as a substitution of self-attention mechanism. We use a straightforward way to build the frequency enhanced cross attention with wavelet decomposition, substituting each FEB-f with a FEA-f module. Besides, another FEA-f module is added to process the coarsest remaining $q(L), k(L), v(L)$ signal.

### 3.4. Mixture of Experts for Seasonal-Trend Decomposition

Because of the commonly observed complex periodic pattern coupled with the trend component on real-world data, extracting the trend can be hard with fixed window average pooling. To overcome such a problem, we design a Mixture Of Experts Decomposition block (MOEDecomp). It contains a set of average filters with different sizes to extract multiple trend components from the input signal and a set of data-dependent weights for combining them as the final trend. Formally, we have

$$
\begin{equation*}
\mathbf{X}_{\text {trend }}=\boldsymbol{\operatorname { S o f t m a x }}(L(x)) *(F(x)), \tag{10}
\end{equation*}
$$

where $F(\cdot)$ is a set of average pooling filters and $\operatorname{Softmax}(L(x))$ is the weights for mixing these extracted trends.

### 3.5. Complexity Analysis

For FEDformer-f, the computational complexity for time and memory is $O(L)$ with a fixed number of randomly selected modes in FEB \& FEA blocks. We set modes number $M=64$ as default value. Though the complexity of full DFT transformation by FFT is $(O(L \log (L))$, our model only needs $O(L)$ cost and memory complexity with the pre-selected set of Fourier basis for quick implementation. For FEDformer-w, when we set the recursive decompose step to a fixed number $L$ and use a fixed number of randomly selected modes the same as FEDformer-f, the time complexity and memory usage are $O(L)$ as well. In practice, we choose $L=3$ and modes number $M=64$ as default value. The comparisons of the time complexity and memory usage in training and the inference steps in testing are summarized in Table 1. It can be seen that the proposed FEDformer achieves the best overall complexity among Transformer-based forecasting models.

## 4. Experiments

To evaluate the proposed FEDformer, we conduct extensive experiments on six popular real-world datasets, including energy, economics, traffic, weather, and disease. Since classic models like ARIMA and basic RNN/CNN models perform relatively inferior as shown in (Zhou et al., 2021) and (Wu et al., 2021), we mainly include four state-of-theart transformer-based models for comparison, i.e., Autoformer (Wu et al., 2021), Informer (Zhou et al., 2021), LogTrans (Li et al., 2019) and Reformer (Kitaev et al., 2020) as baseline models. Note that since Autoformer holds the best performance in all the six benchmarks, it is used as the main baseline model for comparison. More details about baseline models, datasets, and implementation are described in Appendix A.2, F.1, and F.2, respectively.

### 4.1. Main Results

For better comparison, we follow the experiment settings of Autoformer in (Wu et al., 2021) where the input length is fixed to 96 , and the prediction lengths for both training and evaluation are fixed to be $96,192,336$, and 720 , respectively.

---

Multivariate Results. For the multivariate forecasting, FEDformer achieves the best performance on all six benchmark datasets at all horizons as shown in Table 2. Compared with Autoformer, the proposed FEDformer yields an overall $\mathbf{1 4 . 8 \%}$ relative MSE reduction. It is worth noting

- Table 2. Multivariate long-term series forecasting results on six datasets with input length $I=96$ and prediction length $O \in$ $\{96,192,336,720\}$ (For ILI dataset, we use input length $I=36$ and prediction length $O \in\{24,36,48,60\}$ ). A lower MSE indicates better performance, and the best results are highlighted in bold.

---

that for some of the datasets, such as Exchange and ILI, the improvement is even more significant (over $20 \%$ ). Note that the Exchange dataset does not exhibit clear periodicity in its time series, but FEDformer can still achieve superior performance. Overall, the improvement made by FEDformer is consistent with varying horizons, implying its strength in long term forecasting. More detailed results on ETT full benchmark are provided in Appendix F.3.

---

Univariate Results. The results for univariate time series forecasting are summarized in Table 3. Compared with Autoformer, FEDformer yields an overall $\mathbf{2 2 . 6 \%}$ relative MSE reduction, and on some datasets, such as traffic and weather, the improvement can be more than $30 \%$. It again verifies that FEDformer is more effective in long-term forecasting. Note that due to the difference between Fourier and wavelet basis, FEDformer-f and FEDformer-w perform well on different datasets, making them complementary choice for long term forecasting. More detailed results on ETT full benchmark are provided in Appendix F.3.

- Table 3. Univariate long-term series forecasting results on six datasets with input length $I=96$ and prediction length $O \in$ $\{96,192,336,720\}$ (For ILI dataset, we use input length $I=36$ and prediction length $O \in\{24,36,48,60\}$ ). A lower MSE indicates better performance, and the best results are highlighted in bold. 

### 4.2. Ablation Studies

In this section, the ablation experiments are conducted, aiming at comparing the performance of frequency enhanced block and its alternatives. The current SOTA results of Autoformer which uses the autocorrelation mechanism serve as the baseline. Three ablation variants of FEDformer are tested: 1) FEDformer V1: we use FEB to substitute selfattention only; 2) FEDformer V2: we use FEA to substitute cross attention only; 3) FEDFormer V3: we use FEA to substitute both self and cross attention. The ablated versions of FEDformer-f as well as the SOTA models are compared in Table 4, and we use a bold number if the ablated version brings improvements compared with Autoformer. We omit the similar results in FEDformer-w due to space limit. It can be seen in Table 4 that FEDformer V1 brings improvement in 10/16 cases, while FEDformer V2 improves in 12/16 cases. The best performance is achieved in our FEDformer with FEB and FEA blocks which improves performance in all $16 / 16$ cases. This verifies the effectiveness of the designed FEB, FEA for substituting self and cross attention. Furthermore, experiments on ETT and Weather datasets show that the adopted MOEDecomp (mixture of experts decomposition) scheme can bring an average of $2.96 \%$ improvement compared with the single decomposition scheme. More details are provided in Appendix F.5.

- Table 4. Ablation studies: multivariate long-term series forecasting results on ETTm1 and ETTm 2 with input length $I=96$ and prediction length $O \in\{96,192,336,720\}$. Three variants of FEDformer-f are compared with baselines. The best results are highlighted in bold.


### 4.3. Mode Selection Policy

The selection of discrete Fourier basis is the key to effectively representing the signal and maintaining the model's linear complexity. As we discussed in Section 2, random Fourier mode selection is a better policy in forecasting tasks. more importantly, random policy requires no prior knowledge of the input and generalizes easily in new tasks. Here we empirically compare the random selection policy with fixed selection policy, and summarize the experimental results in Figure 6. It can be observed that the adopted random policy achieves better performance than the common fixed policy which only keeps the low frequency modes. Meanwhile, the random policy exhibits some mode saturation effect, indicating an appropriate random number of modes instead of all modes would bring better performance, which is also consistent with the theoretical analysis in Section 2.

- Figure 6. Comparison of two base-modes selection method (Fix\&Rand). Rand policy means randomly selecting a subset of modes, Fix policy means selecting the lowest frequency modes. Two policies are compared on a variety of base-modes number $M \in\{2,4,8 \ldots 256\}$ on ETT full-benchmark (h1, m1, h2, m2). 

### 4.4. Distribution Analysis of Forecasting Output

In this section, we evaluate the distribution similarity between the input sequence and forecasting output of different transformer models quantitatively. In Table 5, we applied the Kolmogrov-Smirnov test to check if the forecasting results of different models made on ETTm1 and ETTm2 are consistent with the input sequences. In particular, we test if the input sequence of fixed 96 -time steps come from the same distribution as the predicted sequence, with the null hypothesis that both sequences come from the same distribution. On both datasets, by setting the common P-value as 0.01 , various existing Transformer baseline models have much less values than 0.01 except Autoformer, which indicates their forecasting output have a higher probability to be sampled from the different distributions compared to the input sequence. In contrast, Autoformer and FEDformer have much larger P-value compared to others, which mainly contributes to their seasonal-trend decomposition mechanism. Though we get close results from ETTm 2 by both models, the proposed FEDformer has much larger P-value in ETTm1. And it's the only model whose null hypothesis can not be rejected with P-value larger than 0.01 in all cases of the two datasets, implying that the output sequence generated by FEDformer shares a more similar distribution as the input sequence than others and thus justifies the our design motivation of FEDformer as discussed in Section 1. More detailed analysis are provided in Appendix E.

- Table 5. P-values of Kolmogrov-Smirnov test of different transformer models for long-term forecasting output on ETTm1 and ETTm2 dataset. Larger value indicates the hypothesis (the input sequence and forecasting output come from the same distribution) is less likely to be rejected. The best results are highlighted.

### 4.5. Differences Compared to Autoformer baseline

Since we use the decomposed encoder-decoder overall architecture as Autoformer, we think it is critical to emphasize the differences. In Autoformer, the authors consider a nice idea to use the top-k sub-sequence correlation (autocorrelation) module instead of point-wise attention, and the Fourier method is applied to improve the efficiency for subsequence level similarity computation. In general, Autoformer can be considered as decomposing the sequence into multiple time domain sub-sequences for feature exaction. In contrast, We use frequency transform to decompose the sequence into multiple frequency domain modes to extract the feature. In particular, we do not use a selective approach in sub-sequence selection. Instead, all frequency features are computed from the whole sequence, and this global property makes our model engage better performance for long sequence.

## 5. Conclusions

This paper proposes a frequency enhanced transformer model for long-term series forecasting which achieves state-of-the-art performance and enjoys linear computational complexity and memory cost. We propose an attention mechanism with low-rank approximation in frequency and a mixture of experts decomposition to control the distribution shifting. The proposed frequency enhanced structure decouples the input sequence length and the attention matrix dimension, leading to the linear complexity. Moreover, we theoretically and empirically prove the effectiveness of the adopted random mode selection policy in frequency. Lastly, extensive experiments show that the proposed model achieves the best forecasting performance on six benchmark datasets in comparison with four state-of-the-art algorithms.


## A. Related Work

In this section, an overview of the literature for time series forecasting will be given. The relevant works include traditional times series models (A.1), deep learning models (A.1), Transformer-based models (A.2), and the Fourier Transform in neural networks (A.3).

## A.1. Traditional Time Series Models

Data-driven time series forecasting helps researchers understand the evolution of the systems without architecting the exact physics law behind them. After decades of renovation, time series models have been well developed and served as the backbone of various projects in numerous application fields. The first generation of data-driven methods can date back to 1970. ARIMA (Box \& Jenkins, 1968; Box \& Pierce, 1970) follows the Markov process and builds an auto-regressive model for recursively sequential forecasting. However, an autoregressive process is not enough to deal with nonlinear and non-stationary sequences. With the bloom of deep neural networks in the new century, recurrent neural networks (RNN) was designed especially for tasks involving sequential data. Among the family of RNNs, LSTM (Hochreiter \& Schmidhuber, 1997) and GRU (Chung et al., 2014) employ gated structure to control the information flow to deal with the gradient vanishing or exploration problem. DeepAR (Flunkert et al., 2017) uses a sequential architecture for probabilistic forecasting by incorporating binomial likelihood. Attention based RNN (Qin et al., 2017) uses temporal attention to capture long-range dependencies. However, the recurrent model is not parallelizable and unable to handle long dependencies. The temporal convolutional network (Sen et al., 2019) is another family efficient in sequential tasks. However, limited to the reception field of the kernel, the features extracted still stay local and long-term dependencies are hard to grasp.

## A.2. Transformers for Time Series Forecasting

With the innovation of transformers in natural language processing (Vaswani et al., 2017; Devlin et al., 2019) and computer vision tasks (Dosovitskiy et al., 2021; Rao et al., 2021), transformer-based models are also discussed, renovated, and applied in time series forecasting (Zhou et al., 2021; Wu et al., 2021). In sequence to sequence time series forecasting tasks an encoder-decoder architecture is popularly employed. The self-attention and cross-attention mechanisms are used as the core layers in transformers. However, when employing a point-wise connected matrix, the transformers suffer from quadratic computation complexity.

---

To get efficient computation without sacrificing too much on performance, the earliest modifications specify the attention matrix with predefined patterns. Examples include: (Qiu et al., 2020) uses block-wise attention which reduces the complexity to the square of block size. Longformer (Beltagy et al., 2020) employs a stride window with fixed intervals. LogTrans (Li et al., 2019) uses logsparse attention and achieves $N \log ^{2} N$ complexity. Htransformer (Zhu \& Soricut, 2021) uses a hierarchical pattern for sparse approximation of attention matrix with $O(n)$ complexity. Some work uses a combination of patterns (BIGBIRD (Zaheer et al., 2020)) mentioned above. Another strategy is to use dynamic patterns: Reformer (Kitaev et al., 2020) introduces a local-sensitive hashing which reduces the complexity to $N \log N$. (Zhu \& Soricut, 2021) introduces a hierarchical pattern. Sinkhorn (Tay et al., 2020) employs a block sorting method to achieve quasi-global attention with only local windows.

---

Similarly, some work employs a top-k truncating to accelerate computing: Informer (Zhou et al., 2021) uses a KLdivergence based method to select top-k in attention matrix. This sparser matrix costs only $N \log N$ in complexity. Autoformer ( Wu et al., 2021) introduces an auto-correlation block in place of canonical attention to get the sub-series level attention, which achieves $N \log N$ complexity with the help of Fast Fourier transform and top-k selection in an auto-correlation matrix.

---

Another emerging strategy is to employ a low-rank approximation of the attention matrix. Linformer (Wang et al., 2020) uses trainable linear projection to compress the sequence length and achieves $O(n)$ complexity and theoretically proves the boundary of approximation error based on JL lemma. Luna (Ma et al., 2021) develops a nested linear structure with $O(n)$ complexity. Nyströformer (Xiong et al., 2021) leverages the idea of Nyström approximation in the attention mechanism and achieves an $O(n)$ complexity. Performer (Choromanski et al., 2021) adopts an orthogonal random features approach to efficiently model kernelizable attention mechanisms.

## A.3. Fourier Transform in Transformers

Thanks to the algorithm of fast Fourier transform (FFT), the computation complexity of Fourier transform is compressed from $N^{2}$ to $N \log N$. The Fourier transform has the property that convolution in the time domain is equivalent to multiplication in the frequency domain. Thus the FFT can be used in the acceleration of convolutional networks (Mathieu et al., 2014). FFT can also be used in efficient computing of auto-correlation function, which can be used as a building neural networks block (Wu et al., 2021) and also useful in numerous anomaly detection tasks (Homayouni et al., 2020). (Li et al., 2020; Gupta et al., 2021) first introduced Fourier Neural Operator in solving partial differential equations (PDEs). FNO is used as an inner block of networks to perform efficient representation learning in the low-frequency domain. FNO is also proved efficient in computer vision tasks (Rao et al., 2021). It also serves as a working horse to build the Wavelet Neural Operator (WNO), which is recently introduced in solving PEDs (Gupta et al., 2021). While FNO keeps the spectrum modes in low frequency, random Fourier method use randomly selected modes. (Rahimi \& Recht, 2008) proposes to map the input data to a randomized low-dimensional feature space to accelerate the training of kernel machines. (Rawat et al., 2019) proposes the Random Fourier softmax (RF-softmax) method that utilizes the powerful Random Fourier Features to enable more efficient and accurate sampling from an approximate softmax distribution.

---

To the best of our knowledge, our proposed method is the first work to achieve fast attention mechanism through low rank approximated transformation in frequency domain for time series forecasting.

## B. Low-rank Approximation of Attention

In this section, we discuss the low-rank approximation of the attention mechanism. First, we present the Restricted Isometry Property (RIP) matrices whose approximate error bound could be theoretically given in B.1. Then in B.2, we follow prior work and present how to leverage RIP matrices and attention mechanisms.

---

If the signal of interest is sparse or compressible on a fixed basis, then it is possible to recover the signal from fewer measurements. (Wang et al., 2020; Xiong et al., 2021) suggest that the attention matrix is low-rank, so the attention matrix can be well approximated if being projected into a subspace where the attention matrix is sparse. For the efficient computation of the attention matrix, how to properly select the basis of the projection yet remains to be an open question. The basis which follows the RIP is a potential candidate.

## B.1. RIP Matrices

The definition of the RIP matrices is:

Definition B.1. RIP matrices. Let $m<n$ be positive integers, $\Phi$ be a $m \times n$ matrix with real entries, $\delta>0$, and $K<m$ be an integer. We say that $\Phi$ is $(K, \delta)-R I P$, if for every $K$-sparse vector $x \in \mathbb{R}^{n}$ we have $(1-\delta)\|x\| \leq$ $\|\Phi x\| \leq(1+\delta)\|x\|$.

RIP matrices are the matrices that satisfy the restricted isometry property, discovered by D. Donoho, E. Candès and T. Tao in the field of compressed sensing. RIP matrices might be good choices for low-rank approximation because of their good properties. A random matrix has a negligible probability of not satisfying the RIP and many kinds of matrices have proven to be RIP, for example, Gaussian basis, Bernoulli basis, and Fourier basis.

---

Theorem 2. Let $m<n$ be positive integers, $\delta>0$, and $K=O\left(\frac{m}{\log ^{4} n}\right)$. Let $\Phi$ be the random matrix defined by one of the following methods:

(Gaussian basis) Let the entries of $\Phi$ be i.i.d. with a normal distribution $N\left(0, \frac{1}{m}\right)$.

(Bernoulli basis) Let the entries of $\Phi$ be i.i.d. with a Bernoulli distribution taking the values $\pm \frac{1}{\sqrt{m}} m$, each with 50\% probability.

(Random selected Discrete Fourier basis) Let $A \subset$ $\{0, \ldots, n-1\}$ be a random subset of size $m$. Let $\Phi$ be the matrix obtained from the Discrete Fourier transform matrix (i.e. the matrix $F$ with entries $F[l, j]=\exp ^{-2 \pi i l j / n} / \sqrt{n}$ ) for $l, j \in\{0, . ., n-1\}$ by selecting the rows indexed by $A$.

Then $\Phi$ is $(K, \sigma)-R I P$ with probability $p \approx 1-e^{-n}$.

Theorem 2 states that Gaussian basis, Bernoulli basis and Fourier basis follow RIP. In the following section, the Fourier basis is used as an example and show how to use RIP basis in low-rank approximation in the attention mechanism.

## B.2. Low-rank Approximation with Fourier Basis/Legendre Polynomials

Linformer (Wang et al., 2020) demonstrates that the attention mechanism can be approximated by a low-rank matrix. Linformer uses a trainable kernel initialized with Gaussian distribution for the low-rank approximation, While our proposed FEDformer uses Fourier basis/Legendre Polynomials, Gaussian basis, Fourier basis, and Legendre Polynomials all obey RIP, so similar conclusions could be drawn.

---

Starting from Johnson-Lindenstrauss lemma (Johnson, 1984) and using the version from (Arriaga \& Vempala, 2006), Linformer proves that a low-rank approximation of the attention matrix could be made.

---

Let $\Phi \in \mathbb{R}^{N \times M}$ be the random selected Fourier basis/Legendre Polynomials. $\Phi$ is RIP matrix. Referring to Theorem 2 , with a probability $p \approx 1-e^{-n}$, for any $x \in \mathbb{R}^{N}$, we have

$$
\begin{equation*}
(1-\delta)\|x\| \leq\|\Phi x\| \leq(1+\delta)\|x\| \tag{11}
\end{equation*}
$$

Referring to (Arriaga \& Vempala, 2006), with a probability $p \approx 1-4 e^{-n}$, for any $x_{1}, x_{2} \in \mathbb{R}^{N}$, we have

$$
\begin{equation*}
(1-\delta)\left\|x_{1} x_{2}^{\top}\right\| \leq\left\|x_{1} \Phi^{\top} \Phi x_{2}^{\top}\right\| \leq(1+\delta)\left\|x_{1} x_{2}^{\top}\right\| \tag{12}
\end{equation*}
$$

With the above inequation function, we now discuss the case in attention mechanism. Let the attention matrix $B=\operatorname{softmax}\left(\frac{Q K^{\top}}{\sqrt{d}}\right)=\exp (A) \cdot D_{A}^{-1}$, where $\left(D_{A}\right)_{i i}=$ $\sum_{n=1}^{N} \exp \left(A_{n i}\right)$. Following Linformer, we can conclude a theorem as (please refer to (Wang et al., 2020) for the detailed proof)

---

Theorem 3. For any row vector $p \in \mathbb{R}^{N}$ of matrix $B$ and any column vector $v \in \mathbb{R}^{N}$ of matrix $V$, with a probability $p=1-o(1)$, we have

$$
\begin{equation*}
\left\|b \Phi^{\top} \Phi v^{\top}-b v^{\top}\right\| \leq \delta\left\|b v^{\top}\right\| \tag{13}
\end{equation*}
$$

Theorem 3 points out the fact that, using Fourier basis/Legendre Polynomials $\Phi$ between the multiplication of attention matrix $(P)$ and values $(V)$, the computation complexity can be reduced from $O\left(N^{2} d\right)$ to $O(N M d)$, where $d$ is the hidden dimension of the matrix. In the meantime, the error of the low-rank approximation is bounded. However, Theorem 3 only discussed the case which is without the activation function.

---

Furthermore, with the Cauchy inequality and the fact that the exponential function is Lipchitz continuous in a compact region (please refer to (Wang et al., 2020) for the proof), we can draw the following theorem:

Theorem 4. For any row vector $A_{i} \in \mathbb{R}^{N}$ in matrix $A$ ( $A=\frac{Q K^{\top}}{\sqrt{d}}$ ), with a probability of $p=1-o(1)$, we have

$$
\begin{equation*}
\left\|\exp \left(A_{i} \Phi^{\top}\right) \Phi v^{\top}-\exp \left(A_{i}\right) v^{\top}\right\| \leq \delta\left\|\exp \left(A_{i}\right) v^{\top}\right\| . \tag{14}
\end{equation*}
$$

Theorem 4 states that with the activation function (softmax), the above discussed bound still holds.

---

In summary, we can leverage RIP matrices for low-rank approximation of attention. Moreover, there exists theoretical error bound when using a randomly selected Fourier basis for low-rank approximation in the attention mechanism.

## C. Fourier Component Selection

Let $X_{1}(t), \ldots, X_{m}(t)$ be $m$ time series. By applying Fourier transform to each time series, we turn each $X_{i}(t)$ into a vector $a_{i}=\left(a_{i, 1}, \ldots, a_{i, d}\right)^{\top} \in \mathbb{R}^{d}$. By putting all the Fourier transform vectors into a matrix, we have $A=\left(a_{1}, a_{2}, \ldots, a_{m}\right)^{\top} \in \mathbb{R}^{m \times d}$, with each row corresponding to a different time series and each column corresponding to a different Fourier component. Here, we propose to select $s$ components from the $d$ Fourier components $(s<d)$ uniformly at random. More specifically, we denote by $i_{1}<i_{2}<\ldots<i_{s}$ the randomly selected components. We construct matrix $S \in\{0,1\}^{s \times d}$, with $S_{i, k}=1$ if $i=i_{k}$ and $S_{i, k}=0$ otherwise. Then, our representation of multivariate time series becomes $A^{\prime}=A S^{\top} \in \mathbb{R}^{m \times s}$. The following theorem shows that, although the Fourier basis is randomly selected, under a mild condition, $A^{\prime}$ can preserve most of the information from $A$.

---

Theorem 5. Assume that $\mu(A)$, the coherence measure of matrix $A$, is $\Omega(k / n)$. Then, with a high probability, we have

$$
\left|A-P_{A^{\prime}}(A)\right| \leq(1+\epsilon)\left|A-A_{k}\right|
$$

if $s=O\left(k^{2} / \epsilon^{2}\right)$.

Proof. Following the analysis in Theorem 3 from (Drineas et al., 2007), we have

$$
\begin{aligned}
\left|A-P_{A^{\prime}}(A)\right| & \leq\left|A-A^{\prime}\left(A^{\prime}\right)^{\dagger} A_{k}\right| \\
& =\left|A-\left(A S^{\top}\right)\left(A S^{\top}\right)^{\dagger} A_{k}\right| \\
& =\left|A-\left(A S^{\top}\right)\left(A_{k} S^{\top}\right)^{\dagger} A_{k}\right| .
\end{aligned}
$$

Using Theorem 5 from (Drineas et al., 2007), we have, with a probability at least 0.7 ,

$$
\left|A-\left(A S^{\top}\right)\left(A_{k} S^{\top}\right)^{\dagger} A_{k}\right| \leq(1+\epsilon)\left|A-A_{k}\right|
$$

if $s=O\left(k^{2} / \epsilon^{2} \times \mu(A) n / k\right)$. The theorem follows because $\mu(A)=O(k / n)$.

## D. Wavelets

In this section, we present some technical background about Wavelet transform which is used in our proposed framework.

## D.1. Continuous Wavelet Transform

First, let's see how a function $f(t)$ is decomposed into a set of basis functions $\psi_{\mathrm{s}, \tau}(t)$, called the wavelets. It is known as the continuous wavelet transform or $C W T$. More formally it is written as

$$
\gamma(s, \tau)=\int f(t) \Psi_{s, \tau}^{*}(t) d t
$$

where * denotes complex conjugation. This equation shows the variables $\gamma(s, \tau), s$ and $\tau$ are the new dimensions, scale, and translation after the wavelet transform, respectively.

---

The wavelets are generated from a single basic wavelet $\Psi(t)$, the so-called mother wavelet, by scaling and translation as

$$
\psi_{s, \tau}(t)=\frac{1}{\sqrt{s}} \psi\left(\frac{t-\tau}{s}\right)
$$

where $s$ is the scale factor, $\tau$ is the translation factor, and $\sqrt{s}$ is used for energy normalization across the different scales.

## D.2. Discrete Wavelet Transform

Continues wavelet transform maps a one-dimensional signal to a two-dimensional time-scale joint representation which is highly redundant. To overcome this problem, people introduce discrete wavelet transformation (DWT) with mother wavelet as

$$
\psi_{j, k}(t)=\frac{1}{\sqrt{s_{0}^{j}}} \psi\left(\frac{t-k \tau_{0} s_{0}^{j}}{s_{0}^{j}}\right)
$$

DWT is not continuously scalable and translatable but can be scaled and translated in discrete steps. Here $j$ and $k$ are integers and $s_{0}>1$ is a fixed dilation step. The translation factor $\tau_{0}$ depends on the dilation step. The effect of discretizing the wavelet is that the time-scale space is now sampled at discrete intervals. We usually choose $s_{0}=2$ so that the sampling of the frequency axis corresponds to dyadic sampling. For the translation factor, we usually choose $\tau_{0}=1$ so that we also have a dyadic sampling of the time axis.

---

When discrete wavelets are used to transform a continuous signal, the result will be a series of wavelet coefficients and it is referred to as the wavelet decomposition.

## D.3. Orthogonal Polynomials

The next thing we need to focus on is orthogonal polynomials (OPs), which will serve as the mother wavelet function we introduce before. A lot of properties have to be maintained to be a mother wavelet, like admissibility condition, regularity conditions, and vanishing moments. In short, we are interested in the OPs that are non-zero over a finite domain and are zero almost everywhere else. Legendre is a popular set of OPs used it in our work here. Some other popular OPs can also be used here like Chebyshev without much modification.

## D.4. Legendre Polynomails

The Legendre polynomials are defined with respect to (w.r.t.) a uniform weight function $w_{L}(x)=1$ for $-1 \leqslant$ $x \leqslant 1$ or $w_{L}(x)=\mathbf{1}_{[-1,1]}(x)$ such that

$$
\int_{-1}^{1} P_{i}(x) P_{j}(x) d x= \begin{cases}\frac{2}{2 i+1} & i=j \\ 0 & i \neq j\end{cases}
$$

Here the function is defined over $[-1,1]$, but it can be extended to any interval $[a, b]$ by performing different shift and scale operations.

## D.5. Multiwavelets

The multiwavelets which we use in this work combine advantages of the wavelet and OPs we introduce before. Other than projecting a given function onto a single wavelet function, multiwavelet projects it onto a subspace of degree-restricted polynomials. In this work, we restricted our exploration to one family of OPs: Legendre Polynomials.

---

First, the basis is defined as: A set of orthonormal basis w.r.t. measure $\mu$, are $\phi_{0}, \ldots, \phi_{k-1}$ such that $\left\langle\phi_{i}, \phi_{j}\right\rangle_{\mu}=\delta_{i j}$. With a specific measure (weighting function $w(x)$ ), the orthonormality condition can be written as $\int \phi_{i}(x) \phi_{j}(x) w(x) d x=\delta_{i j}$.

---

Follow the derivation in (Gupta et al., 2021), through using the tools of Gaussian Quadrature and Gram-Schmidt Orthogonalizaition, the filter coefficients of multiwavelets using Legendre polynomials can be written as

$$
\begin{aligned}
H_{i j}^{(0)} & =\sqrt{2} \int_{0}^{1 / 2} \phi_{i}(x) \phi_{j}(2 x) w_{L}(2 x-1) d x \\
& =\frac{1}{\sqrt{2}} \int_{0}^{1} \phi_{i}(x / 2) \phi_{j}(x) d x \\
& =\frac{1}{\sqrt{2}} \sum_{i=1}^{k} \omega_{i} \phi_{i}\left(\frac{x_{i}}{2}\right) \phi_{j}\left(x_{i}\right)
\end{aligned}
$$

For example, if $k=3$, following the formula, the filter coefficients are derived as follows

$$
\begin{aligned}
& H^{0}=\left[\begin{array}{ccc}
\frac{1}{\sqrt{2}} & 0 & 0 \\
-\frac{\sqrt{3}}{2 \sqrt{2}} & \frac{1}{2 \sqrt{2}} & 0 \\
0 & -\frac{\sqrt{15}}{4 \sqrt{2}} & \frac{1}{4 \sqrt{2}}
\end{array}\right], H^{1}=\left[\begin{array}{ccc}
\frac{1}{\sqrt{2}} & 0 & 0 \\
\frac{\sqrt{3}}{2 \sqrt{2}} & \frac{1}{2 \sqrt{2}} & 0 \\
0 & \frac{\sqrt{15}}{4 \sqrt{2}} & \frac{1}{4 \sqrt{2}}
\end{array}\right], \\
& G^0=\left[\begin{array}{ccc}
\frac{1}{2 \sqrt{2}} & \frac{\sqrt{3}}{2 \sqrt{2}} & 0 \\
0 & \frac{1}{4 \sqrt{2}} & \frac{\sqrt{15}}{4 \sqrt{2}} \\
0 & 0 & \frac{1}{\sqrt{2}}
\end{array}\right],G^1=\left[\begin{array}{ccc}
-\frac{1}{2 \sqrt{2}} & \frac{\sqrt{3}}{2 \sqrt{2}} & 0 \\
0 & -\frac{1}{4 \sqrt{2}} & \frac{\sqrt{15}}{4 \sqrt{2}} \\
0 & 0 & -\frac{1}{\sqrt{2}}
\end{array}\right]
\end{aligned}
$$

## E. Output Distribution Analysis

## E.1. Bad Case Analysis

Using vanilla Transformer as baseline model, we demonstrate two bad long-term series forecasting cases in ETTm1 dataset as shown in the following Figure 7.

- Figure 7. Different distribution between ground truth and forecasting output from vanilla Transformer in a real-world ETTm1 dataset. Left: frequency mode and trend shift. Right: trend shift.

---

The forecasting shifts in Figure 7 is particularly related to the point-wise generation mechanism adapted by the vanilla Transformer model. To the contrary of classic models like Autoregressive integrated moving average (ARIMA) which has a predefined data bias structure for output distribution, Transformer-based models forecast each point independently and solely based on the overall MSE loss learning. This would result in different distribution between ground truth and forecasting output in some cases, leading to performance degradation.

## E.2. Kolmogorov-Smirnov Test

We adopt Kolmogorov-Smirnov (KS) test to check whether the two data samples come from the same distribution. KS test is a nonparametric test of the equality of continuous or discontinuous, two-dimensional probability distributions. In essence, the test answers the question "what is the probability that these two sets of samples were drawn from the same (but unknown) probability distribution". It quantifies a distance between the empirical distribution function of two samples. The Kolmogorov-Smirnov statistic is

$$
D_{n, m}=\sup _{x}\left|F_{1, n}(x)-F_{2, m}(x)\right|
$$

where $F_{1, n}$ and $F_{2, m}$ are the empirical distribution functions of the first and the second sample respectively, and sup is the supremum function. For large samples, the null hypothesis is rejected at level $\alpha$ if

$$
D_{n, m}>\sqrt{-\frac{1}{2} \ln \left(\frac{\alpha}{2}\right)} \cdot \sqrt{\frac{n+m}{n \cdot m}}
$$

where $n$ and $m$ are the sizes of the first and second samples respectively.

## E.3. Distribution Experiments and Analysis

Though the KS test omits the temporal information from the input and output sequence, it can be used as a tool to measure the global property of the foretasting output sequence compared to the input sequence. The null hypothesis is that the two samples come from the same distribution. We can tell that if the P-value of the KS test is large and then the null hypothesis is less likely to be rejected for true output distribution.

---

We applied KS test on the output sequence of 96-720 prediction tasks for various models on the ETTm1 and ETTm2 datasets, and the results are summarized in Table 6. In the test, we compare the fixed 96 -time step input sequence distribution with the output sequence distribution of different lengths. Using a 0.01 P -value as statistics, various existing Transformer baseline models have much less P-value than 0.01 except Autoformer, which indicates they have a higher probability to be sampled from the different distributions. Autoformer and FEDformer have much larger $P$ value compared to other models, which mainly contributes to their seasonal trend decomposition mechanism. Though we get close results from ETTm1 by both models, the proposed FEDformer has much larger P-values in ETTm1. And it is the only model whose null hypothesis can not be rejected with P -value larger than 0.01 in all cases of the two datasets, implying that the output sequence generated by FEDformer shares a more similar distribution as the input sequence than others and thus justifies the our design motivation of FEDformer as discussed in Section 1.

---

Note that in the ETTm1 dataset, the True output sequence has a smaller P-value compared to our FEDformer's predicted output, it shows that the model's close output distribution is achieved through model's control other than merely more accurate prediction. This analysis shed some light on why the seasonal-trend decomposition architecture can give us better performance in long-term forecasting. The design is used to constrain the trend (mean) of the output distribution. Inspired by such observation, we design frequency enhanced block to constrain the seasonality (frequency mode) of the output distribution.

## F. Supplemental Experiments

## F.1. Dataset Details

In this paragraph, the details of the experiment datasets are summarized as follows: 1) ETT (Zhou et al., 2021) dataset contains two sub-dataset: ETT1 and ETT2, collected from two electricity transformers at two stations. Each of them has two versions in different resolutions ( $15 \mathrm{~min} \& 1 \mathrm{~h}$ ). ETT dataset contains multiple series of loads and one series of oil temperatures. 2) Electricity ${ }^{1}$ dataset contains the electricity consumption of clients with each column corresponding to one client. 3) Exchange (Lai et al., 2018) contains the current exchange of 8 countries. 4) Traffic ${ }^{2}$ dataset contains the occupation rate of freeway system across the State of California. 5) Weather ${ }^{3}$ dataset contains 21 meteorological indicators for a range of 1 year in Germany. 6) Illness ${ }^{4}$ dataset contains the influenza-like illness patients in the United States. Table 7 summarizes feature details (Sequence Length: Len, Dimension: Dim, Frequency: Freq) of the six datasets. All datasets are split into the training set, validation set and test set by the ratio of 7:1:2.

## F.2. Implementation Details

Our model is trained using ADAM (Kingma \& Ba, 2017) optimizer with a learning rate of $1 e^{-4}$. The batch size is set to 32 . An early stopping counter is employed to stop the training process after three epochs if no loss degradation on the valid set is observed. The mean square error (MSE) and mean absolute error (MAE) are used as metrics. All experiments are repeated 5 times and the mean of the metrics is used in the final results. All the deep learning networks are implemented in PyTorch (Paszke et al., 2019) and trained on NVIDIA V100 32GB GPUs.

## F.3. ETT Full Benchmark

We present the full-benchmark on the four ETT datasets (Zhou et al., 2021) in Table 8 (multivariate forecasting) and Table 9 (univariate forecasting). The ETTh 1 and ETTh2 are recorded hourly while ETTm1 and ETTm 2 are recorded every 15 minutes. The time series in ETTh1 and ETTm1 follow the same pattern, and the only difference is the sampling rate, similarly for ETTh2 and ETTm2. On average, our FEDformer yields a $\mathbf{1 1 . 5 \%}$ relative MSE reduction for multivariate forecasting, and a $\mathbf{9 . 4 \%}$ reduction for univariate forecasting over the SOTA results from Autoformer.

## F.4. Cross Attention Visualization

The $\sigma\left(\tilde{\boldsymbol{Q}} \cdot \tilde{\boldsymbol{K}}^{\top}\right)$ can be viewed as the cross attention weight for our proposed frequency enhanced cross attention block. Several different activation functions can be used for attention matrix activation. Tanh and softmax are tested in this work with various performances on different datasets. We use tanh as the default one. Different attention patterns are visualized in Figure 8. Here two samples of cross attention maps are shown for FEDformer-f training on the ETTm2 dataset using tanh and softmax respectively. It can be seen that attention with Softmax as activation function seems to be more sparse than using tanh. Overall we can see attention in the frequency domain is much sparser compared to the normal attention graph in the time domain, which indicates our proposed attention can represent the signal more compactly. Also this compact representation supports our random mode selection mechanism to achieve linear complexity.

## F.5. Improvements of Mixture of Experts Decomposition

We design a mixture of experts decomposition mechanism which adopts a set of average pooling layers to extract the trend and a set of data-dependent weights to combine them. The default average pooling layers contain filters with kernel size $7,12,14,24$ and 48 respectively. For comparison, we use single expert decomposition mechanism which employs a single average pooling layer with a fixed kernel size of 24 as the baseline. In Table 10, a comparison study of multivariate forecasting is shown using FEDformer-f model on two typical datasets. It is observed that the designed mixture of experts decomposition brings better performance than the single decomposition scheme.

## F.6. Multiple random runs

Table 11 lists both mean and standard deviation (STD) for FEDformer-f and Autoformer with 5 runs. We observe a small variance in the performance of FEDformer-f, despite the randomness in frequency selection.

## F.7. Sensitivity to the number of modes: ETTx1 vs ETTx2

The choice of modes number depends on data complexity. The time series that exhibits the higher complex patterns requires the larger the number of modes. To verify this claim, we summarize the complexity of ETT datasets, measured by permutation entropy and SVD entropy, in Table 12. It is observed that ETTx1 has a significantly higher complexity (corresponding to a higher entropy value) than ETTx2, thus requiring a larger number of modes.


## F.8. When Fourier/Wavelet model performs better

Our high level principle of model deployment is that Fourier-based model is usually better for less complex time series, while wavelet is normally more suitable for complex ones. Specifically, we found that wavelet-based model is more effective on multivariate time series, while Fourierbased one normally achieves better results on univariate time series. As indicated in Table 13, complexity measures on multivariate time series are higher than those on univariate ones.
