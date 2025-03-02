# iTransformer: Inverted Transformers Are Effective for Time Series Forecasting 


## Abstract

The recent boom of linear forecasting models questions the ongoing passion for architectural modifications of Transformer-based forecasters. These forecasters leverage Transformers to model the global dependencies over temporal tokens of time series, with each token formed by multiple variates of the same timestamp. However, Transformers are challenged in forecasting series with larger lookback windows due to performance degradation and computation explosion. Besides, the embedding for each temporal token fuses multiple variates that represent potential delayed events and distinct physical measurements, which may fail in learning variate-centric representations and result in meaningless attention maps. In this work, we reflect on the competent duties of Transformer components and repurpose the Transformer architecture without any modification to the basic components. We propose iTransformer that simply applies the attention and feed-forward network on the inverted dimensions. Specifically, the time points of individual series are embedded into variate tokens which are utilized by the attention mechanism to capture multivariate correlations; meanwhile, the feed-forward network is applied for each variate token to learn nonlinear representations. The iTransformer model achieves state-of-the-art on challenging real-world datasets, which further empowers the Transformer family with promoted performance, generalization ability across different variates, and better utilization of arbitrary lookback windows, making it a nice alternative as the fundamental backbone of time series forecasting. Code is available at this repository: https://github.com/thuml/iTransformer.


# 1 Introduction

Transformer (Vaswani et al. 2017) has achieved tremendous success in natural language processing (Brown et al., 2020) and computer vision (Dosovitskiy et al. 2021), growing into the foundation model that follows the scaling law (Kaplan et al., 2020). Inspired by the immense success in extensive fields, Transformer with strong capabilities of depicting pairwise dependencies and extracting multi-level representations in sequences is emerging in time series forecasting (Wu et al., 2021, Nie et al. 2023).

---

However, researchers have recently begun to question the validity of Transformer-based forecasters, which typically embed multiple variates of the same timestamp into indistinguishable channels and apply attention on these temporal tokens to capture temporal dependencies. Considering the numerical but less semantic relationship among time points, researchers find that simple linear layers, which can be traced back to statistical forecasters (Box \&Jenkins, 1968), have exceeded complicated Transformers on both performance and efficiency (Zeng et al. 2023, Das et al. 2023). Meanwhile, ensuring the independence of variate and utilizing mutual information is ever more highlighted by recent research that explicitly models multivariate correlations to achieve accurate forecasting (Zhang \& Yan, 2023, Ekambaram et al., 2023), but this goal can be hardly achieved without subverting the vanilla Transformer architecture.

---

Considering the disputes of Transformer-based forecasters, we reflect on why Transformers perform even worse than linear models in time series forecasting while acting predominantly in many other fields. We notice that the existing structure of Transformer-based forecasters may be not suitable for multivariate time series forecasting. As shown on the top of Figure 2, it is notable that the points of the same time step that basically represent completely different physical meanings recorded by inconsistent measurements are embedded into one token with wiped-out multivariate correlations. And the token formed by a single time step can struggle to reveal beneficial information due to excessively local receptive field and time-unaligned events represented by simultaneous time points. Besides, while series variations can be greatly influenced by the sequence order, permutationinvariant attention mechanisms are improperly adopted on the temporal dimension (Zeng et al. 2023). Consequently, Transformer is weakened to capture essential series representations and portray multivariate correlations, limiting its capacity and generalization ability on diverse time series data.

- Figure 2: Comparison between the vanilla Transformer (top) and the proposed iTransformer (bottom). Transformer embeds the temporal token, which contains the multivariate representation of each time step. iTransformer embeds each series independently to the variate token, such that the attention module depicts the multivariate correlations and the feed-forward network encodes series representations.

---

Concerning the potential risks of embedding multivariate points of a timestamp as a (temporal) token, we take an inverted view on time series and embed the whole time series of each variate independently into a (variate) token, the extreme case of Patching (Nie et al., 2023) that enlarges local receptive field. By inverting, the embedded token aggregates the global representations of series that can be more variate-centric and better leveraged by booming attention mechanisms for multivariate correlating. Meanwhile, the feed-forward network can be proficient enough to learn generalizable representations for distinct variates encoded from arbitrary lookback series and decoded to predict future series.

---

Based on the above motivations, we believe it is not that Transformer is ineffective for time series forecasting, but rather it is improperly used. In this paper, we revisit the structure of Transformer and advocate iTransformer as a fundamental backbone for time series forecasting. Technically, we embed each time series as variate tokens, adopt the attention for multivariate correlations, and employ the feed-forward network for series representations. Experimentally, the proposed iTransformer achieves state-of-the-art performance on real-world forecasting benchmarks shown in Figure 1 and surprisingly tackles the pain points of Transformer-based forecasters. Our contributions lie in three aspects:

- We reflect on the architecture of Transformer and refine that the competent capability of native Transformer components on multivariate time series is underexplored.

- We propose iTransformer that regards independent time series as tokens to capture multivariate correlations by self-attention and utilize layer normalization and feed-forward network modules to learn better series-global representations for time series forecasting.

- Experimentally, iTransformer achieves comprehensive state-of-the-art on real-world benchmarks. We extensively analyze the inverted modules and architecture choices, indicating a promising direction for the future improvement of Transformer-based forecasters.


# 2 Related Work

With the progressive breakthrough made in natural language processing and computer vision areas, elaboratively designed Transformer variants are proposed to tackle ubiquitous time series forecasting applications. Going beyond contemporaneous TCNs (Bai et al., 2018; Liu et al., 2022a) and RNNbased forecasters (Zhao et al., 2017, Rangapuram et al., 2018; Salinas et al., 2020), Transformer has exhibited powerful sequence modeling capability and promising model scalability, leading to the trend of passionate modifications adapted for time series forecasting.

---

Through a systematical review of Transformer-based forecasters, we conclude that existing modifications can be divided into four categories by whether to modify the component and architecture. As shown in Figure 3 , the first category (Wu et al., 2021; Li et al., 2021; Zhou et al., 2022), which is the most common practice, mainly concerns the component adaptation, especially the attention module for the temporal dependency modeling and the complexity optimization on long sequences. Nevertheless, with the rapid emergence of linear forecasters (Oreshkin et al., 2019, Zeng et al., 2023, Das et al., 2023, Liu et al., 2023), the impressive performance and efficiency continuously challenge this direction. Soon afterward, the second category attempts to fully utilize Transformer. It pays more attention to the inherent processing of time series, such as Stationarization (Liu et al., 2022b), Channel Independence, and Patching (Nie et al., 2023), which bring about consistently improved performance. Moreover, faced with the increasing significance of the independence and mutual interactions of multiple variates, the third category refurbishes Transformer in both aspects of component and architecture. Representative (Zhang \& Yan, 2023) explicitly captures the cross-time and cross-variate dependencies by the renovated attention mechanism and architecture.

---

Unlike previous works, iTransformer modifies none of the native components of Transformer. Instead, we adopt the components on the inverted dimensions with the altered architecture, as the only one that belongs to the fourth category to our best knowledge. We believe the capabilities of the components have stood the test extensively, the truth is that the architecture of Transformer is improperly adopted.

# 3 ITRANSFORMER

In multivariate time series forecasting, given historical observations $\mathbf{X}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{T}\right\} \in \mathbb{R}^{T \times N}$ with $T$ time steps and $N$ variates, we predict the future $S$ time steps $\mathbf{Y}=\left\{\mathbf{x}_{T+1}, \ldots, \mathbf{x}_{T+S}\right\} \in$ $\mathbb{R}^{S \times N}$. For convenience, we denote $\mathbf{X}_{t,:}$ as the simultaneously recorded time points at the step $t$, and $\mathbf{X}_{:, n}$ as the whole time series of each variate indexed by $n$. It is notable that $\mathbf{X}_{t, \text { : }}$ may not contain time points that essentially reflect the same event in real-world scenarios because of the systematical time lags among variates in the dataset. Besides, the elements of $\mathbf{X}_{t, \text { : }}$ can be distinct from each other in physical measurements and statistical distributions, for which a variate $\mathbf{X}_{:, n}$ generally shares.

## 3.1 Structure Overview

Our proposed iTransformer illustrated in Figure 4 adopts the encoder-only architecture of Transformer (Vaswani et al., 2017), including the embedding, projection, and Transformer blocks.

- Figure 4: Overall structure of iTransformer, which shares the same modular arrangement with the encoder of Transformer. (a) Raw series of different variates are independently embedded as tokens. (b) Self-attention is applied to embedded variate tokens with enhanced interpretability revealing multivariate correlations. (c) Series representations of each token are extracted by the shared feedforward network. (d) Layer normalization is adopted to reduce the discrepancies among variates.

---

Embedding the whole series as the token. Most Transformer-based forecasters typically regard multiple variates of the same time as the (temporal) token and follow the generative formulation of forecasting tasks. However, we find the approach on the numerical modality can be less instructive for learning attention maps, which is supported by increasing applications of Patching (Dosovitskiy et al. 2021; Nie et al. 2023) that broadens the respective field. Meanwhile, the triumph of linear forecasters also challenges the necessity of adopting a heavy encoder-decoder Transformer for generating tokens. Instead, our proposed encoder-only iTransformer focuses on representation learning and adaptive correlating of multivariate series. Each time series driven by the underlying complicated process is firstly tokenized to describe the properties of the variate, applied by self-attention for mutual interactions, and individually processed by feed-forward networks for series representations. Notably, the task to generate the predicted series is essentially delivered to linear layers, which has been proven competent by previous work (Das et al. 2023) and we provide a detailed analysis in the next section.

---

Based on the above considerations, in iTransformer, the process of predicting future series of each specific variate $\hat{\mathbf{Y}}_{:, n}$ based on the lookback series $\mathbf{X}_{:, n}$ is simply formulated as follows:

$$
\begin{align*}
\mathbf{h}_{n}^{0} & =\operatorname{Embedding}\left(\mathbf{X}_{:, n}\right), \\
\mathbf{H}^{l+1} & =\operatorname{TrmBlock}\left(\mathbf{H}^{l}\right), l=0, \ldots, L-1,  \tag{1}\\
\hat{\mathbf{Y}}_{:, n} & =\operatorname{Projection}\left(\mathbf{h}_{n}^{L}\right),
\end{align*}
$$

where $\mathbf{H}=\left\{\mathbf{h}_{1}, \ldots, \mathbf{h}_{N}\right\} \in \mathbb{R}^{N \times D}$ contains $N$ embedded tokens of dimension $D$ and the superscript denotes the layer index. Embedding : $\mathbb{R}^{T} \mapsto \mathbb{R}^{D}$ and Projection : $\mathbb{R}^{D} \mapsto \mathbb{R}^{S}$ are both implemented by multi-layer perceptron (MLP). The obtained variate tokens interact with each other by self-attention and are independently processed by the shared feed-forward network in each TrmBlock. Specifically, as the order of sequence is implicitly stored in the neuron permutation of the feed-forward network, the position embedding in the vanilla Transformer is no longer needed here.

---

iTransformers. The architecture essentially presupposes no more specific requirements on Transformer variants, other than the attention is applicable for multivariate correlation. Thus, a bundle of efficient attention mechanisms (Li et al., 2021, Wu et al., 2022, Dao et al., 2022) can be the plugins, reducing the complexity when the variate number grows large. Besides, with the input flexibility of attention, the token number can vary from training to inference, and the model is allowed to be trained on arbitrary numbers of variates. The inverted Transformers, named iTransformers, are extensively evaluated in experiments of Section 4.2 and demonstrate advantages on time series forecasting.

## 3.2 Inverted Transformer Components

We organize a stack of $L$ blocks composed of the layer normalization, feed-forward network, and self-attention modules. But their duties on the inverted dimension are carefully reconsidered.

Layer normalization. Layer normalization (Ba et al. 2016) is originally proposed to increase the convergence and training stability of deep networks. In typical Transformer-based forecasters, the module normalizes the multivariate representation of the same timestamp, gradually fusing the variates with each other. Once the collected time points do not represent the same event, the operation will also introduce interaction noises between noncausal or delayed processes. In our inverted version, the normalization is applied to the series representation of individual variate as Equation2, which has been studied and proved effective in tackling non-stationary problems (Kim et al. 2021; Liu et al. 2022b). Besides, since all series as (variate) tokens are normalized to a Gaussian distribution, the discrepancies caused by inconsistent measurements can be diminished. By contrast, in previous architecture, different tokens of time steps will be normalized, leading to oversmooth time series.

$$
\begin{equation*}
\operatorname{LayerNorm}(\mathbf{H})=\left\{\left.\frac{\mathbf{h}_{n}-\operatorname{Mean}\left(\mathbf{h}_{n}\right)}{\sqrt{\operatorname{Var}\left(\mathbf{h}_{n}\right)}} \right\rvert\, n=1, \ldots, N\right\} \tag{2}
\end{equation*}
$$

---

Feed-forward network. Transformer adopts the feed-forward network (FFN) as the basic building block for encoding token representation and it is identically applied to each token. As aforementioned, in the vanilla Transformer, multiple variates of the same timestamp that form the token can be malpositioned and too localized to reveal enough information for predictions. In the inverted version, FFN is leveraged on the series representation of each variate token. By the universal approximation theorem (Hornik, 1991), they can extract complicated representations to describe a time series. With the stacking of inverted blocks, they are devoted to encoding the observed time series and decoding the representations for future series using dense non-linear connections, which work effectively as the recent works completely built on MLPs (Tolstikhin et al., 2021, Das et al. 2023).

---

More interestingly, the identical linear operation on independent time series, which serves as the combination of the recent linear forecasters (Zeng et al., 2023) and Channel Independence (Nie et al. 2023), can be instructive for us to understand the series representations. Recent revisiting on linear forecasters (Li et al., 2023) highlights that temporal features extracted by MLPs are supposed to be shared within distinct time series. We propose a rational explanation that the neurons of MLP are taught to portray the intrinsic properties of any time series, such as the amplitude, periodicity, and even frequency spectrums (neuron as a filter), serving as a more advantageous predictive representation learner than the self-attention applied on time points. Experimentally, we validate that the division of labor helps enjoy the benefits of linear layers in Section 4.3. such as the promoted performance if providing enlarged lookback series, and the generalization ability on unseen variates.

---

Self-attention. While the attention mechanism is generally adopted for facilitating the temporal dependencies modeling in previous forecasters, the inverted model regards the whole series of one variate as an independent process. Concretely, with comprehensively extracted representations of each time series $\mathbf{H}=\left\{\mathbf{h}_{0}, \ldots, \mathbf{h}_{N}\right\} \in \mathbb{R}^{N \times D}$, the self-attention module adopts linear projections to get queries, keys, and values $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d_{k}}$, where $d_{k}$ is the projected dimension.

---

With denotation of $\mathbf{q}_{i}, \mathbf{k}_{j} \in \mathbb{R}^{d_{k}}$ as the specific query and key of one (variate) token, we notice that each entry of the pre-Softmax scores is formulated as $\mathbf{A}_{i, j}=\left(\mathbf{Q} \mathbf{K}^{\top} / \sqrt{d_{k}}\right)_{i, j} \propto \mathbf{q}_{i}^{\top} \mathbf{k}_{j}$. Since each token is previously normalized on its feature dimension, the entries can somewhat reveal the variate-wise correlation, and the whole score map $\mathbf{A} \in \mathbb{R}^{N \times N}$ exhibits the multivariate correlations between paired variate tokens. Consequently, highly correlated variate will be more weighted for the next representation interaction with values $\mathbf{V}$. Based on this intuition, the proposed mechanism is believed to be more natural and interpretable for multivariate series forecasting. We further provide the visualization analysis of the score map in Section 4.3 and Appendix E. 1

# 4 EXPERIMENTS

We thoroughly evaluate the proposed iTransformer on various time series forecasting applications, validate the generality of the proposed framework and further dive into the effectiveness of applying the Transformer components on the inverted dimensions of time series.

Datasets. We extensively include 7 real-world datasets in our experiments, including ECL, ETT (4 subsets), Exchange, Traffic, Weather used by Autoformer (Wu et al., 2021), Solar-Energy datasets proposed in LSTNet (Lai et al., 2018), and PEMS (4 subsets) evaluated in SCINet (Liu et al., 2022a). We also provide the experiments on Market ( 6 subsets) in Appendix F.4. It records the minutesampled server load of Alipay online transaction application with hundreds of variates, where we consistently outperform other baselines. Detailed dataset descriptions are provided in Appendix A. 1

## 4.1 Forecasting Results

In this section, we conduct extensive experiments to evaluate the forecasting performance of our proposed model together with advanced deep forecasters.

Baselines. We carefully choose 10 well-acknowledged forecasting models as our benchmark, including (1) Transformer-based methods: Autoformer (Wu et al., 2021), FEDformer (Zhou et al. 2022), Stationary (Liu et al., 2022b), Crossformer (Zhang \& Yan, 2023), PatchTST (Nie et al.,|2023); (2) Linear-based methods: DLinear (Zeng et al., 2023), TiDE (Das et al., 2023), RLinear (Li et al. 2023); and (3) TCN-based methods: SCINet (Liu et al., 2022a), TimesNet (Wu et al., 2023).

---

Main results Comprehensive forecasting results are listed in Table 1 with the best in red and the second underlined. The lower MSE/MAE indicates the more accurate prediction result. Compared with other forecasters, iTransformer is particularly good at forecasting high-dimensional time series. Besides, PatchTST as the previous state-of-the-art, fails in many cases of PEMS, which can stem from the extremely fluctuating series of the dataset, and the patching mechanism of PatchTST may lose focus on specific locality to handle rapid fluctuation. By contrast, the proposed model aggregating the whole series variations for series representations can better cope with this situation. Notably, as the representative that explicitly captures multivariate correlations, the performance of Crossformer is still subpar to iTransformer, indicating the interaction of time-unaligned patches from different multivariate will bring about unnecessary noise for forecasting. Therefore, the native Transformer components are competent for temporal modeling and multivariate correlating, and the proposed inverted architecture can effectively tackle real-world time series forecasting scenarios.

- Table 1: Multivariate forecasting results with prediction lengths $S \in\{12,24,36,48\}$ for PEMS and $S \in\{96,192,336,720\}$ for others and fixed lookback length $T=96$. Results are averaged from all prediction lengths. Avg means further averaged by subsets. Full results are listed in Appendix F. 4

## 4.2 ITRANSFORMERS GENERALITY

In this section, we evaluate iTransformers by applying our framework to Transformer and its variants, which generally address the quadratic complexity of the self-attention mechanism, including Reformer (Kitaev et al., 2020), Informer (Li et al., 2021), Flowformer (Wu et al., 2022) and FlashAttention (Dao et al. 2022). Surprising and promising discoveries are exhibited, indicating the simple inverted perspective can enhance Transformer-based forecasters with promoted performance with efficiency, generalization on unseen variates, and better utilization of historical observations.

---

Performance promotion. We evaluate Transformers and the corresponding iTransformers with the reported performance promotions in Table 2. It is notable that the framework consistently improves various Transformers. Overall, it achieves averaged $\mathbf{3 8 . 9 \%}$ promotion on Transformer, $\mathbf{3 6 . 1 \%}$ on Reformer, $\mathbf{2 8 . 5 \%}$ on Informer, $\mathbf{1 6 . 8 \%}$ on Flowformer and $\mathbf{3 2 . 2 \%}$ on Flashformer, revealing the previous improper usage of the Transformer architecture on time series forecasting. Moreover, since the attention mechanism is adopted on the variate dimension in our inverted structure, the introduction of efficient attentions with linear complexity essentially addresses the computational problem due to numerous variates, which is prevalent in real-world applications but can be resource-consuming for Channel Independence (Nie et al., 2023). Therefore, the idea of iTransformer can be widely practiced on Transformer-based forecasters to take advantage of booming efficient attention mechanisms.

- Table 2: Performance promotion obtained by our inverted framework. Flashformer means Transformer equipped with hardware-accelerated FlashAttention (Dao et al., 2022). We report the average performance and the relative MSE reduction (Promotion). Full results can be found in Appendix F. 2

---

Variate generalization. By inverting vanilla Transformers, it is notable that the models are empowered with the generalization capability on unseen variates. Firstly, benefiting from the flexibility of the number of input tokens, the amount of variate channels is no longer restricted and thus feasible to vary from training and inference. Besides, feed-forward networks are identically applied on independent variate tokens in iTransformer. As aforementioned, the neurons as filters learn the intrinsic patterns of any time series, which are inclined to be shared and transferable among distinct variates.

---

To verify the hypothesis, we compare inverting with another generalizing strategy: Channel Independence, training a shared backbone to forecast all variates. We partition the variates of each dataset into five folders, train models with only $20 \%$ of variates of one folder, and directly forecast all variates without fine-tuning. We compare the performance in Figure 5 and each bar presents the averaged results of all folders to avoid the randomness of partition. CI-Transformers take a long time to predict each variate one by one during inference while iTransformers directly predict all variates and generally present smaller increases, indicating FFN is competent to learn transferable time series representations. It leaves a potential direction to build a foundation model upon iTransformer, where diverse multivariate time series with different numbers of variates can be feasibly trained together.

- Figure 5: Performance of generalization on unseen variates. We partition the variates of each dataset into five folders, train models with $20 \%$ variates, and use the partially trained model to forecast all varieties. iTransformers can be trained efficiently and forecast with good generalizability.

---

Increasing lookback length. Previous works have witnessed the phenomenon that the forecasting performance does not necessarily improve with the increase of lookback length on Transformers (Nie et al. 2023; Zeng et al. 2023), which can be attributed to the distracted attention on the growing input. However, the desired performance improvement is generally held on linear forecasts, theoretically supported by statistical methods (Box \& Jenkins, 1968) with enlarged historical information to be utilized. As the working dimensions of attention and feed-forward network are inverted, we evaluate the performance of Transformers and iTransformer in Figure 6 with increased lookback length. The results surprisingly verify the rationality of leveraging MLPs on the temporal dimension such that Transformers can benefit from the extended lookback window for more precise predictions.

- Figure 6: Forecasting performance with the lookback length $T \in\{48,96,192,336,720\}$ and fixed prediction length $S=96$. While the performance of Transformer-based forecasters does not necessarily benefit from the increased lookback length, the inverted framework empowers the vanilla Transformer and its variants with improved performance on the enlarged lookback window.

### 4.3 Model Analysis

Ablation study. To verify the rational business of Transformer components, we provide detailed ablations covering both replacing components (Replace) and removing components (w/o) experiments. The results are listed in Table 3. iTransformer that utilizes attention on the variate dimension and feed-forward on the temporal dimension generally achieves the best performance. Notably, the performance of vanilla Transformer (the third row) performs the worst among these designs, revealing the potential risks of the conventional architecture, which we describe in detail in Appendix E. 3 .

- Table 3: Ablations on iTransformer. We replace different components on the respective dimension to learn multivariate correlations (Variate) and series representations (Temporal), in addition to component removal. The average results of all predicted lengths are listed here.

Analysis of series representations. To further validate the claim that feed-forward networks are more favored to extract the series representations. We conduct representation analysis based on the centered kernel alignment (CKA) similarity (Kornblith et al., 2019). A higher CKA indicates more similar representations. For Transformer variants and iTransformers, we calculate the CKA between the output features of the first and the last block. Notably, previous works have demonstrated that time series forecasting, as a low-level generative task, prefers the higher CKA similarity (Wu et al., 2023, Dong et al., 2023) for the better performance. As shown in Figure 7, a clear division line is exhibited, implying that iTransformers have learned more appropriate series representations by inverting the dimension and thus achieve more accurate predictions. The results also advocate inverting Transformer deserves a fundamental renovation of the forecasting backbone.

---

Analysis of multivariate correlations. By assigning the duty of multivariate correlation to the attention mechanism, the learned map enjoys enhanced interpretability. We present the case visualization on series from Solar-Energy in Figure 7, which has distinct correlations in the lookback and future windows. It can be observed that in the shallow attention layer, the learned map shares lots of similarities to the correlations of raw input series. As it dives into deeper layers, the learned map become gradually alike to the correlations of future series, which validates the inverted operation empowers interpretable attention for correlating, and the processes of encoding the past and decoding for the future are essentially conducted in series representations during feed-forwarding.

- Figure 7: Analysis of series representations and multivariate correlations. Left: MSE and CKA similarity of representations comparison between Transformers and iTransformers. A higher CKA similarity indicates more favored representations for accurate predictions. Right: A case visualization of multivariate correlations of raw time series and the learned score maps by inverted self-attention.

---

Efficient training strategy Due to the quadratic complexity of self-attention, it can be overwhelming for training on numerous variates, which is very common in real-world scenarios. In addition to efficient attention mechanisms, we propose a novel training strategy for high-dimensional multivariate series by taking advantage of previously demonstrated variate generation capability. Concretely, we randomly choose part of the variates in each batch and only train the model with selected variates. Since the number of variate channels is flexible because of our inverting, the model can predict all the variates for predictions. As shown in Figure 8, the performance of our proposed strategy is still comparable with full-variate training, while the memory footprint can be reduced significantly.

- Figure 8: Analysis of the efficient training strategy. While the performance (left) remains stable on partially trained variates of each batch with different sampled ratios, the memory footprint (right) can be cut off greatly. We provide the comprehensive model efficiency analysis in Appendix D

## 5 CONCLUSION AND FUTURE WORK

Considering the characteristics of multivariate time series, we propose iTransformer that inverts the structure of Transformer without modifying any native modules. iTransformer regards independent series as variate tokens to capture multivariate correlations by attention and utilize layer normalization and feed-forward networks to learn series representations. Experimentally, iTransformer achieves state-of-the-art performance and exhibits remarkable framework generality supported by promising analysis. In the future, we will explore large-scale pre-training and more time series analysis tasks.

## A Implementation Details

## A. 1 DATASET DESCRIPTIONS

We conduct experiments on 7 real-world datasets to evaluate the performance of the proposed iTransformer including (1) ETT (Li et al. 2021) contains 7 factors of electricity transformer from July 2016 to July 2018. There are four subsets where ETTh1 and ETTh2 are recorded every hour, and ETTm1 and ETTm2 are recorded every 15 minutes. (2) Exchange (Wu et al., 2021) collects the panel data of daily exchange rates from 8 countries from 1990 to 2016. (3) Weather (Wu et al., 2021) includes 21 meteorological factors collected every 10 minutes from the Weather Station of the Max Planck Biogeochemistry Institute in 2020. (4) ECL (Wu et al., 2021) records the hourly electricity consumption data of 321 clients. (5) Traffic (Wu et al., 2021) collects hourly road occupancy rates measured by 862 sensors of San Francisco Bay area freeways from January 2015 to December 2016. (6) Solar-Energy (Lai et al., 2018) records the solar power production of 137 PV plants in 2006, which are sampled every 10 minutes. (7) PEMS contains the public traffic network data in California collected by 5-minute windows. We use the same four public subsets (PEMS03, PEMS04, PEMS07, PEMS08) adopted in SCINet (Liu et al., 2022a).

---

Apart from the public datasets widely used as forecasting benchmarks, we also collect a set of Market datasets of a real-world application, which records the minute-sampled server load of Alipay online transactions between January 30th, 2023, and April 9th, 2023 with the number of variates varied from 285 to 759. It includes 6 sub-datasets, which are divided according to diverse transaction domains.

---

We follow the same data processing and train-validation-test set split protocol used in TimesNet ( Wu et al. 2023), where the train, validation, and test datasets are strictly divided according to chronological order to make sure there are no data leakage issues. As for the forecasting settings, we fix the length of the lookback series as 96 in ETT, Weather, ECL, Solar-Energy, PEMS, and Traffic, and the prediction length varies in $\{96,192,336,720\}$. For the PEMS dataset, the prediction length varies in $\{12,24,36,48\}$, which is the same as SCINet, the previous state-of-the-art on this dataset. For the Market dataset, the lookback contains the past one day observations with 144 time points and the forecasting length varies in $\{12,24,72,144\}$. The details of datasets are provided in Table 4

- Table 4: Detailed dataset descriptions. Dim denotes the variate number of each dataset. Dataset Size denotes the total number of time points in (Train, Validation, Test) split respectively. Prediction Length denotes the future time points to be predicted and four prediction settings are included in each dataset. Frequency denotes the sampling interval of time points.

---

All the experiments are implemented in PyTorch (Paszke et al., 2019) and conducted on a single NVIDIA P100 16GB GPU. We utilize ADAM (Kingma \& Ba, 2015) with an initial learning rate in $\left\{10^{-3}, 5 \times 10^{-4}, 10^{-4}\right\}$ and L2 loss for the model optimization. The batch size is uniformly set to 32 and the number of training epochs is fixed to 10 . We set the number of inverted Transformer blocks in our proposed model $L \in\{2,3,4\}$. The dimension of series representations $D$ is set from $\{256,512\}$. All the compared baseline models that we reproduced are implemented based on the benchmark of TimesNet (Wu et al. 2023) Repository, which is fairly built on the configurations provided by each model's original paper or official code. We provide the pseudo-code of iTransformer in Algorithm 1 We also report the standard deviation of iTransformer performance under five runs with different random seeds in Table 5, which exhibits that the performance of iTransformer is stable.


## B Ablation Studies

To elaborate on the rational business of Transformer components, we conduct detailed ablations covering replacing components (Replace) and removing components (w/o). Since the average results are listed in Table 3 due to the paper limit, we provide detailed results and analysis here.

---

As shown in Table 6, among various architectural designs, iTransformer generally exhibits superior performance, which learns multivariate correlations by self-attention and encodes series representations by FFN. Nevertheless, the arrangement of the vanilla Transformer can lead to degenerated performance, indicating the misuse of Transformer components on the time series modality. Based on the relatively poor results of the second (both attentions) and the third (the vanilla Transformer) designs, one of the reasons for that may lie in the attention module over the temporal tokens of the lagged time series, which we elaborate more with the datasets support in Section E. 3 .

- Table 6: Full results of the ablation on iTransformer. We apply different components on the respective dimension to learn multivariate correlations (Variate) and series representations (Temporal), in addition to removing the specific component of Transformer.

---

It is also notable that applying FFN on both dimensions can also lead to fair performance on datasets with small variate numbers (such as Weather with 21 variates). Still, with the increasing of variate numbers in challenging multivariate forecasting tasks, the importance of capturing multivariate correlations is ever more highlighted. We note that the heterogeneity of variates can be hardly considered by the vanilla Transformer. During embedding, the variates are projected into indistinguishable channels, which ignores the inconsistent physical measurements and thus fails to maintain the independence of variates, let alone capture and utilize the multivariate correlation. Consequently, by incorporating the advanced attention module for the variate correlating, the first (iTransformer) and the fifth (attention on variates) designs perform more effectively in challenging multivariate datasets.

---

In a nutshell, both temporal dependencies and multivariate correlations are of importance for multivariate time series forecasting. The proposed iTransformer employing the self-attention module to disentangle the correlations between variate tokens proves to be more powerful and interpretable than feed-forward networks, thereby further boosting the performance on challenging multivariate datasets and enhancing the model capacity.

## C Hyperparameter Sensitivity

We evaluate the hyperparameter sensitivity of iTransformer with respect to the following factors: the learning rate $l r$, the number of Transformer blocks $L$, and the hidden dimension $D$ of variate tokens. The results are shown in Figure 9 . We find that the learning rate, as the most common influencing factor, should be carefully selected when the number of variates is large (ECL, Traffic). The block number and hidden dimension are not essentially favored to be as large as possible in iTransformer.

Figure 9: Hyperparameter sensitivity with respect to the learning rate, the number of Transformer blocks, and the hidden dimension of variate tokens. The results are recorded with the lookback window length $T=96$ and the forecast window length $S=96$.

## D ModEl EfFICIENCY

We comprehensively compare the forecasting performance, training speed, and memory footprint of the following models: iTransformer, iTransformer with our efficient training strategy and iTransformer with the efficient flow attention module (Wu et al., 2022); linear models: DLinear (Zeng et al. 2023) and TiDE (Das et al., 2023); Transformers: Transformer (Vaswani et al., 2017), PatchTST (Nie et al., 2023), and Crossformer (Zhang \& Yan, 2023). The results are recorded with the official model configuration and the same batch size. In Figure 10, we compare the efficiency under two representative datasets ( 21 variates in Weather and 862 in Traffic) with 96 time steps for lookback.

---

In a nutshell, the efficiency of iTransformer exceeds other Transformers in datasets with a relatively small number of variates (Weather). In datasets with numerous variates (Traffic), the memory footprints are basically the same as Transformers variates, but iTransformer can be trained faster. Based on the complexity of $\mathcal{O}\left(N^{2}\right)$ of the attention module, where $N$ is the number of tokens, Transformer surpasses iTransformer on efficiency in this case because of $N=96$ for the temporal token and $N=862$ for the variate token. Meanwhile, iTransformer achieves better performance on numerous variates, since the multivariate correlations can be explicitly utilized. By adopting a linear-complexity attention (Wu et al. 2022) or the proposed efficient training strategy as mentioned in Figure 8 (trained on $20 \%$ variates and forecast all variates), iTransformer can enjoy a comparable speed and memory footprint with linear models. Also, the two strategies can be adopted together.

## E Showcases

## E. 1 Visualization of Multivariate Correlations

By using the attention mechanism on variate tokens, the resulting learned map becomes more interpretable. To present an intuitive understanding of the multivariate correlations, we provide three randomly chosen case visualizations of the time series from Solar-Energy in Figure 11 We provide the Pearson Correlation coefficients of each variate of the raw series by the following equation:

- Figure 11: Multivariate correlations of the lookback series and future series and the learned score maps by inverted self-attention of different layers. Cases all come from the Solar-Energy dataset.

$$
\rho_{x y}=\frac{\sum_{i}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum_{i}\left(x_{i}-\bar{x}\right)^{2}} \sqrt{\sum_{i}\left(y_{i}-\bar{y}\right)^{2}}},
$$

where $x_{i}, y_{i} \in \mathbb{R}$ run through all time points of the paired variates to be correlated. All the cases have distinct multivariate correlations in the lookback and forecast window because the dataset exhibits obvious seasonal changes in the daytime and night. On the second row of each case, we provide the learned pre-Softmax maps of the self-attention module in both the first and the last layers. As we observe in the shallow attention layer (left), we find that the learned map is similar to the correlations of the raw lookback series. As we go deeper into the layers (right), the learned map gradually becomes more similar to the correlations of the future series to be predicted. This demonstrates that the inverted operation allows for interpretable attention in correlating, and that encoding of the past and decoding for the future are conducted through series representations during layer stacking.

---

We present another interesting observation in Figure 12 to show that the attention module of iTransformer has enhanced interpretability. We provide randomly chosen multivariate time series from Market. In this dataset, each variate represents the monitored values of a service interface of a kind, and the service can be further grouped into refined application categories. We divide these variates into corresponding applications (as listed on the top bar App), such that adjacent variates belong to the same application and we reveal the application index by the top bar.

- Figure 12: Visualization of the variates from the Market dataset and the learned multivariate correlations. Each variate represents the monitored interface values of an application, and the applications can be further grouped into refined categories. The color bar is shared with Figure 11

---

We visualize the time series of the variates and plot the learned multivariate correlations with the marks of specific correlations between variates. On the one hand, we observe clear partitioning in the multivariate correlations map, indicating the grouping of variates. On the one hand, the marked correlation values can reflect the correlation of the raw series, where the similarity of variates from the same application becomes closer than the pairs from the different groups. Therefore, highly correlated variate will be leveraged for the next interaction and thus benefit for multivariate forecasting.

## E. 2 Visualization of Prediction Results

To provide a clear comparison among different models, we list supplementary prediction showcases of four representative datasets in Figures 13,16 , which are given by the following models: iTransfomrer, PatchTST (Nie et al., 2023), DLinear (Zeng et al., 2023), Crossformer (Zhang \& Yan, 2023), Autoformer (Wu et al., 2021), Transformer (Vaswani et al., 2017). Among the various models, iTransformer predicts the most precise future series variations and exhibits superior performance.

## E. 3 Risks of Embedding Multivariate Points of A Timestamp

As aforementioned, the embedding approach of the previous Transformer fuses multiple variates representing potentially delayed events and distinct physical measurements, which may fail to learn variate-centric representations and result in meaningless attention maps. We provide the visualization case of Traffic (Liu et al. 2022a), which is collected from sensors on Los Angeles city roads in different areas. As shown in Figure 17, we can observe a strong correlation between the multivariate time series of the dataset, while they also exhibit obvious phase offset, which is due to the systematical time lags in the road occupancy that each series describes. Since the sensors are installed in different areas of the highway, an event (such as a traffic jam) can affect road occupancy with different delays.

- Figure 17: Visualization of partial variates of Traffic. We can observe that several series exhibit strong synchronization (such as Sensor 2 and Sensor 4), and there also exist obvious delays and advances between series (such as Sensor 1 and Sensor 2, Sensor 859 and Sensor 861).

---

Besides, we observe the significantly declined performance on the second and third designs of Traffic in Table 6, which apply attention to temporal tokens. In our opinion, capturing temporal dependencies by attention is not a big problem. But it is based on the fact that the time points of each timestamp essentially reflect the same event to enclose a semantic representation. Since there are inherent delays between the time points, the performance can degrade a lot because of the meaningless attention map, unless the model has an enlarged respective field to learn about the decay or causal process.

---

Other risks can be aroused from the distinct variate measurements, such as organizing together different meteorological indicators (the temperature and rainfall) in the Weather dataset (Wu et al. 2021), and the quantity and proportion of the same observation in ILI (Wu et al., 2023). Given these potential risks, iTransformer proposes a new paradigm that embeds the whole series as the variate token, which can be more robust to extensive real-world scenarios, such as delayed events, inconsistent measurements, irregular (unevenly spaced) time series, systematical delay of monitors, and the time interval of generating and recording different time series.

## F Full Results

## F. 1 Full Promotion Results

We compare the performance of Transformer and iTransformer on all datasets in Table 7. Consistent and great promotions can be achieved, indicating that the attention and feed-forward network on the inverted dimensions greatly empower Transformers in multivariate time series forecasting, leaving an instructive direction to build up the foundation model of extensive time series data.

- Table 7: Full performance comparison between the vanilla Transformer and the proposed iTransformer. The results are averaged from all four prediction lengths.

## F. 2 Full Framework Generality Results

We apply the proposed inverting framework to Transformer and its variants: Transformer (Vaswani et al., 2017), Reformer (Kitaev et al. 2020), Informer (Li et al., 2021), Flowformer (Wu et al. 2022), Flashformer (Dao et al. 2022). The averaged results are shown in Table 2 due to the limited pages. We provide the supplementary forecasting results in Table 8. The results demonstrate that our iTransformers framework can consistently promote these Transformer variants, and take advantage of the booming efficient attention mechanisms.

- Table 8: Full results of Transformers with our inverted framework. Flashformer means Transformer equipped with the hardware-accelerated FlashAttention (Dao et al., 2022).

## F. 3 Full Results of Variate Generalization

We divide the variates of each dataset into five folders, train models with only $20 \%$ of variates of one folder, and directly forecast all variates without fine-tuning. We adopt two strategies for Transformers to generalize on unseen variates: (1) CI-Transformers (Nie et al., 2023): Channel Independence regards each variate of time series as independent channels, and trains with a shared backbone. During inference, the model predicts variates one by one, but the procedure can be time-consuming. (2) iTransformers: with the flexibility of the attention mechanism that the number of input tokens can be dynamically changeable, the amount of variates as tokens is no longer restricted and thus feasible to vary from training and inference, and can even allow the model to be trained on arbitrary variates.

---

As shown in Table 18 , iTransformers can be naturally trained with $20 \%$ variates and accomplish forecast on all variates with the ability to learn transferable representations.

- Figure 18：Full performance of generalization on unseen variates，comparing the iTransformers with CI－Transfomers. We divide the variates of each dataset into five folders，train with $20 \%$ variates，and use the trained model to forecast all varieties. We plot the averaged results of all five folders. 

## F.4 Full Forecasting Results

The full multivariate forecasting results are provided in the following section due to the space limita－ tion of the main text. We extensively evaluate competitive counterparts on challenging forecasting tasks. Table 9 contains the forecasting results on the four public subsets from PEMS（Liu et al.  2022a）. Table 10 contains the detailed results of all prediction lengths of the nine well－acknowledged forecasting benchmarks. And Table 11 records the Market results for Alipay server load forecasting.  The proposed model achieves comprehensive state－of－the－art in real－world forecasting applications. 

- Table 9：Full results of the PEMS forecasting task. We compare extensive competitive models under different prediction lengths following the setting of SCINet（2022a）. The input length is set to 96 for all baselines. Avg means the average results from all four prediction lengths. 

- Table 10：Full results of the long－term forecasting task. We compare extensive competitive models under different prediction lengths following the setting of TimesNet（2023）. The input sequence length is set to 96 for all baselines. Avg means the average results from all four prediction lengths. 

## G Discussions and Further Improvement

## G.1 Discussions on Architecture-free Methods

Channel Independence (CI) (Nie et al., 2023), regarding variates of time series independently and adopting the shared backbone, have gained increasing popularity in forecasting with performance promotions as an architecture-free method. Recent works (Han et al., 2023; Li et al., 2023) found that while Channel Dependence (CD) benefits from a higher capacity ideally, CI can greatly boost the performance because of sample scarcity, since most of the current forecasting benchmarks are not large enough. We think it is essential to make variates independent, especially when there are potential risks of embedding as mentioned in Appendix E. 3 . inducing the ideal model capacity of CD limited by the excessively localized receptive field. However, the essence of CI, regarding multivariate time series univariately, can lead to time-consuming training and inference and become an obstacle to scalability. Still, multivariate correlations can not be explicitly utilized. Perpendicular to these works, iTransformer repurposes an architecture with the native Transformer modules to tackle the issues.

---

RevIN (Kim et al., 2021) and Stationarization (Liu et al., 2022b) have been widely applied for the distribution shift (non-stationarity) as architecture-free techniques. These works strive to reveal the temporal dependency better. This is accomplished by layer normalization in iTransformer and still leaves further improvement for us to tackle the distribution shift.

## G. 2 Discussions on Linear Forecasters

Linear forecasters have natural advantages in modeling temporal dependencies. The dense weighting (Zeng et al., 2023; Li et al., 2023) can reveal measurement-free relationships among the time points of the same variate. More advanced linear forecasters focus on structural point-wise modeling (Oreshkin et al., 2019; Liu et al., 2022, 2023). By contrast, iTransformer is particularly good at forecasting high-dimensional time series (numerous variates with complicated correlations, which can be common and realistic for practitioners in real forecasting applications). For variate correlating, the embedding keeps the variate independent and the attention module can be applied to dig it out. Under univariate scenarios, iTransformer actually becomes a stackable linear forecaster (attention degradation), which leaves further enhancement to exploit the temporal dependency better.

## G. 3 Discussions on Transformers

We emphasize that iTransformer actually proposes a new perspective to think about the multivariate time series modality, specifically, how to consider the variates and the tokenization. We list several representatives in Figure 19 . Transformer treats time series as the natural language but the timealigned embedding may bring about risks in multi-dimensional series. The problem can be alleviated by expanding the receptive field. Although it is believed that Patching (Zhang \& Yan, 2023, Nie et al. 2023) can be more fine-grained, it also brings higher computational complexity and the potential interaction noise between time-unaligned patches. If the current embedding (implemented by MLP) is enhanced with more inductive bias (such as TCN), it may handle more robust cases with the variate token paradigm and enjoy the flexibility of Transformer with changeable numbers of tokens.

---

We believe the capability and scalability of Transformer have stood the test by extensive fields, but there is still improvement room to elaborately design components based on the inverted architecture, such as efficient attention for multivariate correlation, structural temporal dependency modeling under distribution shift, fine-grained variate tokenization and well-designed embedding mechanisms.

Figure 19: Tokenizations for multivariate time series modality of representative Transformers.

