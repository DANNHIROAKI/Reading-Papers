### Time-LLM: Time Series Forecasting by Reprogramming Large Language Models 


## Abstract

Time series forecasting holds significant importance in many real-world dynamic systems and has been extensively studied. Unlike natural language process (NLP) and computer vision (CV), where a single large model can tackle multiple tasks, models for time series forecasting are often specialized, necessitating distinct designs for different tasks and applications. While pre-trained foundation models have made impressive strides in NLP and CV, their development in time series domains has been constrained by data sparsity. Recent studies have revealed that large language models (LLMs) possess robust pattern recognition and reasoning abilities over complex sequences of tokens. However, the challenge remains in effectively aligning the modalities of time series data and natural language to leverage these capabilities. In this work, we present Time-LLM, a reprogramming framework to repurpose LLMs for general time series forecasting with the backbone language models kept intact. We begin by reprogramming the input time series with text prototypes before feeding it into the frozen LLM to align the two modalities. To augment the LLM's ability to reason with time series data, we propose Prompt-as-Prefix ( PaP ), which enriches the input context and directs the transformation of reprogrammed input patches. The transformed time series patches from the LLM are finally projected to obtain the forecasts. Our comprehensive evaluations demonstrate that Time-LLM is a powerful time series learner that outperforms state-of-the-art, specialized forecasting models. Moreover, Time-LLM excels in both few-shot and zero-shot learning scenarios. The code is made available athttps://github.com/KimMeen/Time-LLM


# 1 Introduction

Time series forecasting is a critical capability across many real-world dynamic systems (Jin et al. 2023a), with applications ranging from demand planning (Leonard 2001) and inventory optimization (Li et al. 2022) to energy load forecasting (Liu et al., 2023a) and climate modeling (Schneider \& Dickinson, 1974). Each time series forecasting task typically requires extensive domain expertise and task-specific model designs. This stands in stark contrast to foundation language models like GPT-3 (Brown et al. 2020), GPT-4 (OpenAI, 2023), Llama (Touvron et al., 2023), inter alia, which can perform well on a diverse range of NLP tasks in a few-shot or even zero-shot setting.

---

Pre-trained foundation models, such as large language models (LLMs), have driven rapid progress in computer vision (CV) and natural language processing (NLP). While time series modeling has not benefited from the same significant breakthroughs, LLMs' impressive capabilities have inspired their application to time series forecasting (Jin et al., 2023b). Several desiderata exist for leveraging LLMs to advance forecasting techniques: Generalizability. LLMs have demonstrated a remarkable capability for few-shot and zero-shot transfer learning (Brown et al., 2020). This suggests their potential for generalizable forecasting across domains without requiring per-task retraining from scratch. In contrast, current forecasting methods are often rigidly specialized by domain. Data efficiency. By leveraging pre-trained knowledge, LLMs have shown the ability to perform new tasks with only a few examples. This data efficiency could enable forecasting for settings where historical data is limited. In contrast, current methods typically require abundant in-domain data. Reasoning. LLMs exhibit sophisticated reasoning and pattern recognition capabilities (Mirchandani et al. 2023; Wang et al., 2023; Chu et al., 2023). Harnessing these skills could allow making highly precise forecasts by leveraging learned higher-level concepts. Existing non-LLM methods are largely statistical without much innate reasoning. Multimodal knowledge. As LLM architectures and training techniques improve, they gain more diverse knowledge across modalities like vision, speech, and text (Ma et al. 2023). Tapping into this knowledge could enable synergistic forecasting that fuses different data types. Conventional tools lack ways to jointly leverage multiple knowledge bases. Easy optimization. LLMs are trained once on massive computing and then can be applied to forecasting tasks without learning from scratch. Optimizing existing forecasting models often requires significant architecture search and hyperparameter tuning (Zhou et al., 2023b). In summary, LLMs offer a promising path to make time series forecasting more general, efficient, synergistic, and accessible compared to current specialized modeling paradigms. Thus, adapting these powerful models for time series data can unlock significant untapped potential.

---

The realization of the above benefits hinges on the effective alignment of the modalities of time series data and natural language. However, this is a challenging task as LLMs operate on discrete tokens, while time series data is inherently continuous. Furthermore, the knowledge and reasoning capabilities to interpret time series patterns are not naturally present within LLMs' pre-training. Therefore, it remains an open challenge to unlock the knowledge within LLMs in activating their ability for general time series forecasting in a way that is accurate, data-efficient, and task-agnostic.

---

In this work, we propose TIME-LLM, a reprogramming framework to adapt large language models for time series forecasting while keeping the backbone model intact. The core idea is to reprogram the input time series into text prototype representations that are more naturally suited to language models' capabilities. To further augment the model's reasoning about time series concepts, we introduce Prompt-as-Prefix (PaP), a novel idea in enriching the input time series with additional context and providing task instructions in the modality of natural language. This provides declarative guidance about desired transformations to apply to the reprogrammed input. The output of the language model is then projected to generate time series forecasts. Our comprehensive evaluation demonstrates that large language models can act as effective few-shot and zero-shot time series learners when adopted through this reprogramming approach, outperforming specialized forecasting models. By leveraging LLMs' reasoning capability while keeping the models intact, our work points the way toward multimodal foundation models that can excel on both language and sequential data tasks. Our proposed reprogramming framework offers an extensible paradigm for imbuing large models with new capabilities beyond their original pre-training. Our main contributions in this work can be summarized as follows:

- We introduce a novel concept of reprogramming large language models for time series forecasting without altering the pre-trained backbone model. In doing so, we show that forecasting can be cast as yet another "language" task that can be effectively tackled by an off-the-shelf LLM.

- We propose a new framework, Time-LLM, which encompasses reprogramming the input time series into text prototype representations that are more natural for the LLM, and augmenting the input context with declarative prompts (e.g., domain expert knowledge and task instructions) to guide LLM reasoning. Our technique points towards multimodal foundation models excelling in both language and time series.

- Time-LLM consistently exceeds state-of-the-art performance in mainstream forecasting tasks, especially in few-shot and zero-shot scenarios. Moreover, this superior performance is achieved while maintaining excellent model reprogramming efficiency. Thus, our research is a concrete step in unleashing LLMs' untapped potential for time series and perhaps other sequential data.


# 2 Related Work

Task-specific Learning. Most time series forecasting models are crafted for specific tasks and domains (e.g., traffic prediction), and trained end-to-end on small-scale data. An illustration is in Fig. 1 a). For example, ARIMA models are designed for univariate time series forecasting (Box et al., 2015), LSTM networks are tailored for sequence modeling (Hochreiter \& Schmidhuber 1997), and temporal convolutional networks (Bai et al. 2018) and transformers (Wen et al., 2023) are developed for handling longer temporal dependencies. While achieving good performance on narrow tasks, these models lack versatility and generalizability to diverse time series data.

- Figure 1: Schematic illustration of reprogramming large language models (LLMs) in comparison of (a) task-specific learning and (b) model fine-tuning. Our proposal investigates and demonstrates (c) how to effectively reprogram open-sourced LLMs as powerful time series learners where welldeveloped time series pre-trained models are not readily available.

---

In-modality Adaptation. Relevant research in CV and NLP has demonstrated the effectiveness of pre-trained models that can be fine-tuned for various downstream tasks without the need for costly training from scratch (Devlin et al., 2018; Brown et al., 2020; Touvron et al., 2023). Inspired by these successes, recent studies have focused on the development of time series pre-trained models (TSPTMs). The first step among them involves time series pre-training using different strategies like supervised (Fawaz et al., 2018) or self-supervised learning (Zhang et al. 2022b; Deldari et al. 2022, Zhang et al., 2023). This allows the model to learn representing various input time series. Once pretrained, it can be fine-tuned on similar domains to learn how to perform specific tasks (Tang et al., 2022). An example is in Fig. 1(b). The development of TSPTMs leverages the success of pretraining and fine-tuning in NLP and CV but remains limited on smaller scales due to data sparsity.

---

Cross-modality Adaptation. Building on in-modality adaptation, recent work has further explored transferring knowledge from powerful pre-trained foundations models in NLP and CV to time series modeling, through techniques such as multimodal fine-tuning (Yin et al., 2023) and model reprogramming (Chen, 2022). Our approach aligns with this category; however, there is limited pertinent research available on time series. An example is Voice2Series (Yang et al., 2021), which adapts an acoustic model (AM) from speech recognition to time series classification by editing a time series into a format suitable for the AM. Recently, Chang et al. (2023) proposes LLM4TS for time series forecasting using LLMs. It designs a two-stage fine-tuning process on the LLM first supervised pre-training on time series, then task-specific fine-tuning. Zhou et al. (2023a) leverages pre-trained language models without altering their self-attention and feedforward layers. This model is fine-tuned and evaluated on various time series analysis tasks and demonstrates comparable or state-of-the-art performance by transferring knowledge from natural language pre-training. Distinct from these approach, we neither edit the input time series directly nor fine-tune the backbone LLM. Instead, as illustrated in Fig. 1.(c), we propose reprogramming time series with the source data modality along with prompting to unleash the potential of LLMs as effective time series machines.

# 3 Methodology

Our model architecture is depicted in Fig. 2. We focus on reprogramming an embedding-visible language foundation model, such as Llama (Touvron et al. 2023) and GPT-2 (Radford et al., 2019), for general time series forecasting without requiring any fine-tuning of the backbone model. Specifically, we consider the following problem: given a sequence of historical observations $\mathbf{X} \in \mathbb{R}^{N \times T}$ consisting of $N$ different 1 -dimensional variables across $T$ time steps, we aim to reprogram a large language model $f(\cdot)$ to understand the input time series and accurately forecast the readings at $H$ future time steps, denoted by $\hat{\mathbf{Y}} \in \mathbb{R}^{N \times H}$, with the overall objective to minimize the mean square errors between the ground truths $\mathbf{Y}$ and predictions, i.e., $\frac{1}{H} \sum_{h=1}^{H}\left\|\hat{\mathbf{Y}}_{h}-\mathbf{Y}_{h}\right\|_{F}^{2}$.

- Figure 2: The model framework of Time-LLM. Given an input time series, we first tokenize and embed it via (1) patching along with a (2) customized embedding layer. (3) These patch embeddings are then reprogrammed with condensed text prototypes to align two modalities. To augment the LLM's reasoning ability, (4) additional prompt prefixes are added to the input to direct the transformation of input patches. (5) The output patches from the LLM are projected to generate the forecasts.

---

Our method encompasses three main components: (1) input transformation, (2) a pre-trained and frozen LLM, and (3) output projection. Initially, a multivariate time series is partitioned into $N$ univariate time series, which are subsequently processed independently (Nie et al. 2023). The $i$-th series is denoted as $\mathbf{X}^{(i)} \in \mathbb{R}^{1 \times T}$, which undergoes normalization, patching, and embedding prior to being reprogrammed with learned text prototypes to align the source and target modalities. Then, we augment the LLM's time series reasoning ability by prompting it together with reprogrammed patches to generate output representations, which are projected to the final forecasts $\hat{\mathbf{Y}}^{(i)} \in \mathbb{R}^{1 \times H}$.

---

We note that only the parameters of the lightweight input transformation and output projection are updated, while the backbone language model is frozen. In contrast to vision-language and other multimodal language models, which usually fine-tune with paired cross-modality data, Time-LLM is directly optimized and becomes readily available with only a small set of time series and a few training epochs, maintaining high efficiency and imposing fewer resource constraints compared to building large domain-specific models from scratch or fine-tuning them. To further reduce memory footprints, various off-the-shelf techniques (e.g., quantization) can be seamlessly integrated for slimming Time-LLM.

## 3.1 Model Structure

Input Embedding. Each input channel $\mathbf{X}^{(i)}$ is first individually normalized to have zero mean and unit standard deviation via reversible instance normalization (RevIN) in mitigating the time series distribution shift (Kim et al., 2021). Then, we divide $\mathbf{X}^{(i)}$ into several consecutive overlapped or non-overlapped patches (Nie et al., 2023) with length $L_{p}$; thus the total number of input patches is $P=\left\lfloor\frac{\left(T-L_{p}\right)}{S}\right\rfloor+2$, where $S$ denotes the horizontal sliding stride. The underlying motivations are two-fold: (1) better preserving local semantic information by aggregating local information into each patch and (2) serving as tokenization to form a compact sequence of input tokens, reducing computational burdens. Given these patches $\mathbf{X}_{P}^{(i)} \in \mathbb{R}^{P \times L_{p}}$, we embed them as $\hat{\mathbf{X}}_{P}^{(i)} \in \mathbb{R}^{P \times d_{m}}$, adopting a simple linear layer as the patch embedder to create dimensions $d_{m}$.

---

Patch Reprogramming. Here we reprogram patch embeddings into the source data representation space to align the modalities of time series and natural language to activate the backbone's time series understanding and reasoning capabilities. A common practice is learning a form of "noise" that, when applied to target input samples, allows the pre-trained source model to produce the desired target outputs without requiring parameter updates. This is technically feasible for bridging data modalities that are identical or similar. Examples include repurposing a vision model to work with cross-domain images (Misra et al., 2023) or reprogramming an acoustic model to handle time series data (Yang et al. 2021). In both cases, there are explicit, learnable transformations between the source and target data, allowing for the direct editing of input samples. However, time series can neither be directly edited nor described losslessly in natural language, posing significant challenges to directly bootstrap the LLM for understanding time series without resource-intensive fine-tuning.

---

To close this gap, we propose reprogramming $\hat{\mathbf{X}}_{P}^{(i)}$ using pre-trained word embeddings $\mathbf{E} \in \mathbb{R}^{V \times D}$ in the backbone, where $V$ is the vocabulary size. Nevertheless, there is no prior knowledge indicating which source tokens are directly relevant. Thus, simply leveraging $\mathbf{E}$ will result in large and potentially dense reprogramming space. A simple solution is to maintain a small collection of text prototypes by linearly probing $\mathbf{E}$, denoted as $\mathbf{E}^{\prime} \in \mathbb{R}^{V^{\prime} \times D}$, where $V^{\prime} \ll V$. An illustration is in Fig. 3 a). Text prototypes learn connecting language cues, e.g., "short up" (red lines) and "steady down" (blue lines), which are then combined to represent the local patch information (e.g., "short up then down steadily" for characterizing patch 5) without leaving the space where the language model is pre-trained. This approach is efficient and allows for the adaptive selection of relevant source information. To realize this, we employ a multi-head cross-attention layer. Specifically, for each head $k=\{1, \cdots, K\}$, we define query matrices $\mathbf{Q}_{k}^{(i)}=\hat{\mathbf{X}}_{P}^{(i)} \mathbf{W}_{k}^{Q}$, key matrices $\mathbf{K}_{k}^{(i)}=\mathbf{E}^{\prime} \mathbf{W}_{k}^{K}$, and value matrices $\mathbf{V}_{k}^{(i)}=\mathbf{E}^{\prime} \mathbf{W}_{k}^{V}$, where $\mathbf{W}_{k}^{Q} \in \mathbb{R}^{d_{m} \times d}$ and $\mathbf{W}_{k}^{K}, \mathbf{W}_{k}^{V} \in \mathbb{R}^{D \times d}$. Specifically, $D$ is the hidden dimension of the backbone model, and $d=\left\lfloor\frac{d_{m}}{K}\right\rfloor$. Then, we have the operation to reprogram time series patches in each attention head defined as:

$$
\begin{equation*}
\mathbf{Z}_{k}^{(i)}=\operatorname{ATtention}\left(\mathbf{Q}_{k}^{(i)}, \mathbf{K}_{k}^{(i)}, \mathbf{V}_{k}^{(i)}\right)=\operatorname{Softmax}\left(\frac{\mathbf{Q}_{k}^{(i)} \mathbf{K}_{k}^{(i) \top}}{\sqrt{d_{k}}}\right) \mathbf{V}_{k}^{(i)} . \tag{1}
\end{equation*}
$$

By aggregating each $\mathbf{Z}_{k}^{(i)} \in \mathbb{R}^{P \times d}$ in every head, we obtain $\mathbf{Z}^{(i)} \in \mathbb{R}^{P \times d_{m}}$. This is then linearly projected to align the hidden dimensions with the backbone model, yielding $\mathbf{O}^{(i)} \in \mathbb{R}^{P \times D}$.

---

Prompt-as-Prefix. Prompting serves as a straightforward yet effective approach task-specific activation of LLMs (Yin et al., 2023). However, the direct translation of time series into natural language presents considerable challenges, hindering both the creation of instructionfollowing datasets and the effective utilization of on-thefly prompting without performance compromise (Xue \& Salim, 2022). Recent advancements indicate that other data modalities, such as images, can be seamlessly integrated as the prefixes of prompts, thereby facilitating effective reasoning based on these inputs (Tsimpoukelli et al., 2021). Motivated by these findings, and to render our approach directly applicable to real-world time series, we pose an alternative question: can prompts act as prefixes to enrich the input context and guide the transformation of reprogrammed time series patches? We term this concept as Prompt-as-Prefix ( PaP ) and observe that it significantly enhances the LLM's adaptability to downstream tasks while complementing patch reprogramming (See Sec. 4.5 later).

---

An illustration of the two prompting approaches is in Fig. 3(b). In Patch-as-Prefix, a language model is prompted to predict subsequent values in a time series, articulated in natural language. This approach encounters certain constraints: (1) language models typically exhibit reduced sensitivity in processing high-precision numerals without the aid of external tools, thereby presenting substantial challenges in accurately addressing practical forecasting tasks over long horizons; (2) intricate, customized post-processing is required for different language models, given that they are pre-trained on diverse corpora and may employ different tokenization types in generating high-precision numerals with precision and efficiency. This results in forecasts being represented in disparate natural language formats, such as [' 0 ', '., ' 6 ', ' 1 '] and [' 0 ', ' $\because$, ' 61 '], to denote the decimal 0.61 .

---

Prompt-as-Prefix, on the other hand, tactfully avoids these constraints. In practice, we identify three pivotal components for constructing effective prompts: (1) dataset context, (2) task instruction, and (3) input statistics. A prompt example is in Fig. 4. The dataset context furnishes the LLM with essential background information concerning the input time series, which often exhibits distinct characteristics across various domains. Task instruction serves as a crucial guide for the LLM in the transformation of patch embeddings for specific tasks. We also enrich the input time series with additional crucial statistics, such as trends and lags, to facilitate pattern recognition and reasoning.

---

Output Projection. Upon packing and feedforwarding the prompt and patch embeddings $\mathbf{O}^{(i)}$ through the frozen LLM as shown in Fig. 2, we discard the prefixal part and obtain the output representations. Following this, we flatten and linear project them to derive the final forecasts $\hat{\mathbf{Y}}^{(i)}$.

# 4 Main Results

TIME-LLM consistently outperforms state-of-the-art forecasting methods by large margins across multiple benchmarks and settings, especially in few-shot and zero-shot scenarios. We compared our approach against a broad collection of up-to-date models, including a recent study that fine-tunes language model for time series analysis (Zhou et al., 2023a). To ensure a fair comparison, we adhere to the experimental configurations in (Wu et al., 2023) across all baselines with a unified evaluation pipeling ${ }^{11}$. We use Llama-7B (Touvron et al. 2023) as the default backbone unless stated otherwise.

---

Baselines. We compare with the SOTA time series models, and we cite their performance from (Zhou et al., 2023a) if applicable. Our baselines include a series of Transformer-based methods: PatchTST (2023), ESTformer (2022), Non-Stationary Transformer (2022), FEDformer (2022), Autoformer (2021), Informer (2021), and Reformer (2020). We also select a set of recent competitive models, including GPT4TS (2023a), LLMTime (2023), DLinear (2023), TimesNet (2023), and LightTS (2022a). In short-term forecasting, we further compare our model with N-HiTS (2023b) and N-BEATS (2020). More details are in Appendix A.

## 4.1 LONG-TERM FORECASTING

Setups. We evaluate on ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity (ECL), Traffic, and ILI, which have been extensively adopted for benchmarking long-term forecasting models (Wu et al., 2023). Details of the implementation and datasets can be found in Appendix B. The input time series length $T$ is set as 512 , and we use four different prediction horizons $H \in\{96,192,336,720\}$. The evaluation metrics include mean square error (MSE) and mean absolute error (MAE).

---

Results. Our brief results are shown in Tab. 1, where TIME-LLM outperforms all baselines in most cases and significantly so to the majority of them. The comparison with GPT4TS (Zhou et al., 2023a) is particularly noteworthy. GPT4TS is a very recent work that involves fine-tuning on the backbone language model. We note average performance gains of $\mathbf{1 2 \%}$ and $\mathbf{2 0 \%}$ over GPT4TS and TimesNet, respectively. When compared with the SOTA task-specific Transformer model PatchTST, by reprogramming the smallest Llama, Time-LLM realizes an average MSE reduction of $1.4 \%$. Relative to the other models, e.g., DLinear, our improvements are also pronounced, exceeding $\mathbf{1 2 \%}$.

- Table 1: Long-term forecasting results. All results are averaged from four different forecasting horizons: $H \in$ $\{24,36,48,60\}$ for ILI and $\{96,192,336,720\}$ for the others. A lower value indicates better performance. Red: the best, Blue: the second best. Our full results are in Appendix D

## 4.2 SHORT-TERM FORECASTING

Setups. We choose the M4 benchmark (Makridakis et al. 2018) as the testbed, which contains a collection of marketing data in different sampling frequencies. More details are provided in Appendix B . The prediction horizons in this case are relatively small and in $[6,48]$. The input lengths are twice as prediction horizons. The evaluation metrics are symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MSAE), and overall weighted average (OWA).

---

Results. Our brief results with unified seeds across all methods are in Tab. 2. Time-LLM consistently surpasses all baselines, outperforming GPT4TS by $\mathbf{8 . 7 \%}$. TIME-LLM remains competitive even when compared with the SOTA model, N-HiTS (Challu et al., 2023b) , w.r.t. MASE and OWA.

- Table 2: Short-term time series forecasting results on M4. The forecasting horizons are in $[6,48]$ and the three rows provided are weighted averaged from all datasets under different sampling intervals. A lower value indicates better performance. Red: the best, Blue: the second best. More results are in Appendix D.

## 4.3 Few-Shot Forecasting

Setups. LLMs have recently demonstrated remarkable few-shot learning capabilities (Liu et al. 2023b). In this section, we assess whether our reprogrammed LLM retains this ability in forecasting tasks. We adhere to the setups in (Zhou et al., 2023a) for fair comparisons, and we evaluate on scenarios with limited training data (i.e., $\leq$ first $10 \%$ training time steps).

---

Results. Our brief $10 \%$ and 5\% few-shot learning results are in Tab. 3 and Tab. 4 respectively. TimeLLM remarkably excels over all baseline methods, and we attribute this to the successful knowledge activation in our reprogrammed LLM. Interestingly, both our approach and GPT4TS consistently surpass other competitive baselines, further underscoring the potential prowess of language models as proficient time series machines.

- Table 3: Few-shot learning on $10 \%$ training data. We use the same protocol in Tab. 1 All results are averaged from four different forecasting horizons: $H \in\{96,192,336,720\}$. Our full results are in Appendix E

- Table 4: Few-shot learning on $5 \%$ training data. We use the same protocol in Tab. 1 All results are averaged from four different forecasting horizons: $H \in\{96,192,336,720\}$. Our full results are in Appendix E

---

In the realm of $10 \%$ few-shot learning, our methodology realizes a $\mathbf{5 \%}$ MSE reduction in comparison to GPT4TS, without necessitating any fine-tuning on the LLM. In relation to recent SOTA models such as PatchTST, DLinear, and TimesNet, our average enhancements surpass $\mathbf{8 \%}, \mathbf{1 2 \%}$, and $\mathbf{3 3 \%}$ w.r.t. MSE. Analogous trends are discernible in the $5 \%$ few-shot learning scenarios, where our average advancement over GPT4TS exceeds 5\%. When compared with PatchTST, DLinear, and TimesNet, TIME-LLM manifests a striking average improvement of over $\mathbf{2 0 \%}$.

## 4.4 Zero-Shot Forecasting

Setups. Beyond few-shot learning, LLMs hold potential as effective zero-shot reasoners (Kojima et al. 2022). In this section, we evaluate the zero-shot learning capabilities of the reprogrammed LLM within the framework of cross-domain adaptation. Specifically, we examine how well a model performs on a dataset \& when it is optimized on another dataset $\boldsymbol{\phi}$, where the model has not encountered any data samples from the dataset \&. Similar to few-shot learning, we use long-term forecasting protocol and evaluate on various cross-domain scenarios utilizing the ETT datasets.

---

Results. Our brief results are in Tab. 5. TIME-LLM consistently outperforms the most competitive baselines by a large margin, over $\mathbf{1 4 . 2 \%}$ w.r.t. the second-best in MSE reduction. Considering the few-shot results, we observe that reprogramming an LLM tends to yield significantly better results in data scarcity scenarios. For example, our overall error reductions w.r.t. GPT4TS in $10 \%$ few-shot forecasting, $5 \%$ few-shot forecasting, and zero-shot forecasting are increasing gradually: 7.7\%, $\mathbf{8 . 4 \%}$, and $\mathbf{2 2 \%}$. Even when benchmarked against LLMTime, the most recent approach in this field, with the backbone LLM of comparable size (7B), TIME-LLM shows a substantial improvement exceeding $\mathbf{7 5 \%}$. We attribute this to our approach being better at activating the LLM's knowledge transfer and reasoning capabilities in a resource-efficient manner when performing time series tasks.

## 4.5 Model Analysis

Language Model Variants. We compare two representative backbones with varying capacities (A.1-4 in Tab. 6). Our results indicate that the scaling law retain after the LLM reprogramming. We adopt Llama-7B by default in its full capacity, which manifestly outperforms its $1 / 4$ capacity variant (A.2; inclusive of the first 8 Transformer layers) by $\mathbf{1 4 . 5 \%}$. An average MSE reduction of $\mathbf{1 4 . 7 \%}$ is observed over GPT-2 (A.3), which slightly outperforms its variant GPT-2 (6) (A.4) by $2.7 \%$.

---

Cross-modality Alignment. Our results in Tab. 6 indicate that ablating either patch reprogramming or Prompt-as-Prefix hurts knowledge transfer in reprogramming the LLM for effective time series forecasting. In the absence of representation alignment (B.1), we observe a notable average performance degradation of $\mathbf{9 . 2 \%}$, which becomes more pronounced (exceeding $\mathbf{1 7 \%}$ ) in few-shot tasks. In Time-LLM, the act of prompting stands as a pivotal element in harnessing the LLM's capacity for understanding the inputs and tasks. Ablation of this component (B.2) results in over $\mathbf{8 \%}$ and $\mathbf{1 9 \%}$ degradation in standard and few-shot forecasting tasks, respectively. We find that removing the input statistics (C.1) hurts the most, resulting in an average increase of $\mathbf{1 0 . 2 \%}$ MSE. This is anticipated as external knowledge can be naturally incorporated via prompting to facilitate the learning and inference. Additionally, providing the LLM with clear task instructions and input context (e.g., dataset captioning) is also beneficial (i.e., C. 2 and C.1; eliciting over $\mathbf{7 . 7 \%}$ and $\mathbf{9 . 6 \%}$, respectively).

---

Reprogramming Interpretation. We provide a case study on ETTh1 of reprogramming 48 time series patches with 100 text prototypes in Fig. 5 The top 4 subplots visualize the optimization of reprogramming space from randomly-initialized (a) to welloptimized (d). We find only a small set of prototypes (columns) participated in reprogramming the input patches (rows) in subplot (e). Also, patches undergo different representations through varying combinations of prototypes. This indicates: (1) text prototypes learn to summarize language cues, and a select few are highly relevant for representing information in local time series patches, which we visualize by randomly selecting 10 in subplot (f). Our results suggest a high relevance to the words that describe time series properties (i.e., word sets 1 and 2 ); (2) patches usually have different underlying semantics, necessitating different prototypes to represent.

---

Reprogramming Efficiency. Tab. 7 provides an overall efficiency analysis of TiME-LLM with and without the backbone LLM. Our proposed reprogramming network itself (D.3) is lightweight in activating the LLM's ability for time series forecasting (i.e., fewer than 6.6 million trainable parameters; only around $\mathbf{0 . 2 \%}$ of the total parameters in Llama-7B), and the overall efficiency of TIME-LLM is actually capped by the leveraged backbones (e.g., D. 1 and D.2). This is favorable even compared to the parameter-efficient fine-tuning methods (e.g., QLoRA (Dettmers et al., 2023)) in balancing task performance and efficiency.

# 5 Conclusion and Future Work

TIME-LLM shows promise in adapting frozen large language models for time series forecasting by reprogramming time series data into text prototypes more natural for LLMs and providing natural language guidance via Prompt-as-Prefix to augment reasoning. Evaluations demonstrate the adapted LLMs can outperform specialized expert models, indicating their potential as effective time series machines. Our results also provide a novel insight that time series forecasting can be cast as yet another "language" task that can be tackled by an off-the-shelf LLM to achieve state-of-the-art performance through our Time-LLM framework. Further research should explore optimal reprogramming representations, enrich LLMs with explicit time series knowledge through continued pre-training, and build towards multimodal models with joint reasoning across time series, natural language, and other modalities. Furthermore, applying the reprogramming framework to equip LLMs with broader time series analytical abilities or other new capabilities should also be considered.

# A. More Related Work

Task-specific Learning. We furnish an extension of the related work on task-specific learning, focusing particularly on the most related models to which we made comparisons. Recent works improve Transformer (Vaswani et al. 2017) for time series forecasting by incorporating signal processing principles like patching, exponential smoothing, decomposition, and frequency analysis. For example, PatchTST (Nie et al. 2023) segments time series into patches as input tokens to Transformer. This retains local semantics, reduces computation/memory for attention, and allows longer history. It improves long-term forecast accuracy over other Transformer models. It also achieves excellent performance on self-supervised pretraining and transfer learning. ETSformer (Woo et al. 2022 ) incorporates exponential smoothing principles into Transformer attention to improve accuracy and efficiency. It uses exponential smoothing attention and frequency attention to replace standard self-attention. FEDformer (Zhou et al., 2022) combines Transformer with seasonal-trend decomposition. The decomposition captures the global profile while Transformer captures detailed structures. It also uses frequency enhancement for long-term prediction. This provides better performance and efficiency than the standard Transformer. Autoformer (Wu et al. 2021) uses a decomposition architecture with auto-correlation to enable progressive decomposition capacities for complex series. Auto-correlation is designed based on series periodicity to conduct dependency discovery and representation aggregation. It outperforms self-attention in efficiency and accuracy.

---

Although these methods enhance efficiency and accuracy compared to vanilla Transformer, they are mostly designed and optimized for narrow prediction tasks within specific domains. These models are typically trained end-to-end on small, domain-specific datasets. While achieving strong performance on their target tasks, such specialized models sacrifice versatility and generalizability across the diverse range of time series data encountered in the real world. The narrow focus limits their applicability to new datasets and tasks. To advance time series forecasting, there is a need for more flexible, widely applicable models that can adapt to new data distributions and tasks without extensive retraining. Ideal models would learn robust time series representations that transfer knowledge across domains. Developing such broadly capable forecasting models remains an open challenge. According to our discussions of related previous work, recent studies have begun to explore model versatility through pre-training and architectural innovations. However, further efforts are needed to realize the truly general-purpose forecasting systems that we are advancing in this research.

---

Cross-modality Adaptation. We provide an extended overview of related work in cross-modality adaptation, with a particular focus on recent advancements in model reprogramming for time series and other data modalities. Model reprogramming is a resource-efficient cross-domain learning approach that involves adapting a well-developed, pre-trained model from one domain (source) to address tasks in a different domain (target) without the need for model fine-tuning, even when these domains are significantly distinct, as noted by Chen (2022). In the context of time series data, Voice2Series (Yang et al. 2021) adapts an acoustic model from speech recognition for time series classification by transforming the time series to fit the model and remapping outputs to new labels. Similarly, LLMTime (Gruver et al. 2023) adapts LLMs for zero-shot time series forecasting, focusing on the effective tokenization of input time series for the backbone LLM, which then generates forecasts autoregressively. Diverging from these methods, TIME-LLM does not edit the input time series directly. Instead, it proposes reprogramming time series with the source data modality along with prompting to unleash the full potential of LLMs as versatile forecasters in standard, few-shot, and zero-shot scenarios. Other notable works in this field, mostly in biology, include R2DL (Vinod et al., 2020) and ReproBert (Melnyk et al., 2023), which reprogram amino acids using word embeddings. A key distinction with our patch reprogramming approach is that, unlike the complete set of amino acids, time series patches do not form a complete set. Thus, we propose optimizing a small set of text prototypes and their mapping to time series patches, rather than directly optimizing a large transformation matrix between two complete sets, such as vocabulary and amino acids.

# B Experimental Details

## B. 1 IMPLEMENTATION

We mainly follow the experimental configurations in (Wu et al. 2023) across all baselines within a unified evaluation pipeline in https://github.com/thuml/Time-Series-Libraryfor fair comparisons. We use Llama-7B (Touvron et al., 2023) as the default backbone model unless stated otherwise. All our experiments are repeated three times and we report the averaged results. Our model implementation is on PyTorch (Paszke et al. 2019) with all experiments conducted on NVIDIA A100-80G GPUs. Our detailed model configurations are in Appendix B.4, and our code is made available at https://github.com/KimMeen/Time-LLM

---

Technical Details. We provide additional technical details of Time-LLM in three aspects: (1) the learning of text prototypes, (2) the calculation of trends and lags in time series for use in prompts, and (3) the implementation of the output projection. To identify a small set of text prototypes $\mathbf{E}^{\prime} \in \mathbb{R}^{V^{\prime} \times D}$ from $\mathbf{E} \in \mathbb{R}^{V \times D}$, we learn a matrix $\mathbf{W} \in \mathbb{R}^{V^{\prime} \times V}$ as the intermediary. To describe the overall time series trend in natural language, we calculate the sum of differences between consecutive time steps. A sum greater than 0 indicates an upward trend, while a lesser sum denotes a downward trend. In addition, we calculate the top- 5 lags of the time series, identified by computing the autocorrelation using fast Fourier transformation and selecting the five lags with the highest correlation values. After we pack and feedforward the prompt and patch embeddings $\mathbf{O}^{(i)} \in \mathbb{R}^{P \times D}$ through the frozen LLM, we discard the prefixal part and obtain the output representations, denoted as $\tilde{\mathbf{O}}^{i} \in \mathbb{R}^{P \times D}$. Subsequently, we follow PatchTST (Nie et al. 2023) and flatten $\tilde{\mathbf{O}}^{i}$ into a 1D tensor with the length $P \times D$, which is then linear projected as $\hat{\mathbf{Y}}^{i} \in \mathbb{R}^{H}$.

## B. 2 Dataset Details

Dataset statistics are summarized in Tab. 8 We evaluate the long-term forecasting performance on the well-established eight different benchmarks, including four ETT datasets (Zhou et al., 2021) (i.e., ETTh1, ETTh2, ETTm1, and ETTm2), Weather, Electricity, Traffic, and ILI from (Wu et al.||2023). Furthermore, we evaluate the performance of short-term forecasting on the M4 benchmark (Makridakis et al., 2018) and the quarterly dataset in the M3 benchmark (Makridakis \& Hibon, 2000).

---

The Electricity Transformer Temperature (ETT; An indicator reflective of long-term electric power deployment) benchmark is comprised of two years of data, sourced from two counties in China, and is subdivided into four distinct datasets, each with varying sampling rates: ETTh1 and ETTh2, which are sampled at a 1-hour level, and ETTm1 and ETTm2, which are sampled at a 15-minute level. Each entry within the ETT datasets includes six power load features and a target variable, termed "oil temperature". The Electricity dataset comprises records of electricity consumption from 321 customers, measured at a 1-hour sampling rate. The Weather dataset includes one-year records from 21 meteorological stations located in Germany, with a sampling rate of 10 minutes. The Traffic dataset includes data on the occupancy rates of the freeway system, recorded from 862 sensors across the State of California, with a sampling rate of 1 hour. The influenza-like illness (ILI) dataset contains records of patients experiencing severe influenza with complications.

---

The M4 benchmark comprises 100 K time series, amassed from various domains commonly present in business, financial, and economic forecasting. These time series have been partitioned into six distinctive datasets, each with varying sampling frequencies that range from yearly to hourly. The M3-Quarterly dataset comprises 756 quarterly sampled time series in the M3 benchmark. These series are categorized into five different domains: demographic, micro, macro, industry, and finance.

# B. 3 Evaluation Metrics

For evaluation metrics, we utilize the mean square error (MSE) and mean absolute error (MAE) for long-term forecasting. In terms of the short-term forecasting on M4 benchmark, we adopt the symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MASE), and overall weighted average (OWA) as in N-BEATS (Oreshkin et al. 2020). Note that OWA is a specific metric utilized in the M4 competition. The calculations of these metrics are as follows:

$$
\begin{aligned}
& \operatorname{MSE}=\frac{1}{H} \sum_{h=1}^{T}\left(\mathbf{Y}_{h}-\hat{\mathbf{Y}}_{h}\right)^{2}, \\
& \mathrm{MAE}=\frac{1}{H} \sum_{h=1}^{H}\left|\mathbf{Y}_{h}-\hat{\mathbf{Y}}_{h}\right|, \\
& \text { SMAPE }=\frac{200}{H} \sum_{h=1}^{H} \frac{\left|\mathbf{Y}_{h}-\hat{\mathbf{Y}}_{h}\right|}{\left|\mathbf{Y}_{h}\right|+\left|\hat{\mathbf{Y}}_{h}\right|}, \\
& \text { MAPE }=\frac{100}{H} \sum_{h=1}^{H} \frac{\left|\mathbf{Y}_{h}-\hat{\mathbf{Y}}_{h}\right|}{\left|\mathbf{Y}_{h}\right|}, \\
& \operatorname{MASE}=\frac{1}{H} \sum_{h=1}^{H} \frac{\left|\mathbf{Y}_{h}-\hat{\mathbf{Y}}_{h}\right|}{\frac{1}{H-s} \sum_{j=s+1}^{H}\left|\mathbf{Y}_{j}-\mathbf{Y}_{j-s}\right|}, \\
& \mathrm{OWA}=\frac{1}{2}\left[\frac{\mathrm{SMAPE}}{\mathrm{SMAPE}_{\text {Naïve } 2}}+\frac{\text { MASE }}{\mathrm{MASE}_{\text {Naïve2 }}}\right],
\end{aligned}
$$

where $s$ is the periodicity of the time series data. $H$ denotes the number of data points (i.e., prediction horizon in our cases). $\mathbf{Y}_{h}$ and $\hat{\mathbf{Y}}_{h}$ are the $h$-th ground truth and prediction where $h \in\{1, \cdots, H\}$.

## B. 4 Model Configurations

The configurations of our models, relative to varied tasks and datasets, are consolidated in Tab. 9 By default, the Adam optimizer (Kingma \& Ba, 2015) is employed throughout all experiments. Specifically, the quantity of text prototypes $V^{\prime}$ is held constant at 100 and 1000 for short-term and long-term forecasting tasks, respectively. We utilize the Llama-7B model at full capacity, maintaining the backbone model layers at 32 across all tasks as a standard. The term input length $T$ signifies the number of time steps present in the original input time series data. Patch dimensions $d_{m}$ represent the hidden dimensions of the embedded time series patches prior to reprogramming. Lastly, heads $K$ correlate to the multi-head cross-attention utilized for patch reprogramming. In the four rightmost columns of Tab. 9 , we detail the configurations related to model training.

# C Hyperparameter Sensitivity

We conduct a hyperparameter sensitivity analysis focusing on the four important hyperparameters within TIME-LLM: namely, the number of backbone model layers, the number of text prototypes $V^{\prime}$, the time series input length $T$, and the number of patch reprogramming cross-attention heads $K$. The correlated results can be found in Fig. 6 From our analysis, we derive the following observations: (1) There is a positive correlation between the number of Transformer layers in the backbone LLM and the performance of TIME-LLM, affirming that the scaling law is preserved postLLM reprogramming.; (2) Generally, acquiring more text prototypes enhances performance. We hypothesize that a limited number of prototypes $V^{\prime}$ might induce noise when aggregating language cues, consequently obstructing the efficient learning of highly representative prototypes essential for characterizing the input time series patches; (3) The input time length $T$ exhibits a direct relation with forecasting accuracy, particularly evident when predicting extended horizons. This observation is logical and is in congruence with conventional time series models; (4) Increasing the number of attention heads during the reprogramming of input patches proves to be advantageous.

# D Long-TERM And Short-term Forecasting

## D. 1 LONG-TERM FORECASTING

By solely reprogramming the smallest Llama model while keeping it intact, TimE-LLM attains SOTA performance in $\mathbf{3 6}$ out of 40 instances across eight time series benchmarks. This underscores the considerable potential of LLMs as robust and reliable time series forecasters. Furthermore, we benchmark the proposed method against other well-established baselines in Tab. 11. This comparison includes three notable statistical methods (AutoARIMA, AutoTheta, and AutoETS) (Herzen et al., 2022) and two recent time series models, N-HiTS (Challu et al., 2023b) and N-BEATS (Oreshkin et al. 2020). Remarkably, TiME-LLM secures SOTA performance across all cases, surpassing the second-best results by significant margins of over $\mathbf{2 2 \%}$ and $\mathbf{1 6 \%}$ in terms of MSE and MAE.

- Table 11：Additional comparison with other baselines in long－term forecasting tasks. We set the forecasting horizons $H \in\{24,36,48,60\}$ for ILI and $\{96,192,336,720\}$ for the others. A lower value indicates better performance. Red：the best，Blue：the second best. 

## D.  2 Short－term Forecasting

Our complete results on short－term forecasting are presented in Tab. 12. Time－LLM consistently outperforms the majority of baseline models in most cases. Notably，we surpass GPT4TS by a large margin（e. g. ， $\mathbf{8 . 7 \%}$ overall， $\mathbf{1 3 . 4 \%}$ on M4－Yearly，and an average of $\mathbf{2 1 . 5 \%}$ on M4－Hourly， M4－Daily，and M4－Weekly），as well as TimesNet（e. g. ， $\mathbf{1 0 \%}$ overall， $\mathbf{1 4 . 1 \%}$ on M4－Yearly，and an average of $\mathbf{3 0 . 1 \%}$ on M4－Hourly，M4－Daily，and M4－Weekly）. Compared to the recent state－ of－the－art forecasting models，N－HiTS and PatchTST，TIME－LLM exhibits comparable or superior performances without any parameter updates on the backbone LLM. 

- Table 12：Full short－term time series forecasting results. The forecasting horizons are in $[6,48]$ and the last three rows are weighted averaged from all datasets under different sampling intervals. A lower value indicates better performance. Red：the best，Blue：the second best. 

---

In addition，we conduct a comparative analysis between Time－LLM and the top－performing models on the M3－Quarterly dataset，with the findings presented in Tab.  13 We provide additional metrics， namely MRAE and MAPE，alongside the default SMAPE used in the M3 competition. On this dataset，TIME－LLM attains on－par performance compared to TimesNet and PatchTST，outperform－ ing GPT4TS by substantial margins，achieving reductions of over $\mathbf{2 3 \%}, \mathbf{3 5 \%}$ ，and $\mathbf{2 6 \%}$ in SMAPE， MRAE，and MAPE，respectively. 

- Table 13：Additional short－term time series forecasting results on M3（Quarterly）. The forecasting horizon is 8. A lower value indicates better performance. Red：the best，Blue：the second best. 

# E Few－Shot and Zero－shot Forecasting

## E.1 Few－Shot Forecasting

Our full results in few－shot forecasting tasks are detailed in Tab.  14 and Tab. 15. Within the scope of $10 \%$ few－shot learning，TIME－LLM secures SOTA performance in 32 out of 35 cases，spanning seven different time series benchmarks. Our approach＇s advantage becomes even more pronounced in the context of 5\％few－shot scenarios，achieving SOTA results in 21 out of 32 cases. We attribute this to the successful knowledge activation in our reprogrammed LLM. 

- Table 14：Full few－shot learning results on $10 \%$ training data. We use the same protocol as in Tab.  1

- Table 15: Full few-shot learning results on $5 \%$ training data. We use the same protocol as in Tab. 1 '-' means that 5\% time series is not sufficient to constitute a training set.

## E.2 Zero－Shot Forecasting

The full results of zero－shot forecasting are summarized in Tab.  16 Time－LLM remarkably sur－ passes the six most competitive time series models in zero－shot adaptation. Overall，we observe over $\mathbf{2 3 . 5 \%}$ and $\mathbf{1 2 . 4 \%}$ MSE and MAE reductions across all baselines on average. Our improve－ ments are consistently significant on those typical cross－domain scenarios（e. g. ，ETTh2 $\rightarrow$ ETTh1 and ETTm $2 \rightarrow$ ETTm1），over $\mathbf{2 0 . 8 \%}$ and $\mathbf{1 1 . 3 \%}$ on average w. r. t. MSE and MAE. Significantly， TIME－LLM exhibits superior performance gains in comparison to LLMTime（Gruver et al. 2023）， which employs a similarly sized backbone LLM（7B）and is the latest effort in leveraging LLMs for zero－shot time series forecasting. We attribute this success to our reprogramming framework being better at activating the LLM＇s knowledge transfer and reasoning capabilities in a resource－efficient manner when performing time series tasks. 

- Table 16: Full zero-shot learning results on ETT datasets. A lower value indicates better performance. Red: the best, Blue: the second best.

# F Ablation Study

The full ablation results are in Tab. 17. We additionally compare the model performance under reprogramming and fine－tuning（with QLoRA Dettmers et al. （2023））protocols. Our results indicate a clear performance gain of our approach compared to the QLoRA variant（A. 5）by $\mathbf{1 9 \%}$ in average. 

- Table 17: Full ablations on ETTh1 and ETTm1 in predicting 96 and 192 steps ahead (MSE reported).

# G Efficiency Comparison with Model Fine－Tuning

Setups. We compare the efficiency of model fine－tuning（with QLoRA Dettmers et al. （2023）） and our proposed model reprogramming in this section with two different backbones，that is，Llama in $1 / 4$ capacity（first 8 Transformer layers）and full capacity. Here，we adhere to the long－term forecasting protocol on ETTh1 to forecast two different steps（that is， 96 and 336 in this case） ahead. For the evaluation metrics，we report the total number of trainable parameters（in million）， GPU memory（in mebibyte），and running time（seconds per iteration）. 

---

Results. Our results are given in Tab. 18. We see that model reprogramming remarkably results in better efficiency compared to parameter－efficient fine－tuning（PEFT）with QLoRA on long－range forecasting tasks in terms of the total number of trainable parameters, GPU memory overhead, and training speed. Quantitatively, there is an $\mathbf{7 1 . 2 \%}$ trainable parameter reduction on average over four scenarios, leading to $\mathbf{2 3 . 1 \%}$ smaller memory consumption and $\mathbf{2 5 . 3 \%}$ faster training speed.

- Table 18: Efficiency comparison between model reprogramming and parameter-efficient fine-tuning (PEFT) with QLoRA (Dettmers et al. 2023) on ETTh1 dataset in forecasting two different steps ahead.

# H ERROR BARS

All experiments have been conducted three times, and we present the standard deviations of our model and the runner-up model here. The comparisons between our method and the second-best method, PatchTST (Nie et al., 2023), on long-term forecasting tasks, are delineated in Tab. 19 In this table, the average MSE and MAE have been reported across four ETT datasets, complete with standard deviations. Furthermore, Tab. 20 contrasts the effectiveness of our method with that of the second-best method, N-HiTS (Challu et al., 2023a), employing varying M4 datasets for the comparison.

- Table 19: Standard deviations of our approach and the second-best method (PatchTST) on all time series datasets for long-term forecasting.

- Table 20: Standard deviations of our Time-LLM and the second-best method (N-HiTS) on M4 datasets for short-term forecasting.

# I Visualization

In this part, we visualize the forecasting results of Time-LLM compared with the state-of-theart and representative methods (e.g., GPT4TS (Zhou et al., 2023a), PatchTST (Nie et al., 2023), and Autoformer (Wu et al., 2021) in various scenarios to demonstrate the superior performance of Time-LLM.

---

In Fig. 7 and Fig. 8, the long-term (input-96-predict-96) and short-term (input-36-predict-36) forecasts of various approaches are compared with the ground truth. Here, Time-LLM showcases forecasting accuracy that is notably superior compared to GPT4TS, PatchTST, and a classical Transformer-based method, Autoformer.

- Figure 7: Long-term forecasting cases from ETTh1 by different models under the input-96-predict96 settings. Blue lines are the ground truths and orange lines are the model predictions.

- Figure 8: Short-term forecasting from the M4 dataset by different models under the input-36-predict18 settings.

---

We also offer visual comparisons of the forecasting results in both few-shot and zero-shot scenarios, as depicted in Fig. 9 and Fig. 10. We adhere to the long-term (input-96-predict-96) forecasting setup in both cases. TIME-LLM exhibits remarkable superiority in forecasting with limited data-a fact that becomes particularly salient when compared to GPT4TS.

- Figure 9: Few-shot forecasting cases from ETTm1 by different models under the input-96-predict96 settings. Blue lines are the ground truths and orange lines are the model predictions.

- Figure 10: Zero-shot forecasting cases from ETTh $1 \rightarrow$ ETTh2 by different models under the input96 -predict- 96 settings. Blue lines are the ground truths and orange lines are the model predictions.
