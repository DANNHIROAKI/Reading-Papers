# ColBERT-XM: A Modular Multi-Vector Representation Model for Zero-Shot Multilingual Information Retrieval

# ColBERT-XM：用于零样本多语言信息检索的模块化多向量表示模型

Antoine Louis ${}^{ \oplus  }$ ,Vageesh Saxena ${}^{ \oplus  }$ ,Gijs van Dijck ${}^{ \oplus  }$ ,Gerasimos Spanakis ${}^{ \oplus  }$ Maastricht University, Netherlands

安托万·路易斯 ${}^{ \oplus  }$ 、瓦吉什·萨克塞纳 ${}^{ \oplus  }$ 、吉斯·范·迪克 ${}^{ \oplus  }$ 、格拉西莫斯·斯帕纳基斯 ${}^{ \oplus  }$ 荷兰马斯特里赫特大学

\{a.louis, v.saxena, gijs.vandijck, jerry.spanakis\}@maastrichtuniversity.nl

\{a.louis, v.saxena, gijs.vandijck, jerry.spanakis\}@maastrichtuniversity.nl

## Abstract

## 摘要

State-of-the-art neural retrievers predominantly focus on high-resource languages like English, which impedes their adoption in retrieval scenarios involving other languages. Current approaches circumvent the lack of high-quality labeled data in non-English languages by leveraging multilingual pretrained language models capable of cross-lingual transfer. However, these models require substantial task-specific fine-tuning across multiple languages, often perform poorly in languages with minimal representation in the pretraining corpus, and struggle to incorporate new languages after the pretraining phase. In this work, we present a novel modular dense retrieval model that learns from the rich data of a single high-resource language and effectively zero-shot transfers to a wide array of languages, thereby eliminating the need for language-specific labeled data. Our model, ColBERT-XM, demonstrates competitive performance against existing state-of-the-art multilingual retrievers trained on more extensive datasets in various languages. Further analysis reveals that our modular approach is highly data-efficient, effectively adapts to out-of-distribution data, and significantly reduces energy consumption and carbon emissions. By demonstrating its proficiency in zero-shot scenarios, ColBERT-XM marks a shift towards more sustainable and inclusive retrieval systems, enabling effective information accessibility in numerous languages. We publicly release our code and models for the community.

目前最先进的神经检索器主要关注英语等资源丰富的语言，这阻碍了它们在涉及其他语言的检索场景中的应用。当前的方法通过利用能够进行跨语言迁移的多语言预训练语言模型，来规避非英语语言中高质量标注数据的缺乏。然而，这些模型需要在多种语言上进行大量特定任务的微调，在预训练语料库中代表性极低的语言上往往表现不佳，并且在预训练阶段之后难以纳入新的语言。在这项工作中，我们提出了一种新颖的模块化密集检索模型，该模型从单一资源丰富语言的丰富数据中学习，并能有效地零样本迁移到多种语言，从而消除了对特定语言标注数据的需求。我们的模型 ColBERT - XM 在各种语言中，与在更广泛数据集上训练的现有最先进的多语言检索器相比，表现出了有竞争力的性能。进一步的分析表明，我们的模块化方法具有很高的数据效率，能有效地适应分布外的数据，并显著降低能源消耗和碳排放。通过在零样本场景中展示其能力，ColBERT - XM 标志着向更可持续和更具包容性的检索系统的转变，使多种语言的信息能够有效获取。我们向社区公开发布了我们的代码和模型。

## 1 Introduction

## 1 引言

Text retrieval models are integral to various day-today applications, including search, recommendation, summarization, and question answering. In recent years, transformer-based models have monopolized textual information retrieval and led to significant progress in the field (Lin et al., 2021). However, the existing literature mostly focuses on improving retrieval effectiveness in a handful of widely spoken languages - notably English (Muen-nighoff et al., 2023) and Chinese (Xiao et al., 2023) - whereas other languages receive limited attention.

文本检索模型是各种日常应用不可或缺的一部分，包括搜索、推荐、摘要和问答。近年来，基于Transformer的模型在文本信息检索领域占据主导地位，并推动了该领域的重大进展（Lin等人，2021）。然而，现有文献大多侧重于提高少数广泛使用的语言（特别是英语（Muen - nighoff等人，2023）和中文（Xiao等人，2023））的检索效果，而其他语言受到的关注有限。

As a solution, a few studies have suggested fine-tuning multilingual transformer-based encoders, such as mBERT (Devlin et al., 2019), on aggregated retrieval data across various languages. Nonetheless, this approach faces two major challenges. First, acquiring high-quality relevance labels for various languages proves difficult, particularly for those with fewer resources. Consequently, languages with insufficient representation in training data experience a proficiency gap compared to widely represented ones (MacAvaney et al., 2020). Second, when these multilingual transformers are pretrained on too many languages, their performance on downstream tasks worsens. This issue, known as the curse of multilinguality (Conneau et al., 2020), underscores the challenge of developing models that effectively accommodate a broader spectrum of languages.

作为一种解决方案，一些研究建议在跨多种语言的聚合检索数据上微调基于多语言Transformer的编码器，如mBERT（Devlin等人，2019）。然而，这种方法面临两个主要挑战。首先，为各种语言获取高质量的相关性标签很困难，特别是对于资源较少的语言。因此，与在训练数据中广泛代表的语言相比，训练数据中代表性不足的语言存在性能差距（MacAvaney等人，2020）。其次，当这些多语言Transformer在过多语言上进行预训练时，它们在下游任务上的性能会变差。这个问题被称为多语言诅咒（Conneau等人，2020），凸显了开发能够有效适应更广泛语言范围的模型的挑战。

Our work addresses the challenges above by introducing ColBERT-XM, a novel multilingual dense retrieval model built upon the recent XMOD architecture (Pfeiffer et al., 2022), which combines shared and language-specific parameters pretrained from the start to support the following key features:

我们的工作通过引入ColBERT - XM来解决上述挑战，这是一种基于最近的XMOD架构（Pfeiffer等人，2022）构建的新颖多语言密集检索模型，该架构结合了从一开始就预训练的共享参数和特定语言参数，以支持以下关键特性：

1. Reduced dependency on multilingual data: Our XMOD-based retriever is designed to learn through monolingual fine-tuning, capitalizing on the rich data from a high-resource language, like English, thereby reducing the need for extensive multilingual datasets.

1. 减少对多语言数据的依赖：我们基于XMOD的检索器旨在通过单语言微调进行学习，利用英语等资源丰富语言的丰富数据，从而减少对大量多语言数据集的需求。

2. Zero-shot transfer across languages: Despite being fine-tuned in a single language, our retriever's modular components enable effective knowledge transfer to a variety of underrepresented languages without any further training.

2. 跨语言的零样本迁移：尽管在单一语言中进行了微调，但我们的检索器的模块化组件能够在无需进一步训练的情况下，将知识有效迁移到各种代表性不足的语言。

3. Post-hoc language addition: Unlike conventional multilingual models, our modular retriever is easily extendable to languages un-

3. 事后添加语言：与传统的多语言模型不同，我们的模块化检索器可以轻松扩展到预训练期间未见过的语言，同时减轻多语言诅咒的影响。

<!-- Media -->

<!-- figureText: Add & Norm Add & Norm Add & Norm Add & Norm Add & Norm Add & Norm Feed Feed Forward Forward 个 Add & Norm Add & Norm Multi-Head Multi-Head Attention Attention Input Input Embedding Embedding mMARCO-X Lang. N+1 (c) Zero-shot transfer (d) Post-hoc language extension Add & Norm Add & Norm Feed Feed Forward Forward 个 Add & Norm Add & Norm Multi-Head Multi-Head Attention Attention Input Input Embedding Embedding CC100 MS MARCO (a) Multilingual pre-training (b) Monolingual fine-tuning -->

<img src="https://cdn.noedgeai.com/0195b377-0ecd-714c-9c93-d2f97ba6f989_1.jpg?x=202&y=189&w=1245&h=683&r=0"/>

Figure 1: An illustration of ColBERT-XM's modular architecture during its successive learning stages. Components that are blurred indicate they remain frozen throughout the learning phase. (a) First, the model learns language-specific modular adapters at each transformer layer through MLM pretraining on a large multilingual corpus. (b) Next, the model is adapted to the downstream task by fine-tuning its shared weights on the source language while keeping the modular adapters and the embedding layer frozen. (c) The model is then used in a zero-shot fashion by routing the target language's input text through the corresponding modular units. (d) Finally, extra languages can be added post-hoc by learning new modular components only through lightweight MLM training on the new language.

图1：ColBERT - XM的模块化架构在其连续学习阶段的示意图。模糊的组件表示它们在学习阶段保持冻结。（a）首先，模型通过在大型多语言语料库上进行掩码语言模型（MLM）预训练，在每个Transformer层学习特定语言的模块化适配器。（b）接下来，通过在源语言上微调其共享权重，同时保持模块化适配器和嵌入层冻结，使模型适应下游任务。（c）然后，通过将目标语言的输入文本路由到相应的模块化单元，以零样本方式使用该模型。（d）最后，通过仅在新语言上进行轻量级的MLM训练来学习新的模块化组件，可以事后添加额外的语言。

<!-- Media -->

## seen during pretraining, while mitigating the curse of multilinguality.

## 预训练期间未见过的语言，同时减轻多语言诅咒的影响。

Practically, ColBERT-XM learns to effectively predict relevance between queries and passages using only a limited set of English examples, leveraging the late interaction approach introduced in ColBERT (Khattab and Zaharia, 2020). Our experimental results demonstrate competitive performance across a diverse range of languages against state-of-the-art multilingual models trained on vastly larger datasets and many more languages. Moreover, our analysis shows that ColBERT-XM is highly data efficient, as more training data from the same distribution does not markedly enhance its performance. Even so, further investigations reveal our model's strong ability to generalize to out-of-distribution data, despite its limited training. We also provide evidence that multi-vector representations outperform single-vector approaches within our framework. Finally, we underscore our model's sustainability by examining its environmental impact in comparison to established dense retrievers.

实际上，ColBERT - XM仅使用有限的英语示例，借助ColBERT（Khattab和Zaharia，2020年）中引入的后期交互方法，学会了有效预测查询与段落之间的相关性。我们的实验结果表明，在多种语言中，与在更大数据集和更多语言上训练的最先进的多语言模型相比，它具有有竞争力的性能。此外，我们的分析显示，ColBERT - XM具有很高的数据效率，因为来自相同分布的更多训练数据并不会显著提高其性能。即便如此，进一步的研究表明，尽管训练有限，但我们的模型对分布外数据具有很强的泛化能力。我们还提供证据表明，在我们的框架内，多向量表示优于单向量方法。最后，通过与已有的密集检索器对比其环境影响，我们强调了我们模型的可持续性。

In summary, the contributions of this research are threefold. First, we introduce a novel modular dense retriever that, despite being trained exclusively in one language, demonstrates remarkable adaptability to a broad spectrum of languages in a zero-shot configuration. Second, through comprehensive experiments, we compare the effectiveness of employing multi-vector over single-vector representations, explore the influence of the volume of training examples on the overall model's performance, investigate the model's ability to adapt to out-of-distribution data and languages it has not previously encountered, including low-resource ones, and highlight its sustainable environmental footprint. Finally, we release our source code and model checkpoints at https: //github.com/ant-louis/xm-retrievers.

综上所述，本研究的贡献有三个方面。首先，我们引入了一种新颖的模块化密集检索器，尽管它仅在一种语言上进行训练，但在零样本配置下对广泛的语言表现出了显著的适应性。其次，通过全面的实验，我们比较了使用多向量表示与单向量表示的有效性，探讨了训练示例数量对整体模型性能的影响，研究了模型适应分布外数据和它之前未遇到过的语言（包括低资源语言）的能力，并强调了其可持续的环境影响。最后，我们在https://github.com/ant - louis/xm - retrievers上发布了我们的源代码和模型检查点。

## 2 Related Work

## 2 相关工作

### 2.1 Multilingual Information Retrieval

### 2.1 多语言信息检索

The term "multilingual" typically encompasses a wide range of retrieval tasks using one or more languages (Hull and Grefenstette, 1996). In our study, we define it as performing monolingual retrieval across multiple languages.

“多语言”这一术语通常涵盖了使用一种或多种语言的广泛检索任务（Hull和Grefenstette，1996年）。在我们的研究中，我们将其定义为跨多种语言执行单语言检索。

Monolingual text retrieval approaches have relied on simple statistical metrics based on term frequency, such as TF-IDF and BM25 (Robertson et al., 1994), to represent texts and match documents against a given query. With the advent of transformer-based language models, con-textualized representations rapidly got incorporated into retrieval models and gave rise to various neural-based retrieval techniques, including cross-encoder models such as monoBERT (Nogueira et al., 2019) and monoT5 (Nogueira et al., 2020), single-vector bi-encoders like DPR (Karpukhin et al., 2020) and ANCE (Xiong et al., 2021), multi-vector bi-encoders like ColBERT (Khattab and Za-haria, 2020) and XTR (Lee et al., 2023), and sparse neural models such as uniCOIL (Lin and Ma, 2021) and SPLADE (Formal et al., 2021).

单语言文本检索方法依赖于基于词频的简单统计指标，如TF - IDF和BM25（Robertson等人，1994年），来表示文本并将文档与给定查询进行匹配。随着基于Transformer的语言模型的出现，上下文表示迅速被纳入检索模型，并催生了各种基于神经网络的检索技术，包括交叉编码器模型，如monoBERT（Nogueira等人，2019年）和monoT5（Nogueira等人，2020年）；单向量双编码器，如DPR（Karpukhin等人，2020年）和ANCE（Xiong等人，2021年）；多向量双编码器，如ColBERT（Khattab和Zaharia，2020年）和XTR（Lee等人，2023年）；以及稀疏神经网络模型，如uniCOIL（Lin和Ma，2021年）和SPLADE（Formal等人，2021年）。

Nevertheless, prior work on neural retrievers has predominantly focused on English due to the abundance of labeled training data. In non-English settings, multilingual pretrained language models such as XLM-R (Conneau et al., 2020) and mBERT (Devlin et al., 2019) emerged as an effective solution, capable of adapting the retrieval task across many languages using a shared model (Lawrie et al., 2023). However, these models proved to suffer from the curse of multilinguality (Chang et al., 2023), have shown substantially reduced monolingual abilities for low-resource languages with smaller pretraining data (Wu and Dredze, 2020), and do not effectively extend to unseen languages after the pretraining phase (Pfeiffer et al., 2022).

然而，先前关于神经检索器的研究主要集中在英语上，因为有大量带标签的训练数据。在非英语环境中，多语言预训练语言模型，如XLM - R（Conneau等人，2020年）和mBERT（Devlin等人，2019年）成为了一种有效的解决方案，能够使用共享模型在多种语言中适应检索任务（Lawrie等人，2023年）。然而，这些模型被证明受到多语言诅咒的影响（Chang等人，2023年），对于预训练数据较少的低资源语言，单语言能力大幅下降（Wu和Dredze，2020年），并且在预训练阶段后不能有效地扩展到未见过的语言（Pfeiffer等人，2022年）。

### 2.2 Modular Transformers

### 2.2 模块化Transformer

Traditionally, adapting pretrained transformer-based language models to new data settings involves fully fine-tuning all pretrained weights on relevant data. While effective, this process is computationally expensive. As a parameter-efficient alternative, recent works have proposed inserting lightweight "expert" modules after each transformer layer (Houlsby et al., 2019) to capture specific modeling aspects, such as language-specific (Pfeiffer et al., 2020; Ansell et al., 2021) or task-specific (Bapna and Firat, 2019; He et al., 2021) knowledge. These modular components, commonly referred to as adapters (Rebuffi et al., 2017), are selectively fine-tuned for the downstream task, the core transformer parameters remaining frozen.

传统上，使预训练的基于Transformer的语言模型适应新的数据设置需要在相关数据上对所有预训练权重进行全面微调。虽然这种方法有效，但计算成本很高。作为一种参数高效的替代方法，近期的研究提出在每个Transformer层之后插入轻量级的“专家”模块（Houlsby等人，2019年），以捕捉特定的建模方面，如特定语言（Pfeiffer等人，2020年；Ansell等人，2021年）或特定任务（Bapna和Firat，2019年；He等人，2021年）的知识。这些模块化组件，通常被称为适配器（Rebuffi等人，2017年），会针对下游任务进行选择性微调，而核心Transformer参数保持冻结。

Despite their growing use in NLP, adapter-based approaches remain relatively untouched in multilingual information retrieval, with existing IR research primarily concentrating on cross-language retrieval (Litschko et al., 2022; Yang et al., 2022b), which aims to return documents in a language different from the query. A key limitation of these works is that the additional capacity introduced by adapters after pretraining is not able to mitigate the curse of multilinguality that has already had a catastrophic impact on the shared transformer weights (Pfeif-fer et al., 2022). In contrast, our method employs a model inherently designed for modularity that learns language-specific capacity during pretraining, effectively avoiding this limitation.

尽管基于适配器的方法在自然语言处理（NLP）中的应用日益广泛，但在多语言信息检索领域，这些方法仍相对未被充分探索。现有的信息检索研究主要集中在跨语言检索方面（利奇科等人，2022年；杨等人，2022年b），其目标是返回与查询语言不同的文档。这些研究的一个关键局限性在于，预训练后适配器引入的额外容量无法缓解多语言问题带来的负面影响，而这种影响已经对共享的Transformer权重造成了灾难性后果（普法伊费尔等人，2022年）。相比之下，我们的方法采用了一种本质上为模块化设计的模型，该模型在预训练阶段学习特定语言的容量，从而有效避免了这一局限性。

## 3 Method

## 3 方法

We present a novel multilingual dense retriever that learns to predict relevance between queries and passages via monolingual fine-tuning, while adapting to various languages in a zero-shot configuration. Our model, ColBERT-XM, adopts a traditional bi-encoder architecture (§3.1) based on a modular multilingual text encoder (§3.2), and employs the MaxSim-based late interaction mechanism (§3.3) for relevance assessment. The model is optimized through a contrastive learning strategy (§3.4), and uses a residual compression approach to significantly reduce the space footprint of indexes utilized for fast vector-similarity search at inference time (§3.5). We describe each part in detail below.

我们提出了一种新颖的多语言密集检索器，它通过单语言微调学习预测查询和段落之间的相关性，同时能够在零样本配置下适应多种语言。我们的模型ColBERT - XM采用了基于模块化多语言文本编码器的传统双编码器架构（§3.1），并采用基于最大相似度的后期交互机制（§3.3）进行相关性评估。该模型通过对比学习策略进行优化（§3.4），并使用残差压缩方法显著减少推理时用于快速向量相似度搜索的索引的空间占用（§3.5）。下面我们将详细描述每个部分。

### 3.1 Bi-Encoder Architecture

### 3.1 双编码器架构

To predict relevance between query $q$ and passage $p$ ,ColBERT-XM uses the popular bi-encoder architecture (Gillick et al., 2018), which consists of two learnable text encoding functions $f\left( {\cdot ;{\gamma }_{i}}\right)$ : ${\mathcal{W}}^{n} \mapsto  {\mathbb{R}}^{n \times  d}$ ,parameterized by ${\gamma }_{i}$ ,that map input text sequences of $n$ terms from vocabulary $\mathcal{W}$ to $d$ -dimensional real-valued term vectors,i.e.,

为了预测查询$q$和段落$p$之间的相关性，ColBERT - XM使用了流行的双编码器架构（吉利克等人，2018年），该架构由两个可学习的文本编码函数$f\left( {\cdot ;{\gamma }_{i}}\right)$组成：${\mathcal{W}}^{n} \mapsto  {\mathbb{R}}^{n \times  d}$，由${\gamma }_{i}$参数化，将来自词汇表$\mathcal{W}$的$n$个词的输入文本序列映射到$d$维实值词向量，即

$$
{\widehat{\mathbf{H}}}_{q} = f\left( {\left\lbrack  {{q}_{1},{q}_{2},\cdots ,{q}_{i}}\right\rbrack  ;{\mathbf{\gamma }}_{1}}\right) \text{,and} \tag{1}
$$

$$
{\widehat{\mathbf{H}}}_{p} = f\left( {\left\lbrack  {{p}_{1},{p}_{2},\cdots ,{p}_{j}}\right\rbrack  ;{\mathbf{\gamma }}_{2}}\right) .
$$

The main idea behind this architecture is to find values for parameters ${\gamma }_{i}$ such that a straightforward similarity function $\operatorname{sim} : {\mathbb{R}}^{n \times  d} \times  {\mathbb{R}}^{m \times  d} \mapsto  {\mathbb{R}}_{ + }$ approximates the semantic relevance between $q$ and $p$ by operating on their bags of contextualized term embeddings, i.e.,

这种架构背后的主要思想是找到参数${\gamma }_{i}$的值，使得一个简单的相似度函数$\operatorname{sim} : {\mathbb{R}}^{n \times  d} \times  {\mathbb{R}}^{m \times  d} \mapsto  {\mathbb{R}}_{ + }$通过对查询和段落的上下文词嵌入集合进行操作，来近似$q$和$p$之间的语义相关性，即

$$
\operatorname{score}\left( {q,p}\right)  = \operatorname{sim}\left( {{\widehat{\mathbf{H}}}_{q},{\widehat{\mathbf{H}}}_{p}}\right) . \tag{2}
$$

This scoring approach, known as late interaction (Khattab and Zaharia, 2020), as interactions between the query and passage are delayed after their independent encoding computations, stands out for its computational efficiency (Reimers and Gurevych, 2019). This contrasts with the popular cross-encoder architecture (Nogueira et al., 2019), which encodes the queries and passages jointly to learn rich interactions directly within the model.

这种评分方法被称为后期交互（卡塔布和扎哈里亚，2020年），因为查询和段落之间的交互在它们独立的编码计算之后才进行，其以计算效率高而著称（赖默斯和古雷维奇，2019年）。这与流行的交叉编码器架构（诺盖拉等人，2019年）形成对比，后者将查询和段落联合编码，以直接在模型内部学习丰富的交互。

<!-- Media -->

<!-- figureText: ... Linear Feed Forward Multi-Head Attention Feed Forward Multi-Head Attention Embedding Passage Linear #N Feed Forward Multi-Head Attention #N Feed Forward Multi-Head Attention Embedding Query -->

<img src="https://cdn.noedgeai.com/0195b377-0ecd-714c-9c93-d2f97ba6f989_3.jpg?x=235&y=189&w=528&h=636&r=0"/>

Figure 2: Illustration of the multi-vector late interaction paradigm used in our proposed ColBERT-XM model.

图2：我们提出的ColBERT - XM模型中使用的多向量后期交互范式示意图。

<!-- Media -->

In this work, we use a siamese bi-encoder, where queries and passages are encoded by two identical copies of a shared network (i.e., ${\gamma }_{1} = {\gamma }_{2}$ ).

在这项工作中，我们使用了一个孪生双编码器，其中查询和段落由共享网络的两个相同副本进行编码（即${\gamma }_{1} = {\gamma }_{2}$）。

### 3.2 Modular Language Representation

### 3.2 模块化语言表示

To overcome the limitations posed by multilingual transformer-based encoders outlined in Section 1, we use the XMOD model (Pfeiffer et al., 2022) as our backbone text encoder. As depicted in Figure 1a, XMOD extends the transformer architecture by incorporating language-specific adapters (Houlsby et al., 2019) at every transformer layer, which are learned from the start during the masked language modeling (MLM) pretraining phase. This method contrasts with conventional adapter-based approaches that typically extend pretrained multilingual models post-pretraining, thereby building upon sub-optimal parameter initialization already affected by the curse of multilinguality.

为了克服第1节中概述的基于多语言Transformer的编码器所带来的局限性，我们使用XMOD模型（普法伊费尔等人，2022年）作为我们的骨干文本编码器。如图1a所示，XMOD通过在每个Transformer层中加入特定语言的适配器（豪尔斯比等人，2019年）来扩展Transformer架构，这些适配器在掩码语言建模（MLM）预训练阶段从一开始就进行学习。这种方法与传统的基于适配器的方法形成对比，传统方法通常在预训练后扩展预训练的多语言模型，从而建立在已经受到多语言问题影响的次优参数初始化之上。

Formally, our modular language representation model is defined as a learnable encoding function $g\left( {\cdot ;\mathbf{\theta },{\phi }_{i}}\right)  : \left( {{\mathcal{W}}^{k},\mathcal{L}}\right)  \mapsto  {\mathbb{R}}^{k \times  d}$ ,with shared parameters $\mathbf{\theta }$ and language-specific parameters ${\phi }_{i}$ ,that maps a text sequence $t$ of $k$ terms from vocabulary $\mathcal{W}$ in language ${\mathcal{L}}_{i}$ to $d$ -dimensional real-valued representations. Let ${\mathbf{W}}_{\text{out }} \in  {\mathbb{R}}^{d \times  {d}_{\text{out }}}$ be a linear layer with no activations designed to compress the dimensions of the output representation vectors, Equation (1) then becomes

形式上，我们的模块化语言表示模型被定义为一个可学习的编码函数 $g\left( {\cdot ;\mathbf{\theta },{\phi }_{i}}\right)  : \left( {{\mathcal{W}}^{k},\mathcal{L}}\right)  \mapsto  {\mathbb{R}}^{k \times  d}$，具有共享参数 $\mathbf{\theta }$ 和特定语言参数 ${\phi }_{i}$，该函数将来自语言 ${\mathcal{L}}_{i}$ 的词汇表 $\mathcal{W}$ 中 $k$ 个词项的文本序列 $t$ 映射到 $d$ 维实值表示。设 ${\mathbf{W}}_{\text{out }} \in  {\mathbb{R}}^{d \times  {d}_{\text{out }}}$ 为一个无激活函数的线性层，用于压缩输出表示向量的维度，那么方程 (1) 变为

$$
{\widehat{\mathbf{H}}}_{t} = g\left( {\left\lbrack  {{t}_{1},\cdots ,{t}_{k}}\right\rbrack  ;\mathbf{\theta },{\phi }_{i}}\right)  \cdot  {\mathbf{W}}_{\text{out }}
$$

$$
 = \left\lbrack  {{\widehat{\mathbf{h}}}_{1}^{t},{\widehat{\mathbf{h}}}_{2}^{t},\cdots ,{\widehat{\mathbf{h}}}_{k}^{t}}\right\rbrack  . \tag{3}
$$

A key benefit of employing XMOD over traditional multilingual transformers is its proven adaptability to accommodate new languages after the initial pretraining phase while maintaining performance across previously included languages, thereby effectively counteracting the curse of multilinguality. Furthermore, Pfeiffer et al. (2022) demonstrated that the per-language performance remains consistent whether a language is included during pretraining or added afterward. This suggests that XMOD can potentially encompass numerous languages by pretraining on a subset of languages for which sufficient text data exists, and subsequently adapting to additional, underrepresented languages without deteriorating overall performance. As illustrated in Figure 1d, the post-hoc inclusion of a new language involves learning additional language-specific modular components only through lightweight MLM training on the new language.

与传统的多语言变压器模型相比，采用 XMOD 的一个关键优势在于，它在初始预训练阶段之后，被证明能够适应新语言，同时保持对之前纳入语言的性能，从而有效克服了多语言问题的困境。此外，Pfeiffer 等人（2022 年）证明，无论一种语言是在预训练期间纳入还是之后添加，其特定语言的性能都保持一致。这表明，XMOD 可以通过在有足够文本数据的一部分语言上进行预训练，然后适应更多代表性不足的语言，而不会降低整体性能。如图 1d 所示，事后纳入一种新语言只需要通过对该新语言进行轻量级的掩码语言模型（Masked Language Model，MLM）训练来学习额外的特定语言模块化组件。

### 3.3 MaxSim-based Late Interaction

### 3.3 基于最大相似度（MaxSim）的后期交互

ColBERT-XM adopts the fine-granular late interaction scoring mechanism of ColBERT, depicted in Figure 2. This mechanism calculates the cosine similarity across all pairs of query and passage em-beddings, applies max-pooling across the resulting similarity scores for each query term, and then sum the maximum values across query terms to derive the overall relevance estimate, i.e.,

ColBERT - XM 采用了 ColBERT 的细粒度后期交互评分机制，如图 2 所示。该机制计算查询和段落嵌入的所有对之间的余弦相似度，对每个查询词项的相似度得分应用最大池化，然后对查询词项的最大值求和，以得出整体相关性估计，即

$$
\operatorname{sim}\left( {{\widehat{\mathbf{H}}}_{\widetilde{q}},{\widehat{\mathbf{H}}}_{\widetilde{p}}}\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{j = 1}}^{m}\cos \left( {{\widehat{\mathbf{h}}}_{i}^{\widetilde{q}},{\widehat{\mathbf{h}}}_{j}^{\widetilde{p}}}\right) , \tag{4}
$$

where $\widetilde{q}$ and $\widetilde{p}$ correspond to sequences obtained after incorporating special tokens into $q$ and $p$ ,respectively,and truncating to preset maximum lengths $n$ and $m$ . More specifically,we have

其中 $\widetilde{q}$ 和 $\widetilde{p}$ 分别对应于将特殊标记纳入 $q$ 和 $p$ 后得到的序列，并截断为预设的最大长度 $n$ 和 $m$。更具体地说，我们有

$$
\widetilde{p} = \left\lbrack  {\left\lbrack  \mathrm{{CLS}}\right\rbrack  ,\left\lbrack  \mathrm{P}\right\rbrack  ,{p}_{1},\cdots ,{p}_{j}}\right\rbrack  \text{,and} \tag{5}
$$

$$
\widetilde{q} = \left\lbrack  {\left\lbrack  \mathrm{{CLS}}\right\rbrack  ,\left\lbrack  \mathrm{Q}\right\rbrack  ,{q}_{1},\cdots ,{q}_{i},\left\lbrack  \mathrm{M}\right\rbrack  ,\cdots ,\left\lbrack  \mathrm{M}\right\rbrack  }\right\rbrack  ,
$$

where $\left\lbrack  \mathrm{M}\right\rbrack$ is a mask token appended to queries to reach the predefined length $n$ . This padding strategy serves as a query augmentation technique, enhancing the model's ability to interpret short queries through the generation of extra contextual-ized embeddings at the mask positions. The special tokens [P] and [Q] enable the shared XMOD-based encoder to differentiate between passage and query input sequences, respectively.

其中 $\left\lbrack  \mathrm{M}\right\rbrack$ 是附加到查询中的掩码标记，以使查询达到预定义的长度 $n$。这种填充策略作为一种查询增强技术，通过在掩码位置生成额外的上下文嵌入来增强模型解释短查询的能力。特殊标记 [P] 和 [Q] 使基于 XMOD 的共享编码器能够分别区分段落和查询输入序列。

### 3.4 Supervision

### 3.4 监督

Let $\mathcal{B} = {\left\{  \left( {q}_{i},{p}_{i}^{ + },{p}_{\mathrm{H},i}^{ - }\right) \right\}  }_{i = 1}^{N}$ be a batch of $N$ training instances,each comprising a query ${q}_{i}$ associated with a positive passage ${p}_{i}^{ + }$ and a hard negative passage ${p}_{\mathrm{H},i}^{ - }$ . By considering the passages paired with all other queries within the same batch, we can enrich each training triple with an additional set of $2\left( {N - 1}\right)$ in-batch negatives ${\mathcal{P}}_{\mathrm{{IB}},i}^{ - } = {\left\{  {p}_{j}^{ + },{p}_{\mathrm{H},j}^{ - }\right\}  }_{j \neq  i}^{N}$ . Given these augmented training samples, we optimize our model using a contrastive learning strategy that combines two established ranking loss functions, expressed as

设 $\mathcal{B} = {\left\{  \left( {q}_{i},{p}_{i}^{ + },{p}_{\mathrm{H},i}^{ - }\right) \right\}  }_{i = 1}^{N}$ 为一批 $N$ 个训练实例，每个实例包含一个查询 ${q}_{i}$，该查询与一个正段落 ${p}_{i}^{ + }$ 和一个难负段落 ${p}_{\mathrm{H},i}^{ - }$ 相关联。通过考虑与同一批次中所有其他查询配对的段落，我们可以用一组额外的 $2\left( {N - 1}\right)$ 个批次内负样本 ${\mathcal{P}}_{\mathrm{{IB}},i}^{ - } = {\left\{  {p}_{j}^{ + },{p}_{\mathrm{H},j}^{ - }\right\}  }_{j \neq  i}^{N}$ 丰富每个训练三元组。给定这些增强的训练样本，我们使用一种对比学习策略来优化我们的模型，该策略结合了两个已有的排序损失函数，表达式为

$$
{\mathcal{L}}_{\text{TOTAL }}\left( {{q}_{i},{p}_{i}^{ + },{p}_{\mathrm{H},i}^{ - },{\mathcal{P}}_{\mathrm{{IB}},i}^{ - }}\right)  = {\mathcal{L}}_{\text{PAIR }} + {\mathcal{L}}_{\mathrm{{IB}}} \tag{6}
$$

where ${\mathcal{L}}_{\text{PAIR }}$ is the pairwise softmax cross-entropy loss computed over predicted scores for the positive and hard negative passages, used in ColBERTv1 (Khattab and Zaharia, 2020) and defined as

其中 ${\mathcal{L}}_{\text{PAIR }}$ 是在正段落和难负段落的预测得分上计算的成对 softmax 交叉熵损失，用于 ColBERTv1（Khattab 和 Zaharia，2020 年），定义为

$$
{\mathcal{L}}_{\mathrm{{PAIR}}} =  - \log \frac{{e}^{\operatorname{score}\left( {{q}_{i},{p}_{i}^{ + }}\right) }}{{e}^{\operatorname{score}\left( {{q}_{i},{p}_{i}^{ + }}\right) } + {e}^{\operatorname{score}\left( {{q}_{i},{p}_{\mathrm{H},i}^{ - }}\right) }}, \tag{7}
$$

while ${\mathcal{L}}_{\mathrm{{IB}}}$ is the in-batch sampled softmax cross-entropy loss added as an enhancement for optimizing ColBERTv2 (Santhanam et al., 2022):

而 ${\mathcal{L}}_{\mathrm{{IB}}}$ 是作为优化 ColBERTv2 的增强项添加的批次内采样 softmax 交叉熵损失（Santhanam 等人，2022 年）：

$$
{\mathcal{L}}_{\mathrm{{IB}}} =  - \log \frac{{e}^{\operatorname{score}\left( {{q}_{i},{p}_{i}^{ + }}\right) }}{\mathop{\sum }\limits_{{p \in  {\mathcal{P}}_{\mathrm{{IB}},i}^{ - } \cup  \left\{  {{p}_{i}^{ + },{p}_{\mathrm{H},i}^{ - }}\right\}  }}{e}^{\operatorname{score}\left( {{q}_{i},p}\right) }}. \tag{8}
$$

These contrastive losses aim to learn a high-quality embedding function so that relevant query-passage pairs achieve higher similarity than irrelevant ones.

这些对比损失旨在学习一个高质量的嵌入函数，使相关的查询 - 段落对比不相关的对具有更高的相似度。

### 3.5 Inference

### 3.5 推理

Since passages and queries are encoded independently, passage embeddings can be precomputed and indexed offline through efficient vector-similarity search data structures, using the faiss library (Johnson et al., 2021). Instead of directly indexing the passage representations as in Col-BERTv1, which requires substantial storage even when compressed to 32 or 16 bits, we adopt the centroid-based indexing approach introduced in ColBERTv2, as detailed in Appendix A.

由于段落和查询是独立编码的，因此可以通过高效的向量相似度搜索数据结构（使用faiss库（Johnson等人，2021年））离线预先计算段落嵌入并建立索引。与Col - BERTv1中直接对段落表示进行索引不同（即使压缩到32位或16位也需要大量存储空间），我们采用了ColBERTv2中引入的基于质心的索引方法，具体细节见附录A。

## 4 Experiments

## 4 实验

### 4.1 Experimental Setup

### 4.1 实验设置

Data. For training, we follow ColBERTv1 and use triples from the MS MARCO passage ranking dataset (Nguyen et al., 2018), which contains ${8.8}\mathrm{M}$ passages and ${539}\mathrm{\;K}$ training queries. However, unlike the original work that uses the BM25 negatives provided by the official dataset, we sample harder negatives mined from 12 distinct dense retrievers. ${}^{1}$ For a comprehensive evaluation across various languages, we consider the small development sets from mMARCO (Bonifacio et al., 2021), a machine-translated variant of MS MARCO in 13 languages, each comprising 6980 queries. To assess out-of-distribution performance, we use the test sets from Mr. TYDI (Zhang et al., 2021), another multilingual open retrieval dataset including low-resource languages not present in mMARCO.

数据。在训练方面，我们遵循ColBERTv1的做法，使用MS MARCO段落排名数据集（Nguyen等人，2018年）中的三元组，该数据集包含${8.8}\mathrm{M}$个段落和${539}\mathrm{\;K}$个训练查询。然而，与使用官方数据集提供的BM25负样本的原始工作不同，我们从12个不同的密集检索器中挖掘并采样了更难的负样本。${}^{1}$ 为了对各种语言进行全面评估，我们考虑了mMARCO（Bonifacio等人，2021年）的小型开发集，这是MS MARCO的一个机器翻译版本，包含13种语言，每种语言包含6980个查询。为了评估分布外性能，我们使用了Mr. TYDI（Zhang等人，2021年）的测试集，这是另一个多语言开放检索数据集，包括mMARCO中未出现的低资源语言。

Implementation. We train our model for ${50}\mathrm{k}$ steps using the AdamW optimizer (Loshchilov and Hutter, 2017) with a batch size of 128, a peak learning rate of $3\mathrm{e} - 6$ with warm up along the first ${10}\%$ of training steps and linear scheduling. We set the embedding dimension to ${d}_{\text{out }} = {128}$ , and fix the maximum sequence lengths for questions and passages at $n = {32}$ and $m = {256}$ ,respectively. Training is performed on one ${80}\mathrm{{GB}}$ NVIDIA H100 GPU hosted on a server with a dual 20-core Intel Xeon E5-2698 v4 CPU @2.20GHz and 512GB of RAM. We use the following Python libraries: transformers (Wolf et al., 2020), sentence-transformers (Reimers and Gurevych, 2019), colbert-ir (Khattab and Zaharia, 2020), and wandb (Biewald, 2020).

实现。我们使用AdamW优化器（Loshchilov和Hutter，2017年）对模型进行${50}\mathrm{k}$步训练，批量大小为128，峰值学习率为$3\mathrm{e} - 6$，在前${10}\%$的训练步骤中进行热身并采用线性调度。我们将嵌入维度设置为${d}_{\text{out }} = {128}$，并分别将问题和段落的最大序列长度固定为$n = {32}$和$m = {256}$。训练在一台搭载双20核英特尔至强E5 - 2698 v4 CPU（主频2.20GHz）和512GB内存的服务器上的一个${80}\mathrm{{GB}}$ NVIDIA H100 GPU上进行。我们使用以下Python库：transformers（Wolf等人，2020年）、sentence - transformers（Reimers和Gurevych，2019年）、colbert - ir（Khattab和Zaharia，2020年）和wandb（Biewald，2020年）。

Metrics & evaluation. To measure effectiveness, we use the official metrics for each query set, i.e., mean reciprocal rank at cut-off 10 (MRR@10) for MS MARCO, and recall at cut-off 100 (R@100) along MRR@100 for Mr. TYDI. We compare our model against established multilingual baselines spanning four retrieval methodologies. For lexical matching, we report the widely adopted bag-of-words BM25 function (Robertson et al., 1994). For the cross-encoders, we include two classification models based on mMiniL ${\mathrm{M}}_{\mathrm{L}6}$ (Wang et al.,2021) and ${\mathrm{{mT5}}}_{\text{BASE }}$ (Xue et al.,2021),each fine-tuned on mMARCO pairs across 9 languages (Bonifacio et al., 2021). The dense single-vector bi-encoders are derived from XLM-R (Conneau et al., 2020) and have been fine-tuned on samples in 4 (Yang et al., 2022a) and 16 languages (Wang et al., 2022), respectively. Lastly, we report the performance of a dense multi-vector bi-encoder built on ${\mathrm{{mBERT}}}_{\text{BASE }}$ (Devlin et al., 2019) and fine-tuned on mMARCO samples across 9 languages (Bonifacio et al., 2021).

指标与评估。为了衡量有效性，我们使用每个查询集的官方指标，即MS MARCO的截断10处的平均倒数排名（MRR@10），以及Mr. TYDI的截断100处的召回率（R@100）和MRR@100。我们将我们的模型与涵盖四种检索方法的既定多语言基线进行比较。对于词法匹配，我们报告了广泛采用的词袋BM25函数（Robertson等人，1994年）。对于交叉编码器，我们包括两个基于mMiniL ${\mathrm{M}}_{\mathrm{L}6}$（Wang等人，2021年）和${\mathrm{{mT5}}}_{\text{BASE }}$（Xue等人，2021年）的分类模型，每个模型都在9种语言的mMARCO对上进行了微调（Bonifacio等人，2021年）。密集单向量双编码器源自XLM - R（Conneau等人，2020年），并分别在4种（Yang等人，2022a）和16种语言（Wang等人，2022年）的样本上进行了微调。最后，我们报告了一个基于${\mathrm{{mBERT}}}_{\text{BASE }}$（Devlin等人，2019年）构建并在9种语言的mMARCO样本上进行微调（Bonifacio等人，2021年）的密集多向量双编码器的性能。

---

<!-- Footnote -->

https://huggingface.co/datasets/

sentence-transformers/msmarco-hard-negatives

sentence-transformers/msmarco-hard-negatives

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td>Model</td><td>#Training Examples</td><td>#Training Languages</td><td>#Active $\mathbf{{Params}}$</td><td>en</td><td>es</td><td>${f}_{r}$</td><td>it</td><td>${pt}$</td><td>id</td><td>${de}$</td><td>${ru}$</td><td>${zh}$</td><td>${ja}$</td><td>${nl}$</td><td>${vi}$</td><td>${hi}$</td><td>ar</td><td>$\mathbf{{Avg}}$</td></tr><tr><td colspan="19">Lexical systems</td></tr><tr><td>1 BM25 (Pyserini)</td><td>-</td><td>-</td><td>-</td><td>18.4</td><td>15.8</td><td>15.5</td><td>15.3</td><td>15.2</td><td>14.9</td><td>13.6</td><td>12.4</td><td>11.6</td><td>14.1</td><td>14.0</td><td>13.6</td><td>13.4</td><td>11.1</td><td>14.2</td></tr><tr><td colspan="19">Cross-encoders</td></tr><tr><td>$2{\mathrm{{mT5}}}_{\text{BASE }}$ (Bonifacio et al.,2021)</td><td>12.8M</td><td>9</td><td>390M</td><td>36.6</td><td>31.4</td><td>30.2</td><td>30.3</td><td>30.2</td><td>29.8</td><td>28.9</td><td>26.3</td><td>24.9</td><td>26.7</td><td>29.2</td><td>25.6</td><td>26.6</td><td>23.5</td><td>28.6</td></tr><tr><td>3 mMiniLM (Bonifacio et al., 2021)</td><td>80.0M</td><td>9</td><td>107M</td><td>36.6</td><td>30.9</td><td>29.6</td><td>29.1</td><td>28.9</td><td>29.3</td><td>27.8</td><td>25.1</td><td>24.9</td><td>26.3</td><td>27.6</td><td>24.7</td><td>26.2</td><td>21.9</td><td>27.8</td></tr><tr><td colspan="19">Dense single-vector bi-encoders</td></tr><tr><td>4 DPR-X (Yang et al., 2022a)</td><td>25.6M</td><td>4</td><td>550M</td><td>24.5</td><td>19.6</td><td>18.9</td><td>18.3</td><td>19.0</td><td>16.9</td><td>18.2</td><td>17.7</td><td>14.8</td><td>15.4</td><td>18.5</td><td>15.1</td><td>15.4</td><td>12.9</td><td>17.5</td></tr><tr><td>5 mE5вазε (Wang et al., 2022)</td><td>5.1B</td><td>16</td><td>278M</td><td>35.0</td><td>28.9</td><td>30.3</td><td>28.0</td><td>27.5</td><td>26.1</td><td>27.1</td><td>24.5</td><td>22.9</td><td>25.0</td><td>27.3</td><td>23.9</td><td>24.2</td><td>20.5</td><td>26.5</td></tr><tr><td colspan="19">Dense multi-vector bi-encoders</td></tr><tr><td>6 mColBERT (Bonifacio et al., 2021)</td><td>25.6M</td><td>9</td><td>180M</td><td>35.2</td><td>30.1</td><td>28.9</td><td>29.2</td><td>29.2</td><td>27.5</td><td>28.1</td><td>25.0</td><td>24.6</td><td>23.6</td><td>27.3</td><td>18.0</td><td>23.2</td><td>20.9</td><td>26.5</td></tr><tr><td colspan="19">Ours</td></tr><tr><td>ColBERT-XM</td><td>6.4M</td><td>1</td><td>277M</td><td>37.2</td><td>28.5</td><td>26.9</td><td>26.5</td><td>27.6</td><td>26.3</td><td>27.0</td><td>25.1</td><td>24.6</td><td>24.1</td><td>27.5</td><td>22.6</td><td>23.8</td><td>19.5</td><td>26.2</td></tr></table>

<table><tbody><tr><td>模型</td><td>#训练示例</td><td>#训练语言</td><td>#活跃 $\mathbf{{Params}}$</td><td>英语</td><td>西班牙语</td><td>${f}_{r}$</td><td>意大利语</td><td>${pt}$</td><td>印尼语</td><td>${de}$</td><td>${ru}$</td><td>${zh}$</td><td>${ja}$</td><td>${nl}$</td><td>${vi}$</td><td>${hi}$</td><td>阿拉伯语</td><td>$\mathbf{{Avg}}$</td></tr><tr><td colspan="19">词法系统</td></tr><tr><td>1 BM25（Pyserini）</td><td>-</td><td>-</td><td>-</td><td>18.4</td><td>15.8</td><td>15.5</td><td>15.3</td><td>15.2</td><td>14.9</td><td>13.6</td><td>12.4</td><td>11.6</td><td>14.1</td><td>14.0</td><td>13.6</td><td>13.4</td><td>11.1</td><td>14.2</td></tr><tr><td colspan="19">交叉编码器</td></tr><tr><td>$2{\mathrm{{mT5}}}_{\text{BASE }}$（博尼法西奥等人，2021年）</td><td>12.8M</td><td>9</td><td>390M</td><td>36.6</td><td>31.4</td><td>30.2</td><td>30.3</td><td>30.2</td><td>29.8</td><td>28.9</td><td>26.3</td><td>24.9</td><td>26.7</td><td>29.2</td><td>25.6</td><td>26.6</td><td>23.5</td><td>28.6</td></tr><tr><td>3 mMiniLM（博尼法西奥等人，2021年）</td><td>80.0M</td><td>9</td><td>107M</td><td>36.6</td><td>30.9</td><td>29.6</td><td>29.1</td><td>28.9</td><td>29.3</td><td>27.8</td><td>25.1</td><td>24.9</td><td>26.3</td><td>27.6</td><td>24.7</td><td>26.2</td><td>21.9</td><td>27.8</td></tr><tr><td colspan="19">密集单向量双编码器</td></tr><tr><td>4 DPR - X（杨等人，2022a）</td><td>25.6M</td><td>4</td><td>550M</td><td>24.5</td><td>19.6</td><td>18.9</td><td>18.3</td><td>19.0</td><td>16.9</td><td>18.2</td><td>17.7</td><td>14.8</td><td>15.4</td><td>18.5</td><td>15.1</td><td>15.4</td><td>12.9</td><td>17.5</td></tr><tr><td>5 mE5вазε（王等人，2022年）</td><td>5.1B</td><td>16</td><td>278M</td><td>35.0</td><td>28.9</td><td>30.3</td><td>28.0</td><td>27.5</td><td>26.1</td><td>27.1</td><td>24.5</td><td>22.9</td><td>25.0</td><td>27.3</td><td>23.9</td><td>24.2</td><td>20.5</td><td>26.5</td></tr><tr><td colspan="19">密集多向量双编码器</td></tr><tr><td>6 mColBERT（博尼法西奥等人，2021年）</td><td>25.6M</td><td>9</td><td>180M</td><td>35.2</td><td>30.1</td><td>28.9</td><td>29.2</td><td>29.2</td><td>27.5</td><td>28.1</td><td>25.0</td><td>24.6</td><td>23.6</td><td>27.3</td><td>18.0</td><td>23.2</td><td>20.9</td><td>26.5</td></tr><tr><td colspan="19">我们的方法</td></tr><tr><td>ColBERT - XM</td><td>6.4M</td><td>1</td><td>277M</td><td>37.2</td><td>28.5</td><td>26.9</td><td>26.5</td><td>27.6</td><td>26.3</td><td>27.0</td><td>25.1</td><td>24.6</td><td>24.1</td><td>27.5</td><td>22.6</td><td>23.8</td><td>19.5</td><td>26.2</td></tr></tbody></table>

Table 1: MRR@10 results on mMARCO small dev set. Performance on languages encountered during fine-tuning is highlighted in orange, whereas zero-shot performance is highlighted in blue. ColBERT-XM reaches near state-of-the-art results while trained on one language only with much fewer examples than competitive models.

表1：mMARCO小型开发集上的MRR@10结果。微调期间遇到的语言的性能以橙色突出显示，而零样本性能以蓝色突出显示。ColBERT - XM仅在一种语言上进行训练，且训练示例比竞争模型少得多的情况下，仍能达到接近最先进水平的结果。

<!-- Media -->

### 4.2 Main Results

### 4.2 主要结果

Table 1 reports results using the official MRR@10 metric for the 14 languages included in mMARCO. In its training language (i.e. English), ColBERT-XM outperforms all multilingual baselines. The un-derperformance of certain models,like ${\mathrm{{mT5}}}_{\text{BASE }}$ and mColBERT, can partly be attributed to their exposure to fewer English examples given their training across 9 languages with ${12.8}\mathrm{M}$ and ${25.6}\mathrm{M}$ samples distributed evenly - resulting in only ${1.4}\mathrm{M}$ and 2.8M English examples, respectively, compared to ColBERT-XM's 6.4M training set. Conversely, models such as ${\mathrm{{mMiniLM}}}_{\mathrm{L}6}$ and ${\mathrm{{mE5}}}_{\mathrm{{BASE}}}$ ,despite being exposed to a larger number of English examples, still underperform, suggesting that the modular architecture of ColBERT-XM may offer intrinsic benefits over conventional multilingual models.

表1报告了使用官方MRR@10指标对mMARCO中包含的14种语言的评估结果。在其训练语言（即英语）中，ColBERT - XM的表现优于所有多语言基线模型。某些模型（如${\mathrm{{mT5}}}_{\text{BASE }}$和mColBERT）表现不佳，部分原因在于它们在9种语言上进行训练，且${12.8}\mathrm{M}$和${25.6}\mathrm{M}$样本均匀分布，导致英语示例较少——分别只有${1.4}\mathrm{M}$和280万个英语示例，而ColBERT - XM的训练集有640万个示例。相反，像${\mathrm{{mMiniLM}}}_{\mathrm{L}6}$和${\mathrm{{mE5}}}_{\mathrm{{BASE}}}$这样的模型，尽管接触到更多的英语示例，但表现仍然不佳，这表明ColBERT - XM的模块化架构可能比传统的多语言模型具有内在优势。

In languages on which ColBERT-XM was not trained but the baselines were, we observe comparable performance. For instance, when excluding English, the difference in average performance between our model and ${\mathrm{{mE5}}}_{\text{BASE }}$ is merely ${0.5}\%$ , even though mE5 ${}_{\text{BASE }}$ was trained in 15 additional languages and 800,000 times more data samples. In languages on which neither ColBERT-XM nor the baselines were trained, we note a slight enhancement in performance among the computationally expensive cross-encoder models, while both the non-modular single-vector and multi-vector bi-encoders lag behind our model in performance.

在ColBERT - XM未进行训练但基线模型进行了训练的语言中，我们观察到性能相当。例如，排除英语后，我们的模型与${\mathrm{{mE5}}}_{\text{BASE }}$的平均性能差异仅为${0.5}\%$，尽管mE5 ${}_{\text{BASE }}$在另外15种语言上进行了训练，且数据样本多了80万倍。在ColBERT - XM和基线模型都未进行训练的语言中，我们注意到计算成本较高的交叉编码器模型的性能略有提升，而非模块化的单向量和多向量双编码器在性能上落后于我们的模型。

Overall, ColBERT-XM demonstrates strong knowledge transfer and generalization capabilities across languages while trained on a significantly smaller monolingual set.

总体而言，ColBERT - XM在显著更小的单语训练集上进行训练时，展现出了强大的跨语言知识迁移和泛化能力。

### 4.3 Further Analysis

### 4.3 进一步分析

In this section, we conduct a thorough analysis of several key aspects of our proposed methodology, including the influence of greater volumes of training data on ColBERT-XM's performance (§4.3.1), a performance comparison with a modular single-vector representation variant (§4.3.2), the model's ability to generalize to out-of-distribution data (§4.3.3), and its environmental footprint compared to existing multilingual retrievers (§4.3.4).

在本节中，我们对所提出方法的几个关键方面进行了全面分析，包括更多训练数据量对ColBERT - XM性能的影响（§4.3.1）、与模块化单向量表示变体的性能比较（§4.3.2）、模型对分布外数据的泛化能力（§4.3.3），以及与现有多语言检索器相比的环境影响（§4.3.4）。

#### 4.3.1 How does training on more examples affect ColBERT-XM performance?

#### 4.3.1 更多示例训练对ColBERT - XM性能有何影响？

Despite being trained on substantially fewer examples, ColBERT-XM demonstrates competitive results compared to existing multilingual models, raising the question of whether an increased volume of training data would further enhance its performance. To investigate, we train five instances of our modular retriever on a varying number of MS MARCO training triples, namely 3.2M, 6.4M, ${12.8}\mathrm{M},{19.2}\mathrm{M}$ ,and ${25.6}\mathrm{M}$ examples. Figure 3 shows the resulting models' performance on the mMARCO small dev set across MRR@10 and recall at various cut-offs, alongside the fixed performance of ${\mathrm{{mES}}}_{\text{BASE }}$ for comparison. The results reveal an initial performance boost with an increase in training data, which plateaus quickly after ${6.4}\mathrm{M}$ examples,suggesting diminishing returns from additional data of the same distribution. This contrasts with existing baselines that were trained on comparatively more samples from diverse languages to reach their peak performance, thereby underscoring ColBERT-XM's efficiency in low-resource scenarios. For a comprehensive breakdown of performance across individual languages, we refer to Table 4 in Appendix B.

尽管ColBERT - XM的训练示例大幅减少，但与现有的多语言模型相比，它仍取得了有竞争力的结果，这引发了一个问题：增加训练数据量是否会进一步提升其性能。为了研究这一问题，我们在不同数量的MS MARCO训练三元组（即320万、640万、${12.8}\mathrm{M},{19.2}\mathrm{M}$和${25.6}\mathrm{M}$个示例）上训练了我们的模块化检索器的五个实例。图3展示了所得模型在mMARCO小型开发集上的MRR@10和不同截断点的召回率表现，同时展示了${\mathrm{{mES}}}_{\text{BASE }}$的固定性能以供比较。结果显示，随着训练数据的增加，性能最初会有所提升，但在${6.4}\mathrm{M}$个示例之后迅速趋于平稳，这表明相同分布的额外数据带来的回报递减。这与现有的基线模型形成对比，这些基线模型需要在更多不同语言的样本上进行训练才能达到最佳性能，从而凸显了ColBERT - XM在低资源场景下的效率。关于各语言性能的详细分解，请参考附录B中的表4。

<!-- Media -->

<!-- figureText: MRR@10 R@10 R@100 R@1K 26 Number of training examples (million) Number of training examples (million) 2 Number of training examples (million) Number of training examples (million) -->

<img src="https://cdn.noedgeai.com/0195b377-0ecd-714c-9c93-d2f97ba6f989_6.jpg?x=202&y=190&w=1240&h=255&r=0"/>

Figure 3: Performance of ColBERT-XM on mMARCO small dev set, based on the volume of training examples.

图3：基于训练示例数量，ColBERT - XM在mMARCO小型开发集上的性能表现。

<!-- Media -->

#### 4.3.2 How does a single-vector representation variant compare to ColBERT-XM?

#### 4.3.2 单向量表示变体与ColBERT - XM相比如何？

To analyze the effects of single-vector vs. multi-vector representations on our model's performance, we implement a variant of our modular dense retriever that maintains the bi-encoder architecture and modular encoder outlined in Sections 3.1 and 3.2, respectively, yet adopts a different late interaction scoring mechanism that operates on single-vector representations of the input sequences, i.e.,

为了分析单向量与多向量表示对我们模型性能的影响，我们实现了一种模块化密集检索器的变体，该变体分别保持了第3.1节和第3.2节中概述的双编码器架构和模块化编码器，但采用了一种不同的后期交互评分机制，该机制基于输入序列的单向量表示进行操作，即

$$
\operatorname{sim}\left( {{\widehat{\mathbf{H}}}_{q},{\widehat{\mathbf{H}}}_{p}}\right)  = \cos \left( {\operatorname{pool}\left( {\widehat{\mathbf{H}}}_{q}\right) ,\operatorname{pool}\left( {\widehat{\mathbf{H}}}_{p}\right) }\right) ,
$$

(9)

where pool : ${\mathbb{R}}^{k \times  d} \rightarrow  {\mathbb{R}}^{d}$ distills a global representation for the whole text sequence using mean, max, or [CLS] pooling on the corresponding bags of contextualized term embeddings. We train this model, dubbed DPR-XM, on 25.6M MS MARCO triples with a batch size of 128 and learning rate warm up along the first ${10}\%$ of steps to a maximum value of $2\mathrm{e} - 5$ ,after which linear decay is applied.

其中池化（pool）操作：${\mathbb{R}}^{k \times  d} \rightarrow  {\mathbb{R}}^{d}$ 使用均值、最大值或 [CLS] 池化方法，对相应的上下文词嵌入包进行处理，从而为整个文本序列提取全局表示。我们在 2560 万个 MS MARCO 三元组上训练了这个名为 DPR - XM 的模型，批量大小为 128，学习率在前 ${10}\%$ 步内进行预热，直至达到最大值 $2\mathrm{e} - 5$，之后采用线性衰减。

Figure 4 illustrates the comparative performance of our XMOD-based dense retrievers. We observe that ColBERT-XM surpasses DPR-XM in the training language (i.e., English) by 4.5% on MRR@10.Furthermore, it consistently outperforms DPR-XM across the other 13 languages not encountered during training by an average of ${4.9}\%$ . Supported by findings from Santhanam et al. (2022), our results confirm that multi-vector models bypass the restrictive information bottleneck inherent in single-vector models, enabling a richer and more nuanced representation of queries and passages, thereby yielding higher retrieval performance.

图 4 展示了我们基于 XMOD 的密集检索器的对比性能。我们观察到，在训练语言（即英语）上，ColBERT - XM 在 MRR@10 指标上比 DPR - XM 高出 4.5%。此外，在训练过程中未涉及的其他 13 种语言中，它平均比 DPR - XM 高出 ${4.9}\%$。受 Santhanam 等人（2022 年）研究结果的支持，我们的结果证实，多向量模型突破了单向量模型固有的信息瓶颈限制，能够对查询和段落进行更丰富、更细致的表示，从而实现更高的检索性能。

#### 4.3.3 How does ColBERT-XM generalize to out-of-distribution data?

#### 4.3.3 ColBERT - XM 如何泛化到分布外数据？

To assess ColBERT-XM's capabilities for out-of-distribution generalization, we conduct a zero-shot evaluation on Mr. TYDI, encompassing five languages not covered in mMARCO - notably Swahili, Bengali, and Telugu, which are commonly identified as low-resource. Table 2 reports the zero-shot performance of ColBERT-XM alongside the BM25, mT5-based cross-encoder, and mColBERT baselines. We find that ColBERT-XM shows substantial generalization across the out-of-distribution data. While not as effective as the computationally expensive cross-attentional ${\mathrm{{mT5}}}_{\text{BASE }}$ re-ranking model on the rank-aware MRR@100 metrics, ColBERT-XM outperforms its non-modular mColBERT counterpart. Notably, on the rank-unaware R@100 metrics, ColBERT-XM matches closely and even surpasses the more resource-intensive mColBERT and mT5 retrieval models, which have been trained on many more samples and languages. These findings highlight our model's ability to efficiently adapt to domains and languages beyond its original training scope.

为了评估 ColBERT - XM 在分布外数据上的泛化能力，我们在 Mr. TYDI 数据集上进行了零样本评估，该数据集涵盖了 mMARCO 中未包含的五种语言，特别是斯瓦希里语、孟加拉语和泰卢固语，这些语言通常被认为是低资源语言。表 2 报告了 ColBERT - XM 与 BM25、基于 mT5 的交叉编码器和 mColBERT 基线模型的零样本性能。我们发现，ColBERT - XM 在分布外数据上表现出了显著的泛化能力。虽然在考虑排名的 MRR@100 指标上，它不如计算成本高昂的交叉注意力 ${\mathrm{{mT5}}}_{\text{BASE }}$ 重排模型有效，但 ColBERT - XM 优于非模块化的 mColBERT 模型。值得注意的是，在不考虑排名的 R@100 指标上，ColBERT - XM 与资源消耗更大的 mColBERT 和 mT5 检索模型表现相近，甚至超过了它们，而这些模型是在更多的样本和语言上进行训练的。这些发现凸显了我们的模型能够有效适应其原始训练范围之外的领域和语言。

<!-- Media -->

<!-- figureText: ColBERT-XM DPR-XM pt -->

<img src="https://cdn.noedgeai.com/0195b377-0ecd-714c-9c93-d2f97ba6f989_6.jpg?x=997&y=551&w=310&h=342&r=0"/>

Figure 4: MRR@10 results of our multi-vector representation retriever (ColBERT-XM) compared to its single-vector counterpart (DPR-XM) on mMARCO dev set.

图 4：在 mMARCO 开发集上，我们的多向量表示检索器（ColBERT - XM）与单向量对应模型（DPR - XM）的 MRR@10 结果对比。

<!-- Media -->

#### 4.3.4 What is the environmental footprint of ColBERT-XM?

#### 4.3.4 ColBERT - XM 的环境足迹如何？

Given the growing concerns over carbon emissions and climate change, the environmental impact of AI models has become a crucial issue. In a quest for achieving ever-increasing performance, many works prioritize effectiveness over efficiency, leading to models whose training requires significant energy consumption often derived from non-renewable resources, thereby exacerbating the global carbon footprint. Our comparative analysis demonstrates that ColBERT-XM exhibits reduced energy consumption and carbon emissions while performing comparably to leading retrieval models, underscoring its economic and environmental advantages. ${}^{2}$ Table 3 reveals that ColBERT-XM, trained for 7.5 hours only on private infrastructure with a carbon efficiency of ${0.432}\mathrm{\;{kg}}{\mathrm{{CO}}}_{2}\mathrm{{eq}}/\mathrm{{kWh}}$ , utilized only ${2.3}\mathrm{\;{kWh}}$ of power for a carbon footprint of about ${1.01}\mathrm{\;{kg}}\mathrm{{CO}}2\mathrm{{eq}}$ ,which is approximately the amount of emissions produced by burning ${0.5}\mathrm{\;{kg}}$ of coal. This contrasts significantly with competing models like mE5, which, despite its high performance,consumed about ${100} \times$ more power during training (i.e., ${230.4}\mathrm{\;{kWh}}$ ),emitting carbon emissions equivalent to burning ${49.6}\mathrm{\;{kg}}$ of coal. For reference, we compute the estimated carbon emissions ${E}_{\mathrm{c}}$ as

鉴于人们对碳排放和气候变化的担忧日益增加，人工智能模型的环境影响已成为一个关键问题。为了追求不断提高的性能，许多研究工作更注重有效性而非效率，导致一些模型的训练需要消耗大量能源，而这些能源往往来自不可再生资源，从而加剧了全球碳足迹。我们的对比分析表明，ColBERT - XM 在与领先的检索模型性能相当的情况下，能源消耗和碳排放有所降低，凸显了其经济和环境优势。${}^{2}$ 表 3 显示，ColBERT - XM 仅在私有基础设施上训练了 7.5 小时，碳效率为 ${0.432}\mathrm{\;{kg}}{\mathrm{{CO}}}_{2}\mathrm{{eq}}/\mathrm{{kWh}}$，仅消耗了 ${2.3}\mathrm{\;{kWh}}$ 的电力，碳足迹约为 ${1.01}\mathrm{\;{kg}}\mathrm{{CO}}2\mathrm{{eq}}$，这大约相当于燃烧 ${0.5}\mathrm{\;{kg}}$ 煤炭所产生的排放量。这与竞争模型 mE5 形成了鲜明对比，尽管 mE5 性能很高，但在训练过程中消耗的电力大约多 ${100} \times$（即 ${230.4}\mathrm{\;{kWh}}$），碳排放相当于燃烧 ${49.6}\mathrm{\;{kg}}$ 煤炭。作为参考，我们将估计的碳排放 ${E}_{\mathrm{c}}$ 计算为

(10)

<!-- Media -->

<table><tr><td>Model</td><td>Type</td><td>ar</td><td>${bn}$</td><td>en</td><td>${fi}$</td><td>id</td><td>${ja}$</td><td>${ko}$</td><td>ru</td><td>${SW}$</td><td>te</td><td>$\mathbf{{Avg}}$</td></tr><tr><td/><td/><td/><td/><td/><td/><td>MRR@100</td><td/><td/><td/><td/><td/><td/></tr><tr><td>1BM25 (Pyserini)</td><td>LEXICAL</td><td>36.8</td><td>41.8</td><td>14.0</td><td>28.4</td><td>37.6</td><td>21.1</td><td>28.5</td><td>31.3</td><td>38.9</td><td>34.3</td><td>31.3</td></tr><tr><td>2mT5 ${}_{\text{BASE }}$ (Bonifacio et al.,2021)</td><td>CROSS</td><td>62.2</td><td>65.1</td><td>${35.7}^{ \dagger  }$</td><td>49.5</td><td>${61.1}^{ \dagger  }$</td><td>48.1</td><td>47.4</td><td>${\mathbf{{52.6}}}^{ \dagger  }$</td><td>62.9</td><td>66.6</td><td>55.1</td></tr><tr><td>3mColBERT (Bonifacio et al., 2021)</td><td>MULTI</td><td>55.3</td><td>48.8</td><td>${32.9}^{ \dagger  }$</td><td>41.3</td><td>${55.5}^{ \dagger  }$</td><td>36.6</td><td>36.7</td><td>${48.2}^{ \dagger  }$</td><td>44.8</td><td>61.6</td><td>46.1</td></tr><tr><td>4ColBERT-XM (ours)</td><td>MULTI</td><td>55.2</td><td>56.6</td><td>${\mathbf{{36.0}}}^{ \dagger  }$</td><td>41.8</td><td>57.1</td><td>42.1</td><td>41.3</td><td>52.2</td><td>56.8</td><td>50.6</td><td>49.0</td></tr><tr><td colspan="13">R@100</td></tr><tr><td>5BM25 (Pyserini)</td><td>LEXICAL</td><td>79.3</td><td>86.9</td><td>53.7</td><td>71.9</td><td>84.3</td><td>64.5</td><td>61.9</td><td>64.8</td><td>76.4</td><td>75.8</td><td>72.0</td></tr><tr><td>mT5 ${}_{\text{BASE }}$ (Bonifacio et al.,2021)</td><td>CROSS</td><td>88.4</td><td>92.3</td><td>${72.4}^{ \dagger  }$</td><td>85.1</td><td>${92.8}^{ \dagger  }$</td><td>83.2</td><td>76.5</td><td>${76.3}^{ \dagger  }$</td><td>83.8</td><td>85.0</td><td>83.5</td></tr><tr><td>7mColBERT (Bonifacio et al., 2021)</td><td>MULTI</td><td>85.9</td><td>91.8</td><td>${78.6}^{ \dagger  }$</td><td>82.6</td><td>${91.1}^{ \dagger  }$</td><td>70.9</td><td>72.9</td><td>${86.1}^{ \dagger  }$</td><td>80.8</td><td>96.9</td><td>83.7</td></tr><tr><td>8ColBERT-XM (ours)</td><td>MULTI</td><td>89.6</td><td>91.4</td><td>${\mathbf{{83.7}}}^{ \dagger  }$</td><td>84.4</td><td>93.8</td><td>84.9</td><td>77.6</td><td>89.1</td><td>87.1</td><td>93.3</td><td>87.5</td></tr></table>

<table><tbody><tr><td>模型</td><td>类型</td><td>阿拉伯语（ar）</td><td>${bn}$</td><td>英语（en）</td><td>${fi}$</td><td>印尼语（id）</td><td>${ja}$</td><td>${ko}$</td><td>俄语（ru）</td><td>${SW}$</td><td>泰卢固语（te）</td><td>$\mathbf{{Avg}}$</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>前100的平均倒数排名（MRR@100）</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1BM25（Pyserini）</td><td>词法的（LEXICAL）</td><td>36.8</td><td>41.8</td><td>14.0</td><td>28.4</td><td>37.6</td><td>21.1</td><td>28.5</td><td>31.3</td><td>38.9</td><td>34.3</td><td>31.3</td></tr><tr><td>2mT5 ${}_{\text{BASE }}$（博尼法西奥等人，2021年）</td><td>交叉的（CROSS）</td><td>62.2</td><td>65.1</td><td>${35.7}^{ \dagger  }$</td><td>49.5</td><td>${61.1}^{ \dagger  }$</td><td>48.1</td><td>47.4</td><td>${\mathbf{{52.6}}}^{ \dagger  }$</td><td>62.9</td><td>66.6</td><td>55.1</td></tr><tr><td>3mColBERT（博尼法西奥等人，2021年）</td><td>多语言的（MULTI）</td><td>55.3</td><td>48.8</td><td>${32.9}^{ \dagger  }$</td><td>41.3</td><td>${55.5}^{ \dagger  }$</td><td>36.6</td><td>36.7</td><td>${48.2}^{ \dagger  }$</td><td>44.8</td><td>61.6</td><td>46.1</td></tr><tr><td>4ColBERT - XM（我们的）</td><td>多语言的（MULTI）</td><td>55.2</td><td>56.6</td><td>${\mathbf{{36.0}}}^{ \dagger  }$</td><td>41.8</td><td>57.1</td><td>42.1</td><td>41.3</td><td>52.2</td><td>56.8</td><td>50.6</td><td>49.0</td></tr><tr><td colspan="13">R@100</td></tr><tr><td>5BM25（Pyserini）</td><td>词法的（LEXICAL）</td><td>79.3</td><td>86.9</td><td>53.7</td><td>71.9</td><td>84.3</td><td>64.5</td><td>61.9</td><td>64.8</td><td>76.4</td><td>75.8</td><td>72.0</td></tr><tr><td>mT5 ${}_{\text{BASE }}$（博尼法西奥等人，2021年）</td><td>交叉的（CROSS）</td><td>88.4</td><td>92.3</td><td>${72.4}^{ \dagger  }$</td><td>85.1</td><td>${92.8}^{ \dagger  }$</td><td>83.2</td><td>76.5</td><td>${76.3}^{ \dagger  }$</td><td>83.8</td><td>85.0</td><td>83.5</td></tr><tr><td>7mColBERT（博尼法西奥等人，2021年）</td><td>多语言的（MULTI）</td><td>85.9</td><td>91.8</td><td>${78.6}^{ \dagger  }$</td><td>82.6</td><td>${91.1}^{ \dagger  }$</td><td>70.9</td><td>72.9</td><td>${86.1}^{ \dagger  }$</td><td>80.8</td><td>96.9</td><td>83.7</td></tr><tr><td>8ColBERT - XM（我们的）</td><td>多语言的（MULTI）</td><td>89.6</td><td>91.4</td><td>${\mathbf{{83.7}}}^{ \dagger  }$</td><td>84.4</td><td>93.8</td><td>84.9</td><td>77.6</td><td>89.1</td><td>87.1</td><td>93.3</td><td>87.5</td></tr></tbody></table>

Table 2: Out-of-domain retrieval performance on Mr. TYDI test set. All supervised models were fine-tuned on one or more languages from mMARCO. $\dagger$ indicates performance on languages encountered during fine-tuning. The best results are marked in bold, and the second best are underlined.

表2：在Mr. TYDI测试集上的域外检索性能。所有有监督模型都在mMARCO中的一种或多种语言上进行了微调。$\dagger$表示在微调过程中遇到的语言的性能。最佳结果用粗体标记，次佳结果用下划线标记。

<!-- figureText: ${E}_{\mathrm{c}} = \overset{\text{ Power consumption }}{\overbrace{\underset{\begin{matrix} \text{ Thermal } \\  \text{ Design Power } \end{matrix}}{\underbrace{{P}_{\text{TDP }}}} \times  \underset{\begin{matrix} \text{ Training } \\  \text{ efficiency } \end{matrix}}{\underbrace{{T}_{\text{train }}}}}} + \underset{\begin{matrix} \text{ Carbon } \\  \text{ efficiency } \end{matrix}}{\underbrace{{C}_{\text{effi }}}}.$ -->

<img src="https://cdn.noedgeai.com/0195b377-0ecd-714c-9c93-d2f97ba6f989_7.jpg?x=242&y=1537&w=459&h=152&r=0"/>

<!-- Media -->

Our analysis not only highlights the potential for reduced carbon emissions associated with multilingual dense retrievers, but also reflects a deliberate stride toward aligning AI models with the pressing need for environmental sustainability. By demonstrating a comparable performance with a fraction of the energy and carbon output, we hope to set a precedent for future research and development in the field, emphasizing the importance of eco-friendly retrieval systems.

我们的分析不仅凸显了多语言密集检索器在减少碳排放方面的潜力，也体现了我们朝着使人工智能模型满足紧迫的环境可持续性需求迈出的审慎一步。通过展示在消耗一小部分能源和碳排放的情况下达到相当的性能，我们希望为该领域未来的研究和发展树立一个先例，强调环保型检索系统的重要性。

<!-- Media -->

<table><tr><td>Model</td><td>Hardware</td><td>TDP (W)</td><td>Training time (h)</td><td>Power (kWh)</td><td>Emission (kg ${\mathrm{{CO}}}_{2}$ eq)</td></tr><tr><td>1${\mathrm{{mE5}}}_{\text{BASE }}$</td><td>${32} \times$ V100</td><td>300</td><td>24</td><td>230.4</td><td>99.52</td></tr><tr><td>2${\mathrm{{mMiniLM}}}_{\mathrm{L}6}$</td><td>$1 \times  {A100}$</td><td>400</td><td>50</td><td>20.0</td><td>8.64</td></tr><tr><td>3mColBERT</td><td>1 x V100</td><td>300</td><td>36</td><td>10.8</td><td>4.67</td></tr><tr><td>4${\mathrm{{mT5}}}_{\text{BASE }}$</td><td>$1 \times$ TPUv3</td><td>283</td><td>27</td><td>7.6</td><td>3.30</td></tr><tr><td>5ColBERT-XM</td><td>$1 \times  \mathrm{H}{100}$</td><td>310</td><td>7.5</td><td>2.3</td><td>1.01</td></tr></table>

<table><tbody><tr><td>模型</td><td>硬件</td><td>热设计功耗（瓦）</td><td>训练时间（小时）</td><td>耗电量（千瓦时）</td><td>排放量（千克 ${\mathrm{{CO}}}_{2}$ 当量）</td></tr><tr><td>1${\mathrm{{mE5}}}_{\text{BASE }}$</td><td>${32} \times$ V100</td><td>300</td><td>24</td><td>230.4</td><td>99.52</td></tr><tr><td>2${\mathrm{{mMiniLM}}}_{\mathrm{L}6}$</td><td>$1 \times  {A100}$</td><td>400</td><td>50</td><td>20.0</td><td>8.64</td></tr><tr><td>3mColBERT</td><td>1 块 V100</td><td>300</td><td>36</td><td>10.8</td><td>4.67</td></tr><tr><td>4${\mathrm{{mT5}}}_{\text{BASE }}$</td><td>$1 \times$ TPUv3</td><td>283</td><td>27</td><td>7.6</td><td>3.30</td></tr><tr><td>5ColBERT - XM</td><td>$1 \times  \mathrm{H}{100}$</td><td>310</td><td>7.5</td><td>2.3</td><td>1.01</td></tr></tbody></table>

Table 3: Power efficiency and carbon footprint of existing multilingual retrieval models.

表3：现有多语言检索模型的能效和碳足迹。

<!-- Media -->

## 5 Conclusion

## 5 结论

This research presents ColBERT-XM, a multilingual dense retrieval model built upon the XMOD architecture, which effectively learns from monolingual fine-tuning in a high-resource language and performs zero-shot retrieval across multiple languages. Despite being trained solely in English, ColBERT-XM demonstrates competitive performance compared to existing state-of-the-art neural retrievers trained on more extensive datasets in various languages. An in-depth analysis reveals that our modular model learns faster, consumes a fraction of energy, and has a lower carbon footprint than existing multilingual models, thereby balancing its efficacy with environmental sustainabil-ity goals. Additionally, ColBERT-XM generalizes on out-of-distribution data and low-resource languages without further training, performing closely or surpassing strong retrievers. We believe that our research can help build effective retrieval systems for many languages while eliminating the need for language-specific labeled data, thus fostering inclu-sivity and linguistic diversity by helping individuals access information in their native languages.

本研究提出了ColBERT - XM，这是一种基于XMOD架构构建的多语言密集检索模型，它能有效地从高资源语言的单语微调中学习，并能跨多种语言进行零样本检索。尽管仅用英语进行训练，但与在多种语言的更广泛数据集上训练的现有最先进的神经检索器相比，ColBERT - XM表现出了有竞争力的性能。深入分析表明，与现有的多语言模型相比，我们的模块化模型学习速度更快，能耗仅为其一小部分，碳足迹也更低，从而在其有效性和环境可持续性目标之间取得了平衡。此外，ColBERT - XM无需进一步训练就能在分布外数据和低资源语言上实现泛化，表现接近或超越强大的检索器。我们相信，我们的研究有助于为多种语言构建有效的检索系统，同时消除对特定语言标注数据的需求，从而通过帮助个人以母语获取信息来促进包容性和语言多样性。

## Limitations

## 局限性

This section enumerates our work's limitations.

本节列举了我们工作的局限性。

---

<!-- Footnote -->

${}^{2}$ Estimations were conducted using the MachineLearning Impact calculator (Lacoste et al., 2019).

${}^{2}$ 估算使用了机器学习影响计算器（Lacoste等人，2019年）进行。

<!-- Footnote -->

---

Broader evaluation across diverse datasets. While our model's evaluation predominantly relies on the mMARCO dataset (Bonifacio et al., 2021), future investigations could benefit from exploring a broader spectrum of multilingual retrieval datasets, such as MIRACL (Zhang et al., 2022), SWIM-IR (Thakur et al., 2023), and MLDR (Chen et al., 2024). Additionally, examining the model's proficiency in domain-specific retrieval could offer valuable insights into its adaptability to specialized knowledge areas. Unfortunately, such benchmarks are scarce in multilingual contexts.

跨不同数据集的更广泛评估。虽然我们模型的评估主要依赖于mMARCO数据集（Bonifacio等人，2021年），但未来的研究可以从探索更广泛的多语言检索数据集（如MIRACL（Zhang等人，2022年）、SWIM - IR（Thakur等人，2023年）和MLDR（Chen等人，2024年））中受益。此外，检查模型在特定领域检索中的能力可以为其对专业知识领域的适应性提供有价值的见解。不幸的是，在多语言环境中，此类基准测试很少。

Distillation of expressive retrieval models. Instead of the standard pairwise cross-entropy loss employed in ColBERTv1, a KL-divergence loss aimed at distilling the scores from a more sophisticated cross-encoder model, as introduced in Col-BERTv2, could yield notable performance improvement (Santhanam et al., 2022). Nevertheless, our estimates suggest this supervision scheme would require approximately 9.3 times more computational time for training on our system, surpassing our current resource allocation. As such, we let this exploration for future work.

表达性检索模型的蒸馏。与ColBERTv1中采用的标准成对交叉熵损失不同，如Col - BERTv2中引入的，旨在从更复杂的交叉编码器模型中蒸馏分数的KL散度损失可能会显著提高性能（Santhanam等人，2022年）。然而，我们的估计表明，这种监督方案在我们的系统上训练大约需要多9.3倍的计算时间，超出了我们目前的资源分配。因此，我们将这一探索留待未来工作。

Adaptability to cross-lingual retrieval. While this study presents a multilingual model designed for information retrieval within the same language, investigating its cross-lingual retrieval capabilities - i.e., identifying relevant passages in a target language based on queries in a different source language - represents a compelling direction for future research, especially in light of increasing needs for systems that can transcend language barriers.

对跨语言检索的适应性。虽然本研究提出了一个用于同一语言内信息检索的多语言模型，但研究其跨语言检索能力（即根据不同源语言的查询识别目标语言中的相关段落）是未来研究的一个有吸引力的方向，特别是考虑到对能够跨越语言障碍的系统的需求不断增加。

Model interpretability. Enhancing the interpretability of dense retrieval model predictions is essential for boosting user confidence and ensuring system transparency, particularly given the complex linguistic and cultural nuances present in multilingual contexts. Building on seminal works in the area (Sudhi et al., 2022; Anand et al., 2023), our future efforts will focus on deepening our understanding of ColBERT-XM's decision-making mechanisms through detailed analysis of the model's interpretability features.

模型可解释性。提高密集检索模型预测的可解释性对于增强用户信心和确保系统透明度至关重要，特别是考虑到多语言环境中存在的复杂语言和文化细微差别。基于该领域的开创性工作（Sudhi等人，2022年；Anand等人，2023年），我们未来的工作将通过详细分析模型的可解释性特征，专注于加深对ColBERT - XM决策机制的理解。

## Broader Impacts

## 更广泛的影响

In this section, we delve into the ethical considerations, societal implications, and potential risks of our proposed methodology.

在本节中，我们深入探讨了我们提出的方法的伦理考量、社会影响和潜在风险。

Ethical considerations. Our work mostly leverages the widely recognized MS MARCO dataset, which contains over half a million anonymized queries collected from Bing's search logs, ensuring that our data sourcing practices are ethical and protect individual privacy. By leveraging mMARCO's direct translations, we ensure a fair and unbiased distribution of samples across languages, thereby avoiding the reinforcement of stereotypes. Furthermore, the combination of automated translation and manual labeling of the dataset ensures the reliability and precision of the ground truth data. This approach is essential for reducing label bias, which can arise from human annotators' varying proficiency levels and backgrounds.

伦理考量。我们的工作主要利用了广泛认可的MS MARCO数据集，该数据集包含从必应搜索日志中收集的超过50万个匿名查询，确保了我们的数据来源实践符合伦理并保护个人隐私。通过利用mMARCO的直接翻译，我们确保了样本在各语言间的公平和无偏分布，从而避免了刻板印象的强化。此外，数据集的自动翻译和手动标注相结合，确保了真实数据的可靠性和精确性。这种方法对于减少标签偏差至关重要，标签偏差可能源于人类标注者不同的熟练程度和背景。

Societal implications. Multilingual retrieval models significantly impact society by reducing language barriers and improving information accessibility for all. Our research aims to foster inclu-sivity and linguistic diversity, helping non-English speakers and those desiring information in their native languages. By developing models capable of effectively retrieving information in lesser-used languages, we contribute to equitable learning opportunities worldwide, enable businesses to serve a diverse international clientele, and prevent the digital marginalization of linguistic minorities.

社会影响。多语言检索模型通过减少语言障碍和提高所有人的信息可获取性，对社会产生了重大影响。我们的研究旨在促进包容性和语言多样性，帮助非英语使用者和那些希望以母语获取信息的人。通过开发能够有效检索较少使用语言信息的模型，我们为全球公平的学习机会做出了贡献，使企业能够服务于多样化的国际客户群体，并防止语言少数群体在数字领域被边缘化。

Potential misuse. The premature deployment of a modular retrieval system presents a few risks. Notably, flaws or biases acquired during the monolingual fine-tuning phase could be inadvertently propagated to other languages when performing zero-shot transfer, thus perpetuating these malfunctions. More generally, the integrity of the underlying knowledge corpus is crucial, as even an effective system may retrieve relevant yet factually inaccurate content, thus unwittingly spreading misinformation. These concerns underscore the need for interpretability of retrieval model predictions to bolster user trust in such systems, which ColBERT-XM addresses with its interpretable MaxSim-based scoring mechanism.

潜在滥用。模块化检索系统的过早部署存在一些风险。值得注意的是，在单语微调阶段获得的缺陷或偏差可能会在进行零样本迁移时无意中传播到其他语言，从而使这些故障持续存在。更普遍地说，基础知识语料库的完整性至关重要，因为即使是一个有效的系统也可能检索到相关但事实不准确的内容，从而在不知不觉中传播错误信息。这些担忧凸显了检索模型预测可解释性对于增强用户对这类系统信任的必要性，ColBERT - XM通过其基于MaxSim的可解释评分机制解决了这一问题。

## Acknowledgments

## 致谢

This research is partially supported by the Sector Plan Digital Legal Studies of the Dutch Ministry of Education, Culture, and Science. In addition, this research was made possible, in part, using the Data Science Research Infrastructure (DSRI) hosted at Maastricht University.

本研究部分得到了荷兰教育、文化和科学部数字法律研究部门计划的支持。此外，本研究部分得益于马斯特里赫特大学托管的数据科学研究基础设施（DSRI）才得以开展。

## References

## 参考文献

Avishek Anand, Procheta Sen, Sourav Saha, Manisha Verma, and Mandar Mitra. 2023. Explainable information retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3448- 3451. ACM. [Page 9]

阿维谢克·阿南德（Avishek Anand）、普罗切塔·森（Procheta Sen）、苏拉夫·萨哈（Sourav Saha）、玛尼沙·维尔马（Manisha Verma）和曼达尔·米特拉（Mandar Mitra）。2023年。可解释的信息检索。见《第46届ACM SIGIR国际信息检索研究与发展会议论文集》，第3448 - 3451页。美国计算机协会（ACM）。[第9页]

Alan Ansell, Edoardo Maria Ponti, Jonas Pfeiffer, Sebastian Ruder, Goran Glavas, Ivan Vulic, and Anna Korhonen. 2021. MAD-G: multilingual adapter generation for efficient cross-lingual transfer. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4762-4781. ACL. [Page 3]

艾伦·安塞尔（Alan Ansell）、爱德华多·玛丽亚·庞蒂（Edoardo Maria Ponti）、乔纳斯·普法伊费尔（Jonas Pfeiffer）、塞巴斯蒂安·鲁德尔（Sebastian Ruder）、戈兰·格拉瓦斯（Goran Glavas）、伊万·武利奇（Ivan Vulic）和安娜·科尔霍宁（Anna Korhonen）。2021年。MAD - G：用于高效跨语言迁移的多语言适配器生成。见《计算语言学协会研究成果：2021年自然语言处理经验方法会议》，第4762 - 4781页。计算语言学协会（ACL）。[第3页]

Ankur Bapna and Orhan Firat. 2019. Simple, scalable adaptation for neural machine translation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 1538-1548. ACL. [Page 3]

安库尔·巴普纳（Ankur Bapna）和奥尔汗·菲拉特（Orhan Firat）。2019年。神经机器翻译的简单可扩展自适应方法。见《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议论文集》，第1538 - 1548页。计算语言学协会（ACL）。[第3页]

Lukas Biewald. 2020. Experiment tracking with weights and biases. Software available from wandb.com. [Page 5]

卢卡斯·比瓦尔德（Lukas Biewald）。2020年。使用Weights & Biases进行实验跟踪。软件可从wandb.com获取。[第5页]

Luiz Henrique Bonifacio, Israel Campiotti, Roberto de Alencar Lotufo, and Rodrigo Nogueira. 2021. mmarco: A multilingual version of MS MARCO passage ranking dataset. CoRR, abs/2108.13897. [Pages 5, 6, 8, and 9]

路易斯·恩里克·博尼法西奥（Luiz Henrique Bonifacio）、伊斯雷尔·坎皮奥蒂（Israel Campiotti）、罗伯托·德·阿伦卡尔·洛图福（Roberto de Alencar Lotufo）和罗德里戈·诺盖拉（Rodrigo Nogueira）。2021年。mmarco：MS MARCO段落排名数据集的多语言版本。计算机研究报告库（CoRR），编号abs/2108.13897。[第5、6、8和9页]

Tyler A. Chang, Catherine Arnett, Zhuowen Tu, and Benjamin K. Bergen. 2023. When is multilinguality a curse? language modeling for 250 high- and low-resource languages. CoRR, abs/2311.09205. [Page 3]

泰勒·A·张（Tyler A. Chang）、凯瑟琳·阿尼特（Catherine Arnett）、涂卓文（Zhuowen Tu）和本杰明·K·伯根（Benjamin K. Bergen）。2023年。多语言何时会成为障碍？针对250种高资源和低资源语言的语言建模。计算机研究报告库（CoRR），编号abs/2311.09205。[第3页]

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. CoRR, abs/2402.03216. [Page 9]

陈健律、肖世涛、张培天、罗坤、连德富和刘政。2024年。Bge m3 - 嵌入：通过自我知识蒸馏实现的多语言、多功能、多粒度文本嵌入。计算机研究报告库（CoRR），编号abs/2402.03216。[第9页]

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettle-moyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440-8451. ACL. [Pages 1, 3, and 6]

亚历克西斯·科诺（Alexis Conneau）、卡蒂凯·坎德尔瓦尔（Kartikay Khandelwal）、纳曼·戈亚尔（Naman Goyal）、维什拉夫·乔杜里（Vishrav Chaudhary）、纪尧姆·温泽克（Guillaume Wenzek）、弗朗西斯科·古兹曼（Francisco Guzmán）、爱德华·格雷夫（Edouard Grave）、迈尔·奥特（Myle Ott）、卢克·泽特尔莫耶（Luke Zettle - moyer）和韦塞林·斯托亚诺夫（Veselin Stoyanov）。2020年。大规模无监督跨语言表征学习。见《计算语言学协会第58届年会论文集》，第8440 - 8451页。计算语言学协会（ACL）。[第1、3和6页]

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4171-4186. ACL. [Pages 1, 3, and 6]

雅各布·德夫林（Jacob Devlin）、张明伟（Ming - Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练。见《计算语言学协会北美分会2019年会议：人类语言技术》，第4171 - 4186页。计算语言学协会（ACL）。[第1、3和6页]

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE: sparse lexical and expansion model for first stage ranking. In Proceedings of the 44th International ACM SIGIR conference on research and development in Information Retrieval, pages 2288-2292. ACM. [Page 3]

蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。SPLADE：用于第一阶段排名的稀疏词法和扩展模型。见《第44届ACM SIGIR国际信息检索研究与发展会议论文集》，第2288 - 2292页。美国计算机协会（ACM）。[第3页]

Daniel Gillick, Alessandro Presta, and Gaurav Singh Tomar. 2018. End-to-end retrieval in continuous

丹尼尔·吉利克（Daniel Gillick）、亚历山德罗·普雷斯塔（Alessandro Presta）和高拉夫·辛格·托马尔（Gaurav Singh Tomar）。2018年。连续空间中的端到端检索

space. CoRR, abs/1811.08008. [Page 3]

。计算机研究报告库（CoRR），编号abs/1811.08008。[第3页]

Ruidan He, Linlin Liu, Hai Ye, Qingyu Tan, Bosheng Ding, Liying Cheng, Jia-Wei Low, Lidong Bing, and Luo Si. 2021. On the effectiveness of adapter-based tuning for pretrained language model adaptation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pages 2208-2222. ACL. [Page 3]

何瑞丹、刘琳琳、叶海、谭清玉、丁博生、程丽英、罗佳伟、邴立东和司罗。2021年。基于适配器的微调对预训练语言模型自适应的有效性研究。见《计算语言学协会第59届年会和第11届自然语言处理国际联合会议论文集》，第2208 - 2222页。计算语言学协会（ACL）。[第3页]

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Ges-mundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, pages 2790-2799. PMLR. [Pages 3 and 4]

尼尔·霍尔兹比（Neil Houlsby）、安德烈·久尔久（Andrei Giurgiu）、斯坦尼斯瓦夫·亚斯特热布斯基（Stanislaw Jastrzebski）、布鲁娜·莫罗内（Bruna Morrone）、昆汀·德拉鲁西耶（Quentin de Laroussilhe）、安德里亚·杰斯蒙多（Andrea Gesmundo）、莫娜·阿塔里扬（Mona Attariyan）和西尔万·热利（Sylvain Gelly）。2019年。自然语言处理的参数高效迁移学习。见《第36届国际机器学习会议论文集》，第2790 - 2799页。机器学习研究会议录（PMLR）。[第3页和第4页]

David A. Hull and Gregory Grefenstette. 1996. Querying across languages: A dictionary-based approach to multilingual information retrieval. In Proceedings of the 19th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 49-57. ACM. [Page 2]

大卫·A·赫尔（David A. Hull）和格雷戈里·格雷芬施泰特（Gregory Grefenstette）。1996年。跨语言查询：一种基于词典的多语言信息检索方法。见《第19届年度国际计算机协会信息检索研究与发展会议论文集》，第49 - 57页。美国计算机协会（ACM）。[第2页]

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535-547. [Page 5]

杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2021年。基于图形处理器（GPU）的十亿级相似度搜索。《电气与电子工程师协会大数据汇刊》（IEEE Transactions on Big Data），7(3):535 - 547。[第5页]

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 6769-6781. ACL. [Page 3]

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世文（Sewon Min）、帕特里克·S·H·刘易斯（Patrick S. H. Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wentau Yih）。2020年。开放域问答的密集段落检索。见《2020年自然语言处理经验方法会议论文集》，第6769 - 6781页。计算语言学协会（ACL）。[第3页]

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextual-ized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39-48. ACM. [Pages 2, 3, and 5]

奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。科尔伯特（Colbert）：通过基于BERT的上下文延迟交互实现高效有效的段落搜索。见《第43届国际计算机协会信息检索研究与发展会议论文集》，第39 - 48页。美国计算机协会（ACM）。[第2页、第3页和第5页]

Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. 2019. Quantifying the carbon emissions of machine learning. CoRR, abs/1910.09700. [Page 8]

亚历山大·拉科斯特（Alexandre Lacoste）、亚历山德拉·卢乔尼（Alexandra Luccioni）、维克多·施密特（Victor Schmidt）和托马斯·丹德雷斯（Thomas Dandres）。2019年。量化机器学习的碳排放。计算机研究存储库（CoRR），abs/1910.09700。[第8页]

Dawn J. Lawrie, Eugene Yang, Douglas W. Oard, and James Mayfield. 2023. Neural approaches to multilingual information retrieval. In Proceedings of the 45th European Conference on Information Retrieval, pages 521-536. Springer. [Page 3]

道恩·J·劳里（Dawn J. Lawrie）、杨宇（Eugene Yang）、道格拉斯·W·奥尔德（Douglas W. Oard）和詹姆斯·梅菲尔德（James Mayfield）。2023年。多语言信息检索的神经方法。见《第45届欧洲信息检索会议论文集》，第521 - 536页。施普林格出版社（Springer）。[第3页]

Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Y. Zhao. 2023. Rethinking the role of token retrieval in multi-vector retrieval. CoRR, abs/2304.01982. [Page 3]

李镇赫（Jinhyuk Lee）、戴竹云（Zhuyun Dai）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、雷涛（Tao Lei）、伊夫泰哈尔·奈姆（Iftekhar Naim）、张明伟（Ming-Wei Chang）和赵文森（Vincent Y. Zhao）。2023年。重新思考多向量检索中词元检索的作用。计算机研究存储库（CoRR），abs/2304.01982。[第3页]

Jimmy Lin and Xueguang Ma. 2021. A few brief notes on deepimpact, coil, and a conceptual framework for information retrieval techniques. CoRR, abs/2106.14807. [Page 3]

林吉米（Jimmy Lin）和马学光（Xueguang Ma）。2021年。关于深度影响（deepimpact）、线圈（coil）以及信息检索技术概念框架的几点简要说明。计算机研究存储库（CoRR），abs/2106.14807。[第3页]

Jimmy Lin, Rodrigo Frassetto Nogueira, and Andrew Yates. 2021. Pretrained Transformers for Text Ranking: ${BERT}$ and Beyond. Synthesis Lectures on Human Language Technologies. Morgan & Claypool Publishers. [Page 1]

林吉米（Jimmy Lin）、罗德里戈·弗拉塞托·诺盖拉（Rodrigo Frassetto Nogueira）和安德鲁·耶茨（Andrew Yates）。2021年。用于文本排序的预训练Transformer：${BERT}$及其他。《人类语言技术综合讲座》。摩根与克莱普尔出版社（Morgan & Claypool Publishers）。[第1页]

Robert Litschko, Ivan Vulic, and Goran Glavas. 2022. Parameter-efficient neural reranking for cross-lingual and multilingual retrieval. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1071-1082. International Committee on Computational Linguistics. [Page 3]

罗伯特·利奇科（Robert Litschko）、伊万·武利奇（Ivan Vulic）和戈兰·格拉瓦斯（Goran Glavas）。2022年。跨语言和多语言检索的参数高效神经重排序。见《第29届国际计算语言学会议论文集》，第1071 - 1082页。国际计算语言学委员会。[第3页]

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. In Proceedings of the 7th International Conference on Learning Representations. [Page 5]

伊利亚·洛希洛夫（Ilya Loshchilov）和弗兰克·胡特（Frank Hutter）。2017年。解耦权重衰减正则化。见《第7届国际学习表征会议论文集》。[第5页]

Sean MacAvaney, Luca Soldaini, and Nazli Goharian. 2020. Teaching a new dog old tricks: Resurrecting multilingual retrieval using zero-shot learning. In Proceedings of the 42nd European Conference on Information Retrieval, pages 246-254. Springer. [Page 1]

肖恩·麦卡瓦尼（Sean MacAvaney）、卢卡·索尔代尼（Luca Soldaini）和纳兹利·戈哈里安（Nazli Goharian）。2020年。教新狗学旧把戏：利用零样本学习复兴多语言检索。见《第42届欧洲信息检索会议论文集》，第246 - 254页。施普林格出版社（Springer）。[第1页]

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. MTEB: massive text embedding benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 2006-2029. ACL. [Page 1]

尼克拉斯·米宁霍夫（Niklas Muennighoff）、努阿曼·塔齐（Nouamane Tazi）、洛里克·马涅（Loïc Magne）和尼尔斯·赖默斯（Nils Reimers）。2023年。MTEB：大规模文本嵌入基准。见《计算语言学协会欧洲分会第17届会议论文集》，第2006 - 2029页。计算语言学协会（ACL）。[第1页]

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2018. MS MARCO: A human generated machine reading comprehension dataset. CoRR, abs/1611.09268v3. [Page 5]

阮特里（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）和李登（Li Deng）。2018年。MS MARCO：一个人工生成的机器阅读理解数据集。计算机研究存储库（CoRR），abs/1611.09268v3。[第5页]

Rodrigo Frassetto Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Document ranking with a pretrained sequence-to-sequence model. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 708-718. ACL. [Page 3]

罗德里戈·弗拉塞托·诺盖拉（Rodrigo Frassetto Nogueira）、蒋志英（Zhiying Jiang）、罗纳克·普拉迪普（Ronak Pradeep）和吉米·林（Jimmy Lin）。2020年。使用预训练的序列到序列模型进行文档排名。见《计算语言学协会成果：2020年自然语言处理经验方法会议》，第708 - 718页。计算语言学协会。[第3页]

Rodrigo Frassetto Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. 2019. Multi-stage document ranking with BERT. CoRR, abs/1910.14424. [Pages 3 and 4]

罗德里戈·弗拉塞托·诺盖拉（Rodrigo Frassetto Nogueira）、杨威（Wei Yang）、赵京焕（Kyunghyun Cho）和吉米·林（Jimmy Lin）。2019年。使用BERT进行多阶段文档排名。预印本论文库（CoRR），编号：abs/1910.14424。[第3页和第4页]

Jonas Pfeiffer, Naman Goyal, Xi Victoria Lin, Xian Li, James Cross, Sebastian Riedel, and Mikel Artetxe. 2022. Lifting the curse of multilinguality by pretraining modular transformers. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3479-3495. ACL. [Pages 1, 3, and 4]

乔纳斯·普法伊费尔（Jonas Pfeiffer）、纳曼·戈亚尔（Naman Goyal）、林希·维多利亚（Xi Victoria Lin）、李贤（Xian Li）、詹姆斯·克罗斯（James Cross）、塞巴斯蒂安·里德尔（Sebastian Riedel）和米克尔·阿泰特克（Mikel Artetxe）。2022年。通过预训练模块化Transformer解除多语言的诅咒。见《计算语言学协会北美分会2022年会议：人类语言技术》论文集，第3479 - 3495页。计算语言学协会。[第1页、第3页和第4页]

Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, and Sebastian Ruder. 2020. MAD-X: an adapter-based framework for multi-task cross-lingual transfer. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 7654- 7673. ACL. [Page 3]

乔纳斯·普法伊费尔（Jonas Pfeiffer）、伊万·武利奇（Ivan Vulic）、伊琳娜·古列维奇（Iryna Gurevych）和塞巴斯蒂安·鲁德尔（Sebastian Ruder）。2020年。MAD - X：一种基于适配器的多任务跨语言迁移框架。见《2020年自然语言处理经验方法会议》论文集，第7654 - 7673页。计算语言学协会。[第3页]

Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. 2017. Learning multiple visual domains with residual adapters. Advances in Neural Information Processing Systems, 30:506-516. [Page 3]

西尔维斯特雷 - 阿尔维斯·雷比菲（Sylvestre - Alvise Rebuffi）、哈坎·比伦（Hakan Bilen）和安德里亚·韦尔代利（Andrea Vedaldi）。2017年。使用残差适配器学习多个视觉领域。《神经信息处理系统进展》，30：506 - 516。[第3页]

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 3980-3990. ACL. [Pages 4 and 5]

尼尔斯·赖默斯（Nils Reimers）和伊琳娜·古列维奇（Iryna Gurevych）。2019年。句子BERT：使用孪生BERT网络的句子嵌入。见《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议》论文集，第3980 - 3990页。计算语言学协会。[第4页和第5页]

Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. 1994. Okapi at TREC-3. In Proceedings of the 3rd Text REtrieval Conference, volume 500-225 of NIST Special Publication, pages 109-126. National Institute of Standards and Technology. [Pages 3 and 5]

斯蒂芬·E·罗伯逊（Stephen E. Robertson）、史蒂夫·沃克（Steve Walker）、苏珊·琼斯（Susan Jones）、米歇琳·汉考克 - 博勒伊（Micheline Hancock - Beaulieu）和迈克·加特福德（Mike Gatford）。1994年。TREC - 3会议上的Okapi系统。见《第3届文本检索会议》论文集，美国国家标准与技术研究院特别出版物第500 - 225卷，第109 - 126页。美国国家标准与技术研究院。[第3页和第5页]

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. Col-bertv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3715-3734. ACL. [Pages 5, 7, 9, and 12]

凯沙夫·桑塔南（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马特伊·扎哈里亚（Matei Zaharia）。2022年。Col - bertv2：通过轻量级后期交互实现高效有效的检索。见《计算语言学协会北美分会2022年会议：人类语言技术》论文集，第3715 - 3734页。计算语言学协会。[第5页、第7页、第9页和第12页]

Viju Sudhi, Sabine Wehnert, Norbert Michael Hom-ner, Sebastian Ernst, Mark Gonter, Andreas Krug, and Ernesto William De Luca. 2022. Bite-rex: An explainable bilingual text retrieval system in the automotive domain. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3251- 3255. ACM. [Page 9]

维朱·苏迪（Viju Sudhi）、萨宾·韦纳特（Sabine Wehnert）、诺伯特·迈克尔·霍姆纳（Norbert Michael Hom - ner）、塞巴斯蒂安·恩斯特（Sebastian Ernst）、马克·贡特（Mark Gonter）、安德里亚斯·克鲁格（Andreas Krug）和埃内斯托·威廉·德·卢卡（Ernesto William De Luca）。2022年。Bite - rex：汽车领域的可解释双语文本检索系统。见《第45届ACM信息检索研究与发展国际会议》论文集，第3251 - 3255页。美国计算机协会。[第9页]

Nandan Thakur, Jianmo Ni, Gustavo Hernández Ábrego, John Wieting, Jimmy Lin, and Daniel Cer. 2023. Leveraging llms for synthesizing training data across many languages in multilingual dense retrieval. CoRR, abs/2311.05800. [Page 9]

南丹·塔库尔（Nandan Thakur）、倪建谟（Jianmo Ni）、古斯塔沃·埃尔南德斯·阿布雷戈（Gustavo Hernández Ábrego）、约翰·维廷（John Wieting）、吉米·林（Jimmy Lin）和丹尼尔·塞尔（Daniel Cer）。2023年。在多语言密集检索中利用大语言模型合成多种语言的训练数据。预印本论文库（CoRR），编号：abs/2311.05800。[第9页]

Liang Wang, Nan Yang, Xiaolong Huang, Binx-ing Jiao, Linjun Yang, Daxin Jiang, Rangan Ma-jumder, and Furu Wei. 2022. Text embeddings by weakly-supervised contrastive pre-training. CoRR, abs/2212.03533. [Page 6]

王亮（Liang Wang）、杨楠（Nan Yang）、黄晓龙（Xiaolong Huang）、焦彬星（Binxing Jiao）、杨林军（Linjun Yang）、蒋大新（Daxin Jiang）、兰甘·马宗德（Rangan Majumder）和魏富如（Furu Wei）。2022年。通过弱监督对比预训练进行文本嵌入。预印本论文库（CoRR），编号：abs/2212.03533。[第6页]

Wenhui Wang, Hangbo Bao, Shaohan Huang, Li Dong, and Furu Wei. 2021. Minilmv2: Multi-head self-attention relation distillation for compressing pre-trained transformers. In Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, pages 2140-2151. ACL. [Page 5]

王文慧（Wenhui Wang）、鲍航波（Hangbo Bao）、黄少涵（Shaohan Huang）、李东（Li Dong）和魏富如（Furu Wei）。2021年。MiniLMv2：用于压缩预训练Transformer的多头自注意力关系蒸馏。见《计算语言学协会成果：2021年ACL/IJCNLP会议》，第2140 - 2151页。计算语言学协会。[第5页]

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pier-ric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38-45. Association for Computational Linguistics. [Page 5]

托马斯·沃尔夫（Thomas Wolf）、利桑德尔·迪布（Lysandre Debut）、维克多·桑（Victor Sanh）、朱利安·肖蒙（Julien Chaumond）、克莱门特·德朗格（Clement Delangue）、安东尼·莫伊（Anthony Moi）、皮埃尔 - 里克·西斯塔克（Pier - ric Cistac）、蒂姆·劳（Tim Rault）、雷米·卢夫（Rémi Louf）、摩根·丰托维奇（Morgan Funtowicz）、乔·戴维森（Joe Davison）、山姆·施莱弗（Sam Shleifer）、帕特里克·冯·普拉滕（Patrick von Platen）、克拉拉·马（Clara Ma）、亚辛·杰尔尼（Yacine Jernite）、朱利安·普鲁（Julien Plu）、徐灿文（Canwen Xu）、特文·勒·斯考（Teven Le Scao）、西尔万·古格（Sylvain Gugger）、玛丽亚玛·德拉梅（Mariama Drame）、昆汀·勒斯特（Quentin Lhoest）和亚历山大·M·拉什（Alexander M. Rush）。2020年。Transformer：最先进的自然语言处理技术。见《2020年自然语言处理经验方法会议：系统演示文集》，第38 - 45页。计算语言学协会。[第5页]

Shijie Wu and Mark Dredze. 2020. Are all languages created equal in multilingual bert? In Proceedings of the 5th Workshop on Representation Learning for ${NLP}$ ,pages 120-130. ACL. [Page 3]

吴世杰（Shijie Wu）和马克·德雷兹（Mark Dredze）。2020年。多语言BERT中所有语言是否平等？见《第五届${NLP}$表示学习研讨会文集》，第120 - 130页。计算语言学协会。[第3页]

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighof. 2023. C-pack: Packaged resources to advance general chinese embedding. CoRR, abs/2309.07597. [Page 1]

肖诗涛（Shitao Xiao）、刘政（Zheng Liu）、张培天（Peitian Zhang）和尼克拉斯·米宁霍夫（Niklas Muennighof）。2023年。C - pack：推动通用中文嵌入的打包资源。计算机研究存储库（CoRR），论文编号：abs/2309.07597。[第1页]

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In Proceedings of the 9th International Conference on Learning Representations. OpenRe-view.net. [Page 3]

李雄（Lee Xiong）、熊晨燕（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·N·贝内特（Paul N. Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2021年。用于密集文本检索的近似最近邻负对比学习。见《第九届学习表示国际会议文集》。OpenReview.net。[第3页]

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mt5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483-498. ACL. [Page 5]

薛林婷（Linting Xue）、诺亚·康斯坦特（Noah Constant）、亚当·罗伯茨（Adam Roberts）、米希尔·卡尔（Mihir Kale）、拉米·阿尔 - 鲁福（Rami Al - Rfou）、阿迪蒂亚·西丹特（Aditya Siddhant）、阿迪蒂亚·巴鲁阿（Aditya Barua）和科林·拉菲尔（Colin Raffel）。2021年。mt5：一个大规模多语言预训练的文本到文本Transformer。见《2021年计算语言学协会北美分会会议：人类语言技术文集》，第483 - 498页。计算语言学协会。[第5页]

Eugene Yang, Suraj Nair, Ramraj Chandradevan, Rebecca Iglesias-Flores, and Douglas W. Oard. 2022a. C3: continued pretraining with contrastive weak supervision for cross language ad-hoc retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2507-2512. ACM. [Page 6]

尤金·杨（Eugene Yang）、苏拉杰·奈尔（Suraj Nair）、拉姆拉杰·钱德拉德万（Ramraj Chandradevan）、丽贝卡·伊格莱西亚斯 - 弗洛雷斯（Rebecca Iglesias - Flores）和道格拉斯·W·奥尔德（Douglas W. Oard）。2022a。C3：用于跨语言临时检索的对比弱监督持续预训练。见《第45届ACM SIGIR国际信息检索研究与发展会议文集》，第2507 - 2512页。美国计算机协会（ACM）。[第6页]

Eugene Yang, Suraj Nair, Dawn J. Lawrie, James Mayfield, and Douglas W. Oard. 2022b. Parameter-efficient zero-shot transfer for cross-language dense retrieval with adapters. CoRR, abs/2212.10448. [Page 3]

尤金·杨（Eugene Yang）、苏拉杰·奈尔（Suraj Nair）、道恩·J·劳里（Dawn J. Lawrie）、詹姆斯·梅菲尔德（James Mayfield）和道格拉斯·W·奥尔德（Douglas W. Oard）。2022b。使用适配器进行跨语言密集检索的参数高效零样本迁移。计算机研究存储库（CoRR），论文编号：abs/2212.10448。[第3页]

Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin. 2021. Mr. tydi: A multi-lingual benchmark for dense retrieval. CoRR, abs/2108.08787. [Page 5]

张新宇（Xinyu Zhang）、马学光（Xueguang Ma）、史鹏（Peng Shi）和吉米·林（Jimmy Lin）。2021年。Mr. tydi：一个用于密集检索的多语言基准。计算机研究存储库（CoRR），论文编号：abs/2108.08787。[第5页]

Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xi-aoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. 2022. Making a MIRACL: multilingual information retrieval across a continuum of languages. CoRR, abs/2210.09984. [Page 9]

张新宇（Xinyu Zhang）、南丹·塔库尔（Nandan Thakur）、奥敦ayo·奥贡德波（Odunayo Ogundepo）、埃桑·卡马卢（Ehsan Kamalloo）、大卫·阿方索 - 赫尔梅洛（David Alfonso - Hermelo）、李晓光（Xiaoguang Li）、刘群（Qun Liu）、梅赫迪·雷扎戈利扎德（Mehdi Rezagholizadeh）和吉米·林（Jimmy Lin）。2022年。打造MIRACL：跨连续语言的多语言信息检索。计算机研究存储库（CoRR），论文编号：abs/2210.09984。[第9页]

## A Centroid-based Indexing

## 基于质心的索引

ColBERTv2's centroid-based indexing consists of three main stages (Santhanam et al., 2022).

ColBERTv2的基于质心的索引包括三个主要阶段（桑塔纳姆等人，2022年）。

First,we select a set of cluster centroids $\mathcal{C} =$ ${\left\{  {\mathbf{c}}_{j} \in  {\mathbb{R}}^{{d}_{\text{out }}}\right\}  }_{j = 1}^{N}$ of size $N$ ,proportional to the square root of the estimated number of term em-beddings across the entire passage collection, by applying $k$ -means clustering to the contextualized term embeddings of only a sample of all passages.

首先，我们通过对所有段落样本的上下文词嵌入应用$k$ - 均值聚类，选择一组大小为$N$的聚类质心$\mathcal{C} =$ ${\left\{  {\mathbf{c}}_{j} \in  {\mathbb{R}}^{{d}_{\text{out }}}\right\}  }_{j = 1}^{N}$，该大小与整个段落集合中估计的词嵌入数量的平方根成正比。

Then, every passage in the corpus is processed using the modular language representation model, as detailed in Section 3.2, and the resulting contex-tualized term embeddings are assigned the identifier of the closest centroid ${\mathbf{c}}_{j} \in  \mathcal{C}$ ,which requires $\left\lceil  {{\log }_{2}\left| \mathcal{C}\right| }\right\rceil$ bits to be encoded. Additionally,a residual representation ${\mathbf{r}}_{i}^{p} \in  {\mathbb{R}}^{{d}_{\text{out }}}$ is computed for each term embedding to facilitate its reconstruction given ${\mathbf{r}}_{i}^{p} = {\widehat{\mathbf{h}}}_{i}^{p} - {\mathbf{c}}_{j}$ . To enhance storage efficiency, each dimension of this residual vector is quantized into 2-bit values. Consequently, storing each term vector requires $2{d}_{\text{out }} + \left\lceil  {{\log }_{2}\left| \mathcal{C}\right| }\right\rceil$ bits,i.e. roughly $7 \times$ less than the ${16}{d}_{\text{out }}$ bits needed for the 16 -bit precision compression used in ColBERTv1, without compromising on retrieval quality.

然后，使用模块化语言表示模型处理语料库中的每个段落，具体细节见3.2节。将得到的上下文相关词项嵌入分配给最接近的质心标识符 ${\mathbf{c}}_{j} \in  \mathcal{C}$，这需要 $\left\lceil  {{\log }_{2}\left| \mathcal{C}\right| }\right\rceil$ 位进行编码。此外，为每个词项嵌入计算一个残差表示 ${\mathbf{r}}_{i}^{p} \in  {\mathbb{R}}^{{d}_{\text{out }}}$，以便在给定 ${\mathbf{r}}_{i}^{p} = {\widehat{\mathbf{h}}}_{i}^{p} - {\mathbf{c}}_{j}$ 的情况下对其进行重构。为提高存储效率，将该残差向量的每个维度量化为2位值。因此，存储每个词项向量需要 $2{d}_{\text{out }} + \left\lceil  {{\log }_{2}\left| \mathcal{C}\right| }\right\rceil$ 位，即比ColBERTv1中使用的16位精度压缩所需的 ${16}{d}_{\text{out }}$ 位大约少 $7 \times$，同时不影响检索质量。

Finally, the identifiers of the compressed term embeddings linked to each centroid are grouped together and saved on disk within an inverted list. At search time,the ${n}_{\text{probe }}$ centroids closest to every term representation of a given query are identified, and the embeddings indexed under these centroids are fetched for a first-stage candidate generation. Specifically, the compressed embeddings associated with the selected centroids are accessed via the inverted list structure, decompressed, and scored against each query vector using the similarity metric. The computed similarities are then aggregated by passage for each query term and subjected to a max-pooling operation. Since not all terms from a given passage are evaluated but only those associated with the selected centroids, the scores from this preliminary retrieval stage serve as an approximate of the MaxSim operation described in Section 3.3, thus providing a lower bound on actual scores. These approximated values are summed across query terms,and the $k$ passages with the highest scores undergo a secondary ranking phase. Here, the full set of term embeddings for each candidate passage is considered to calculate the exact MaxSim scores. The selected passages are then reordered based on these refined scores and returned.

最后，将与每个质心关联的压缩词项嵌入的标识符分组，并以倒排表的形式保存在磁盘上。在搜索时，识别出与给定查询的每个词项表示最接近的 ${n}_{\text{probe }}$ 个质心，并提取这些质心下索引的嵌入以进行第一阶段的候选生成。具体而言，通过倒排表结构访问与所选质心关联的压缩嵌入，对其进行解压缩，并使用相似度度量对每个查询向量进行评分。然后，针对每个查询词项按段落汇总计算出的相似度，并进行最大池化操作。由于并非评估给定段落中的所有词项，而仅评估与所选质心关联的词项，因此该初步检索阶段的得分可作为3.3节中描述的MaxSim操作的近似值，从而为实际得分提供一个下限。将这些近似值在查询词项上求和，得分最高的 $k$ 个段落将进入第二阶段的排序。在此阶段，考虑每个候选段落的完整词项嵌入集，以计算精确的MaxSim得分。然后根据这些优化后的得分对所选段落重新排序并返回。

## B Experimental Details

## B 实验细节

Table 4 provides a comprehensive breakdown of ColBERT-XM's performance across individual languages on mMARCO small dev set, depending on the number of examples used for training.

表4详细列出了ColBERT - XM在mMARCO小型开发集上针对每种语言的性能，具体取决于训练所用的示例数量。

<!-- Media -->

<table><tr><td>#Training Examples</td><td>en</td><td>es</td><td>fr</td><td>it</td><td>${pt}$</td><td>id</td><td>de</td><td>${ru}$</td><td>${zh}$</td><td>${ja}$</td><td>${nl}$</td><td>${vi}$</td><td>hi</td><td>ar</td><td>$\mathbf{{Avg}}$</td></tr><tr><td colspan="16">MRR@10</td></tr><tr><td>3.2M</td><td>35.7</td><td>27.7</td><td>25.9</td><td>26.2</td><td>26.9</td><td>25.3</td><td>26.2</td><td>24.4</td><td>24.0</td><td>23.9</td><td>26.5</td><td>21.8</td><td>23.2</td><td>19.2</td><td>25.5</td></tr><tr><td>6.4M</td><td>37.2</td><td>28.5</td><td>26.9</td><td>26.5</td><td>27.6</td><td>26.3</td><td>27.0</td><td>25.1</td><td>24.6</td><td>24.1</td><td>27.5</td><td>22.6</td><td>23.8</td><td>19.5</td><td>26.2</td></tr><tr><td>12.8M</td><td>38.1</td><td>28.6</td><td>26.8</td><td>26.9</td><td>27.5</td><td>26.6</td><td>27.1</td><td>25.4</td><td>24.9</td><td>24.2</td><td>27.3</td><td>22.9</td><td>23.8</td><td>19.5</td><td>26.4</td></tr><tr><td>19.2M</td><td>38.2</td><td>28.7</td><td>26.8</td><td>26.7</td><td>27.9</td><td>26.7</td><td>27.1</td><td>25.7</td><td>25.0</td><td>24.1</td><td>27.5</td><td>23.2</td><td>23.7</td><td>19.3</td><td>26.5</td></tr><tr><td>25.6M</td><td>38.0</td><td>28.4</td><td>26.7</td><td>26.8</td><td>27.8</td><td>26.6</td><td>27.1</td><td>26.0</td><td>25.2</td><td>24.2</td><td>27.5</td><td>23.2</td><td>23.8</td><td>19.6</td><td>26.5</td></tr><tr><td colspan="16">R@10</td></tr><tr><td>3.2M</td><td>63.8</td><td>50.4</td><td>48.2</td><td>47.8</td><td>49.6</td><td>46.8</td><td>48.3</td><td>46.1</td><td>44.9</td><td>44.3</td><td>49.2</td><td>41.2</td><td>43.4</td><td>35.6</td><td>47.1</td></tr><tr><td>6.4M</td><td>65.7</td><td>52.0</td><td>49.2</td><td>48.2</td><td>50.5</td><td>48.3</td><td>49.5</td><td>47.3</td><td>46.0</td><td>44.6</td><td>49.8</td><td>42.4</td><td>44.2</td><td>36.5</td><td>48.2</td></tr><tr><td>12.8M</td><td>66.4</td><td>51.8</td><td>48.7</td><td>48.6</td><td>50.5</td><td>48.3</td><td>49.6</td><td>47.1</td><td>45.9</td><td>45.0</td><td>50.0</td><td>42.3</td><td>43.8</td><td>36.4</td><td>48.2</td></tr><tr><td>19.2M</td><td>67.0</td><td>52.0</td><td>49.1</td><td>48.2</td><td>50.4</td><td>48.9</td><td>49.6</td><td>47.8</td><td>46.0</td><td>44.8</td><td>50.0</td><td>42.8</td><td>43.6</td><td>35.7</td><td>48.3</td></tr><tr><td>25.6M</td><td>67.0</td><td>51.9</td><td>48.7</td><td>48.8</td><td>50.5</td><td>48.6</td><td>49.7</td><td>47.9</td><td>46.4</td><td>45.0</td><td>50.0</td><td>42.7</td><td>43.8</td><td>36.2</td><td>48.4</td></tr><tr><td colspan="16">R@100</td></tr><tr><td>3.2M</td><td>88.5</td><td>77.2</td><td>75.1</td><td>73.6</td><td>75.3</td><td>73.3</td><td>73.1</td><td>73.0</td><td>71.6</td><td>71.2</td><td>74.4</td><td>66.8</td><td>68.6</td><td>59.3</td><td>72.9</td></tr><tr><td>6.4M</td><td>89.3</td><td>77.5</td><td>75.2</td><td>74.1</td><td>75.8</td><td>74.5</td><td>73.9</td><td>73.6</td><td>72.2</td><td>71.4</td><td>75.2</td><td>67.5</td><td>69.8</td><td>60.4</td><td>73.6</td></tr><tr><td>12.8M</td><td>90.1</td><td>77.7</td><td>75.3</td><td>73.8</td><td>75.6</td><td>73.9</td><td>73.9</td><td>73.6</td><td>72.2</td><td>71.4</td><td>75.0</td><td>67.2</td><td>69.0</td><td>59.7</td><td>73.5</td></tr><tr><td>19.2M</td><td>90.0</td><td>77.4</td><td>75.2</td><td>73.6</td><td>75.7</td><td>74.4</td><td>74.1</td><td>73.8</td><td>72.5</td><td>71.3</td><td>75.1</td><td>67.9</td><td>69.1</td><td>59.7</td><td>73.6</td></tr><tr><td>25.6M</td><td>90.0</td><td>77.5</td><td>75.3</td><td>73.6</td><td>75.7</td><td>74.1</td><td>74.2</td><td>73.9</td><td>72.7</td><td>71.4</td><td>75.3</td><td>67.8</td><td>69.4</td><td>59.7</td><td>73.6</td></tr><tr><td colspan="16">R@1000</td></tr><tr><td>3.2M</td><td>96.3</td><td>88.7</td><td>87.5</td><td>86.3</td><td>87.4</td><td>86.2</td><td>85.5</td><td>85.7</td><td>84.7</td><td>83.8</td><td>86.8</td><td>81.5</td><td>81.4</td><td>75.1</td><td>85.5</td></tr><tr><td>6.4M</td><td>96.5</td><td>88.4</td><td>87.3</td><td>86.1</td><td>87.1</td><td>86.7</td><td>86.0</td><td>85.7</td><td>84.8</td><td>83.6</td><td>86.8</td><td>81.6</td><td>82.2</td><td>74.8</td><td>85.5</td></tr><tr><td>12.8M</td><td>96.5</td><td>88.0</td><td>87.5</td><td>85.8</td><td>86.9</td><td>86.0</td><td>85.4</td><td>85.6</td><td>84.7</td><td>83.6</td><td>86.4</td><td>80.9</td><td>81.4</td><td>74.3</td><td>85.2</td></tr><tr><td>19.2M</td><td>96.6</td><td>87.8</td><td>87.2</td><td>85.9</td><td>86.8</td><td>86.5</td><td>85.2</td><td>85.4</td><td>84.4</td><td>83.2</td><td>86.9</td><td>81.1</td><td>81.5</td><td>74.0</td><td>85.2</td></tr><tr><td>25.6M</td><td>96.7</td><td>87.8</td><td>87.3</td><td>85.8</td><td>87.0</td><td>86.2</td><td>85.3</td><td>85.4</td><td>84.3</td><td>83.4</td><td>87.0</td><td>80.9</td><td>81.6</td><td>74.0</td><td>85.2</td></tr></table>

<table><tbody><tr><td>#训练示例</td><td>英语（en）</td><td>西班牙语（es）</td><td>法语（fr）</td><td>意大利语（it）</td><td>${pt}$</td><td>印尼语（id）</td><td>德语（de）</td><td>${ru}$</td><td>${zh}$</td><td>${ja}$</td><td>${nl}$</td><td>${vi}$</td><td>印地语（hi）</td><td>阿拉伯语（ar）</td><td>$\mathbf{{Avg}}$</td></tr><tr><td colspan="16">前10名平均倒数排名（MRR@10）</td></tr><tr><td>3.2M</td><td>35.7</td><td>27.7</td><td>25.9</td><td>26.2</td><td>26.9</td><td>25.3</td><td>26.2</td><td>24.4</td><td>24.0</td><td>23.9</td><td>26.5</td><td>21.8</td><td>23.2</td><td>19.2</td><td>25.5</td></tr><tr><td>6.4M</td><td>37.2</td><td>28.5</td><td>26.9</td><td>26.5</td><td>27.6</td><td>26.3</td><td>27.0</td><td>25.1</td><td>24.6</td><td>24.1</td><td>27.5</td><td>22.6</td><td>23.8</td><td>19.5</td><td>26.2</td></tr><tr><td>12.8M</td><td>38.1</td><td>28.6</td><td>26.8</td><td>26.9</td><td>27.5</td><td>26.6</td><td>27.1</td><td>25.4</td><td>24.9</td><td>24.2</td><td>27.3</td><td>22.9</td><td>23.8</td><td>19.5</td><td>26.4</td></tr><tr><td>19.2M</td><td>38.2</td><td>28.7</td><td>26.8</td><td>26.7</td><td>27.9</td><td>26.7</td><td>27.1</td><td>25.7</td><td>25.0</td><td>24.1</td><td>27.5</td><td>23.2</td><td>23.7</td><td>19.3</td><td>26.5</td></tr><tr><td>25.6M</td><td>38.0</td><td>28.4</td><td>26.7</td><td>26.8</td><td>27.8</td><td>26.6</td><td>27.1</td><td>26.0</td><td>25.2</td><td>24.2</td><td>27.5</td><td>23.2</td><td>23.8</td><td>19.6</td><td>26.5</td></tr><tr><td colspan="16">R@10</td></tr><tr><td>3.2M</td><td>63.8</td><td>50.4</td><td>48.2</td><td>47.8</td><td>49.6</td><td>46.8</td><td>48.3</td><td>46.1</td><td>44.9</td><td>44.3</td><td>49.2</td><td>41.2</td><td>43.4</td><td>35.6</td><td>47.1</td></tr><tr><td>6.4M</td><td>65.7</td><td>52.0</td><td>49.2</td><td>48.2</td><td>50.5</td><td>48.3</td><td>49.5</td><td>47.3</td><td>46.0</td><td>44.6</td><td>49.8</td><td>42.4</td><td>44.2</td><td>36.5</td><td>48.2</td></tr><tr><td>12.8M</td><td>66.4</td><td>51.8</td><td>48.7</td><td>48.6</td><td>50.5</td><td>48.3</td><td>49.6</td><td>47.1</td><td>45.9</td><td>45.0</td><td>50.0</td><td>42.3</td><td>43.8</td><td>36.4</td><td>48.2</td></tr><tr><td>19.2M</td><td>67.0</td><td>52.0</td><td>49.1</td><td>48.2</td><td>50.4</td><td>48.9</td><td>49.6</td><td>47.8</td><td>46.0</td><td>44.8</td><td>50.0</td><td>42.8</td><td>43.6</td><td>35.7</td><td>48.3</td></tr><tr><td>25.6M</td><td>67.0</td><td>51.9</td><td>48.7</td><td>48.8</td><td>50.5</td><td>48.6</td><td>49.7</td><td>47.9</td><td>46.4</td><td>45.0</td><td>50.0</td><td>42.7</td><td>43.8</td><td>36.2</td><td>48.4</td></tr><tr><td colspan="16">R@100</td></tr><tr><td>3.2M</td><td>88.5</td><td>77.2</td><td>75.1</td><td>73.6</td><td>75.3</td><td>73.3</td><td>73.1</td><td>73.0</td><td>71.6</td><td>71.2</td><td>74.4</td><td>66.8</td><td>68.6</td><td>59.3</td><td>72.9</td></tr><tr><td>6.4M</td><td>89.3</td><td>77.5</td><td>75.2</td><td>74.1</td><td>75.8</td><td>74.5</td><td>73.9</td><td>73.6</td><td>72.2</td><td>71.4</td><td>75.2</td><td>67.5</td><td>69.8</td><td>60.4</td><td>73.6</td></tr><tr><td>12.8M</td><td>90.1</td><td>77.7</td><td>75.3</td><td>73.8</td><td>75.6</td><td>73.9</td><td>73.9</td><td>73.6</td><td>72.2</td><td>71.4</td><td>75.0</td><td>67.2</td><td>69.0</td><td>59.7</td><td>73.5</td></tr><tr><td>19.2M</td><td>90.0</td><td>77.4</td><td>75.2</td><td>73.6</td><td>75.7</td><td>74.4</td><td>74.1</td><td>73.8</td><td>72.5</td><td>71.3</td><td>75.1</td><td>67.9</td><td>69.1</td><td>59.7</td><td>73.6</td></tr><tr><td>25.6M</td><td>90.0</td><td>77.5</td><td>75.3</td><td>73.6</td><td>75.7</td><td>74.1</td><td>74.2</td><td>73.9</td><td>72.7</td><td>71.4</td><td>75.3</td><td>67.8</td><td>69.4</td><td>59.7</td><td>73.6</td></tr><tr><td colspan="16">R@1000</td></tr><tr><td>3.2M</td><td>96.3</td><td>88.7</td><td>87.5</td><td>86.3</td><td>87.4</td><td>86.2</td><td>85.5</td><td>85.7</td><td>84.7</td><td>83.8</td><td>86.8</td><td>81.5</td><td>81.4</td><td>75.1</td><td>85.5</td></tr><tr><td>6.4M</td><td>96.5</td><td>88.4</td><td>87.3</td><td>86.1</td><td>87.1</td><td>86.7</td><td>86.0</td><td>85.7</td><td>84.8</td><td>83.6</td><td>86.8</td><td>81.6</td><td>82.2</td><td>74.8</td><td>85.5</td></tr><tr><td>12.8M</td><td>96.5</td><td>88.0</td><td>87.5</td><td>85.8</td><td>86.9</td><td>86.0</td><td>85.4</td><td>85.6</td><td>84.7</td><td>83.6</td><td>86.4</td><td>80.9</td><td>81.4</td><td>74.3</td><td>85.2</td></tr><tr><td>19.2M</td><td>96.6</td><td>87.8</td><td>87.2</td><td>85.9</td><td>86.8</td><td>86.5</td><td>85.2</td><td>85.4</td><td>84.4</td><td>83.2</td><td>86.9</td><td>81.1</td><td>81.5</td><td>74.0</td><td>85.2</td></tr><tr><td>25.6M</td><td>96.7</td><td>87.8</td><td>87.3</td><td>85.8</td><td>87.0</td><td>86.2</td><td>85.3</td><td>85.4</td><td>84.3</td><td>83.4</td><td>87.0</td><td>80.9</td><td>81.6</td><td>74.0</td><td>85.2</td></tr></tbody></table>

Table 4: Influence of training samples on the performance of ColBERT-XM model on mMARCO small dev set.

表4：训练样本对ColBERT - XM模型在mMARCO小型开发集上性能的影响。

<!-- Media -->

## C Reproducibility

## C 可重复性

We ensure the reproducibility of the experimental results by releasing our code on Github at https: //github.com/ant-louis/xm-retrievers. In addition, we release our model checkpoints on Hugging Face at https://huggingface.co/antoinelouis/colbert-xm and https: //huggingface.co/antoinelouis/dpr-xm.

我们通过在Github（https://github.com/ant - louis/xm - retrievers ）上发布代码来确保实验结果的可重复性。此外，我们还在Hugging Face（https://huggingface.co/antoinelouis/colbert - xm 和https://huggingface.co/antoinelouis/dpr - xm ）上发布了模型检查点。