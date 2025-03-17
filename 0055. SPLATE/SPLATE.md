# SPLATE: Sparse Late Interaction Retrieval

# SPLATE：稀疏晚期交互检索

Thibault Formal

蒂博·福尔马尔（Thibault Formal）

Naver Labs Europe

欧洲NAVER实验室（Naver Labs Europe）

Meylan, France

法国梅扬（Meylan）

thibault.formal@naverlabs.com

Stéphane Clinchant

斯特凡·克兰尚（Stéphane Clinchant）

Naver Labs Europe

欧洲NAVER实验室（Naver Labs Europe）

Meylan, France

法国梅扬

stephane.clinchant@naverlabs.com

Hervé Déjean

埃尔韦·德让

Naver Labs Europe

欧洲纳弗实验室

Meylan, France

法国梅扬

herve.dejean@naverlabs.com

Carlos Lassance*

卡洛斯·拉桑斯*

Cohere

科希尔

Toronto, Canada

加拿大，多伦多

cadurosar@gmail.com

## ABSTRACT

## 摘要

The late interaction paradigm introduced with ColBERT stands out in the neural Information Retrieval space, offering a compelling effectiveness-efficiency trade-off across many benchmarks. Efficient late interaction retrieval is based on an optimized multi-step strategy, where an approximate search first identifies a set of candidate documents to re-rank exactly. In this work, we introduce SPLATE, a simple and lightweight adaptation of the ColBERTv2 model which learns an "MLM adapter", mapping its frozen token embeddings to a sparse vocabulary space with a partially learned SPLADE module. This allows us to perform the candidate generation step in late interaction pipelines with traditional sparse retrieval techniques, making it particularly appealing for running ColBERT in CPU environments. Our SPLATE ColBERTv2 pipeline achieves the same effectiveness as the PLAID ColBERTv2 engine by re-ranking 50 documents that can be retrieved under ${10}\mathrm{{ms}}$ .

与ColBERT（上下文双向编码器表示法）一同提出的后期交互范式在神经信息检索领域脱颖而出，在许多基准测试中都展现出了令人信服的有效性与效率的平衡。高效的后期交互检索基于一种优化的多步骤策略，即首先通过近似搜索确定一组候选文档，然后对这些文档进行精确重排序。在这项工作中，我们推出了SPLATE，这是对ColBERTv2模型的一种简单轻量级改进，它学习一个“掩码语言模型适配器（MLM adapter）”，通过部分学习的SPLADE模块将其冻结的词元嵌入映射到一个稀疏词汇空间。这使我们能够在后期交互管道中使用传统的稀疏检索技术执行候选生成步骤，这对于在CPU环境中运行ColBERT尤其有吸引力。我们的SPLATE ColBERTv2管道通过对在${10}\mathrm{{ms}}$条件下可检索到的50篇文档进行重排序，达到了与PLAID ColBERTv2引擎相同的效果。

## CCS CONCEPTS

## 计算机与通信安全会议概念

- Information systems $\rightarrow$ Retrieval models and ranking.

- 信息系统 $\rightarrow$ 检索模型与排序。

## KEYWORDS

## 关键词

Late Interaction, Sparse Retrieval, Hybrid Models

晚期交互、稀疏检索、混合模型

## ACM Reference Format:

## ACM引用格式：

Thibault Formal, Stéphane Clinchant, Hervé Déjean, and Carlos Lassance. 2024. SPLATE: Sparse Late Interaction Retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24), July 14-18, 2024, Washington, DC, USA. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3626772.3657968

蒂博·福尔马尔（Thibault Formal）、斯特凡·克兰尚（Stéphane Clinchant）、埃尔韦·德让（Hervé Déjean）和卡洛斯·拉桑斯（Carlos Lassance）。2024年。SPLATE：稀疏晚期交互检索。见第47届ACM信息检索研究与发展国际会议（SIGIR '24）论文集，2024年7月14 - 18日，美国华盛顿特区。美国纽约州纽约市ACM，6页。https://doi.org/10.1145/3626772.3657968

## 1 INTRODUCTION

## 1 引言

In the landscape of neural retrieval models based on Pre-trained Language Models (PLMs), the late interaction paradigm - introduced with the ColBERT model [16] - delivers state-of-the-art results across many benchmarks. ColBERT - and its variants [11,

在基于预训练语言模型（PLMs）的神经检索模型领域，由ColBERT模型[16]引入的晚期交互范式在许多基准测试中取得了最先进的成果。ColBERT及其变体[11，

${12},{21},{25},{33},{38},{45},{48}\rbrack$ - enjoys many good properties,ranging from interpretability $\left\lbrack  {8,{46}}\right\rbrack$ to robustness $\left\lbrack  {{10},{26},{47},{49}}\right\rbrack$ . The fine-grained interaction mechanism, based on a token-level dense vector representation of documents and queries, alleviates the inherent limitation of single-vector models such as DPR [15]. Due to its MaxSim formulation, late interaction retrieval requires a dedicated multi-step search pipeline. In the meantime, Learned Sparse Retrieval [30] has emerged as a new paradigm to reconcile the traditional search infrastructure with PLMs. In particular, SPLADE models $\left\lbrack  {6,7,9}\right\rbrack$ exhibit strong in-domain and zero-shot capabilities at a fraction of the cost of late interaction approaches - both in terms of memory footprint and search latency $\left\lbrack  {{18},{19},{34},{35}}\right\rbrack$ .

${12},{21},{25},{33},{38},{45},{48}\rbrack$ 具有许多优良特性，范围从可解释性 $\left\lbrack  {8,{46}}\right\rbrack$ 到鲁棒性 $\left\lbrack  {{10},{26},{47},{49}}\right\rbrack$。基于文档和查询的词元级密集向量表示的细粒度交互机制，缓解了诸如DPR [15] 等单向量模型的固有局限性。由于其最大相似度（MaxSim）公式，后期交互检索需要一个专门的多步骤搜索流程。与此同时，学习型稀疏检索 [30] 已成为一种新的范式，用于协调传统搜索基础设施与预训练语言模型（PLMs）。特别是，SPLADE模型 $\left\lbrack  {6,7,9}\right\rbrack$ 在内存占用和搜索延迟 $\left\lbrack  {{18},{19},{34},{35}}\right\rbrack$ 方面，以远低于后期交互方法的成本展现出强大的领域内和零样本能力。

In this work, we draw a parallel between these two lines of works, and show how we can simply "adapt" ColBERTv2 frozen representations with a light SPLADE module to effectively map queries and documents in a sparse vocabulary space. Based on this idea, we introduce SPLATE - for SParse LATE interaction - as an alternative approximate scoring method for late interaction pipelines. Contrary to optimized engines like PLAID [37], our method relies on traditional sparse techniques, making it particularly appealing to run ColBERT in mono-CPU environments.

在这项工作中，我们将这两类工作进行了类比，并展示了如何通过一个轻量级的SPLADE模块简单地“适配”冻结的ColBERTv2表征，从而在稀疏词汇空间中有效地映射查询和文档。基于这一想法，我们引入了SPLATE（稀疏后期交互，SParse LATE interaction），作为后期交互流水线的一种替代近似评分方法。与像PLAID [37]这样的优化引擎不同，我们的方法依赖于传统的稀疏技术，这使得它在单核CPU环境下运行ColBERT特别有吸引力。

## 2 RELATED WORKS

## 2 相关工作

Efficient Late Interaction Retrieval. Late interaction retrieval is a powerful paradigm, that requires complex engineering to scale up efficiently. Specifically, it resorts to a multi-step pipeline, where an initial set of candidate documents is retrieved based on approximate scores [16]. While it is akin to the traditional retrieve-and-rank pipeline in IR, it still fundamentally differs in that the same (PLM) model is used for both steps ${}^{1}$ . Late interaction models offer advantages over cross-encoders because they allow for pre-computation of document representations offline, thus improving efficiency in theory. However, this comes at the cost of storing large indexes of dense term representations. Various optimizations of the ColBERT engine have thus been introduced $\lbrack 5,{12},{20},{23},{27},{29},{33}$ , ${37},{38},{41},{43}\rbrack$ . ColBERTv2 [38] refines the original ColBERT by introducing residual compression to reduce the space footprint of late interaction approaches. Yet, search speed remains a bottleneck, mostly due to the large number of candidates to re-rank exactly $\left( { > {10k}}\right)$ [27]. Santhanam et al. identify the major bottlenecks - in terms of search speed - of the vanilla ColBERTv2 pipeline, and introduce PLAID [37], a new optimized late interaction pipeline that can largely reduce the number of candidate passages without impacting ColBERTv2's effectiveness. In particular, PLAID candidate generation is based on three steps that leverage centroid interaction and centroid pruning - emulating traditional Bag-of-Words (BoW) retrieval - as well as dedicated CUDA kernels. It reduces the large number of candidate documents to re-rank, greatly offloading subsequent steps (index lookup, decompression, and scoring).

高效的后期交互检索。后期交互检索是一种强大的范式，需要复杂的工程技术才能有效地进行扩展。具体而言，它采用多步骤流程，即根据近似得分检索出一组初始候选文档[16]。虽然它与信息检索（IR）中传统的检索 - 排序流程类似，但本质上的不同之处在于，这两个步骤使用的是同一个（预训练语言模型，PLM）模型${}^{1}$。与交叉编码器相比，后期交互模型具有优势，因为它们允许离线预先计算文档表示，从而理论上提高了效率。然而，这需要存储大量的密集词项表示索引。因此，人们对ColBERT引擎进行了各种优化$\lbrack 5,{12},{20},{23},{27},{29},{33}$、${37},{38},{41},{43}\rbrack$。ColBERTv2 [38]通过引入残差压缩来改进原始的ColBERT，以减少后期交互方法的空间占用。然而，搜索速度仍然是一个瓶颈，这主要是因为需要精确重新排序的候选文档数量众多$\left( { > {10k}}\right)$ [27]。桑塔纳姆（Santhanam）等人确定了原始ColBERTv2流程在搜索速度方面的主要瓶颈，并推出了PLAID [37]，这是一种新的优化后期交互流程，它可以在不影响ColBERTv2有效性的情况下大幅减少候选段落的数量。具体来说，PLAID的候选生成基于三个步骤，这些步骤利用质心交互和质心剪枝（模拟传统的词袋（BoW）检索）以及专用的CUDA内核。它减少了需要重新排序的大量候选文档，极大地减轻了后续步骤（索引查找、解压缩和评分）的负担。

---

<!-- Footnote -->

*Work done while at Naver Labs Europe.

*在欧洲Naver实验室工作期间完成。

${}^{1}$ On the contrary,a standard DPR [15] $\gg$ MonoBERT [31] pipeline would require feeding the query twice to a PLM at inference time.

${}^{1}$ 相反，标准的DPR [15] $\gg$ MonoBERT [31] 流水线在推理时需要将查询输入到预训练语言模型（PLM）两次。

<!-- Footnote -->

---

Hybrid Models. Several works have identified similarities between the representations learned by different neural ranking models. For instance, UNIFIER [40] jointly learns dense and sparse single-vector bi-encoders by sharing intermediate transformer layers. Similarly, the BGE-M3 embedding model [3] can perform dense, multi-vector, and sparse retrieval indifferently. SparseEmbed [17] extends SPLADE with dense contextual embeddings - borrowing ideas from ColBERT and COIL [11]. SLIM [22] adapts ColBERT to perform late interaction on top of SPLADE-like representations - making it fully compatible with traditional search techniques. Ram et al. [36] show that mapping representations of a dense bi-encoder to the vocabulary space - via the Masked Language Modeling (MLM) head - can also be used for interpretation purposes.

混合模型。多项研究已经发现了不同神经排序模型所学习的表示之间的相似性。例如，UNIFIER [40] 通过共享中间的Transformer层联合学习密集和稀疏的单向量双编码器。同样，BGE - M3嵌入模型 [3] 可以无差别地执行密集、多向量和稀疏检索。SparseEmbed [17] 用密集上下文嵌入扩展了SPLADE，借鉴了ColBERT和COIL [11] 的思想。SLIM [22] 对ColBERT进行调整，以在类似SPLADE的表示之上进行后期交互，使其与传统搜索技术完全兼容。Ram等人 [36] 表明，通过掩码语言建模（MLM）头将密集双编码器的表示映射到词汇空间也可用于解释目的。

## 3 METHOD

## 3 方法

SPLATE is motivated by two core ideas: (1) PLAID [37] draws inspiration from traditional BoW retrieval to optimize the late interaction pipeline; (2) dense embeddings can seemingly be mapped to the vocabulary space [36]. Rather than proposing a new standalone model, we show how SPLATE can be used to approximate the candidate generation step in late interaction retrieval, by bridging the gap between sparse and dense models.

SPLATE受两个核心思想启发：（1）PLAID [37]从传统的词袋（BoW）检索中汲取灵感，以优化后期交互流程；（2）密集嵌入似乎可以映射到词汇空间 [36]。我们并非提出一个新的独立模型，而是展示了如何通过弥合稀疏模型和密集模型之间的差距，使用SPLATE来近似后期交互检索中的候选生成步骤。

Adapting Representations. SPLATE builds on the similarities between the representations learned by sparse and dense IR models. For instance, Ram et al. [36] show that mapping representations of a dense bi-encoder with the MLM head can produce meaningful BoW. We take one step further and hypothesize that effective sparse models can be derived - or at least adapted - from frozen embeddings of dense IR models in a SPLADE-like fashion. We, therefore, propose to "branch" an MLM head on top of a frozen ColBERT model.

适配表示。SPLATE基于稀疏信息检索（IR）模型和密集信息检索模型所学习到的表示之间的相似性构建。例如，Ram等人 [36] 表明，使用掩码语言模型（MLM）头对密集双编码器的表示进行映射可以产生有意义的词袋。我们更进一步，假设可以以类似SPLADE的方式从密集信息检索模型的冻结嵌入中推导出——或者至少适配出——有效的稀疏模型。因此，我们提议在冻结的ColBERT模型之上“分支”一个掩码语言模型头。

SPLATE. Given ColBERT’s contextual embeddings ${\left( {h}_{i}\right) }_{i \in  t}$ of an input query or document $t$ ,we can define a simple "adapted" MLM head, by linearly mapping transformed representations back to the vocabulary. Inspired by Adapter modules [14, 32], SPLATE thus simply adapts frozen representations ${\left( {h}_{i}\right) }_{i \in  t}$ by learning a simple two-layer MLP, whose output is recombined in a residual fashion before "MLM" vocabulary projection:

SPLATE。给定输入查询或文档$t$的ColBERT上下文嵌入${\left( {h}_{i}\right) }_{i \in  t}$，我们可以通过将转换后的表示线性映射回词汇表来定义一个简单的“适配”掩码语言模型（MLM）头部。受适配器模块[14, 32]的启发，SPLATE通过学习一个简单的两层多层感知器（MLP）来简单地适配冻结表示${\left( {h}_{i}\right) }_{i \in  t}$，其输出在进行“MLM”词汇投影之前以残差方式重新组合：

$$
{w}_{iv} = {\left( {h}_{i} + ML{P}_{\theta }\left( {h}_{i}\right) \right) }^{T}{E}_{v} + {b}_{v} \tag{1}
$$

where ${w}_{i}$ corresponds to an unnormalized log-probability distribution over the vocabulary $\mathcal{V}$ for the token ${t}_{i},{E}_{v}$ is the (Col)BERT input embedding for the token $v$ and ${b}_{v}$ is a token-level bias. The residual guarantees a near-identity initialization - making training stable [14]. We can then derive sparse SPLADE vectors as follows:

其中${w}_{i}$对应于词汇表$\mathcal{V}$上未归一化的对数概率分布，对于标记${t}_{i},{E}_{v}$，$v$是（Col）BERT输入嵌入，${b}_{v}$是标记级偏置。残差保证了接近恒等的初始化，从而使训练稳定[14]。然后我们可以按如下方式推导出稀疏SPLADE向量：

$$
{w}_{v} = \mathop{\max }\limits_{{i \in  t}}\log \left( {1 + \operatorname{ReLU}\left( {w}_{iv}\right) }\right) ,\;v \in  \{ 1,\ldots ,\left| \mathcal{V}\right| \}  \tag{2}
$$

<!-- Media -->

<!-- figureText: Adapter (sparse) Colbertv2 PLM -->

<img src="https://cdn.noedgeai.com/01958ba7-1221-783f-849a-947faf162941_1.jpg?x=978&y=244&w=616&h=570&r=0"/>

Figure 1: (Left) SPLATE relies on the same representations ${\left( {h}_{i}\right) }_{i \in  t}$ to learn sparse BoW with SPLADE (candidate generation) and to compute late interactions (re-ranking). (Right) Inference: SPLATE ColBERTv2 maps the representations of the query tokens to a sparse vector, which is used to retrieve $k$ documents from a pre-computed sparse index (R setting). In the ${e2e}$ setting,representations are gathered from the ColBERT index to re-rank the candidates exactly with MaxSim.

图1：（左）SPLATE依赖相同的表示形式${\left( {h}_{i}\right) }_{i \in  t}$，使用SPLADE学习稀疏词袋模型（候选生成）并计算后期交互（重排序）。（右）推理：SPLATE ColBERTv2将查询词元的表示形式映射到一个稀疏向量，该向量用于从预先计算的稀疏索引中检索$k$文档（R设置）。在${e2e}$设置中，从ColBERT索引中收集表示形式，以使用MaxSim精确地对候选文档进行重排序。

<!-- Media -->

We then train the parameters of the MLM head $\left( {\mathbf{\theta },\mathbf{b}}\right)$ with distillation based on the derived SPLADE vectors to reproduce ColBERT's scores - see Section 4. Our approach is very light, as the ColBERT backbone model is entirely frozen - including the (tied) projection layer $E$ . In our default setting,the MLP first down-projects representations by a factor of two, then up-projects back to the original dimension. This corresponds to a latent dimension of ${768}/2 = {384}$ - early experiments indicate that the choice of this hyperparameter is not critical - and amounts to roughly ${0.6M}$ trainable parameters only (yellow blocks in Figure 1, (Left)).

然后，我们基于导出的SPLADE向量，通过蒸馏来训练掩码语言模型（MLM）头部 $\left( {\mathbf{\theta },\mathbf{b}}\right)$ 的参数，以重现ColBERT的得分 - 详见第4节。我们的方法非常轻量级，因为ColBERT骨干模型完全被冻结，包括（绑定的）投影层 $E$。在我们的默认设置中，多层感知器（MLP）首先将表征下投影至原来的二分之一，然后再上投影回原始维度。这对应着 ${768}/2 = {384}$ 的潜在维度 - 早期实验表明，该超参数的选择并不关键 - 并且仅相当于大约 ${0.6M}$ 个可训练参数（图1（左）中的黄色块）。

Efficient Candidate Generation for Late Interaction. By adapting ColBERT's frozen dense representations with a SPLADE module, SPLATE aims to approximate late interaction scoring with an efficient sparse dot product. Thus,the same representations ${\left( {h}_{i}\right) }_{i \in  t}$ can function in both retrieval (SPLATE module) and re-ranking (ColBERT's MaxSim) scenarios - requiring a single transformer inference step on query and document sides. Thus, it becomes possible to replace the existing candidate generation step in late retrieval pipelines such as PLAID with traditional sparse retrieval to efficiently provide ColBERT with documents to re-rank. SPLATE is therefore not a model per se, but rather offers an alternative implementation to late-stage pipelines by bridging the gap between sparse and dense models. SPLATE however differs from PLAID in various aspects:

延迟交互的高效候选生成。通过使用SPLADE模块调整ColBERT的冻结密集表示，SPLATE旨在通过高效的稀疏点积来近似延迟交互评分。因此，相同的表示 ${\left( {h}_{i}\right) }_{i \in  t}$ 可以在检索（SPLATE模块）和重排序（ColBERT的最大相似度）场景中发挥作用——只需在查询和文档端进行一次Transformer推理步骤。因此，有可能用传统的稀疏检索替换延迟检索管道（如PLAID）中现有的候选生成步骤，从而有效地为ColBERT提供待重排序的文档。因此，SPLATE本身并不是一个模型，而是通过弥合稀疏模型和密集模型之间的差距，为后期管道提供了一种替代实现方式。然而，SPLATE在多个方面与PLAID不同：

- While PLAID implicitly derives sparse BoW representations from ColBERTv2's centroid mapping, SPLATE explicitly learns such representations by adapting a pseudo-MLM head to ColBERT frozen representations. The approximate step becomes supervised rather than (yet efficiently) "engineered".

- 虽然PLAID从ColBERTv2的质心映射中隐式导出稀疏词袋（Bag-of-Words，BoW）表示，但SPLATE通过将伪掩码语言模型（Masked Language Model，MLM）头应用于ColBERT的冻结表示来显式学习此类表示。近似步骤变成了有监督的，而不是（尽管高效）“设计”出来的。

- The candidate generation can benefit from the long-standing efficiency of inverted indexes and query processing techniques such as MaxScore [44] or WAND [2], making end-to-end ColBERT more "CPU-friendly" - see Table 1.

- 候选生成可以受益于倒排索引和查询处理技术（如MaxScore [44] 或WAND [2]）长期以来的高效性，使端到端的ColBERT更 “对CPU友好” —— 见表1。

- It is more controllable and directly amenable to all sorts of recent optimizations for learned sparse models [18, 19].

- 它更易于控制，并且可以直接应用于对学习型稀疏模型的各种最新优化 [18, 19]。

- ColBERT's pipeline becomes even more interpretable, as SPLATE's candidate generation simply operates in the vocabulary space - rather than representing documents as a lightweight bag of centroids - see Table 3 for examples.

- ColBERT的流程变得更具可解释性，因为SPLATE的候选生成仅在词汇空间中进行操作 —— 而不是将文档表示为一个轻量级的质心集合 —— 示例见表3。

Nonetheless, SPLATE requires an additional - although light - training round for the parameters of the Adapter module. It also requires indexing SPLATE's sparse document vectors, therefore adding a small memory footprint overhead ${}^{2}$ . Also,note that hybrid approaches like BGE-M3 [3] - that can output sparse and multi-vector representations - could in theory be used in late interaction pipelines. However, SPLATE is directly optimized to approximate ColBERTv2, and we leave for future work the study of jointly training the candidate generation and re-ranking modules.

尽管如此，SPLATE需要为适配器模块的参数进行额外的（尽管是轻量级的）训练轮次。它还需要对SPLATE的稀疏文档向量进行索引，因此会增加少量的内存开销 ${}^{2}$。此外，请注意，像BGE - M3 [3] 这样可以输出稀疏和多向量表示的混合方法理论上可以用于后期交互流程。然而，SPLATE是直接针对近似ColBERTv2进行优化的，我们将联合训练候选生成和重排序模块的研究留待未来工作。

## 4 EXPERIMENTS

## 4 实验

Setting. We initialize SPLATE with ColBERTv2 [38] weights which are kept frozen. We rely on top- ${k}_{q,d}$ pooling to obtain respectively query and document BoW SPLADE representations ${}^{3}$ . We train the MLM parameters $\left( {\mathbf{\theta },\mathbf{b}}\right)$ on the MS MARCO passage dataset [1], using both distillation and hard negative sampling. More specifically, we distill ColBERTv2's scores based on a weighted combination of marginMSE [13] and KLDiv [24] losses for 3 epochs. We set the batch size to 24 , and select 20 hard negatives per query - coming from ColBERTv2's top-1000. By using ColBERTv2 as both the teacher and the source of hard negatives, SPLATE aims to approximate late interaction with sparse retrieval. SPLATE models are trained with the SPLADE codebase on 2 Tesla V100 GPUs with 32GB memory in less than two hours ${}^{4}$ . SPLATE can be evaluated as a standalone sparse retriever(R),but more interestingly in an end-to-end late interaction pipeline (e2e) where it provides ColBERTv2 with candidates to re-rank (see Figure 1, ${\left( \text{Right}\right) }^{5}$ . For the former, we rely on the PISA engine [28] to conduct sparse retrieval with block-max WAND and provide latency measurements as the Mean Response Time (MRT), i.e., the average search latency measured on the MS MARCO dataset using one core of an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz CPU. For the latter, we perform on-the-fly re-ranking with the ColBERT library ${}^{6}$ . Note that naive re-ranking with ColBERT is sub-optimal - compared to pipelines that precompute document term embeddings. We leave the end-to-end latency measurements for future work - but we believe the integration of SPLATE into ColBERT's pipelines such as PLAID should be seamless, as it would only require modifying the candidate generation step. We evaluate models on the MS MARCO dev set and the TREC DL19 queries [4] (in-domain), and provide out-of-domain evaluations on the 13 readily available BEIR datasets [42], as well as the test pooled Search dataset of the LoTTE benchmark [38].

设置。我们使用ColBERTv2 [38]的权重初始化SPLATE，这些权重保持冻结状态。我们依靠top- ${k}_{q,d}$池化分别获得查询和文档的词袋（Bag-of-Words，BoW）SPLADE表示 ${}^{3}$。我们在MS MARCO段落数据集 [1]上训练掩码语言模型（Masked Language Model，MLM）参数 $\left( {\mathbf{\theta },\mathbf{b}}\right)$，同时使用蒸馏和难负样本采样。更具体地说，我们基于边际均方误差（marginMSE） [13]和KL散度（KLDiv） [24]损失的加权组合对ColBERTv2的得分进行3个轮次的蒸馏。我们将批量大小设置为24，并为每个查询选择20个难负样本——这些样本来自ColBERTv2的前1000个结果。通过将ColBERTv2既作为教师模型又作为难负样本的来源，SPLATE旨在通过稀疏检索近似后期交互。SPLATE模型使用SPLADE代码库在2块显存为32GB的特斯拉V100图形处理器（GPU）上进行训练，训练时间不到两小时 ${}^{4}$。SPLATE可以作为独立的稀疏检索器（R）进行评估，但更有趣的是，它可以用于端到端的后期交互管道（e2e），在该管道中，它为ColBERTv2提供待重排序的候选结果（见图1， ${\left( \text{Right}\right) }^{5}$）。对于前者，我们依靠比萨引擎（PISA engine） [28]使用块最大波前算法（block-max WAND）进行稀疏检索，并将平均响应时间（Mean Response Time，MRT）作为延迟度量，即使用英特尔（Intel）至强（Xeon）金牌6338中央处理器（CPU）（主频2.00GHz）的一个核心在MS MARCO数据集上测量的平均搜索延迟。对于后者，我们使用ColBERT库 ${}^{6}$进行实时重排序。请注意，与预先计算文档词嵌入的管道相比，使用ColBERT进行简单的重排序并不是最优的。我们将端到端的延迟测量留待未来工作——但我们相信，将SPLATE集成到ColBERT的管道（如PLAID）中应该是无缝的，因为这只需要修改候选生成步骤。我们在MS MARCO开发集和TREC DL19查询 [4]（领域内）上评估模型，并在13个现成的BEIR数据集 [42]以及LoTTE基准测试 [38]的测试合并搜索数据集上进行领域外评估。

The following experiments investigate three different Research Questions: (1) How does the sparsity of SPLATE vectors affect latency and re-ranking performance? (2) How accurate SPLATE candidate generation is compared to ColBERTv2? (3) How does it perform overall for in-domain and out-of-domain scenarios?

以下实验研究了三个不同的研究问题：（1）SPLATE向量的稀疏性如何影响延迟和重排序性能？（2）与ColBERTv2相比，SPLATE候选生成的准确性如何？（3）它在领域内和领域外场景中的整体表现如何？

Latency Results. Table 1 reports in-domain results on MS MARCO, in both retrieval-only (R) and end-to-end (e2e) settings. Overall, the results show that it is possible to "convert" a frozen Col-BERTv2 model to an effective SPLADE, with a lightweight residual adaptation of its token embeddings. We consider several SPLATE models trained with varying pooling sizes $\left( {{k}_{q},{k}_{d}}\right)$ - those parameters controlling the size of the query and document representations. We observe the standard effectiveness-efficiency trade-off for SPLADE, where pooling affects both the performance and average latency. These results indicate that one can easily control the latency of the candidate generation step by selecting appropriate pooling sizes. However, after re-ranking with ColBERTv2, all the models perform comparably, which is interesting from an efficiency perspective, as it becomes possible to use very lightweight models to cheaply provide candidates (e.g.,as low as ${2.9}\mathrm{\;{ms}}$ Mean Response Time), while achieving performance on par with the original ColBERTv2 (see Table 2). For comparison, the end-to-end latency reported in PLAID [37] (single CPU core, less conservative setting with $k = {10}$ ) is around ${186}\mathrm{\;{ms}}$ on MS MARCO. Given that candidate generation accounts for around two-thirds of the complete pipeline [37], SPLATE thus offers an interesting alternative for running ColBERT on mono-CPU environments.

延迟结果。表1报告了在MS MARCO数据集上的领域内结果，涵盖仅检索（R）和端到端（e2e）两种设置。总体而言，结果表明，可以通过对其词元嵌入进行轻量级的残差调整，将冻结的Col - BERTv2模型“转换”为有效的SPLADE模型。我们考虑了几个使用不同池化大小$\left( {{k}_{q},{k}_{d}}\right)$训练的SPLATE模型——这些参数控制着查询和文档表示的大小。我们观察到SPLADE存在标准的有效性 - 效率权衡，其中池化会影响性能和平均延迟。这些结果表明，通过选择合适的池化大小，可以轻松控制候选生成步骤的延迟。然而，在使用ColBERTv2进行重排序后，所有模型的表现相当，从效率的角度来看这很有趣，因为可以使用非常轻量级的模型以低成本提供候选结果（例如，平均响应时间低至${2.9}\mathrm{\;{ms}}$），同时实现与原始ColBERTv2相当的性能（见表2）。作为对比，PLAID [37]中报告的端到端延迟（单CPU核心，使用$k = {10}$的较宽松设置）在MS MARCO上约为${186}\mathrm{\;{ms}}$。鉴于候选生成约占整个流程的三分之二[37]，因此SPLATE为在单核CPU环境中运行ColBERT提供了一个有趣的替代方案。

<!-- Media -->

Table 1: Retrieval latency (MRT), retrieval-only (R) and end-to-end (e2e, $k = {50}$ ) MRR@10 on MS MARCO dev.

表1：在MS MARCO开发集上的检索延迟（平均响应时间，MRT）、仅检索（R）和端到端（e2e，$k = {50}$ ）的前10名平均倒数排名（MRR@10）。

<table><tr><td rowspan="2">$\left( {{k}_{q},{k}_{d}}\right)$</td><td colspan="2">R</td><td>e2e $\left( {k = {50}}\right)$</td></tr><tr><td>MRT (ms)</td><td>MRR@10</td><td>MRR@10</td></tr><tr><td>(5,30)</td><td>2.9</td><td>34.5</td><td>39.5</td></tr><tr><td>(5,50)</td><td>4.3</td><td>35.5</td><td>39.7</td></tr><tr><td>(5, 100)</td><td>7.4</td><td>35.6</td><td>39.8</td></tr><tr><td>(10, 100)</td><td>24.0</td><td>36.7</td><td>40.0</td></tr><tr><td>(20,200)</td><td>106.0</td><td>37.4</td><td>40.0</td></tr></table>

<table><tbody><tr><td rowspan="2">$\left( {{k}_{q},{k}_{d}}\right)$</td><td colspan="2">R</td><td>端到端 $\left( {k = {50}}\right)$</td></tr><tr><td>平均响应时间（毫秒）</td><td>前10召回率（MRR@10）</td><td>前10召回率（MRR@10）</td></tr><tr><td>(5,30)</td><td>2.9</td><td>34.5</td><td>39.5</td></tr><tr><td>(5,50)</td><td>4.3</td><td>35.5</td><td>39.7</td></tr><tr><td>(5, 100)</td><td>7.4</td><td>35.6</td><td>39.8</td></tr><tr><td>(10, 100)</td><td>24.0</td><td>36.7</td><td>40.0</td></tr><tr><td>(20,200)</td><td>106.0</td><td>37.4</td><td>40.0</td></tr></tbody></table>

<!-- Media -->

Approximation Quality. To assess the quality of SPLATE approximation,we compare the top- $k$ passages retrieved by PLAID ColBERTv2 to the ones retrieved by SPLATE (R). We report in Figure 2 the average fraction $R\left( k\right)$ of documents in SPLATE’s top- ${k}^{\prime }$ that also appear in the top- $k$ documents retrieved by ColBERTv2 on MS MARCO,for $k \in  \{ {10},{100}\}$ and ${k}^{\prime } = i \times  k,i \in  \{ 1,\ldots ,5\}$ . When $k = {10}$ ,SPLATE can retrieve more than 90% of ColBERTv2’s documents in its top-50 $\left( {i = 5}\right)$ ,for all levels of $\left( {{k}_{q},{k}_{d}}\right)$ . This explains the ability of SPLATE to fully recover ColBERT's performance by re-ranking a handful of documents (e.g., 50 only). We additionally observe that the quality of approximation falls short for efficient models (i.e.,lower $\left. \left( {{k}_{q},{k}_{d}}\right) \right)$ when $k$ is higher.

近似质量。为了评估SPLATE近似的质量，我们将PLAID ColBERTv2检索到的前$k$个段落与SPLATE（R）检索到的段落进行比较。我们在图2中报告了在MS MARCO数据集上，SPLATE的前${k}^{\prime }$个文档中同时也出现在ColBERTv2检索到的前$k$个文档中的平均比例$R\left( k\right)$，其中$k \in  \{ {10},{100}\}$和${k}^{\prime } = i \times  k,i \in  \{ 1,\ldots ,5\}$。当$k = {10}$时，对于所有$\left( {{k}_{q},{k}_{d}}\right)$级别，SPLATE在前50个$\left( {i = 5}\right)$文档中能够检索到ColBERTv2超过90%的文档。这解释了SPLATE通过对少量文档（例如，仅50个）进行重新排序就能完全恢复ColBERT性能的能力。此外，我们还观察到，对于高效模型（即，当$k$较高时$\left. \left( {{k}_{q},{k}_{d}}\right) \right)$较低），近似质量有所不足。

---

<!-- Footnote -->

${}^{2}$ Note however that this is negligible compared to ColBERT’s index - for instance,the MS MARCO PISA index for the SPLATE model in Table 2 weighs around 2.2GB.

${}^{2}$ 不过请注意，与ColBERT的索引相比，这可以忽略不计——例如，表2中SPLATE模型的MS MARCO PISA索引大小约为2.2GB。

${}^{3}$ While SPLADE is usually trained with sparse regularization,top- ${k}_{q,d}$ was shown to be almost as effective - while being much simpler [30].

${}^{3}$ 虽然SPLADE通常使用稀疏正则化进行训练，但研究表明top - ${k}_{q,d}$几乎同样有效，而且要简单得多[30]。

${}^{4}$ https://github.com/naver/splade

${}^{4}$ https://github.com/naver/splade

${}^{5}$ Note that SPLATE (e2e) is an alternative implementation of ColBERTv2. We use SPLATE (resp. PLAID) or SPLATE ColBERTv2 (resp. PLAID ColBERTv2) indifferently. ${}^{6}$ https://github.com/stanford-futuredata/ColBERT

${}^{5}$ 请注意，SPLATE (e2e)是ColBERTv2的另一种实现方式。我们可以随意使用SPLATE（或PLAID）或SPLATE ColBERTv2（或PLAID ColBERTv2）。 ${}^{6}$ https://github.com/stanford-futuredata/ColBERT

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: - ${k}_{q} = {20},{k}_{d} = {200}$ -->

<img src="https://cdn.noedgeai.com/01958ba7-1221-783f-849a-947faf162941_3.jpg?x=286&y=237&w=444&h=321&r=0"/>

Figure 2: Candidate generation approximate accuracy on MS MARCO dev - SPLATE (R). Dotted lines (∎) represent $R\left( {10}\right)$ , solid lines represent (*) $R\left( {100}\right)$ .

图2：MS MARCO开发集上的候选生成近似准确率 - SPLATE (R)。虚线（∎）表示$R\left( {10}\right)$，实线表示(*) $R\left( {100}\right)$。

<!-- Media -->

Figure 3 further reports the performance of SPLATE (e2e) on out-of-domain. We observe similar trends, where increasing both the number $k$ of documents to re-rank and $\left( {{k}_{q},{k}_{d}}\right)$ leads to better generalization. Overall, re-ranking only 50 documents provides a good trade-off across all settings - echoing previous findings [27, 37]. Yet,the most efficient scenario $\left( {\left( {{k}_{q},{k}_{d}}\right)  = \left( {5,{50}}\right) ,k = {10}}\right)$ still leads to impressive results: 38.4 MRR@10 on MS MARCO dev (not shown), 70.0 S@5 on LoTTE (purple line on Figure 3).

图3进一步展示了SPLATE（端到端）在域外数据上的性能。我们观察到了类似的趋势，即增加待重排序文档的数量$k$和$\left( {{k}_{q},{k}_{d}}\right)$都会带来更好的泛化能力。总体而言，仅对50篇文档进行重排序在所有设置下都能实现良好的权衡——这与之前的研究结果[27, 37]相呼应。然而，最有效的场景$\left( {\left( {{k}_{q},{k}_{d}}\right)  = \left( {5,{50}}\right) ,k = {10}}\right)$仍能取得令人瞩目的结果：在MS MARCO开发集上的MRR@10达到38.4（未展示），在LoTTE数据集上的S@5达到70.0（图3中的紫色线）。

Overall Results. Finally, Table 2 compares SPLATE ColBERTv2 with the reference points ColBERTv2 [38] and PLAID ColBERTv2 $\left( {k = {1000}}\right) \left\lbrack  {37}\right\rbrack   -$ in both $\mathrm{R}$ and $\mathrm{e}2\mathrm{e}$ settings. We also include results from SPLADE++ [7], as well as the hybrid methods SparseEm-bed [17] and SLIM++ [22] - even though they are not entirely comparable to SPLATE. While SparseEmbed and SLIM introduce new models, SPLATE rather proposes an alternative implementation to ColBERT's late retrieval pipeline. We further report the two baselines consisting of retrieving documents with BM25 (resp. SPLADE++) and re-ranking those with ColBERTv2 (BM25 $\gg  \mathrm{C}$ and $\mathrm{S} \gg  \mathrm{C}$ respectively,with $k = {50}$ ). Note that we expect SPLATE to perform in between,as BM25 $\gg  \mathrm{C}$ relies on a less effective retriever, while $\mathrm{S} \gg  \mathrm{C}$ fundamentally differs from SPLATE,as it is based on two different models. Specifically, it requires feeding the query to a PLM twice at inference time. Overall, SPLATE (R) is effective as a standalone retriever (e.g., reaching almost 37 MRR@10 on MS MARCO dev). On the other hand, SPLATE (e2e) performs comparably to ColBERTv2 and PLAID on MS MARCO, BEIR, and LoTTE. Additionally, we conducted a meta-analysis against PLAID with RANGER [39] over the 13 BEIR datasets, and found no statistical differences on 10 datasets, and statistical improvement (resp. loss) on one (resp. two) dataset(s). Finally, we provide in Table 3 some examples of predicted BoW for queries in MS MARCO dev - highlighting the interpretable nature of the retrieval step in SPLATE-based ColBERT's pipeline.

总体结果。最后，表2比较了SPLATE ColBERTv2与参考点ColBERTv2 [38]和PLAID ColBERTv2 $\left( {k = {1000}}\right) \left\lbrack  {37}\right\rbrack   -$在$\mathrm{R}$和$\mathrm{e}2\mathrm{e}$两种设置下的表现。我们还纳入了SPLADE++ [7]的结果，以及混合方法SparseEmbed [17]和SLIM++ [22]的结果——尽管它们与SPLATE并非完全可比。SparseEmbed和SLIM引入了新模型，而SPLATE则为ColBERT的后期检索流程提出了一种替代实现方式。我们进一步报告了两个基线结果，即使用BM25（分别对应SPLADE++）检索文档，然后用ColBERTv2对这些文档进行重排序（分别为BM25 $\gg  \mathrm{C}$和$\mathrm{S} \gg  \mathrm{C}$，使用$k = {50}$）。请注意，我们预计SPLATE的表现会介于两者之间，因为BM25 $\gg  \mathrm{C}$依赖于效果较差的检索器，而$\mathrm{S} \gg  \mathrm{C}$与SPLATE有根本区别，因为它基于两种不同的模型。具体而言，它在推理时需要将查询两次输入到预训练语言模型（PLM）中。总体而言，SPLATE（R）作为独立的检索器很有效（例如，在MS MARCO开发集上的MRR@10几乎达到37）。另一方面，SPLATE（e2e）在MS MARCO、BEIR和LoTTE上的表现与ColBERTv2和PLAID相当。此外，我们使用RANGER [39]针对PLAID对13个BEIR数据集进行了元分析，发现10个数据集上没有统计学差异，在一个（分别对应两个）数据集上有统计学上的改进（分别对应损失）。最后，我们在表3中提供了MS MARCO开发集中查询的预测词袋（BoW）示例——突出了基于SPLATE的ColBERT流程中检索步骤的可解释性。

## 5 CONCLUSION

## 5 结论

We propose SPLATE, a new lightweight candidate generation technique simplifying ColBERTv2's candidate generation for late interaction retrieval. SPLATE adapts ColBERTv2's frozen embeddings

我们提出了SPLATE（一种新的轻量级候选生成技术），用于简化ColBERTv2在后期交互检索中的候选生成过程。SPLATE采用了ColBERTv2的冻结嵌入（frozen embeddings）

<!-- Media -->

<!-- figureText: top- $k$ (to re-rank) -->

<img src="https://cdn.noedgeai.com/01958ba7-1221-783f-849a-947faf162941_3.jpg?x=1064&y=238&w=443&h=318&r=0"/>

Figure 3: Impact of $k$ and $\left( {{k}_{q},{k}_{d}}\right)$ on SPLATE (e2e) ouf-of-domain performance - Success@5 on LoTTE (test pooled Search). The orange line represents ColBERTv2.

图3：$k$和$\left( {{k}_{q},{k}_{d}}\right)$对SPLATE（端到端）域外性能的影响——LoTTE（测试池搜索）的前5准确率（Success@5）。橙色线代表ColBERTv2。

<!-- Media -->

Table 2: Evaluation of SPLATE with $\left( {{k}_{q},{k}_{d}}\right)  = \left( {{10},{100}}\right)$ and $k = {50}$ . abcde denote significant improvements over the corresponding rows,for a paired $t$ -test with $p$ -value $= {0.01}$ and Bonferroni correction (MS MARCO dev set and DL19). PLAID ColBERTv2 [37] $\left( {k = {1000}}\right)$ reports the dev LoTTE ${}^{ * }$ S@5.

表2：对使用$\left( {{k}_{q},{k}_{d}}\right)  = \left( {{10},{100}}\right)$和$k = {50}$的SPLATE的评估。abcde表示与相应行相比有显著改进，采用配对$t$检验，$p$值为$= {0.01}$，并进行了邦费罗尼校正（MS MARCO开发集和DL19）。PLAID ColBERTv2 [37] $\left( {k = {1000}}\right)$报告了开发集LoTTE ${}^{ * }$的前5成功率（S@5）。

<!-- Media -->

<table><tr><td/><td>MS MARCO MRR@10</td><td>DL19 nDCG@10</td><td>R@1k</td><td>BEIR nDCG@10</td><td>LoTTE S@5</td></tr><tr><td colspan="6">- Sparse/Hybrid</td></tr><tr><td>SPLADE++ [7]</td><td>38.0</td><td>73.2</td><td>87.5</td><td>50.7</td><td>-</td></tr><tr><td>SparseEmbed [17]</td><td>39.2</td><td>-</td><td>-</td><td>50.9</td><td>-</td></tr><tr><td>SLIM++ [22]</td><td>40.4</td><td>71.4</td><td>84.2</td><td>49.0</td><td>-</td></tr><tr><td colspan="6">- References</td></tr><tr><td>ColBERTv2 [38]</td><td>39.7</td><td>-</td><td>-</td><td>49.7</td><td>71.6</td></tr><tr><td>(a) PLAID ColBERTv2 [37]</td><td>${39.8}^{bd}$</td><td>74.6</td><td>${85.2}^{b}$</td><td>-</td><td>${69.6}^{ * }$</td></tr><tr><td>(b) $\mathrm{{BM}}{25} \gg  \mathrm{C}\left( {k = {50}}\right)$</td><td>34.3</td><td>68.7</td><td>73.9</td><td>49.0</td><td>62.8</td></tr><tr><td>(c) $\mathrm{S} \gg  \mathrm{C}\left( {k = {50}}\right)$</td><td>${40.4}^{bd}$</td><td>74.4</td><td>${87.5}^{b}$</td><td>49.9</td><td>72.0</td></tr><tr><td colspan="6">- SPLATE ColBERTv2 $\left( {k = {50}}\right)$</td></tr><tr><td>(d) SPLATE (R)</td><td>${36.7}^{b}$</td><td>72.9</td><td>${84.4}^{b}$</td><td>46.5</td><td>66.7</td></tr><tr><td>(e) SPLATE (e2e)</td><td>${40.0}^{bd}$</td><td>74.2</td><td>${84.4}^{b}$</td><td>49.6</td><td>71.0</td></tr></table>

<table><tbody><tr><td></td><td>MS MARCO 前10名平均倒数排名（MRR@10）</td><td>DL19 前10名归一化折损累积增益（nDCG@10）</td><td>前1000名召回率（R@1k）</td><td>BEIR 前10名归一化折损累积增益（nDCG@10）</td><td>LoTTE 前5名成功率（S@5）</td></tr><tr><td colspan="6">- 稀疏/混合</td></tr><tr><td>SPLADE++（SPLADE++） [7]</td><td>38.0</td><td>73.2</td><td>87.5</td><td>50.7</td><td>-</td></tr><tr><td>SparseEmbed（稀疏嵌入） [17]</td><td>39.2</td><td>-</td><td>-</td><td>50.9</td><td>-</td></tr><tr><td>SLIM++（SLIM++） [22]</td><td>40.4</td><td>71.4</td><td>84.2</td><td>49.0</td><td>-</td></tr><tr><td colspan="6">- 参考文献</td></tr><tr><td>ColBERTv2（ColBERTv2） [38]</td><td>39.7</td><td>-</td><td>-</td><td>49.7</td><td>71.6</td></tr><tr><td>(a) PLAID ColBERTv2（PLAID ColBERTv2） [37]</td><td>${39.8}^{bd}$</td><td>74.6</td><td>${85.2}^{b}$</td><td>-</td><td>${69.6}^{ * }$</td></tr><tr><td>(b) $\mathrm{{BM}}{25} \gg  \mathrm{C}\left( {k = {50}}\right)$</td><td>34.3</td><td>68.7</td><td>73.9</td><td>49.0</td><td>62.8</td></tr><tr><td>(c) $\mathrm{S} \gg  \mathrm{C}\left( {k = {50}}\right)$</td><td>${40.4}^{bd}$</td><td>74.4</td><td>${87.5}^{b}$</td><td>49.9</td><td>72.0</td></tr><tr><td colspan="6">- SPLATE ColBERTv2 $\left( {k = {50}}\right)$（SPLATE ColBERTv2 $\left( {k = {50}}\right)$）</td></tr><tr><td>(d) SPLATE (R)（(d) SPLATE (R)）</td><td>${36.7}^{b}$</td><td>72.9</td><td>${84.4}^{b}$</td><td>46.5</td><td>66.7</td></tr><tr><td>(e) SPLATE (e2e)（(e) SPLATE (e2e)）</td><td>${40.0}^{bd}$</td><td>74.2</td><td>${84.4}^{b}$</td><td>49.6</td><td>71.0</td></tr></tbody></table>

<!-- Media -->

Table 3: BoW SPLATE representations for queries in the MS MARCO dev set with $\left( {{k}_{q},{k}_{d}}\right)  = \left( {{10},{100}}\right)$ (model from Table 2). to conduct efficient sparse retrieval with SPLADE. When evaluated end-to-end, the SPLATE implementation of ColBERTv2 performs comparably to ColBERTv2 and PLAID on several benchmarks, by re-ranking a handful of documents. The sparse term-based nature of the candidate generation step makes it particularly appealing in mono-CPU environments efficiency-wise. Beyond optimizing late interaction retrieval, our work opens the path to a deeper study of the link between the representations trained from different architectures.

表3：使用$\left( {{k}_{q},{k}_{d}}\right)  = \left( {{10},{100}}\right)$（表2中的模型）为MS MARCO开发集中的查询生成的词袋（BoW）SPLATE表示，以便使用SPLADE进行高效的稀疏检索。在进行端到端评估时，ColBERTv2的SPLATE实现通过对少量文档进行重排序，在多个基准测试中的表现与ColBERTv2和PLAID相当。候选生成步骤基于稀疏词项的特性，使其在单核CPU环境下的效率方面特别有吸引力。除了优化后期交互检索之外，我们的工作为深入研究不同架构训练的表示之间的联系开辟了道路。

<!-- Media -->

<table><tr><td>SPLATE BoW</td></tr><tr><td>$Q \rightarrow$ "what is the medium for an artisan" - (medium, 2.2), (art, 1.8), (##isan, 1.7), (media, 1.1), (craftsman, 0.9), (arts, 0.6 ), (carpenter, 0.6 ), (artist, 0.5 ), (##vre, 0.4 ), (draper, 0.3)</td></tr><tr><td>$Q \rightarrow$ "treating tension headaches without medication" - (headache, 2.1), (tension, 1.8), (without, 1.6), (treatment, 1.5), (treat, 1.4), (medication, 1.3), (drug, 0.8), (baker, 0.7), (no, 0.6), (stress, 0.5)</td></tr></table>

<table><tbody><tr><td>SPLATE词袋模型（SPLATE BoW）</td></tr><tr><td>$Q \rightarrow$ “工匠的媒介是什么” - （媒介（medium）, 2.2）, （艺术（art）, 1.8）, （##isan, 1.7）, （媒体（media）, 1.1）, （工匠（craftsman）, 0.9）, （艺术（arts）, 0.6 ）, （木匠（carpenter）, 0.6 ）, （艺术家（artist）, 0.5 ）, （##vre, 0.4 ）, （布商（draper）, 0.3）</td></tr><tr><td>$Q \rightarrow$ “不使用药物治疗紧张性头痛” - （头痛（headache）, 2.1）, （紧张（tension）, 1.8）, （不（without）, 1.6）, （治疗（treatment）, 1.5）, （治疗（treat）, 1.4）, （药物（medication）, 1.3）, （药品（drug）, 0.8）, （面包师（baker）, 0.7）, （无（no）, 0.6）, （压力（stress）, 0.5）</td></tr></tbody></table>

<!-- Media -->

## REFERENCES

## 参考文献

[1] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2016. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. In InCoCo@NIPS.

[1] 帕亚尔·巴贾杰（Payal Bajaj）、丹尼尔·坎波斯（Daniel Campos）、尼克·克拉斯韦尔（Nick Craswell）、李邓（Li Deng）、高剑锋（Jianfeng Gao）、刘旭东（Xiaodong Liu）、兰甘·马朱姆德（Rangan Majumder）、安德鲁·麦克纳马拉（Andrew McNamara）、巴斯卡尔·米特拉（Bhaskar Mitra）、特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、阿利娜·斯托伊卡（Alina Stoica）、索拉布·蒂瓦里（Saurabh Tiwary）和王彤（Tong Wang）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。发表于神经信息处理系统大会附属研讨会InCoCo（InCoCo@NIPS）。

[2] Andrei Z. Broder, David Carmel, Michael Herscovici, Aya Soffer, and Jason Zien. 2003. Efficient Query Evaluation Using a Two-Level Retrieval Process. In Proceedings of the Twelfth International Conference on Information and Knowledge Management (New Orleans, LA, USA) (CIKM '03). Association for Computing Machinery, New York, NY, USA, 426-434. https://doi.org/10.1145/956863.956944

[2] 安德烈·Z·布罗德（Andrei Z. Broder）、大卫·卡梅尔（David Carmel）、迈克尔·赫斯科维奇（Michael Herscovici）、阿亚·索弗（Aya Soffer）和杰森·齐恩（Jason Zien）。2003年。使用两级检索过程进行高效查询评估。发表于第十二届信息与知识管理国际会议论文集（美国路易斯安那州新奥尔良）（CIKM '03）。美国计算机协会，美国纽约州纽约市，426 - 434页。https://doi.org/10.1145/956863.956944

[3] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL]

[3] 陈俭律（Jianlv Chen）、肖诗涛（Shitao Xiao）、张培天（Peitian Zhang）、罗坤（Kun Luo）、连德富（Defu Lian）和刘政（Zheng Liu）。2024年。BGE M3嵌入：通过自知识蒸馏实现多语言、多功能、多粒度的文本嵌入。arXiv:2402.03216 [计算机科学 - 计算语言学（cs.CL）]

[4] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen Voorhees. 2019. Overview of the TREC 2019 deep learning track. In TREC 2019.

[4] 尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）、丹尼尔·坎波斯（Daniel Campos）和埃伦·沃里斯（Ellen Voorhees）。2019年。2019年文本检索会议（TREC）深度学习赛道概述。收录于《2019年文本检索会议论文集》。

[5] Joshua Engels, Benjamin Coleman, Vihan Lakshman, and Anshumali Shrivastava. 2023. DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=kXfrlWXLwH

[5] 约书亚·恩格尔斯（Joshua Engels）、本杰明·科尔曼（Benjamin Coleman）、维汉·拉克什曼（Vihan Lakshman）和安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）。2023年。DESSERT：一种用于向量集查询的向量集搜索高效算法。收录于《第三十七届神经信息处理系统大会论文集》。https://openreview.net/forum?id=kXfrlWXLwH

[6] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. arXiv:2109.10086 [cs.IR]

[6] 蒂博·福尔马尔（Thibault Formal）、卡洛斯·拉桑斯（Carlos Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。SPLADE v2：用于信息检索的稀疏词汇与扩展模型（SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval）。arXiv:2109.10086 [计算机科学.信息检索（cs.IR）]

[7] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2022. From distillation to hard negative sampling: Making sparse neural ir models more effective. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2353-2359.

[7] 蒂博·福尔马尔（Thibault Formal）、卡洛斯·拉桑斯（Carlos Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2022年。从蒸馏到难负样本采样：让稀疏神经信息检索模型更有效（From distillation to hard negative sampling: Making sparse neural ir models more effective）。见第45届ACM信息检索研究与发展国际会议论文集（Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval）。2353 - 2359页。

[8] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2020. A White Box Analysis of ColBERT. arXiv:2012.09650 [cs.IR]

[8] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2020年。ColBERT的白盒分析（A White Box Analysis of ColBERT）。arXiv:2012.09650 [计算机科学.信息检索（cs.IR）]

[9] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In Proc. SIGIR. 2288-2292.

[9] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。SPLADE：用于第一阶段排序的稀疏词汇与扩展模型（SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking）。见SIGIR会议论文集（Proc. SIGIR）。2288 - 2292页。

[10] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2022. Match Your Words! A Study of Lexical Matching in Neural Information Retrieval. In Advances in Information Retrieval, Matthias Hagen, Suzan Verberne, Craig Macdonald, Christin Seifert, Krisztian Balog, Kjetil Nørvâg, and Vinay Setty (Eds.). Springer International Publishing, Cham, 120-127.

[10] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2022 年。匹配你的词汇！神经信息检索中的词汇匹配研究。见《信息检索进展》，马蒂亚斯·哈根（Matthias Hagen）、苏珊·韦尔伯恩（Suzan Verberne）、克雷格·麦克唐纳（Craig Macdonald）、克里斯汀·塞弗特（Christin Seifert）、克里斯蒂安·巴洛格（Krisztian Balog）、克耶蒂尔·诺尔瓦格（Kjetil Nørvâg）和维奈·塞蒂（Vinay Setty）（编）。施普林格国际出版公司，尚姆，第 120 - 127 页。

[11] Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List. In Proc. NAACL-HLT. 3030-3042.

[11] 高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡伦（Jamie Callan）。2021 年。COIL：借助上下文倒排列表重新审视信息检索中的精确词汇匹配。见《北美计算语言学协会人类语言技术分会会议录》。第 3030 - 3042 页。

[12] Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions Using Enhanced Reduction. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (Atlanta, GA, USA) (CIKM '22). Association for Computing Machinery, New York, NY, USA, 737-747. https://doi.org/10.1145/3511808.3557367

[12] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、奥马尔·哈塔卜（Omar Khattab）、索菲娅·阿尔塔默（Sophia Althammer）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2022年。使用ColBERTer引入全词神经词袋：利用增强约简的上下文延迟交互。见第31届ACM信息与知识管理国际会议论文集（美国佐治亚州亚特兰大）（CIKM '22）。美国计算机协会，美国纽约州纽约市，737 - 747页。https://doi.org/10.1145/3511808.3557367

[13] Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2021. Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. arXiv:2010.02666 [cs.IR]

[13] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、索菲娅·阿尔塔默（Sophia Althammer）、迈克尔·施罗德（Michael Schröder）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2021年。利用跨架构知识蒸馏改进高效神经排序模型。预印本arXiv:2010.02666 [计算机科学.信息检索]

[14] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-Efficient Transfer Learning for NLP. In Proceedings of the 36th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 97), Kamalika Chaudhuri and Ruslan Salakhutdinov (Eds.). PMLR, 2790-2799. https://proceedings.mlr.press/v97/houlsby19a.html

[14] 尼尔·霍尔兹比（Neil Houlsby）、安德烈·久尔久（Andrei Giurgiu）、斯坦尼斯瓦夫·亚斯特热布斯基（Stanislaw Jastrzebski）、布鲁娜·莫罗内（Bruna Morrone）、昆汀·德拉鲁西耶（Quentin De Laroussilhe）、安德里亚·格斯蒙多（Andrea Gesmundo）、莫娜·阿塔里扬（Mona Attariyan）和西尔万·热利（Sylvain Gelly）。2019年。自然语言处理的参数高效迁移学习。见第36届国际机器学习会议论文集（机器学习研究会议录，第97卷），卡玛利卡·乔杜里（Kamalika Chaudhuri）和鲁斯兰·萨拉胡季诺夫（Ruslan Salakhutdinov）（编）。机器学习研究会议录，第2790 - 2799页。https://proceedings.mlr.press/v97/houlsby19a.html

[15] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Online, 6769-6781. https://doi.org/10.18653/v1/2020.emnlp-main.550

[15] 弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen - tau Yih）。2020年。开放域问答的密集段落检索。见2020年自然语言处理经验方法会议（EMNLP）论文集。计算语言学协会，线上会议，第6769 - 6781页。https://doi.org/10.18653/v1/2020.emnlp - main.550

[16] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Proc. SIGIR. 39-48.

[16] 奥马尔·哈塔卜（Omar Khattab）和马泰·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文延迟交互实现高效的段落搜索。见《SIGIR会议论文集》。第39 - 48页。

[17] Weize Kong, Jeffrey M. Dudek, Cheng Li, Mingyang Zhang, and Mike Bendersky. 2023. SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23).

[17] 孔伟泽（Weize Kong）、杰弗里·M·杜德克（Jeffrey M. Dudek）、李成（Cheng Li）、张明阳（Mingyang Zhang）和迈克·本德尔斯基（Mike Bendersky）。2023年。SparseEmbed：利用上下文嵌入学习用于检索的稀疏词汇表示。见《第46届ACM国际信息检索研究与发展会议论文集》（SIGIR '23）。

[18] Carlos Lassance and Stéphane Clinchant. 2022. An Efficiency Study for SPLADE Models. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (Madrid, Spain) (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 2220-2226. https://doi.org/10.1145/3477495.3531833

[18] 卡洛斯·拉桑斯（Carlos Lassance）和斯特凡·克兰尚（Stéphane Clinchant）。2022年。SPLADE模型的效率研究。见《第45届ACM国际信息检索研究与发展会议论文集》（西班牙马德里）（SIGIR '22）。美国计算机协会，美国纽约州纽约市，第2220 - 2226页。https://doi.org/10.1145/3477495.3531833

[19] Carlos Lassance, Simon Lupart, Hervé Déjean, Stéphane Clinchant, and Nicola Tonellotto. 2023. A Static Pruning Study on Sparse Neural Retrievers. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (Taipei, Taiwan) (SIGIR '23). Association for Computing Machinery, New York, NY, USA, 1771-1775. https://doi.org/10.1145/3539618.3591941

[19] 卡洛斯·拉桑斯（Carlos Lassance）、西蒙·卢帕尔（Simon Lupart）、埃尔韦·德让（Hervé Déjean）、斯特凡·克兰尚（Stéphane Clinchant）和尼古拉·托内洛托（Nicola Tonellotto）。2023年。稀疏神经检索器的静态剪枝研究。见第46届国际计算机协会信息检索研究与发展会议论文集（中国台湾台北）（SIGIR '23）。美国计算机协会，美国纽约州纽约市，1771 - 1775页。https://doi.org/10.1145/3539618.3591941

[20] Carlos Lassance, Maroua Maachou, Joohee Park, and Stéphane Clinchant. 2022. Learned Token Pruning in Contextualized Late Interaction over BERT (ColBERT). In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (Madrid, Spain) (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 2232-2236. https://doi.org/10.1145/ 3477495.3531835

[20] 卡洛斯·拉桑斯（Carlos Lassance）、马鲁阿·马乔（Maroua Maachou）、朴珠熙（Joohee Park）和斯特凡·克兰尚（Stéphane Clinchant）。2022年。基于BERT的上下文晚期交互（ColBERT）中的学习型Token剪枝。见第45届国际计算机协会信息检索研究与发展会议论文集（西班牙马德里）（SIGIR '22）。美国计算机协会，美国纽约州纽约市，2232 - 2236页。https://doi.org/10.1145/ 3477495.3531835

[21] Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Y. Zhao. 2023. Rethinking the Role of Token Retrieval in Multi-Vector Retrieval. arXiv:2304.01982 [cs.CL]

[21] 李镇赫（Jinhyuk Lee）、戴竹云（Zhuyun Dai）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、雷涛（Tao Lei）、伊夫特哈尔·奈姆（Iftekhar Naim）、张明伟（Ming-Wei Chang）和赵文森（Vincent Y. Zhao）。2023年。重新思考Token检索在多向量检索中的作用。预印本：arXiv:2304.01982 [计算机科学 - 计算语言学（cs.CL）]

[22] Minghan Li, Sheng-Chieh Lin, Xueguang Ma, and Jimmy Lin. 2023. SLIM: Sparsi-fied Late Interaction for Multi-Vector Retrieval with Inverted Indexes. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM. https://doi.org/10.1145/3539618.3591977

[22] 李明翰（Minghan Li）、林圣杰（Sheng-Chieh Lin）、马学光（Xueguang Ma）和林吉米（Jimmy Lin）。2023年。SLIM：使用倒排索引进行多向量检索的稀疏化后期交互。收录于第46届ACM国际信息检索研究与发展会议论文集。美国计算机协会（ACM）。https://doi.org/10.1145/3539618.3591977

[23] Minghan Li, Sheng-Chieh Lin, Barlas Oguz, Asish Ghoshal, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2023. CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Toronto, Canada, 11891-11907. https://doi.org/10.18653/v1/2023.acl-long.663

[23] 李明翰、林圣杰、巴拉斯·奥古兹、阿西什·戈沙尔、林吉米、亚沙尔·梅赫达德、易文涛和陈希伦。2023 年。CITADEL：通过动态词法路由实现条件式Token交互以进行高效有效的多向量检索。见《第 61 届计算语言学协会年会论文集（第 1 卷：长论文）》。计算语言学协会，加拿大多伦多，11891 - 11907。https://doi.org/10.18653/v1/2023.acl-long.663

[24] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2021. In-Batch Negatives for Knowledge Distillation with Tightly-Coupled Teachers for Dense Retrieval. In Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP- 2021). Association for Computational Linguistics, Online, 163-173. https://doi.org/10.18653/v1/2021.repl4nlp-1.17

[24] 林圣杰、杨政宏和林吉米。2021 年。使用紧密耦合教师进行知识蒸馏以用于密集检索的批次内负样本。见《第 6 届自然语言处理表示学习研讨会（RepL4NLP - 2021）论文集》。计算语言学协会，线上，163 - 173。https://doi.org/10.18653/v1/2021.repl4nlp-1.17

[25] Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, and Bill Byrne. 2023. Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=IWWWulAX7g

[25] 林伟哲、陈景鸿、梅景标、亚历山德鲁·科卡（Alexandru Coca）和比尔·伯恩（Bill Byrne）。2023 年。用于检索增强视觉问答的细粒度后期交互多模态检索。收录于第三十七届神经信息处理系统大会。https://openreview.net/forum?id=IWWWulAX7g

[26] Simon Lupart, Thibault Formal, and Stéphane Clinchant. 2023. MS-Shift: An Analysis of MS MARCO Distribution Shifts on Neural Retrieval. In Advances in Information Retrieval, Jaap Kamps, Lorraine Goeuriot, Fabio Crestani, Maria Maistro, Hideo Joho, Brian Davis, Cathal Gurrin, Udo Kruschwitz, and Annalina Caputo (Eds.). Springer Nature Switzerland, Cham, 636-652.

[26] 西蒙·卢帕特（Simon Lupart）、蒂博·福尔马尔（Thibault Formal）和斯特凡·克兰尚（Stéphane Clinchant）。2023 年。MS - 偏移：对神经检索中 MS MARCO 分布偏移的分析。收录于《信息检索进展》，雅普·坎普斯（Jaap Kamps）、洛林·戈里奥（Lorraine Goeuriot）、法比奥·克雷斯塔尼（Fabio Crestani）、玛丽亚·梅斯特罗（Maria Maistro）、乔保夫（Hideo Joho）、布赖恩·戴维斯（Brian Davis）、卡哈尔·古林（Cathal Gurrin）、乌多·克鲁施维茨（Udo Kruschwitz）和安娜莉娜·卡普托（Annalina Caputo）（编）。瑞士斯普林格自然出版社，尚贝里，第 636 - 652 页。

[27] Craig Macdonald and Nicola Tonellotto. 2021. On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (Virtual Event, Queensland, Australia) (CIKM '21). Association for Computing Machinery, New York, NY, USA, 3318-3322. https://doi.org/10.1145/3459637.3482156

[27] 克雷格·麦克唐纳（Craig Macdonald）和尼科拉·托内洛托（Nicola Tonellotto）。2021年。关于多阶段密集检索的近似最近邻选择。见第30届ACM信息与知识管理国际会议论文集（线上会议，澳大利亚昆士兰州）（CIKM '21）。美国计算机协会，美国纽约州纽约市，3318 - 3322。https://doi.org/10.1145/3459637.3482156

[28] Antonio Mallia, Michal Siedlaczek, Joel Mackenzie, and Torsten Suel. 2019. PISA: Performant Indexes and Search for Academia. In Proceedings of the Open-Source IR Replicability Challenge co-located with 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, OSIRRC@SIGIR 2019, Paris, France, July 25, 2019. 50-56. http://ceur-ws.org/Vol-2409/docker08.pdf

[28] 安东尼奥·马利亚（Antonio Mallia）、米哈尔·谢德拉切克（Michal Siedlaczek）、乔尔·麦肯齐（Joel Mackenzie）和托尔斯滕·苏埃尔（Torsten Suel）。2019年。PISA：面向学术界的高性能索引与搜索。见与第42届ACM SIGIR信息检索研究与发展国际会议同期举办的开源信息检索可复现性挑战赛论文集，OSIRRC@SIGIR 2019，法国巴黎，2019年7月25日。50 - 56。http://ceur-ws.org/Vol-2409/docker08.pdf

[29] Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini. 2024. Efficient Multi-vector Dense Retrieval with Bit Vectors. In Advances in Information Retrieval, Nazli Goharian, Nicola Tonellotto, Yulan He, Aldo Lipani, Graham McDonald, Craig Macdonald, and Iadh Ounis (Eds.). Springer Nature Switzerland, Cham, 3-17.

[29] 佛朗哥·玛丽亚·纳尔迪尼（Franco Maria Nardini）、科西莫·鲁利（Cosimo Rulli）和罗萨诺·文图里尼（Rossano Venturini）。2024年。基于位向量的高效多向量密集检索。见《信息检索进展》，纳兹利·戈哈里安（Nazli Goharian）、尼古拉·托内洛托（Nicola Tonellotto）、何玉兰（Yulan He）、阿尔多·利帕尼（Aldo Lipani）、格雷厄姆·麦克唐纳（Graham McDonald）、克雷格·麦克唐纳（Craig Macdonald）和伊阿德·乌尼斯（Iadh Ounis）（编）。施普林格自然瑞士出版公司，尚姆，第3 - 17页。

[30] Thong Nguyen, Sean MacAvaney, and Andrew Yates. 2023. A Unified Framework for Learned Sparse Retrieval. In European Conference on Information Retrieval. Springer, 101-116.

[30] 通·阮（Thong Nguyen）、肖恩·麦卡瓦尼（Sean MacAvaney）和安德鲁·耶茨（Andrew Yates）。2023年。学习型稀疏检索的统一框架。见《欧洲信息检索会议》。施普林格出版社，第101 - 116页。

[31] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv:Preprint: arXiv:1901.04085

[31] 罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。基于BERT的段落重排序。预印本arXiv:1901.04085

[32] Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, and Sebastian Ruder. 2020. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Online, 7654- 7673. https://doi.org/10.18653/v1/2020.emnlp-main.617

[32] 乔纳斯·普法伊费尔（Jonas Pfeiffer）、伊万·武利奇（Ivan Vulić）、伊琳娜·古列维奇（Iryna Gurevych）和塞巴斯蒂安·鲁德尔（Sebastian Ruder）。2020年。MAD - X：基于适配器的多任务跨语言迁移框架。见《2020年自然语言处理经验方法会议论文集》（EMNLP）。计算语言学协会，线上会议，第7654 - 7673页。https://doi.org/10.18653/v1/2020.emnlp - main.617

[33] Yujie Qian, Jinhyuk Lee, Sai Meher Karthik Duddu, Zhuyun Dai, Siddhartha Brahma, Iftekhar Naim, Tao Lei, and Vincent Y. Zhao. 2022. Multi-Vector Retrieval as Sparse Alignment. arXiv:2211.01267 [cs.CL]

[33] 钱玉洁、李晋赫、赛·梅赫尔·卡尔蒂克·杜杜、戴竹云、悉达多·布拉马、伊夫特哈尔·奈姆、雷涛和赵文森·Y。2022年。多向量检索作为稀疏对齐。预印本arXiv:2211.01267 [计算机科学 - 计算语言学（cs.CL）]

[34] Yifan Qiao, Yingrui Yang, Shanxiu He, and Tao Yang. 2023. Representation Sparsification with Hybrid Thresholding for Fast SPLADE-based Document

[34] 乔一帆、杨英锐、何善秀和杨涛。2023年。基于混合阈值的表示稀疏化用于快速基于SPLADE的文档

Retrieval. arXiv preprint arXiv:2306.11293 (2023).

检索。预印本arXiv:2306.11293 (2023)。

[35] Yifan Qiao, Yingrui Yang, Haixin Lin, and Tao Yang. 2023. Optimizing Guided Traversal for Fast Learned Sparse Retrieval. In Proceedings of the ACM Web Conference 2023. 3375-3385.

[35] 乔一帆、杨英锐、林海鑫和杨涛。2023年。优化引导遍历以实现快速学习型稀疏检索。收录于《2023年ACM网络会议论文集》。3375 - 3385。

[36] Ori Ram, Liat Bezalel, Adi Zicher, Yonatan Belinkov, Jonathan Berant, and Amir Globerson. 2023. What Are You Token About? Dense Retrieval as Distributions Over the Vocabulary. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Toronto, Canada, 2481-2498. https: //doi.org/10.18653/v1/2023.acl-long.140

[36] 奥里·拉姆（Ori Ram）、利亚特·贝扎莱尔（Liat Bezalel）、阿迪·齐彻（Adi Zicher）、约纳坦·贝林科夫（Yonatan Belinkov）、乔纳森·贝兰特（Jonathan Berant）和阿米尔·格洛伯森（Amir Globerson）。2023年。你在说什么Token？将密集检索视为词汇表上的分布。见《计算语言学协会第61届年会论文集（第1卷：长论文）》。计算语言学协会，加拿大多伦多，2481 - 2498页。https: //doi.org/10.18653/v1/2023.acl-long.140

[37] Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022. PLAID: An Efficient Engine for Late Interaction Retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (Atlanta, GA, USA) (CIKM '22). Association for Computing Machinery, New York, NY, USA, 1747-1756. https://doi.org/10.1145/3511808.3557325

[37] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特伊·扎哈里亚（Matei Zaharia）。2022年。PLAID：一种用于后期交互检索的高效引擎。见《第31届ACM国际信息与知识管理大会论文集（美国佐治亚州亚特兰大）（CIKM '22）》。美国计算机协会，美国纽约州纽约市，1747 - 1756页。https://doi.org/10.1145/3511808.3557325

[38] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2021. Colbertv2: Effective and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488 (2021).

[38] 凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad-Falcon）、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）。2021年。Colbertv2：通过轻量级后期交互实现有效且高效的检索。预印本 arXiv:2112.01488 (2021)。

[39] Mete Sertkan, Sophia Althammer, and Sebastian Hofstätter. 2023. Ranger: A Toolkit for Effect-Size Based Multi-Task Evaluation. arXiv preprint arXiv:2305.15048 (2023).

[39] 梅特·塞尔特坎（Mete Sertkan）、索菲娅·阿尔塔默（Sophia Althammer）和塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）。2023年。Ranger：基于效应量的多任务评估工具包。预印本 arXiv:2305.15048 (2023)。

[40] Tao Shen, Xiubo Geng, Chongyang Tao, Can Xu, Guodong Long, Kai Zhang, and Daxin Jiang. 2023. UnifieR: A Unified Retriever for Large-Scale Retrieval. arXiv:2205.11194 [cs.IR]

[40] 沈涛、耿秀波、陶重阳、徐灿、龙国栋、张凯和蒋大新。2023年。UnifieR：用于大规模检索的统一检索器。arXiv:2205.11194 [计算机科学 - 信息检索（cs.IR）]

[41] Susav Shrestha, Narasimha Reddy, and Zongwang Li. 2023. ESPN: Memory-Efficient Multi-Vector Information Retrieval. arXiv:2312.05417 [cs.IR]

[41] 苏萨夫·什雷斯塔（Susav Shrestha）、纳拉辛哈·雷迪（Narasimha Reddy）和李宗旺。2023年。ESPN：内存高效的多向量信息检索。arXiv:2312.05417 [计算机科学 - 信息检索（cs.IR）]

[42] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). https://openreview.net/forum?id=wCu6T5xFjeJ

[42] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021年。BEIR：信息检索模型零样本评估的异构基准。收录于第三十五届神经信息处理系统数据集与基准赛道会议（第二轮）。https://openreview.net/forum?id=wCu6T5xFjeJ

[43] Nicola Tonellotto and Craig Macdonald. 2021. Query Embedding Pruning for Dense Retrieval. In Proc. CIKM. 3453-3457.

[43] 尼古拉·托内洛托（Nicola Tonellotto）和克雷格·麦克唐纳（Craig Macdonald）。2021年。用于密集检索的查询嵌入剪枝。收录于《CIKM会议论文集》。第3453 - 3457页。

[44] Howard Turtle and James Flood. 1995. Query Evaluation: Strategies and Optimizations. Inf. Process. Manage. 31, 6 (nov 1995), 831-850. https://doi.org/10.1016/0306- 4573(95)00020-H

[44] 霍华德·特特尔（Howard Turtle）和詹姆斯·弗洛德（James Flood）。1995年。查询评估：策略与优化。《信息处理与管理》，第31卷，第6期（1995年11月），第831 - 850页。https://doi.org/10.1016/0306- 4573(95)00020-H

[45] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval (Virtual Event, Canada) (ICTIR '21). Association for Computing Machinery, New York, NY, USA, 297-306. https://doi.org/10.1145/3471158.3472250

[45] 小王、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2021年。多表示密集检索的伪相关反馈。收录于《2021年ACM SIGIR信息检索理论国际会议论文集》（线上会议，加拿大）（ICTIR '21）。美国计算机协会，美国纽约州纽约市，297 - 306页。https://doi.org/10.1145/3471158.3472250

[46] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2023. Reproducibility, Replicability, and Insights into Dense Multi-Representation Retrieval Models: From ColBERT to Col*. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (Taipei, Taiwan) (SIGIR '23). Association for Computing Machinery, New York, NY, USA, 2552-2561. https://doi.org/10.1145/3539618.3591916

[46] 小王、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2023年。密集多表示检索模型的可复现性、可重复性及洞察：从ColBERT到Col*。收录于《第46届ACM SIGIR信息检索研究与发展国际会议论文集》（中国台湾台北）（SIGIR '23）。美国计算机协会，美国纽约州纽约市，2552 - 2561页。https://doi.org/10.1145/3539618.3591916

[47] Orion Weller, Dawn Lawrie, and Benjamin Van Durme. 2023. NevIR: Negation in Neural Information Retrieval. arXiv:2305.07614 [cs.IR]

[47] 奥赖恩·韦勒（Orion Weller）、道恩·劳里（Dawn Lawrie）和本杰明·范·杜尔姆（Benjamin Van Durme）。2023年。NevIR：神经信息检索中的否定。arXiv:2305.07614 [计算机科学.信息检索（cs.IR）]

[48] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. 2022. FILIP: Fine-grained Interactive Language-Image Pre-Training. In International Conference on Learning Representations. https://openreview.net/forum?id=cpDhesEDC2

[48] 姚乐伟（Lewei Yao）、黄润辉（Runhui Huang）、侯璐（Lu Hou）、卢冠松（Guansong Lu）、牛敏哲（Minzhe Niu）、徐航（Hang Xu）、梁晓丹（Xiaodan Liang）、李震国（Zhenguo Li）、蒋鑫（Xin Jiang）和徐春静（Chunjing Xu）。2022年。FILIP：细粒度交互式语言 - 图像预训练。发表于国际学习表征会议。https://openreview.net/forum?id=cpDhesEDC2

[49] Jingtao Zhan, Xiaohui Xie, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2022. Evaluating Interpolation and Extrapolation Performance of Neural Retrieval Models. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (Atlanta, GA, USA) (CIKM '22). Association for Computing Machinery, New York, NY, USA, 2486-2496. https://doi.org/10.1145/3511808.3557312

[49] 詹景涛、谢晓辉、毛佳欣、刘奕群、郭佳峰、张敏和马少平。2022年。评估神经检索模型的内插和外推性能。见第31届ACM信息与知识管理国际会议论文集（美国佐治亚州亚特兰大）（CIKM '22）。美国计算机协会，美国纽约州纽约市，2486 - 2496。https://doi.org/10.1145/3511808.3557312