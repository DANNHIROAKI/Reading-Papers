## ColBERT-serve: Efficient Multi-Stage Memory-Mapped Scoring

## ColBERT服务：高效的多阶段内存映射评分

Kaili Huang ${}^{\boxtimes 1 \dagger  }$ ,Thejas Venkatesh ${}^{2 \dagger  }$ ,Uma Dingankar ${}^{3 \dagger   \ddagger  }$ ,Antonio Mallia ${}^{4}$ , Daniel Campos ${}^{5}$ ,Jian ${\mathrm{{Jiao}}}^{1}$ ,Christopher Potts ${}^{6}$ ,Matei Zaharia ${}^{7}$ ,Kwabena Boahen ${}^{6}$ ,Omar Khattab ${}^{6}$ ,Saarthak Sarup ${}^{6}$ ,and Keshav Santhanam ${}^{6}$

黄凯莉 ${}^{\boxtimes 1 \dagger  }$ 、泰贾斯·文卡特什 ${}^{2 \dagger  }$ 、乌玛·丁坎卡尔 ${}^{3 \dagger   \ddagger  }$ 、安东尼奥·马利亚 ${}^{4}$ 、丹尼尔·坎波斯 ${}^{5}$ 、简 ${\mathrm{{Jiao}}}^{1}$ 、克里斯托弗·波茨 ${}^{6}$ 、马泰·扎哈里亚 ${}^{7}$ 、夸贝纳·博阿亨 ${}^{6}$ 、奥马尔·哈塔卜 ${}^{6}$ 、萨尔萨克·萨鲁普 ${}^{6}$ 和凯沙夫·桑塔南 ${}^{6}$

${}^{1}$ Microsoft,Redmond,WA,USA

${}^{1}$ 美国华盛顿州雷德蒙德市微软公司

kaili.khuang@gmail.com, jian.jiao@microsoft.com

kaili.khuang@gmail.com, jian.jiao@microsoft.com

${}^{2}$ Samaya AI,Mountain View,CA,USA

${}^{2}$ 美国加利福尼亚州山景城萨玛雅人工智能公司

thejas@stanford.edu

${}^{3}$ Foundry,Palo Alto,CA,USA

${}^{3}$ 美国加利福尼亚州帕洛阿尔托市铸造厂

uma@mlfoundry.com

${}^{4}$ Pinecone,New York,NY,USA

${}^{4}$ 美国纽约州纽约市松果公司

me@antoniomallia.it

${}^{5}$ Snowflake,New York,NY,USA

${}^{5}$ 美国纽约州纽约市雪花公司

daniel.campos@snowflake.com

${}^{6}$ Stanford University,Stanford,CA,USA

${}^{6}$ 美国加利福尼亚州斯坦福市斯坦福大学

\{cgpotts, boahen, okhattab, ssarup, keshav2\}@stanford.edu

\{cgpotts, boahen, okhattab, ssarup, keshav2\}@stanford.edu

${}^{7}\mathrm{{UC}}$ Berkeley,Berkeley,CA,USA

${}^{7}\mathrm{{UC}}$ 美国加利福尼亚州伯克利市伯克利

matei@berkeley.edu

Abstract. We study serving retrieval models, particularly late interaction retrievers like ColBERT, to many concurrent users at once and under a small budget, in which the index may not fit in memory. We present ColBERT-serve, a serving system that applies a memory-mapping strategy to the ColBERT index, reducing RAM usage by 90% and permitting its deployment on cheap servers, and incorporates a multi-stage architecture with hybrid scoring, reducing ColBERT's query latency and supporting many concurrent queries in parallel.

摘要。我们研究了在预算有限的情况下，同时为大量并发用户提供检索模型服务的问题，特别是像ColBERT这样的后期交互检索器，在这种情况下，索引可能无法全部装入内存。我们提出了ColBERT服务系统，该系统对ColBERT索引应用了内存映射策略，将随机存取存储器（RAM）的使用量减少了90%，并允许在廉价服务器上部署该系统。此外，该系统还采用了具有混合评分的多阶段架构，减少了ColBERT的查询延迟，并支持并行处理大量并发查询。

Keywords: Information Retrieval - ColBERT - Efficiency

关键词：信息检索 - ColBERT - 效率

## 1 Introduction

## 1 引言

Multi-vector late-interaction retrievers like ColBERT [10] and ColPali [5] have demonstrated state-of-the-art quality and superior generalization [31] while maintaining low latency, but despite major progress in compressing their embeddings 28 8, hosting a ColBERT index of Wikipedia (20M passages) via PLAID [27 demands nearly ${100}\mathrm{{GB}}$ of RAM. This poses a challenge for serving such models on cheap servers with little RAM, especially if we need to serve many concurrent users with low latency. Unfortunately, cost, latency, and quality tradeoffs in such a high-concurrency, low-memory regime are rarely considered jointly in the existing neural IR literature.

像ColBERT [10] 和ColPali [5] 这样的多向量后期交互检索器在保持低延迟的同时，展现出了最先进的性能和出色的泛化能力 [31] 。然而，尽管在压缩其嵌入方面取得了重大进展 [28, 8] ，但通过PLAID [27] 托管维基百科（2000万个段落）的ColBERT索引仍需要近 ${100}\mathrm{{GB}}$ 的随机存取存储器（RAM）。这给在内存有限的廉价服务器上部署此类模型带来了挑战，特别是当我们需要以低延迟为大量并发用户提供服务时。不幸的是，在现有的神经信息检索文献中，很少同时考虑在这种高并发、低内存情况下的成本、延迟和性能之间的权衡。

---

<!-- Footnote -->

${}^{ \dagger  }$ K. Huang,T. Venkatesh,and U. Dingankar contributed equally to this work.

${}^{ \dagger  }$ 黄凯莉、泰贾斯·文卡特什和乌玛·丁坎卡尔对这项工作贡献相同。

${}^{ \ddagger  }$ Work by U. Dingankar was done while at Stanford.

${}^{ \ddagger  }$ U. 丁坎卡尔（U. Dingankar）的工作是在斯坦福大学时完成的。

<!-- Footnote -->

---

We tackle this with the following contributions. First, we present a methodology and benchmark for evaluating the concurrent serving of neural IR models under different traffic workloads and memory budgets. Second, we introduce ColBERT-serve ${}^{1}$ which (1) incorporates a new memory-mapping architecture, permitting the bulk of ColBERTv2's index to reside on disk, (2) minimizes access to this index via a multi-stage retrieval process, (3) handles concurrent requests in parallel with low latency and scales gracefully under load by adapting PISA and ColBERTv2, and (4) preserves the quality of full ColBERTv2 retrieval through a hybrid scoring technique. Third, we conduct an empirical evaluation that demonstrates the first ColBERT serving system that can serve up to 4 queries per second on a server with as little as a few GBs of RAM (90% reduction in RAM usage for loading the model compared to the full ColBERTv2) for massive collections while preserving quality.

我们通过以下贡献来解决这个问题。首先，我们提出了一种方法和基准，用于评估在不同流量负载和内存预算下神经信息检索（IR）模型的并发服务情况。其次，我们引入了 ColBERT - serve ${}^{1}$，它（1）采用了一种新的内存映射架构，允许 ColBERTv2 的大部分索引驻留在磁盘上；（2）通过多阶段检索过程最小化对该索引的访问；（3）通过适配 PISA 和 ColBERTv2，以低延迟并行处理并发请求，并在负载下实现良好的扩展；（4）通过混合评分技术保留完整 ColBERTv2 检索的质量。第三，我们进行了实证评估，展示了首个 ColBERT 服务系统，该系统可以在仅配备几 GB 随机存取存储器（RAM）的服务器上每秒处理多达 4 个查询（与完整的 ColBERTv2 相比，加载模型所需的 RAM 使用量减少了 90%），同时保持检索质量。

## 2 Related Work

## 2 相关工作

Memory-mapping is a technique for accessing data from disk while only materializing accessed portions in memory on demand. It is used in approximate nearest neighbor search [9.4], database systems [11], and concurrently with our work also in neural IR [30]. Memory-mapped indexes pose the challenge of minimizing the latency overhead incurred by page misses. Whereas [30] built a prefetcher to reduce the impact of SSD latencies, we seek to reduce the number of accesses to disk directly via multi-stage retrieval. Much existing work has studied improving the latency or memory footprint of ColBERT-like models [28 27 29 17 25 15 6 or the quality-cost tradeoff [10 21 19 3] of using ColBERT to re-rank results produced by simpler systems like BM25 [26 or LADR [12]. This general strategy is generally known to lead to improved latency but comes at the cost of a reduction in MRR and recall. We build a concurrent serving system for ColBERTv2 that permits the index to mostly reside on disk without sacrificing retrieval quality or latency under high traffic. We achieve this by leveraging a combination of memory-mapping and a multi-stage retrieval approach that utilizes scores from both the candidate generation and the re-ranking steps. This hybrid scoring method leverages the strengths of both stages, resulting in performance that can surpass fully in-memory ColBERTv2 retrieval.

内存映射是一种从磁盘访问数据的技术，仅在需要时将访问的部分数据加载到内存中。它被用于近似最近邻搜索 [9.4]、数据库系统 [11]，并且在我们开展工作的同时也被用于神经信息检索 [30]。内存映射索引面临的挑战是最小化页面缺失带来的延迟开销。[30] 构建了一个预取器来减少固态硬盘（SSD）延迟的影响，而我们则试图通过多阶段检索直接减少对磁盘的访问次数。许多现有工作研究了如何改善类似 ColBERT 模型的延迟或内存占用 [28 27 29 17 25 15 6]，或者研究了使用 ColBERT 对像 BM25 [26] 或 LADR [12] 这样的简单系统生成的结果进行重新排序时的质量 - 成本权衡 [10 21 19 3]。这种通用策略通常可以改善延迟，但会导致平均倒数排名（MRR）和召回率下降。我们为 ColBERTv2 构建了一个并发服务系统，该系统允许索引大部分驻留在磁盘上，同时在高流量下不牺牲检索质量或延迟。我们通过结合使用内存映射和多阶段检索方法来实现这一目标，该多阶段检索方法利用了候选生成和重新排序步骤的得分。这种混合评分方法发挥了两个阶段的优势，其性能可以超越完全在内存中运行的 ColBERTv2 检索。

## 3 ColBERT-serve

## 3 ColBERT - serve

Memory-Mapped Storage To deploy ColBERTv2 on memory-constrained machines, we introduce memory-mapping into the ColBERT implementation, specifically for the tensors encoding the compressed ColBERTv2 embeddings. This bypasses loading the index upfront and instead enables the operating system to manage limited memory resources, by bringing only accessed data into memory at the page granularity and evicting pages when RAM is insufficient. This reduces the RAM requirements by over 90%.

内存映射存储 为了在内存受限的机器上部署 ColBERTv2，我们将内存映射引入到 ColBERT 的实现中，具体是针对编码压缩后的 ColBERTv2 嵌入的张量。这避免了预先加载索引，而是让操作系统管理有限的内存资源，仅以页面为粒度将访问的数据加载到内存中，并在 RAM 不足时淘汰页面。这将 RAM 需求降低了 90% 以上。

---

<!-- Footnote -->

1 https://github.com/stanford-futuredata/colbert-serve

1 https://github.com/stanford - futuredata/colbert - serve

<!-- Footnote -->

---

Concurrent Requests We build a server-client architecture for deployment as well as experimentation for ColBERTv2. To this end, we improve Col-BERTv2's multithreading compatibility by releasing Python's Global Interpreter Lock while invoking all underlying functionality of ColBERTv2 implemented as $\mathrm{C} +  +$ extensions ${}^{2}$ Without this,multithreading for ColBERTv2 was prohibitively expensive as each query would block when extensions are invoked, so concurrency was only possible by launching multiple processes, which - without memory mapping-would scale memory consumption linearly with the number of processes. With support for memory mapping, we tune the number of threads used to serve each ColBERTv2 request and find that though multithreading improves performance under low load, single-threaded performance dominates under higher loads; hence, we use only a single thread for all our experiments. In addition, we adapt the PISA [22] engine for this setting to support our server-client architecture with the multi-stage retrieval discussed next. We leave the comparison with more recent dynamic pruning strategies specifically designed for learned sparse retrieval models [2420] as future work.

并发请求 我们为 ColBERTv2 的部署和实验构建了一个服务器 - 客户端架构。为此，我们通过在调用以 $\mathrm{C} +  +$ 扩展 ${}^{2}$ 形式实现的 ColBERTv2 的所有底层功能时释放 Python 的全局解释器锁，提高了 ColBERTv2 的多线程兼容性。如果不这样做，ColBERTv2 的多线程开销会非常大，因为每次调用扩展时查询都会阻塞，所以只能通过启动多个进程来实现并发，而在没有内存映射的情况下，内存消耗会随着进程数量线性增长。在支持内存映射的情况下，我们调整了用于处理每个 ColBERTv2 请求的线程数量，发现虽然多线程在低负载下可以提高性能，但在高负载下单线程性能更优；因此，我们在所有实验中都只使用单线程。此外，我们针对这种情况对 PISA [22] 引擎进行了适配，以支持我们的服务器 - 客户端架构和接下来要讨论的多阶段检索。我们将与专门为学习型稀疏检索模型设计的最新动态剪枝策略 [24 20] 进行比较作为未来的工作。

Multi-Stage Retrieval Memory-mapping introduces a key challenge: due to the latency incurred by page misses, searching over MS MARCO with a memory-mapped index is approximately $2 \times$ slower than an in-memory index. We tackle this via a multi-stage ranking architecture, in which SPLADEv2 [7], a learned sparse model [18,32,33,2], serves as the first-stage retriever to minimize the number of documents we need to access from the ColBERT index. As a baseline ${}^{3}$ we use the standard ColBERTv2 with PLAID [27] with a machine capable of fitting the entire index in memory. Then, we implement and study four different systems: (1) MMAP ColBERTv2, in which we apply memory-mapping to the end-to-end process of PLAID; (2) SPLADEv2 w/ PISA, in which SPLADEv2 expands queries and the PISA engine performs efficient retrieval [22; (3) MMAP Rerank, in which SPLADEv2 generates top-200 candidates per query and MMAP ColBERTv2 re-ranks them; and MMAP Hybrid, in which SPLADEv2's top-200 results are re-ranked via a linear interpolation between SPLADEv2 and MMAP ColBERTv2. For a given query $Q$ and document $D$ ,the hybrid score is given by:

多阶段检索内存映射带来了一个关键挑战：由于页面缺失导致的延迟，使用内存映射索引在MS MARCO上进行搜索比内存内索引大约慢$2 \times$。我们通过一种多阶段排序架构来解决这个问题，在该架构中，SPLADEv2 [7]，一种学习型稀疏模型 [18,32,33,2]，作为第一阶段的检索器，以最小化我们需要从ColBERT索引中访问的文档数量。作为基线 ${}^{3}$，我们使用配备PLAID [27] 的标准ColBERTv2，其所在机器能够将整个索引装入内存。然后，我们实现并研究了四种不同的系统：(1) MMAP ColBERTv2，我们将内存映射应用于PLAID的端到端过程；(2) SPLADEv2 w/ PISA，其中SPLADEv2扩展查询，PISA引擎执行高效检索 [22]；(3) MMAP重排序，其中SPLADEv2为每个查询生成前200个候选文档，MMAP ColBERTv2对它们进行重排序；以及MMAP混合，其中SPLADEv2的前200个结果通过SPLADEv2和MMAP ColBERTv2之间的线性插值进行重排序。对于给定的查询 $Q$ 和文档 $D$，混合得分由以下公式给出：

$$
{S}_{\text{hybrid }}\left( {D,Q}\right)  = {\alpha N}\left( {{S}_{\text{SPLADE }}\left( {D,Q}\right) }\right)  + \left( {1 - \alpha }\right) N\left( {{S}_{\text{ColBERT }}\left( {D,Q}\right) }\right) 
$$

where $S\left( {*, * }\right)$ is the score function, $N\left( *\right)$ is the normalization function,and $\alpha$ is a coefficient between 0 and 1. SPLADEv2 and ColBERTv2 produce scores of drastically different distributions, a likely source of quality for hybrid scoring.

其中 $S\left( {*, * }\right)$ 是得分函数，$N\left( *\right)$ 是归一化函数，$\alpha$ 是一个介于0和1之间的系数。SPLADEv2和ColBERTv2产生的得分分布差异巨大，这可能是混合评分质量的来源。

---

<!-- Footnote -->

${}^{2}$ This optimization was implemented in May 2024. Since then,Python 3.13 has introduced experimental support for a GIL-free mode.

${}^{2}$ 此优化于2024年5月实现。从那时起，Python 3.13引入了对无全局解释器锁（GIL）模式的实验性支持。

${}^{3}$ We build on code from https://github.com/stanford-futuredata/ColBERT,https: //github.com/naver/splade, and https://github.com/pisa-engine/pisa

${}^{3}$ 我们基于来自https://github.com/stanford-futuredata/ColBERT、https://github.com/naver/splade和https://github.com/pisa-engine/pisa的代码进行开发。

<!-- Footnote -->

---

To combine these scores, we explored (1) linearly mapping each to the range of $\left\lbrack  {0,1}\right\rbrack  ,\left( 2\right)$ min-max norm,and (3) z-norm. Among these,z-norm yielded the best results,so we select that as the normalization function,defined as: $N\left( x\right)  = \frac{x - \bar{x}}{S}$ where $\bar{x}$ denotes the mean of samples and $S$ denotes the standard deviation.

为了组合这些得分，我们探索了(1) 将每个得分线性映射到 $\left\lbrack  {0,1}\right\rbrack  ,\left( 2\right)$ 的范围、(2) 最小 - 最大归一化和(3) z - 归一化。在这些方法中，z - 归一化产生了最佳结果，因此我们选择它作为归一化函数，定义为：$N\left( x\right)  = \frac{x - \bar{x}}{S}$ 其中 $\bar{x}$ 表示样本的均值，$S$ 表示标准差。

## 4 Evaluation

## 4 评估

We now test the impact of multi-stage retrieval on quality, of memory-mapping on RAM usage, and of both together on latency under varying traffic.

我们现在测试多阶段检索对质量的影响、内存映射对随机存取存储器（RAM）使用的影响，以及在不同流量下两者共同对延迟的影响。

Methodology We use MS MARCO Passage Ranking development set that contains $7\mathrm{\;K}$ queries and ${8.8}\mathrm{M}$ passages [1] as an "in-domain" benchmark for ColBERTv2 and SPLADEv2 and report MRR@10, Recall@5 and Recall@50.To test out-of-domain (OOD) generalization, we use Wikipedia Open-QA NQ-dev with ${8.7}\mathrm{\;K}$ queries and ${21}\mathrm{M}$ passages [1316] and LoTTE Search Lifestyle-dev with 417 queries and ${269}\mathrm{\;K}$ passages [28]. These popular datasets differ dramatically in size, with Wikipedia stressing RAM usage and LoTTE Lifestyle always fitting in memory. Following [28], we report Success@5.We report the mean latency and tail (95th and the 99th percentiles) latency observed by the concurrent clients in our client-server architecture under varying degrees of server load. We measure latency over the first $1\mathrm{\;K}$ queries from each dataset,a sufficient size to saturate the system under high load conditions. The number of queries per second (QPS) is sampled using a Poisson distribution.

方法 我们使用包含 $7\mathrm{\;K}$ 个查询和 ${8.8}\mathrm{M}$ 个段落的MS MARCO段落排名开发集 [1] 作为ColBERTv2和SPLADEv2的“领域内”基准，并报告MRR@10、Recall@5和Recall@50。为了测试领域外（OOD）泛化能力，我们使用包含 ${8.7}\mathrm{\;K}$ 个查询和 ${21}\mathrm{M}$ 个段落的维基百科开放问答NQ - 开发集 [1316] 以及包含417个查询和 ${269}\mathrm{\;K}$ 个段落的LoTTE搜索生活方式 - 开发集 [28]。这些流行的数据集在规模上差异巨大，维基百科数据集对RAM使用要求较高，而LoTTE生活方式数据集始终能装入内存。按照 [28] 的做法，我们报告Success@5。我们报告在不同服务器负载程度下，客户端 - 服务器架构中并发客户端观察到的平均延迟和尾部（第95和第99百分位数）延迟。我们对每个数据集中的前 $1\mathrm{\;K}$ 个查询测量延迟，这个规模足以在高负载条件下使系统达到饱和。每秒查询数（QPS）采用泊松分布进行采样。

Choice of Hardware Since the ColBERTv2 baseline loads the entire index in memory, its experiments require a machine with a high-capacity RAM. In contrast, ColBERT-serve can run on significantly smaller machines. To highlight this important benefit, we run experiments for our method on strictly smaller, less powerful, cheaper machines rather than using the same machines as the control experiments (namely, the full ColBERTv2 baseline). This demonstrates that the proposed method has comparable quality and latency, while running on significantly cheaper and resource-constrained machines. For MS MARCO and Wikipedia, we use an AWS r6a.4xlarge instance for the control experiment, and m5ad.xlarge and r6id.xlarge instances for SPLADEv2/MMAP experiments, respectively. LoTTE Lifestyle's index is small enough to fit in a memory-restricted machine, so we run all experiments on a c5ad. xlarge instance. The key machine specifications are provided in Table 1.

硬件选择 由于ColBERTv2基线模型会将整个索引加载到内存中，因此其实验需要一台具有大容量随机存取存储器（RAM）的机器。相比之下，ColBERT-serve可以在配置显著更低的机器上运行。为了突出这一重要优势，我们在配置严格更低、性能更弱、价格更便宜的机器上对我们的方法进行实验，而不是使用与对照实验相同的机器（即完整的ColBERTv2基线模型所使用的机器）。这表明，所提出的方法在质量和延迟方面表现相当，同时可以在价格显著更低且资源受限的机器上运行。对于MS MARCO和维基百科数据集，我们在对照实验中使用AWS r6a.4xlarge实例，在SPLADEv2/MMAP实验中分别使用m5ad.xlarge和r6id.xlarge实例。LoTTE Lifestyle数据集的索引足够小，可以装入内存受限的机器，因此我们在c5ad.xlarge实例上运行所有实验。关键的机器规格见表1。

<!-- Media -->

Table 1: AWS machine specifications

表1：AWS机器规格

<table><tr><td/><td>Control</td><td>MMAP MARCO</td><td>MMAP Wiki</td><td>LoTTE</td></tr><tr><td>AWS machine</td><td>r6a.4xlarge</td><td>m5ad.xlarge</td><td>r6id.xlarge</td><td>c5ad.xlarge</td></tr><tr><td>Disk Size (GB)</td><td>950</td><td>150</td><td>237</td><td>150</td></tr><tr><td>CPU Count</td><td>16</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Memory (GB)</td><td>128</td><td>16 (-88%)</td><td>32 (-75%)</td><td>8</td></tr><tr><td>Cost (\$/month)</td><td>438</td><td>95 (-78%)</td><td>139 (-68%)</td><td>54</td></tr></table>

<table><tbody><tr><td></td><td>控制</td><td>MMAP马可（MMAP MARCO）</td><td>MMAP维基百科（MMAP Wiki）</td><td>乐天（LoTTE）</td></tr><tr><td>亚马逊云服务机器（AWS machine）</td><td>r6a.4xlarge</td><td>m5ad.xlarge</td><td>r6id.xlarge</td><td>c5ad.xlarge</td></tr><tr><td>磁盘大小（GB）</td><td>950</td><td>150</td><td>237</td><td>150</td></tr><tr><td>CPU数量</td><td>16</td><td>4</td><td>4</td><td>4</td></tr><tr><td>内存（GB）</td><td>128</td><td>16 (-88%)</td><td>32 (-75%)</td><td>8</td></tr><tr><td>成本（美元/月）</td><td>438</td><td>95 (-78%)</td><td>139 (-68%)</td><td>54</td></tr></tbody></table>

Table 2: Results on MS MARCO, Wikipedia (NQ-dev) and LoTTE (Lifestyle-dev) datasets. For SPLADEv2, we use the BT-SPLADE-L [14] checkpoint and a PISA index compressed with a block_simdbp encoding, following [23], and block of size 40 with a quantized scorer. For MS MARCO, we report development results on Dev,on which we tune $\alpha$ for all datasets,and report evaluation results on the held-out evaluation set used by the ColBERT authors [10 28].

表2：在MS MARCO、维基百科（NQ-dev）和LoTTE（生活方式开发集，Lifestyle-dev）数据集上的结果。对于SPLADEv2，我们使用BT - SPLADE - L [14]检查点和采用块相似性双字节编码（block_simdbp）压缩的PISA索引（遵循文献[23]），块大小为40，并使用量化评分器。对于MS MARCO，我们报告开发集（Dev）上的开发结果，在该开发集上我们对所有数据集调整$\alpha$，并报告由ColBERT作者使用的保留评估集上的评估结果[10 28]。

<table><tr><td>Method</td><td colspan="3">MS MARCO Dev</td><td colspan="3">MS MARCO5K Test</td></tr><tr><td/><td>MRR@10</td><td>R@5</td><td>R@50</td><td>MRR@10</td><td>R@5</td><td>R@50</td></tr><tr><td>ColBERTv2</td><td>39.51</td><td>56.62</td><td>86.30</td><td>40.57</td><td>57.78</td><td>86.14</td></tr><tr><td>SPLADEv2</td><td>38.00</td><td>54.70</td><td>85.04</td><td>38.62</td><td>54.92</td><td>84.84</td></tr><tr><td>Rerank</td><td>39.50</td><td>56.65</td><td>86.64</td><td>40.55</td><td>57.78</td><td>86.36</td></tr><tr><td>Hybrid $\left( {\alpha  = {0.3}}\right)$</td><td>40.22</td><td>57.38</td><td>86.98</td><td>41.11</td><td>58.23</td><td>86.91</td></tr></table>

<table><tbody><tr><td>方法</td><td colspan="3">MS MARCO开发集</td><td colspan="3">MS MARCO 5K测试集</td></tr><tr><td></td><td>前10名平均倒数排名（MRR@10）</td><td>R@5</td><td>R@50</td><td>前10名平均倒数排名（MRR@10）</td><td>R@5</td><td>R@50</td></tr><tr><td>ColBERTv2</td><td>39.51</td><td>56.62</td><td>86.30</td><td>40.57</td><td>57.78</td><td>86.14</td></tr><tr><td>SPLADEv2</td><td>38.00</td><td>54.70</td><td>85.04</td><td>38.62</td><td>54.92</td><td>84.84</td></tr><tr><td>重排序</td><td>39.50</td><td>56.65</td><td>86.64</td><td>40.55</td><td>57.78</td><td>86.36</td></tr><tr><td>混合 $\left( {\alpha  = {0.3}}\right)$</td><td>40.22</td><td>57.38</td><td>86.98</td><td>41.11</td><td>58.23</td><td>86.91</td></tr></tbody></table>

<table><tr><td>Method</td><td colspan="2">Wikipedia</td><td colspan="2">LoTTE</td></tr><tr><td/><td>S@5</td><td>$\Delta$</td><td>S@5</td><td>$\Delta$</td></tr><tr><td>ColBERTv2</td><td>67.51</td><td/><td>74.6</td><td/></tr><tr><td>SPLADEv2</td><td>59.60</td><td>-11.7%</td><td>70.7</td><td>-5.6%</td></tr><tr><td>Rerank</td><td>66.29</td><td>-1.8%</td><td>74.3</td><td>-0.4%</td></tr><tr><td>Hybrid $\left( {\alpha  = {0.3}}\right)$</td><td>65.78</td><td>-2.6%</td><td>74.8</td><td>+0.3%</td></tr><tr><td>Hybrid (optimal $\alpha$ )</td><td>66.34</td><td>-1.7%</td><td>75.3</td><td>+0.9%</td></tr></table>

<table><tbody><tr><td>方法</td><td colspan="2">维基百科</td><td colspan="2">乐天（LoTTE）</td></tr><tr><td></td><td>S@5</td><td>$\Delta$</td><td>S@5</td><td>$\Delta$</td></tr><tr><td>ColBERTv2（原文未变，可根据具体领域确定是否有特定译法）</td><td>67.51</td><td></td><td>74.6</td><td></td></tr><tr><td>SPLADEv2（原文未变，可根据具体领域确定是否有特定译法）</td><td>59.60</td><td>-11.7%</td><td>70.7</td><td>-5.6%</td></tr><tr><td>重排序</td><td>66.29</td><td>-1.8%</td><td>74.3</td><td>-0.4%</td></tr><tr><td>混合 $\left( {\alpha  = {0.3}}\right)$</td><td>65.78</td><td>-2.6%</td><td>74.8</td><td>+0.3%</td></tr><tr><td>混合（最优 $\alpha$ ）</td><td>66.34</td><td>-1.7%</td><td>75.3</td><td>+0.9%</td></tr></tbody></table>

<!-- Media -->

Retrieval Quality Table 2 reports the quality of full ColBERTv2 scoring against more efficient approaches based on SPLADEv2, Rerank, and Hybrid scoring. We tune the parameter $\alpha$ for Hybrid on MS MARCO Dev and report the results of this setting (i.e., $\alpha  = {0.3}$ ) across all datasets. We can observe that Hybrid scoring is the most effective method on MS MARCO and that it outperforms the SPLADEv2 model and Rerank across every dataset. On Wikipedia, however,using a non-optimal $\alpha$ results in lower performance than Rerank. This suggests that tuning $\alpha$ on a dedicated set of queries can be important to OOD settings, though we leave this exploration for future work. Having confirmed the quality of the Rerank and especially Hybrid methods, we now proceed to evaluate the different efficiency dimensions.

检索质量 表2报告了完整的ColBERTv2评分相对于基于SPLADEv2、重排（Rerank）和混合（Hybrid）评分的更高效方法的质量。我们在MS MARCO开发集上调整混合评分的参数$\alpha$，并报告此设置（即$\alpha  = {0.3}$）在所有数据集上的结果。我们可以观察到，混合评分在MS MARCO上是最有效的方法，并且在每个数据集上都优于SPLADEv2模型和重排方法。然而，在维基百科数据集上，使用非最优的$\alpha$会导致性能比重排方法更低。这表明，在一组专门的查询上调整$\alpha$对于分布外（OOD）设置可能很重要，不过我们将这一探索留待未来工作。在确认了重排方法尤其是混合方法的质量后，我们现在开始评估不同的效率维度。

<!-- Media -->

Table 3: Retrieval quality of Hybrid with different $\alpha$ . When $\alpha  = 0$ ,the method is equivalent to Rerank; when $\alpha  = 1$ ,it’s equivalent to SPLADEv2.

表3：不同$\alpha$下混合评分的检索质量。当$\alpha  = 0$时，该方法等同于重排方法；当$\alpha  = 1$时，等同于SPLADEv2方法。

<table><tr><td>$\alpha$</td><td>0.0</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td>1.0</td></tr><tr><td>Wiki S@5</td><td>66.29</td><td>66.34</td><td>66.21</td><td>65.78</td><td>65.33</td><td>64.76</td><td>63.80</td><td>63.01</td><td>62.12</td><td>61.04</td><td>59.60</td></tr><tr><td>LoTTE S@5</td><td>74.3</td><td>74.1</td><td>75.3</td><td>74.8</td><td>74.6</td><td>74.3</td><td>73.4</td><td>72.4</td><td>71.5</td><td>71.2</td><td>70.7</td></tr><tr><td>MARCOMRR@10</td><td>39.50</td><td>39.95</td><td>40.06</td><td>40.22</td><td>40.08</td><td>40.05</td><td>39.81</td><td>39.53</td><td>38.98</td><td>38.54</td><td>38.00</td></tr></table>

<table><tbody><tr><td>$\alpha$</td><td>0.0</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td>1.0</td></tr><tr><td>维基S@5</td><td>66.29</td><td>66.34</td><td>66.21</td><td>65.78</td><td>65.33</td><td>64.76</td><td>63.80</td><td>63.01</td><td>62.12</td><td>61.04</td><td>59.60</td></tr><tr><td>乐天S@5</td><td>74.3</td><td>74.1</td><td>75.3</td><td>74.8</td><td>74.6</td><td>74.3</td><td>73.4</td><td>72.4</td><td>71.5</td><td>71.2</td><td>70.7</td></tr><tr><td>马尔科MRR@10</td><td>39.50</td><td>39.95</td><td>40.06</td><td>40.22</td><td>40.08</td><td>40.05</td><td>39.81</td><td>39.53</td><td>38.98</td><td>38.54</td><td>38.00</td></tr></tbody></table>

<!-- Media -->

RAM Usage We measure memory usage for loading ColBERTv2 on MS MARCO and Wikipedia by recording the difference in RSS memory before and after loading. For the memory-mapped approaches, only the model checkpoint and index metadata are loaded into memory, resulting in a substantial reduction of RAM usage,by ${90}\%$ for MS MARCO (from 23.4 GB to 2.3 GB) and ${92}\%$ for Wikipedia (from 98.3 GB to 8.2 GB). Our approach allows us to host the indexes on machines with significantly lower RAM capacities, and reduces machine cost by 78% for MS MARCO and 68% for Wikipedia, as shown in Table 1,

内存使用情况 我们通过记录加载前后常驻集大小（RSS）内存的差异，来衡量在MS MARCO和维基百科数据集上加载ColBERTv2模型的内存使用情况。对于内存映射方法，只有模型检查点和索引元数据会被加载到内存中，从而显著降低了随机存取存储器（RAM）的使用量。在MS MARCO数据集上，内存使用量从23.4GB降至2.3GB，降幅为${90}\%$；在维基百科数据集上，内存使用量从98.3GB降至8.2GB，降幅为${92}\%$。我们的方法使我们能够在随机存取存储器容量显著较低的机器上托管索引，并且如表格1所示，在MS MARCO数据集上机器成本降低了78%，在维基百科数据集上降低了68%。

<!-- Media -->

<!-- figureText: ...表示 Full ColBERTv2 MMAP ColBERTv2 Full ColBERTv2 MMAP ColBERTv2 SPLADEv2 MMAP Rerank/Hybrid 20 / 3 latency (s) ${2}^{-2}$ .5 15.0 7.5 20.0 Average (b) MS MARCO (c) LoTTE SPLADEv2 SPLADEv2 - MMAP Rerank/Hybrid ${2}^{-2}$ (a) Wikipedia -->

<img src="https://cdn.noedgeai.com/0195afbf-03cb-77ee-ba8c-55864b08e8f0_5.jpg?x=386&y=345&w=1035&h=345&r=0"/>

Fig. 1: P95 latency for Wikipedia, MS MARCO, and LoTTE. Note that full ColBERTv2 on MS MARCO is evaluated on a higher-end and more expensive machine (refer to Table 1) with a different physical processor, so its latency is only for reference and is not directly comparable to the MMAP methods.

图1：维基百科、MS MARCO和LoTTE数据集的P95延迟。请注意，在MS MARCO数据集上对完整的ColBERTv2模型的评估是在一台更高端、更昂贵且配备不同物理处理器的机器上进行的（参见表格1），因此其延迟仅作参考，不能直接与内存映射方法进行比较。

<!-- Media -->

Latency on Varying Traffic Figure 1a compares the P95 latency across methods on Wikipedia. The optimized PISA implementation of SPLADEv2, using the efficiency-optimized BT-SPLADE-L model checkpoint [14], has the lowest latency, although this comes at the steep reduction in quality, especially out of domain, presented earlier. Next, although the Rerank/Hybrid methods incur higher latency than SPLADE, they are markedly faster than the memory-mapped ColBERTv2 method. The Rerank/Hybrid methods maintain low latency with QPS up to $1/{0.2} = 5$ queries per second. When QPS exceeds this,the system is saturated, leading to a sharper increase in latency due to queuing time. Note that as shown in Table 1, full ColBERTv2 experiments were conducted on a more expensive machine that fits the index in RAM, for reference. Despite this, the Rerank/Hybrid methods still achieve lower latency than full ColBERTv2 on QPS $< 1/{0.3} = {3.3}$ ,highlighting the value of multi-stage retrieval.

不同流量下的延迟 图1a比较了在维基百科数据集上不同方法的P95延迟。使用效率优化的BT - SPLADE - L模型检查点[14]对SPLADEv2进行优化后的PISA实现，具有最低的延迟，不过这是以质量的大幅下降为代价的，尤其是在域外数据上，这一点我们之前已经提到过。其次，尽管重排/混合（Rerank/Hybrid）方法的延迟比SPLADE方法高，但它们明显比内存映射的ColBERTv2方法快。重排/混合方法在每秒查询率（QPS）达到$1/{0.2} = 5$时仍能保持较低的延迟。当每秒查询率超过这个值时，系统会达到饱和状态，由于排队时间的增加，延迟会急剧上升。请注意，如表1所示，完整的ColBERTv2实验是在一台更昂贵且能将索引完全加载到随机存取存储器中的机器上进行的，仅供参考。尽管如此，在每秒查询率为$< 1/{0.3} = {3.3}$时，重排/混合方法的延迟仍然低于完整的ColBERTv2方法，这凸显了多阶段检索的价值。

Figure 1b shows similar trends on MS MARCO, where our Rerank/Hybrid systems greatly reduce the latency of memory-mapping ColBERTv2 across every traffic load. Note that full ColBERTv2 is evaluated on a machine with a different physical processor, so its latency is only for reference and is not directly comparable to the memory-mapped methods. Lastly, Figure 1c reports very similar patterns for for LoTTE. Note that we do not apply memory mapping for LoTTE, whose ColBERTv2 index fits easily in the RAM of our smallest machines. We also report mean latency and P99 latency as additional metrics in Figure 2, with similar trends as P95 latency.

图1b显示了在MS MARCO数据集上类似的趋势，我们的重排/混合系统在各种流量负载下都显著降低了内存映射ColBERTv2的延迟。请注意，完整的ColBERTv2是在一台配备不同物理处理器的机器上进行评估的，因此其延迟仅作参考，不能直接与内存映射方法进行比较。最后，图1c显示了LoTTE数据集上非常相似的模式。请注意，我们没有对LoTTE数据集应用内存映射，因为其ColBERTv2索引可以轻松地加载到我们最小型机器的随机存取存储器中。我们还在图2中报告了平均延迟和P99延迟作为额外的指标，其趋势与P95延迟相似。

<!-- Media -->

<!-- figureText: Full ColBERTv2 MMAP ColBERTv2 MMAP Rerank/Hybrid SPLADEv2 Mean latency (s) $\begin{array}{ll} {5.0} & 7 \end{array}$ Average incomi (queries/s) (b) Mean, MS MARCO (c) Mean, LoTTE MMAP ColBERTv2 ColBERTv2 Rerank/Hybrid MMAP Rerank/Hybrid SPLADEv2 ${2}^{4}$ P99 latency (s) ${2}^{2}$ ${2}^{-7}$ ${2}^{-4}$ Average (e) P99, MS MARCO (f) P99, LoTTE SPLADEv2 MMAP Rerank/Hybrid SPLADEv2 verage (a) Mean Latency, Wiki 110 Full ColBERTv2 MMAP ColBERTv2 Full ColBERTv2 SPLADEv2 SPLADEv2 MMAP Rerank/Hybrid ${2}^{7}$ 299 latency (s) ${2}^{5}$ ${2}^{1}$ ${2}^{-1}$ (d) P99 Latency, Wiki -->

<img src="https://cdn.noedgeai.com/0195afbf-03cb-77ee-ba8c-55864b08e8f0_6.jpg?x=386&y=337&w=1043&h=714&r=0"/>

Fig. 2: Mean Latency and P99 Latency on Three Datasets.

图2：三个数据集的平均延迟和P99延迟。

<!-- Media -->

## 5 Conclusion

## 5 结论

We presented a highly practical serving system for ColBERT models that combines memory-mapping, hybrid scoring, and support for concurrent requests. We introduced an evaluation methodology for assessing the neural IR tradeoffs in the concurrent, memory-constrained regime and demonstrated for the first time to our knowledge that a ColBERT serving system can serve several queries per second over large datasets on a server with as little as a few GBs of RAM. While we expect that serving multi-vector models will continue to become faster and cheaper in other ways, this work presents that a simple yet effective strategy to balance a large number of deployment tradeoffs.

我们提出了一个高度实用的ColBERT模型服务系统，该系统结合了内存映射、混合评分和对并发请求的支持。我们引入了一种评估方法，用于评估在并发、内存受限的情况下神经信息检索（IR）的权衡。据我们所知，我们首次证明了一个ColBERT服务系统可以在仅配备几GB随机存取存储器的服务器上，每秒处理大型数据集上的多个查询。虽然我们预计通过其他方式服务多向量模型将继续变得更快、更便宜，但这项工作提出了一种简单而有效的策略，用于平衡大量的部署权衡。

Acknowledgments. This work was partially supported by a Stanford HAI Hoffman-Yee Research Grant and by IBM as a founding member of the Stanford Institute for Human-Centered Artificial Intelligence (HAI), Oracle, Virtusa, and Cigna Healthcare. This research was supported in part by affiliate members and other supporters of the Stanford DAWN project-Facebook, Google, and VMware.

致谢。这项工作部分得到了斯坦福以人为本人工智能研究所（HAI）的霍夫曼 - 伊研究资助，以及IBM（斯坦福以人为本人工智能研究所的创始成员）、甲骨文（Oracle）、维鲁萨（Virtusa）和信诺医疗（Cigna Healthcare）的支持。这项研究还得到了斯坦福黎明（DAWN）项目的附属成员和其他支持者——脸书（Facebook）、谷歌（Google）和威睿（VMware）的部分支持。

## References

## 参考文献

1. Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X., Majumder, R., McNamara, A., Mitra, B., Nguyen, T., et al.: Ms marco: A human generated machine reading comprehension dataset. arXiv preprint arXiv:1611.09268 (2016)

1. 巴贾杰（Bajaj），P.；坎波斯（Campos），D.；克拉斯韦尔（Craswell），N.；邓（Deng），L.；高（Gao），J.；刘（Liu），X.；马朱姆德（Majumder），R.；麦克纳马拉（McNamara），A.；米特拉（Mitra），B.；阮（Nguyen），T.等：MS MARCO：一个人工生成的机器阅读理解数据集。预印本arXiv：1611.09268（2016）

2. Basnet, S., Gou, J., Mallia, A., Suel, T.: Deeperimpact: Optimizing sparse learned index structures. arXiv preprint arXiv:2405.17093 (2024)

2. 巴斯内特（Basnet），S.；苟（Gou），J.；马利亚（Mallia），A.；苏埃尔（Suel），T.：深度影响：优化稀疏学习索引结构。预印本arXiv：2405.17093（2024）

3. Bergum, J.K.: Improving zero-shot ranking with vespa hybrid search - part two (2023), https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa-part-two/

3. 伯古姆（Bergum），J.K.：使用Vespa混合搜索改进零样本排序 - 第二部分（2023），https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa-part-two/

4. Bernhardsson, E.: Spotify/annoy: Approximate nearest neighbors in c++/python optimized for memory usage and loading/saving to disk, https://github.com/ spotify/annoy

4. 伯恩哈德松（Bernhardsson），E.：Spotify/annoy：针对内存使用以及磁盘加载/保存进行优化的C++/Python近似最近邻算法，https://github.com/ spotify/annoy

5. Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., Colombo, P.: Colpali: Efficient document retrieval with vision language models (2024), https: //arxiv.org/abs/2407.01449

5. 费塞（Faysse），M.，西比尔（Sibille），H.，吴（Wu），T.，奥姆拉尼（Omrani），B.，维奥（Viaud），G.，于德洛（Hudelot），C.，科伦坡（Colombo），P.：Colpali：利用视觉语言模型进行高效文档检索（2024），https: //arxiv.org/abs/2407.01449

6. Formal, T., Clinchant, S., Déjean, H., Lassance, C.: Splate: Sparse late interaction retrieval. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 2635-2640. SIGIR '24, Association for Computing Machinery, New York, NY, USA (2024). https://doi.org/10.1145/3626772.3657968 https://doi.org/10.1145/3626772.3657968

6. 福尔马尔（Formal），T.，克兰尚（Clinchant），S.，德让（Déjean），H.，拉桑斯（Lassance），C.：Splate：稀疏延迟交互检索。见：第47届ACM信息检索研究与发展国际会议论文集。第2635 - 2640页。SIGIR '24，美国计算机协会，美国纽约州纽约市（2024）。https://doi.org/10.1145/3626772.3657968 https://doi.org/10.1145/3626772.3657968

7. Formal, T., Lassance, C., Piwowarski, B., Clinchant, S.: SPLADE v2: Sparse lexical and expansion model for information retrieval. CoRR abs/2109.10086 (2021), https://arxiv.org/abs/2109.10086

7. 福尔马尔（Formal），T.，拉桑斯（Lassance），C.，皮沃瓦尔斯基（Piwowarski），B.，克兰尚（Clinchant），S.：SPLADE v2：用于信息检索的稀疏词汇与扩展模型。计算机研究存储库（CoRR）abs/2109.10086（2021），https://arxiv.org/abs/2109.10086

8. Hofstätter, S., Khattab, O., Althammer, S., Sertkan, M., Hanbury, A.: Introducing neural bag of whole-words with colberter: Contextualized late interactions using enhanced reduction. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. p. 737-747. CIKM '22, Association for Computing Machinery, New York, NY, USA (2022). https://doi.org/10.1145/ 3511808.3557367, https://doi.org/10.1145/3511808.3557367

8. 霍夫施泰特（Hofstätter），S.，哈塔布（Khattab），O.，阿尔塔默（Althammer），S.，塞尔特坎（Sertkan），M.，汉伯里（Hanbury），A.：通过ColBERTer引入全词神经词袋：使用增强约简的上下文延迟交互。见：第31届ACM信息与知识管理国际会议论文集。第737 - 747页。CIKM '22，美国计算机协会，美国纽约州纽约市（2022）。https://doi.org/10.1145/ 3511808.3557367，https://doi.org/10.1145/3511808.3557367

9. Johnson, J., Douze, M., Jégou, H.: Billion-scale similarity search with gpus. IEEE Transactions on Big Data 7(3), 535-547 (2019)

9. 约翰逊（Johnson），J.，杜泽（Douze），M.，热古（Jégou），H.：使用GPU进行十亿级相似度搜索。《电气与电子工程师协会大数据汇刊》（IEEE Transactions on Big Data）7(3)，535 - 547（2019）

10. Khattab, O., Zaharia, M.: Colbert: Efficient and effective passage search via con-textualized late interaction over bert. In: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 39-48. SIGIR '20, Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3397271.3401075.https://doi.org/10.1145/ 3397271.3401075

10. 哈塔布（Khattab），O.，扎哈里亚（Zaharia），M.：ColBERT：通过基于BERT的上下文延迟交互实现高效有效的段落搜索。见：第43届ACM信息检索研究与发展国际会议论文集。第39 - 48页。SIGIR '20，美国计算机协会，美国纽约州纽约市（2020）。https://doi.org/10.1145/3397271.3401075.https://doi.org/10.1145/ 3397271.3401075

11. Kotek, J.: Jankotek/mapdb: Mapdb provides concurrent maps, sets and queues backed by disk storage or off-heap-memory. it is a fast and easy to use embedded java database engine., https://github.com/jankotek/mapdb/

11. 科泰克（Kotek），J.：Jankotek/mapdb：Mapdb提供由磁盘存储或堆外内存支持的并发映射、集合和队列。它是一个快速且易于使用的嵌入式Java数据库引擎。，https://github.com/jankotek/mapdb/

12. Kulkarni, H., MacAvaney, S., Goharian, N., Frieder, O.: Lexically-accelerated dense retrieval. In: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 152-162. SIGIR '23, Association for Computing Machinery, New York, NY, USA (2023). https: $//\mathrm{{doi}.{org}}/{10.1145}/{3539618.3591715},\mathrm{{https}} : //\mathrm{{doi}.{org}}/{10.1145}/{3539618.3591715}$

12. 库尔卡尼（Kulkarni），H.，麦卡瓦尼（MacAvaney），S.，戈哈里安（Goharian），N.，弗里德（Frieder），O.：词汇加速的密集检索。见：第46届ACM信息检索研究与发展国际会议论文集。第152 - 162页。SIGIR '23，美国计算机协会，美国纽约州纽约市（2023）。https: $//\mathrm{{doi}.{org}}/{10.1145}/{3539618.3591715},\mathrm{{https}} : //\mathrm{{doi}.{org}}/{10.1145}/{3539618.3591715}$

13. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M.W., Dai, A.M., Uszkoreit, J., Le, Q., Petrov, S.: Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics 7, 453-466 (08 2019). https://doi.org/10.1162/tacl_ a 00276 https://doi.org/10.1162/tacl_a_00276

13. 夸特科夫斯基（Kwiatkowski），T.，帕洛马基（Palomaki），J.，雷德菲尔德（Redfield），O.，柯林斯（Collins），M.，帕里克（Parikh），A.，阿尔贝蒂（Alberti），C.，爱泼斯坦（Epstein），D.，波洛苏金（Polosukhin），I.，德夫林（Devlin），J.，李（Lee），K.，图托纳娃（Toutanova），K.，琼斯（Jones），L.，凯尔西（Kelcey），M.，张（Chang），M.W.，戴（Dai），A.M.，乌兹科赖特（Uszkoreit），J.，勒（Le），Q.，彼得罗夫（Petrov），S.：自然问题：问答研究基准。《计算语言学协会汇刊》（Transactions of the Association for Computational Linguistics）7，453 - 466（2019年8月）。https://doi.org/10.1162/tacl_ a 00276 https://doi.org/10.1162/tacl_a_00276

14. Lassance, C., Clinchant, S.: An efficiency study for splade models. In: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 2220-2226. SIGIR '22, Association for Computing Machinery, New York, NY, USA (2022). https://doi.org/10.1145/3477495.3531833.https://doi.org/10.1145/3477495.3531833

14. 拉桑斯（Lassance），C.；克林尚（Clinchant），S.：关于SPLADE模型的效率研究。见：第45届ACM国际信息检索研究与发展会议论文集。第2220 - 2226页。SIGIR '22，美国计算机协会，美国纽约州纽约市（2022年）。https://doi.org/10.1145/3477495.3531833.https://doi.org/10.1145/3477495.3531833

15. Lee, J., Dai, Z., Duddu, S.M.K., Lei, T., Naim, I., Chang, M.W., Zhao, V.: Rethinking the role of token retrieval in multi-vector retrieval. Advances in Neural Information Processing Systems 36 (2024)

15. 李（Lee），J.；戴（Dai），Z.；杜杜（Duddu），S.M.K.；雷（Lei），T.；奈姆（Naim），I.；张（Chang），M.W.；赵（Zhao），V.：重新思考多向量检索中词元检索的作用。《神经信息处理系统进展》36（2024年）

16. Lee, K., Chang, M.W., Toutanova, K.: Latent retrieval for weakly supervised open domain question answering. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics (2019)

16. 李（Lee），K.；张（Chang），M.W.；图托纳娃（Toutanova），K.：弱监督开放域问答的潜在检索。见：第57届计算语言学协会年会论文集。计算语言学协会（2019年）

17. Li, M., Lin, S.C., Oguz, B., Ghoshal, A., Lin, J., Mehdad, Y., Yih, W.t., Chen, X.: Citadel: Conditional token interaction via dynamic lexical routing for efficient and effective multi-vector retrieval. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 11891-11907 (2023)

17. 李（Li），M.；林（Lin），S.C.；奥古兹（Oguz），B.；戈沙尔（Ghoshal），A.；林（Lin），J.；梅赫达德（Mehdad），Y.；易（Yih），W.t.；陈（Chen），X.：城堡（Citadel）：通过动态词法路由实现条件词元交互以进行高效有效的多向量检索。见：第61届计算语言学协会年会论文集（第1卷：长论文）。第11891 - 11907页（2023年）

18. Lin, J., Ma, X.: A few brief notes on deepimpact, coil, and a conceptual framework for information retrieval techniques. arXiv preprint arXiv:2106.14807 (2021)

18. 林（Lin），J.；马（Ma），X.：关于深度影响（DeepImpact）、线圈（Coil）以及信息检索技术概念框架的几点简要说明。预印本arXiv:2106.14807（2021年）

19. MacAvaney, S., Tonellotto, N.: A reproducibility study of plaid. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 1411-1419. SIGIR '24, Association for Computing Machinery, New York, NY, USA (2024). https://doi.org/10.1145/3626772.3657856.https://doi.org/10.1145/3626772.3657856

19. 麦卡瓦尼（MacAvaney），S.；托内洛托（Tonellotto），N.：格子（Plaid）的可重复性研究。见：第47届ACM国际信息检索研究与发展会议论文集。第1411 - 1419页。SIGIR '24，美国计算机协会，美国纽约州纽约市（2024年）。https://doi.org/10.1145/3626772.3657856.https://doi.org/10.1145/3626772.3657856

20. Mackenzie, J., Mallia, A., Moffat, A., Petri, M.: Accelerating learned sparse indexes via term impact decomposition. In: Findings of the Association for Computational Linguistics: EMNLP 2022. pp. 2830-2842 (2022)

20. 麦肯齐（Mackenzie），J.；马利亚（Mallia），A.；莫法特（Moffat），A.；佩特里（Petri），M.：通过词项影响分解加速学习型稀疏索引。见：计算语言学协会研究成果：EMNLP 2022。第2830 - 2842页（2022年）

21. Mallia, A., Khattab, O., Suel, T., Tonellotto, N.: Learning passage impacts for inverted indexes. In: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 1723-1727 (2021)

21. 马利亚（Mallia），A.；哈塔布（Khattab），O.；苏埃尔（Suel），T.；托内洛托（Tonellotto），N.：为倒排索引学习段落影响。见：第44届ACM国际信息检索研究与发展会议论文集。第1723 - 1727页（2021年）

22. Mallia, A., Siedlaczek, M., Mackenzie, J., Suel, T.: PISA: performant indexes and search for academia. In: Proceedings of the Open-Source IR Replicability Challenge co-located with 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, OSIRRC@SIGIR 2019, Paris, France, July 25, 2019. pp. 50-56 (2019), http://ceur-ws.org/Vol-2409/docker08.pdf

22. 马利亚（Mallia），A.；西德拉塞克（Siedlaczek），M.；麦肯齐（Mackenzie），J.；苏埃尔（Suel），T.：PISA：面向学术界的高性能索引与搜索。见：与第42届ACM国际信息检索研究与发展会议同期举办的开源信息检索可重复性挑战赛论文集，OSIRRC@SIGIR 2019，法国巴黎，2019年7月25日。第50 - 56页（2019年），http://ceur - ws.org/Vol - 2409/docker08.pdf

23. Mallia, A., Siedlaczek, M., Suel, T.: An experimental study of index compression and daat query processing methods. In: Advances in Information Retrieval: 41st European Conference on IR Research, ECIR 2019, Cologne, Germany, April 14-18, 2019, Proceedings, Part I 41. pp. 353-368. Springer (2019)

23. 马利亚（Mallia），A.；西德拉塞克（Siedlaczek），M.；苏埃尔（Suel），T.：索引压缩和按需查询处理方法的实验研究。见：《信息检索进展：第41届欧洲信息检索研究会议，ECIR 2019，德国科隆，2019年4月14 - 18日，会议论文集，第一部分41》。第353 - 368页。施普林格出版社（2019年）

24. Mallia, A., Suel, T., Tonellotto, N.: Faster learned sparse retrieval with block-max pruning. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 2411-2415 (2024)

24. 马利亚（Mallia），A.；苏埃尔（Suel），T.；托内洛托（Tonellotto），N.：通过块最大剪枝实现更快的学习型稀疏检索。见：第47届ACM国际信息检索研究与发展会议论文集。第2411 - 2415页（2024年）

25. Nardini, F.M., Rulli, C., Venturini, R.: Efficient multi-vector dense retrieval with bit vectors. In: European Conference on Information Retrieval. pp. 3-17. Springer (2024)

25. 纳尔迪尼（Nardini），F.M.；鲁利（Rulli），C.；文图里尼（Venturini），R.：使用位向量进行高效的多向量密集检索。见：欧洲信息检索会议。第3 - 17页。施普林格出版社（2024年）

26. Robertson, S., Walker, S., Jones, S., Hancock-Beaulieu, M.M., Gatford, M.: Okapi at trec-3. In: Overview of the Third Text REtrieval Conference (TREC-3). pp. 109-126. Gaithersburg, MD: NIST (January 1995), https://www.microsoft.com/ en-us/research/publication/okapi-at-trec-3/

26. 罗伯逊（Robertson），S.；沃克（Walker），S.；琼斯（Jones），S.；汉考克 - 博利厄（Hancock - Beaulieu），M.M.；加特福德（Gatford），M.：奥卡皮（Okapi）在第三届文本检索会议（TREC - 3）上。见：《第三届文本检索会议（TREC - 3）综述》。第109 - 126页。美国马里兰州盖瑟斯堡：美国国家标准与技术研究院（1995年1月），https://www.microsoft.com/ en - us/research/publication/okapi - at - trec - 3/

27. Santhanam, K., Khattab, O., Potts, C., Zaharia, M.: Plaid: an efficient engine for late interaction retrieval. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. pp. 1747-1756 (2022)

27. 桑塔纳姆（Santhanam），K.；卡塔布（Khattab），O.；波茨（Potts），C.；扎哈里亚（Zaharia），M.：Plaid：一种用于后期交互检索的高效引擎。见：第31届ACM国际信息与知识管理会议论文集。第1747 - 1756页（2022年）

28. Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., Zaharia, M.: ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In: Carpuat, M., de Marneffe, M.C., Meza Ruiz, I.V. (eds.) Proceedings of the 2022 Conference of the

28. 桑塔纳姆（Santhanam），K.；卡塔布（Khattab），O.；萨德 - 法尔孔（Saad - Falcon），J.；波茨（Potts），C.；扎哈里亚（Zaharia），M.：ColBERTv2：通过轻量级后期交互实现高效检索。见：卡尔普亚特（Carpuat），M.；德马尔内夫（de Marneffe），M.C.；梅萨·鲁伊斯（Meza Ruiz），I.V.（编）2022年北美计算语言学协会人类语言技术会议论文集

North American Chapter of the Association for Computational Linguistics: Human Language Technologies. pp. 3715-3734. Association for Computational Linguistics, Seattle, United States (Jul 2022). https://doi.org/10.18653/v1/2022.naacl-main.272, https://aclanthology.org/2022.naacl-main.272

第3715 - 3734页。计算语言学协会，美国西雅图（2022年7月）。https://doi.org/10.18653/v1/2022.naacl - main.272，https://aclanthology.org/2022.naacl - main.272

29. Santhanam, K., Saad-Falcon, J., Franz, M., Khattab, O., Sil, A., Florian, R., Sultan, M.A., Roukos, S., Zaharia, M., Potts, C.: Moving beyond downstream task accuracy for information retrieval benchmarking. In: Rogers, A., Boyd-Graber, J., Okazaki, N. (eds.) Findings of the Association for Computational Linguistics: ACL 2023. pp. 11613-11628. Association for Computational Linguistics, Toronto, Canada (Jul 2023). https://doi.org/10.18653/v1/2023.findings-acl.738, https://aclanthology.org/2023.findings-acl.738

29. 桑塔纳姆（Santhanam），K.；萨德 - 法尔孔（Saad - Falcon），J.；弗朗茨（Franz），M.；卡塔布（Khattab），O.；西尔（Sil），A.；弗洛里安（Florian），R.；苏丹（Sultan），M.A.；鲁科斯（Roukos），S.；扎哈里亚（Zaharia），M.；波茨（Potts），C.：超越下游任务准确率进行信息检索基准测试。见：罗杰斯（Rogers），A.；博伊德 - 格拉伯（Boyd - Graber），J.；冈崎（Okazaki），N.（编）计算语言学协会研究成果：ACL 2023。第11613 - 11628页。计算语言学协会，加拿大多伦多（2023年7月）。https://doi.org/10.18653/v1/2023.findings - acl.738，https://aclanthology.org/2023.findings - acl.738

30. Shrestha, S., Reddy, N., Li, Z.: Espn: Memory-efficient multi-vector information retrieval. In: Proceedings of the 2024 ACM SIGPLAN International Symposium on Memory Management. p. 95-107. ISMM 2024, Association for Computing Machinery, New York, NY, USA (2024). https://doi.org/10.1145/3652024.3665515.https://doi.org/10.1145/3652024.3665515

30. 什雷斯塔（Shrestha），S.；雷迪（Reddy），N.；李（Li），Z.：Espn：内存高效的多向量信息检索。见：2024年ACM SIGPLAN国际内存管理研讨会论文集。第95 - 107页。ISMM 2024，美国计算机协会，美国纽约州纽约市（2024年）。https://doi.org/10.1145/3652024.3665515.https://doi.org/10.1145/3652024.3665515

31. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I.: Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663 (2021)

31. 塔库尔（Thakur），N.；赖默斯（Reimers），N.；吕克莱（Rücklé），A.；斯里瓦斯塔瓦（Srivastava），A.；古雷维奇（Gurevych），I.：Beir：一个用于信息检索模型零样本评估的异构基准。预印本arXiv：2104.08663（2021年）

32. Yu, P., Mallia, A., Petri, M.: Improved learned sparse retrieval with corpus-specific vocabularies. In: European Conference on Information Retrieval. pp. 181-194. Springer (2024)

32. 于（Yu），P.；马利亚（Mallia），A.；彼得里（Petri），M.：利用特定语料库词汇改进学习型稀疏检索。见：欧洲信息检索会议。第181 - 194页。施普林格出版社（2024年）

33. Zhuang, S., Zuccon, G.: Fast passage re-ranking with contextualized exact term matching and efficient passage expansion. arXiv preprint arXiv:2108.08513 (2021)

33. 庄（Zhuang），S.；祖克康（Zuccon），G.：基于上下文精确词匹配和高效段落扩展的快速段落重排序。预印本arXiv：2108.08513（2021年）