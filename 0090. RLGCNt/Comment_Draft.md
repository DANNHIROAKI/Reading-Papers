## 1. Paper Significance  

- Above Average  

## 2. Paper Clarity  

- Above Average  

## 3. Summary of the paper  

This paper proposes a novel cardinality estimation method called RLGCNt. The main workflow of the model is as follows:

**Query Encoding**: The Rank Gauss Transform (Equations 1-3) is used to encode query conditions such as SJCB, SARC, and BRTL. Through a three-stage transformation process—ranking → scaling → inverse CDF mapping—non-uniformly distributed features are converted into a standard normal distribution, effectively mitigating distribution shift issues in dynamic workload scenarios. Histogram statistics are used to generate K/V features, which are concatenated with the encoded Q features to form a joint feature vector tz

**Locality-Sensitive Hashing (LSH)**: Learnable LSH (Equations 4-7) maps tz into hash buckets, ensuring that similar queries fall into the same bucket, thereby reducing subsequent computational overhead.

**Gaussian Kernel Similarity Calculation**: Within each hash bucket, the Euclidean distance of tz is computed (Equations 10-11), and a Gaussian kernel function (Equation 12) generates a similarity matrix.

**Causal Attention Mechanism**: The similarity matrix is used to generate Q/K/V vectors. An upper triangular causal mask (Equation 13) eliminates interference from predicate sequence positions on attention weights. Attention weights adjusted by the mask (Equation 14) enable temporal-order-independent feature aggregation, dynamically updating similarity representations.

**Adaptive Regression**: The output of the attention mechanism is processed by an MLP layer. The Reg module (a stack of multiple MLPs) generates the final cardinality estimate. Reg is integrated into the loss function (Equations 15-16), using MSE and regularization terms for backpropagation to adaptively optimize attention weights.

In experimental results, RLGCNt significantly reduces Q-error under both static and dynamic workloads (with a 10.7% improvement in dynamic scenarios) and achieves the shortest end-to-end runtime (only 2,623 seconds in dynamic scenarios). Ablation studies validate the synergistic effects of LSH, Gaussian kernels, and causal attention. Comparative experiments show its superiority over traditional methods (e.g., PG, Uni-Samp) and mainstream models (e.g., MSCN, ALECE). Its performance advantages stem from normal distribution encoding and dynamic weight optimization.

## 4. Strong points of the paper  

**S1  Outlier Robustness**: By leveraging Gaussian transformation encoding, non-uniform query features are forcibly converted into a normal distribution, significantly reducing the interference of outliers (99th percentile Q-error decreased by 29%).

**S2  Local Feature Capture**: The introduction of Gaussian kernel functions combined with LSH hash buckets enables a "coarse-filtering → fine-calculation" synergy, enhancing the ability to capture local similarity in non-uniform data (ablation studies show an 18.4% reduction in 95th percentile error).

**S3  Permutation-Invariance Optimization**: A causal attention mask is designed to eliminate the influence of predicate order on attention weights, improving the model's generalization capability for unordered queries (99th percentile error under dynamic workloads reduced by 57.2%).

**S4  Dynamic Workload Adaptability**: Through dynamic encoding and adaptive weight mechanisms, the model maintains stability under data distribution drift (accuracy improved by 10.7% in dynamic workloads).

**S5  Efficiency-Accuracy Balance**: LSH hash buckets reduce over 90% of computational redundancy, achieving end-to-end runtime of just 2,623 seconds—2-10x faster than mainstream methods—meeting industrial-grade real-time requirements while maintaining high precision.

## 5. Weak points of the paper

**W1  Research Motivation**: The paper does not fully explain why the proposed model is based on Transformer. It's noteworthy that recent studies have shown GNN models outperform Transformer-based models under instance-specific training, as referenced in [arXiv:2408.16170](https://doi.org/10.48550/arXiv.2408.16170).

**W2  Theoretical Foundation**: It is recommended to include an analysis of the model's time and space complexity. For example, while the original text vaguely claims that LSH can accelerate the computation of attention weights, it fails to further analyze whether this reduces the time complexity from $O(N^2)$ to $O(N\log N)$. Utilizing LSH to improve the attention mechanism is a common practice, and for such analysis, one could refer to the Reformer model ([arXiv:2001.04451](https://doi.org/10.48550/arXiv.2001.04451)).

**W3  Innovation**: There seems to be a lack of novelty, given that components of RLGCNt (such as Rank Gauss Transform, LSH, Gaussian kernel functions, and causal attention) are mostly combinations of existing technologies. The paper does not clearly articulate how the integration of these components results in a qualitative leap.

**W4  Experimental Persuasiveness**: The experimental section lacks sufficient persuasiveness. It could include more of the most recent state-of-the-art cardinality estimation models, such as those mentioned in [DOI:10.1007/s00778-023-00808-x](https://doi.org/10.1007/s00778-023-00808-x), [DOI:10.1145/3639300](https://doi.org/10.1145/3639300), and [DOI:10.1145/3639309](https://doi.org/10.1145/3639309). In terms of workload, it is suggested that future work should further improve and evaluate the scalability of the model (e.g., handling data at the billion level).

## 6. Detailed comments

Some high-level suggestions were provided. Here, we will discuss some specific details.

**D1** The overall quality of the figures in the document is suboptimal. For instance, in Fig. 1, "Cardinality estimation based on RLGCNt," the lower-left part appears unclear. It would be helpful to clarify in the main text whether the symbol Q originates from the query conditions encoded through the rank Gaussian transformation. Similarly, Fig. 2, "The LSH of RLGCNt," also seems somewhat confusing, as the explanation of the LSH process is vague. In general, it is recommended to redraw these figures for better clarity.

**D2** The writing in the section 4 needs to be more explicit and clear. Specifically, it would be beneficial to clearly specify the input content and shape/output content and shape/purpose of each component of RLGCNt. For example, what is the shape of the `hash_value` tensor in Equation (5)? Furthermore, after concatenating the hash values along `axis=-1` in Equation (6), what is the resulting shape?

## 7. Overall Rating

## 8. Review confidence  

