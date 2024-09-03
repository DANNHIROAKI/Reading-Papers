## :wheel_of_dharma:Keynotes  

#### :point_right:==**The Limitations of Data, Machine Learning and Us**==

:classical_building:机构：智利大学

:arrow_right:领域：

- Social and professional topics → Computing / technology policy
- Computing methodologies → Machine learning
- Information systems → Data management systems

:books:摘要：

- 讨论了以下主题
  - 监督学习/输入监督学习的数据的局限
  - 人类适用机器学习时的社会/认知偏见
  - 人工智能使用的监管措施



#### :point_right:==**The Journey to A Knowledgeable Assistant with Retrieval-Augmented Generation (RAG)  **==  

:classical_building:机构：Facebook (Meta)

:books:摘要

- 背景：
  - 多个研究领域(DB/NLP/AI)都致力于在一定时间提供正确信息
  - 近年LLM提出，但也可能会输出错误/虚假信息
- 本文工作：
  - 通过实验，评估LLMs在回答**事实性问题**的可靠性
  - 构建Retrieval-Augmented Generation(RAG)联邦系统，整合LLM训练集以外知识，提高回答可靠性
  - 将RAG用到多模态/不同文化/个性化回答



#### :point_right:**==Making Data Management Better with Vectorized Query Processing==** 

:books:摘要：

-  主要回顾/展望了矢量化查询
-  矢量化查询是啥
   - 传统的查询：逐行处理(tuple-at-a-time)
   - 矢量化查询：每次处理一批固定大小的数据(称之为Vector)，可实现CPU优化/缓存友好等



## :wheel_of_dharma:Industry Session 1: Query Engines   

#### **:point_right:==Apache Arrow DataFusion: A Fast, Embeddable, Modular Analytic Query Engine==**   

:classical_building:机构：Apache 

:books:摘要：

- 介绍了Apache Arrow DataFusion：一个基于Apache Arrow的查询引擎，强调快速/可嵌入/可扩展
  - Apache Arrow：跨平台数据处理工具，提供高效的内存模型
  - DataFusion：用Rust编写，具有性能+安全性的优势



#### **:point_right:==Unified Query Optimization in the Fabric Data Warehouse==** 

:classical_building:机构：Microsoft

:arrow_right:领域：Information systems → Query optimization  

:books:摘要：

- 背景：微软曾推出了Parallel Data Warehouse，是一种查询大量数据的并行系统
- 本文：介绍了微软最新提出的Fabric DW
  - 文章对比了Fabric DW与传统的Parallel Data Warehouse
  - 新的优化器考虑了现代环境中的需求，如动态资源分配/计算存储分离等



#### **:point_right:==Measures in SQL==  ** 

:classical_building:机构：Google  

:arrow_right:领域：

- Information systems → Relational database query languages  
- Data analytics  
- Online analytical processing  

:books:摘要：

- 背景：SQL已被广泛采用，但传统的SQL任然缺乏*可组合计算*的能力
- 本文：提出一种新型的附加列，叫做Measure(度量)
  - 如何操作带度量的表：和普通表操作方法一样
  - 带度量的SQL的优势：可在保留SQL语义同时，通过调用Measure解决更复杂的查询
  - 度量如何计算得到：通过上下文(上下文敏感表达式)得到度量的值



#### **:point_right:==ByteCard: Enhancing ByteDance’s Data Warehouse with Learned Cardinality Estimation==** 

:classical_building:机构：ByteDance  

:arrow_right:领域：

- Information systems → Data management systems  
- Computing methodologies → Machine learning  

:books:摘要：​​

- 背景：
  - 关于ByteHouse：字节公司开发的云原生数据分析引擎，用于处理超大规模数据的复杂分析任务
  - 关于基数估计：预测查询结果的数量(大小)，直接影响优化器的决策，是有护额的瓶颈所在
- ByteCard的引入：融合最近在基数估计方面的进展，构建了兼顾可靠/实用的基数估计模型



#### **:point_right:==Automated Multidimensional Data Layouts in Amazon Redshift==** 

:classical_building:机构：Amazon

:arrow_right:领域：

- Information systems → Data layout
- Autonomous database administration
- Online analytical processing engines  

:books:摘要：

- 背景：关于数据布局技术，其是DB/DW中优化存储和访问效率的策略，常见为以下几种

  |  种类  | 概述                     | 示例(T=Tuple/A=Attribute) |
  | :----: | ------------------------ | -------------- |
  | 行存储 | 一行数据所有字段连续存储 | T1/A1→T1/A2→....→T1/An→T2/A1→....→Tm/An |
  | 列存储 | 一列数据所有字段连续存储 | T1/A1→T2/A1→....→Tm/A1→T1/A2→....→Tm/An |
  | 排序键 | 数据按Key(单一/复合)排序后存储 | N/A |
  | [索引](https://blog.csdn.net/qq_64091900/article/details/141219405) | 建立数据在表中$\xleftrightarrow{}$内存中位置的索引 | B+树，哈希表 |
  
- 本文的工作1：提出了多维数据布局(MDDL)

  - 核心方法：传统方法是基于一组列对表进行排序，MDDL是==基于一组谓词(查询条件)对表进行排序==
  - 优点：是的查询高度的定制化

- 本文的工作2：提出一种自动化学习算法，基于历史工作负载，==自动学习每个表最佳的MDDL== 



#### **:point_right:==Automated Clustering Recommendation With Database Zone Maps==**  

:classical_building:机构：Oracle  

:arrow_right:领域：Theory of computation → Database query processing and optimization (theory)

:books:摘要：一言蔽之，主要讲了区间图/自动聚类在数据仓库中的应用

- 背景：关于区间图(Zone Maps)
  - 结构：将表划分为Zone，存储每个区域的最大/最小值
  - 工作原理：支持查询时，读取区间的最大/最小值，选择跳过/不跳过该区间，从而减少扫描工作量
  - 优势：**在按某列排序/聚类处理后的数据上表现优越** 
- 本文的工作：自动分析工作负载→推荐聚类方案(线性聚类和z-order聚类)→建区间图→提高查询性能



## :wheel_of_dharma:Industry Session 2: LLMs and ML Applications  



#### **:point_right:==Similarity Joins of Sparse Features==**

:classical_building:机构：Uber

:arrow_right:领域：

- Information systems → Clustering  
- Theory of computation → MapReduce algorithms  

:books:摘要：提出了Fast Scalable Sparse Joiner (FSSJ)算法，用于在大规模系数数据尚进行相似性连接

- 背景
  - 相似性连接：在两个数据集中，找出相似性超过某个阈值的记录对
  - 稀疏特征：平均每个Tuple只有少数Attributes被赋值
- 关于FSSJ：引入Quasi-Prefix Filtering的新方法



## :wheel_of_dharma:Industry Session 3: Cloud Storage  



## :wheel_of_dharma:Indusrty Session 4: Cloud Databases  



## :wheel_of_dharma:Industry Session 5: Cloud Database Architecture  



## :wheel_of_dharma:Industry Session 6: Graph Data Management  



## :wheel_of_dharma:Demonstrations Group A  



## :wheel_of_dharma:Demonstrations Group B  



## :wheel_of_dharma:Panels  



## :wheel_of_dharma:Tutorials  



## :wheel_of_dharma:Workshop Summaries  
