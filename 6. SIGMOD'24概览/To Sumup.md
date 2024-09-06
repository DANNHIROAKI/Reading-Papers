# 1. ​Keynotes  

> :point_down:==**The Limitations of Data, Machine Learning and Us**==
>
> > :classical_building:机构：智利大学
> >
> > :arrow_right:领域：
> >
> > - Social and professional topics → Computing / technology policy
> > - Computing methodologies → Machine learning
> > - Information systems → Data management systems
> >
> > :books:摘要：
> >
> > - 讨论了以下主题
> >   - 监督学习/输入监督学习的数据的局限
> >   - 人类适用机器学习时的社会/认知偏见
> >   - 人工智能使用的监管措施
>
> ##### :point_down:==**The Journey to A Knowledgeable Assistant with Retrieval-Augmented Generation (RAG)  **==  
>
> > :classical_building:机构：Facebook (Meta)
> >
> > :books:摘要
> >
> > - 背景：
> >   - 多个研究领域(DB/NLP/AI)都致力于在一定时间提供正确信息
> >   - 近年LLM提出，但也可能会输出错误/虚假信息
> > - 本文工作：
> >   - 通过实验，评估LLMs在回答**事实性问题**的可靠性
> >   - 构建Retrieval-Augmented Generation(RAG)联邦系统，整合LLM训练集以外知识，提高回答可靠性
> >   - 将RAG用到多模态/不同文化/个性化回答
>
> :point_down:**==Making Data Management Better with Vectorized Query Processing==** 
>
> > :books:摘要：
> >
> > -  主要回顾/展望了矢量化查询
> > -  矢量化查询是啥
> >    - 传统的查询：逐行处理(tuple-at-a-time)
> >    - 矢量化查询：每次处理一批固定大小的数据(称之为Vector)，可实现CPU优化/缓存友好等

# 2. Industry Session

> ## 2.1. Query Engines
>
> > **:point_down:==Apache Arrow DataFusion: A Fast, Embeddable, Modular Analytic Query Engine==**   
> >
> > > :classical_building:机构：Apache 
> > >
> > > :books:摘要：
> > >
> > > - 介绍了Apache Arrow DataFusion：一个基于Apache Arrow的查询引擎，强调快速/可嵌入/可扩展
> > >   - Apache Arrow：跨平台数据处理工具，提供高效的内存模型
> > >   - DataFusion：用Rust编写，具有性能+安全性的优势
> >
> > **:point_down:==Unified Query Optimization in the Fabric Data Warehouse==** 
> >
> > > :classical_building:机构：微软
> > >
> > > :arrow_right:领域：Information systems → Query optimization  
> > >
> > > :books:摘要：
> > >
> > > - 背景：微软曾推出了Parallel Data Warehouse，是一种查询大量数据的并行系统
> > > - 本文：介绍了微软最新提出的Fabric DW
> > >   - 文章对比了Fabric DW与传统的Parallel Data Warehouse
> > >   - 新的优化器考虑了现代环境中的需求，如动态资源分配/计算存储分离等
> >
> > **:point_down:==Measures in SQL==  ** 
> >
> > > :classical_building:机构：Google  
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Relational database query languages  
> > > - Data analytics  
> > > - Online analytical processing  
> > >
> > > :books:摘要：
> > >
> > > - 背景：SQL已被广泛采用，但传统的SQL任然缺乏*可组合计算*的能力
> > > - 本文：提出一种新型的附加列，叫做Measure(度量)
> > >   - 如何操作带度量的表：和普通表操作方法一样
> > >   - 带度量的SQL的优势：可在保留SQL语义同时，通过调用Measure解决更复杂的查询
> > >   - 度量如何计算得到：通过上下文(上下文敏感表达式)得到度量的值
> >
> > **:point_down:==ByteCard: Enhancing ByteDance’s Data Warehouse with Learned Cardinality Estimation==** 
> >
> > > :classical_building:机构：ByteDance  
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data management systems  
> > > - Computing methodologies → Machine learning  
> > >
> > > :books:摘要：​​
> > >
> > > - 背景：
> > >
> > >   - 关于ByteHouse：字节公司开发的云原生数据分析引擎，用于处理超大规模数据的复杂分析任务
> > >
> > >     :warning:云原生数据库：指专门为云环境设计和优化的数据库系统
> > >
> > >   - 关于基数估计：预测查询结果的数量(大小)，直接影响优化器的决策，是有护额的瓶颈所在
> > >
> > > - ByteCard的引入：融合最近在基数估计方面的进展，构建了兼顾可靠/实用的基数估计模型
> >
> > **:point_down:==Automated Multidimensional Data Layouts in Amazon Redshift==** 
> >
> > > :classical_building:机构：Amazon
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data layout
> > > - Autonomous database administration
> > > - Online analytical processing engines  
> > >
> > > :books:摘要：
> > >
> > > - 背景：关于数据布局技术，其是DB/DW中优化存储和访问效率的策略，常见为以下几种
> > >
> > >   |                             种类                             | 概述                                     | 示例(T=Tuple/A=Attribute)               |
> > >   | :----------------------------------------------------------: | ---------------------------------------- | --------------------------------------- |
> > >   |                            行存储                            | 一行数据连续存储                         | T1/A1→T1/A2→....→T1/An→T2/A1→....→Tm/An |
> > >   |                            列存储                            | 一列数据连续存储                         | T1/A1→T2/A1→....→Tm/A1→T1/A2→....→Tm/An |
> > >   |                            排序键                            | 数据按Key排序后存储                      | N/A                                     |
> > >   | [索引](https://blog.csdn.net/qq_64091900/article/details/141219405) | 建立表$\xleftrightarrow{}$内存位置的索引 | B+树，哈希表                            |
> > >
> > > - 本文的工作1：提出了多维数据布局(MDDL)
> > >
> > >   - 核心方法：传统方法是基于一组列对表进行排序，MDDL是==基于一组谓词(查询条件)对表进行排序==
> > >   - 优点：是的查询高度的定制化
> > >
> > > - 本文的工作2：提出一种自动化学习算法，基于历史工作负载，==自动学习每个表最佳的MDDL== 
> > >
> >
> > **:point_down:==Automated Clustering Recommendation With Database Zone Maps==**  
> >
> > > :classical_building:机构：Oracle  
> > >
> > > :arrow_right:领域：Theory of computation → Database query processing and optimization (theory)
> > >
> > > :books:摘要：一言蔽之，主要讲了区间图/自动聚类在数据仓库中的应用
> > >
> > > - 背景：关于区间图(Zone Maps)
> > >   - 结构：将表划分为Zone，存储每个区域的最大/最小值
> > >   - 工作原理：支持查询时，读取区间的最大/最小值，选择跳过/不跳过该区间，从而减少扫描工作量
> > >   - 优势：**在按某列排序/聚类处理后的数据上表现优越** 
> > > - 本文的工作：自动分析工作负载→推荐聚类方案(线性聚类和z-order聚类)→建区间图→提高查询性能
> > >
>
> ## 2.2. LLMs and ML Applications
>
> > **:point_down:==Similarity Joins of Sparse Features==**
> >
> > > :classical_building:机构：Uber
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Clustering  
> > > - Theory of computation → MapReduce algorithms  
> > >
> > > :books:摘要：提出了Fast Scalable Sparse Joiner (FSSJ)算法，用于在大规模稀疏数据上进行相似性连接
> > >
> > > - 一些前置知识和背景
> > >   - 相似性连接：在两个数据集中，找出相似性超过某个阈值的记录对
> > >   - 前缀过滤：相似性连接的一种技术
> > >     - 含义：对比属性的前N个属性(前缀)，如果两个记录的前缀不匹配，则默认不相似
> > >     - 存在的问题：某些元素在数据集中很流行/元素分布极其不均时，过滤效率会下降
> > >   - 稀疏特征：比如平均每个Tuple只有少数Attributes被赋值
> > > - 本文工作：关于FSSJ，引入Quasi-Prefix Filtering的新方法
> > >   - 针对频繁出现的流行元素做出优化，最流行元素不会被当作前缀来过滤
> > >   - 传统前缀过滤需对所有记录排序，然后广播给所有计算结点。准前缀过滤避免了广播操作
> > >
> >
> > **:point_down:==FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis==** 
> >
> > > :classical_building:机构：浙江大学
> > >
> > > :arrow_right:领域：Information systems → Structured Query Language  
> > >
> > > :books:摘要：金融领域Text-to-SQL的挑战与解决
> > >
> > > - 背景：Text-to-SQL
> > >   - 含义：通过自然语言生成SQL
> > >   - 问题与挑战：金融领域缺乏实用的Text-to-SQL基准数据集，现有Text-to-SQL没考虑金融数据库特点
> > > - 本文的工作
> > >   - BULL数据集：收集的一个实用的Text-to-SQL基准数据集
> > >   - FinSQL框架：一个基于大语言模型的Text-to-SQL框架，处理方法包括提示词构建/参数微调/输出校准
> > >
> >
> > **:point_down:==Rock: Cleaning Data by Embedding ML in Logic Rules==** 
> >
> > > :classical_building:机构：关河智图/深圳计算机研究院
> > >
> > > :arrow_right:领域：Information systems → Information integration  
> > >
> > > :books:摘要：提出一个基于ML的Rock系统，用来清洗Relational Data(就是Relational Database中的数据)
> > >
> > > - Rock的核心：结合机器学习/逻辑推理，通过将ML分类器嵌入为谓词来清洗数据
> > > - Rock的清洗任务：注意以下任务在Rock中可做到多任务协同处理
> > >   - 实体解析：将不同事物指向(识别并归类为)一个实体
> > >   - 冲突解决：捕捉不同实体之间的语义不一致(比如数据源1说A是20岁/数据源2说A是30岁)并解决
> > >   - 及时性推断：根据数据的属性值，判断这些值是否过期并更新
> > >   - 不完整信息补全
> > > - Rock的其它功能
> > >   - 自发从数据中发现规则
> > >   - 对大规模数据采取批处理模式
> > >   - 随数据更新而逐步更新
> > >
> >
> > **:point_down:==Data-Juicer: A One-Stop Data Processing System for Large Language Models==**
> >
> > > :classical_building:机构：阿里巴巴
> > >
> > > :arrow_right:领域：Information systems → Information integration  
> > >
> > > :books:摘要：提出了一个新的Data-Juicer系统，能够为LLM的训练生成多样化的数据组合(data recipes)
> > >
> > > - 背景：数据与LLM
> > >   - 数据在LLM的重要性：LLM的关键在于使用了==庞大的/异构的/高质量的==数据
> > >   - 数据组合：从不用来源混合而成的数据，用于训练LLM，决定了LLM的性能
> > > - 现有的问题：开源工具无法满足多样化数据需求，以及新数据源
> > > - Data-Juicer能干啥
> > >   - 对于异构且庞大的数据，能高效生成各种数据组合
> > >   - 能更高效评估数据组合对LLMs性能的影响
> > >
> >
> > **:point_down:==The Hopsworks Feature Store for Machine Learning==** 
> >
> > > :classical_building:机构：Hopsworks(瑞典软件公司)
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Database design and models  
> > > - Database management system engines.  
> > >
> > > :books:摘要：提出了Hopsworks机器学习特征存储(Feature Store)系统
> > >
> > > - 背景：ML系统中的数据管理
> > >   - 含义：是ML-Sys中处理/存储/组织数据，确保数据用于训练推理的过程，是ML-Sys最具挑战的部分
> > >   - 特征存储：管理ML数据的统一平台，贯穿了特征工程/训练/推理
> > > - Hopsworks特征存储平台：用于管理特征数据，解决了如下问题
> > >   - 特征重用：特征在不同机器学习任务中重复使用
> > >   - 数据转换：组织/执行特征过程的数据转换过程
> > >   - 确保一致性：保证特征工程/训练/推理时，数据是正确且一致的
> > >
> >
> > **:point_down:==COSMO: A Large-Scale E-commerce Common Sense Knowledge Generation and Serving System at Amazon==** 
> >
> > > :classical_building:机构：Amazon  
> > >
> > > :arrow_right:领域：
> > >
> > > - Computing methodologies → Knowledge representation and reasoning
> > > - Information systems → Web mining  
> > >
> > > :books:摘要：
> > >
> > > - 背景：现有电商图谱(产品属性-用户-商家关系)无法有效发现用户意图/反应用户思维
> > > - COSMO是个啥：可扩展系统，基于用户行为→构建用户知识图谱→为搜索导航提供服务
> > > - COSMO构建流程：
> > >   - 知识提取：用LLM从亚马逊大数据中提取初始知识
> > >   - 筛选：引入一个(基于人工标注数据)分类器，判断哪些知识可靠/不可靠并筛选
> > >   - 去噪：采用指令微调，进一步筛掉与人类认知有偏差的知识，==最终得到高质量的知识==
> > > - COSMO已经被部署在亚马逊的搜索和导航系统中
> > >
>
> ## 2.3. Cloud Storage
>
> > **:point_down:==LETUS: A Log-Structured Efficient Trusted Universal BlockChain Storage==**
> >
> > > :classical_building:机构：蚂蚁集团
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data management systems  
> > > - Security and privacy → Database and storage security  
> > >
> > > :books:摘要：提出了LETUS，用于区块链的高效/安全的通用存储系统
> > >
> > > - 背景：区块链爆炸增长，传统两层式存储结构已无法满足需求
> > > - LETUS系统的主要特点
> > >   - 打破传统两层架构：将认证数据结构(ADS)放到存储引擎，从而优化了存储和IO
> > >   - 提出了新型ADS：结合Merkle树+增量编码(delta-encoding)功能，称作DMM-Tree
> > >   - 改进的索引机制：基于版本的索引，用变种B树来索引ADS生成的数据页
> > >   - 通用性：适用各种区块链
> > >
> > > - LETUS已经在蚂蚁链的商业应用中部署，例如2023年亚运会的NFT项目和数字火炬点燃活动
> >
> > **:point_down:==Vortex: A Stream-oriented Storage Engine For Big Data Analytics==** 
> >
> > > :classical_building:机构：Google
> > >
> > > :arrow_right:领域：Information systems → Stream management
> > >
> > > :books:摘要：提出了Vortex，一个为Google BigQuery构建的**实时分析存储引擎**，支持对数据流的实时分析
> > >
> > > - 背景：
> > >   - 企业需要处理海量数据，尤其是对于连续数据流(streaming data)
> > >   - 传统数据系统分为流处理引擎/批处理系统，后者在处理实时数据时不佳
> > > - 关于Vortex
> > >   - 设计：专为数据流设计但也支持批处理，将两种操作集成到了同一个系统中
> > >   - 能力：处理PB级别的数据摄取(持续流入与分析)，能以亚秒级响应用户的实时查询
> > >
> >
> > **:point_down:==Native Cloud Object Storage in Db2 Warehouse: Implementing a Fast and Cost-Efficient Cloud Storage Architecture==** 
> >
> > > :classical_building:机构：IBM
> > >
> > > :arrow_right:领域：Information systems → Database management system engines  
> > >
> > > :books:摘要：提出了**Db2 Warehouse**存储架构的现代化改造，以适应云环境
> > >
> > > - 背景
> > >   - 传统小块存储：以4KB大小的数据页为存储单位(适合随机存取/块级IO)，但在云环境数据库中成本高
> > >   - 云对象存储：在处理大规模数据时，比传统小块存储成本更低
> > > - 存在的问题：将传统存储$\xrightarrow{迁移}$云对象存储成本巨大，因此需要新的架构
> > > - 对**Db2 Warehouse**架构的改进
> > >   - 将Log-Structured Merge(LSM)树整合到Db2 Warehouse系统，以管理大规模写入/查询
> > >   - 保留传统数据页格式，避免对传统数据库内核大幅重构
> > >
> >
> > **:point_down:==ESTELLE: An Efficient and Cost-effective Cloud Log Engine==** 
> >
> > > :classical_building:机构：电子科大/华为
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → DBMS engine architectures  
> > > - Structured text search  
> > >
> > > :books:摘要：提出了ESTELLE，转为云环境设计的日志引擎，用于管理大规模的日志数据
> > >
> > > - 背景：
> > >   - 日志的重要性：监控/调试/分析的核心数据
> > >   - 日志的特性：高频写入，低频检索，大量存储；这也是本文模型所要满足的
> > > - ESTELLE的设计与特点
> > >   - 采用了一种低成本日志索引框架，可根据需求灵活引用索引机制
> > >   - 分离计算和存储，以分离读写操作，从而确保系统能同时查询和写入
> > >   - 设计了一个近乎无锁的写入过程，以适应高频快速写入需求
> > > - ESTELLE存储与查询优化
> > >   - 采取对象存储技术(以对象为单位存储，包含数据/元数据/主键)
> > >   - 采取Log Bloom Filter和近似倒排索引，根据场景优化查询
> > >
> >
> > **:point_down:==TimeCloth: Fast Point-in-Time Database Recovery in The Cloud==** 
> >
> > > :classical_building:机构：阿里巴巴
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Database utilities and tools  
> > > - Point-in-time copies  
> > > - Storage recovery strategies  
> > > - Database recovery  
> > >
> > > :books:摘要：提出了TimeCloth，一种专为云环境设计的通用恢复机制，以优化用户触发的数据库恢复
> > >
> > > - 背景：关于用户触发的数据库恢复
> > >   - 特点：相比于因故障触发的恢复，需要更加考量用户的需求，如细粒度(精确程度)/时间点
> > >   - 现有方案：与底层数据库引擎高度集成，难以处理用户触发的恢复
> > > - TimeCloth的设计：专注实现**次线性恢复时间**，满足用户对恢复的特定要求
> > >   - 恢复模块：包括了几种机制，高效日志过滤/将非冲突日志并行回放/合并日志以减少工作量
> > >   - 导入模块：实现了透明的基于FUSE的延迟加载机制+智能预取功能
> > > - TimeCloth已经在阿里云上投入生产
> > >
>
> ## 2.4. Cloud Databases  
>
> > **:point_down:==Proactive Resume and Pause of Resources for Microsoft Azure SQL Database Serverless==**  
> >
> > > :classical_building:机构：微软
> > >
> > > :arrow_right:领域：Computer systems organization → Self-organizing autonomic computing  
> > >
> > > :books:摘要：提出了一种针对云数据库的**主动资源分配**基础设施，并用于无服务器的Azure SQL数据库
> > >
> > > - 背景：为云数据库分配资源
> > >   - 反应式：传统的方法，即根据当前需求分配资源
> > >   - 主动式：创新方法，结合当前需求+预期需求来分配资源
> > > - 本文的模型
> > >   - 要干啥：在**资源的高可用性**/**运营成本的降低**/**主动策略的计算开销**之间找到接近最优的平衡点
> > >   - 干了啥：用于管理数百万个无服务器的Azure SQL数据库
> > >
> >
> > **:point_down:==Vertically Autoscaling Monolithic Applications with CaaSPER==**  
> >
> > > :classical_building:机构：微软
> > >
> > > :arrow_right:领域：Information systems → Data management systems  
> > >
> > > :books:摘要：提出了CaaSPER垂直自动扩展算法，旨在优化Kubernetes平台上DBaaS的资源管理
> > >
> > > - 一些基本概念
> > >
> > >   - Kubernetes平台：管理云应用程序的开源平台，云应用分为有状态(对DB操作有赖于历史数据)/无状态
> > >   - 垂直扩展/水平扩展：增加单个服务器或节点的资源来提升处理能力/增加服务器节点数
> > >
> > >   :warning:Kubernetes通过垂直扩展来应对负载波动 
> > >
> > > - 现状问题
> > >
> > >   - Kubernetes平台上，客户为应对峰值负载会过度分配资源(负载下降时也没有缩减资源)
> > >   - 现有的垂直自动扩展工具在及时缩减资源或应对CPU限流时表现不佳
> > >
> > > - CaaSPER的提出
> > >
> > >   - 是个啥：结合反应式(负载临界时主动调整)+主动式(预测负载变化以主动调整)的垂直自动扩展算法
> > >   - 为了啥：保持**最佳的CPU利用率**，减少资源浪费
> > >   - 其它特性：允许用户选择能效模式/性能模式，可扩展性(与平台无关)
> >
> > **:point_down:==Flux: Decoupled Auto-Scaling for Heterogeneous Query Workload in Alibaba AnalyticDB==** 
> >
> > >   :classical_building:机构：阿里巴巴
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data warehouses  
> > > - Autonomous database administration
> > >
> > > :books:摘要：提出了Flux，一个专为阿里巴巴AnalyticDB设计的云原生**负载==自动扩展==平台**，用于优化异构查询
> > >
> > > - 背景(当前遇到的问题)
> > >   - 云数据仓库需要处理各种异构工作负载，比如在线事务/临时查询/ETL(抽取+转换+加载)
> > >   - 当长/短期查询混合执行时，并发控制+多任务执行会过于复杂
> > >   - 传统**自动扩展机制**在处理混合工作时，可能导致资源利用不平衡(有些过度分配/有些又不足)
> > > - 关于Flux
> > >   - 是啥：云原生的自动扩展平台，具有**解耦的自动扩展架构**，专用于==处理异构查询工作负载==
> > >   - 架构：
> > >     - 性能优化：将长/短期查询机制分开处理$\to$消除了传统系统中由于并发控制导致的瓶颈
> > >     - 资源弹性：利用无服务器容器实例来动态分配资源$\to$资源分配可快速响应负载变化
> > >
> >
> > **:point_down:==Intelligent Scaling in Amazon Redshift==**  
> >
> > > :classical_building:机构：Amazon
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → DBMS engine architectures  
> > > - Relational parallel and distributed DBMSs  
> > > - Autonomous database administration
> > > - Online analytical processing engines  
> > >
> > > :books:摘要：提出了基于AI的RAIS，用于解决云数据仓库在处理多样化工作负载时的==自动扩展==问题
> > >
> > > - 背景：阿里巴巴和亚马逊真是神奇的对手，两篇论文的论调都差不多，什么工作负载多样云云
> > > - 关于RAIS
> > >   - 是啥：一组基于AI驱动的扩展/优化技术
> > >   - 干啥：确保数据仓库能根据负载需求，从垂直/水平扩展(动态调整)资源
> > >   - 咋干：动态(响应)分配资源+自动优化数据仓库规模，这二者都是基于AI所完成的
> > >
> >
> > **:point_down:==Stage: Query Execution Time Prediction in Amazon Redshift==**  
> >
> > > :classical_building:机构：Amazon/MIT
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Database performance evaluation;  
> > > - Relational database model 
> > >
> > > :books:摘要：这个好理解，就是一种新的查询时间预测器，称之为Stage predictor，应用在Amazon Redshift
> > >
> > > - 背景：
> > >   - 在DBMS中查询时间的准确预测极为关键，关系到优化/资源分配等
> > >   - 现有预测技术存在一些问题，比如Cold Start(无历史数据时表现差)，工作负载变化大时预测不准
> > > - Stage predictor：一个分层执行的时间预测器，结合了以下三种模型
> > >   - 执行时间缓存：缓存过去的执行时间，预测时优先使用历史数据
> > >   - 轻量级本地模型：针对特定数据库实例进行优化，即对每个实例个性化预测
> > >   - 复杂的全局模型：一个可在Redshift实例剑转移的复杂模型，基于不同实例的共享知识预测
> > >
>
> ## 2.5. Cloud Database Architecture(**==<span style="color: red;">最佳论文</span>==**)
>
> > **:point_down:==PolarDB-MP: A Multi-Primary Cloud-Native Database via Disaggregated Shared Memory==**  
> >
> > > :classical_building:机构：阿里巴巴
> > >
> > > :arrow_right:领域：Information systems → Relational database model  
> > >
> > > :books:摘要：提出了PolarDB-MP，多主结构+云原生数据库，旨在解决主从数据库中写入吞吐量受限问题
> > >
> > > - 关于什么是主从数据库
> > >
> > >   - 主数据库：位于核心结点，处理所有写操作，将写操作同步到从数据库
> > >   - 从数据库：位于辅助结点，处理所有读操作，接收来自主数据库的更新从而保持一致
> > >
> > >   :waning_crescent_moon:这种做法的好处在于提高了读性能，坏处在于写必须经过主数据库→==限制了写性能==
> > >
> > > - 关于PolarDB-MP
> > >
> > >   - 是多主数据库，即允许多个结点成为主数据库，分散了写的负载
> > >   - 利用了==**分离式共享内存和存储**==架构
> > >     - 分离式架构：计算资源与存储资源分开设置在不同结点，二者都可独立扩展
> > >     - 共享内存/存储：多个结点可访问同一组内存/存储资源，本模型实质上每个结点可访问所有数据
> > >   - 允许事务在单个节点上处理
> > >
> > > - 关于PolarDB-MP的核心组件Polar Multi-Primary Fusion Server (PMFS)
> > >
> > >   - 设计思想：建立在分离式内存共享上，负责全局事务调节+缓冲区融合，采取了远程直接内存访问
> > >   - 主要功能：事务融合(跨结点事务一致)，缓冲区融合(跨结点内存共享)，锁融合(跨界点并发控制)
> > >
> > > - 关于**PolarDB-MP**引入的LLSN设计：为不同结点生成的写前日志，建立一个部分顺序的结构
> > >
> >
> > **:point_down:==Amazon MemoryDB: A Fast and Durable Memory-First Cloud Database==**  
> >
> > > :classical_building:机构：Amazon
> > >
> > > :arrow_right:领域：Information systems → Main memory engines  
> > >
> > > :books:摘要：提出了**基于云内存的数据库服务**Amazon MemoryDB for Redis
> > >
> > > - 关于Amazon MemoryDB for Redis的主要特点
> > >   - 内存高性能：MemoryDB将数据直接放在内存中，可以高速读写
> > >   - 高耐久性：除了将数据放在内存中，MemoryDB还会异步地将数据复制到外存中，防丢失
> > >   - 与Redis：MemoryDB基于Redis，兼容Redis(在Redis上运行的app可直接在MemoryDB上运行)
> > >   - 可扩展性：用户可按需(负载增加时)扩展存储/计算资源
> > >   - 高可用性：可多区部署，多区备份
> >
> > **:point_down:==Extending Polaris to Support Transactions==**  
> >
> > > :classical_building:机构：微软
> > >
> > > :arrow_right:领域：Information systems → Data management systems  
> > >
> > > :books:摘要：对Polaris系统的增强
> > >
> > > - 关于Polaris：一个云原生的分布式查询处理器
> > >
> > >   - 传统的Polaris：仅支持只读事务(查询)
> > >   - 增强的Polaris：支持所有常规事务(插入/删除/更新/加载)
> > >
> > > - 关于日志结构存储
> > >
> > >   - 原理：当插入/跟新/删除时，先把变更按顺序写入日志，一段时间后合并执行日志以更改实际数据
> > >   - 不可变性：一旦数据被写入，就不会再发生改变，新的数据不会覆盖而是追加 (避免了频繁磁盘修改)
> > >
> > >   :ocean:增强的Polaris正是采用了日志结构存储，利用其不可变性，大大提高了写入效率
> > >
> > > - 其它Polaris的技术特性
> > >
> > >   - 使用**快照隔离**语义(Snapshot Isolation/一种事务隔离级别)来保持数据一致，支持多表/多语句事务
> > >   - 支持T-SQL，即为微软的Fabric平台提供完整的T-SQL支持
> >
> > **:point_down:==BigLake: BigQuery’s Evolution toward a Multi-Cloud Lakehouse==**  
> >
> > > :classical_building:机构：Google
> > >
> > > :arrow_right:领域：Information systems → Data management systems  engines  
> > >
> > > :books:摘要：介绍了**BigLake**的设计及其在Google Cloud的BigQuery中的演变
> > >
> > > - BigQuery是啥：Google Cloud的云原生分布式查询处理器
> > >
> > > - 现今遇到的挑战
> > >
> > >   - 数据管理的复杂性：很多企业需要统一管理数据仓库/数据湖，但这又是俩不同结构的系统
> > >
> > >     |   结构   | 简单说明                                                   |
> > >     | :------: | ---------------------------------------------------------- |
> > >     |  数据库  | 用于实时存储、管理结构化数据，支持事务处理。               |
> > >     | 数据仓库 | 集成多个数据源，用于大规模数据分析和报表生成。             |
> > >     |  数据湖  | 存储原始、未处理的多种格式数据，支持大数据分析和机器学习。 |
> > >
> > >   - 如何整合不同格式的数据和表格
> > >
> > >   - 非结构化数据的处理：AI/ML工作负载处理需要处理的正是非结构化的数据，如何让它们高效处理？
> > >
> > >   - 多云部署：很多企业会用不同的云平台，如何让多个云平台运行相同服务？
> > >
> > > - BigLake：通过以下创新，是的数据仓库和数据湖得以结合
> > >
> > >   - BigLake Tables：使得BigQuery能处理分析不同格式的数据
> > >   - BigLake Object Tables：使BigQuery能处理非结构化数据，从而进行AI/ML处理
> > >   - Omni平台：使得BigQuery可以在非谷歌云平台运行
> >
> > **:point_down:==Predicate Caching: Query-Driven Secondary Indexing for Cloud Data Warehouses==**  
> >
> > > :classical_building:机构：Amazon
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data scans  
> > > - Online analytical processing engines  
> > > - Data warehouses  
> > >
> > > :books:摘要：提出了云数据仓库中**提高查询性能**的新方法，叫做谓词缓存
> > >
> > > - 背景
> > >
> > >   - 云数据仓库(比如Amazon Redshift)已成为查询处理的标准
> > >   - 用户和系统经常发送相同的查询，导致查询性能遇到瓶颈
> > >   - 当前系统的优化有赖于查询结果的缓存，但结果缓存会因为插入/删除/更新而过时
> > >
> > > - 为了解决上述问题，提出了谓词缓存(一种新的二级索引)
> > >
> > >   - 是啥：一种用于优化数据库查询性能的二级索引技术
> > >   - 干啥：解决传统缓存方法，在处理重复查询时面临的**缓存过时**问题
> > >
> > > - 谓词缓存的原理：以如下为例子阐述
> > >
> > >   | UserID (基础表) | Name  | Age  |
> > >   | :-------------: | :---: | :--: |
> > >   |        1        | Alice |  25  |
> > >   |        2        |  Bob  |  30  |
> > >   |        3        | Carol |  35  |
> > >   |        4        | Dave  |  40  |
> > >
> > >   ```sql
> > >   SELECT * FROM Users WHERE Age > 30; -- 查询结果如下
> > >   ```
> > >
> > >   | UserID (结果表) | Name  | Age  |
> > >   | :-------------: | :---: | :--: |
> > >   |        3        | Carol |  35  |
> > >   |        4        | Dave  |  40  |
> > >
> > >   - 传统的查询：缓存结果表的结果，下次发起相同查询时(若基础表没更新)直接输出缓存
> > >   - 谓词缓存查询：不会缓存结果，转而缓存基础表中==满足查询条件==的对象的范围，例如
> > >     - 缓存：执行上述查询，缓存会记录范围[35, 40]
> > >     - 更新：当基础表发生改变时，缓存也只要改变谓词范围(相比换掉整个结果表好得多)
> > >     - 再查询：利用缓存的范围信息，快速定位符合的数据
> > >
> > > - 谓词缓存的其它特性
> > >
> > >   - 可在查询执行时动态构建(摘要里也没细说)
> > >   - 谓词缓存是轻量级的(还是那句话，比缓存整张表好多了)，并且能够在线维护
>
> ## 2.6. Graph Data Management 
>
> > **:point_down:==BG3: A Cost Effective and I/O Efficient Graph Database in ByteDance==**  
> >
> > > :classical_building:机构：字节
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data management systems  
> > > - Storage management
> > >
> > > :books:摘要：介绍了字节新提出的ByteGraph 3.0(BG3)模型，用来处理大规模图结构数据
> > >
> > > - 背景
> > >   - 字节旗下产品(Tiktok/抖音/头条)每天生成大量图
> > >   - ByteGraph是字节的分布式图数据库，但随负载量增加显得力不从心
> > > - 关于ByteGraph 3.0：ByteGraph的升级版本，主要结构包含
> > >   - 图存储引擎：内存索引是基于BW-Tree(一种适合图数据存储的树形数据结构)，采取云存储
> > >   - 负载感知的空间回收机制：根据负载情况优化存储空间，减少写放大(写入量<<请求空间)
> > >   - 轻量级主从同步机制：保证扩展系统时，多个结点间的数据同步且一致，有利于实时处理
> >
> > **:point_down:==PG-Triggers: Triggers for Property Graphs==**  
> >
> > > :classical_building:机构：米兰理工
> > >
> > > :arrow_right:领域：Triggers(触发器)是数据库的一种自动化操作，即==特定事件发生时自动执行一系列预定操作==
> > >
> > > - Information systems → Triggers and rules  
> > > - Graph-based database models
> > > - Theory of computation → Database query languages (principles)  
> > >
> > > :books:摘要：提出了PG-Triggers的概念，是一个针对属性图(Property Graphs)添加触发器的方案
> > >
> > > - 回顾一下属性图：一种图数据库模型，用于存储+管理图数据库，由以下三种结构组成
> > >
> > >   | 图数据库结构 | 对应关系数据库结构 |                  举例                   |
> > >   | :----------: | :----------------: | :-------------------------------------: |
> > >   |    Nodes     |       Entity       |               学生，老师                |
> > >   |  Properties  |     Attribute      | 学生(StuID/成绩)，老师(Course/TecherID) |
> > >   |    Edges     |    Relationship    |       学生$\xleftarrow{授课}$老师       |
> > >
> > > - 背景：
> > >
> > >   - 现状：图数据库正在进行标准化工作
> > >   - 作者要干啥呢：为图数据库引入PG-Triggers，以支持类似SQL的触发机制
> > >
> > > - PG-Triggers (说的是写啥，反正在我辽阔的知识盲区中)
> > >
> > >   - 定义了触发器的语法和语义
> > >   - Neo4j 实现：将PG-Triggers翻译成Neo4j图数据库中的APOC触发器
> > >     - Neo4j：一个库，用于增强Neo4j的Cypher查询语言的功能
> > >     - APOC触发器：一种触发结构，让Neo4j在数据变化时执行预定操作
> > >   - Memgraph 实现：在这个库也实现了以下，为证明PG-Triggers的机制不仅适用于Neo4j
> >
> > **:point_down:==GraphScope Flex: LEGO-like Graph Computing Stack==**  
> >
> > > :classical_building:机构：阿里巴巴
> > >
> > > :arrow_right:领域：Computer systems organization → n-tier architectures.  
> > >
> > > :books:摘要：提出了GraphScope Flex(GraphScope系统的升级)，
> > >
> > > 

# 3. Demonstrations

> ## 3.1. Group A
>
> > 
>
> ## 3.1. Group B
>
> > 

# 4. Panels  

> 

# 5. ​​Tutorials

> 

# 6. Workshop Summaries  

> 
