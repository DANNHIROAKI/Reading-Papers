# 1. Keynotes  

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
> > :books:概述：
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
> > :books:概述：
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
> > :books:概述：
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
> > > :books:概述：
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
> > > :books:概述：
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
> > > :books:概述：
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
> > > :books:概述：
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
> > > :books:概述：
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
> > > :books:概述：一言蔽之，主要讲了区间图/自动聚类在数据仓库中的应用
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
> > > :books:概述：提出了Fast Scalable Sparse Joiner (FSSJ)算法，用于在大规模稀疏数据上进行相似性连接
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
> > > :books:概述：金融领域Text-to-SQL的挑战与解决
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
> > > :books:概述：提出一个基于ML的Rock系统，用来清洗Relational Data(就是Relational Database中的数据)
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
> > > :books:概述：提出了一个新的Data-Juicer系统，能够为LLM的训练生成多样化的数据组合(data recipes)
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
> > > :books:概述：提出了Hopsworks机器学习特征存储(Feature Store)系统
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
> > > :books:概述：
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
> > > :books:概述：提出了LETUS，用于区块链的高效/安全的通用存储系统
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
> > > :books:概述：提出了Vortex，一个为Google BigQuery构建的**实时分析存储引擎**，支持对数据流的实时分析
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
> > > :books:概述：提出了**Db2 Warehouse**存储架构的现代化改造，以适应云环境
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
> > > :books:概述：提出了ESTELLE，转为云环境设计的日志引擎，用于管理大规模的日志数据
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
> > > :books:概述：提出了TimeCloth，一种专为云环境设计的通用恢复机制，以优化用户触发的数据库恢复
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
> > > :books:概述：提出了一种针对云数据库的**主动资源分配**基础设施，并用于无服务器的Azure SQL数据库
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
> > > :books:概述：提出了CaaSPER垂直自动扩展算法，旨在优化Kubernetes平台上DBaaS的资源管理
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
> > >   :arrow_right:领域：
> > >
> > >   - Information systems → Data warehouses  
> > >   - Autonomous database administration
> > >
> > >   :books:概述：提出了Flux，一个专为阿里巴巴AnalyticDB设计的云原生**负载==自动扩展==平台**，用于优化异构查询
> > >
> > >   - 背景(当前遇到的问题)
> > >   - 云数据仓库需要处理各种异构工作负载，比如在线事务/临时查询/ETL(抽取+转换+加载)
> > >   - 当长/短期查询混合执行时，并发控制+多任务执行会过于复杂
> > >   - 传统**自动扩展机制**在处理混合工作时，可能导致资源利用不平衡(有些过度分配/有些又不足)
> > >   - 关于Flux
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
> > > :books:概述：提出了基于AI的RAIS，用于解决云数据仓库在处理多样化工作负载时的==自动扩展==问题
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
> > > :books:概述：这个好理解，就是一种新的查询时间预测器，称之为Stage predictor，应用在Amazon Redshift
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
> ## 2.5. Cloud Database Architecture
>
> > **:point_down:==PolarDB-MP: A Multi-Primary Cloud-Native Database via Disaggregated Shared Memory==**(**==<span style="color: red;">最佳论文</span>==**)  
> >
> > > :classical_building:机构：阿里巴巴
> > >
> > > :arrow_right:领域：Information systems → Relational database model  
> > >
> > > :books:概述：提出了PolarDB-MP，多主结构+云原生数据库，旨在解决主从数据库中写入吞吐量受限问题
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
> > > :books:概述：提出了**基于云内存的数据库服务**Amazon MemoryDB for Redis
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
> > > :books:概述：对Polaris系统的增强
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
> > > :books:概述：介绍了**BigLake**的设计及其在Google Cloud的BigQuery中的演变
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
> > > :books:概述：提出了云数据仓库中**提高查询性能**的新方法，叫做谓词缓存
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
> > > :books:概述：介绍了字节新提出的ByteGraph 3.0(BG3)模型，用来处理大规模图结构数据
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
> > > :books:概述：提出了PG-Triggers的概念，是一个针对属性图(Property Graphs)添加触发器的方案
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
> > > :books:概述：提出了GraphScope Flex(GraphScope系统的升级)
> > >
> > > - 关于GraphScope
> > >   - 是个啥：用于图遍历+分析+学习的综合解决方案
> > >   - 遇到的困难：不够万能(处理各种编程接口/App/Data时不够多样)
> > > - 关于GraphScope Flex
> > >   - 目标：解决GraphScope所面对的多样性挑战，权衡资源和效益，提供灵活和用户友好
> > >   - 模块化：采取类似乐高积木的模块化，允许用户根据需求组合定制
> > > - 结果评估
> > >   - GraphScope Flex在LDBC社交网络基准测试中实现了2.4倍的吞吐量提升
> > >   - GraphScope Flex在Graphalytics基准测试中达到了最高55.7倍的加速比
> > >   - 在实际应用中，GraphScope Flex表现出高达2,400倍的性能提升
> > >
> >
> > **:point_down:==Bouncer: Admission Control with Response Time Objectives for Low-latency Online Data Systems==**  
> >
> > > :classical_building:机构：领英/微软
> > >
> > > :arrow_right:领域：
> > >
> > > - General and reference → Empirical studies  
> > > - Information systems → Main memory engines
> > > - Database utilities and tools  
> > >
> > > :books:概述：提出了Bouncer(一种查询接纳控制策略)，在流量激增时确保查询能尽快响应
> > >
> > > - 背景：
> > >   - 现实背景：互联网公司在经历突发流量时，需采取策略让查询满足**响应时间目标(SLOs)**
> > >   - 查询接纳控制策略：(尤其在流量暴增时)用于控制接受/拒绝用户的查询请求
> > > - 关于Bouncer策略
> > >   - 是啥：一种查询接纳控制策略
> > >   - 基本原理：低成本估算当前响应时间分布→判断新查询是否能在SLOs内完成→拒绝/接受
> > >   - 其它策略/原理
> > >     - 查询分类：为不同类别的查询设置不同的SLO
> > >     - 早期拒绝策略：帮助客户端迅速作出反应，避免系统浪费资源在无效的查询上
> > >     - 避免饥饿策略：确保某些类别的查询不会被完全拒绝，防止查询类型长期得不到服务
> > > - 评估与结果
> > >   - Bouncer有效避免了饥饿
> > >   - Bouncer通过较少的总体拒绝次数，达到总体较小的系统开销
> > >     - 高负载下，让已接纳的查询保持接近其SLO
> > >     - 其它查询则不能达到SLO
> >
> > **:point_down:==NPA: Improving Large-scale Graph Neural Networks with Non-parametric Attention==**  
> >
> > > :classical_building:机构：北京大学
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data mining  
> > > - Computing methodologies → Machine learning  
> > >
> > > :books:概述：设计了非参数化(Non-parametric)GNN与非参数化注意力(PNA)
> > >
> > > - 基础背景：
> > >   - 传统GNN：GNN处理大规模图数据时，可扩展性差
> > >   - 以往研究：通过GNN高采样技术来提交扩展性
> > >   - 现在研究：非参数化GNN训练不依赖大量可训练参数，许多场景下扩展性都很强
> > > - 另一个背景：非参数化GNN的局限
> > >   - 过平滑问题：由于特征的过度传播，随着传播层数增加，网络性能急剧下降
> > >   - 忽略了特征的影响：非参数化GNN传播时只考虑了图结构，忽略了特征的影响
> > > - NPA模块的提出
> > >   - 是啥：一个可插拔的模块，兼容现有非参数化GNN，使其同时支持可扩展性+更深架构
> > >   - 原理：引入注意力机制，通过传播时权衡特诊&图结构的重要性，来优化特征传播
> > > - 验证与实验
> > >   - NPA在七个同构图/五个异构图中表现优异
> > >   - 在大规模数据集**ogbn-papers100M**上，NPA 实现了**最先进的性能**
> > >   - 一言蔽之：**高性能，高扩展性，支持更深网络结构**

# 3. Demonstrations

> ## 3.1. Group A
>
> > **:point_down:==Demonstration of Ver: View Discovery in the Wild==**  
> >
> > > :classical_building:机构：芝加哥大学
> > >
> > > :arrow_right:领域：Information systems → Information integration  
> > >
> > > :books:概述：展示了Ver1数据发现系统
> > >
> > > - 能干啥：在没提供连接路径信息的大型表格库中，识别出Project-Join视图
> > > - 解决了啥问题
> > >   - 技术问题：面对大规模表格，要能快速找出视图
> > >   - 认为问题：如何帮用户理解+使用这些视图(因为导航结果复杂性/路径链接多样性等)
> >
> > **:point_down:==Comquest: Large Scale User Comment Crawling and Integration==**  
> >
> > > :classical_building:机构：天普大学/IBM
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Deep web  
> > > - Web crawling  
> > > - Information integration
> > > - Information systems applications.  
> > >
> > > :books:概述：展示了名为 **Comquest** 的评论抓取系统，利用Web API来收集大量网站用户评论
> > >
> > > - 问题背景
> > >   - 用户的评论对于下游应用有重要价值
> > >   - 评论数据受限于特定平台，使得数据可用性受限，群体多样化受限
> > > - Comquest 系统的设计
> > >   - 能干啥：(跨平台)抓取与特定新闻话题或故事相关的评论数据
> > >   - 怎么干：通过深度学习抓取API参数→发送HTTP请求到第三方评论系统的API→收集评论
> > >   - 广泛性：不仅适用于新闻网站，还可与任何用户评论网站配合使用
> >
> > **:point_down:==QueryShield: Cryptographically Secure Analytics in the Cloud==**  
> >
> > > :classical_building:机构：波士顿大学
> > >
> > > :arrow_right:领域：
> > >
> > > - Security and privacy → Cryptography  
> > > - Information systems → Data management systems  
> > >
> > > :books:概述：展示了QueryShield，为云端数据分析提供加密安全服务，以保护隐私+简化多方安全计算
> > >
> > > - QueryShield 的功能
> > >   - 数据分析描述发布：
> > >     - 数据分析师$\xrightarrow[\text{QueryShield}]{发布分析描述}$数据所有者
> > >     - 数据所有者在保证隐私前提下，选择参与计算以获利/公益
> > >   - 数据隐私保障：提供多方安全计算技术，为关系数据库/时间序列分析，提供隐私保护
> > > - QueryShield 的特性：用户友好，封装了多方安全计算(MPC)的复杂计算，非专家也可使用
> > > - 文中演示的三个场景：四人就业信息调查+信用评分 异常分析+医学场景
> >
> > **:point_down:==SIERRA: A Counterfactual Thinking-based Visual Interface for Property Graph Query Construction==**  
> >
> > > :classical_building:机构：南洋理工
> > >
> > > :arrow_right:领域：
> > >
> > > - Human-centered computing → Visualization systems and tools  
> > > - Information systems → Query languages  
> > >
> > > :books:概述：展示了新型视觉查询界面(VQI) SIERRA，帮不会图查询语言(Cypher)用户构建属性图数据库
> > >
> > > -  背景知识
> > >
> > >   - 属性图：一种图数据库模型，用于存储+管理图数据库，由以下三种结构组成
> > >
> > >     | 图数据库结构 | 对应关系数据库结构 |                  举例                   |
> > >     | :----------: | :----------------: | :-------------------------------------: |
> > >     |    Nodes     |       Entity       |               学生，老师                |
> > >     |  Properties  |     Attribute      | 学生(StuID/成绩)，老师(Course/TecherID) |
> > >     |    Edges     |    Relationship    |       学生$\xleftarrow{授课}$老师       |
> > >
> > >   - 视觉查询界面：一种帮助用户建立数据库的图形化界面，而无需编写代码(比如SQL)
> > >
> > > - 背景：
> > >
> > >   - 属性图大受欢迎，但特定查询语言构成了门槛→视觉查询界面
> > >   - 现有视觉查询界面虽然易用，但未充分考虑HCI规律和心理学
> > >
> > > - SIERRA 的设计创新：解决了现有视觉查询界面在可用性和美观性上的不足
> > >
> > >   - 理论驱动的设计：采用**反事实思维**，结合HCI/可视化/心理学原则，使得界面直观易用
> > >   - 标签复合图(LCG)：引入标签复合图，展示图的结构
> > >   - 视觉形状定义语言：融入在SIERRA的设计里，在查询构建过程中引导用户创建和维护LCG
> >
> > **:point_down:==Sawmill: From Logs to Causal Diagnosis of Large Systems==**  
> >
> > > :classical_building:机构： MIT
> > >
> > > :arrow_right:领域：
> > >
> > > - Software and its engineering → System administration  
> > > - Computing methodologies → Causal reasoning and diagnostics  
> > > - Natural language generation  
> > >
> > > :books:概述：展示了Swamill系统，用来从复杂**日志文件**中提取**因果关系**
> > >
> > > - 背景：
> > >   - 因果分析在**复杂系统**的动态中至关重要
> > >   - 计算机作为复杂系统，很多信息都在**半结构化**的日志文件中，难以提取因果
> > > - Sawmill 系统的设计与功能
> > >   - 数据转换与清理：**半结构化**原始日志数据$\xrightarrow{\text{Sawmill}}$适合因果分析的**结构化**表示形式
> > >   - 可理解的变量命名：系统会自动地，将从日志中提取出的变量，命名为人类可理解的名称
> > >   - 聚合变量生成：Sawmill根据用户选择的因果单元，生成相关的聚合变量
> > > - Sawmill能干啥
> > >   - 高效地将日志数据转化为可以进行因果推理的模型，并进行探索式因果发现
> > >   - 允许用户通过交互式界面参与，从而使用现有的工具进行因果推理
> >
> > **:point_down:==Demonstrating REmatch: A Novel RegEx Engine for Finding all Matches==**  
> >
> > > :classical_building:机构： 牛津大学/智利天主教大学
> > >
> > > :arrow_right:领域：
> > >
> > > - Theory of computation → Regular languages  
> > > - Information systems → Information retrieval
> > >
> > > :books:概述：展示了名为REmatch的正则表达式(RegEx)引擎
> > >
> > > - 背景知识
> > >
> > >   - 正则表达式：一种用于**模式匹配**的工具，如以下示例
> > >
> > >     ```txt
> > >     (1) 电子邮件的正则匹配表达式^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$
> > >     (2) 123-456-7890类型的表达式被正则表达式\d{3}-\d{3}-\d{4}匹配
> > >     ```
> > >
> > >   - 正则表达式引擎：用于解析+匹配正则表达式，并返回结果
> > >
> > > - REmatch引擎的设计
> > >
> > >   - 基于枚举算法理论，找到文档中给定模式的所有匹配项
> > >   - 区别于传统正则引擎，REmatch无需使用复杂非标准操作符，就能找到嵌套和重叠的匹配项
> > >   - 时间复杂度与逐字符输出匹配结果的时间成比例
> > >
> > > - 用户界面：https://rematch.cl
> > >
> > > - 应用场景：DNA序列分析，语言分析，如本文展示例子所示
> >
> > **:point_down:==ASQP-RL Demo: Learning Approximation Sets for Exploratory Queries==**  
> >
> > > :classical_building:机构： 宾夕法尼亚大学/Aviv大学
> > >
> > > :arrow_right:领域：Information systems → Data management systems   
> > >
> > > :books:概述：展示了ASQP-RL系统，用于优化针对大规模外部数据的非聚合查询
> > >
> > > - 背景：处理大规模外部数据库的查询很耗时，尤其当内存有限时
> > > - ASQP-RL原理
> > >   - 用户发起**非聚合查询**(SELECT/PROJECT/JOIN)
> > >   - ASQP-RL运行强化学习算法选择外部数据库一个子集
> > >     - 此处强化学习算法的原理：通过局部数据子集来近似全局数据查询的结果
> > >   - ASQP-RL将选择的子集加载到本地，成为**近似集**
> > >   - ASQP-RL对已经物理化(本地化)的子集执行快速的查询
> > > - ASQP-RL的优势：
> > >   - 查询时间速快
> > >   - 查询结果准确(虽然只取了子集，但效果和取整体差不多)
> > >   - 针对聚合查询也有良好效果
> >
> > **:point_down:==IMBridge: Impedance Mismatch Mitigation between Database Engine and Prediction Query Execution==**  
> >
> > > :classical_building:机构： 华东师大/蚂蚁集团
> > >
> > > :arrow_right:领域：Information systems → Query optimization  
> > >
> > > :books:概述：展示了IMBridge系统，旨在弥合[数据库引擎$\leftrightarrow$机器学习预测]间的**阻抗不匹配**问题
> > >
> > > - 背景知识
> > >   - ML×DB：机器学习模型可用于对存储在数据库的数据执行分析
> > >   - Python UDF：看起来很高级，其实就是Python User-Defined Function的意思
> > >   - 阻抗不匹配：两个系统或组件之间差异过大，导致无法交互，协调效率差
> > > - 研究背景
> > >   - 阻抗不匹配：
> > >     - 当前数据库在查询引擎中引入Python UDF(预测函数)，以在处理查询时执行ML推理
> > >     - 数据库无法理解预测函数语义
> > >   - 推理上下文重复：传统方法中，没调用一次预测函数，都要重新设置上下文环境
> > >   - 不匹配的批量大小：源于数据库操作与预测函数的批量处理之间缺乏协调，影响吞吐
> > > - IMBridge 系统的解决方案
> > >   - 通过预测函数重写器→消除多余的推理上下文设置
> > >   - 引入了一个解耦的预测操作符→统一数据库与预测函数的批大小
> >
> > **:point_down:==ASM in Action: Fast and Practical Learned Cardinality Estimation==**  
> >
> > > :classical_building:机构： 浦项科技大学/洛桑联邦理工大学
> > >
> > > :arrow_right:领域：Information systems → Query optimization  
> > >
> > > :books:概述：展示了名为ASM的基数估计器
> > >
> > > - 背景
> > >   - 基数估算：用于估计查询结果/中间结果的大小，从而反向优化查询
> > >   - 现有问题：
> > >     - 基于机器学习的基数估算器能够显著提高估算精度
> > >     - 实际部署中，ML无法与数据库查询优化器结合，导致性能不佳
> > > - ASM的改进
> > >   - 使用自动回归模型，即利用历史数据对当前查询结果进行预测
> > >   - 从数据库中进行适当的采样
> > >   - 利用多维统计合并，在复杂多维数据上提供更高效的基数估算
> > > - ASM能干啥：
> > >   - 显著提升了基数估算器的效率，尤其是在复杂/多维查询情况下
> > >   - 更容易与现有的数据库查询优化器集成，避免了“估算精度高但执行效率低”的问题
> >
> > **:point_down:==The Game Of Recourse: Simulating Algorithmic Recourse over Time to Improve Its Reliability and Fairness==**  
> >
> > > :classical_building:机构： 纽约大学
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data management systems;  
> > > - Social and professional topics → Socio-technical systems
> > > - Human centered computing;  
> > >
> > > :books:概述：讨论了算法反应(Algorithmic Recourse)概念，并提供了一种通过模拟生成相关数据的方法
> > >
> > > - 背景知识
> > >   - 算法反应：为在算法系统中得到不利结果的人提供建议，使其采取行动改变结果
> > >   - 算法反应的目的：发挥人的主观能动性，从而让人对算法有更多控制权
> > >   - 算法反应的困境：缺乏公开可用的数据集
> > > - 关于The Game Of Recourse
> > >   - 是啥：一个基于代理的模拟
> > >   - 干啥：生成现实的算法反应数据
> > >   - 灵感：来自于康威的“生命游戏”Conway’s Game of Life (笑)
> > >   - 特性：可靠性+公平性
> > > - 开放访问： https://game-of-recourse.streamlit.app 
> >
> > **:point_down:==RobOpt: A Tool for Robust Workload Optimization Based on Uncertainty-Aware Machine Learning==**  
> >
> > > :classical_building:机构： 渥太华大学/IBM
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Query optimization  
> > > - Computing methodologies → Uncertainty quantification  
> > > - Supervised learning by regression
> > >
> > > :books:概述：展示了RobOpt系统，旨在解决关系型数据库管理系统(R-DBMS)中的查询优化问题
> > >
> > > - 背景知识
> > >   - 优化器：R-DBMS依赖于查询优化器，为从查询选择最优计划，以达到优化目的
> > >   - 优化器原理：依赖于数据库中达到统计信息(数据分布/查询条件)→估计查询的代价和参数
> > > - 研究背景
> > >   - 传统优化器的缺陷：
> > >     - 传统优化器的参数估计准确性差，执行查询总是非最优
> > >     - 传统优化器基于特定场景
> > >   - 基于机器学习优化器的缺陷：处理不同工作负载时，通常会选择次优方案，从而优化不力
> > > - RobOpt 的提出
> > >   - 是啥：针对工作负载的**鲁棒查询优化器**，使得查询计划的选择更加稳健
> > >     - **鲁棒查询优化器**：在面对不确定性/系统波动，仍然选出最优计划的优化器
> > >   - 原理
> > >     - 使用数据库的**查询日志**作为输入
> > >     - 通过日志中的历史数据，训练出一个**基于风险感知的学习代价模型**
> > >     - 在执行优化器时，考虑风险因素，并采取风险感知的计划策略
> > >     - 可以在工作负载级别/单个查询级别上分析查询样本。都可做出最优选择
> > > - RobOpt 的优势
> > >   - 鲁棒性：在不确定较大的场景任然保持稳定性+高性能
> > >   - 灵活性：可部署在任何R-DBMS上
> >
> > **:point_down:==Demonstrating CAESURA: Language Models as Multi-Modal Query Planners==**  
> >
> > > :classical_building:机构： 达姆城工业大学
> > >
> > > :arrow_right:领域：Information systems → Semi-structured data  
> > >
> > > :books:概述：展示了**CAESURA**系统，用于将数据库技术与LLM结合，从而处理多模态数据
> > >
> > > - 背景与背景知识
> > >   - 多模态数据
> > >     - 含义：指包含不同形式的数据，比如表格/文本/图像
> > >     - 应用：在基于LLM的问答系统中，需要enable用户去查询多模态数据
> > >   - RAG(Retrieval Augmented Generation)
> > >     - 是啥：一种扩展LLM的技术
> > >     - 干啥：先从向量数据库中检索相关数据→将数据输入LLM来计算查询结果
> > >     - 弊端：LLM推理成本很高，LLM只能处理有限数据(对大规模RAG束手无策)
> > > - CAESURA 的提出
> > >   - 是啥：一种数据库优先的多模态问答系统
> > >   - 核心思想：使用 LLM 的推理能力→翻译自然语言查询→生成数据库执行计划
> > >   - 工作流程
> > >     - 用户通过自然语言，提出查询
> > >     - CAESURA 使用 LLM 将查询翻译成数据库可以执行的查询计划
> > >     - 数据库系统(而非LLM)执行查询
> > >   - 优势：
> > >     - 得益于LLM，从而(通过转换自然语言)能处理多模态数据
> > >     - 得益于数据库系统，可以快速处理(而非是让LLM进行高成本的推理)
> > >     - 扩展性好，能够处理大规模的数据集，不想RAG数据一大就阿巴阿巴
> >
> > **:point_down:==Demonstration of Udon: Line-by-line Debugging of User-Defined Functions in Data Workflows==**  
> >
> > > :classical_building:机构： 加州大学欧文分校
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data management systems  
> > > - Software and its engineering → Software testing and debugging  
> > >
> > > :books:概述：展示了Udon调试器，用于在大数据处理系统中，逐行调试复杂用户自定义函数(UDF)
> > >
> > > - 背景
> > >   - 编程语言的差异：
> > >     - 大数据系统由C/C++/Java编写
> > >     - 用户用Python分析处理，比如机器学习API有99%都是python
> > >     - UDF成为bridge their gap的重要工具
> > >   - UDF调试的挑战：需要协同不同编程语言+大数据规模庞大(开销高)
> > > - Udon 的解决方案
> > >   - 逐行调试：用户可设断点+逐行单步走，可以在调试UDF时修改代码
> > >   - 单个元组调试：允许UDF在单个Tuple上执行，逐行检查运行情况
> > >   - 调试原语：包含了现代化调试原语，比如设断点+代码检查+动态修改代码
> >
> > **:point_down:==UniTS: A Universal Time Series Analysis Framework Powered by Self-Supervised Representation Learning==**  
> >
> > > :classical_building:机构： 哈工大
> > >
> > > :arrow_right:领域：
> > >
> > > - Computing methodologies → Machine learning  
> > > - Mathematics of computing → Time series analysis  
> > >
> > > :books:概述：展示UniTS框架，用于解决时序分析中的问题，比如部分标注数据/领域漂移
> > >
> > > - 背景问题：时间序列预测的一些挑战
> > >   - 部分标注：即不是所有数据都被完整标注，即不是所有数据都有正确标签
> > >   - 领域漂移：应用于某模型的领域，在新的领域表现不佳
> > > - UniTS 框架的设计
> > >   - 自监督表征学习：使得模型在标签不完整时，通过学习内在结构+表征，提升分析效果
> > >   - Sklearn 风格 API：尊重用户习惯，开发者可灵活使用该架构
> > >   - 用户友好GUI：高度封装，好看，傻子也能用
> >
> > **:point_down:==ChatPipe: Orchestrating Data Preparation Pipelines by Optimizing Human-ChatGPT Interactions==**  
> >
> > > :classical_building:机构： 人大
> > >
> > > :arrow_right:领域：Information systems → Data analytics  
> > >
> > > :books:概述：展示了ChatPipe新系统，通过与ChatGPT对话来简化机器学习的数据准备过程
> > >
> > > - 关于数据准备
> > >   - 是啥：数据清洗，转换。处理等
> > >   - 将ChatGPT用于数据准备
> > >     - 咋整：根据用户提示生成代码，运行代码来进行数据准备
> > >     - 缺陷：需用户引导ChatGPT因此需具备一定编程基础+生成的代码无法滚回(需从头引导)
> > > - ChatPipe 系统的设计
> > >   - 套壳ChatGPT：不仅与ChatGPT无缝交互
> > >   - 操作推荐：智能提示用户下一步操作，从而更好的引导ChatGPT
> > >   - 版本控制与滚回：允许用户滚回到以前的版本。无需从头还是引导
> > > - Demo Session
> > >   - 被集成到了一个Web应用里
> > >   - 在Kaggle数据集上可完成高效准备
>
> ## 3.1. Group B
>
> > **:point_down:==Responsible Model Selection with Virny and VirnyView==**  
> >
> > > :classical_building:机构： 乌克兰天主教大学/纽约大学
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data management systems  
> > > - Social and professional topics → Socio-technical systems
> > > - Human centered computing  
> > >
> > > :books:概述：展示了Virny软件库和与之配套的交互工具VirnyView，用户模型审计+模型选择
> > >
> > > - 模型审计：对机器学习模型进行系统性评估和分析，涵盖准确性/稳定性/鲁棒性
> > > - 关于Virny软件库
> > >   - 特性：模块化+可扩展性，用户可根据需求扩展其功能
> > >   - 技术手段
> > >     - 具有一套评估机器学习性能的公平性指标，其中包括很多新指标
> > >     - 提供了一套基于多个敏感属性(性别/种族)的分析功能，用于评估在不同人群的表现
> > > - VirnyView工具：一个配套的交互工具，提供可视化界面，封装了模型审计和选择的过程
> > >
> > > - 开放访问：https://github.com/DataResponsibly/Virny and https://r-ai.co/VirnyView
> >
> > **:point_down:==Property Graph Stream Processing In Action with Seraph==**  
> >
> > > :classical_building:机构： 乱七八糟
> > >
> > > :arrow_right:领域： 乱七八糟
> > >
> > > :books:概述：介绍了Seraph，一种基于Cypher的查询语言，专注于处理流图数据+连续查询 
> > >
> > > - 背景
> > >
> > >   - 图数据模型的普及+Cypher查询语言的推广→图数据分析越来越重要
> > >   - 现有的图查询语言(Cypher)在处理流图数据存在局限，如不可连续查询
> > >     - 流图数据，就是实时性高的图数据
> > >
> > > - 关于两种编程语言
> > >
> > >   - 声明式(declarative)：用户只需描述需求就可得结果，具体每一步怎么做不用管，比如SQL
> > >
> > >     ```sql
> > >     SELECT name FROM students WHERE age > 18;
> > >     ```
> > >
> > >   - 命令式(Imperative)：需用户明确每一步该怎么做，比如Python
> > >
> > >     ```python
> > >     result = []
> > >     for student in students:
> > >         if student.age > 18:
> > >             result.append(student.name)
> > >     ```
> > >
> > > - 关于Seraph
> > >
> > >   - 核心创新点：基于Cypher，支持**本地连续查询**，可在**流图数据**上查询并给出实时结果
> > >   - 特性
> > >     - 是声明式(declarative)语言
> > >     - 向后兼容了Cypher，即Cypher语言也可在Seraph中使用
> > >     - 有严格的形式化定义，即用符号+表达式来描述问题
> > >
> > > - 其它
> > >
> > >   - Seraph还提供了一个web用户界面
> > >   - 演示视频：https://riccardotommasini.github.io/seraph/
> >
> > **:point_down:==Property Graph Stream Processing In Action with Seraph==**  
> >
> > > :classical_building:机构：智利的一堆大学
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Data management systems  
> > > - Database query processing  
> > > - Graph-based database models  
> > >
> > > :books:概述：展示了MillenniumDB，一种高性能开源图数据库
> > >
> > > - 背景与问题
> > >   - 知识图谱的数据多样性：包含文本/图像/表格/视频/音频，图数据库还需支持多个共存DB
> > >   - 多种数据具备需要互相操作的需求，因此更需要处理和查询多样化的数据格式
> > > - MillenniumDB 的特点
> > >   - 支持多模态+多模型：支持属性图模型，语义网络式RDF，以及结合这二者的多层图模型
> > >   - 支持的查询语言：
> > >     - 支持属性图和多层图上的类似 Cypher 的查询语言
> > >     - 支持在 RDF 数据上执行 SPARQL 1.1 查询
> > >   - 优化的查询引擎：
> > >     - 结合了最坏情况最优连接算法+统的关系型查询优化技术
> > >     - 支持多种图特定任务，如路径查找、模式识别和多模态数据的相似性搜索
> > > - Demo Session：在TelarKG/BibKG/Wikidata等图谱上表现良好
> >
> > **:point_down:==IDE: A System for Iterative Mislabel Detection==**  
> >
> > > :classical_building:机构：北理
> > >
> > > :arrow_right:领域：Information systems → Data cleaning  
> > >
> > > :books:概述：介绍了IDE系统，用于在ML训练种解决标签错误的问题，提高标签的质量以利于训练
> > >
> > > -  背景及背景知识
> > >   - 标签错误：即在数据集种标注错误标签
> > >   - 标签错误的后果：会让DL模型性能雪崩，因为DL赖于高质量标签
> > >   - 获取高质量标签过程需要人工验证，成本极高
> > > - IDE 系统的介绍
> > >   - 采用一种**迭代检测**和修复错误标签的方法
> > >     - 每次迭代种，IDE使用早期损失观察+基于影响的验证，来识别错误标签
> > >     - 对于识别出的错误标签，系统随之做出修复
> > >   - 当系统检测到早期损失观察不再有效时，自动终止迭代
> > >   - 对于难以确定标签的实例，IDE会生成伪标签，这也可以提高总体的标签质量
> >
> > **:point_down:==A Demonstration of GPTuner: A GPT-Based Manual-Reading Database Tuning System==**  
> >
> > > :classical_building:机构：四川大学
> > >
> > > :arrow_right:领域：Information systems → Database administration  
> > >
> > > :books:概述：提出了名为GPTuner的DBMS自动调优系统
> > >
> > > - 背景
> > >   - 可配置参数(knobs)对数据库系统影响很大，但人为调整这些参数到最优及其困难
> > >   - 目前已有的机器学习自动调整系统有赖于黑箱优化，忽略了数据库领域知识
> > > - GPTuner 的提出
> > >   - 核心：GPTuner系统利用LLM，通过阅读数据库文档/手册等，讲黑箱优化与领域知识结合
> > >   - 用户与专家的合作
> > >     - 用户：GPTuner解读参数特性提供定见解，以帮助优化，无需用户深入掌握优化知识
> > >     - 专家：通过自然语言输入调优建议，进一步增强GPTuner功能
> >
> > **:point_down:==Demonstrating 𝜆-Tune: Exploiting Large Language Models for Workload-Adaptive Database System Tuning==**  
> >
> > > :classical_building:机构：康奈尔大学
> > >
> > > :arrow_right:领域：
> > >
> > > - Information systems → Query optimization  
> > > - Autonomous database administration
> > > - Human-centered computing → Natural language interfaces
> > >
> > > :books:概述：展示了$\lambda$-Tune模型，可根据工作负载自动化自适应为数据库系统调优
> > >
> > > - 背景
> > >   - 还是knobs的调优，目的在于根据数据库的硬件+查询负载来优化配置
> > >   - 传统调优方法依赖大量计算资源(GPU)和时间
> > > - 𝜆-Tune 的创新之处
> > >   - 利用LLMs来理解和处理文本数据，不需要额外训练(零次学习)直接生成配置建议
> > >     - 零次学习：模型在**没有见过**某类训练数据情况下，对这些类别做出正确预测
> > >   - 系统根据DB系统+硬件规格+查询负载，通过自动生成提示，生成适合的调优建议
> > >   - 采取一种工作负载压缩方法，只提取最优洞察力的工作负载特征
> > > - 𝜆-Tune 的优势
> > >   - 计算资源小：无需耗时的调优与训练(零次学习)
> > >   - 性能提升
> >
> > **:point_down:==User-friendly, Interactive, and Configurable Explanations for Graph Neural Networks with Graph Views==**  
> >
> > > :classical_building:机构：浙江大学等
> > >
> > > :arrow_right:领域：
> > >
> > > - Computing methodologies → Neural networks  
> > > - Information systems → Graph-based database models  
> > >
> > > :books:概述：介绍了名为GVEX的系统，用于为用户提供友好+可交互的GNNs行为解释
> > >
> > > - 问题背景
> > >   - GNN对图数据分析表现优异，但其具有黑箱特性，内部工作原理难以解释
> > >   - 目前对于GNNs的解释方法，仅限于对特定实例，且生成的解释结构过大(难以直观理解)
> > > - GVEX 系统的创新点
> > >   - 提供了用户友好+交互式的界面，以及个性化的配置(选择感兴趣类别/结点数量)
> > >   - 利用事实+反事实属性，以及这些节点在GNN消息传递的聚合影响，生成高质量解释子图
> > >   - 生成双层解释结构，包含图模式+解释子图
> >
> > **:point_down:==OpenIVM: a SQL-to-SQL Compiler for Incremental Computations==**  
> >
> > > :classical_building:机构：荷兰国家数学和计算机科学研究学会/滑铁卢大学
> > >
> > > :arrow_right:领域：Information systems → Database query processing  
> > >
> > > :books:概述：展示了名为OpenIVM的SQL-to-SQL编译器，专用于增量视图维护(IVM)
> > >
> > > - 增量视图维护(IVM)
> > >   - 作用：用于在基础数据插入/更新/删除时，快速更新数据库中物化视图
> > >     - 物化视图：所预测的查询结果，存储在数据库中，用于查询优化
> > >   - 现有IVM的局限：通常在独立的系统中实现IVM的计算，需要额外计算系统及资源
> > > - OpenIVM 的创新之处
> > >   - 核心理念：通过现有的SQL查询引擎执行所有IVM，而非额外系统，减少开发/计算成本
> > >   - 支持跨系统：能协调OLTP和OLAP系统工作
> > >     - OLTP(在线事务处理)：负责处理DBMS基础的表插入/更新/删除
> > >     - OLAP(在线分析处理)：存储和维护物化视图
> > >     - 二者协调的方式：OLTP将基础操作处理后，通过SQL传递给OLAP后续处理
> > > - 技术实现
> > >   - SQL编译器：
> > >     - OpenIVM将视图定义编译为SQL
> > >     - OpenIVM根据数据库的基础表变化，增量地更新物化视图 (基于DBSP增量计算原理)
> > >   - DuckDB 的集成
> > >     - DuckDB：一个轻量级数据库管理系统
> > >     - OpenIVM用DuckDB来编译/解析/转换/优化物化视图维护的逻辑
> > > - Demo Session
> > >   - OpenIVM作为DuckDB的一个扩展模块，给 DuckDB 添加 IVM 功能
> > >   - OpenIVM 在跨系统 IVM 中应用
> > >     - PostgreSQL 处理基础表的更新操作
> > >     - DuckDB 用于存储和维护这些表的物化视图
> >
> > 

# 4. Panels  

> 

# 5. Tutorials

> 

# 6. Workshop Summaries  

> 