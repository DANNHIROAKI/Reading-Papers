# 1. Intro

:one:基本概念

1. 基数估计：关系数据库中，预测查询结果的大小(返回多少行记录)
2. 成本估计：预测执行特定查询，需要消耗多少计算资源
3. 谓词：例如`where age > 60`中，`age > 60`就是谓词

:two:回顾一些过往方法

1. 传统基数估计器：基于统计信息(如直方图)，假如某数据库中简历了关于年龄的直方图，当查询`where age > 60`时就会返回1000的基数估计

   |  Age  | Num  |
   | :---: | :--: |
   | 0-30  | 1000 |
   | 30-60 | 2000 |
   | 60-90 | 1000 |

2. 采样方法：从数据库中抽取一部分样本，来估计查询结果的基数

3. 深度学习方法：学习数据分布和查询模式之间的复杂关系，从而得到预估的基数

:three:现有模型的局限性

1. MSCN
   - 原理：将查询拆分为多个集合(表/连接条件/过滤条件)，再用特殊神经网络处理这些数据
   - 缺点：需要大量的样本来训练，无法捕捉复杂数据的模式，泛化能力有限
2. ALECE
   - 原理：通过注意力机制获得每个谓词的重要性权重
   - 缺点：注意力机制依赖位置编码，导致相同查询(不同排序)的基数估计不同
3. FACE
   - 原理：类似于一个语言模型，来预测下一个谓词的影响
   - 缺陷：存在谓词顺序依赖，要求数据中各列独立(实际不可能)
4. PRICE
   - 原理：通过高维联合概率，从而来估计基数
   - 缺陷：存在谓词顺序依赖，具有维度灾难

:four:RLGCNt基数估计模型架构

1. 引入了LSH技术：使得能快速找到相似数据，缓解了维度灾难
2. 高斯核函数：将原始数据映射到高维特征空间，即$K(x,y) = \exp^{-\frac{\|x-y\|^2}{2\varepsilon^2}}$使得非线性关系变为线性关系，用于捕捉数据间非线性关系
3. 引入注意力机制：能动态调整，以重点关注对基数估计最重要的谓词

:five:基于秩高斯变换的查询编码方法

1. 传统方法：ALIECE使用独热编码，在数据分布不均时效果极差
2. 秩高斯：先将数据按照值大小排序分配排名，将排名通过高斯函数映射到正态分布，最终得到归一化值
   - 一方面保留了数据项之间的大小关系
   - 另一方面无论数据多么偏斜，转换后都接近正态分布

:six:因果注意力机制

1. 传统模型：对谓词的顺序敏感，即相同条件不同为此顺序，得到的基数估值也不同
2. 引入因果掩码：控制信息的流动方向，减少复杂度，捕捉谓词因果性(而非相关性)，抹去顺序敏感

:seven:实验性能

1. 动态工作负载测试：RLGCNt比基线模型提高了10.7%的准确率
2. 静态工作负载测试：RLGCNt比基线模型提高了5.8%的准确率

# 2. 相关工作

:one:两种模型及其代表方法

1. 查询驱动的模型：直接学习查询→结果大小的映射关系，以查询特征(谓词/条件/连接)作为输入，将查询编码后通过神经网络输出基数估计值
   - MSCN模型：分解查询为特征集(表/连接条件/过滤条件)，输入卷积从而输出基数估计
   - Deep Sketches模型：也使用卷积，但采用物化样本(预计算的查询结果)辅助估计，以及通过草图结构(大规模数据的少量关键特征)压缩查询特征
   - DBEst模型：同样使用卷积，但结合了多种编码方式表示查询特征(对于不同类型数据Adaptive优化)
   - NNGP模型：将神经网络与高斯过程结合，用神经网络提取查询特征+高斯过程建模查询结果的概率分布
2. 数据驱动的模型：先对数据分布进行建模，理解底层数据分布，再基于数据分布模型预测基数估计值
   - DeepDB：基于RSPN数据结构，但泛化和处理离群值乏力
   - Neurocard：通过采样连接结果构建自回归模型，但难以处理复杂序列
     - 马可夫：序列当前值和有限个之前的值有关
     - 自回归：序列当前值和有所有之前值有关
3. 混合模型
   - ALECE：用注意力机制的MLP层，将数据和查询工作负载进行合并，来估计基数
   - PRICE：忽略了查询中数据排列对基数的影响

:two:现有模型的弊端：本文的Motivation

1. 基于深度学习的基数估计，不能很好地处理查询中的离群值
2. 基于深度学习的基数估计，难以处理数据的局部特征(不均匀数据)
3. 传统的注意力机制，忽略了查询数据(谓词)的排列
4. 动态工作负载下，以上问题会被进一步突出

# 3. 基数估计

:one:关系数据库及基数的表示

1. 令关系$R$种有$P$个元组，即表$R$中有$P$个行
2. 每行用$m$个属性去表述，即定义属性集为$\{B_1,B_2,...,.B_m\}$，每行可用$m$个属性值表示为$u=\{b_1,b_2,...,.b_m\}$
3. 定义查询条件为一个函数$Q$，元组$u$满足查询条件时为$Q(u)=1$
4. $Q$的基数$\text{card}(Q)$用于表示满足查询$Q$的元组数量，$Q$的选择性$\text{sel}(Q)$表示满足查询条件$Q$的元组的比率

:two:RLGCNt概览

<img src="https://raw.githubusercontent.com/DANNHIROAKI/New-Picture-Bed/main/img/0195b83e-3266-7865-aa76-bf1a6ae5de97_4.jpg" style="zoom:60%;" />   

:one:数据输入

1. 对于原始数据：将原始数据处理成直方图($h_1,h_2,h_3,...$)

2. 对于查询负载

   - 提取表连接条件：表连接条件SJCB(如`A.id=B.userid`)，属性范围SARC(如`age>18`)，查询涉及的相关列表BRTL
   - 高斯排序转换：对查询进行排序，归一化缩放，通过CDF计算每个查询的百分位数，进一步将数据归一化到0-1
   

:two:特征增强与相似性提取 


3. 将二者合并为一个大的特征空间，输入到LSH中，让特征空间中临近的点分到同一桶

4. 进一步用高斯核函数(GKF)接收LSH输出，进一步捕捉数据相似性，输出KQV等注意力参数

:two:注意力机制：数据经过LSH和GKF处理后输入注意力层

1. 多头自注意力机制：应用因果掩码控制信息流向，处理数据内部上下文关系
2. 交叉注意力机制：同样使用因果掩码，建立不同特征表示间的关系

:three:输出：

1. 用单个MLP处理输出，再通过一个多个MLP构建的Reg对象得到最终基数
2. Reg被集成到了损失函数中，用于调整注意力机制的权值

# 4. 基于注意力机制的RLGCNt

## 4.1. 秩高斯变换编码

:one:独热编码存在严重局限：处理大规模数据时向量非常稀疏，无法表达两值的相对关系，对离群值敏感，数据分布极其不均

:two:秩高斯变换原理

1. 输入：表连接条件SJCB(如`A.id=B.userid`)，属性范围SARC(如`age>18`)，查询涉及的相关列表BRTL

   - 后续要进行排序和处理的是查询条件中每个属性的值，而不是谓词本身

   - 例如处理如下查询

     ```sql
     SELECT * FROM employees WHERE age > 25 AND salary < 10000
     SELECT * FROM employees WHERE age > 30 AND salary < 20000
     SELECT * FROM employees WHERE age > 30 AND salary < 30000
     SELECT * FROM employees WHERE age > 40 AND salary < 40000
     SELECT * FROM employees WHERE age > 20 AND salary < 50000
     SELECT * FROM employees WHERE age > 50 AND salary < 60000
     SELECT * FROM employees WHERE age > 30 AND salary < 70000
     ```

   - 参与Age属性排名的就是[25, 30, 30, 40, 20, 50, 30]

2. 排序：对查询数据中的每个元素进行排序，将每个元素转换为排名，如果重复了则都取最小排名

   ```txt
   年龄字段: [25, 30, 30, 40, 20, 50, 30]
   排序之后: [20, 25, 30, 30, 30, 40, 50]
   排名代之: [01, 02, 03, 03, 03, 06, 07]
   回归原序: [02, 03, 03, 06, 01, 07, 03]
   ```

3. 缩放：将排名归一化到[0,1]区间，例如这里有七个数据即ng=7，除以8以归一化

   ```txt
   归一化后: [0.250, 0.375, 0.375, 0.750, 0.125, 0.875, 0.375]
   ```

4. 正态变换：将归一化的[0,1]区间映射到正态分布，具体来说是用$\Phi^{-1}$变换

   - 对于$\Phi(x_0)=p$，意思是正太分布中$x<x_0$的概率是$p$
   - 对于$\Phi^{-1}(p)=x_0$，意思是给定一个概率$p$，反过来找到满足这个概率分位的值

## 4.2. LSH

:one:LSH和注意力机制结合的好处：让相似的查询被分到同一个桶中，计算注意力权重时只需在桶内进行，而非所有组合，从而加速计算

- 但是这里缺乏复杂度的分析，比如是否能将$O(N^2)$的复杂度降低到$O(N\log N)$
- 参考Reformer模型，它也提出了与本文类似的基于LSH对注意力机制进行改进方法，但是对复杂度进行了详细的分析

:two:输入的特征

1. 注意力查询特征Q：就是通过秩高斯变换编码的查询条件
2. 注意力的K/V特征：来自数据表的特征
3. 将这些特征合并为tz作为输入特征，这些特征是LSH分桶的直接对象

:three:LSH的实现：

1. 创建哈希函数：预定义nl个哈希函数，每个函数由权重矩阵W+偏置向量b组成
   - 处理的特征是tz合并特征向量
   - 每个tz特征会被nl个哈希函数处理nl次，得到nl个哈希值
2. 计算哈希值：对于每个哈希函数i，计算输入特征tz与权重矩阵的点积，加上偏置，最后用ReLU激活
   - $\text{hash\_value}_i\text{=}\text{ReLU}(tz\times W_i+b_i)$
3. 连接哈希值：将所有计算得到的哈希值沿着最后一个维度拼接起来，得到一个大特征向量，作为该tz的最终哈希值
   - 沿最后一个维度是什么意思？举个例子加入每个hash_value的张量形状为[64,10,16]
   - 有n个这样的张量，连接后就是[64,10,16n] 

## 4.3. 高斯核函数

:one:输入和输出

1. 输入：上一步LSH输出的哈希特征向量
2. 输出：相似度矩阵，用于衡量输入的LSH特征向量中不同部分间的相似度
3. 目的：基于欧几里得距离量化特征之间的相似性，捕捉局部结构，处理非线性关系，减少噪声

:two:具体实现：

1. 对LSH步骤输出的哈希特征向量进行扩展
   - 对原始特征向量$x$进行两个扩展得到$x_1,x_2$，假设原来$x$形状为64×16，则$x_1$为64×16×1则$x_2$为64×1×16
   - 这种结构就是为了计算$x$中每两个元素间欧几里得距离而设计的
2. 计算欧几里得距离$S = {\left( {x}_{1,d} - {x}_{2,d}\right) }^{2}$，维度为64×16×16
3. 只需在$S$的指定维度求和$\text{squared\_distance} = \mathop{\sum }\limits_{0}^{D}S \tag{11}$，就可得到每对特征向量之间的平方欧几里得距离，由此构成相似度矩阵
4. 高斯核函数：将上一步得到的相似度矩阵的每个元素应用高斯核函数，即$\text{ Gaussian Kernel } = \exp \left( {-\frac{1}{2} \cdot  \frac{\text{ squared\_distance }}{{\sigma }^{2}}}\right)$，目的是将相似度映射到[0,1]间

## 4.4. 因果注意力机制

:one:黑箱分析

1. 一种对传统多头注意力机制的改进，通过因果掩码，模型在计算注意力时只关注过去和当前数据，避免了对未来数据的依赖。
2. 输入：就是上一步高斯核函数输出的相似度矩阵，是查询和数据特征之间的相似度矩阵
3. 操作：
   - 将特征矩阵分解为Q，进一步线性变换出K/V值
   - 通过因果掩码调整注意力得分，确保未来位置的权重为0
   - 将注意力输出反馈到原始特征矩阵，得到更新后的特征矩阵
4. 输出：因果掩码矩阵，以及更新后的特征矩阵

:two:实现细节

1. 提取查询特征向量：从特征矩阵中，提取特定的行，作为查询特征向量$Q_p$
2. 分解特征向量为键值对和查询：通过线性变换将特征矩阵转化为$K_p/V_p$
3. 生成因果掩码：创建一个全为1的 $ m \times m $ 矩阵，然后按照${\mathbf{M}}_{cj} = \left\{  \begin{array}{ll} 1, & \text{ if }c \geq  j \\  0, & \text{ if }c < j \end{array}\right.$将其转换为上三角矩阵
4. 最后应用因果掩码注意力$\operatorname{Attention}\left( {{Q}_{p},{K}_{p},{V}_{p}}\right)  = \operatorname{softmax}\left( {\frac{{Q}_{p}{K}_{p}^{\top }}{\sqrt{{d}_{k}}} + M}\right) {V}_{p}$ 
5. 反馈到原始特征：将掩码和注意力输出集成回原始特征矩阵，输出更新后的特征矩阵

