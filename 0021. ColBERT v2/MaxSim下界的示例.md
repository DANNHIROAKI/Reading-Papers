假设我有查询，嵌入后得到了向量集{q1, q2, q3}

假设只有两个文档嵌入{d1, d2, d3, d4,} {p1, p2, p3, p4} {o1, o2, o3, o4}



ColBERT的做法

计算距离q1↔{d1, d2, d3, d4}，得到MaxSim-d1

计算距离q2↔{d1, d2, d3, d4}，得到MaxSim-d2

计算距离q3↔{d1, d2, d3, d4}，得到MaxSim-d3

最后q-d的相似得分为MaxSim-d1 + MaxSim-d2 + MaxSim-d3

计算距离q1↔{p1, p2, p3, p4}，得到MaxSim-p1

计算距离q2↔{p1, p2, p3, p4}，得到MaxSim-p2

计算距离q3↔{p1, p2, p3, p4}，得到MaxSim-p3

最后q-p的相似得分为MaxSim-p1 + MaxSim-p2 + MaxSim-p3

计算距离q1↔{o1, p2, o3, p4}，得到MaxSim-p1

计算距离q2↔{o1, p2, o3, p4}，得到MaxSim-p2

计算距离q3↔{o1, o2, o3, p4}，得到MaxSim-p3

最后q-p的相似得分为MaxSim-o1 + MaxSim-o2 + MaxSim-o3

对d/p/o三者的得分进行重排，得到最相似的文档



本文的初排

先经过候选过程，找到与q1, q2, q3相关的候选集

- 此处假设候选集为{d2, d3, d4, p1, p2, o1}，源于一个假设，即经过候选后的嵌入可以只包含某个文档的部分嵌入

计算距离

计算距离q1↔{d2, d3, d4, p1, p2, o1}

计算距离q2↔{d2, d3, d4, p1, p2, o1}

计算距离q3↔{d2, d3, d4, p1, p2, o1}

分组

文档d对应的组：

距离q1↔{d2, d3, d4}，取距离最大值作为近似MaxSim，近似值一定小于实际值MaxSim-d1

距离q2↔{d2, d3, d4}，取距离最大值作为近似MaxSim，近似值一定小于实际值MaxSim-d2

距离q3↔{d2, d3, d4}，取距离最大值作为近似MaxSim，近似值一定小于实际值MaxSim-d3

将3个近似的MaxSim相加作为文档d的得分，也一定小于实际值

文档p对应的组

距离q1↔{p1, p2}，取距离最大值作为近似MaxSim，近似值一定小于实际值MaxSim-q1

距离q2↔{p1, p2}，取距离最大值作为近似MaxSim，近似值一定小于实际值MaxSim-q2

距离q3↔{p1, p2}，取距离最大值作为近似MaxSim，近似值一定小于实际值MaxSim-q3

将3个近似的MaxSim相加作为文档p的得分，也一定小于实际值

文档o对应的组

距离q1↔{o1}，取距离最大值作为近似MaxSim，近似值一定小于实际值MaxSim-o1

距离q2↔{o1}，取距离最大值作为近似MaxSim，近似值一定小于实际值MaxSim-o2

距离q3↔{o1}，取距离最大值作为近似MaxSim，近似值一定小于实际值MaxSim-o3

将3个近似的MaxSim相加作为文档p的得分，也一定小于实际值

将dpo三者的近似MaxSim排序，即初排结果，假设为d>p>o



本文的重排

选取初排结果的前若干个，例如此处为两个，即选取d和p

对d和p进行重排

加载d/p的所有精确嵌入：{d1, d2, d3, d4,} {p1, p2, p3, p4}

按照ColBERT模式计算MaxSim

计算距离q1↔{d1, d2, d3, d4}，得到MaxSim-d1

计算距离q2↔{d1, d2, d3, d4}，得到MaxSim-d2

计算距离q3↔{d1, d2, d3, d4}，得到MaxSim-d3

最后q-d的相似得分为MaxSim-d1 + MaxSim-d2 + MaxSim-d3

计算距离q1↔{p1, p2, p3, p4}，得到MaxSim-p1

计算距离q2↔{p1, p2, p3, p4}，得到MaxSim-p2

计算距离q3↔{p1, p2, p3, p4}，得到MaxSim-p3

最后q-p的相似得分为MaxSim-p1 + MaxSim-p2 + MaxSim-p3

再将d/p的精确相似度排名，例如p>d，则p为最相似文档