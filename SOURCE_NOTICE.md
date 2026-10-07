# 数据与方法来源说明

本项目使用 MHGNN 公开项目所提供的 HIT、TCMIO、PPI 与蛋白质嵌入数据，并参考其网络传播、超图传播和药材–症状关联分类流程重新实现了不依赖 DHG/DGL 的 CPU 版本。请在使用数据或开展派生研究时引用原论文：

Liang, X., Lin, T., Xie, B., Tang, Y., & Wang, W. “MHGNN: Multiplex Hypergraph Neural Networks for Predicting Herb–Symptom Interactions.” *IEEE Transactions on Neural Networks and Learning Systems*, 2026. DOI: 10.1109/TNNLS.2026.3677056.

本项目新增内容包括可移植数据转换、严格去重划分、可审计负采样、纯 PyTorch CPU 实现、残差式结构专家 MoE、环境记录以及迁移脚本。数据的使用与再分发仍应遵循原始数据源和公开仓库的授权条件；用于论文时，应明确区分原始 MHGNN、本文重实现基线与 MoE 改造结果。
