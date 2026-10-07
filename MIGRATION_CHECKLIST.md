# CPU 机器迁移检查清单

## 复制前

- 保留整个 `MHGNN-MoE-CPU` 目录；
- 确认 `data/processed` 下有 5 个数据文件；
- 建议核对 `SHA256SUMS.txt`；
- 不复制旧机器的 `.venv`，在目标机重新创建环境；
- 若目标机离线，提前生成并一并复制 `wheelhouse`。

## 目标机要求

- 64 位 Windows 10/11 或常见 x86-64 Linux；
- Python 3.10（推荐）或兼容的 3.9–3.11；
- 建议内存不少于 8 GB，磁盘可用空间不少于 3 GB；
- 不要求 CUDA、显卡、DHG 或 DGL。

## 安装后验收

1. `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 能输出版本且 CUDA 为 `False`；
2. `python -m mhgnn_moe.audit --data-dir data/processed` 输出 `AUDIT PASS`；
3. 冒烟测试生成 baseline 和 MoE 两个结果目录；
4. 每个目录均包含配置、划分、指标、预测和检查点；
5. MoE 预测文件中的三个专家平均利用率均非零。
6. `python -m unittest discover -s tests -v`的5项测试全部通过；
7. 已有正式基线时，执行`audit_feature_leakage.py`并确认正式模型配置中的
   `allow_proximity_router`为`false`；
8. 运行HistGB或融合实验时，确认`summary.json`中的
   `leaky_proximity_included`为`false`。

## 常见问题

- PowerShell 禁止脚本：先执行 `Set-ExecutionPolicy -Scope Process Bypass`；
- 找不到模块：确认当前目录是项目根目录，并使用 `.venv` 内的 Python；
- 安装慢：使用离线 wheelhouse，或确认没有误装 CUDA 版 PyTorch；
- 内存不足：保持 `device=cpu`，关闭其他大型程序；模型是全图训练，不能用减小 batch size 解决；
- 结果有差异：首先核对配置快照、Python/PyTorch 版本、线程数、划分协议和随机种子。
