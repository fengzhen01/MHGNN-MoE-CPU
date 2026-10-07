# 结果目录使用状态

## 可用于当前正式分析

- `hit_baseline_strict_full`、`tcmio_baseline_strict_full`：严格去重随机10折MHGNN；
- `hit_moe_safe_strict_seed0`、`tcmio_moe_safe_strict_seed0`：屏蔽proximity代理特征的MoE；
- `hit_histgb_structural_strict`、`tcmio_histgb_structural_strict`：无泄漏结构特征HistGB；
- `hit_baseline_histgb_ensemble`、`tcmio_baseline_histgb_ensemble`：预设0.5/0.5融合；
- `hit_hard_baseline_seed0`、`hit_hard_moe_safe_seed0`、`hit_hard_histgb_structural_seed0`：
  网络近邻困难未标记样本实验；
- `hit_hard_baseline_histgb_ensemble_seed0`：困难样本固定均值融合；
- `audit_hit_feature_leakage`、`audit_tcmio_feature_leakage`：泄漏证据。
- `hit_baseline_strict_seed17`、`hit_baseline_strict_seed42`及对应的
  `hit_histgb_structural_*`、`hit_baseline_histgb_ensemble_*`：多随机种子复核；
- `multiseed_hit_baseline`、`multiseed_hit_hybrid`：以完整CV运行均值为单位的汇总。
- `hit_full_ranking_*`：filtered全候选排序审计；
- `hit_cold_herb_hard_seed0`、`hit_cold_symptom_hard_seed0`、
  `hit_cold_both_hard_seed0`：无泄漏冷启动困难样本结果；
- `hit_ranking_bpr_seed0`、`hit_ranking_bpr_broad_seed0`：局部困难与广覆盖BPR对照。
- `hit_ranking_bpr_broad_seed17`、`hit_ranking_bpr_broad_seed42`及对应的
  `hit_full_ranking_bpr_broad_hybrid_*`：广覆盖BPR全候选排序多种子复核；
- `multiseed_hit_full_ranking_bpr_broad_hybrid`：以完整运行宏平均为单位的全候选
  排序跨种子汇总。
- `tcmio_ranking_bpr_broad_seed0`、`tcmio_ranking_bpr_broad_seed17`、
  `tcmio_ranking_bpr_broad_seed42`及对应的`tcmio_full_ranking_bpr_broad_*`：
  TCMIO广覆盖BPR与全候选排序复核；
- `multiseed_tcmio_full_ranking_bpr_broad_hybrid`：TCMIO全候选排序跨种子汇总。
- `bootstrap_hit_ranking_ci`、`bootstrap_tcmio_ranking_ci`：以seed–药材查询簇
  为单位的全候选排序bootstrap区间与配对增益。

## 仅作诊断，不进入正式方法对比

- `hit_logistic_strict`、`hit_random_forest_strict`、`hit_hist_gb_strict`：包含
  proximity覆盖代理，满分结果无效；
- `hit_moe_strict_full*`、`tcmio_moe_strict_full*`：早期MoE可能读取proximity代理；
- `hit_confidence_strict_seed0`：置信度BCE负结果，仅作为方法筛选记录；
- `hit_nnpu_pilot_fold1`：单折先导实验，不是完整比较；
- `final_*`、`smoke_*`、`verify_*`、`v2_smoke_*`：工程验收或单折调试。

## 未完成

- `hit_cold_herb_bce_seed0`、`hit_cold_symptom_bce_seed0`、
  `hit_cold_both_bce_seed0`：在发现代理特征问题后中止，不汇报其部分折结果。

正式汇总以`PHASE3_RESULTS.md`为准。不得从诊断、冒烟或未完成目录中挑选高分替代
完整交叉验证结果。
