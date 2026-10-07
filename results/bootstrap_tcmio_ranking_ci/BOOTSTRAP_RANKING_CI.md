# TCMIO full-ranking cluster bootstrap

Bootstrap unit: seed–herb query cluster; intervals are descriptive and do not replace an external test set.

| Method | Metric | Mean | 95% CI |
|---|---|---:|---:|
| hybrid | ap | 0.5006 | [0.4853, 0.5162] |
| hybrid | mrr | 0.5706 | [0.5533, 0.5870] |
| hybrid | recall@10 | 0.7267 | [0.7104, 0.7426] |
| hybrid | ndcg@10 | 0.5606 | [0.5454, 0.5765] |
| deep | ap | 0.2875 | [0.2792, 0.2961] |
| deep | mrr | 0.3243 | [0.3151, 0.3339] |
| deep | recall@10 | 0.6898 | [0.6736, 0.7063] |
| deep | ndcg@10 | 0.3855 | [0.3754, 0.3959] |
| histgb | ap | 0.4560 | [0.4408, 0.4713] |
| histgb | mrr | 0.5381 | [0.5210, 0.5543] |
| histgb | recall@10 | 0.6070 | [0.5895, 0.6243] |
| histgb | ndcg@10 | 0.5005 | [0.4842, 0.5159] |

| Paired difference | Metric | Mean difference | 95% CI | P(draw > 0) |
|---|---|---:|---:|---:|
| hybrid_minus_deep | ap | 0.2131 | [0.2018, 0.2251] | 1.0000 |
| hybrid_minus_deep | mrr | 0.2464 | [0.2336, 0.2591] | 1.0000 |
| hybrid_minus_deep | recall@10 | 0.0369 | [0.0239, 0.0499] | 1.0000 |
| hybrid_minus_deep | ndcg@10 | 0.1749 | [0.1651, 0.1847] | 1.0000 |
| hybrid_minus_histgb | ap | 0.0448 | [0.0408, 0.0491] | 1.0000 |
| hybrid_minus_histgb | mrr | 0.0327 | [0.0284, 0.0371] | 1.0000 |
| hybrid_minus_histgb | recall@10 | 0.1196 | [0.1084, 0.1310] | 1.0000 |
| hybrid_minus_histgb | ndcg@10 | 0.0599 | [0.0549, 0.0646] | 1.0000 |
