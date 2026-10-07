# HIT full-ranking cluster bootstrap

Bootstrap unit: seed–herb query cluster; intervals are descriptive and do not replace an external test set.

| Method | Metric | Mean | 95% CI |
|---|---|---:|---:|
| hybrid | ap | 0.4869 | [0.4715, 0.5022] |
| hybrid | mrr | 0.5585 | [0.5422, 0.5748] |
| hybrid | recall@10 | 0.6920 | [0.6763, 0.7081] |
| hybrid | ndcg@10 | 0.5421 | [0.5267, 0.5569] |
| deep | ap | 0.2686 | [0.2607, 0.2767] |
| deep | mrr | 0.3020 | [0.2929, 0.3108] |
| deep | recall@10 | 0.6383 | [0.6224, 0.6537] |
| deep | ndcg@10 | 0.3561 | [0.3462, 0.3666] |
| histgb | ap | 0.4354 | [0.4208, 0.4508] |
| histgb | mrr | 0.5186 | [0.5027, 0.5352] |
| histgb | recall@10 | 0.5924 | [0.5757, 0.6093] |
| histgb | ndcg@10 | 0.4814 | [0.4662, 0.4960] |

| Paired difference | Metric | Mean difference | 95% CI | P(draw > 0) |
|---|---|---:|---:|---:|
| hybrid_minus_deep | ap | 0.2183 | [0.2076, 0.2295] | 1.0000 |
| hybrid_minus_deep | mrr | 0.2567 | [0.2444, 0.2693] | 1.0000 |
| hybrid_minus_deep | recall@10 | 0.0538 | [0.0428, 0.0650] | 1.0000 |
| hybrid_minus_deep | ndcg@10 | 0.1860 | [0.1769, 0.1957] | 1.0000 |
| hybrid_minus_histgb | ap | 0.0516 | [0.0471, 0.0561] | 1.0000 |
| hybrid_minus_histgb | mrr | 0.0401 | [0.0354, 0.0449] | 1.0000 |
| hybrid_minus_histgb | recall@10 | 0.0995 | [0.0894, 0.1099] | 1.0000 |
| hybrid_minus_histgb | ndcg@10 | 0.0607 | [0.0560, 0.0654] | 1.0000 |
