# MHGNN-MoE-CPU

MHGNN-MoE-CPU is a **CPU-portable, leakage-aware reproduction and extension** for herb–symptom association prediction. It provides a pure PyTorch implementation of MHGNN-style protein/PPI and hypergraph propagation, strict duplicate-aware evaluation, pairwise full-candidate ranking, residual mixture-of-experts (MoE), and a fixed fusion with explicit target-set statistics.

> The project supports research and candidate association discovery. It does not provide clinical diagnosis, prescription, or treatment recommendations.

## Main contributions

- CPU-only implementation without DHG, DGL, PyTorch Geometric, or GPU requirements;
- strict pair-level deduplication and auditable cross-validation splits;
- explicit diagnosis of proximity-derived label proxies;
- BPR learning with broad unlabeled sampling for full-candidate ranking;
- MHGNN-style deep representation plus HistGradientBoosting target-statistics branch;
- fixed equal-weight fusion without test-fold weight search;
- multi-seed evaluation and query-cluster bootstrap confidence intervals;
- cold-herb, cold-symptom, and double-cold diagnostic protocols;
- optional residual MoE for controlled architecture experiments.

## Data

The processed HIT and TCMIO arrays are included because the complete processed package is approximately 43 MB and no individual file exceeds GitHub's 100 MB file limit.

| Dataset | Herbs | Symptoms | Unique positive associations | Hyperedges |
|---|---:|---:|---:|---:|
| HIT | 798 | 174 | 1,436 | 972 |
| TCMIO | 461 | 174 | 1,327 | 635 |

The repository also includes 18,505 protein feature vectors and 327,924 PPI edges used by the propagation module. See [DATA.md](DATA.md) and [SOURCE_NOTICE.md](SOURCE_NOTICE.md) for provenance and use boundaries.

## Repository layout

```text
MHGNN-MoE-CPU/
├── mhgnn_moe/        # data loaders, models, losses, metrics, and runner
├── configs/          # baseline, MoE, BPR, and cold-start configurations
├── data/processed/   # portable NumPy arrays for HIT, TCMIO, PPI, and proteins
├── scripts/          # Windows/Linux setup and experiment scripts
├── tests/            # implementation and protocol checks
└── results/          # compact authoritative summaries; checkpoints excluded
```

## Installation

Using Conda:

```bash
conda env create -f environment-cpu.yml
conda activate mhgnn-moe-cpu
```

Or create a Python 3.10 environment and install the CPU build of PyTorch followed by:

```bash
pip install -r requirements.txt
```

Windows helpers:

```powershell
.\scripts\setup_windows_cpu.ps1
.\scripts\smoke_windows.ps1
```

Linux helpers:

```bash
chmod +x scripts/*.sh
./scripts/setup_linux_cpu.sh
./scripts/smoke_linux.sh
```

## Quick start

Audit the processed data:

```bash
python -m mhgnn_moe.audit --data-dir data/processed
```

Run a strict baseline:

```bash
python -m mhgnn_moe.run --config configs/hit_baseline.json --run-name hit_baseline_strict
```

Run the multi-seed full-candidate ranking study:

```powershell
.\scripts\run_ranking_multiseed_windows.ps1
```

```bash
./scripts/run_ranking_multiseed_linux.sh
```

## Main full-candidate ranking results

The primary ranking results use seeds 0, 17, and 42, five-fold cross-validation, one positive association with five broad unlabeled samples for BPR, and a fixed 0.5/0.5 fusion of the deep and target-statistics branches.

| Dataset | Macro-AP | MRR | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| HIT | 0.4869±0.0045 | 0.5585±0.0052 | 0.6921±0.0092 | 0.5419±0.0065 |
| TCMIO | 0.5007±0.0010 | 0.5709±0.0037 | 0.7266±0.0096 | 0.5606±0.0013 |

Query-cluster bootstrap estimated the fusion-minus-HistGB Macro-AP gain as:

- HIT: `0.0516`, 95% CI `[0.0471, 0.0561]`;
- TCMIO: `0.0448`, 95% CI `[0.0408, 0.0491]`.

These are internal cross-validation estimates and do not replace external or temporal validation.

## Leakage and reporting boundaries

- HIT duplicate positive rows are removed before splitting.
- Proximity coverage is a strong label proxy in the public preprocessing and is excluded from formal model input.
- Balanced-sampling AUPRC is not treated as equivalent to full-candidate Macro-AP.
- Fixed fusion weights are chosen in advance, not optimized on test folds.
- MoE improvements are small and should be reported with all folds, seeds, and paired comparisons.
- Unobserved herb–symptom pairs are unlabeled candidates, not confirmed biological negatives.

## Citation and license status

Please cite the original MHGNN paper listed in [SOURCE_NOTICE.md](SOURCE_NOTICE.md) when using the processed data or MHGNN-style propagation. No open-source software license has been selected yet; contact the repository owner before redistributing modified code.
