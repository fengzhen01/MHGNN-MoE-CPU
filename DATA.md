# Data provenance and redistribution notes

## Included data

`data/processed/` contains the portable arrays required to run the project on a CPU-only machine:

- `HIT.npz` and `TCMIO.npz`: herbs, symptoms, positive associations, and hypergraph incidences;
- `protein_features.npy`: 18,505 protein embeddings with 512 dimensions;
- `ppi_edges.npy`: 327,924 protein–protein interaction edges;
- `metadata.json`: shapes, counts, and format version.

The processed data package is approximately 43 MB; its largest file is approximately 36 MB.

## Provenance

The arrays were converted from the public data released with the MHGNN study:

Liang, X., Lin, T., Xie, B., Tang, Y., and Wang, W. “MHGNN: Multiplex Hypergraph Neural Networks for Predicting Herb–Symptom Interactions.” *IEEE Transactions on Neural Networks and Learning Systems*, 2026. DOI: `10.1109/TNNLS.2026.3677056`.

Users must also comply with the terms of the underlying HIT, TCMIO, PPI, and protein-feature sources. Inclusion in this repository does not supersede original ownership or licensing conditions.

## Important exclusions

- Proximity coverage and related fields are not formal predictive inputs because their missingness strongly tracks the sampled label.
- Files under `data/external_ptm/` are exploratory and marked non-model-ready until symptom mappings and fold-wise leakage controls are completed.
- Unobserved herb–symptom pairs are treated as unlabeled candidates, not verified negatives.
