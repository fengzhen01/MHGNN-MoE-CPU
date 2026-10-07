"""Convert the published MHGNN files into compact, portable NumPy files."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def read_embedding(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    tensor = payload["hypergraph"]["protein"]
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    result = np.asarray(tensor, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"Expected a 2-D protein embedding, got {result.shape}")
    return result


def prepare_dataset(source: Path, output: Path, dataset: str) -> dict:
    folder = source / dataset
    hsi = pd.read_excel(folder / "herb_symptom.xlsx").iloc[:, :2].to_numpy(np.int64)
    incidence_frame = pd.read_excel(folder / "hg_data.xlsx").iloc[:, :2]
    incidence = incidence_frame.to_numpy(np.int64)
    proximity_file = folder / ("HIT_Proximity.csv" if dataset == "HIT" else "TCMIO_Proximity.xlsx")
    proximity = pd.read_csv(proximity_file) if proximity_file.suffix == ".csv" else pd.read_excel(proximity_file)
    proximity_arrays = proximity[
        ["herb_node_id", "symptom_node_id", "proximity_d", "proximity_z"]
    ].to_numpy(np.float64)

    num_herbs = int(pd.read_excel(folder / "herb_target.xlsx").iloc[:, 0].max()) + 1
    num_symptoms = int(pd.read_excel(folder / "symptom_gene.xlsx").iloc[:, 0].max()) + 1
    num_hyperedges = int(incidence[:, 0].max()) + 1
    if num_hyperedges != num_herbs + num_symptoms:
        raise ValueError(
            f"{dataset}: expected {num_herbs + num_symptoms} hyperedges, got {num_hyperedges}"
        )

    np.savez_compressed(
        output / f"{dataset}.npz",
        positives_raw=hsi,
        positives_unique=np.unique(hsi, axis=0),
        incidence_edge=incidence[:, 0],
        incidence_vertex=incidence[:, 1],
        proximity_herb=proximity_arrays[:, 0].astype(np.int64),
        proximity_symptom=proximity_arrays[:, 1].astype(np.int64),
        proximity_d=proximity_arrays[:, 2].astype(np.float32),
        proximity_z=proximity_arrays[:, 3].astype(np.float32),
        num_herbs=np.asarray(num_herbs),
        num_symptoms=np.asarray(num_symptoms),
        num_hyperedges=np.asarray(num_hyperedges),
    )
    return {
        "dataset": dataset,
        "num_herbs": num_herbs,
        "num_symptoms": num_symptoms,
        "num_hyperedges": num_hyperedges,
        "positive_rows": int(len(hsi)),
        "positive_unique": int(len(np.unique(hsi, axis=0))),
        "incidences": int(len(incidence)),
        "proximity_rows": int(len(proximity_arrays)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Original MHGNN-main directory")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data_source = args.source / "data" if (args.source / "data").is_dir() else args.source
    required = [data_source / name for name in ("HIT", "TCMIO")]
    if not all(path.is_dir() for path in required):
        raise FileNotFoundError("The source directory must contain HIT/ and TCMIO/")

    feature_target = args.output / "protein_features.npy"
    ppi_target = args.output / "ppi_edges.npy"
    if args.overwrite or not feature_target.exists():
        np.save(feature_target, read_embedding(data_source / "HIT" / "initial_embeddings.pkl"))
    if args.overwrite or not ppi_target.exists():
        ppi = pd.read_excel(data_source / "HIT" / "interactome.xlsx").iloc[:, :2].to_numpy(np.int64)
        np.save(ppi_target, ppi)

    summaries = [prepare_dataset(data_source, args.output, dataset) for dataset in ("HIT", "TCMIO")]
    features = np.load(feature_target, mmap_mode="r")
    ppi = np.load(ppi_target, mmap_mode="r")
    metadata = {
        "format_version": 1,
        "num_proteins": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "ppi_edges": int(ppi.shape[0]),
        "datasets": summaries,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
