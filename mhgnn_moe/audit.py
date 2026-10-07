"""Check processed data and split integrity before expensive training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    problems = []
    for dataset in ("HIT", "TCMIO"):
        payload = np.load(args.data_dir / f"{dataset}.npz")
        raw = payload["positives_raw"]
        unique = payload["positives_unique"]
        duplicate_count = len(raw) - len(unique)
        if dataset == "HIT" and duplicate_count != 44:
            problems.append(f"Expected 44 HIT duplicate rows, observed {duplicate_count}")
        print(
            f"{dataset}: positive_rows={len(raw)} unique_pairs={len(unique)} "
            f"duplicates={duplicate_count} incidences={len(payload['incidence_edge'])}"
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit("; ".join(problems))
    print("AUDIT PASS")


if __name__ == "__main__":
    main()

