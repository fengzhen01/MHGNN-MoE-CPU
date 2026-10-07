"""Prepare PTM knowledge as an auditable external evidence table.

The generated files are not consumed by the model until a reviewed Chinese
symptom-to-UMLS crosswalk and fold-wise leakage controls are available.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import pandas as pd


FILES = (
    "herb_symptom_knowledge.txt",
    "herb_symptom_db_knowledge.txt",
    "symptom_herb_tcm_mesh.txt",
    "symptom_herb_tcm_mesh_long.txt",
    "symptom_herb_tcm_mesh_cooccur.txt",
)


def normalize(value: str) -> str:
    return re.sub(r"[\s·•・（）()\-—_/]", "", value.strip())


def read_knowledge(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t", 1)
        if not fields[0].strip() or len(fields) != 2:
            continue
        if path.name.startswith("symptom_herb"):
            symptom = fields[0].strip()
            pairs.extend((herb, symptom) for herb in fields[1].split() if herb)
        else:
            herb = fields[0].strip()
            pairs.extend((herb, symptom) for symptom in fields[1].split() if symptom)
    return pairs


def load_hit_names(mhgnn_root: Path) -> pd.DataFrame:
    folder = mhgnn_root / "data" / "HIT" / "original_data"
    ids = pd.read_excel(folder / "herb.xlsx")
    names = pd.read_csv(folder / "herb_chinese.csv", encoding="gb18030")
    result = ids.merge(names, on="tcm_id", how="left")
    result["normalized_herb"] = result["herb_name"].astype(str).map(normalize)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptm-root", type=Path, required=True)
    parser.add_argument("--mhgnn-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/external_ptm"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    evidence: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    source_rows = {}
    for name in FILES:
        pairs = read_knowledge(args.ptm_root / "data" / name)
        source_rows[name] = len(pairs)
        for herb, symptom in pairs:
            evidence[(normalize(herb), normalize(symptom))].add(name)
    rows = [
        {
            "herb_name": herb,
            "symptom_name": symptom,
            "source_count": len(sources),
            "sources": "|".join(sorted(sources)),
        }
        for (herb, symptom), sources in sorted(evidence.items())
    ]
    knowledge = pd.DataFrame(rows)
    knowledge.to_csv(args.output / "ptm_unique_knowledge.csv", index=False, encoding="utf-8-sig")

    hit = load_hit_names(args.mhgnn_root)
    herbs = knowledge[["herb_name"]].drop_duplicates().rename(columns={"herb_name": "normalized_herb"})
    crosswalk = herbs.merge(
        hit[["node_id", "tcm_id", "herb_name", "normalized_herb"]], on="normalized_herb", how="left"
    )
    crosswalk["match_status"] = crosswalk["node_id"].notna().map({True: "exact_normalized", False: "unmatched"})
    crosswalk["reviewed"] = False
    crosswalk.to_csv(args.output / "ptm_to_hit_herb_crosswalk.csv", index=False, encoding="utf-8-sig")

    ptm_symptoms = pd.DataFrame({"ptm_symptom_name": sorted(knowledge["symptom_name"].unique())})
    ptm_symptoms["umls_cui"] = ""
    ptm_symptoms["umls_preferred_name"] = ""
    ptm_symptoms["match_type"] = "unreviewed"
    ptm_symptoms["reviewed"] = False
    ptm_symptoms.to_csv(args.output / "symptom_crosswalk_template.csv", index=False, encoding="utf-8-sig")

    umls = pd.read_excel(args.mhgnn_root / "data" / "HIT" / "original_data" / "symptom.xlsx")
    umls.to_csv(args.output / "mhgnn_hit_umls_symptoms.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "unique_knowledge_pairs": len(knowledge),
        "unique_ptm_herbs": int(knowledge["herb_name"].nunique()),
        "unique_ptm_symptoms": int(knowledge["symptom_name"].nunique()),
        "hit_exact_herb_matches": int(crosswalk["node_id"].notna().sum()),
        "source_rows": source_rows,
        "model_ready": False,
        "blocking_requirement": "Reviewed Chinese symptom-to-UMLS mapping and fold-wise label masking",
        "license_note": "PTM README states CKCEST data are for research use only.",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
