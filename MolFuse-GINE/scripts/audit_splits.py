from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molfuse_gine.data import make_splits


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _scaffold_label(smiles: str, acyclic_unique: bool = True) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return scaffold
    if acyclic_unique:
        return f"acyclic:{Chem.MolToSmiles(mol, canonical=True)}"
    return ""


def _target_stats(df: pd.DataFrame, target_col: str) -> dict:
    y = pd.to_numeric(df[target_col], errors="coerce")
    return {
        "n": int(len(df)),
        "target_mean": float(y.mean()),
        "target_std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        "target_min": float(y.min()),
        "target_max": float(y.max()),
    }


def _split_summary(
    df: pd.DataFrame,
    split_name: str,
    smiles_col: str,
    target_col: str,
) -> dict:
    scaffolds = [_scaffold_label(smi, acyclic_unique=True) for smi in df[smiles_col].astype(str)]
    murcko_empty = [_scaffold_label(smi, acyclic_unique=False) == "" for smi in df[smiles_col].astype(str)]
    counts = Counter(scaffolds)
    target = _target_stats(df, target_col)
    return {
        "split": split_name,
        **target,
        "source_index_min": int(df["source_index"].min()) if "source_index" in df and len(df) else None,
        "source_index_max": int(df["source_index"].max()) if "source_index" in df and len(df) else None,
        "n_scaffolds": int(len(counts)),
        "largest_scaffold_size": int(max(counts.values(), default=0)),
        "acyclic_fraction": float(sum(murcko_empty) / len(df)) if len(df) else 0.0,
    }


def _source_index_stats(path: Path) -> dict:
    if not path.exists():
        return {"values": set(), "row_count": 0, "unique_count": 0, "duplicate_count": 0}
    values = pd.read_csv(path)["source_index"].astype(int)
    return {
        "values": set(values),
        "row_count": int(len(values)),
        "unique_count": int(values.nunique()),
        "duplicate_count": int(values.duplicated().sum()),
    }


def _canonical_to_splits(split_frames: dict[str, pd.DataFrame], smiles_col: str) -> pd.DataFrame:
    rows = []
    mapping: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for split, frame in split_frames.items():
        for smi in frame[smiles_col].astype(str):
            mapping.setdefault(smi, set()).add(split)
            counts[smi] = counts.get(smi, 0) + 1
    for smiles, splits in sorted(mapping.items()):
        if counts[smiles] > 1:
            rows.append(
                {
                    "canonical_smiles": smiles,
                    "n_rows": counts[smiles],
                    "n_splits": len(splits),
                    "splits": ",".join(sorted(splits)),
                    "cross_split_duplicate": len(splits) > 1,
                }
            )
    return pd.DataFrame(rows)


TARGET_COLUMNS = [
    "config_name",
    "seed",
    "split",
    "n",
    "target_mean",
    "target_std",
    "target_min",
    "target_max",
    "source_index_min",
    "source_index_max",
    "n_scaffolds",
    "largest_scaffold_size",
    "acyclic_fraction",
]
SCAFFOLD_COLUMNS = ["config_name", "seed", "split", "scaffold", "size"]
ACYCLIC_COLUMNS = ["config_name", "seed", "split", "acyclic_fraction", "n"]
DUPLICATE_COLUMNS = ["config_name", "seed", "canonical_smiles", "n_rows", "n_splits", "splits", "cross_split_duplicate"]
ACTUAL_COLUMNS = [
    "config_name",
    "seed",
    "split",
    "actual_split_dir",
    "actual_exists",
    "generated_n",
    "generated_unique_n",
    "generated_duplicate_n",
    "actual_n",
    "actual_unique_n",
    "actual_duplicate_n",
    "source_index_match",
    "missing_from_actual_n",
    "extra_in_actual_n",
]
OVERLAP_COLUMNS = ["config_name", "left_seed", "right_seed", "test_overlap_n", "test_jaccard"]


def audit_config(
    config_path: Path,
    out_dir: Path,
    seeds: list[int],
    actual_split_template: str | None = None,
) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    cfg_name = config_path.stem
    rows: list[dict] = []
    scaffold_rows: list[dict] = []
    target_rows: list[dict] = []
    acyclic_rows: list[dict] = []
    duplicate_rows: list[dict] = []
    actual_rows: list[dict] = []
    test_sets: dict[int, set[int]] = {}
    cfg_out = out_dir / cfg_name
    cfg_out.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        split_dir = cfg_out / "split_files" / f"seed{seed}"
        paths = make_splits(
            csv_path=data["csv"],
            out_dir=split_dir,
            smiles_col=data["smiles_col"],
            split=data["split"],
            train_ratio=float(data["train_ratio"]),
            val_ratio=float(data["val_ratio"]),
            seed=seed,
        )
        split_frames = {name: pd.read_csv(path) for name, path in paths.items()}
        dup_df = _canonical_to_splits(split_frames, data["smiles_col"])
        if not dup_df.empty:
            dup_df.insert(0, "seed", seed)
            dup_df.insert(0, "config_name", cfg_name)
            duplicate_rows.extend(dup_df.to_dict("records"))
        if actual_split_template:
            actual_dir = Path(actual_split_template.format(config_name=cfg_name, name=cfg_name, seed=seed))
            for split, generated_path in paths.items():
                generated_stats = _source_index_stats(generated_path)
                generated = generated_stats["values"]
                actual_path = actual_dir / f"{split}.csv"
                actual_stats = _source_index_stats(actual_path)
                actual = actual_stats["values"]
                actual_exists = actual_path.exists()
                source_index_match = (
                    actual_exists
                    and generated == actual
                    and generated_stats["row_count"] == actual_stats["row_count"]
                    and generated_stats["duplicate_count"] == 0
                    and actual_stats["duplicate_count"] == 0
                )
                actual_rows.append(
                    {
                        "config_name": cfg_name,
                        "seed": seed,
                        "split": split,
                        "actual_split_dir": str(actual_dir),
                        "actual_exists": actual_exists,
                        "generated_n": generated_stats["row_count"],
                        "generated_unique_n": generated_stats["unique_count"],
                        "generated_duplicate_n": generated_stats["duplicate_count"],
                        "actual_n": actual_stats["row_count"],
                        "actual_unique_n": actual_stats["unique_count"],
                        "actual_duplicate_n": actual_stats["duplicate_count"],
                        "source_index_match": source_index_match,
                        "missing_from_actual_n": len(generated - actual) if actual_exists else None,
                        "extra_in_actual_n": len(actual - generated) if actual_exists else None,
                    }
                )
        with (split_dir / "split_metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        per_split = [
            _split_summary(frame, split, data["smiles_col"], data["target_col"])
            for split, frame in split_frames.items()
        ]
        record = {
            "config": str(config_path),
            "config_name": cfg_name,
            "dataset": cfg_name.replace("_gine_desc16", "").replace("_scaffold", ""),
            "requested_split": data["split"],
            "seed": int(seed),
            "metadata": metadata,
            "splits": per_split,
        }
        rows.append(record)
        with (cfg_out / f"seed{seed}.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        for split, frame in split_frames.items():
            summary = _split_summary(frame, split, data["smiles_col"], data["target_col"])
            target_rows.append({"config_name": cfg_name, "seed": seed, **summary})
            acyclic_rows.append(
                {
                    "config_name": cfg_name,
                    "seed": seed,
                    "split": split,
                    "acyclic_fraction": summary["acyclic_fraction"],
                    "n": summary["n"],
                }
            )
            labels = [_scaffold_label(smi, acyclic_unique=True) for smi in frame[data["smiles_col"]].astype(str)]
            for label, size in Counter(labels).items():
                scaffold_rows.append(
                    {
                        "config_name": cfg_name,
                        "seed": seed,
                        "split": split,
                        "scaffold": label,
                        "size": int(size),
                    }
                )
        test_sets[seed] = set(pd.read_csv(paths["test"])["source_index"].astype(int))

    overlap_rows = []
    for left_seed, left in sorted(test_sets.items()):
        for right_seed, right in sorted(test_sets.items()):
            inter = left & right
            union = left | right
            overlap_rows.append(
                {
                    "config_name": cfg_name,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "test_overlap_n": len(inter),
                    "test_jaccard": float(len(inter) / len(union)) if union else 0.0,
                }
            )

    pd.DataFrame(target_rows, columns=TARGET_COLUMNS).to_csv(cfg_out / "target_distribution_by_split.csv", index=False)
    pd.DataFrame(scaffold_rows, columns=SCAFFOLD_COLUMNS).to_csv(cfg_out / "scaffold_size_distribution.csv", index=False)
    pd.DataFrame(acyclic_rows, columns=ACYCLIC_COLUMNS).to_csv(cfg_out / "acyclic_fraction_by_split.csv", index=False)
    pd.DataFrame(duplicate_rows, columns=DUPLICATE_COLUMNS).to_csv(cfg_out / "canonical_duplicate_overlap.csv", index=False)
    pd.DataFrame(actual_rows, columns=ACTUAL_COLUMNS).to_csv(cfg_out / "actual_split_comparison.csv", index=False)
    pd.DataFrame(overlap_rows, columns=OVERLAP_COLUMNS).to_csv(cfg_out / "split_overlap_matrix.csv", index=False)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit random/scaffold split protocol and distributions.")
    parser.add_argument("--configs", required=True, help="Comma-separated config YAML files.")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--out", default="reports/split_audit")
    parser.add_argument("--actual-split-template", default="")
    args = parser.parse_args(argv)
    configs = [Path(path) for path in _as_list(args.configs)]
    seeds = [int(seed) for seed in _as_list(args.seeds)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for config in configs:
        all_records.extend(
            audit_config(
                config,
                out_dir,
                seeds,
                actual_split_template=args.actual_split_template or None,
            )
        )

    summary_rows = []
    for record in all_records:
        for split in record["splits"]:
            summary_rows.append(
                {
                    "config_name": record["config_name"],
                    "requested_split": record["requested_split"],
                    "seed": record["seed"],
                    "actual_split_template": args.actual_split_template,
                    **split,
                }
            )
    pd.DataFrame(summary_rows).to_csv(out_dir / "split_audit_summary.csv", index=False)
    with (out_dir / "split_audit_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "configs": [str(config) for config in configs],
                "seeds": seeds,
                "actual_split_template": args.actual_split_template,
                "n_records": len(all_records),
            },
            f,
            indent=2,
        )
    print(f"wrote split audit to {out_dir}")


if __name__ == "__main__":
    main()
