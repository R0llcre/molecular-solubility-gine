"""Build fresh "balanced scaffold" confirmation splits for ESOL (seeds 1-5).

These splits exist so a final model can be confirmed on test molecules that no
earlier experiment was selected on. The ChemPath deterministic scaffold split
(chempath_gine/data/split.py) always yields the same test set; here each seed
gives a different, scaffold-disjoint test set.

Algorithm (chemprop v1 ``scaffold_split(..., balanced=True)``):
  1. clean the CSV with chempath_gine.data.clean.clean_molecules (keeps source_index);
  2. group molecules by Bemis-Murcko scaffold (MurckoScaffoldSmiles,
     includeChirality=False; acyclic molecules share the empty scaffold), the
     same definition as the deterministic ChemPath split;
  3. groups larger than half the val or test size ("big") go first, the rest
     follow; both lists are shuffled with random.Random(seed);
  4. each group goes to train if it still fits under train_ratio * n, else to
     val if it fits under val_ratio * n, else to test.

Outputs (same columns as the cleaned CSV, rows in cleaned order):
  <out>/scaffold_balanced_seed{s}/{train,val,test}.csv
and, unless --no-configs, run configs with explicit split paths:
  <configs-out>/{desc6,desc7}_scaffold_balanced_seed{s}.yaml
Run a config with ``scripts/run_calibration.py --seeds s`` (s = its split seed);
with explicit paths, run.seed only sets the initialisation/shuffle seed.

No model is trained and no test-set metric is computed here; the printed
statistics describe the data only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import tempfile
import time
from collections import OrderedDict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chempath_gine.data.clean import clean_molecules  # noqa: E402
from chempath_gine.data.split import _scaffold  # noqa: E402  (same scaffold definition as split.py)

RDLogger.DisableLog("rdApp.*")

SPLIT_NAME = "scaffold_balanced"
PARTS = ("train", "val", "test")
DEFAULT_CSV = "calibration_runs/perf/data/esol_with_desc_ap.csv"
DEFAULT_OUT = "calibration_runs/confirm/splits"
DEFAULT_CONFIGS_OUT = "calibration_runs/confirm/configs"
DEFAULT_REFERENCE = "calibration_runs/esol_scaffold/runs/faithful_seed1_gine_desc/splits/test.csv"
BASE_CONFIGS = OrderedDict(
    [
        ("desc6", "configs/esol_fusion_ablation_scaffold.yaml"),
        ("desc7", "calibration_runs/perf/configs/desc7_logp_scaffold.yaml"),
    ]
)


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def read_bytes_retry(path: Path, tries: int = 20, wait: float = 1.0) -> bytes:
    """Read a file that may be evicted by iCloud: retry until len(bytes) == st_size > 0."""
    last = None
    for _ in range(tries):
        try:
            size = os.stat(path).st_size
            data = path.read_bytes()
            if size > 0 and len(data) == size:
                return data
            last = f"short read {len(data)}/{size} bytes"
        except OSError as exc:  # TimeoutError is an OSError
            last = repr(exc)
        time.sleep(wait)
    raise OSError(f"could not read {path}: {last}")


def read_csv_retry(path: Path, **kwargs) -> pd.DataFrame:
    import io

    return pd.read_csv(io.BytesIO(read_bytes_retry(path)), **kwargs)


def load_clean(csv_path: Path, smiles_col: str) -> pd.DataFrame:
    """Clean via clean_molecules on a verified local copy (row order, hence source_index, unchanged)."""
    with tempfile.TemporaryDirectory(prefix="confirm_splits_") as tmp:
        src = Path(tmp) / csv_path.name
        src.write_bytes(read_bytes_retry(csv_path))
        clean_path = clean_molecules(src, Path(tmp) / "clean.csv", smiles_col)
        # round_trip keeps every float bit-identical to the cleaned CSV
        return pd.read_csv(clean_path, float_precision="round_trip")


def scaffold_groups(smiles: list[str]) -> "OrderedDict[str, list[int]]":
    groups: "OrderedDict[str, list[int]]" = OrderedDict()  # first-occurrence order, like chemprop's dict
    for i, smi in enumerate(smiles):
        groups.setdefault(_scaffold(smi), []).append(i)
    return groups


def balanced_scaffold_split(groups, n: int, train_ratio: float, val_ratio: float, seed: int):
    """chemprop scaffold_split(balanced=True) on positional row indices."""
    test_ratio = 1.0 - train_ratio - val_ratio
    train_size, val_size, test_size = train_ratio * n, val_ratio * n, test_ratio * n
    big, small = [], []
    for scaffold, idx in groups.items():
        if len(idx) > val_size / 2 or len(idx) > test_size / 2:
            big.append((scaffold, idx))
        else:
            small.append((scaffold, idx))
    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)
    parts = {p: [] for p in PARTS}
    part_scaffolds = {p: [] for p in PARTS}
    for scaffold, idx in big + small:
        if len(parts["train"]) + len(idx) <= train_size:
            part = "train"
        elif len(parts["val"]) + len(idx) <= val_size:
            part = "val"
        else:
            part = "test"
        parts[part].extend(idx)
        part_scaffolds[part].append(scaffold)
    meta = {
        "n": n,
        "target_sizes": {"train": train_size, "val": val_size, "test": test_size},
        "big_group_threshold": min(val_size, test_size) / 2,
        "big_groups": sorted(
            ({"scaffold": s or "<acyclic>", "size": len(i)} for s, i in big), key=lambda d: -d["size"]
        ),
    }
    return {p: sorted(v) for p, v in parts.items()}, part_scaffolds, meta


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def part_stats(df: pd.DataFrame, idx: list[int], scaffolds: list[str], smiles_col: str, target_col: str):
    sub = df.iloc[idx]
    heavy = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in sub[smiles_col]]
    return {
        "n": len(idx),
        "frac": len(idx) / len(df),
        "n_scaffold_groups": len(scaffolds),
        "acyclic_in_part": "" in scaffolds,
        "mean_heavy_atoms": float(np.mean(heavy)),
        "mean_logS": float(sub[target_col].mean()),
        "std_logS": float(sub[target_col].std(ddof=1)),
    }


def write_configs(configs_out: Path, split_dirs: dict[int, Path], csv_rel: str) -> list[Path]:
    written = []
    configs_out.mkdir(parents=True, exist_ok=True)
    for tag, base_rel in BASE_CONFIGS.items():
        base = yaml.safe_load(read_bytes_retry(_abs(base_rel)).decode("utf-8"))
        for seed, split_dir in split_dirs.items():
            cfg = copy.deepcopy(base)
            cfg["run"]["name"] = f"{tag}_{SPLIT_NAME}_seed{seed}"
            cfg["run"]["seed"] = int(seed)
            # data.csv is kept as in the base config; the explicit paths below take precedence
            # in train.build_datasets. data.split only feeds run_calibration's cache key and the
            # "split" field of its summaries (make_splits is never called with explicit paths).
            cfg["data"]["split"] = SPLIT_NAME
            for part in PARTS:
                cfg["data"][f"{part}_csv"] = str((split_dir / f"{part}.csv").resolve())
            header = (
                f"# Generated by scripts/make_confirmation_splits.py from {base_rel}\n"
                f"# Confirmation split: balanced scaffold (chemprop), split seed {seed}, cleaned from {csv_rel}.\n"
                f"# Run with: scripts/run_calibration.py --config <this file> --seeds {seed} ...\n"
            )
            path = configs_out / f"{tag}_{SPLIT_NAME}_seed{seed}.yaml"
            path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            written.append(path)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--configs-out", default=DEFAULT_CONFIGS_OUT)
    ap.add_argument("--no-configs", action="store_true")
    ap.add_argument("--reference-test", default=DEFAULT_REFERENCE,
                    help="deterministic ChemPath scaffold test.csv to measure overlap against")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--target-col", default="Solubility")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--verify", action="store_true",
                    help="regenerate in memory and check the files on disk match; write nothing")
    ap.add_argument("--report", default=None, help="optional JSON path for the statistics")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    csv_path, out_dir = _abs(args.csv), _abs(args.out)
    df = load_clean(csv_path, args.smiles_col)
    smiles = df[args.smiles_col].astype(str).tolist()
    groups = scaffold_groups(smiles)
    n = len(df)
    print(f"cleaned {csv_path}: {n} molecules, {len(groups)} scaffold groups "
          f"(acyclic group size {len(groups.get('', []))})")

    ref_path = _abs(args.reference_test) if args.reference_test else None
    ref_ids = None
    if ref_path is not None and ref_path.exists():
        ref = read_csv_retry(ref_path)
        ref_ids = set(ref["source_index"].astype(int))
        # source_index is only comparable if the reference was cleaned from the same row order
        ref_smiles = dict(zip(ref["source_index"].astype(int), ref[args.smiles_col].astype(str)))
        by_id = dict(zip(df["source_index"].astype(int), smiles))
        bad = [i for i, s in ref_smiles.items() if by_id.get(i) != s]
        if bad:
            raise SystemExit(f"reference source_index does not line up with {csv_path} ({len(bad)} rows)")

    report = {"csv": str(csv_path), "n": n, "n_scaffold_groups": len(groups),
              "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
              "reference_test": str(ref_path) if ref_ids is not None else None,
              "reference_test_n": len(ref_ids) if ref_ids is not None else None, "seeds": {}}
    test_sets, split_dirs, mismatches = {}, {}, []
    for seed in seeds:
        parts, part_scaffolds, meta = balanced_scaffold_split(
            groups, n, args.train_ratio, args.val_ratio, seed
        )
        all_idx = parts["train"] + parts["val"] + parts["test"]
        assert len(all_idx) == n and len(set(all_idx)) == n, "parts must partition the data"
        sets = {p: set(part_scaffolds[p]) for p in PARTS}
        assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
        assert min(len(parts[p]) for p in PARTS) > 0, "empty part"

        split_dir = out_dir / f"{SPLIT_NAME}_seed{seed}"
        split_dirs[seed] = split_dir
        for part in PARTS:
            sub = df.iloc[parts[part]]
            path = split_dir / f"{part}.csv"
            if args.verify:
                disk = read_csv_retry(path)
                if sorted(disk["source_index"].astype(int)) != sorted(sub["source_index"].astype(int)):
                    mismatches.append(str(path))
            else:
                split_dir.mkdir(parents=True, exist_ok=True)
                sub.to_csv(path, index=False)

        ids = {p: set(df.iloc[parts[p]]["source_index"].astype(int)) for p in PARTS}
        test_sets[seed] = ids["test"]
        entry = {
            "dir": str(split_dir),
            "big_group_threshold": meta["big_group_threshold"],
            "big_groups": meta["big_groups"],
            "parts": {p: part_stats(df, parts[p], part_scaffolds[p], args.smiles_col, args.target_col)
                      for p in PARTS},
        }
        if ref_ids is not None:
            entry["test_vs_reference"] = {
                "jaccard": jaccard(ids["test"], ref_ids),
                "intersection": len(ids["test"] & ref_ids),
                # where the reference (deterministic) test molecules land in this split
                "reference_in_train": len(ref_ids & ids["train"]),
                "reference_in_val": len(ref_ids & ids["val"]),
                "reference_in_test": len(ref_ids & ids["test"]),
            }
        report["seeds"][str(seed)] = entry

    report["pairwise_test_jaccard"] = {
        f"{a}-{b}": jaccard(test_sets[a], test_sets[b]) for a, b in combinations(seeds, 2)
    }
    union = set().union(*test_sets.values()) if test_sets else set()
    report["test_union_n"] = len(union)
    report["test_union_frac"] = len(union) / n

    if args.verify:
        if mismatches:
            raise SystemExit("split files differ from regenerated splits:\n  " + "\n  ".join(mismatches))
        print("verify: all split files match the regenerated splits")
    elif not args.no_configs:
        try:
            csv_rel = str(csv_path.relative_to(ROOT))
        except ValueError:
            csv_rel = str(csv_path)
        for path in write_configs(_abs(args.configs_out), split_dirs, csv_rel):
            print(f"wrote {path}")

    # ---- printed summary (data statistics only) ----
    print(f"\nbig-group threshold = {min(args.val_ratio, 1 - args.train_ratio - args.val_ratio) * n / 2:.1f} molecules")
    hdr = f"{'seed':>4} {'part':>5} {'n':>4} {'groups':>6} {'heavy':>6} {'logS':>7} {'sdlogS':>6}"
    print(hdr)
    for seed in seeds:
        e = report["seeds"][str(seed)]
        for p in PARTS:
            s = e["parts"][p]
            print(f"{seed:>4} {p:>5} {s['n']:>4} {s['n_scaffold_groups']:>6} {s['mean_heavy_atoms']:>6.2f} "
                  f"{s['mean_logS']:>7.3f} {s['std_logS']:>6.3f}")
        if "test_vs_reference" in e:
            r = e["test_vs_reference"]
            print(f"     test vs deterministic scaffold test: jaccard={r['jaccard']:.3f} "
                  f"(shared {r['intersection']}; reference molecules here in train/val/test = "
                  f"{r['reference_in_train']}/{r['reference_in_val']}/{r['reference_in_test']})")
    print("pairwise test Jaccard: " + ", ".join(f"{k}={v:.3f}" for k, v in report["pairwise_test_jaccard"].items()))
    print(f"union of {len(seeds)} test sets: {report['test_union_n']} molecules ({report['test_union_frac']:.1%})")

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {rp}")
    return report


if __name__ == "__main__":
    main()
