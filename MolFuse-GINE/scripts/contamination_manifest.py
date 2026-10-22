"""Pretrain<->test contamination manifest.

For each TEST molecule, measure how close it is to a (large) PRETRAINING corpus:
  - exact canonical-SMILES membership (the molecule is literally in the corpus), and
  - the maximum ECFP4 Tanimoto similarity to any corpus molecule (near-duplicate).

This is the core instrument for the contamination audit: it lets us stratify a
foundation model's gain-over-baseline by how contaminated each test molecule is,
and to build a decontaminated test set. Provider-agnostic — point it at any
corpus SMILES file (PubChem / ZINC15 / ChEMBL / a model's released pretrain set).

Memory-safe: the corpus is streamed in chunks; we keep a running max-Tanimoto per
test molecule plus a set of canonical corpus SMILES for the exact-match flag.

Example:
  python scripts/contamination_manifest.py \
    --test-csv data/processed/freesolv_with_desc.csv --smiles-col SMILES \
    --corpus corpora/zinc15_2m.smi \
    --out reports/contamination/freesolv_vs_zinc.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def fingerprint(smiles: str, radius: int, nbits: int):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def _iter_corpus_smiles(path: Path, smiles_col: str | None):
    """Yield raw SMILES strings from a .csv/.smi/.txt corpus, streaming."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        for chunk in pd.read_csv(path, sep=sep, chunksize=200_000):
            col = smiles_col or ("smiles" if "smiles" in chunk.columns else chunk.columns[0])
            for s in chunk[col].astype(str):
                yield s
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # ".smi" lines are often "SMILES<space/tab>id"; take the first token.
                yield line.replace("\t", " ").split(" ", 1)[0]


def build_manifest(
    test_df: pd.DataFrame,
    smiles_col: str,
    corpus_path: Path,
    corpus_smiles_col: str | None,
    radius: int,
    nbits: int,
    chunk_size: int,
    max_corpus: int | None,
) -> tuple[pd.DataFrame, dict]:
    test_can = [canonical(s) for s in test_df[smiles_col]]
    test_fps = [fingerprint(s, radius, nbits) if s else None for s in test_can]
    n_test = len(test_can)

    max_tani = np.zeros(n_test, dtype=np.float64)
    nearest = [None] * n_test
    corpus_canon_set: set[str] = set()

    fp_chunk: list = []
    can_chunk: list[str] = []
    seen = 0

    def flush():
        if not fp_chunk:
            return
        for i in range(n_test):
            if test_fps[i] is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(test_fps[i], fp_chunk)
            j = int(np.argmax(sims))
            if sims[j] > max_tani[i]:
                max_tani[i] = float(sims[j])
                nearest[i] = can_chunk[j]

    for raw in _iter_corpus_smiles(corpus_path, corpus_smiles_col):
        if max_corpus is not None and seen >= max_corpus:
            break
        can = canonical(raw)
        if can is None:
            continue
        seen += 1
        corpus_canon_set.add(can)
        fp = fingerprint(can, radius, nbits)
        if fp is None:
            continue
        fp_chunk.append(fp)
        can_chunk.append(can)
        if len(fp_chunk) >= chunk_size:
            flush()
            fp_chunk, can_chunk = [], []
    flush()

    exact = [bool(c is not None and c in corpus_canon_set) for c in test_can]
    out = pd.DataFrame(
        {
            "source_index": test_df.get("source_index", test_df.index),
            "smiles": test_df[smiles_col].astype(str).values,
            "canonical_smiles": test_can,
            "in_corpus_exact": exact,
            "max_tanimoto": max_tani,
            "nearest_corpus_smiles": nearest,
        }
    )
    finite = out["max_tanimoto"].to_numpy()
    summary = {
        "n_test": int(n_test),
        "corpus_size_scanned": int(seen),
        "exact_match_frac": float(np.mean(exact)),
        "tanimoto_ge_0.4_frac": float(np.mean(finite >= 0.4)),
        "tanimoto_ge_0.6_frac": float(np.mean(finite >= 0.6)),
        "tanimoto_ge_0.8_frac": float(np.mean(finite >= 0.8)),
        "tanimoto_eq_1.0_frac": float(np.mean(finite >= 0.999)),
        "max_tanimoto_mean": float(np.mean(finite)),
    }
    return out, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--corpus", required=True, help="SMILES corpus file (.smi/.txt/.csv)")
    p.add_argument("--corpus-smiles-col", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--chunk-size", type=int, default=50_000)
    p.add_argument("--max-corpus", type=int, default=None, help="cap corpus size (smoke tests)")
    args = p.parse_args(argv)

    test_df = pd.read_csv(args.test_csv)
    if args.smiles_col not in test_df.columns:
        raise KeyError(f"{args.test_csv} has no column {args.smiles_col!r}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest, summary = build_manifest(
        test_df=test_df,
        smiles_col=args.smiles_col,
        corpus_path=Path(args.corpus),
        corpus_smiles_col=args.corpus_smiles_col,
        radius=args.radius,
        nbits=args.nbits,
        chunk_size=args.chunk_size,
        max_corpus=args.max_corpus,
    )
    manifest.to_csv(out_path, index=False)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
