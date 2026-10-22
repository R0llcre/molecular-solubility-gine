from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .clean import clean_molecules


def _random_split(df: pd.DataFrame, smiles_col: str, train_ratio: float, val_ratio: float, seed: int):
    groups = defaultdict(list)
    for idx, smiles in enumerate(df[smiles_col].astype(str)):
        groups[smiles].append(idx)
    ordered = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(df)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_idx, val_idx, test_idx = [], [], []
    for group in ordered:
        if len(train_idx) < n_train or not train_idx:
            train_idx.extend(group)
        elif len(val_idx) < n_val or not val_idx:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
    if not train_idx or not val_idx or not test_idx:
        raise ValueError("Random split could not produce non-empty train/val/test partitions")
    meta = {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "num_canonical_molecule_groups": len(groups),
        "num_duplicate_canonical_groups": sum(1 for group in groups.values() if len(group) > 1),
        "largest_canonical_group_size": max((len(group) for group in groups.values()), default=0),
    }
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx], meta


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return scaffold
    return f"acyclic:{Chem.MolToSmiles(mol, canonical=True)}"


def _scaffold_split(
    df: pd.DataFrame,
    smiles_col: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    groups = defaultdict(list)
    for idx, smiles in enumerate(df[smiles_col].astype(str)):
        groups[_scaffold(smiles)].append(idx)
    by_size: dict[int, list[list[int]]] = defaultdict(list)
    for group in groups.values():
        by_size[len(group)].append(group)
    rng = random.Random(seed)
    ordered = []
    for size in sorted(by_size, reverse=True):
        tied = sorted(by_size[size], key=lambda g: g[0])
        rng.shuffle(tied)
        ordered.extend(tied)

    n = len(df)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train_idx, val_idx, test_idx = [], [], []
    for group in ordered:
        if len(train_idx) < n_train or not train_idx:
            train_idx.extend(group)
        elif len(val_idx) < n_val or not val_idx:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            "Scaffold split could not produce non-empty train/val/test partitions; "
            "use split='random' explicitly or adjust ratios/dataset."
        )
    meta = {
        "num_scaffolds": len(groups),
        "largest_scaffold_size": max((len(g) for g in groups.values()), default=0),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx], meta


def make_splits(
    csv_path: str | Path,
    out_dir: str | Path,
    smiles_col: str,
    split: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_molecules(csv_path, out_dir / "clean.csv", smiles_col)
    df = pd.read_csv(clean_path)
    if len(df) < 3:
        raise ValueError("Need at least 3 rows for train/val/test split")

    if split == "scaffold":
        train, val, test, split_meta = _scaffold_split(df, smiles_col, train_ratio, val_ratio, seed)
    elif split == "random":
        train, val, test, split_meta = _random_split(df, smiles_col, train_ratio, val_ratio, seed)
    else:
        raise ValueError(f"Unknown split={split!r}; use random or scaffold")

    paths = {
        "train": out_dir / "train.csv",
        "val": out_dir / "val.csv",
        "test": out_dir / "test.csv",
    }
    train.to_csv(paths["train"], index=False)
    val.to_csv(paths["val"], index=False)
    test.to_csv(paths["test"], index=False)
    with (out_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "requested_split": split,
                "actual_split": split,
                "seed": int(seed),
                "train_ratio": float(train_ratio),
                "val_ratio": float(val_ratio),
                **split_meta,
            },
            f,
            indent=2,
        )
    return paths
