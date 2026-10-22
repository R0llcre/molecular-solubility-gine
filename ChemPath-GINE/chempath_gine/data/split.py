from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .clean import clean_molecules


def _random_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


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
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)

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
        return _random_split(df, train_ratio, val_ratio, seed=seed)
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


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
        train, val, test = _scaffold_split(df, smiles_col, train_ratio, val_ratio, seed)
    elif split == "random":
        train, val, test = _random_split(df, train_ratio, val_ratio, seed)
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
    return paths
