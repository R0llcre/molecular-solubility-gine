from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem


def clean_molecules(
    csv_path: str | Path,
    out_path: str | Path,
    smiles_col: str,
) -> Path:
    """Drop invalid SMILES before splitting/scaling and keep source_index."""
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"{csv_path} must contain {smiles_col!r}")

    rows = []
    has_source_index = "source_index" in df.columns
    for source_index, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is None:
            continue
        clean = row.to_dict()
        clean["source_index"] = int(row["source_index"]) if has_source_index else int(source_index)
        clean[smiles_col] = Chem.MolToSmiles(mol, canonical=True)
        rows.append(clean)

    if not rows:
        raise ValueError(f"No valid molecules found in {csv_path}")
    out_df = pd.DataFrame(rows)
    if out_df["source_index"].isna().any() or out_df["source_index"].duplicated().any():
        raise ValueError(f"{csv_path} produced non-unique or missing source_index values")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_path
