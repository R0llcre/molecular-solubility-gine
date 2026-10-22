from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS

from molfuse_gine.features.motifs import MOTIF_SMARTS


def brics_fragments(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return []
    try:
        return sorted(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def build_brics_vocab(smiles: Iterable[str], max_fragments: int = 256) -> list[str]:
    counts: Counter[str] = Counter()
    for smi in smiles:
        counts.update(brics_fragments(str(smi)))
    return [frag for frag, _ in counts.most_common(max(0, int(max_fragments)))]


def compute_brics_count_frame(smiles: Iterable[str], vocab: list[str]) -> pd.DataFrame:
    smiles_list = list(smiles)
    columns = [f"brics_{i}" for i in range(len(vocab))]
    vocab_index = {frag: i for i, frag in enumerate(vocab)}
    arr = np.zeros((len(smiles_list), len(vocab)), dtype=np.float32)
    for row_idx, smi in enumerate(smiles_list):
        counts = Counter(brics_fragments(str(smi)))
        for frag, count in counts.items():
            idx = vocab_index.get(frag)
            if idx is not None:
                arr[row_idx, idx] = float(count)
    return pd.DataFrame(arr, columns=columns)


def compute_functional_group_count_frame(smiles: Iterable[str]) -> pd.DataFrame:
    smiles_list = list(smiles)
    columns = [f"fg_{name}" for name in MOTIF_SMARTS]
    patterns = [(name, Chem.MolFromSmarts(smarts)) for name, smarts in MOTIF_SMARTS.items()]
    arr = np.zeros((len(smiles_list), len(patterns)), dtype=np.float32)
    for row_idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        for col_idx, (_, patt) in enumerate(patterns):
            if patt is not None:
                arr[row_idx, col_idx] = float(len(mol.GetSubstructMatches(patt)))
    return pd.DataFrame(arr, columns=columns)
