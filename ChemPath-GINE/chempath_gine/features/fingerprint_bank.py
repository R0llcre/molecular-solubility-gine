from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors


RDLogger.DisableLog("rdApp.warning")


FINGERPRINT_SPECS = {
    "morgan2": ("morgan", 2, 2048),
    "morgan3": ("morgan", 3, 2048),
    "maccs": ("maccs", 0, 167),
    "rdkitfp": ("rdkitfp", 0, 2048),
    "atom_pair": ("atom_pair", 0, 2048),
    "torsion": ("torsion", 0, 2048),
}


def fingerprint_names(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        names = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip().lower() for item in value if str(item).strip()]
    unknown = [name for name in names if name not in FINGERPRINT_SPECS]
    if unknown:
        raise KeyError(f"Unknown fingerprint(s): {unknown}. Options: {sorted(FINGERPRINT_SPECS)}")
    return names


def _bitvect_to_array(bitvect, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def compute_fingerprint(mol: Chem.Mol, name: str) -> np.ndarray:
    kind, radius, n_bits = FINGERPRINT_SPECS[name]
    if kind == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    elif kind == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif kind == "rdkitfp":
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
    elif kind == "atom_pair":
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    elif kind == "torsion":
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
    else:  # pragma: no cover - guarded by FINGERPRINT_SPECS
        raise ValueError(name)
    return _bitvect_to_array(fp, n_bits)


def compute_fingerprint_frame(smiles: Iterable[str], fingerprints: str | Iterable[str]) -> pd.DataFrame:
    names = fingerprint_names(fingerprints)
    if not names:
        return pd.DataFrame(index=range(len(list(smiles))))

    # Materialize once because callers often pass a pandas Series iterator.
    smiles_list = list(smiles)
    blocks: list[np.ndarray] = []
    columns: list[str] = []
    for fp_name in names:
        _, _, n_bits = FINGERPRINT_SPECS[fp_name]
        arr = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                arr[i] = compute_fingerprint(mol, fp_name)
        blocks.append(arr)
        columns.extend([f"{fp_name}_{j}" for j in range(n_bits)])
    if not blocks:
        return pd.DataFrame(index=range(len(smiles_list)))
    return pd.DataFrame(np.concatenate(blocks, axis=1), columns=columns)
