from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from molfuse_gine.features.atom_bond import is_hbond_acceptor, is_hbond_donor


CONFORMER3D_STATS = [
    "RadiusOfGyration",
    "PMI1",
    "PMI2",
    "PMI3",
    "NPR1",
    "NPR2",
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "SpherocityIndex",
    "PairDistMean",
    "PairDistStd",
    "PairDistMin",
    "PairDistMax",
    "HBDHBADistMean",
    "HBDHBADistMin",
    "AromaticCentroidDistMean",
    "AromaticCentroidDistMin",
]


def _embed_mol(smiles: str, num_confs: int, seed: int) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 0
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(num_confs), params=params))
    except Exception:
        return None
    if not conf_ids:
        return None
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    for conf_id in conf_ids:
        try:
            if props is not None:
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", confId=int(conf_id), maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol, confId=int(conf_id), maxIters=200)
        except Exception:
            continue
    return mol


def _pairwise_distances(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    conf = mol.GetConformer(conf_id)
    coords = np.asarray(conf.GetPositions(), dtype=np.float64)
    if coords.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    tri = np.triu_indices(coords.shape[0], k=1)
    return dist[tri]


def _hbd_hba_distances(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    donors = [atom.GetIdx() for atom in mol.GetAtoms() if is_hbond_donor(atom)]
    acceptors = [atom.GetIdx() for atom in mol.GetAtoms() if is_hbond_acceptor(atom)]
    if not donors or not acceptors:
        return np.zeros(0, dtype=np.float64)
    conf = mol.GetConformer(conf_id)
    out = []
    for i in donors:
        pi = np.asarray(conf.GetAtomPosition(i), dtype=np.float64)
        for j in acceptors:
            if i == j:
                continue
            pj = np.asarray(conf.GetAtomPosition(j), dtype=np.float64)
            out.append(float(np.linalg.norm(pi - pj)))
    return np.asarray(out, dtype=np.float64)


def _aromatic_centroid_distances(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    rings = [
        tuple(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if ring and all(mol.GetAtomWithIdx(int(idx)).GetIsAromatic() for idx in ring)
    ]
    if len(rings) < 2:
        return np.zeros(0, dtype=np.float64)
    conf = mol.GetConformer(conf_id)
    centroids = []
    for ring in rings:
        coords = [np.asarray(conf.GetAtomPosition(int(idx)), dtype=np.float64) for idx in ring]
        centroids.append(np.mean(np.vstack(coords), axis=0))
    out = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            out.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return np.asarray(out, dtype=np.float64)


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}Mean": 0.0, f"{prefix}Std": 0.0, f"{prefix}Min": 0.0, f"{prefix}Max": 0.0}
    return {
        f"{prefix}Mean": float(values.mean()),
        f"{prefix}Std": float(values.std()),
        f"{prefix}Min": float(values.min()),
        f"{prefix}Max": float(values.max()),
    }


def _conformer_row(mol: Chem.Mol, conf_id: int) -> dict[str, float]:
    row = {
        "RadiusOfGyration": float(rdMolDescriptors.CalcRadiusOfGyration(mol, confId=conf_id)),
        "PMI1": float(rdMolDescriptors.CalcPMI1(mol, confId=conf_id)),
        "PMI2": float(rdMolDescriptors.CalcPMI2(mol, confId=conf_id)),
        "PMI3": float(rdMolDescriptors.CalcPMI3(mol, confId=conf_id)),
        "NPR1": float(rdMolDescriptors.CalcNPR1(mol, confId=conf_id)),
        "NPR2": float(rdMolDescriptors.CalcNPR2(mol, confId=conf_id)),
        "Asphericity": float(rdMolDescriptors.CalcAsphericity(mol, confId=conf_id)),
        "Eccentricity": float(rdMolDescriptors.CalcEccentricity(mol, confId=conf_id)),
        "InertialShapeFactor": float(rdMolDescriptors.CalcInertialShapeFactor(mol, confId=conf_id)),
        "SpherocityIndex": float(rdMolDescriptors.CalcSpherocityIndex(mol, confId=conf_id)),
    }
    row.update(_safe_stats(_pairwise_distances(mol, conf_id), "PairDist"))
    hbd_hba = _safe_stats(_hbd_hba_distances(mol, conf_id), "HBDHBADist")
    aromatic = _safe_stats(_aromatic_centroid_distances(mol, conf_id), "AromaticCentroidDist")
    row["HBDHBADistMean"] = hbd_hba["HBDHBADistMean"]
    row["HBDHBADistMin"] = hbd_hba["HBDHBADistMin"]
    row["AromaticCentroidDistMean"] = aromatic["AromaticCentroidDistMean"]
    row["AromaticCentroidDistMin"] = aromatic["AromaticCentroidDistMin"]
    return row


def _stable_seed(smiles: str, seed: int) -> int:
    digest = hashlib.sha256(str(smiles).encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(seed)) % (2**31 - 1)


@lru_cache(maxsize=20000)
def compute_conformer3d_row(
    smiles: str,
    num_confs: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    mol = _embed_mol(smiles, num_confs=num_confs, seed=seed)
    if mol is None:
        return {f"conf3d_{name}_mean": 0.0 for name in CONFORMER3D_STATS} | {
            f"conf3d_{name}_std": 0.0 for name in CONFORMER3D_STATS
        }
    rows = []
    for conf in mol.GetConformers():
        try:
            rows.append(_conformer_row(mol, conf.GetId()))
        except Exception:
            continue
    if not rows:
        return {f"conf3d_{name}_mean": 0.0 for name in CONFORMER3D_STATS} | {
            f"conf3d_{name}_std": 0.0 for name in CONFORMER3D_STATS
        }
    df = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for name in CONFORMER3D_STATS:
        values = pd.to_numeric(df.get(name, 0.0), errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        out[f"conf3d_{name}_mean"] = float(values.mean()) if values.size else 0.0
        out[f"conf3d_{name}_std"] = float(values.std()) if values.size else 0.0
    return out


def compute_conformer3d_frame(
    smiles: Iterable[str],
    num_confs: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        canonical = Chem.MolToSmiles(mol, canonical=True) if mol is not None else str(smi)
        rows.append(
            compute_conformer3d_row(
                canonical,
                num_confs=num_confs,
                seed=_stable_seed(canonical, seed),
            )
        )
    return pd.DataFrame(rows)
