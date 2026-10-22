from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch, Data, Dataset

from molfuse_gine.data.scaler import (
    fit_standard_scaler,
    load_scaler,
    save_scaler,
    scaler_signature,
    transform_standard,
)
from molfuse_gine.features import CHEM_PATH_DIM, compute_chem_path_features, featurize_mol

RDLogger.DisableLog("rdApp.*")


FEATURE_VERSION = 4


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise_cols(cols: Sequence[str] | str | None) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        cols = cols.strip()
        return [] if not cols else [c.strip() for c in cols.split(",") if c.strip()]
    return list(cols)


class MoleculeDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        mode: str,
        smiles_col: str,
        target_col: str | None,
        descriptor_cols: Sequence[str] | str | None,
        scaler_dir: str | Path,
        scaler_key: str,
        cache_dir: str | Path,
        max_path_distance: int = 12,
        skip_path_features: bool = False,
        teacher_col: str | None = None,
        force_rebuild: bool = False,
    ):
        super().__init__(root=None)
        self.csv_path = Path(csv_path).resolve()
        self.mode = mode.lower()
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.teacher_col = teacher_col
        self.descriptor_cols = _normalise_cols(descriptor_cols)
        self.max_path_distance = int(max_path_distance)
        self.skip_path_features = bool(skip_path_features)
        self.scaler_dir = Path(scaler_dir).resolve()
        self.scaler_key = scaler_key
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.desc_scaler_path = self.scaler_dir / f"{scaler_key}.desc.npz"
        self.y_scaler_path = self.scaler_dir / f"{scaler_key}.y.npz"

        self.df = pd.read_csv(self.csv_path)
        if self.smiles_col not in self.df.columns:
            raise ValueError(f"{self.csv_path} must contain {self.smiles_col!r}")
        missing_desc = [c for c in self.descriptor_cols if c not in self.df.columns]
        if missing_desc:
            raise KeyError(f"Descriptor columns not found: {missing_desc}")
        if self.target_col and self.target_col not in self.df.columns:
            raise KeyError(f"Target column not found: {self.target_col!r}")
        if self.teacher_col and self.teacher_col not in self.df.columns:
            raise KeyError(f"Teacher column not found: {self.teacher_col!r}")

        self.desc_raw = self._descriptor_matrix()
        self.y_raw = self._target_matrix()
        self.teacher_raw = self._teacher_matrix()
        self.desc, self.y, self._desc_sig, self._y_sig = self._scale_arrays()
        self.teacher = self._scale_teacher()
        self._metadata = self._cache_metadata()
        self.cache_path = self.cache_dir / f"{self.csv_path.stem}.{self._metadata['cache_hash'][:16]}.pt"

        cached = None if force_rebuild else self._load_cache()
        if cached is None:
            self.data_list = self._build_data()
            torch.save({"metadata": self._metadata, "data_list": self.data_list}, self.cache_path)
        else:
            self.data_list = cached

    @property
    def descriptor_dim(self) -> int:
        return len(self.descriptor_cols)

    @property
    def path_dim(self) -> int:
        return CHEM_PATH_DIM

    def _descriptor_matrix(self) -> np.ndarray:
        if not self.descriptor_cols:
            return np.zeros((len(self.df), 0), dtype=np.float32)
        arr = self.df[self.descriptor_cols].astype("float32").values
        if not np.isfinite(arr).all():
            row, col = np.argwhere(~np.isfinite(arr))[0]
            raise ValueError(f"Non-finite descriptor at row={row}, col={self.descriptor_cols[col]}")
        return arr

    def _target_matrix(self) -> np.ndarray:
        if not self.target_col:
            return np.zeros((len(self.df), 1), dtype=np.float32)
        arr = self.df[self.target_col].astype("float32").values.reshape(-1, 1)
        if not np.isfinite(arr).all():
            row = np.argwhere(~np.isfinite(arr))[0][0]
            raise ValueError(f"Non-finite target at row={row}, col={self.target_col}")
        return arr

    def _teacher_matrix(self) -> np.ndarray | None:
        if not self.teacher_col:
            return None
        arr = self.df[self.teacher_col].astype("float32").values.reshape(-1, 1)
        if not np.isfinite(arr).all():
            row = np.argwhere(~np.isfinite(arr))[0][0]
            raise ValueError(f"Non-finite teacher target at row={row}, col={self.teacher_col}")
        return arr

    def _scale_arrays(self):
        fit = self.mode == "train"
        if fit:
            desc_scaler = fit_standard_scaler(self.desc_raw)
            y_scaler = fit_standard_scaler(self.y_raw)
            save_scaler(self.desc_scaler_path, desc_scaler)
            save_scaler(self.y_scaler_path, y_scaler)
        else:
            desc_scaler = load_scaler(self.desc_scaler_path)
            y_scaler = load_scaler(self.y_scaler_path)

        if desc_scaler["mean"].shape[0] != self.desc_raw.shape[1]:
            raise ValueError("Descriptor scaler dimension does not match descriptor_cols")

        desc = transform_standard(self.desc_raw, desc_scaler)
        y = transform_standard(self.y_raw, y_scaler)
        return desc, y, scaler_signature(desc_scaler), scaler_signature(y_scaler)

    def _scale_teacher(self) -> np.ndarray | None:
        if self.teacher_raw is None:
            return None
        y_scaler = load_scaler(self.y_scaler_path)
        return transform_standard(self.teacher_raw, y_scaler)

    def _cache_metadata(self) -> dict:
        meta = {
            "feature_version": FEATURE_VERSION,
            "csv_sha256": _sha256_file(self.csv_path),
            "mode": self.mode,
            "smiles_col": self.smiles_col,
            "target_col": self.target_col,
            "teacher_col": self.teacher_col,
            "descriptor_cols": self.descriptor_cols,
            "max_path_distance": self.max_path_distance,
            "skip_path_features": self.skip_path_features,
            "desc_scaler": self._desc_sig,
            "y_scaler": self._y_sig,
        }
        h = hashlib.sha256(repr(sorted(meta.items())).encode("utf-8")).hexdigest()
        meta["cache_hash"] = h
        return meta

    def _load_cache(self):
        if not self.cache_path.exists():
            return None
        try:
            try:
                payload = torch.load(self.cache_path, weights_only=False)
            except TypeError:
                payload = torch.load(self.cache_path)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("metadata") != self._metadata:
            return None
        return payload.get("data_list")

    def _build_data(self):
        data_list = []
        for row_idx, row in self.df.iterrows():
            smiles = str(row[self.smiles_col])
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            feats = featurize_mol(mol)
            if self.skip_path_features:
                n = mol.GetNumAtoms()
                path_attr = np.zeros((n, n, CHEM_PATH_DIM), dtype=np.float32)
            else:
                path_attr = compute_chem_path_features(mol, max_distance=self.max_path_distance)
            data = Data(
                atom_cat=feats["atom_cat"],
                atom_float=feats["atom_float"],
                edge_index=feats["edge_index"],
                edge_cat=feats["edge_cat"],
                edge_float=feats["edge_float"],
                path_attr=torch.tensor(path_attr, dtype=torch.float32),
                desc=torch.tensor(self.desc[row_idx], dtype=torch.float32).view(1, -1),
                y=torch.tensor(self.y[row_idx], dtype=torch.float32).view(1),
                row_idx=torch.tensor([row_idx], dtype=torch.long),
                smiles=smiles,
                num_nodes=feats["atom_cat"].size(0),
            )
            if self.teacher is not None:
                data.teacher = torch.tensor(self.teacher[row_idx], dtype=torch.float32).view(1)
            data_list.append(data)
        if not data_list:
            raise ValueError(f"No valid molecules found in {self.csv_path}")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_molecules(data_list: list[Data]) -> Batch:
    batch = Batch.from_data_list(data_list, exclude_keys=["path_attr", "smiles"])
    max_nodes = max(data.num_nodes for data in data_list)
    path_dim = data_list[0].path_attr.size(-1)
    path = data_list[0].path_attr.new_zeros((len(data_list), max_nodes, max_nodes, path_dim))
    node_mask = torch.zeros((len(data_list), max_nodes), dtype=torch.bool)
    smiles = []
    for i, data in enumerate(data_list):
        n = data.num_nodes
        path[i, :n, :n, :] = data.path_attr
        node_mask[i, :n] = True
        smiles.append(getattr(data, "smiles", ""))
    batch.path_attr = path
    batch.node_mask = node_mask
    batch.smiles = smiles
    return batch
