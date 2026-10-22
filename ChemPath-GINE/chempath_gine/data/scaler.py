from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def fit_standard_scaler(x: np.ndarray) -> dict[str, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={x.shape}")
    if x.shape[1] == 0:
        return {
            "mean": np.zeros((0,), dtype=np.float32),
            "scale": np.ones((0,), dtype=np.float32),
        }
    mean = x.mean(axis=0).astype(np.float32)
    scale = x.std(axis=0).astype(np.float32)
    scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    return {"mean": mean, "scale": scale}


def transform_standard(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    if x.shape[1] == 0:
        return x.astype(np.float32)
    return ((x - scaler["mean"]) / scaler["scale"]).astype(np.float32)


def save_scaler(path: Path, scaler: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=scaler["mean"], scale=scaler["scale"])


def load_scaler(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing scaler: {path}")
    arr = np.load(path)
    return {"mean": arr["mean"].astype(np.float32), "scale": arr["scale"].astype(np.float32)}


def scaler_signature(scaler: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(scaler["mean"]).tobytes())
    h.update(np.ascontiguousarray(scaler["scale"]).tobytes())
    return h.hexdigest()
