"""Hybrid system row: deep-ensemble mean blended with a train-only GBDT.

    y = (1 - w) * mu_ensemble + w * GBDT

GBDT: sklearn HistGradientBoostingRegressor with hyperparameters fixed in
advance (no tuning), fitted on each seed's train split only, on all RDKit 2D
descriptors (Descriptors._descList) plus a Morgan radius-2 count fingerprint
(1024 bits). One blend weight per split is chosen from {0, 0.05, ..., 0.50} by
the 5-seed mean validation RMSE.

Two phases keep test numbers out of the selection step:
  --phase select  fit GBDTs, predict and write validation only, choose w
  --phase reveal  read the frozen w from the lock file and evaluate test once

Example
-------
python scripts/run_hybrid_blend.py --phase select \
  --ens-roots calibration_runs/locked/ensembles_desc7/esol_random,calibration_runs/locked/ensembles_desc7/esol_scaffold \
  --mode faithful_ens --out calibration_runs/locked/hybrid_desc7
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor

RDLogger.DisableLog("rdApp.*")
GBDT_PARAMS = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=10,
                   l2_regularization=1.0, random_state=0)
W_GRID = np.round(np.arange(0, 0.501, 0.05), 2)
TARGET = "Solubility"


def _parse_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


_FEATURE_CACHE: dict[str, np.ndarray] = {}
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def features(smiles: str) -> np.ndarray:
    if smiles not in _FEATURE_CACHE:
        mol = Chem.MolFromSmiles(smiles)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            desc = []
            for _, fn in Descriptors._descList:
                try:
                    desc.append(float(fn(mol)))
                except Exception:
                    desc.append(np.nan)
        desc = np.array(desc, dtype=np.float64)
        desc[~np.isfinite(desc)] = np.nan
        fp = _MORGAN.GetCountFingerprintAsNumPy(mol).astype(np.float64)
        _FEATURE_CACHE[smiles] = np.concatenate([desc, fp])
    return _FEATURE_CACHE[smiles]


def _matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack([features(s) for s in df["SMILES"]])


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def fit_seed(run_dir: Path, parts=("val", "test")) -> dict:
    """Fit the GBDT on train, predict the requested parts; align with the ensemble predictions."""
    train = pd.read_csv(run_dir / "splits" / "train.csv")
    gbdt = HistGradientBoostingRegressor(**GBDT_PARAMS).fit(_matrix(train), train[TARGET].values)
    out = {}
    for part in parts:
        pred = pd.read_csv(run_dir / f"predictions_{part}.csv")
        split_df = pd.read_csv(run_dir / "splits" / f"{part}.csv")
        g = pd.Series(gbdt.predict(_matrix(split_df)), index=split_df["source_index"].values)
        pred = pred.assign(gbdt=g.loc[pred["source_index"].values].values)
        out[part] = pred[["source_index", "y_true", "mu", "gbdt"]]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "reveal"), required=True)
    parser.add_argument("--ens-roots", required=True)
    parser.add_argument("--mode", default="faithful_ens")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lock", default=None, help="lock JSON holding frozen w per split (reveal phase)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in _parse_list(args.seeds)]

    results = {}
    for root in [Path(r) for r in _parse_list(args.ens_roots)]:
        per_seed = {}
        split = None
        for s in seeds:
            summary = json.loads((root / "summaries" / f"{args.mode}_seed{s}.json").read_text())
            split = summary["split"]
            parts = ("val",) if args.phase == "select" else ("val", "test")
            per_seed[s] = fit_seed(Path(summary["run_dir"]), parts)
            for part in parts:
                per_seed[s][part].to_csv(out / f"{split}_seed{s}_{part}_gbdt.csv", index=False)
        val_curve = {float(w): float(np.mean([
            _rmse((1 - w) * per_seed[s]["val"]["mu"] + w * per_seed[s]["val"]["gbdt"], per_seed[s]["val"]["y_true"])
            for s in seeds])) for w in W_GRID}
        w_sel = min(val_curve, key=lambda w: (val_curve[w], w))
        res = {
            "split": split, "w_selected_on_val": w_sel, "val_curve": val_curve,
            "val_rmse_ensemble": float(np.mean([_rmse(per_seed[s]["val"]["mu"], per_seed[s]["val"]["y_true"]) for s in seeds])),
            "val_rmse_gbdt": float(np.mean([_rmse(per_seed[s]["val"]["gbdt"], per_seed[s]["val"]["y_true"]) for s in seeds])),
            "val_rmse_blend": val_curve[w_sel],
        }
        if args.phase == "reveal":
            lock = json.loads(Path(args.lock).read_text())
            w = float(lock["hybrid_w"][split])
            t_ens = [_rmse(per_seed[s]["test"]["mu"], per_seed[s]["test"]["y_true"]) for s in seeds]
            t_gb = [_rmse(per_seed[s]["test"]["gbdt"], per_seed[s]["test"]["y_true"]) for s in seeds]
            t_bl = [_rmse((1 - w) * per_seed[s]["test"]["mu"] + w * per_seed[s]["test"]["gbdt"],
                          per_seed[s]["test"]["y_true"]) for s in seeds]
            res.update({"w_frozen": w, "test_rmse_ensemble_per_seed": t_ens, "test_rmse_gbdt_per_seed": t_gb,
                        "test_rmse_blend_per_seed": t_bl,
                        "test_rmse_ensemble": [float(np.mean(t_ens)), float(np.std(t_ens, ddof=1))],
                        "test_rmse_gbdt": [float(np.mean(t_gb)), float(np.std(t_gb, ddof=1))],
                        "test_rmse_blend": [float(np.mean(t_bl)), float(np.std(t_bl, ddof=1))]})
        results[split] = res
    (out / f"hybrid_{args.phase}.json").write_text(json.dumps(results, indent=2))
    for split, r in results.items():
        line = (f"{split}: w*={r['w_selected_on_val']:.2f} val ens {r['val_rmse_ensemble']:.4f} "
                f"gbdt {r['val_rmse_gbdt']:.4f} blend {r['val_rmse_blend']:.4f}")
        if args.phase == "reveal":
            line += (f" | TEST (w={r['w_frozen']:.2f}) ens {r['test_rmse_ensemble'][0]:.4f} "
                     f"gbdt {r['test_rmse_gbdt'][0]:.4f} blend {r['test_rmse_blend'][0]:.4f}"
                     f"±{r['test_rmse_blend'][1]:.4f}")
        print(line)


if __name__ == "__main__":
    main()
