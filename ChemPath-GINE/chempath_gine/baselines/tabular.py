from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.svm import SVR

from chempath_gine.features.conformer3d_bank import compute_conformer3d_frame
from chempath_gine.features.descriptor_bank import compute_descriptor_frame, descriptor_names
from chempath_gine.features.fingerprint_bank import compute_fingerprint_frame
from chempath_gine.features.fragment_bank import (
    build_brics_vocab,
    compute_brics_count_frame,
    compute_functional_group_count_frame,
)
from chempath_gine.selection import FeaturePreprocessor, FeatureSelectionSpec


@dataclass(frozen=True)
class TabularFeatureSpec:
    name: str
    descriptor_set: str = "none"
    fingerprints: tuple[str, ...] = ()
    brics_fragments: int = 0
    functional_groups: bool = False
    conformer3d: bool = False
    conformer3d_num_confs: int = 10
    selection_method: str = "none"
    selected_k: int | None = None


def parse_feature_spec(name: str) -> TabularFeatureSpec:
    key = name.lower()
    tokens = [tok for tok in key.split("_") if tok]

    descriptor_set = "none"
    fingerprints: list[str] = []
    brics_fragments = 0
    functional_groups = False
    conformer3d = False
    conformer3d_num_confs = 10
    selection_method = "none"
    selected_k: int | None = None

    if "desc6" in tokens:
        descriptor_set = "desc6"
    if "desc16" in tokens:
        descriptor_set = "desc16"
    if "rdkit2d" in tokens or "desc50" in tokens:
        descriptor_set = "rdkit2d"
    for fp in ("morgan2", "morgan3", "maccs", "rdkitfp", "atom_pair", "torsion"):
        if fp in tokens or fp in key:
            fingerprints.append(fp)
    for tok in tokens:
        if tok.startswith("selected"):
            selection_method = "pearson"
            suffix = tok.removeprefix("selected")
            selected_k = int(suffix) if suffix else 32
        elif tok.startswith("mi"):
            selection_method = "mutual_info"
            suffix = tok.removeprefix("mi")
            selected_k = int(suffix) if suffix else 32
        elif tok.startswith("enet"):
            selection_method = "elasticnet"
            suffix = tok.removeprefix("enet")
            selected_k = int(suffix) if suffix else 32
        elif tok.startswith("brics"):
            suffix = tok.removeprefix("brics")
            brics_fragments = int(suffix) if suffix else 256
        elif tok.startswith("conf3d") or tok.startswith("3d"):
            conformer3d = True
            suffix = tok.removeprefix("conf3d").removeprefix("3d")
            conformer3d_num_confs = int(suffix) if suffix else 10
    if "fg" in tokens or "functional" in tokens:
        functional_groups = True
    if "conf3d" in tokens or "3d" in tokens:
        conformer3d = True

    if key == "desc16":
        descriptor_set = "desc16"
    elif key == "rdkit2d":
        descriptor_set = "rdkit2d"
    elif key == "morgan2":
        fingerprints = ["morgan2"]
    elif key == "maccs":
        fingerprints = ["maccs"]

    return TabularFeatureSpec(
        name=name,
        descriptor_set=descriptor_set,
        fingerprints=tuple(dict.fromkeys(fingerprints)),
        brics_fragments=brics_fragments,
        functional_groups=functional_groups,
        conformer3d=conformer3d,
        conformer3d_num_confs=conformer3d_num_confs,
        selection_method=selection_method,
        selected_k=selected_k,
    )


def make_tabular_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    smiles_col: str,
    spec: TabularFeatureSpec,
):
    vocab: list[str] = []
    blocks = []
    for df in (train_df, val_df, test_df):
        blocks.append(_make_raw_features(df, smiles_col, spec, vocab=vocab))

    if spec.brics_fragments > 0 and not vocab:
        vocab.extend(build_brics_vocab(train_df[smiles_col], max_fragments=spec.brics_fragments))
        blocks = [_make_raw_features(df, smiles_col, spec, vocab=vocab) for df in (train_df, val_df, test_df)]

    return blocks[0], blocks[1], blocks[2], vocab


def _make_raw_features(
    df: pd.DataFrame,
    smiles_col: str,
    spec: TabularFeatureSpec,
    vocab: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    smiles = df[smiles_col].astype(str)
    if spec.descriptor_set != "none":
        names = descriptor_names(spec.descriptor_set)
        existing = [name for name in names if name in df.columns]
        missing = [name for name in names if name not in df.columns]
        desc_parts = []
        if existing:
            desc_parts.append(df[existing].reset_index(drop=True))
        if missing:
            desc_parts.append(compute_descriptor_frame(smiles, missing).reset_index(drop=True))
        parts.append(pd.concat(desc_parts, axis=1)[names])
    if spec.fingerprints:
        parts.append(compute_fingerprint_frame(smiles, spec.fingerprints).reset_index(drop=True))
    if spec.brics_fragments > 0:
        parts.append(compute_brics_count_frame(smiles, vocab).reset_index(drop=True))
    if spec.functional_groups:
        parts.append(compute_functional_group_count_frame(smiles).reset_index(drop=True))
    if spec.conformer3d:
        parts.append(
            compute_conformer3d_frame(smiles, num_confs=spec.conformer3d_num_confs).reset_index(
                drop=True
            )
        )
    if not parts:
        raise ValueError(f"Feature set {spec.name!r} did not produce any columns")
    out = pd.concat(parts, axis=1)
    out.columns = [str(col) for col in out.columns]
    return out


def build_preprocessor(spec: TabularFeatureSpec) -> FeaturePreprocessor:
    return FeaturePreprocessor(
        FeatureSelectionSpec(method=spec.selection_method, top_k=spec.selected_k),
        scale=True,
    )


def build_model(name: str, seed: int):
    key = name.lower()
    if key == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 25))
    if key == "elasticnet":
        return ElasticNetCV(
            alphas=np.logspace(-4, 1, 12),
            l1_ratio=[0.1, 0.25, 0.5, 0.75],
            cv=3,
            max_iter=8000,
            tol=1e-3,
            random_state=seed,
            n_jobs=-1,
        )
    if key == "svr":
        return SVR(C=10.0, epsilon=0.05, gamma="scale")
    if key == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        )
    if key == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Install xgboost to run model=xgb") from exc
        return XGBRegressor(
            n_estimators=900,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5.0,
            reg_alpha=0.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=60,
        )
    if key == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("Install lightgbm to run model=lightgbm") from exc
        return LGBMRegressor(
            n_estimators=900,
            learning_rate=0.02,
            max_depth=-1,
            num_leaves=15,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if key == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("Install catboost to run model=catboost") from exc
        return CatBoostRegressor(
            iterations=900,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=10.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown tabular model {name!r}")


def fit_predict(model, model_name: str, x_train, y_train, x_val, y_val, x_test):
    key = model_name.lower()
    if key == "xgb":
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    elif key == "lightgbm":
        try:
            from lightgbm import early_stopping, log_evaluation

            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
            )
        except Exception:
            model.fit(x_train, y_train)
    elif key == "catboost":
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    else:
        model.fit(x_train, y_train)
    return model.predict(x_test).reshape(-1)
