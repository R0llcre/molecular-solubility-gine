from __future__ import annotations

from dataclasses import dataclass
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.svm import SVR

from molfuse_gine.features.conformer3d_bank import compute_conformer3d_frame
from molfuse_gine.features.descriptor_bank import compute_descriptor_frame, descriptor_names
from molfuse_gine.features.fingerprint_bank import compute_fingerprint_frame
from molfuse_gine.features.fragment_bank import (
    build_brics_vocab,
    compute_brics_count_frame,
    compute_functional_group_count_frame,
)
from molfuse_gine.selection import FeaturePreprocessor, FeatureSelectionSpec


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
    selection_scope: str = "all"


def parse_feature_spec(name: str) -> TabularFeatureSpec:
    raw_key = name.lower()
    key = (
        raw_key.replace("atom_pair", "atompair")
        .replace("atom-pair", "atompair")
        .replace("topological_torsion", "torsion")
        .replace("topological-torsion", "torsion")
    )
    tokens = [tok for tok in key.split("_") if tok]
    allowed_literals = {
        "desc6",
        "desc16",
        "rdkit2d",
        "desc50",
        "morgan2",
        "morgan3",
        "maccs",
        "rdkitfp",
        "atompair",
        "torsion",
        "fg",
        "functional",
        "conf3d",
        "3d",
    }
    allowed_prefixes = (
        "selected",
        "mi",
        "enet",
        "descsel",
        "midesc",
        "enetdesc",
        "brics",
        "conf3d",
        "3d",
    )
    unknown = [
        tok
        for tok in tokens
        if tok not in allowed_literals
        and not any(re.fullmatch(prefix + r"\d*", tok) for prefix in allowed_prefixes)
    ]
    if unknown:
        raise ValueError(
            f"Unknown tabular feature token(s) in {name!r}: {', '.join(unknown)}"
        )

    descriptor_set = "none"
    fingerprints: list[str] = []
    brics_fragments = 0
    functional_groups = False
    conformer3d = False
    conformer3d_num_confs = 10
    selection_method = "none"
    selected_k: int | None = None
    selection_scope = "all"

    if "desc6" in tokens:
        descriptor_set = "desc6"
    if "desc16" in tokens:
        descriptor_set = "desc16"
    if "rdkit2d" in tokens or "desc50" in tokens:
        descriptor_set = "rdkit2d"
    fingerprint_tokens = {
        "morgan2": "morgan2",
        "morgan3": "morgan3",
        "maccs": "maccs",
        "rdkitfp": "rdkitfp",
        "atompair": "atom_pair",
        "torsion": "torsion",
    }
    for tok, fp in fingerprint_tokens.items():
        if tok in tokens:
            fingerprints.append(fp)
    for tok in tokens:
        if tok.startswith("selected"):
            selection_method = "pearson"
            suffix = tok.removeprefix("selected")
            selected_k = int(suffix) if suffix else 32
        elif tok.startswith("descsel"):
            descriptor_set = "rdkit2d" if descriptor_set == "none" else descriptor_set
            selection_method = "pearson"
            selection_scope = "descriptor"
            suffix = tok.removeprefix("descsel")
            selected_k = int(suffix) if suffix else 64
        elif tok.startswith("midesc"):
            descriptor_set = "rdkit2d" if descriptor_set == "none" else descriptor_set
            selection_method = "mutual_info"
            selection_scope = "descriptor"
            suffix = tok.removeprefix("midesc")
            selected_k = int(suffix) if suffix else 64
        elif tok.startswith("enetdesc"):
            descriptor_set = "rdkit2d" if descriptor_set == "none" else descriptor_set
            selection_method = "elasticnet"
            selection_scope = "descriptor"
            suffix = tok.removeprefix("enetdesc")
            selected_k = int(suffix) if suffix else 64
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
        selection_scope=selection_scope,
    )


def feature_group_counts(columns: list[str]) -> dict[str, int]:
    groups = {
        "descriptor": 0,
        "morgan2": 0,
        "morgan3": 0,
        "maccs": 0,
        "rdkitfp": 0,
        "atom_pair": 0,
        "torsion": 0,
        "brics": 0,
        "functional_group": 0,
        "conformer3d": 0,
        "other": 0,
    }
    descriptor_set = set(descriptor_names("rdkit2d"))
    for col in columns:
        if col in descriptor_set:
            groups["descriptor"] += 1
        elif col.startswith("morgan2_"):
            groups["morgan2"] += 1
        elif col.startswith("morgan3_"):
            groups["morgan3"] += 1
        elif col.startswith("maccs_"):
            groups["maccs"] += 1
        elif col.startswith("rdkitfp_"):
            groups["rdkitfp"] += 1
        elif col.startswith("atom_pair_"):
            groups["atom_pair"] += 1
        elif col.startswith("torsion_"):
            groups["torsion"] += 1
        elif col.startswith("brics_"):
            groups["brics"] += 1
        elif col.startswith("fg_"):
            groups["functional_group"] += 1
        elif col.startswith("conf3d_"):
            groups["conformer3d"] += 1
        else:
            groups["other"] += 1
    return groups


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
    candidate_columns = None
    candidate_prefixes: tuple[str, ...] = ()
    if spec.selection_scope == "descriptor":
        candidate_columns = tuple(descriptor_names(spec.descriptor_set))
    return FeaturePreprocessor(
        FeatureSelectionSpec(
            method=spec.selection_method,
            top_k=spec.selected_k,
            candidate_columns=candidate_columns,
            candidate_prefixes=candidate_prefixes,
            keep_non_candidates=True,
        ),
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
            precompute=False,
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
    if key in {"extratrees", "extra_trees", "et"}:
        return ExtraTreesRegressor(
            n_estimators=600,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
            max_features="sqrt",
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
        except TypeError as exc:
            warnings.warn(
                f"LightGBM early-stopping callbacks failed ({exc}); fitting without eval_set",
                RuntimeWarning,
                stacklevel=2,
            )
            model.fit(x_train, y_train)
    elif key == "catboost":
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    else:
        model.fit(x_train, y_train)
    return model.predict(x_test).reshape(-1)
