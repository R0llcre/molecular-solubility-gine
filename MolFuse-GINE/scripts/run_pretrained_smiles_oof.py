from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molfuse_gine.baselines.tabular import build_model, fit_predict
from molfuse_gine.baselines.tabular import build_preprocessor, make_tabular_features, parse_feature_spec
from molfuse_gine.data import make_splits
from molfuse_gine.features.pretrained_smiles_bank import SmilesEmbeddingConfig, embed_smiles_frame
from molfuse_gine.selection import FeaturePreprocessor, FeatureSelectionSpec
from molfuse_gine.training.metrics import regression_metrics


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _target(df: pd.DataFrame, target_col: str) -> np.ndarray:
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite targets found")
    return y


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def _json_fingerprint(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _smiles_fingerprint(smiles: pd.Series) -> str:
    hasher = hashlib.sha256()
    for value in smiles.astype(str).tolist():
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def _expected_protocol(
    cfg: dict,
    model_name: str,
    head_model: str,
    feature_set: str,
    seed: int,
    folds: int,
    random_smiles: int,
    batch_size: int,
    device: str,
) -> dict:
    return {
        "seed": int(seed),
        "folds_requested": int(folds),
        "model_name": model_name,
        "head_model": head_model,
        "feature_set": feature_set or "embedding",
        "random_smiles": int(random_smiles),
        "batch_size": int(batch_size),
        "device": device,
        "config_fingerprint": _json_fingerprint({"data": cfg.get("data", {})}),
    }


def _summary_matches_protocol(summary: dict, expected: dict) -> bool:
    return all(summary.get(key) == value for key, value in expected.items())


def _embed_cached(smiles: pd.Series, config: SmilesEmbeddingConfig, cache_root: Path, split: str) -> pd.DataFrame:
    smiles_fp = _smiles_fingerprint(smiles)
    cache_dir = (
        cache_root
        / "embedding_cache"
        / _safe_name(config.model_name)
        / f"seed{config.seed}_random{config.n_random}_max{config.max_length}_smiles{smiles_fp}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{split}.csv"
    meta_file = cache_dir / f"{split}.json"
    expected_meta = {
        "model_name": config.model_name,
        "seed": int(config.seed),
        "n_random": int(config.n_random),
        "max_length": int(config.max_length),
        "smiles_fingerprint": smiles_fp,
        "n_smiles": int(len(smiles)),
    }
    if cache_file.exists():
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if all(meta.get(key) == value for key, value in expected_meta.items()):
                return pd.read_csv(cache_file)
        else:
            cache_file.unlink()
    frame = embed_smiles_frame(smiles, config)
    frame.to_csv(cache_file, index=False)
    meta_file.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return frame


def run_one(
    cfg: dict,
    model_name: str,
    head_model: str,
    feature_set: str,
    seed: int,
    folds: int,
    random_smiles: int,
    batch_size: int,
    device: str,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = _expected_protocol(
        cfg=cfg,
        model_name=model_name,
        head_model=head_model,
        feature_set=feature_set,
        seed=seed,
        folds=folds,
        random_smiles=random_smiles,
        batch_size=batch_size,
        device=device,
    )
    split_paths = make_splits(
        csv_path=cfg["data"]["csv"],
        out_dir=out_dir / "splits",
        smiles_col=cfg["data"]["smiles_col"],
        split=cfg["data"]["split"],
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        seed=seed,
    )
    train_df = pd.read_csv(split_paths["train"])
    val_df = pd.read_csv(split_paths["val"])
    test_df = pd.read_csv(split_paths["test"])
    emb_cfg = SmilesEmbeddingConfig(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        n_random=random_smiles,
        seed=seed,
    )
    cache_root = out_dir.parent.parent if out_dir.parent.name == "runs" else out_dir.parent
    x_train_emb = _embed_cached(train_df[cfg["data"]["smiles_col"]], emb_cfg, cache_root, f"seed{seed}_train")
    x_val_emb = _embed_cached(val_df[cfg["data"]["smiles_col"]], emb_cfg, cache_root, f"seed{seed}_val")
    x_test_emb = _embed_cached(test_df[cfg["data"]["smiles_col"]], emb_cfg, cache_root, f"seed{seed}_test")
    extra_spec = None
    if feature_set and feature_set.lower() not in {"embedding", "emb", "none"}:
        extra_spec = parse_feature_spec(feature_set)
    y_train = _target(train_df, cfg["data"]["target_col"])
    y_val = _target(val_df, cfg["data"]["target_col"])
    y_test = _target(test_df, cfg["data"]["target_col"])

    n_splits = min(max(2, int(folds)), len(train_df))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_df), dtype=np.float64)
    val_fold_preds = []
    test_fold_preds = []
    last_tabular_columns: list[str] = []
    last_brics_vocab_size = 0

    def _concat_features(base: pd.DataFrame, extra_arr: np.ndarray | None, extra_columns: list[str]) -> pd.DataFrame:
        base = base.reset_index(drop=True)
        if extra_arr is None:
            return base
        extra_df = pd.DataFrame(extra_arr, columns=[f"tabular_{col}" for col in extra_columns])
        return pd.concat([base, extra_df], axis=1)

    for fold_idx, (fit_idx, hold_idx) in enumerate(kfold.split(x_train_emb), 1):
        if len(fit_idx) >= 4:
            inner_fit_idx, inner_eval_idx = train_test_split(
                fit_idx,
                test_size=max(1, int(round(len(fit_idx) * 0.15))),
                random_state=seed + fold_idx,
                shuffle=True,
            )
        else:
            inner_fit_idx = fit_idx
            inner_eval_idx = fit_idx
        if extra_spec is not None:
            fit_df = train_df.iloc[inner_fit_idx].reset_index(drop=True)
            eval_df = train_df.iloc[inner_eval_idx].reset_index(drop=True)
            hold_df = train_df.iloc[hold_idx].reset_index(drop=True)
            extra_fit, extra_eval, extra_hold, vocab = make_tabular_features(
                fit_df,
                eval_df,
                hold_df,
                cfg["data"]["smiles_col"],
                extra_spec,
            )
            extra_fit_for_vt, extra_val, extra_test, vocab_vt = make_tabular_features(
                fit_df,
                val_df,
                test_df,
                cfg["data"]["smiles_col"],
                extra_spec,
            )
            if list(extra_fit.columns) != list(extra_fit_for_vt.columns):
                raise RuntimeError("Fold tabular feature columns changed between holdout and val/test builds")
            del extra_fit_for_vt
            last_brics_vocab_size = int(max(len(vocab), len(vocab_vt)))
            tabular_pre = build_preprocessor(extra_spec)
            extra_fit = tabular_pre.fit_transform(
                extra_fit.reset_index(drop=True),
                y_train[inner_fit_idx],
            )
            extra_eval = tabular_pre.transform(extra_eval.reset_index(drop=True))
            extra_hold = tabular_pre.transform(extra_hold.reset_index(drop=True))
            extra_val = tabular_pre.transform(extra_val.reset_index(drop=True))
            extra_test = tabular_pre.transform(extra_test.reset_index(drop=True))
            last_tabular_columns = list(tabular_pre.selected_columns_)
        else:
            extra_fit = extra_eval = extra_hold = extra_val = extra_test = None
            last_tabular_columns = []
        x_fit_raw = _concat_features(
            x_train_emb.iloc[inner_fit_idx],
            extra_fit,
            last_tabular_columns,
        )
        x_eval_raw = _concat_features(
            x_train_emb.iloc[inner_eval_idx],
            extra_eval,
            last_tabular_columns,
        )
        x_hold_raw = _concat_features(
            x_train_emb.iloc[hold_idx],
            extra_hold,
            last_tabular_columns,
        )
        x_val_raw = _concat_features(x_val_emb, extra_val, last_tabular_columns)
        x_test_raw = _concat_features(x_test_emb, extra_test, last_tabular_columns)
        pre = FeaturePreprocessor(FeatureSelectionSpec(method="none", top_k=None), scale=True)
        x_fit = pre.fit_transform(x_fit_raw, y_train[inner_fit_idx])
        x_eval = pre.transform(x_eval_raw)
        x_hold = pre.transform(x_hold_raw)
        x_val = pre.transform(x_val_raw)
        x_test = pre.transform(x_test_raw)
        model = build_model(head_model, seed + fold_idx)
        oof[hold_idx] = fit_predict(
            model,
            head_model,
            x_fit,
            y_train[inner_fit_idx],
            x_eval,
            y_train[inner_eval_idx],
            x_hold,
        )
        val_fold_preds.append(model.predict(x_val).reshape(-1))
        test_fold_preds.append(model.predict(x_test).reshape(-1))

    feature_suffix = _safe_name(feature_set or "embedding")
    col = f"{_safe_name(model_name)}__{feature_suffix}__{head_model}"
    val_pred = np.mean(np.vstack(val_fold_preds), axis=0)
    test_pred = np.mean(np.vstack(test_fold_preds), axis=0)
    train_out = pd.DataFrame(
        {
            "source_index": train_df.get("source_index", train_df.index),
            "y_true": y_train,
            col: oof,
        }
    )
    val_out = pd.DataFrame(
        {"source_index": val_df.get("source_index", val_df.index), "y_true": y_val, col: val_pred}
    )
    test_out = pd.DataFrame(
        {"source_index": test_df.get("source_index", test_df.index), "y_true": y_test, col: test_pred}
    )
    train_out.to_csv(out_dir / "expert_oof_train.csv", index=False)
    val_out.to_csv(out_dir / "expert_val_predictions.csv", index=False)
    test_out.to_csv(out_dir / "expert_test_predictions.csv", index=False)
    summary = {
        **protocol,
        "protocol_fingerprint": _json_fingerprint(protocol),
        "seed": int(seed),
        "folds": int(n_splits),
        "n_features": int(x_train_emb.shape[1] + len(last_tabular_columns)),
        "n_embedding_features": int(x_train_emb.shape[1]),
        "n_tabular_features_last_fold": int(len(last_tabular_columns)),
        "brics_vocab_size": int(last_brics_vocab_size),
        **{f"val_{k}": v for k, v in regression_metrics(val_pred, y_val).items()},
        **{f"test_{k}": v for k, v in regression_metrics(test_pred, y_test).items()},
    }
    with (out_dir / "pretrained_oof_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-names", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--head-models", default="ridge")
    parser.add_argument("--feature-sets", default="embedding")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-smiles", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="pretrained_smiles_oof_runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "pretrained_oof_summary.csv"
    rows = []
    for seed_text in _parse_list(args.seeds):
        seed = int(seed_text)
        for model_name in _parse_list(args.model_names):
            for feature_set in _parse_list(args.feature_sets):
                for head in _parse_list(args.head_models):
                    expected_protocol = _expected_protocol(
                        cfg=cfg,
                        model_name=model_name,
                        head_model=head,
                        feature_set=feature_set,
                        seed=seed,
                        folds=args.folds,
                        random_smiles=args.random_smiles,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                    run_dir = (
                        out_root
                        / "runs"
                        / (
                            f"seed{seed}_{_safe_name(model_name)}_{_safe_name(feature_set)}_{head}"
                            f"_folds{args.folds}_rand{args.random_smiles}_bs{args.batch_size}"
                        )
                    )
                    run_summary = run_dir / "pretrained_oof_summary.json"
                    if run_summary.exists() and not args.overwrite:
                        summary = json.loads(run_summary.read_text(encoding="utf-8"))
                        if _summary_matches_protocol(summary, expected_protocol):
                            print(f"resume existing {run_summary}")
                        else:
                            print(f"protocol changed; rerun {run_summary}")
                            summary = run_one(
                                cfg=cfg,
                                model_name=model_name,
                                head_model=head,
                                feature_set=feature_set,
                                seed=seed,
                                folds=args.folds,
                                random_smiles=args.random_smiles,
                                batch_size=args.batch_size,
                                device=args.device,
                                out_dir=run_dir,
                            )
                    else:
                        summary = run_one(
                            cfg=cfg,
                            model_name=model_name,
                            head_model=head,
                            feature_set=feature_set,
                            seed=seed,
                            folds=args.folds,
                            random_smiles=args.random_smiles,
                            batch_size=args.batch_size,
                            device=args.device,
                            out_dir=run_dir,
                        )
                    rows.append({"run_dir": str(run_dir), **summary})

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
